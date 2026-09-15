"""Explicit optional NVIDIA/Triton runtime detection, without kernel execution."""

import os
import platform
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec

from quantaforge._validation import integer


@dataclass(frozen=True, slots=True)
class GPUStatus:
    usable: bool
    reason: str
    torch_available: bool
    triton_available: bool
    cuda_available: bool
    platform: str
    python_version: str
    torch_version: str | None = None
    triton_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    device_index: int | None = None


class GPUUnavailableError(RuntimeError):
    """An explicitly requested GPU backend cannot run in this environment."""


def gpu_status(device: int | None = None) -> GPUStatus:
    """Probe prerequisites; a usable status does not establish kernel correctness.

    Missing top-level packages are ordinary capability results. Broken imports,
    driver queries, or invalid explicit device indices raise their original errors.
    No failures are caught and converted into a misleading missing-GPU result.
    """
    if device is not None:
        device = integer(device, "device")
    system = platform.system()
    has_torch = find_spec("torch") is not None
    has_triton = find_spec("triton") is not None
    details = {
        "platform": system,
        "python_version": platform.python_version(),
        "torch_available": has_torch,
        "triton_available": has_triton,
        "cuda_available": False,
    }
    reasons = []
    if system != "Linux":
        reasons.append(f"Linux/NVIDIA required; host platform is {system}")
    # Match Triton's case-insensitive boolean environment parser, not just '1'.
    if os.environ.get("TRITON_INTERPRET", "").lower() in {"1", "y", "on", "yes", "true"}:
        reasons.append("TRITON_INTERPRET enables CPU interpretation, not NVIDIA execution")
    if not has_torch:
        reasons.append("PyTorch is not installed")
    else:
        torch = import_module("torch")
        details["torch_version"] = str(torch.__version__)
        details["cuda_version"] = torch.version.cuda
        details["cuda_available"] = bool(torch.cuda.is_available())
        if torch.version.cuda is None:
            reasons.append("PyTorch has no NVIDIA CUDA build (CPU/ROCm builds are unsupported)")
        if not details["cuda_available"]:
            reasons.append("CUDA is unavailable to PyTorch; check the device/driver/build")
        else:
            selected = torch.cuda.current_device() if device is None else device
            if selected >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {selected} is outside the visible device range")
            details["device_index"] = selected
            details["gpu_name"] = torch.cuda.get_device_name(selected)
            capability = tuple(torch.cuda.get_device_capability(selected))
            details["compute_capability"] = capability
            if capability < (8, 0):
                reasons.append(f"compute capability >= 8.0 required; found {capability}")
    if not has_triton:
        reasons.append("Triton is not installed")
    else:
        # Import errors here must remain visible, including broken binary dependencies.
        triton = import_module("triton")
        details["triton_version"] = str(triton.__version__)
        # Newer Triton also permits in-process overrides, e.g. in a notebook.
        knobs = getattr(triton, "knobs", None)
        runtime_knobs = getattr(knobs, "runtime", None)
        if getattr(runtime_knobs, "interpret", False):
            reasons.append(
                "Triton's runtime interpreter is enabled; real NVIDIA execution required"
            )
    return GPUStatus(
        usable=not reasons,
        reason="; ".join(reasons)
        or "NVIDIA/Triton prerequisites available; kernel execution unverified",
        **details,
    )


def gpu_available(device: int | None = None) -> bool:
    return gpu_status(device).usable


def require_gpu(device: int | None = None) -> GPUStatus:
    status = gpu_status(device)
    if not status.usable:
        raise GPUUnavailableError(status.reason)
    return status
