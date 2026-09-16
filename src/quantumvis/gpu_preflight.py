"""check gpu prerequisites, with an optional single production-kernel launch."""

import argparse
import json
import platform
import sys
import traceback
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

from quantumvis.gpu.runtime import gpu_status

JETSON_FILES = {
    "l4t_release": Path("/etc/nv_tegra_release"),
    "model": Path("/proc/device-tree/model"),
}


def jetson_info() -> dict:
    info = {}
    for name, path in JETSON_FILES.items():
        try:
            info[name] = path.read_text().rstrip("\x00\n")
        except FileNotFoundError:
            info[name] = None
    return info


def runtime_version() -> tuple[str | None, str]:
    """query an already-installed public cuda binding; never guess from torch's build."""
    try:
        runtime = import_module("cuda.bindings.runtime")
    except ModuleNotFoundError as error:
        if error.name not in {"cuda", "cuda.bindings", "cuda.bindings.runtime"}:
            raise
        return None, "unavailable: optional CUDA Python bindings are not installed"
    query = getattr(runtime, "getLocalRuntimeVersion", None)
    if query is None:
        return None, "unavailable: CUDA Python has no getLocalRuntimeVersion API"
    error, value = query()
    if int(error) != 0:
        raise RuntimeError(f"CUDA runtime version query failed: {error}")
    return f"{value // 1000}.{(value % 1000) // 10}", "cuda.bindings.runtime.getLocalRuntimeVersion"


def kernel_smoke() -> dict:
    """use the full production path for x|0>, including blocking download."""
    import numpy as np

    from quantumvis import Circuit
    from quantumvis.gpu import GPUSimulator
    from quantumvis.validate_gpu import check_output

    actual = GPUSimulator().run(Circuit(1).x(0)).amplitudes
    check = check_output("preflight-X-zero", actual, np.array([0, 1], dtype=np.complex128))
    return asdict(check)


def preflight(*, smoke: bool = False) -> tuple[dict, int]:
    report = {
        "schema_version": 1,
        "status": "BLOCKED",
        "os": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": None,
        "triton_version": None,
        "cuda_available": False,
        "cuda_build_version": None,
        "cuda_runtime_version": None,
        "cuda_runtime_note": "not queried: CUDA prerequisites unavailable",
        "gpu_name": None,
        "compute_capability": None,
        "device_index": None,
        "jetson": None,
        "minimum_requirements_satisfied": False,
        "kernel_import": "NOT_ATTEMPTED",
        "smoke": "NOT_RUN",
        "reason": None,
    }
    try:
        report["jetson"] = jetson_info()
        status = asdict(gpu_status())
        status["cuda_build_version"] = status.pop("cuda_version")
        report.update(status)
        if not status["usable"]:
            return report, 2
        if sys.version_info < (3, 11) or report["architecture"] not in {"x86_64", "aarch64"}:
            report["reason"] = "Python >= 3.11 and Linux x86_64/aarch64 required"
            return report, 2
        for name, minimum, maximum in (("torch", (2, 6), (3, 0)), ("triton", (3, 2), (4, 0))):
            version = tuple(int(part) for part in report[f"{name}_version"].split(".")[:2])
            if not minimum <= version < maximum:
                report["reason"] = f"{name} major/minor outside declared GPU dependency range"
                return report, 2
        report["minimum_requirements_satisfied"] = True
        report["cuda_runtime_version"], report["cuda_runtime_note"] = runtime_version()
        import_module("quantumvis.gpu.kernels.single_qubit")
        report["kernel_import"] = "PASS"
        if smoke:
            report["smoke_result"] = kernel_smoke()
            report["smoke"] = "PASS"
        report["status"] = "READY"
        report["reason"] = (
            "production X smoke passed; full device acceptance still required"
            if smoke
            else "prerequisites and kernel import passed; no kernel was executed"
        )
        return report, 0
    except Exception as error:
        report["status"] = "ERROR"
        report["reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        return report, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="launch production X on one qubit")
    parser.add_argument("--json", action="store_true", help="emit a structured report")
    args = parser.parse_args(argv)
    report, code = preflight(smoke=args.smoke)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"{report['status']}: {report['reason']}")
        for key, value in report.items():
            if key not in {"status", "reason", "schema_version"}:
                print(f"{key}: {value}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
