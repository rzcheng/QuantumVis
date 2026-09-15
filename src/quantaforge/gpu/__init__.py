"""Optional NVIDIA backend; importing this interface does not import torch/Triton."""

from quantaforge.gpu.runtime import GPUStatus, GPUUnavailableError, gpu_available, gpu_status
from quantaforge.gpu.simulator import GPUResult, GPUSimulator

__all__ = [
    "GPUResult",
    "GPUSimulator",
    "GPUStatus",
    "GPUUnavailableError",
    "gpu_available",
    "gpu_status",
]
