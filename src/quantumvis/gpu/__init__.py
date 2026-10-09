"""optional nvidia backend; importing this interface does not import torch/triton."""

from quantumvis.gpu.runtime import GPUStatus, GPUUnavailableError, gpu_available, gpu_status
from quantumvis.gpu.simulator import GPUResult, GPUSimulator

__all__ = [
    "GPUResult",
    "GPUSimulator",
    "GPUStatus",
    "GPUUnavailableError",
    "gpu_available",
    "gpu_status",
]
