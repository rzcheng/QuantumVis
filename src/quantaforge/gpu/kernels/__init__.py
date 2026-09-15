"""Lazy entry point and host-side checks for the first Triton kernel.

Importing this module does not import PyTorch or Triton. The low-level operation
expects CUDA tensors and enqueues work asynchronously on their current stream.
"""

import numpy as np
from numpy.typing import ArrayLike

from quantaforge._validation import integer

BLOCK_SIZE = 256
# NVIDIA's one-dimensional grid is limited to 2**31 - 1 programs. A 40-qubit
# state needs 2**31 blocks of 256 pairs, so reject it before allocating storage.
# This is an index/launch bound; available device memory imposes a much lower
# practical limit (the split state alone would occupy 4 TiB at 39 qubits).
MAX_NUM_QUBITS = 39


def _validate_num_amplitudes(size: int) -> int:
    size = integer(size, "state length", minimum=2)
    if size & (size - 1):
        raise ValueError("state length must be a power of two")
    num_qubits = size.bit_length() - 1
    if num_qubits > MAX_NUM_QUBITS:
        raise ValueError(f"this kernel supports at most {MAX_NUM_QUBITS} qubits")
    return num_qubits


def _matrix_coefficients(matrix: ArrayLike) -> tuple[float, ...]:
    """Round eight real coefficients once, checking float32 representability."""
    values = np.asarray(matrix, dtype=np.complex128)
    if values.shape != (2, 2):
        raise ValueError("single-qubit matrix must have shape (2, 2)")
    components = np.stack((values.real, values.imag), axis=-1)
    if not np.all(np.isfinite(components)):
        raise ValueError("single-qubit matrix coefficients must be finite")
    if np.any(np.abs(components) > np.finfo(np.float32).max):
        raise ValueError("single-qubit matrix coefficients must fit float32")
    return tuple(float(value) for value in components.astype(np.float32).ravel())


def apply_single_qubit(real, imag, matrix: ArrayLike, target: int) -> None:
    """Apply a general complex 2x2 matrix in-place to split CUDA float32 arrays.

    This is a low-level asynchronous API. Callers own stream synchronization;
    neither normalized input nor matrix unitarity is enforced here. Use
    GPUSimulator for validated quantum-state inputs and synchronous CPU results.
    """
    from quantaforge.gpu.kernels.single_qubit import apply_single_qubit as apply

    apply(real, imag, matrix, target)
