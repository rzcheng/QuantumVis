"""kernel entry point; keeps optional imports lazy."""

import numpy as np
from numpy.typing import ArrayLike

from quantaforge._validation import integer

BLOCK_SIZE = 256
# 40 qubits exceed the 2**31 - 1 grid limit at 256 pairs per block.
# memory runs out much earlier: even 39 qubits need 4 TiB of state storage.
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
    """round eight real coefficients once, checking float32 representability."""
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
    """apply a general complex 2x2 matrix in-place to split cuda float32 arrays.

    callers handle synchronization. normalization and unitarity are unchecked;
    use GPUSimulator for validated inputs and synchronous results.
    """
    from quantaforge.gpu.kernels.single_qubit import apply_single_qubit as apply

    apply(real, imag, matrix, target)
