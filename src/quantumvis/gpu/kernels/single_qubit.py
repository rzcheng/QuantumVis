"""in-place complex 2x2 kernel; nvidia validation is still pending."""

import torch
import triton
import triton.language as tl
from numpy.typing import ArrayLike

from quantumvis._validation import integer
from quantumvis.gpu.kernels import BLOCK_SIZE, _matrix_coefficients, _validate_num_amplitudes


@triton.jit
def _single_qubit_kernel(
    real_ptr,
    imag_ptr,
    num_pairs,
    u00r,
    u00i,
    u01r,
    u01i,
    u10r,
    u10i,
    u11r,
    u11i,
    TARGET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # widen before multiplication so pair indices beyond 2**31 remain valid.
    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = pair < num_pairs
    # insert a zero target bit to give each pair exactly one owner.
    low_mask: tl.constexpr = (1 << TARGET) - 1
    i0 = ((pair >> TARGET) << (TARGET + 1)) | (pair & low_mask)
    i1 = i0 | (1 << TARGET)

    # load both original amplitudes before writing either output.
    a0r = tl.load(real_ptr + i0, mask=valid, other=0)
    a0i = tl.load(imag_ptr + i0, mask=valid, other=0)
    a1r = tl.load(real_ptr + i1, mask=valid, other=0)
    a1i = tl.load(imag_ptr + i1, mask=valid, other=0)

    b0r = (u00r * a0r - u00i * a0i) + (u01r * a1r - u01i * a1i)
    b0i = (u00r * a0i + u00i * a0r) + (u01r * a1i + u01i * a1r)
    b1r = (u10r * a0r - u10i * a0i) + (u11r * a1r - u11i * a1i)
    b1i = (u10r * a0i + u10i * a0r) + (u11r * a1i + u11i * a1r)

    tl.store(real_ptr + i0, b0r, mask=valid)
    tl.store(imag_ptr + i0, b0i, mask=valid)
    tl.store(real_ptr + i1, b1r, mask=valid)
    tl.store(imag_ptr + i1, b1i, mask=valid)


def apply_single_qubit(
    real: torch.Tensor, imag: torch.Tensor, matrix: ArrayLike, target: int
) -> None:
    """validate split storage and enqueue one in-place generic matrix operation."""
    # notebooks can retain an interpreted wrapper after disabling interpreter mode.
    if not isinstance(_single_qubit_kernel, triton.JITFunction):
        raise RuntimeError(
            "the kernel was imported in Triton interpreter mode; "
            "restart the process with interpreter mode disabled for real GPU execution"
        )
    for name, tensor in (("real", real), ("imag", imag)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
        if tensor.layout != torch.strided or tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous one-dimensional tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on a CUDA device")
        if tensor.requires_grad:
            raise ValueError("the Triton operation does not support autograd")
    if real.shape != imag.shape:
        raise ValueError("real and imag must have equal shape")
    if real.device != imag.device:
        raise ValueError("real and imag must be on the same CUDA device")
    if real.untyped_storage().data_ptr() == imag.untyped_storage().data_ptr():
        raise ValueError("real and imag must use distinct storage")
    num_qubits = _validate_num_amplitudes(real.numel())
    target = integer(target, "target")
    if target >= num_qubits:
        raise ValueError("target qubit is outside the state")
    coefficients = _matrix_coefficients(matrix)
    num_pairs = real.numel() // 2
    grid = (triton.cdiv(num_pairs, BLOCK_SIZE),)
    # use the tensors' device and its current stream.
    with torch.cuda.device(real.device):
        _single_qubit_kernel[grid](
            real,
            imag,
            num_pairs,
            *coefficients,
            TARGET=target,
            BLOCK=BLOCK_SIZE,
            num_warps=4,
            enable_fp_fusion=False,
        )
