# GPU amplitude representation decision

Decision date: 2026-09-14. Status: implemented in initial source for Milestone 2;
no GPU kernel has been compiled, executed, or benchmarked in this repository yet.

## Observed environment and upstream evidence

The local inspection found macOS on arm64 (Apple M4 Pro), Python 3.14.2,
PyTorch 2.10.0, `torch.version.cuda is None`,
`torch.cuda.is_available() == False`, and no installed Triton. These are local
observations, not a statement about the latest PyTorch release.

The upstream release listing identified Triton **3.8.0** as the latest release
when checked. Its scalar dtype definitions include real floating point and
integer types, with no native complex scalar dtype. Therefore, PyTorch complex
tensor support does not imply that a Triton kernel can use complex loads and
arithmetic directly. Use real pointers and expand the arithmetic explicitly.
[Triton releases](https://github.com/triton-lang/triton/releases),
[Triton 3.8.0 dtype implementation](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/core.py).

Triton 3.8.0 lists Linux and NVIDIA GPUs with compute capability 8.0 or later,
or supported AMD GPUs with ROCm 6.2 or later. The Apple GPU cannot validate this
backend. The first QuantaForge GPU target will be Linux/NVIDIA; a CUDA-enabled
PyTorch installation, compatible driver, and compatible Triton/PyTorch versions
must be selected and recorded on that machine before implementation. Do not
assume that independently installing the newest releases makes a compatible
pair. [Triton 3.8.0 compatibility](https://github.com/triton-lang/triton/blob/v3.8.0/README.md#compatibility).

## Representation choice

The source uses **two contiguous, one-dimensional `torch.float32` tensors** of length
`N = 2**n`: one real array and one imaginary array. Together these represent
complex64 amplitudes. Keep the CPU oracle in NumPy complex128. This is a choice
for transparent indexing and arithmetic, not a performance conclusion.

| Candidate | Kernel implications | Initial decision |
| --- | --- | --- |
| Native PyTorch complex64 | Convenient host API, but requires a real view before entering this Triton kernel. | Keep at interoperability boundaries if useful. |
| Two contiguous real/imag arrays | Two pointers with identical amplitude indices; explicit real arithmetic. Conversion from interleaved complex storage requires copies. | First baseline. |
| Contiguous `[N, 2]` real array | One pointer; real/imag offsets are `2*i` and `2*i+1`. Can share storage with a PyTorch complex tensor. | Viable later comparison; no claim it is slower. |

PyTorch 2.10 documents copy-free `view_as_real`/`view_as_complex` conversion and
noncontiguous `.real`/`.imag` views. A local CPU probe confirmed a complex64
tensor's real view has strides `(2, 1)`, shares its pointer, and `.real` has
stride `(2,)`. **Passing `.real` and `.imag` directly does not produce the
chosen contiguous split representation.** Explicitly copy them once at the
execution boundary. [PyTorch 2.10 complex tensors](https://docs.pytorch.org/docs/2.10/complex_numbers.html).

For a trailing-dimension alternative, `view_as_complex` requires float32 or
float64, final dimension size two and stride one, and even outer strides.
[PyTorch 2.10 view constraints](https://docs.pytorch.org/docs/2.10/generated/torch.view_as_complex.html).

## First kernel and index mapping

The source implements one generic 2-by-2 single-qubit matrix kernel. Host code supplies its
eight real coefficients; rotations are constructed on the host. No complex
class, fusion system, or specialized gate family is needed to prove this slice.

Qubit zero is the least significant state index bit. For target `q`, each
logical element handles pair number `p` in `[0, N/2)`:

```text
low = p & ((1 << q) - 1)
i0 = ((p >> q) << (q + 1)) | low
i1 = i0 | (1 << q)

b0 = U00*a[i0] + U01*a[i1]
b1 = U10*a[i0] + U11*a[i1]
```

Load both complex amplitudes before storing either result. Expand each
multiply as `(u_r*a_r - u_i*a_i, u_r*a_i + u_i*a_r)`. The pairs are disjoint
and cover the state exactly once, so this mapping permits in-place updates
without atomics or cross-program communication. Mask the final partial block.
Validate index width and shift bounds explicitly; do not rely on overflow.
Triton schedules blocks of logical elements, so this is not a claim that one
element always corresponds to one CUDA thread. Its vector tutorial demonstrates
the program/block/mask pattern.
[Triton vector addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).

The initial implementation uses 256 pair elements per program, four warps, int64
index arithmetic widened before multiplying the program index, and masked loads
and stores. A 39-qubit cap protects the one-dimensional launch-grid bound; actual
device memory is far more restrictive (39 qubits would require 4 TiB raw storage).
The same storage is updated in place because pairs are disjoint and all four
real input components are loaded before writes. An out-of-place buffer would add
allocation/storage without resolving any cross-pair dependency for this operation.
Real and imaginary tensors must have distinct underlying storage.

At target zero, target-zero indices are even and target-one indices are odd: each
load stream has stride two. At high targets, consecutive pair numbers traverse
long contiguous runs in each half, with paired values separated by `2**target`.
All arithmetic and coefficients are float32. The initial launch disables floating
point fusion (`enable_fp_fusion=False`) to keep the baseline operation ordering
explicit; this is not a performance optimization or a claim about fastest settings.

The state occupies `8*N` bytes at this precision. A full gate logically reads
and writes it once, totaling `16*N` bytes, excluding coefficients, conversions,
and cache/transaction effects. The target bit changes spacing between paired
amplitudes and contiguous runs within each array. Both split and interleaved
layouts have plausible access tradeoffs; actual coalescing, bandwidth, and
latency remain measurements to make. Keep upload, layout conversion, and
download outside kernel timings, while reporting them in end-to-end timings.

## Acceptance and remaining uncertainty

Compare public CPU/GPU execution from the same normalized complex128 input,
including float32 input quantization in the GPU error budget. Separately compare
the raw kernel with independent dense NumPy arithmetic on exactly the same rounded
inputs. Do not feed rounded inputs into the current public CPU oracle: its strict
`1e-12` norm validation correctly rejects many float32-rounded states. For example,
rounded H|0> has squared norm about `0.99999996577`. GPU result ownership will also
use the explicit `GPUResult` boundary rather than wrapping float32 results in
the current strict CPU `StateVector` or silently renormalizing them.

Cover basis states, random normalized complex states, several sizes, every target
position, and partial launch blocks. Test H twice,
inverse rotations, state norm, and relative L2 error as well as amplitudes. The
initial **proposed, unverified** single-gate thresholds are amplitude
`atol=1e-6, rtol=1e-5`, relative L2 error `<=1e-6`, and norm drift `<=1e-6`.
Absolute tolerance alone can hide errors in small amplitudes of large states.
Long-circuit tolerances need a separate depth/error study; never loosen a test
just to make a failure disappear. Floating point operation ordering can cause
CPU/GPU differences. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html).

The validation CLI and pytest suite now encode these single-gate thresholds and
prespecified depth budgets through 64 gates; see [GPU validation](gpu-validation.md)
for exact formulas and rationale. They are still unverified on NVIDIA hardware.

CPU use must require neither PyTorch nor Triton. Future GPU tests should skip
with an explicit missing-hardware/dependency reason in CPU-only environments;
compilation or numerical failures on supported hardware must fail. Explicit
GPU execution requests should raise a clear error instead of silently running
on CPU. No GPU speedup, numerical agreement, or Milestone 2 completion is claimed.
