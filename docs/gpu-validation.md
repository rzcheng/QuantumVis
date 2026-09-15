# Validate the first Triton backend

The source exists; **Milestone 2 remains unaccepted until these commands run on
compatible NVIDIA hardware**. The Apple development machine cannot establish
kernel compilation, device execution, or numerical correctness. A clean local
skip is evidence about capability detection only.

## Prepare Linux/NVIDIA

Use Linux, Python 3.12 (the first validation target; the package requires 3.12+),
a working NVIDIA driver, and a GPU with compute capability at least 8.0. This is
QuantaForge's conservative initial support boundary, not a claim that all earlier
hardware is inherently incapable of Triton. Confirm the installed Triton release's
[upstream compatibility requirements](https://github.com/triton-lang/triton#compatibility).

From this repository checkout:

```bash
nvidia-smi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,dev]'
python -m pip check
python -c 'from quantaforge.gpu import gpu_status; print(gpu_status())'
```

The optional Linux GPU extra declares `torch>=2.6,<3` and `triton>=3.2,<4`.
Allow pip to resolve PyTorch's own Triton dependency rather than independently
upgrading Triton past the version PyTorch expects. These ranges are not a claim
that every allowed combination has been tested. If PyTorch lacks a CUDA build,
select a driver-compatible CUDA wheel using the official
[PyTorch installation instructions](https://pytorch.org/get-started/locally/),
then rerun the extra installation and `pip check`. CPU users need only
`python -m pip install -e .`; GPU dependencies are never necessary for CPU use.

Do not enable `TRITON_INTERPRET`: capability detection rejects all Triton truthy
spellings (`1`, `true`, `y`, `on`, `yes`, case-insensitively) and in-process
interpreter overrides. If a notebook already imported the kernel in interpreter
mode, restart its process with interpretation disabled; the wrapper rejects cached
interpreted functions. An interpreter run cannot satisfy device acceptance.

## Run acceptance

The one-command first check is:

```bash
python -m quantaforge.validate_gpu
```

To capture metadata and per-check errors as a structured artifact, use:

```bash
python -m quantaforge.validate_gpu --json > gpu-validation.json
python -m pytest tests/test_gpu_correctness.py -ra
python -m pip freeze > gpu-validation-requirements.txt
git rev-parse HEAD > gpu-validation-commit.txt
```

Inspect the exit status of **each command** before proceeding. The validation
command returns 0 only after its deterministic checks pass, 1 for a numerical
failure, and 2 when the runtime is unavailable. Unexpected configuration,
compiler, or launch errors retain their original traceback and a nonzero exit.
When capability detection succeeds, an execution exception also writes a `FAIL`
report with environment metadata before re-raising. An exception during capability
detection itself may occur before a report can be constructed.

The readable report prints environment metadata, precision, seed, and a final
`PASS`, `FAIL`, or `UNAVAILABLE` summary. JSON additionally records the UTC
timestamp, NumPy version, tolerance policy, each completed check's maximum
amplitude error, relative L2 error, and squared-norm drift. Metadata includes
GPU name/compute capability, device index, Python, PyTorch, Triton, and the CUDA
version associated with PyTorch. Save `nvidia-smi` output separately to retain the
actual driver version; PyTorch's CUDA version is not the driver version.

The first command currently performs 108 deterministic checks: analytical states,
all nine supported gates on seeded inputs, inverse rotations, circuits up to
64 gates, and independent dense comparisons for arbitrary complex 2x2 unitaries.
`--device 1` selects another visible CUDA device. The full pytest file expands
sizes/targets/seeds and also exercises input preservation, output ownership and
probabilities. Run the full repository suite with `python -m pytest -ra` as well.

Pytest deliberately skips real-GPU tests when capability is absent, so pytest's
exit code alone is insufficient for GPU acceptance: verify that GPU tests
**passed with zero skips**, and that the one-command validation exited 0. On the
Apple development host, no GPU tests pass; all are explicitly skipped.

## Numerical contract

CPU reference amplitudes remain `complex128`. Device storage and arithmetic use
two contiguous `float32` arrays, with downloaded results in `complex64`. The
public backend validates the original normalized input using the unchanged CPU
contract, then rounds for upload. It does not renormalize GPU output or wrap it
in the CPU `StateVector` type.

These thresholds are **prespecified, not yet validated on NVIDIA hardware**:

| Quantity | One gate | Depth d, 2 through 64 |
| --- | --- | --- |
| Componentwise absolute tolerance | `1e-6` | `1e-6 + (d-1) * 4 * eps32` |
| Componentwise relative tolerance | `1e-5` | `1e-5` |
| Relative L2 error | `<= 1e-6` | `<= 1e-6 + (d-1) * 4 * eps32` |
| Absolute squared-norm drift | `<= 1e-6` | `<= 1e-6 + (d-1) * 8 * eps32` |

Here `eps32 = 2^-23`, approximately `1.1921e-7`. Input and gate coefficient
rounding and the short real multiply/add chains give single-gate errors on this
scale. The `1e-6` allowance is roughly eight float32 epsilons; the relative term
covers nonzero components, while the absolute term handles cancellation near
zero. These are conservative engineering thresholds, not a proven universal
forward-error theorem. For short circuits, unitary transformations do not amplify
vector errors in exact arithmetic; the linear depth terms budget accumulation
of local rounding error, with twice the coefficient for squared-norm drift.
At depth 64 the L2 limit is about `3.10e-5` and the norm limit `6.11e-5`.
No arbitrary-depth guarantee is made. See PyTorch's
[numerical accuracy discussion](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
for why arithmetic ordering and platforms can change floating-point results.

Diagnostics accumulate in complex128/float64. Relative L2 is
`norm(actual - expected) / norm(expected)`; squared-norm drift is
`abs(vdot(actual, actual).real - initial_squared_norm)`. Amplitude agreement is
phase-sensitive; global phase is not discarded. Norm alone would accept a wrong
unitary or a gate acting on the wrong qubit, so it cannot replace differential
or analytical checks.

Public comparisons include input quantization relative to the original CPU
state. Raw-kernel tests instead round both the input and arbitrary unitary
coefficients first, then use an independent small dense Kronecker operator in
complex128. This separates indexing/arithmetic errors from input rounding without
feeding a float32-rounded state into the strict CPU normalization validator.
The raw check uses the rounded input's squared norm as its drift reference.

The full suite covers every target at 1, 2, 5, 9, and 10 qubits. With the current
256-pair launch block these include masked small blocks, one full block, and
multiple full blocks. State sizes are powers of two, so a final partial block in
a multi-block launch cannot occur with this block size. Results are downloaded
with blocking CPU transfers before comparison; tests do not time asynchronous
launch submission or make performance claims.

## First failure points

- **UNAVAILABLE:** Inspect the reason: missing package, unsupported OS, CPU-only
  PyTorch build, unavailable CUDA, insufficient capability, or interpreter mode.
  Check `nvidia-smi`, the active virtual environment, and `CUDA_VISIBLE_DEVICES`.
- **Installed-package import failure:** Keep the traceback. Check `pip check` and
  the installed PyTorch/Triton pair. This is not a numerical test skip.
- **First kernel compilation/launch:** Save the traceback and environment report.
  Inspect Triton API compatibility, CUDA driver/runtime compatibility, and the
  coefficient launch arguments before changing the mathematical code.
- **Only some targets/sizes fail:** Inspect insertion of the zero target bit,
  masks, index widths, and duplicate writes. Target 0 mixes adjacent amplitudes;
  the highest target mixes register halves.
- **Phase or rotation failures:** Inspect signs of imaginary coefficients and the
  half-angle convention. The tests retain global phase intentionally.
- **NaNs or norm drift:** Inspect both original amplitudes being loaded before
  stores and the split real/imag layout. Reproduce the recorded case/seed; never
  normalize away the error or loosen tolerances to get a pass.

Remote acceptance requires kernel compilation, analytical checks, randomized
CPU/GPU checks, the complete GPU pytest file passing, and saved environment
metadata. Performance measurements and optimization follow that acceptance.
