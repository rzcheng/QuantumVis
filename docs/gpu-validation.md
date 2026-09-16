# Validate the first Triton backend

The source exists; **Milestone 2 remains unaccepted until these commands run on
compatible NVIDIA hardware**. The Apple development machine cannot establish
kernel compilation, device execution, or numerical correctness. A clean local
skip is evidence about capability detection only.

## Evidence levels

| Level | What it establishes | What it does not establish |
| --- | --- | --- |
| CPU reference correctness | NumPy results against analytical, dense, and optional Qiskit references | Triton execution |
| Triton interpreter validation | Production kernel indexing/arithmetic under CPU interpretation | NVIDIA compilation, parallel execution, CUDA streams, or performance |
| Triton compile-only validation | Compilation for the recorded explicit target, when the compiler succeeds | Launch success or numerical correctness |
| Real NVIDIA device validation | Actual kernel results and runtime contracts on the recorded device/software | Performance advantage or all-device compatibility |
| Performance benchmarking | Timings for stated hardware, precision, workload, and timing boundary | Correctness without separate checks |

These are separate records. None of the first three can accept Milestone 2.

## Compatibility and installation audit

No platform has confirmed QuantaForge **device correctness** yet. The table
distinguishes package/hardware eligibility from actual execution evidence.

| Platform | Architecture | Expected status | Blocker / requirement |
| --- | --- | --- | --- |
| Desktop Linux + recent NVIDIA GPU | x86_64 | Expected, device-unverified | Python 3.11+, CUDA PyTorch, matching Triton, driver/toolchain, SM >= 8.0; pass preflight smoke |
| Jetson AGX Orin | aarch64 | Conditional, unverified | SM 8.7 eligible; exact JetPack/Python/PyTorch/Triton/NumPy stack unresolved |
| Jetson Orin NX | aarch64 | Conditional, unverified | Same software requirements; model label is insufficient |
| Jetson Orin Nano | aarch64 | Conditional, unverified | Same requirements; this is not the original Nano |
| Xavier-class Jetson | aarch64 | Unsupported by current backend | SM 7.2 is below the 8.0 minimum |
| Original Jetson Nano | aarch64 | Unsupported by current backend | SM 5.3 is below the 8.0 minimum |
| macOS Apple Silicon | arm64 | GPU unsupported; CPU confirmed | Linux/NVIDIA required; Mac CPU tests do not validate GPU execution |

Hardware references: [current NVIDIA devices](https://developer.nvidia.com/cuda/gpus),
[legacy devices](https://developer.nvidia.com/cuda/gpus/legacy), and
[Triton requirements](https://github.com/triton-lang/triton#compatibility).

Package audit on 2026-09-16: PyPI publishes CPython 3.11/3.12 Linux aarch64
manylinux 2.27/2.28 wheels for [Triton 3.6.0](https://pypi.org/project/triton/3.6.0/#files)
and [3.8.0](https://pypi.org/project/triton/3.8.0/#files). This establishes wheel
availability, **not JetPack or Orin runtime compatibility**. PyTorch 2.10's Linux
x86_64 distribution pins Triton 3.6.0; that dependency is architecture-qualified,
so an aarch64 Torch install must not be assumed to install Triton. Generic ARM
wheel availability also does not prove Jetson CUDA support.

The source uses `datetime.UTC` (Python 3.11+); pinned NumPy 2.3.5 also needs 3.11+.
Python 3.11 passed the complete locally available CPU/host suite before support
was widened from 3.12. CI now covers 3.11–3.14. Python 3.10 remains unsupported:
supporting common older JetPack Python environments would require code/pin changes
and separate verification, not just lowering package metadata. No GPU dependencies
are added to the base NumPy installation.

## Preflight

```bash
python -m quantaforge.gpu_preflight
python -m quantaforge.gpu_preflight --smoke --json
```

Default preflight checks runtime eligibility, architecture, dependency major/minor
ranges, and production kernel import without launching. `--smoke` additionally
executes X|0> through `GPUSimulator` and verifies the result with the existing
numerical check. READY means plausible for acceptance, not accepted. Exit codes
are 0 READY, 2 BLOCKED/unavailable, and 1 unexpected import/query/launch/numerical
error, with the original traceback retained on stderr.

Reports include OS/architecture, Python/PyTorch/Triton, CUDA availability,
device index/name/capability, and available Jetson files. `cuda_build_version`
comes from PyTorch. `cuda_runtime_version` uses the already-installed public
[CUDA Python getLocalRuntimeVersion API](https://nvidia.github.io/cuda-python/cuda-bindings/12.9.0/module/runtime.html)
when available; otherwise it is explicitly null with a reason. No package is
installed for metadata, and neither value is presented as the NVIDIA driver
version. The queried CUDA Python library need not be PyTorch's bundled runtime.

Follow [the three-stage runbook](first-nvidia-run.md) to save one device session.

## Hardware-free Linux checks

Use a separate Linux x86_64/Python 3.12 environment. The development Mac has no
Triton installation; neither interpreter execution nor compile-only execution
has been performed locally. Hosted interpreter validation has now passed on
Linux with Triton 3.6.0 and 3.8.0:
[147 passed, zero skips per version](https://github.com/rzcheng/quantum-sim/actions/runs/35063367779).
No NVIDIA execution or compilation is implied. No emulator or alternate kernel is added.

```bash
python3.12 -m venv .venv-interpreter
source .venv-interpreter/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install triton==3.8.0
python -m pip install --no-build-isolation -e .
python -m pip check
TRITON_INTERPRET=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_triton_interpreter.py --junitxml=validation/results/interpreter.xml
```

The `interpreter` job in `.github/workflows/triton-validation.yml` runs this path
for Triton 3.6.0 (the desktop Torch pin) and 3.8.0.
Its 147 cases use the actual `_single_qubit_kernel`, CPU PyTorch storage, and the
existing coefficient conversion and NumPy oracle. They cover X/H/phase/rotation
identities, two seeds across all nine gates, low/middle/high targets at 1/5/10
qubits, masked small blocks, multiple programs, and 8/16-gate circuits. Existing
amplitude, relative-L2, and norm budgets are reused unchanged. JUnit records the
Python/Triton/PyTorch/NumPy versions, kernel hash, seed, and evidence level.

The tests skip unless explicitly enabled. Once enabled, an unsupported platform,
missing dependency, import error, or numerical failure fails the job. Run them in
a fresh process: the public CUDA wrapper still rejects interpreted kernels and
CPU tensors. The normal CPU suite and real-device runner retain their meanings.

Triton's [documented interpreter](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
executes program instances sequentially on CPU. It cannot expose inter-program
races, compiler transformations, or device scheduling failures.

### Compile-only boundary

The manual `compile-only` job invokes Triton's public AOT CLI for `cuda:87:32`
(SM 8.7, Orin), target qubit 0 and block size 256. Once the workflow is on the
default branch and this branch is pushed:

```bash
gh workflow run triton-validation.yml --ref feat/triton-backend -f compile_only=true
```

The underlying Linux command, after the environment setup above, is:

```bash
mkdir -p validation/results/compile-only
TRITON_INTERPRET=0 python -m triton.tools.compile \
  src/quantaforge/gpu/kernels/single_qubit.py \
  --kernel-name _single_qubit_kernel --target cuda:87:32 \
  --signature '*fp32,*fp32,i32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,fp32,0,256' \
  --grid '1,1,1' --num-warps 4 --out-path validation/results/compile-only/kernel
```

**This is a diagnostic probe, not an established working CPU-only compilation
path.** In [Triton 3.8.0's CLI source](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/tools/compile.py),
output generation accesses `triton.runtime.driver.active` even with an explicit
target. The CLI also does not expose production's `enable_fp_fusion=False`
option. These limitations stop this subtask: no driver patch, internal compiler
API, or alternate kernel is introduced. The manual job may fail on a CPU runner;
its failure is retained, never converted into a pass or skip. It is opt-in so a
known upstream limitation does not block CPU development on every push.

The job saves stdout/stderr and JSON with Python/Triton versions, target,
capability, source hash, command, exit code, and status. `compilation_succeeded`
is true only after CLI success; a failed CLI leaves it null because failure may
occur after compilation but before output. Even success covers only that CLI
specialization and its default options, not the production launch configuration.
No compile-only result has been recorded yet. Do not spend another session
working around this boundary before the first device run.

## Jetson preparation and metadata

The metadata fallback currently recognizes **Jetson Orin only** (AGX Orin, Orin
NX, Orin Nano), with compute capability exactly 8.7. NVIDIA lists these models at
[SM 8.7](https://developer.nvidia.com/cuda/gpus). Xavier and the original Nano
fall below the existing 8.0 minimum; an unknown model is not assumed supported.
This is metadata compatibility, not evidence of a working Jetson kernel.

Do not install desktop CUDA wheels blindly on Jetson. Check the board, JetPack,
Python ABI, and the [NVIDIA PyTorch installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).
QuantaForge requires Python 3.11+, which may not match the available JetPack wheel.
The compatible CUDA PyTorch build and an importable Triton on Linux aarch64 must
be established first. The CPU-only interpreter environment above cannot run the
device suite. After installing the matching GPU stack, install this project with
`python -m pip install -e '.[dev]'` to avoid replacing that stack via the GPU extra.
Run `python -m pip check` and the preflight before making the trip if remote shell
access is available:

```bash
cat /etc/nv_tegra_release
cat /proc/device-tree/model
python -m quantaforge.gpu_preflight --smoke --json
```

If `nvidia-smi` runs successfully, its output remains the metadata source. If the
executable is missing, the runner requires Linux aarch64, recognizable Orin and
L4T release files, and complete CUDA device metadata from the real validator.
It saves both platform file contents in `summary.json` along with PyTorch/Triton
versions, CUDA build version, device name/index, and capability. The CUDA build
version is explicitly **not** a queried driver or runtime version; L4T identifies
the installed Jetson platform release. A failing or timed-out `nvidia-smi` is not
hidden by this fallback. Missing/malformed platform data remains a failure.

CUDA availability, a visible supported device, Triton import, 108 deterministic
checks, and all 673 real-device pytest cases with zero skips are still required.
For the first Jetson acceptance run in the prepared environment:

```bash
env -u TRITON_INTERPRET python scripts/validate_nvidia.py \
  --output validation/results/jetson-first-run
```

Use a fresh output directory when retrying. Neither the Orin metadata fallback
nor interpreter tests certify CUDA wheel/driver/toolchain compatibility.

There is no established universal Orin installation command. First obtain the
board/L4T version and match an NVIDIA wheel's Python ABI and JetPack release using
the [NVIDIA compatibility table](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).
Confirm NumPy 2 support, Torch major/minor >= 2.6 and < 3, and Triton >= 3.2 and
< 4 in that environment. NVIDIA prerelease builds still need the matching
platform stack; the preflight's version-range check is not a wheel certification.
After that stack is established, use the checkout's `.[dev]` installation and
the same preflight/runbook. Do not replace a vendor CUDA build with generic Torch
to satisfy a resolver. No source-build workaround is attempted here. If a matching
stack is unavailable, the fastest alternative is an existing x86_64 Linux
workstation with an Ampere/Ada NVIDIA GPU and working CUDA PyTorch/Triton.

## Prepare Linux/NVIDIA

Use Linux x86_64, Python 3.12 (the first device target; the package requires 3.11+),
a working NVIDIA driver, and a GPU with compute capability at least 8.0. This is
QuantaForge's conservative initial support boundary, not a claim that all earlier
hardware is inherently incapable of Triton. Confirm the installed Triton release's
[upstream compatibility requirements](https://github.com/triton-lang/triton#compatibility).

For a machine with a driver compatible with the CUDA 12.6 PyTorch build, a C
compiler, and matching Python development headers, use this explicit baseline.
The [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v2100)
provide the CUDA wheel index. A different driver/toolchain may require a different
official wheel; do not substitute an unverified combination silently.

```bash
set -e
unset TRITON_INTERPRET
nvidia-smi
git clone --branch feat/triton-backend https://github.com/rzcheng/quantum-sim.git
cd quantum-sim
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-build-isolation -e '.[gpu]'
python -m pip check
python -m quantaforge.gpu_preflight --smoke
python scripts/validate_nvidia.py --output validation/results/nvidia-first-run
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

From the checkout, capture the complete acceptance run with one command:

```bash
python scripts/validate_nvidia.py --output validation/results/nvidia-first-run
```

Use a new output directory for each attempt; existing evidence is never overwritten.
This runs the deterministic validator first and, only after it passes, the full
GPU pytest file. It requires nonempty passing validator output, positive pytest
case counts with **zero failures/errors/skips**, driver/package metadata, and
unchanged source hashes throughout the run. Otherwise it exits nonzero. On this
Mac it exits 2 with `UNAVAILABLE` and does not launch the GPU pytest step.

The directory contains `summary.json`, validator JSON in `validator.stdout`, the
full pytest JUnit report in `gpu-tests.xml` when run, and separate stdout/stderr
logs for every command, including compilation tracebacks. It also records
`nvidia-smi` (or the guarded Jetson metadata above), package versions, Git
revision/status when available, and SHA-256
hashes of simulator, validator, tests, scripts, and dependency configuration.
Source archives without Git work because the source hashes still identify the
code. Reports under `validation/results/` are ignored by Git; retain and review
them before committing a successful NVIDIA result.

Both subprocesses execute this checkout's source. The runner clears inherited
`PYTEST_ADDOPTS`/`PYTEST_PLUGINS`, disables third-party pytest plugin autoload, and
overrides configured pytest `addopts` so an accidental selection option cannot
reduce the intended full run. Set `CUDA_VISIBLE_DEVICES` before starting to select
the same visible device for both commands. Each command has a 900-second timeout;
use `--timeout SECONDS` for a slower compilation environment. Timeouts remain
failures with retained logs, not skips or evidence of numerical correctness.

Actual NVIDIA execution of this wrapper and both suites remains unverified.

### Individual commands

The quick deterministic check alone is:

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
actual driver version on desktop NVIDIA systems; use the recorded L4T/platform
metadata on the supported Jetson path. PyTorch's CUDA version is not the driver version.

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

The full suite covers every target at 1, 2, 5, 9, and 10 qubits. It also checks all
nine gates at targets 0, 8, and 17 in 18-qubit states, plus two seeded 64-gate
circuits at that size. An 18-qubit split state uses 2 MiB on the device and launches
512 programs with the current 256-pair block; host references and temporary arrays
add memory. This covers substantially more programs and higher target bits than
the two-program 10-qubit cases without requiring a large-memory GPU.

Raw-kernel tests additionally place contiguous state views at offsets 1 and 17
inside separate sentinel-filled allocations. They check both numerical output
and untouched prefix/suffix guards for masked small blocks and multi-block
states. These checks can reveal out-of-view writes and mishandling of nonzero
storage offsets; they are not a proof of arbitrary memory safety. State sizes are
powers of two, so a final partial block in a multi-block launch cannot occur with
this block size. Results are downloaded with blocking CPU transfers before
comparison; tests do not time launch submission or make performance claims.

There are now 673 real-device pytest cases. All remain skipped on the Apple host;
the additional 39 cases have not executed on NVIDIA hardware. The quick standalone
validator remains the same 108-check suite. Neither suite establishes correctness
beyond its tested state sizes, depths, precision, and device/software versions.

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
