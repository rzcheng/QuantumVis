# QuantaForge

QuantaForge is a state-vector quantum circuit simulator being built to investigate
GPU numerical computing: mapping gates to parallel work, understanding memory
access, validating floating-point results, and measuring performance honestly.
The CPU reference is validated locally. An initial custom Triton single-qubit
backend now has source and validation tools, but has **not executed on NVIDIA
hardware**. There are no GPU correctness results or speedup claims yet.

Milestones 0 and 1 are implemented and locally validated: **376 CPU tests pass**
on Python 3.12.11 / NumPy 2.3.5. CPU CI is configured for Python 3.12–3.14;
the hosted workflow has not yet run. See the [validation record](docs/validation.md).

Implemented: the CPU engine and measurement API, optional GPU runtime detection,
one generic Triton single-qubit kernel and simulator interface, deterministic GPU
validation, and a reproducible CPU gate-call benchmark harness. Validated locally:
CPU behavior, GPU host-side boundaries, missing-runtime reporting, packaging, and
CPU benchmark prechecks. Awaiting NVIDIA hardware: Triton compilation/execution,
analytical and differential GPU checks, the full GPU pytest suite, and GPU timings.
**Milestone 2 remains incomplete.**

The current local suite reports **552 passed, 0 failed, 634 GPU cases skipped**.
The added passing cases test host boundaries, validation reporting/error budgets,
acceptance evidence, and benchmark correctness; they do not establish Triton execution.

The current scope is pure-state simulation of X, Y, Z, H, S, T, RX, RY, RZ, CX
(CNOT), and CZ, with probabilities and seeded computational-basis sampling.
It does not model noise, mixed states, or mid-circuit measurement.

## Run it

Use Python 3.12 or newer. NumPy is the only runtime dependency. The development
dependencies provide pytest, Ruff, and the package build frontend.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python examples/bell_state.py
python examples/ghz_state.py
```

For development without the pinned reference environment, use
`python -m pip install -e '.[dev]'`. The GPU extra is separate and Linux-only;
see the [NVIDIA validation guide](docs/gpu-validation.md). NumPy remains the only
required runtime dependency. PyTorch/Triton are never imported by CPU execution.

```python
from quantaforge import Circuit, CPUSimulator

circuit = Circuit(2).h(0).cx(0, 1)
result = CPUSimulator().run(circuit)

print(result.amplitudes)
# [0.70710678+0.j 0.+0.j 0.+0.j 0.70710678+0.j]
print(result.probabilities())
# [0.5 0.  0.  0.5]
print(result.sample(10, seed=42))  # Integer basis indices, each either 0 or 3.
```

Qubit 0 is the least significant bit of a basis index. The integer `1` in a
two-qubit register denotes `|01>`, with printed kets ordered `|q1 q0>`.
Angles are in radians; rotations use `circuit.rx(target, angle)` and likewise for
RY and RZ. Sampling returns independent draws from the final state's distribution
without collapsing or modifying the state. A seed makes repeated runs reproducible
within the recorded NumPy environment.

## How the simulator works

An n-qubit pure state has 2^n complex amplitudes. A single-qubit gate updates
disjoint pairs whose indices differ only at the target bit. The CPU implementation
uses vectorized NumPy array operations on those pairs; it never builds a dense
2^n by 2^n circuit operator. Controlled gates select pairs for which the control
bit is one.

The package separates three concerns:

| Boundary | Responsibility |
| --- | --- |
| `Circuit` and gate definitions | Describe and validate a sequential circuit. |
| `StateVector` and measurement | Own amplitudes, calculate probabilities, and sample outcomes. |
| `CPUSimulator` | Execute the shared circuit description with NumPy. |

CPU amplitudes use `complex128`. The simulator starts in `|0...0>` unless given a
normalized input state, preserves caller inputs, and returns read-only amplitudes.
Gate execution does not renormalize the state: drift must remain visible to tests.
The raw state occupies `16 * 2**n` bytes, before working copies and temporary arrays;
30 qubits alone require 16 GiB. This is exponential-memory simulation, not a way
to simulate arbitrary quantum computers cheaply.

The gate definitions, circuit execution, state validation, and measurement wrapper
are implemented here. NumPy supplies numerical storage, vectorized arithmetic, and
random sampling. The initial GPU backend reuses the circuit description, uses
PyTorch for device storage and transfers, and supplies its own Triton kernel for
state updates. It does not delegate execution to another quantum simulator.

Read the [architecture](docs/architecture.md), [mathematical conventions](docs/math.md),
and [GPU complex-number representation decision](docs/gpu-representation.md).

## Correctness and reproducibility

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m build
```

The test strategy combines analytical states (including Bell and GHZ), inverse
gates and rotations, randomized normalized inputs, norm preservation, measurement
behavior, and invalid-input checks. Small-system tests construct independent dense
operators to check indexing and gate action; these operators are test oracles, not
the simulator implementation. CPU tests use `atol=1e-12, rtol=0` for small tested
circuits in `complex128` (example assertions use `1e-12` for both). This tolerance is not a
promise for arbitrary circuit depth or future `float32` GPU execution.

CPU CI runs the checks without installing a GPU stack. Random tests use explicit
seeds. A trusted external simulator is an optional additional cross-check, not a
runtime dependency or a substitute for analytical tests.

## GPU status and performance work

The development host is an Apple M4 Pro running macOS. It has no NVIDIA CUDA device
or driver and no Triton installation, so it cannot validate the intended GPU
backend. Source exists for one generic in-place complex 2x2 kernel over separate
contiguous `float32` real and imaginary arrays. All nine single-qubit gates use
that kernel; CX/CZ explicitly raise `NotImplementedError` on the GPU backend.

`from quantaforge.gpu import GPUSimulator, gpu_status` imports without loading
optional frameworks. `gpu_status()` probes prerequisites and explains failures;
`GPUSimulator().run(circuit)` explicitly requires them. There is no CPU fallback.
Its `GPUResult` contains downloaded read-only complex64 amplitudes and probabilities
without silently correcting norm drift; GPU sampling is not implemented yet.

After installing on compatible Linux/NVIDIA hardware, run:

```bash
python scripts/validate_nvidia.py --output validation/results/nvidia-first-run
```

This runs the validator and full GPU tests, captures driver/package/source evidence,
and requires zero failed or skipped GPU cases before reporting PASS. It creates a
new evidence directory for every attempt and exits nonzero if unavailable. The
quick check remains `python -m quantaforge.validate_gpu`; the full suite is
`python -m pytest -q -rs tests/test_gpu_correctness.py`.

The validator exits nonzero if unavailable or failing. GPU pytest tests skip only
for missing prerequisites, with explicit reasons. Compilation and numerical
failures on compatible hardware fail. A capability probe or skipped test suite is
not GPU validation. Installation and failure diagnosis are in the
[GPU validation guide](docs/gpu-validation.md).

Later benchmarks will compare a vectorized NumPy baseline with straightforward
and optimized GPU kernels. They must record hardware, software, precision, seeds,
state size, and workload; separate compilation, transfers, kernel execution,
gate-call latency, and whole-circuit runtime; warm up and synchronize GPU work;
and retain repeated trials with median and p95 in JSON artifacts. The current
CPU harness already records those host-side fields and raw timing samples:

```bash
python -m benchmarks.benchmark_gates --qubits 8 12 16 18 \
  --operations H RX --warmups 5 --repetitions 31 \
  --output benchmarks/results/cpu-local.json
```

This measures complete one-gate `CPUSimulator.run()` latency, including validation,
copies and result construction, after warmups. It excludes process startup and
test/setup work. See [benchmark methodology](docs/benchmarking.md) and saved small
CPU artifacts in `benchmarks/results/`. There is no evidence yet that this
implementation beats an established simulator, and no CPU/GPU ratio is reported.

The [recorded M4 Pro sweep](benchmarks/results/cpu-m4-pro-2026-09-14.json) retains
16 cases and 496 raw trials, including source fingerprints and numerical prechecks.

One optimization will be selected only after correctness and measurements identify
a bottleneck. A static React/TypeScript demonstration follows meaningful simulator
results. Its circuit outputs and benchmark charts will be labeled precomputed or
recorded; GitHub Pages will not run the Python/Triton engine live.

## Milestones

0. Repository foundation — implemented; local tooling passes, hosted CI pending.
1. Correct CPU reference — accepted locally with 376 passing tests.
2. One GPU kernel — source and validation tooling implemented; NVIDIA acceptance pending.
3. Required GPU gate coverage and randomized circuits.
4. Reproducible benchmark harness and saved results.
5. One measured optimization with before/after evidence.
6. Static engineering demonstration and GitHub Pages deployment.

See the [initial implementation plan](docs/implementation-plan.md) for environment
findings and acceptance boundaries, and [AGENTS.md](AGENTS.md) for engineering rules.
The [follow-up plan](docs/triton-implementation-plan.md) separates local preparation
from remote acceptance. [Technical understanding notes](docs/interview-notes.md)
identify the concepts to work through before making portfolio claims.
