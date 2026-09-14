# QuantaForge

QuantaForge is a state-vector quantum circuit simulator being built to investigate
GPU numerical computing: mapping gates to parallel work, understanding memory
access, validating floating-point results, and measuring performance honestly.
The CPU reference is the first implementation. Custom Triton GPU kernels are the
next milestone; there are no GPU results or speedup claims yet.

Milestones 0 and 1 are implemented and locally validated: **376 CPU tests pass**
on Python 3.12.11 / NumPy 2.3.5. CPU CI is configured for Python 3.12–3.14;
the hosted workflow has not yet run. See the [validation record](docs/validation.md).

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
random sampling. A future GPU backend will reuse the circuit description and use
PyTorch for device storage and Triton for custom kernels, rather than delegating
execution to an existing quantum simulator.

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
backend. No custom GPU kernel has been implemented or accepted on this host.
The representation decision proposes separate contiguous `float32` real and
imaginary arrays; it remains a baseline to verify on a compatible device.

The next milestone is one generic single-qubit Triton kernel, an explicit runtime
availability check, and CPU/GPU differential tests. Tests must skip clearly when
CUDA/Triton is unavailable; requesting GPU execution must report the missing
capability rather than silently run on the CPU.

Later benchmarks will compare a vectorized NumPy baseline with straightforward
and optimized GPU kernels. They must record hardware, software, precision, seeds,
state size, and workload; separate compilation, transfers, kernel execution,
gate-call latency, and whole-circuit runtime; warm up and synchronize GPU work;
and retain repeated trials with median and p95 in JSON artifacts. No performance
measurements or benchmark artifacts are claimed in this milestone. There is not
yet evidence that this implementation beats any established simulator.

One optimization will be selected only after correctness and measurements identify
a bottleneck. A static React/TypeScript demonstration follows meaningful simulator
results. Its circuit outputs and benchmark charts will be labeled precomputed or
recorded; GitHub Pages will not run the Python/Triton engine live.

## Milestones

0. Repository foundation — implemented; local tooling passes, hosted CI pending.
1. Correct CPU reference — accepted locally with 376 passing tests.
2. One GPU kernel and differential correctness on compatible hardware.
3. Required GPU gate coverage and randomized circuits.
4. Reproducible benchmark harness and saved results.
5. One measured optimization with before/after evidence.
6. Static engineering demonstration and GitHub Pages deployment.

See the [initial implementation plan](docs/implementation-plan.md) for environment
findings and acceptance boundaries, and [AGENTS.md](AGENTS.md) for engineering rules.
