# QuantaForge

A state-vector quantum simulator built around a NumPy reference and custom Triton
kernels. The main work is mapping quantum gates to GPU memory access and checking
the numerical results before optimizing anything.

The CPU backend works. The first GPU kernel is implemented, but **has not been
compiled or run on NVIDIA hardware yet**. There are no GPU performance claims.

## status

- CPU: X, Y, Z, H, S, T, RX, RY, RZ, CX/CNOT, CZ, probabilities, and seeded sampling.
- GPU: one generic single-qubit kernel, runtime checks, and validation tools.
  Controlled GPU gates and sampling are not implemented.
- CPU/host tests: **765 passed, 0 failed** on the development Mac, including
  185 optional Qiskit comparisons.
- Interpreter validation: 147 cases prepared using the production kernel;
  all skipped locally. Separate Linux CI added; execution is still pending.
- Compile-only validation: manual Linux probe prepared for SM 8.7, unexecuted.
  The public Triton CLI has a driver-dependent output path; no internal workaround.
- Real NVIDIA validation: all 673 device cases skipped locally; still pending.
- Benchmarks: a saved CPU baseline with raw timings. GPU measurements come later.

See [validation records](docs/validation.md) for environments and exact checks.

## quick start

From the checkout, with Python 3.12+:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/bell_state.py
python examples/ghz_state.py
```

NumPy is the only required dependency.

```python
from quantaforge import Circuit, CPUSimulator

circuit = Circuit(2).h(0).cx(0, 1)
result = CPUSimulator().run(circuit)

print(result.amplitudes)
# [0.70710678+0.j 0.+0.j 0.+0.j 0.70710678+0.j]
print(result.probabilities())
# [0.5 0.  0.  0.5]
print(result.sample(10, seed=42))  # basis indices: 0 or 3
```

Qubit 0 is the least significant bit: index 1 in a two-qubit state means `|01>`.
Rotation methods take `(target, angle)` with angles in radians. Sampling returns
basis indices without collapsing or modifying the state.

## how it works

An n-qubit state has `2**n` complex amplitudes. A gate on qubit `t` mixes pairs
whose indices differ by `1 << t`. The CPU implementation uses reshaped NumPy
arrays; it does not construct a full circuit matrix.

| Part | Role |
| --- | --- |
| `Circuit` | Ordered gates and qubit indices, shared by both backends |
| `StateVector` | Owned, read-only CPU amplitudes and measurement methods |
| `CPUSimulator` | Vectorized complex128 reference |
| `GPUSimulator` | PyTorch allocation/transfers and a custom Triton gate kernel |

The GPU kernel stores real and imaginary components in separate float32 arrays.
Each logical element owns one amplitude pair, loads both originals, then writes
both outputs in place. This avoids overlapping writes within a gate. The target
bit determines the spacing between paired amplitudes and the access pattern.

The circuit handling, gate updates, validation, and benchmark harness are written
here. NumPy supplies CPU arrays and arithmetic; PyTorch manages device storage;
Triton compiles the GPU kernel. Qiskit is used only as an optional test reference.

State-vector storage grows exponentially: a complex128 state alone needs
`16 * 2**n` bytes, or 16 GiB at 30 qubits, before temporary arrays. This version
does not support noise, mixed states, or mid-circuit measurement.

## tests

```bash
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
ruff format --check .
python -m build
```

For the pinned CPU environment, use `requirements-dev.txt`. Tests cover analytical
states, independent dense operators, seeded random circuits, inverse gates, norms,
and input validation. CPU comparisons use `atol=1e-12, rtol=0`; GPU tests have
separate float32 error budgets. Neither backend silently renormalizes its output.

Optional Qiskit checks compare amplitudes directly, including global phase:

```bash
python -m pip install -e '.[verification]'
python -m pytest -q tests/test_qiskit_reference.py
```

## gpu validation

The initial target is Linux/NVIDIA with compute capability 8.0+. Follow the
[setup guide](docs/gpu-validation.md) before installing the optional GPU stack.
Once the environment is ready:

```bash
python scripts/validate_nvidia.py --output validation/results/nvidia-first-run
```

This runs the deterministic validator and full GPU tests, saving results,
environment details, and source hashes in a new directory. It requires zero
failed or skipped GPU cases. Missing hardware returns `UNAVAILABLE`; a compiler
or numerical failure remains an error. There is no automatic CPU fallback.
Identified Jetson Orin boards can use L4T/device-tree and CUDA metadata when
`nvidia-smi` is absent. This has host-side tests, but has not run on a Jetson.
The [guide](docs/gpu-validation.md) separates CPU correctness, interpreter,
compile-only, device validation, and performance evidence, with exact commands.

## benchmarks

```bash
python -m benchmarks.benchmark_gates --qubits 8 12 16 18 \
  --operations H RX --warmups 5 --repetitions 31 \
  --output benchmarks/results/cpu-local.json
```

The timer measures a complete single-gate CPU `run()` call, including validation
and copies, after warmup. JSON output retains every trial, median/p95, precision,
hardware/software metadata, and source hashes. It is not a kernel-only timing.

The [M4 Pro baseline](benchmarks/results/cpu-m4-pro-2026-09-14.json) contains
16 cases and 496 trials. No CPU/GPU speedup or advantage over an established
simulator has been measured. See [benchmark methodology](docs/benchmarking.md).

## next steps

1. Run the prepared Linux interpreter job; retain the public compile probe's limitation.
2. Validate the single-qubit backend on compatible NVIDIA hardware; this is the next acceptance milestone.
3. Add controlled GPU gates, then measure the GPU baseline.
4. Pick one optimization from profiling evidence.
5. Build a static demo using clearly labeled recorded results.

## notes

- [architecture](docs/architecture.md) and [math](docs/math.md)
- [GPU representation and indexing](docs/gpu-representation.md)
- [external verification](docs/external-verification.md)
- [concepts to work through](docs/interview-notes.md)
- [contribution rules](AGENTS.md)
