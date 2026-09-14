# Initial implementation plan

## Environment observed on 2026-09-14

The workspace started empty, without Git metadata. Host: macOS arm64, Apple M4 Pro
(16 GPU cores). Python 3.12.11 has NumPy 2.3.5 and pytest 9.0.3. The default Python
is 3.14.2, with NumPy 2.4.2 and PyTorch 2.10.0; that PyTorch build has no CUDA
support, reports CUDA unavailable, and reports MPS unavailable in this session.
Triton and `nvidia-smi` are absent. Node 25.5.0 and npm 11.8.0 are available.
No GPU framework installation is needed to complete the CPU work.

## Milestone 0: foundation

Create a Python 3.12+ package in `src/quantaforge`, a small development dependency
set (pytest, Ruff, build), CPU CI, engineering rules, and technical documentation.
Initialize one coherent Git branch and record logical commits where permitted.

## Milestone 1: CPU reference

- Circuit descriptions contain validated gate operations, independent of execution.
- `Circuit(n).h(0).cx(0, 1)` builds a circuit; `CPUSimulator().run(circuit)` returns
  a `StateVector` with amplitudes, probabilities, and seeded measurement sampling.
- Use NumPy `complex128`, little-endian indexing (qubit 0 is the least significant
  bit), and vectorized disjoint amplitude pairs. Implement X/Y/Z/H/S/T/RX/RY/RZ
  and CX/CZ without constructing full-system matrices in the simulator.
- Accept a normalized input state for differential tests; otherwise start at zero.
  Copy caller inputs, reject malformed/nonfinite/unnormalized inputs, and do not
  renormalize after gates to hide errors. Sampling is non-collapsing, in the
  computational basis, using a local NumPy random generator.
- Validate analytical states, inverse operations, all target/control positions,
  several sizes, seeded random states, norm preservation, sampling behavior, and
  invalid inputs. Tests use independent dense Kronecker matrices for small systems.
- Run examples, the complete test suite, lint/format checks, packaging checks, and
  inspect the resulting diff. Add an optional trusted-simulator cross-check if it
  is practical without expanding the required dependency set.

## Milestone 2: smallest GPU slice (requires another host)

First document native complex, split real/imaginary, and interleaved representations
against official PyTorch/Triton support. The proposed baseline is two contiguous
`float32` arrays owned by PyTorch, with explicit complex arithmetic in one generic
single-qubit Triton kernel; confirm this decision in the representation note.
Each logical kernel element owns one disjoint amplitude pair. Avoid a complex-number framework.

On a compatible Linux CUDA host, pin a working PyTorch/Triton pair, implement one
generic single-qubit kernel, and test every target on small/random normalized states
against the CPU oracle, with justified float32 tolerances. GPU tests must skip with
an explicit reason when the runtime/device is unavailable; requesting GPU execution
must fail clearly rather than silently falling back to CPU.

This host cannot satisfy GPU acceptance criteria. Do not add unexecuted kernel
code or call Milestone 2 complete. GPU coverage, timing, optimization, and the static
React demo remain subsequent milestones.

## Risks and boundaries

- State memory grows as 2^n; vectorized CPU temporaries add to the raw 16 * 2^n bytes.
- Endianness, Y/rotation phases, and controlled-bit masking are primary correctness risks.
- Float32 GPU tolerances and error growth require device evidence; CPU agreement
  alone is insufficient if both implementations share an indexing bug.
- GPU compilation, synchronization, launch overhead, transfers, and steady-state
  kernel work must be measured separately before any performance claim.
- The first pass does not claim speedups or implement speculative optimizations.
