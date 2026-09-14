# Architecture

QuantaForge separates circuit description, state ownership, and execution. The
first backend is a NumPy CPU reference. GPU execution is a later implementation
of the same gate semantics, not a dependency of the CPU package.

## Current data flow

```text
Circuit(num_qubits)
    | validated ordered gate operations
    v
CPUSimulator.run(circuit, initial_state=...)
    | vectorized complex128 amplitude updates
    v
StateVector
    |                     |
    v                     v
probabilities()       sample(shots, seed=...)
```

`Circuit` records gate operations and the register size. Its small chainable API
supports the required gates without a parser, optimizer, plugin registry, or DSL.
Each operation records a name and target, plus a control or angle where needed.
Execution does not change the circuit description.

`StateVector` owns a one-dimensional `complex128` array of length `2**num_qubits`.
States must have at least one qubit, finite amplitudes, and squared norm within
`1e-12` of one (an absolute tolerance with no relative term). Caller-owned arrays
are copied, and the exposed amplitude array is read-only. `StateVector.zero(n)`
initializes the computational basis state with amplitude one at index zero.
`CPUSimulator.run` defaults to this
state or accepts a compatible normalized input state.

`CPUSimulator` creates working storage, applies operations sequentially, and returns
a new state. There is no persistent register hidden inside the backend. Probabilities
are the squared amplitude magnitudes. Sampling uses a local NumPy random generator
and returns integer basis indices without state collapse or global RNG mutation.
After state validation, only the RNG's probability weights are rescaled to sum to
one; returned probabilities and stored amplitudes are not rescaled.

## CPU execution

The basis is little endian: bit `q` of an amplitude index is qubit `q`. To apply a
gate at target `t`, the simulator groups amplitudes separated by `2**t`. Reshaped
views expose the target-zero and target-one halves of each block so NumPy can apply
the 2 by 2 matrix to all pairs. Original pair values must be preserved until both
outputs are computed; an overwritten first output cannot be used as an input to
the second.

CX swaps the target amplitudes only where the control is one. CZ negates the
amplitude only where control and target are both one. Distinct target/control bits
are required. Full-system dense matrices belong only in small independent tests.

The raw CPU state uses `16 * 2**n` bytes. Working-state copies, pair temporaries,
and returned state storage can multiply the peak footprint. This reference favors
clear, vectorized correctness over minimizing every allocation. It remains a
meaningful NumPy baseline, rather than an intentionally slow Python amplitude loop.

## Validation boundaries

Gate and circuit validation rejects unknown operations, invalid targets/controls,
and malformed rotation angles before execution. State validation rejects shapes,
values, or norms that do not describe an accepted pure state. A supplied state must
match the circuit register size. Measurement validation rejects invalid shot counts.

Unitary operations do not trigger automatic renormalization: a numerical bug must
not be concealed by normalization after every gate. Small `complex128` tests compare
amplitudes and norms at `rtol=0`, `atol=1e-12`; a future GPU backend requires a
separate tolerance justified against precision, circuit depth, and device results.

Independent dense Kronecker operators test small-system semantics and endianness.
Analytical circuits and inverse operations provide checks that do not depend on the
same implementation path. Randomized tests cover multiple register sizes and all
valid target/control positions, with fixed seeds for reproducibility.

## GPU boundary and next slice

The next implementation will use PyTorch for device allocation and Triton for one
generic single-qubit kernel. The proposed representation is split contiguous real
and imaginary `float32` arrays; see the [decision note](gpu-representation.md) for
library support, alternatives, and reasons. There is no GPU backend yet and no
unverified kernel scaffold in the CPU milestone.

Each future logical kernel element will own one disjoint amplitude pair. This follows
directly from the math and avoids overlapping writes for a single gate. Device availability
will be checked explicitly. CPU-only installations must remain usable without
importing PyTorch or Triton. GPU tests will skip with a concrete reason if unavailable;
an explicit request to run on the GPU will fail clearly instead of switching backend.

The first GPU acceptance requires randomized and analytical agreement with the CPU
reference on a compatible device. Adding all GPU gates, measuring runtime, and
optimizing are separate steps. No memory layout or kernel configuration is considered
optimal before it is measured.

## Deferred systems

Benchmarks will consume the public backend APIs and save metadata plus trial data
as JSON. The static website will consume exported circuit and benchmark artifacts,
not call Python. Neither has been implemented yet. There is no backend server,
database, authentication system, cloud infrastructure, or browser GPU runtime in
this first milestone.
