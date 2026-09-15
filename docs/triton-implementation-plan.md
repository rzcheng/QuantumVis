# Initial Triton source and CPU baseline plan

This follow-up starts from `fe09a9c` on `feat/triton-backend`. Existing CPU
acceptance was rechecked: 376 tests passed. The prior decision to defer GPU source
is superseded by the request to prepare it on the Mac, with NVIDIA execution still
required for acceptance. No CPU behavior or numerical tolerance will be changed.

1. Add optional dependencies and one runtime probe. Report missing packages, CUDA,
   platform, and device capability explicitly; propagate unexpected configuration
   failures. Capability detection is not proof of successful kernel execution.
2. Add one generic in-place Triton single-qubit kernel over two contiguous float32
   arrays. One logical element owns a disjoint amplitude pair under little-endian
   indexing. Use explicit complex arithmetic, masked launches, and int64 indices.
3. Connect all nine existing single-qubit gate matrices through that kernel.
   Reject CX/CZ. Download to an explicit complex64 GPU result, preserving numerical
   drift without weakening the CPU `StateVector` contract. Imports remain lazy.
4. Add deterministic analytical, randomized, circuit and raw-kernel differential
   tests. Missing hardware skips tests with reasons; compilation and numerical
   failures on compatible hardware must fail. Provide a standalone validation CLI
   that exits nonzero when unavailable or failing and captures environment details.
5. Measure complete CPU single-gate call latency with the existing public backend,
   fixed normalized inputs, correctness checks outside timing, warmups, repeated
   trials, median/p95, and raw samples plus provenance in modest JSON artifacts.
6. Document remote validation, timing boundaries, float32 error budgets, and the
   questions Ryan needs to understand. Run local tests, skip checks, lint, builds,
   examples, and benchmark smoke tests; inspect diffs and create logical commits.

Remote acceptance remains open: kernel compilation, analytical checks, randomized
differential checks, the full GPU test suite, and captured NVIDIA environment metadata.
No controlled kernels, fusion, website, or GPU performance claims belong in this slice.
