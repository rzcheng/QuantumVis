# CPU benchmark methodology

Run from a repository checkout with QuantaForge installed and Python 3.12+:

```bash
python -m benchmarks.benchmark_gates --qubits 8 12 16 18 \
  --operations H RX --warmups 5 --repetitions 31 \
  --seed 20260914 --output benchmarks/results/cpu-local.json
```

This measures the vectorized NumPy CPU implementation. There are no GPU timing,
speedup, or optimization results. The default sweep uses 8, 10, 12, 14, 16, and 18
qubits. All nine single-qubit gate names are accepted by `--operations`; rotations
use the recorded angle 0.731 radians. Target 0 and target `n - 1` are measured
(deduplicated for one qubit). The command rejects duplicate cases, fewer than three
repetitions, missing warmups, and states larger than 20 qubits before allocation.
Twenty qubits use 16 MiB for one raw complex128 state; execution and validation need
several additional arrays. This guard is deliberate; extend it only after checking
the available memory and peak working footprint. Trials and warmups are capped at
10,000 per case. An existing output file is never overwritten.

## Exact timing boundary

The metric is **complete single-gate `CPUSimulator.run()` latency**, in nanoseconds:

1. Generate one seeded normalized random complex128 input for each register size.
   Construct its one-gate circuit outside the timer.
2. Check the output against independent full-index partner arithmetic, using
   `atol=1e-12, rtol=0`, and verify squared norm within `1e-12` of one.
   This reference uses direct gate formulas, not the simulator's matrix or reshape
   functions; tests additionally check it against independent small dense operators.
3. Warm the same call, then time repeated independent `run()` calls using
   `time.perf_counter_ns()`. Each call starts from the same read-only input, so
   trials do not accumulate gate effects or normalization drift.
4. Record every duration, its median, and its p95 (`numpy.percentile`, linear
   interpolation). Report min/max to make outliers visible. All trials are retained.

The timer includes input validation, the simulator's input copy, gate matrix
construction, vectorized NumPy work and temporaries, output norm validation, and
the owned result copy. It excludes process startup, imports, state generation,
circuit construction, correctness checks, warmups, destruction of the previous
result, metadata collection, and JSON serialization. It therefore measures the
current public API's cost, **not just amplitude arithmetic**. The returned state is
created within the timer, then released outside it. No Python amplitude loop is
used as a performance baseline. This CPU work is synchronous; no GPU timing or
synchronization is involved.

"Steady state" here means warmed repeated calls from one unchanged state, not a
persistent in-place register or an entire circuit. Cache reuse is intentional and
should be preserved in comparisons. The sweep orders sizes, operations, then low
and high targets deterministically. It is not randomized across cases; thermal
drift, frequency scaling, allocator state, BLAS threading, and concurrent processes
can affect results. Close other heavy workloads before a recorded run. A p95 from
31 observations is descriptive and noisy, not a confidence interval. Repeat full
runs to assess variability before claiming a trend or improvement.

## Artifact and reproduction

Schema version 1 records the metric boundary, precision, seed and RNG rule,
timestamp, platform, CPU model when discoverable, Python/NumPy/package versions,
NumPy build configuration, thread environment settings, optional runtime thread
pool introspection, git revision and dirty flag, SHA-256 hashes of the harness and
CPU source files, and each workload's state size, target, operation/angle,
warmup/repetition count, precheck errors, raw samples, and
statistics. Missing hardware/git/thread-pool information is `null`, not invented.
The harness does not set thread counts; reproduce the recorded environment or
explicitly choose counts in the shell before Python starts and save a new artifact.
The benchmark modules are repository tools, not a second simulator API.

Before any allocation or timing, the harness also hashes the **imported** CPU
modules and checks them against this checkout's source. This prevents a different
installed QuantaForge version from being timed under the checkout's hashes. A
byte-identical wheel or source copy is accepted; a mismatch exits with an editable
installation hint and creates no result. New reports include
`metadata.loaded_simulator_sources` with paths and hashes. This is additional
schema-version-1 provenance metadata; older artifacts lack that field. The timer
boundary and statistical methodology are unchanged.

Seeds use `SeedSequence([seed, num_qubits])`, so selecting a different subset or
reordering the qubit sweep preserves the input for each size. All operations and
targets at one size share that exact input. Saved files are small CPU snapshots,
not universal performance guarantees. Dirty working-tree artifacts record that
fact; the recorded source hashes identify the measured code even before a commit.
Interpret these artifacts alongside their containing commit. Raw timings
should not be edited, filtered, or regenerated just to improve appearances.

The first [recorded CPU snapshot](../benchmarks/results/cpu-m4-pro-2026-09-14.json)
contains H/RX at 8, 12, 16, and 18 qubits, two targets each, five warmups and 31
trials per case, on Apple M4 Pro / Python 3.12.11 / NumPy 2.3.5. Its timestamp is
2026-09-15 UTC (September 14 locally). CPU brand discovery was blocked by the
development sandbox; a preliminary metadata check reproduced that restriction,
then the same sweep ran with host metadata access. The saved file contains that
complete run, without filtering trials. It reports a dirty working tree plus
source hashes because the harness was still uncommitted when measured.

For future GPU benchmarks, use explicit device synchronization before and after
timed regions and report compilation, transfers, kernel execution, complete gate
latency, and full circuit runtime as distinct metrics. GPU correctness must pass
first. Comparing different precisions requires explicit labels and an appropriate
correctness budget; no CPU/GPU ratio is currently reported.
