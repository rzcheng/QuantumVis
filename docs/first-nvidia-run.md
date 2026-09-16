# First NVIDIA run

Use the installed environment from [gpu-validation.md](gpu-validation.md). Keep
the same checkout, environment, and `CUDA_VISIBLE_DEVICES` through all stages.
These bash commands stop on errors. Use a new `first-nvidia` directory for retries.

## STAGE A — PRECHECK

```bash
set -e
unset TRITON_INTERPRET
mkdir -p validation/results
mkdir validation/results/first-nvidia
python -m pip check
git rev-parse HEAD > validation/results/first-nvidia/commit.txt
python -m pip list --format=json > validation/results/first-nvidia/packages.json
python -m quantumvis.gpu_preflight --smoke --json \
  > validation/results/first-nvidia/preflight.json \
  2> validation/results/first-nvidia/preflight.stderr
```

Require exit 0, `status: READY`, and `smoke: PASS`. The report captures OS/CPU
architecture, Python/framework/CUDA/device versions and model/L4T files when
present. CUDA build and runtime versions are separate; a runtime query may be
unavailable without optional CUDA Python bindings. The smoke compiles (or loads
cached code), launches, and checks production X|0>. It is not full acceptance.
If blocked/error, read both preflight files and stop; do not start pytest.

## STAGE B — CORRECTNESS

```bash
python scripts/validate_nvidia.py \
  --output validation/results/first-nvidia/acceptance
```

This runs the standalone `python -m quantumvis.validate_gpu --json` first, then
the complete `tests/test_gpu_correctness.py` file. Require runner exit 0 and
`acceptance/summary.json` status PASS: 108 standalone checks and 673 pytest cases,
with zero failures/errors/skips. `validator.stdout`, `gpu-tests.xml`, and command
stdout/stderr files preserve the results and any compilation traceback.
Metadata capture and unchanged source hashes are also required. Keep the whole
directory, including failed attempts. No benchmark belongs in this stage.

## STAGE C — ONLY IF CORRECTNESS PASSES

```bash
python -m benchmarks.benchmark_gpu_smoke \
  --acceptance validation/results/first-nvidia/acceptance/summary.json \
  --output validation/results/first-nvidia/gpu-smoke.json
```

This refuses failed/skipped/stale acceptance reports and existing output files.
It measures H on target 0 at 8, 12, and 16 qubits: five warmups and 31 trials per
size, synchronized before and after each timed call. JSON retains raw trials,
median/p95, precision, GPU/software/source metadata, and the acceptance reference.
The metric is complete `GPUSimulator.run()` latency, including allocation,
transfers, validation, and synchronization; it is not isolated kernel time.
The first correctness call handles compilation before timing. Results are also
checked against the CPU oracle outside timed regions. Save the artifact, then
stop. Do not optimize or infer a speedup from this smoke run.
