# Validation records

## NVIDIA acceptance handoff, 2026-09-15

Added `scripts/validate_nvidia.py` to collect complete remote acceptance evidence
with one command, while retaining the existing quick validator. Simulator code,
CPU/GPU numerical tests, tolerances, and benchmark methodology are unchanged.

Reproduced the acceptance hazard on this Mac: the real GPU pytest file returned
exit code 0 with all 634 tests skipped. The new runner checks the JUnit case
outcomes, rejects zero-test/inconsistent reports, and requires both validation
stages and driver/package capture to pass before reporting PASS. Source hashes
must also be unchanged across execution. Existing output directories are refused.

Local checks: 33 host-side evidence/command tests passed. A real invocation
created `validation/results/mac-handoff-2026-09-15`, retained failure diagnostics
and environment/source details, reported UNAVAILABLE, and exited 2 without
launching GPU pytest. These records are locally ignored artifacts, not a GPU
acceptance result. Actual NVIDIA execution is still required for Milestone 2.

Final local checks: **552 passed, 0 failed, 634 explicitly skipped** in 1.20 s.
Ruff lint passed and all 33 Python files satisfied formatting. Source distribution
and wheel builds passed; the extracted source archive includes the new script,
helper, and tests, and its script's `--help` command passed. The 33 additional
passing tests exercise evidence collection, not device execution. CPU benchmarks
were not rerun because neither the simulator nor timing methodology changed.

## Triton source and CPU benchmark follow-up

Session: 2026-09-14 local time (artifact timestamps use UTC). Branch:
`feat/triton-backend`, starting from `fe09a9c`. Local acceptance is complete;
**Milestone 2 remote acceptance remains incomplete**.

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | 519 passed, 0 failed, 634 explicitly skipped, in 1.07 s |
| Existing CPU cases | All original 376 pass; CPU source and existing CPU tests unchanged |
| Added passing host cases | 29 capability decisions, 38 GPU interface boundaries, 27 CLI/error-budget checks, 49 benchmark checks |
| Real GPU cases | 634 skipped: Darwin, CPU-only PyTorch, CUDA unavailable, Triton missing |
| `.venv/bin/python -m quantaforge.validate_gpu --json` | UNAVAILABLE, exit 2, 0 device checks; environment metadata emitted |
| Ruff lint / format | All checks passed; 30 files already formatted |
| Package build | Source distribution and wheel built successfully without installing GPU dependencies |
| Extracted source archive | 519 passed, 0 failed, 634 skipped in 1.31 s; benchmark/test helpers included |
| Isolated wheel target | GPU source/validator present; dependencies marked optional; Mac validator exits 2 with metadata |
| Bell / GHZ examples | Analytical assertions and seeded sampling passed |
| CPU CLI smoke | 8 H/RX cases at 8/10 qubits, 24 raw timings, valid JSON and successful prechecks |
| Recorded CPU sweep | 16 H/RX cases at 8/12/16/18 qubits, 496 timings; all 9 source hashes verified |

The increased test count comes from host-side failure/precision/benchmark checks
and a parameterized real-device matrix across gates, qubit counts, targets, seeds,
analytical states, inverse rotations, circuits, raw storage contracts and streams.
Host tests do not mock numerical GPU success. Skipped cases are not GPU validation.

The saved [CPU artifact](../benchmarks/results/cpu-m4-pro-2026-09-14.json) records
Apple M4 Pro, Python 3.12.11, NumPy 2.3.5, complex128, raw trials, NumPy/BLAS/thread
configuration and source hashes. Example complete H-call timings: 8 qubits/target
0 median 20,958 ns, p95 22,187.5 ns; 18 qubits/target 0 median 1,931,458 ns, p95
2,529,562.5 ns. These are a single warmed sweep, not a speedup or kernel-only
measurement. The methodology and metadata correction are recorded in
[benchmarking.md](benchmarking.md).

Review reproduced five capability-test failures caused by accepting only the
string `1` for interpreter mode. Triton also recognizes other truthy spellings.
Detection now matches those spellings and checks in-process overrides; all 29
runtime tests pass. The raw wrapper also rejects a cached interpreted function.
The validator now emits FAIL with metadata before re-raising compilation/launch
exceptions, preserving debugging information rather than treating them as skips.

Unverified: kernel compilation, numerical GPU agreement, CUDA stream behavior,
GPU timings, hosted CI, and external-simulator comparisons. Error thresholds are
prespecified engineering budgets, not hardware-calibrated tolerances. Follow
[gpu-validation.md](gpu-validation.md) and capture all five remote acceptance
conditions before marking Milestone 2 complete.

## Initial CPU validation record (before this follow-up)

Date: 2026-09-14. Milestones 0 and 1 accepted locally. Milestone 2 is not complete.

## Environment

- macOS arm64, Apple M4 Pro; Python 3.12.11.
- NumPy 2.3.5, pytest 9.0.3, Ruff 0.15.7, build 1.4.0.
- setuptools 80.9.0 and wheel 0.45.1 for packaging.
- Local `.venv` reused installed NumPy/pytest through `--system-site-packages`;
  only the small missing lint/build tools were downloaded. Standard clean setup
  commands and pinned direct development dependencies are provided in the README.
- No NVIDIA driver, CUDA device, or Triton installation. The default Python 3.14.2
  has PyTorch 2.10.0 without CUDA; MPS also reports unavailable in this session.

## Checks

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest` | 376 passed, 0 failed, 0 skipped; 0.27 s in the recorded run |
| `.venv/bin/ruff check .` | All checks passed |
| `.venv/bin/ruff format --check .` | 17 files already formatted |
| Bell example | Analytical amplitude assertion passes; probabilities 0.5 at 00 and 11 |
| Four-qubit GHZ example | Analytical amplitude assertion passes; probabilities 0.5 at 0000 and 1111 |
| `.venv/bin/python -m build --no-isolation` | Source distribution and wheel built successfully |
| Extracted source archive, with plugin autoload disabled | 376 passed in 0.30 s; test helper and reproduction files present |
| Wheel installed to an isolated temporary target | Import resolved to installed wheel; both analytical examples passed |
| Fresh dependency resolution (`pip install --dry-run --ignore-installed -r requirements-dev.txt`) | All pinned dependencies resolved successfully |

Test runtime is a test-runner report, **not a simulator benchmark**. No runtime
comparison or performance improvement is claimed.

The suite includes independent dense single-qubit/controlled operators, all target
positions over multiple sizes, seeded random normalized states, 60-gate random
circuits, analytical phase checks, inverse rotations, Bell/GHZ, input preservation,
strict validation, probabilities, deterministic sampling, and sampling statistics.
Gate tests use absolute tolerance `1e-12` with zero relative tolerance.

An additional read-only review independently checked 660 CX/CZ cases over every
distinct control/target pair for 2–10 qubits using full-index permutation/phase
arithmetic. Maximum amplitude error was exactly zero. This manual check is
supplementary; the committed pytest suite is the reproducible acceptance check.

## Issues found and resolved

- The first source archive omitted the independent oracle helper and other
  reproduction files because setuptools' default selection only included test
  filenames. Archive inspection established the cause; `MANIFEST.in` now includes
  all Python tests, examples, technical docs, and development requirements.
- Review exposed a future GPU precision mismatch: ordinary float32-rounded states
  can fail strict complex128 input validation. The GPU decision note now separates
  public-backend comparison from raw-kernel comparison on rounded inputs. No CPU
  tolerance was changed.

## Unverified boundaries

No GPU kernel execution, CPU/GPU differential testing, trusted external simulator
comparison, profiler run, benchmark, or hosted CI run has occurred. Python 3.13/3.14
are configured in CI but have not run the complete suite locally. GPU dtype/layout
and tolerances remain proposed until measured on a supported device. Very deep
circuits may exceed the strict CPU norm tolerance; no arbitrary-depth guarantee is
made. The backend deliberately incurs working copies and exponential memory use.
