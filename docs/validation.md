# Validation records

## Project rename, 2026-09-16

The project was renamed to QuantumVis on 2026-09-16. Earlier records refer to the
package and source paths at their recorded revisions; their commands and hashes
are historical evidence. Current commands use `quantumvis`. The saved CPU baseline
and raw validation artifacts remain unchanged.

Starting revision: `a303d8d`. Local Python 3.12 verification: **797 passed, 0 failed,
820 skipped** (673 device and 147 interpreter cases). Ruff lint/format, package
build, dependency check, and Bell/GHZ examples passed. The renamed preflight exits
2 with BLOCKED on this Mac. All 39 Python files differ only by project naming;
no numerical operations or tolerances changed. The built wheel/source archive
contain the renamed package. No NVIDIA execution or new GPU timing is claimed.

## First-device preparation, 2026-09-16

Starting revision: `aeaedce`. Added preflight, a three-stage device runbook, a
restricted post-acceptance H timing smoke, and a platform/package audit. Production
kernel arithmetic, CPU reference, device tests, and numerical tolerances are
unchanged. No NVIDIA computation or GPU benchmark ran during this preparation.

| Check | Result |
| --- | --- |
| macOS Python 3.12.11, with Qiskit | **797 passed, 0 failed, 820 skipped** |
| Clean temporary Python 3.11.9 CPU environment | **612 passed, 0 failed, 1005 skipped** |
| New host-only preflight / benchmark protocol tests | 32 passed; no simulated GPU numerical success |
| Mac preflight with `--smoke --json` | BLOCKED, exit 2; kernel import not attempted, smoke NOT_RUN |
| Ruff lint / format | Passed; 39 Python files formatted |
| Build / pip check / examples | Wheel and source archive built; both environments passed pip check; Bell/GHZ assertions passed |
| Hosted interpreter, starting revision `aeaedce` | **147 passed, 0 failed, 0 skipped**, Triton 3.8.0, Torch 2.10.0+cpu, Python 3.12.14 |
| Hosted CPU matrix, implementation revision `1a9a20a` | Python 3.11/3.12/3.13/3.14: **612 passed, 0 failed, 1005 skipped each**; lint, format, examples, CPU smoke and builds passed |
| Hosted Qiskit job, `1a9a20a` | **185 passed, 0 failed, 0 skipped** |
| Hosted interpreter matrix, `1a9a20a` | Triton 3.6.0 and 3.8.0: **147 passed, 0 failed, 0 skipped each** |

The [CPU matrix](https://github.com/rzcheng/QuantumVis/actions/runs/35063367767)
and [interpreter matrix](https://github.com/rzcheng/QuantumVis/actions/runs/35063367779)
both completed successfully. Logs and both interpreter JUnit artifacts were
downloaded and inspected. The later documentation commit only records these
results; production/test code is the tested implementation revision. Compile-only
was skipped by its manual-job condition, and remains unexecuted.

The [hosted interpreter run](https://github.com/rzcheng/QuantumVis/actions/runs/35062312183)
executed actual production source with `TRITON_INTERPRET=1`; its JUnit artifact
was downloaded and inspected. The manual compile-only job was skipped, so no
compilation result is claimed. An additional interpreter job now targets Triton
3.6.0, matching the desktop Torch 2.10 dependency. The unchanged kernel hash in the
artifact is `244a64f123384262a8bddb3e3ae0b070252415baddcbab3c49a2743846bcc3bc`.

Python support is widened to 3.11+ after local suite verification; 3.11–3.14 are
configured in CPU CI. Python 3.10 is not accepted: `datetime.UTC` and the pinned
NumPy version require 3.11. The clean Python 3.11 environment lacks Qiskit, so its
1005 skips are 185 external comparisons, 147 interpreter, and 673 device cases.
The main Mac environment skips only the latter two groups. Skips are not validation.

Preflight's default READY is an eligibility/import result; `--smoke` additionally
requires the production X result. Metadata-only tests do not establish a working
Jetson stack. Published Triton aarch64 wheels were confirmed through PyPI metadata,
but Ryan's model/JetPack/Python ABI and compatible NVIDIA Torch stack remain unknown.
The preflight reports CUDA build and queried runtime versions separately; a
missing optional CUDA Python runtime-query binding is explicitly recorded.

The timing smoke is prepared but unexecuted. It requires a passing, zero-skip
acceptance report for unchanged source, records complete synchronized `run()`
latency, and preserves raw trials and provenance. It adds no performance claim.
Local logs/JUnit/CI downloads are under ignored
`validation/results/device-prep-2026-09-16/`. The next device acceptance still
requires all 108 standalone checks and 673 real-device pytest cases.

## Hardware-free preparation and Jetson metadata, 2026-09-16

Starting revision: `eaefc21`, branch `feat/triton-backend`. Production kernel,
CPU/GPU simulator code, numerical budgets, real-device suites, and benchmark
methodology are unchanged. No GPU dependencies were installed on this Mac.

| Evidence level | Result this session |
| --- | --- |
| CPU reference correctness / host tooling | **765 passed, 0 failed, 820 skipped** in 2.37 s; includes 185 Qiskit comparisons |
| Triton interpreter validation | **0 executed**; 147 cases collected and skipped; Linux CI prepared |
| Triton compile-only validation | **0 executed**; manual explicit-SM-8.7 Linux CLI probe prepared; upstream limitation documented |
| Real NVIDIA device validation | **0 executed**; 673 cases skipped; actual acceptance runner exited 2, UNAVAILABLE, without GPU pytest |
| Performance benchmarking | Not run; existing saved CPU baseline is unchanged; no GPU timing or speedup claim |

Environment: macOS 15.7 arm64, Apple M4 Pro, Python 3.12.11, NumPy 2.3.5,
pytest 9.0.3, Ruff 0.15.7, build 1.4.0, CPU-only PyTorch 2.10.0, Qiskit 2.5.2.
Triton and NVIDIA hardware are unavailable. Passing host tests do not count as
interpreter or device validation. The 26 new passing cases test metadata policy
and report aggregation only, bringing the acceptance-runner file to 59 tests.

The original runner's missing-`nvidia-smi` rejection was reproduced against its
previous source using synthetic passing child reports; no GPU computation was
simulated. Its cause was unconditional acceptance of metadata only through
`nvidia-smi` exit code zero. The new path records recognizable Orin/L4T files and
complete real-validator CUDA metadata when that executable is absent. Host tests
reject unknown models, unsupported capability, unavailable CUDA, incomplete
metadata, driver errors/timeouts, failed validation, and skipped GPU cases.
Actual Jetson execution remains unverified.

Diff review caught an overly broad Orin label match: a synthetic unknown Orin
board initially passed metadata identification. The strengthened regression
failed, the matcher was restricted to AGX Orin / Orin NX / Orin Nano, and the
full local suite was rerun. No hardware was involved in this policy check.

Ruff lint passed; all **35 Python files** passed formatting. Source distribution
and wheel builds succeeded. Workflow YAML and the embedded compile-probe Python
syntax were checked locally; neither Linux job was executed in this session.
The existing CPU workflow and optional dependency boundaries are unchanged.

Local JUnit/build logs and the actual Mac UNAVAILABLE run are retained under
ignored `validation/results/hardware-free-2026-09-16/`. No skipped check is
recorded as a numerical success. See [the GPU guide](gpu-validation.md) for the
separate commands and limits of each evidence level.

The compile-only subtask stops at Triton 3.8.0's public CLI: its output path still
accesses an active driver and does not expose the production fusion option.
The opt-in CI probe records its actual outcome, including failure; no undocumented
compiler API or driver patch is added. On CLI failure, compilation success is
unknown rather than inferred from partial output. Do not build additional
hardware-free infrastructure before the first NVIDIA run.

Next acceptance milestone: run the unchanged 108-check validator and all 673
device cases on compatible NVIDIA hardware, preserve their environment/logs, and
resolve real failures without weakening tests. Milestone 2 is not complete.

## Independent reference and robustness review, 2026-09-15

Starting revision: `3cc8293`, branch `feat/triton-backend`. CPU simulator source,
the original 376 CPU tests, Triton source, numerical tolerances, and timing
boundaries are unchanged. This review adds optional Qiskit verification, expands
real-device test coverage, and fixes an observed benchmark provenance defect.

| Check | Result |
| --- | --- |
| Full local suite with Qiskit 2.5.2 | **739 passed, 0 failed, 673 GPU cases skipped**, 2.14 s |
| Qiskit comparison file | 185 passed, 0 failed/skipped, 1.04 s; Python/NumPy/Qiskit/seed/tolerances recorded in JUnit |
| Benchmark tests | 51 passed, including two real subprocess installed-source cases |
| Fresh CPU-only wheel environment | **554 passed, 0 failed, 858 skipped**, 1.83 s; 673 GPU + 185 optional Qiskit skips |
| Fresh environment dependency boundary | PyTorch, Triton, and Qiskit absent; import resolves to the installed wheel |
| CPU-only wheel GPU validator | UNAVAILABLE, exit 2, 0 checks; missing packages recorded |
| Ruff lint / format | All checks passed; 34 Python files formatted |
| Packaging / dependencies | Wheel and source archive built; optional metadata and archive contents inspected; `pip check` passed |
| Bell / GHZ examples | Analytical assertions and deterministic samples passed |
| CPU benchmark smoke | 8 H/RX cases at 8/10 qubits; 24 raw timings; correctness prechecks and all 8 imported-module hashes verified |
| CI configuration | YAML parses; separate Qiskit job added with JUnit retention; hosted jobs not run |

The 187 new passing cases are 185 external comparisons and 2 provenance
regressions. The 39 new GPU cases cover all nine gates at three targets in
18-qubit states, two 64-gate circuits, and ten offset-view/guard-region cases.
They **only collected and skipped** here. Their device allocations, execution,
and assertions remain unverified until the full NVIDIA suite actually passes.
The existing one-command acceptance runner includes them automatically.

Qiskit comparisons use named gates and its own state evolution, never production
matrices. A supplementary sensitivity experiment changed three temporary copies
of the CPU source: reversed Y phase caused 10 failures, reversed CX direction
caused 28, and replacing RZ with a globally shifted phase gate caused 1. All three
incorrect implementations were rejected while preserving unitary norm; no such
mutation touched the repository. These are manual checks of test sensitivity,
not additional passing pytest cases. See [external verification](external-verification.md).

The provenance defect was reproduced by importing a different temporary copy of
the CPU implementation: the old harness recorded checkout hashes that did not
match the imported code. The fix checks imported-module hashes before timing and
records their paths. A mismatch now exits 2 without an artifact; a byte-identical
copy runs successfully. The numerical implementation, timed region, raw-trial
retention, statistics, and existing saved benchmark artifact are unchanged.

Local smoke/JUnit output is retained under ignored `validation/results/`; it is
not a performance comparison or GPU acceptance result. No new speedup claim is
made. NVIDIA compilation/correctness, GPU performance, and hosted CI remain open.

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
