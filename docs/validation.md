# Initial CPU validation record

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
