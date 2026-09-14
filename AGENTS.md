# QuantaForge engineering rules

- Establish numerical correctness before optimizing. Numerical changes require analytical and randomized tests.
- Performance claims require reproducible measurements and saved artifacts. Never invent numbers or compare against intentionally weak baselines.
- Keep dependencies small and backend boundaries simple. Justify broad refactors before making them.
- Preserve seeds, precision, hardware/software metadata, and benchmark methodology so results can be reproduced.
- Do not silently weaken tests or change benchmark methodology to obtain better-looking results.
- For failures: reproduce the failure, identify its cause, fix it, then rerun the relevant checks.
- Explain uncertainty, unavailable hardware, skipped checks, and limitations explicitly.
- Before finishing, run relevant tests and lint/format checks, inspect the diff, and check git status.
