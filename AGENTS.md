# engineering notes

- correctness first. numerical changes need analytical and randomized tests.
- performance claims need saved, reproducible measurements. no invented numbers or weak baselines.
- keep dependencies and backend boundaries small. justify broad refactors.
- preserve seeds, precision, environment metadata, and benchmark methodology.
- do not weaken tests or change methodology to make results look better.
- reproduce failures, find the cause, fix it, and rerun the relevant checks.
- state skipped checks, hardware limits, and uncertainty.
- run tests and lint, inspect the diff, and check git status before finishing.
- keep comments and docstrings short and lowercase; preserve exact identifiers and units when needed.
- use short lowercase commit messages, such as `add gate tests` or `fix sampling seed`.
