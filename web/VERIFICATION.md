# Local verification

## review refresh, 2026-10-09

Starting revision: `64991da`, branch `codex/web-demo`. The existing implementation
was rechecked before publication for review. No NVIDIA hardware was available.

- Python 3.12.11: 840 passed, 820 skipped, zero failures; the skips remain 673
  device cases and 147 interpreter cases. Qiskit comparisons ran.
- Ruff lint/format, `pip check`, and source/wheel builds passed.
- Node 24.21.0: clean `npm ci`, TypeScript check, and all 26 Playwright tests
  passed. The browser suite builds and serves the production bundle.
- The install exposed [GHSA-68fv-2mgg-jv7q](https://github.com/advisories/GHSA-68fv-2mgg-jv7q)
  in the transitive development dependency `source-map-js`. Only its lockfile
  entry changed, from 1.2.1 to 1.2.2. Clean install, typecheck, production build,
  and all 26 browser tests passed again; `npm audit` reported zero vulnerabilities.
- Desktop and 360px expanded-metadata screenshots were visually inspected.
  Independent review verified source hashes, benchmark byte identity, and all
  16 median/p95 values against the saved raw trials.

Local Python JUnit is saved under ignored
`validation/results/session-20261009/web-base-tests.xml`. Browser screenshots
remain under ignored `web/test-results/`. This refresh uses the installed
Playwright Chromium cache and an isolated Node 24 runtime in `/tmp`.
Other browser engines and physical mobile devices remain untested.

Hosted checks passed at `89ff50f`:
[CPU/Qiskit](https://github.com/rzcheng/QuantumVis/actions/runs/37897424950),
[interpreter](https://github.com/rzcheng/QuantumVis/actions/runs/37897424973), and
[web](https://github.com/rzcheng/QuantumVis/actions/runs/37897424991).
Python 3.11–3.14 each reported 655 passed and 1005 skips. The separate Qiskit
job passed 185 cases; Triton 3.6.0 and 3.8.0 each passed 147 interpreter cases
with zero skips. Chromium passed all 26 web tests. Compile-only was intentionally
skipped; no device execution occurred. The CPU-job skips include all 673 device,
147 interpreter, and 185 optional Qiskit cases.

Integration remains tracked in [issue #3](https://github.com/rzcheng/QuantumVis/issues/3)
and [draft PR #4](https://github.com/rzcheng/QuantumVis/pull/4), based on
`feat/triton-backend`. This record does not establish merge or deployment.
The historical verification below describes its original session.

## initial verification, 2026-09-16

Verified on 2026-09-16 (America/New_York), on the development Mac. This record
covers the static engineering demo; it is not NVIDIA validation or a performance
improvement claim. The GitHub workflows are prepared but have not run remotely.

## environment and checks

| Check | Result |
| --- | --- |
| Python export tests | 43 passed, 0 skipped |
| Full existing pytest suite plus export tests | 840 passed, 820 skipped, 0 failed |
| Ruff 0.15.7 lint | Passed |
| Ruff format | 41 files already formatted |
| Clean `npm ci` | Passed; lockfile installed successfully |
| Playwright 1.63.0 / Chromium 153.0.8010.12 | 26 passed, 0 skipped |
| TypeScript 5.9.3 | Passed (`tsc --noEmit`) |
| Vite 8.3.0 production build | Passed, base `/QuantumVis/` |
| Diff whitespace check | Passed |

Python export/test environment: Python 3.12.11, NumPy 2.3.5. The final web checks
ran on Node 24.21.0, matching the CI major version. React/React DOM are 19.3.0.
The host's default Node was 25.5.0; an isolated Node 24 runtime was installed
under `/private/tmp/quantumvis-node-runtime` for the final checks. A shared
Playwright cache lock required the isolated browser cache below:

```bash
cd web
PATH=/private/tmp/quantumvis-node-runtime/node_modules/.bin:$PATH \
  PLAYWRIGHT_BROWSERS_PATH=/private/tmp/quantumvis-browsers npm test
```

The 820 Python skips comprise 673 Linux/NVIDIA device cases and 147 Triton
interpreter cases. They are not GPU validation. The existing optional Qiskit
comparisons ran in the full suite. Python simulator code and its existing
workflows were not changed.

## browser inspection

The production preview was opened in the app's real browser and inspected at
1280×900, 390×900, and 360×900. Playwright separately exercises the real production
bundle in Chromium and saves screenshots under ignored `web/test-results/`.
Desktop, mobile, narrow, and malformed-data screenshots were visually inspected.

- Bell/GHZ/interference presets, Initial/Previous/Next, gate selection, keyboard
  focus and all 33 recorded angles work. The π endpoint shows amplitude −i.
- All three targets have four unique, disjoint pairs covering every index, with
  the correct differing bit. Both arrays expose selected indices in text.
- Both benchmark selections reproduce the raw file's converted values in the
  table. The mobile SVG shows all four measured sizes with readable axis text.
- No unwanted page overflow was found, including expanded metadata/source hashes.
  The mobile state table retains all four accessible column headers.
- Section links, keyboard navigation, raw JSON download, loading, missing data,
  malformed schema/JSON, retry, and reduced motion are checked.
- Normal loading produces no browser console errors, JavaScript exceptions, or
  missing assets. Deliberately missing/malformed data yields visible errors.
- Production JS/CSS/JSON use nested `/QuantumVis/` URLs. Unknown routes and missing
  assets return 404, with no SPA fallback.
- The specified text/accent colors exceed 4.5:1 against the page, panel and
  selected-control backgrounds. Other browser engines and physical mobile
  devices were not tested; no screen-reader certification is claimed.

## data provenance

`public/data/circuits.json` was produced by the existing CPU simulator, with
backend/precision, bit order, Python/NumPy versions and source SHA-256 metadata.
Analytical and seeded randomized tests check complex amplitudes and probabilities;
the committed snapshots are also checked against the independent dense oracle.
The exporter imports neither Torch nor Triton.

The benchmark source and exported download are byte-identical. Both have SHA-256:

```text
a0fa85352314d91e9b5f7647ca9798ecaf3b30854050be7e1b7ac13f851d07a5
```

Original source: `benchmarks/results/cpu-m4-pro-2026-09-14.json`. Recorded timestamp:
2026-09-15T01:38:52.119259+00:00; Apple M4 Pro; complex128; Python 3.12.11; NumPy
2.3.5; seed 20260914; 5 warmups and 31 trials per case. The original dirty-tree
flag, historical project name, revision and source hashes remain unchanged.

## handoff

The local production preview is [http://127.0.0.1:4173/QuantumVis/](http://127.0.0.1:4173/QuantumVis/).
See [README.md](README.md) for normal export/dev/test/build commands and the
remaining Pages setup. No push, merge, Pages enablement, or publication occurred.
The user's pre-existing untracked `docs/web-demo-build-prompt.md` was preserved.
