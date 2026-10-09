# QuantumVis static demo

A single React + TypeScript page for recorded CPU circuit states, the Triton
kernel's index arithmetic, and the historical M4 Pro CPU baseline. React and
React DOM are the only runtime dependencies. There is no browser simulator,
Python runtime, GPU runtime, service, routing, or state interpolation.

## local development

Use Node 24 LTS (`nvm use` in this directory), with npm:

```bash
cd web
npm ci
npm run dev
```

Open http://127.0.0.1:5173/QuantumVis/. The committed JSON assets mean npm commands
do not require Python. Fetches and download links use Vite's `BASE_URL`.

## export data

From the repository root, with the existing CPU package installed:

```bash
python -m pip install -e '.[dev]'
python scripts/export_web_demo.py
python -m pytest -q tests/test_web_export.py
```

The exporter runs every prefix, including the initial state, using `Circuit` and
`CPUSimulator`: Bell, three-qubit GHZ, and H–RZ(θ)–H at 33 recorded angles from
0 through 2π in π/16 increments. It stores full-precision real/imaginary values
and probabilities as finite JSON. Metadata records the actual CPU backend,
complex128 precision, bit order, Python/NumPy versions, and SHA-256 hashes of
the exporter and imported CPU sources. Imported files must match the checkout.
No Torch or Triton is imported. Exports are deterministic in an unchanged source
and software environment; no wall-clock timestamp or changing git state is added.

`benchmarks/results/cpu-m4-pro-2026-09-14.json` is copied byte-for-byte to
`public/data/cpu-baseline.json`. The original contains H/RX, 8/12/16/18 qubits,
low/high targets, 5 warmups and 31 trials per case. Its timestamp, old project
name, dirty-tree flag, source hashes and raw samples are historical provenance.
The UI converts nanoseconds to milliseconds only for display. p95 is a
percentile, not a confidence interval. See the [methodology](../docs/benchmarking.md).

## checks and production preview

See the [local verification record](VERIFICATION.md) for exact results and limits.

```bash
cd web
npm run typecheck
npx playwright install chromium
npm test
npm run build
npm run preview
```

Open http://127.0.0.1:4173/QuantumVis/. `npm test` uses one Playwright/Chromium
setup, builds the production bundle, and starts its own preview on port 4174.
It checks controls, all angles, unique complete pair mappings for all targets,
benchmark conversion, loading/error/retry behavior, keyboard navigation, raw
downloads, nested asset URLs, reduced motion, contrast, and page overflow at
1280/390/360px. Screenshots for human inspection are saved under ignored
`test-results/`; they are not decorative snapshot assertions. The benchmark plot
uses a compact view box on mobile to keep all measured sizes and labels visible.

Run repository checks separately from the root:

```bash
python -m pytest
ruff check .
ruff format --check .
git diff --check
git status --short
```

No NVIDIA measurements or real-device validation are added by this demo.
The kernel panel illustrates logical index ownership, not physical CUDA threads.
Circuit recordings use complex128 CPU values; the illustrated Triton storage is
split float32. Display rounding and near-zero phase suppression do not modify
the exported data.

## deployment, prepared but not published

Vite uses base `/QuantumVis/` and static multi-page mode (no SPA fallback).
`web.yml` is independent of the existing lightweight Python CPU workflow and
uses Node 24, npm's lockfile, TypeScript, a production build and browser tests.
No optional GPU packages are installed by web CI.

`pages.yml` builds/tests on pushes to `main` or manual dispatch and checks the
repository's default branch for both build and deployment. Non-default manual
runs skip both jobs. Only the deploy job has Pages write/OIDC permissions.
The default branch was verified as `main`; update the push filter if renamed.

The remaining publication step is to review/merge these changes onto the default
branch, enable **Settings → Pages → GitHub Actions**, and restrict the
`github-pages` environment's deployment branches to the default branch. Then
run the Pages workflow from that branch. None of those remote changes are made
by the local implementation. Source/documentation links are pinned to the
implementation commit used when the demo was built.
