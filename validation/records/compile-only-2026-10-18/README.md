# Failed public compile-only probe

These are byte-for-byte copies of the compile artifact from
[run 39154822044](https://github.com/rzcheng/QuantumVis/actions/runs/39154822044),
executed on 2026-10-18 at source revision `3814b0f`.

- `summary.json`, `stdout.txt`, and `stderr.txt` are unmodified raw outputs.
- `provenance.json` is a derived manifest: run/source identity, raw-file hashes,
  original archive digests, and the separately verified interpreter counters.
- Full logs, original ZIP archives, and interpreter JUnit files remain in Actions
  and the original local evidence directory; they are not duplicated here.

The CLI failed at `kernel.create_binder()` because it found no active driver.
The summary remains FAIL, exit 1, with `compilation_succeeded: null`; no compiled
output was saved. The two passing interpreter suites are separate evidence.
No NVIDIA correctness or performance claim follows from this record.

See the [validation record](../../../docs/validation.md#public-compile-only-probe-2026-10-18)
for the environment and limitations.
