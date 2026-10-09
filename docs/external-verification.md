# Independent Qiskit comparisons

The optional `verification` extra installs Qiskit for a second CPU correctness
reference. It does not participate in QuantumVis execution, benchmarking, or GPU
runtime detection. The normal installation still requires only NumPy. Qiskit Aer,
IBM services, credentials, and network calls are not used by these tests.

```bash
python -m pip install -e '.[dev,verification]'
python -m pytest -q tests/test_qiskit_reference.py --junitxml=validation/results/qiskit.xml
```

The tests construct Qiskit circuits from named operations, qubit positions, and
angles, then use its [Statevector.evolve API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector).
They never supply QuantumVis's gate matrices to the reference. Both simulators
use [qubit 0 as the least significant bit](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering),
so amplitude arrays are compared directly, without reversing bits. Comparisons
use complex128, absolute tolerance `1e-12`, and relative tolerance zero. Raw
amplitudes retain global phase; Qiskit's phase-insensitive `equiv` is not used.

The suite covers all nine single-qubit gates, both controlled gates in all
distinct control/target orders for selected sizes, rotation phase conventions,
Bell/GHZ states, and basis-index anchors. Twelve seeded mixed circuits of 48 gates
are compared after **every prefix**, so later cancellation cannot conceal an
earlier disagreement. Circuit sizes range from one to six qubits. Probabilities,
norm, and preservation of caller input are also checked. This supplements the
existing analytical and independent dense-oracle tests; it does not replace them.

When Qiskit is absent these tests explicitly skip. If an installed Qiskit fails
to import or execute, the error propagates. CI has a separate job that installs
the extra, imports Qiskit explicitly, runs this file, and retains its JUnit report
with Python, NumPy, Qiskit, seed, and tolerance metadata. CPU-only CI continues
without this extra. Hosted CI execution remains pending until the repository is
pushed to a configured remote.

The first local run used Python 3.12.11, NumPy 2.3.5, and Qiskit 2.5.2 on Apple
M4 Pro: **185 passed, 0 failed, 0 skipped**. The installation added Qiskit,
rustworkx, and stevedore (about 11 MB of wheels); the existing environment already
provided SciPy and other dependencies.
A clean verification environment also resolves those dependencies. Qiskit 2.x
is allowed by the extra; only versions recorded in validation results are claimed
as tested. These are correctness comparisons, not performance comparisons, and
provide no evidence of Triton compilation or GPU execution.
