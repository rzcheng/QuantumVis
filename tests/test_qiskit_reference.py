"""optional independent simulator comparisons; no quantaforge matrices in the oracle."""

import platform
from importlib.util import find_spec

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantaforge import Circuit, CPUSimulator
from quantaforge.gates import Gate

from .oracle import ATOL, random_state

pytestmark = pytest.mark.external
SEED = 20260915
SINGLE_GATES = ("X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ")


@pytest.fixture(scope="module")
def qiskit_api(record_testsuite_property):
    if find_spec("qiskit") is None:
        pytest.skip("optional Qiskit comparison: install quantaforge[verification]")
    # a broken installed reference must fail visibly, not turn into a skip.
    import qiskit
    from qiskit.quantum_info import Statevector

    for name, value in {
        "qiskit_version": qiskit.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "amplitude_atol": ATOL,
        "amplitude_rtol": 0,
    }.items():
        record_testsuite_property(name, value)
    return qiskit.QuantumCircuit, Statevector


def reference_circuit(circuit, qiskit_api):
    """translate gate names/arguments only; qiskit owns its matrices and evolution."""
    quantum_circuit, _ = qiskit_api
    reference = quantum_circuit(circuit.num_qubits)
    for gate in circuit.operations:
        method = getattr(reference, gate.name.lower())
        if gate.control is not None:
            method(gate.control, gate.target)
        elif gate.angle is not None:
            method(gate.angle, gate.target)
        else:
            method(gate.target)
    return reference


def compare(circuit, initial, qiskit_api, *, context=""):
    _, reference_state = qiskit_api
    original = initial.copy()
    expected = reference_state(initial.copy()).evolve(reference_circuit(circuit, qiskit_api))
    actual = CPUSimulator().run(circuit, initial_state=initial)
    # compare raw amplitudes: statevector.equiv would hide global-phase errors.
    assert_allclose(actual.amplitudes, expected.data, atol=ATOL, rtol=0, err_msg=context)
    assert_allclose(actual.probabilities(), expected.probabilities(), atol=ATOL, rtol=0)
    assert_allclose(np.vdot(actual.amplitudes, actual.amplitudes), 1, atol=ATOL, rtol=0)
    assert_array_equal(initial, original)
    return actual


@pytest.mark.parametrize("name", SINGLE_GATES)
@pytest.mark.parametrize(("num_qubits", "target"), [(n, t) for n in (1, 3, 6) for t in range(n)])
def test_single_gate_on_random_complex_input(qiskit_api, name, num_qubits, target):
    angle = -0.731 if name.startswith("R") else None
    circuit = Circuit(num_qubits).add(Gate(name, target, angle=angle))
    compare(circuit, random_state(num_qubits, SEED + target), qiskit_api)


@pytest.mark.parametrize("name", ("CX", "CZ"))
@pytest.mark.parametrize(
    ("num_qubits", "control", "target"),
    [(n, c, t) for n in (2, 3, 5) for c in range(n) for t in range(n) if c != t],
)
def test_controlled_gate_both_qubit_orders(qiskit_api, name, num_qubits, control, target):
    circuit = Circuit(num_qubits).add(Gate(name, target, control=control))
    compare(circuit, random_state(num_qubits, SEED + control * 11 + target), qiskit_api)


@pytest.mark.parametrize("name", ("RX", "RY", "RZ"))
@pytest.mark.parametrize("angle", (0.0, np.pi, -np.pi, 2 * np.pi, 17.2))
def test_rotation_phase_conventions(qiskit_api, name, angle):
    circuit = Circuit(3).add(Gate(name, 2, angle=angle))
    compare(circuit, random_state(3, SEED), qiskit_api)


@pytest.mark.parametrize("num_qubits", (2, 3, 6))
@pytest.mark.parametrize("root", (0, -1))
def test_bell_and_ghz_in_both_directions(qiskit_api, num_qubits, root):
    root %= num_qubits
    circuit = Circuit(num_qubits).h(root)
    for target in range(num_qubits):
        if target != root:
            circuit.cx(root, target)
    initial = np.zeros(1 << num_qubits, dtype=np.complex128)
    initial[0] = 1
    actual = compare(circuit, initial, qiskit_api)
    analytical = np.zeros_like(initial)
    analytical[[0, -1]] = 1 / np.sqrt(2)
    assert_allclose(actual.amplitudes, analytical, atol=ATOL, rtol=0)


@pytest.mark.parametrize("num_qubits", (1, 2, 4, 6))
@pytest.mark.parametrize("seed_offset", (0, 101, 912))
def test_mixed_circuit_every_prefix(qiskit_api, num_qubits, seed_offset):
    seed = SEED + seed_offset
    rng = np.random.default_rng(seed)
    names = SINGLE_GATES + (("CX", "CZ") if num_qubits > 1 else ())
    sequence = list(names) + list(rng.choice(names, size=48 - len(names)))
    rng.shuffle(sequence)
    initial = random_state(num_qubits, seed + 1)
    circuit = Circuit(num_qubits)
    for depth, name in enumerate(sequence, start=1):
        target = int(rng.integers(num_qubits))
        control = None
        if name in ("CX", "CZ"):
            control = int(rng.choice([q for q in range(num_qubits) if q != target]))
        angle = float(rng.uniform(-4 * np.pi, 4 * np.pi)) if name.startswith("R") else None
        circuit.add(Gate(name, target, control=control, angle=angle))
        compare(circuit, initial, qiskit_api, context=f"n={num_qubits}, seed={seed}, depth={depth}")


@pytest.mark.parametrize("target", range(5))
def test_basis_labels_are_not_bit_reversed(qiskit_api, target):
    initial = np.zeros(32, dtype=np.complex128)
    initial[0] = 1
    actual = compare(Circuit(5).x(target), initial, qiskit_api)
    expected = np.zeros_like(initial)
    expected[1 << target] = 1
    assert_array_equal(actual.amplitudes, expected)


def test_global_phase_is_not_discarded(qiskit_api):
    initial = random_state(3, SEED)
    actual = compare(Circuit(3).rz(1, 2 * np.pi), initial, qiskit_api)
    assert_allclose(actual.amplitudes, -initial, atol=ATOL, rtol=0)
