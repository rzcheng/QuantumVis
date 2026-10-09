import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantumvis import Circuit, CPUSimulator
from quantumvis.gates import Gate, single_qubit_matrix

from .oracle import ATOL, dense_operator, local_matrix, random_state

SINGLE_GATES = ("X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ")
SINGLE_CASES = [(n, q) for n in (1, 2, 3, 5) for q in range(n)]
CONTROLLED_CASES = [
    (n, control, target)
    for n in (2, 3, 4)
    for control in range(n)
    for target in range(n)
    if control != target
]


@pytest.mark.parametrize("name", SINGLE_GATES)
@pytest.mark.parametrize(("num_qubits", "target"), SINGLE_CASES)
def test_single_gate_matches_independent_dense_oracle(name, num_qubits, target):
    state = random_state(num_qubits, seed=1700 + num_qubits * 19 + target)
    original = state.copy()
    angle = -0.731 if name.startswith("R") else None
    circuit = Circuit(num_qubits).add(Gate(name, target, angle=angle))

    actual = CPUSimulator().run(circuit, initial_state=state).amplitudes
    expected = dense_operator(num_qubits, name, target, angle=angle) @ state

    assert_allclose(actual, expected, atol=ATOL, rtol=0)
    assert_allclose(np.vdot(actual, actual), 1, atol=ATOL, rtol=0)
    assert_array_equal(state, original)


@pytest.mark.parametrize("name", ("CX", "CZ"))
@pytest.mark.parametrize(("num_qubits", "control", "target"), CONTROLLED_CASES)
def test_controlled_gate_matches_independent_dense_oracle(name, num_qubits, control, target):
    state = random_state(num_qubits, seed=2100 + num_qubits * 19 + control * 7 + target)
    original = state.copy()
    circuit = Circuit(num_qubits).add(Gate(name, target, control=control))

    actual = CPUSimulator().run(circuit, initial_state=state).amplitudes
    expected = dense_operator(num_qubits, name, target, control=control) @ state

    assert_allclose(actual, expected, atol=ATOL, rtol=0)
    assert_allclose(np.vdot(actual, actual), 1, atol=ATOL, rtol=0)
    assert_array_equal(state, original)


@pytest.mark.parametrize("name", ("CX", "CZ"))
@pytest.mark.parametrize(("control", "target"), ((0, 1), (1, 0)))
@pytest.mark.parametrize("basis_index", range(4))
def test_controlled_gate_truth_table(name, control, target, basis_index):
    state = np.eye(4, dtype=np.complex128)[:, basis_index]
    circuit = Circuit(2).add(Gate(name, target, control=control))
    actual = CPUSimulator().run(circuit, initial_state=state).amplitudes
    expected = dense_operator(2, name, target, control=control) @ state
    assert_allclose(actual, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    ("name", "initial", "expected"),
    (
        ("X", [1, 0], [0, 1]),
        ("X", [0, 1], [1, 0]),
        ("Y", [1, 0], [0, 1j]),
        ("Y", [0, 1], [-1j, 0]),
        ("Z", [0, 1], [0, -1]),
        ("H", [1, 0], np.array([1, 1]) / np.sqrt(2)),
        ("H", [0, 1], np.array([1, -1]) / np.sqrt(2)),
        ("S", [0, 1], [0, 1j]),
        ("T", [0, 1], [0, np.exp(1j * np.pi / 4)]),
    ),
)
def test_analytical_single_qubit_states(name, initial, expected):
    actual = CPUSimulator().run(Circuit(1).add(Gate(name, 0)), initial_state=initial)
    assert_allclose(actual.amplitudes, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", ("X", "Y", "Z", "H"))
@pytest.mark.parametrize("target", range(4))
def test_self_inverse_gate_restores_complex_state(name, target):
    state = random_state(4, seed=24)
    circuit = Circuit(4).add(Gate(name, target)).add(Gate(name, target))
    actual = CPUSimulator().run(circuit, initial_state=state)
    assert_allclose(actual.amplitudes, state, atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", ("RX", "RY", "RZ"))
@pytest.mark.parametrize("angle", (0, 0.031, -0.91, np.pi, 2 * np.pi, 17.2))
@pytest.mark.parametrize("target", (0, 2))
def test_inverse_rotations_restore_state(name, angle, target):
    state = random_state(3, seed=81)
    circuit = Circuit(3).add(Gate(name, target, angle=angle)).add(Gate(name, target, angle=-angle))
    actual = CPUSimulator().run(circuit, initial_state=state)
    assert_allclose(actual.amplitudes, state, atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    ("name", "angle", "expected"),
    (
        ("RX", np.pi, [0, -1j]),
        ("RY", np.pi, [0, 1]),
        ("RZ", np.pi, [-1j, 0]),
        ("RX", 2 * np.pi, [-1, 0]),
        ("RY", 2 * np.pi, [-1, 0]),
        ("RZ", 2 * np.pi, [-1, 0]),
    ),
)
def test_rotation_sign_and_global_phase(name, angle, expected):
    circuit = Circuit(1).add(Gate(name, 0, angle=angle))
    assert_allclose(CPUSimulator().run(circuit).amplitudes, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", SINGLE_GATES)
def test_public_single_qubit_matrix_matches_oracle_and_is_unitary(name):
    angle = 0.71 if name.startswith("R") else None
    actual = single_qubit_matrix(Gate(name, 0, angle=angle))
    assert_allclose(actual, local_matrix(name, angle), atol=ATOL, rtol=0)
    assert_allclose(actual.conj().T @ actual, np.eye(2), atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", ("CX", "CZ"))
def test_single_qubit_matrix_rejects_controlled_gates(name):
    with pytest.raises(ValueError):
        single_qubit_matrix(Gate(name, 1, control=0))
