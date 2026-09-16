import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantaforge import Circuit, CPUSimulator, StateVector
from quantaforge.gates import Gate

from .oracle import ATOL, dense_operator, random_state


def test_bell_state():
    circuit = Circuit(2).h(0).cx(0, 1)
    result = CPUSimulator().run(circuit)
    assert_allclose(result.amplitudes, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=ATOL, rtol=0)
    assert_allclose(result.probabilities(), [0.5, 0, 0, 0.5], atol=ATOL, rtol=0)


@pytest.mark.parametrize("num_qubits", (2, 3, 5, 8))
def test_ghz_state(num_qubits):
    circuit = Circuit(num_qubits).h(0)
    for target in range(1, num_qubits):
        circuit.cx(target - 1, target)
    result = CPUSimulator().run(circuit)
    expected = np.zeros(2**num_qubits, dtype=np.complex128)
    expected[[0, -1]] = 1 / np.sqrt(2)
    assert_allclose(result.amplitudes, expected, atol=ATOL, rtol=0)
    assert_allclose(np.sum(result.probabilities()), 1, atol=ATOL, rtol=0)


@pytest.mark.parametrize("num_qubits", (1, 2, 5))
def test_empty_circuit_initializes_zero_state(num_qubits):
    result = CPUSimulator().run(Circuit(num_qubits))
    expected = np.zeros(2**num_qubits, dtype=np.complex128)
    expected[0] = 1
    assert_array_equal(result.amplitudes, expected)
    assert result.num_qubits == num_qubits


def test_empty_circuit_preserves_input_and_does_not_alias():
    initial = StateVector(random_state(3, seed=242))
    result = CPUSimulator().run(Circuit(3), initial_state=initial)
    assert_array_equal(result.amplitudes, initial.amplitudes)
    assert not np.shares_memory(result.amplitudes, initial.amplitudes)


@pytest.mark.parametrize("target", range(5))
def test_little_endian_target_index(target):
    result = CPUSimulator().run(Circuit(5).x(target))
    expected = np.zeros(32, dtype=np.complex128)
    expected[2**target] = 1
    assert_array_equal(result.amplitudes, expected)


@pytest.mark.parametrize("seed", (5, 37, 913))
@pytest.mark.parametrize("num_qubits", (2, 3, 5))
def test_random_circuit_matches_independent_dense_oracle(num_qubits, seed):
    rng = np.random.default_rng(seed)
    initial = random_state(num_qubits, seed=seed + 1000)
    expected = initial.copy()
    circuit = Circuit(num_qubits)
    names = ("X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ", "CX", "CZ")
    # include each gate at least once; vary the remaining order and placements.
    sequence = list(names) + list(rng.choice(names, size=49))
    rng.shuffle(sequence)
    for name in sequence:
        target = int(rng.integers(num_qubits))
        angle = float(rng.uniform(-2 * np.pi, 2 * np.pi)) if name.startswith("R") else None
        control = None
        if name in ("CX", "CZ"):
            control = int(rng.choice([q for q in range(num_qubits) if q != target]))
        circuit.add(Gate(name, target, angle=angle, control=control))
        expected = dense_operator(num_qubits, name, target, angle=angle, control=control) @ expected

    result = CPUSimulator().run(circuit, initial_state=initial)
    assert_allclose(result.amplitudes, expected, atol=ATOL, rtol=0)
    assert_allclose(np.vdot(result.amplitudes, result.amplitudes), 1, atol=ATOL, rtol=0)


def test_run_is_repeatable_and_does_not_consume_circuit():
    circuit = Circuit(3).h(2).ry(0, 0.39).cx(2, 1).t(1)
    operations = circuit.operations
    simulator = CPUSimulator()
    first = simulator.run(circuit)
    second = simulator.run(circuit)
    assert_array_equal(first.amplitudes, second.amplitudes)
    assert circuit.operations == operations


def test_cnot_alias_and_chainable_builder():
    circuit = Circuit(2)
    assert circuit.x(0) is circuit
    assert circuit.y(0) is circuit
    assert circuit.z(0) is circuit
    assert circuit.h(0) is circuit
    assert circuit.s(0) is circuit
    assert circuit.t(0) is circuit
    assert circuit.rx(0, 0.1) is circuit
    assert circuit.ry(0, 0.2) is circuit
    assert circuit.rz(0, 0.3) is circuit
    assert circuit.cx(0, 1) is circuit
    assert circuit.cz(0, 1) is circuit
    assert circuit.cnot(1, 0) is circuit
    assert circuit.add(Gate("X", 1)) is circuit
    assert Gate("CNOT", 1, control=0) == Gate("CX", 1, control=0)
    assert_allclose(
        CPUSimulator().run(Circuit(2).h(0).cnot(0, 1)).amplitudes,
        CPUSimulator().run(Circuit(2).h(0).cx(0, 1)).amplitudes,
        atol=ATOL,
        rtol=0,
    )


def test_operation_sequence_is_immutable_snapshot():
    circuit = Circuit(2).h(0)
    operations = circuit.operations
    assert isinstance(operations, tuple)
    circuit.cx(0, 1)
    assert len(operations) == 1
    assert len(circuit.operations) == 2


@pytest.mark.parametrize("num_qubits", (0, -1))
def test_circuit_rejects_invalid_qubit_count(num_qubits):
    with pytest.raises(ValueError):
        Circuit(num_qubits)


@pytest.mark.parametrize("num_qubits", (True, False, 1.5, "2", None))
def test_circuit_rejects_nonintegral_qubit_count(num_qubits):
    with pytest.raises(TypeError):
        Circuit(num_qubits)


def test_numpy_integer_indices_are_supported():
    circuit = Circuit(np.int64(2)).h(np.int64(0)).cx(np.int64(0), np.int64(1))
    assert_allclose(
        CPUSimulator().run(circuit).amplitudes,
        np.array([1, 0, 0, 1]) / np.sqrt(2),
        atol=ATOL,
        rtol=0,
    )


@pytest.mark.parametrize("gate", (Gate("X", 2), Gate("CX", 1, control=2), Gate("CZ", 2, control=0)))
def test_circuit_rejects_out_of_range_gate_without_mutation(gate):
    circuit = Circuit(2).h(0)
    operations = circuit.operations
    with pytest.raises(ValueError):
        circuit.add(gate)
    assert circuit.operations == operations


@pytest.mark.parametrize("operation", (None, "H", ["X", 0], 1))
def test_circuit_rejects_non_gate_operations(operation):
    with pytest.raises(TypeError):
        Circuit(1).add(operation)


@pytest.mark.parametrize("initial", ([1, 0], StateVector.zero(3)))
def test_simulator_rejects_initial_state_with_wrong_size(initial):
    with pytest.raises(ValueError):
        CPUSimulator().run(Circuit(2), initial_state=initial)


@pytest.mark.parametrize("circuit", (None, [], "H", 2))
def test_simulator_rejects_non_circuit_input(circuit):
    with pytest.raises(TypeError):
        CPUSimulator().run(circuit)
