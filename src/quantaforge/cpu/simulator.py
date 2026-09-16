"""state-vector execution with vectorized numpy amplitude-pair operations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantaforge.circuit import Circuit
from quantaforge.gates import Gate, single_qubit_matrix
from quantaforge.measurement import _validated_amplitudes
from quantaforge.state import StateVector


def _apply_single_qubit(state: NDArray[np.complex128], gate: Gate) -> None:
    matrix = single_qubit_matrix(gate)
    # reshape groups by target bit: each column position is one disjoint pair.
    pairs = state.reshape(-1, 2, 1 << gate.target)
    low = pairs[:, 0, :].copy()
    high = pairs[:, 1, :].copy()
    pairs[:, 0, :] = matrix[0, 0] * low + matrix[0, 1] * high
    pairs[:, 1, :] = matrix[1, 0] * low + matrix[1, 1] * high


def _apply_controlled(state: NDArray[np.complex128], gate: Gate) -> None:
    # insert zero target/control bits to enumerate each controlled pair once.
    first, second = sorted((gate.target, gate.control))
    low = np.arange(state.size >> 2, dtype=np.intp)
    for bit in (first, second):
        mask = (1 << bit) - 1
        low = (low & mask) | ((low >> bit) << (bit + 1))
    low |= 1 << gate.control
    high = low | (1 << gate.target)
    if gate.name == "CX":
        saved = state[low].copy()
        state[low] = state[high]
        state[high] = saved
    else:  # cz: only amplitudes whose control and target bits are both 1 change.
        state[high] *= -1


class CPUSimulator:
    """execute a circuit from zero or an explicitly supplied normalized state."""

    def run(
        self, circuit: Circuit, *, initial_state: ArrayLike | StateVector | None = None
    ) -> StateVector:
        if not isinstance(circuit, Circuit):
            raise TypeError("circuit must be a Circuit")
        if initial_state is None:
            state = np.zeros(1 << circuit.num_qubits, dtype=np.complex128)
            state[0] = 1
        else:
            amplitudes = (
                initial_state.amplitudes
                if isinstance(initial_state, StateVector)
                else initial_state
            )
            state = _validated_amplitudes(amplitudes)
            if state.size != 1 << circuit.num_qubits:
                raise ValueError("initial state size does not match circuit num_qubits")
            state = state.copy()
        for gate in circuit.operations:
            if gate.control is None:
                _apply_single_qubit(state, gate)
            else:
                _apply_controlled(state, gate)
        return StateVector(state)
