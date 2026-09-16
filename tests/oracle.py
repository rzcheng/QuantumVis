"""small dense test oracles, deliberately independent of production gate code."""

import numpy as np
from numpy.typing import NDArray

ATOL = 1e-12


def random_state(num_qubits: int, seed: int) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
    return state / np.linalg.norm(state)


def local_matrix(name: str, angle: float | None = None) -> NDArray[np.complex128]:
    """build rotations from their pauli generator, unlike closed-form kernels."""
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.diag(np.array([1, -1], dtype=np.complex128))
    fixed = {
        "X": x,
        "Y": y,
        "Z": z,
        "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
        "S": np.diag(np.array([1, 1j], dtype=np.complex128)),
        "T": np.diag(np.exp(1j * np.array([0, np.pi / 4]))),
    }
    if name in fixed:
        return fixed[name]
    assert angle is not None
    pauli = {"RX": x, "RY": y, "RZ": z}[name]
    return np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * pauli


def dense_operator(
    num_qubits: int,
    name: str,
    target: int,
    *,
    control: int | None = None,
    angle: float | None = None,
) -> NDArray[np.complex128]:
    """use kronecker products or basis-wise construction, never pair indexing."""
    if name in {"CX", "CNOT", "CZ"}:
        assert control is not None
        matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        for source in range(2**num_qubits):
            bits = list(f"{source:0{num_qubits}b}")
            is_control_set = bits[num_qubits - 1 - control] == "1"
            target_position = num_qubits - 1 - target
            phase = 1
            if is_control_set and name in {"CX", "CNOT"}:
                bits[target_position] = "0" if bits[target_position] == "1" else "1"
            elif is_control_set and bits[target_position] == "1":
                phase = -1
            destination = int("".join(bits), 2)
            matrix[destination, source] = phase
        return matrix
    matrix = np.ones((1, 1), dtype=np.complex128)
    for qubit in reversed(range(num_qubits)):
        factor = local_matrix(name, angle) if qubit == target else np.eye(2)
        matrix = np.kron(matrix, factor)
    return matrix
