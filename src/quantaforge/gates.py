"""validated gate descriptions and their small single-qubit matrices."""

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from quantaforge._validation import integer

SINGLE_QUBIT_GATES = frozenset({"X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"})
CONTROLLED_GATES = frozenset({"CX", "CZ"})
ROTATION_GATES = frozenset({"RX", "RY", "RZ"})


@dataclass(frozen=True, slots=True)
class Gate:
    """an operation on named qubits; angles are radians.

    qubit bounds are validated when an operation is added to a circuit.
    """

    name: str
    target: int
    control: int | None = field(default=None, kw_only=True)
    angle: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("gate name must be a string")
        name = "CX" if self.name == "CNOT" else self.name
        if name not in SINGLE_QUBIT_GATES | CONTROLLED_GATES:
            raise ValueError(f"unsupported gate: {self.name!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "target", integer(self.target, "target"))
        if name in CONTROLLED_GATES:
            if self.control is None:
                raise ValueError(f"{name} requires a control qubit")
            object.__setattr__(self, "control", integer(self.control, "control"))
            if self.control == self.target:
                raise ValueError("control and target must be distinct")
        elif self.control is not None:
            raise ValueError(f"{name} does not accept a control qubit")
        if name in ROTATION_GATES:
            if self.angle is None:
                raise ValueError(f"{name} requires an angle")
            if isinstance(self.angle, bool) or not isinstance(self.angle, Real):
                raise TypeError("angle must be a real number")
            angle = float(self.angle)
            if not np.isfinite(angle):
                raise ValueError("angle must be finite")
            object.__setattr__(self, "angle", angle)
        elif self.angle is not None:
            raise ValueError(f"{name} does not accept an angle")


def single_qubit_matrix(gate: Gate) -> NDArray[np.complex128]:
    """return the 2x2 unitary; controlled operations have no 2x2 matrix here."""
    if not isinstance(gate, Gate):
        raise TypeError("gate must be a Gate")
    name = gate.name
    match name:
        case "X":
            values = [[0, 1], [1, 0]]
        case "Y":
            values = [[0, -1j], [1j, 0]]
        case "Z":
            values = [[1, 0], [0, -1]]
        case "H":
            return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        case "S":
            values = [[1, 0], [0, 1j]]
        case "T":
            values = [[1, 0], [0, np.exp(1j * np.pi / 4)]]
        case "RX" | "RY" | "RZ":
            # gate validation guarantees the angle exists and is finite.
            half_angle = gate.angle / 2
            c, s = np.cos(half_angle), np.sin(half_angle)
            if name == "RX":
                values = [[c, -1j * s], [-1j * s, c]]
            elif name == "RY":
                values = [[c, -s], [s, c]]
            else:
                values = [[np.exp(-1j * half_angle), 0], [0, np.exp(1j * half_angle)]]
        case _:
            raise ValueError(f"{name} is not a single-qubit gate")
    return np.array(values, dtype=np.complex128)
