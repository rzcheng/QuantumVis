"""An ordered circuit description, independent of execution backend."""

from __future__ import annotations

from quantaforge._validation import integer
from quantaforge.gates import Gate


class Circuit:
    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = integer(num_qubits, "num_qubits", minimum=1)
        self._operations: list[Gate] = []

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def operations(self) -> tuple[Gate, ...]:
        return tuple(self._operations)

    def add(self, gate: Gate) -> Circuit:
        if not isinstance(gate, Gate):
            raise TypeError("operation must be a Gate")
        if gate.target >= self.num_qubits:
            raise ValueError("target qubit is outside the circuit")
        if gate.control is not None and gate.control >= self.num_qubits:
            raise ValueError("control qubit is outside the circuit")
        self._operations.append(gate)
        return self

    def x(self, target: int) -> Circuit:
        return self.add(Gate("X", target))

    def y(self, target: int) -> Circuit:
        return self.add(Gate("Y", target))

    def z(self, target: int) -> Circuit:
        return self.add(Gate("Z", target))

    def h(self, target: int) -> Circuit:
        return self.add(Gate("H", target))

    def s(self, target: int) -> Circuit:
        return self.add(Gate("S", target))

    def t(self, target: int) -> Circuit:
        return self.add(Gate("T", target))

    def rx(self, target: int, angle: float) -> Circuit:
        return self.add(Gate("RX", target, angle=angle))

    def ry(self, target: int, angle: float) -> Circuit:
        return self.add(Gate("RY", target, angle=angle))

    def rz(self, target: int, angle: float) -> Circuit:
        return self.add(Gate("RZ", target, angle=angle))

    def cx(self, control: int, target: int) -> Circuit:
        return self.add(Gate("CX", target, control=control))

    def cnot(self, control: int, target: int) -> Circuit:
        return self.cx(control, target)

    def cz(self, control: int, target: int) -> Circuit:
        return self.add(Gate("CZ", target, control=control))
