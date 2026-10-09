"""owned, read-only cpu state-vector results."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantumvis._validation import integer
from quantumvis.measurement import _validated_amplitudes, probabilities, sample


class StateVector:
    """a normalized complex128 state with qubit 0 at the least significant bit."""

    def __init__(self, amplitudes: ArrayLike) -> None:
        self._amplitudes = _validated_amplitudes(amplitudes).copy()
        self._amplitudes.flags.writeable = False

    @classmethod
    def zero(cls, num_qubits: int) -> StateVector:
        num_qubits = integer(num_qubits, "num_qubits", minimum=1)
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1
        return cls(state)

    @property
    def num_qubits(self) -> int:
        return self._amplitudes.size.bit_length() - 1

    @property
    def amplitudes(self) -> NDArray[np.complex128]:
        """read-only view; use .copy() to obtain independently mutable storage."""
        return self._amplitudes.view()

    def probabilities(self) -> NDArray[np.float64]:
        return probabilities(self._amplitudes)

    def sample(self, shots: int, *, seed: int | None = None) -> NDArray[np.int64]:
        return sample(self._amplitudes, shots, seed=seed)
