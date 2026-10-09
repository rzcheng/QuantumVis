"""single-qubit gpu execution with strict cpu inputs and visible float32 drift."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantumvis._validation import integer
from quantumvis.circuit import Circuit
from quantumvis.gates import single_qubit_matrix
from quantumvis.gpu.kernels import MAX_NUM_QUBITS
from quantumvis.gpu.runtime import require_gpu
from quantumvis.measurement import _validated_amplitudes
from quantumvis.state import StateVector


class GPUResult:
    """read-only complex64 result; preserves norm drift and supports probabilities."""

    def __init__(self, amplitudes: ArrayLike) -> None:
        values = np.asarray(amplitudes)
        if values.dtype != np.complex64:
            raise TypeError("GPUResult amplitudes must have dtype complex64")
        if values.ndim != 1 or values.size < 2 or values.size & (values.size - 1):
            raise ValueError("state must be one-dimensional with length 2**n for n >= 1")
        if not np.all(np.isfinite(values)):
            raise ValueError("GPUResult amplitudes must be finite")
        self._amplitudes = values.copy()
        self._amplitudes.flags.writeable = False

    @property
    def num_qubits(self) -> int:
        return self._amplitudes.size.bit_length() - 1

    @property
    def amplitudes(self) -> NDArray[np.complex64]:
        """read-only view; .copy() returns independently mutable storage."""
        return self._amplitudes.view()

    def probabilities(self) -> NDArray[np.float64]:
        """compute squared magnitudes in float64, preserving total norm drift."""
        real = self._amplitudes.real.astype(np.float64)
        imag = self._amplitudes.imag.astype(np.float64)
        return real * real + imag * imag


class GPUSimulator:
    """single-qubit backend with lazy imports and no cpu fallback.

    run includes validation, transfers, and launches; downloads block until ready.
    """

    def __init__(self, device: int | None = None) -> None:
        self.device = None if device is None else integer(device, "device")

    def run(
        self, circuit: Circuit, *, initial_state: ArrayLike | StateVector | None = None
    ) -> GPUResult:
        if not isinstance(circuit, Circuit):
            raise TypeError("circuit must be a Circuit")
        operations = circuit.operations
        for gate in operations:
            if gate.control is not None:
                raise NotImplementedError(
                    f"the initial Triton backend does not support {gate.name}; "
                    "only single-qubit gates are implemented"
                )
        if circuit.num_qubits > MAX_NUM_QUBITS:
            raise ValueError(f"this kernel supports at most {MAX_NUM_QUBITS} qubits")
        state = None
        if initial_state is not None:
            amplitudes = (
                initial_state.amplitudes
                if isinstance(initial_state, StateVector)
                else initial_state
            )
            state = _validated_amplitudes(amplitudes)
            if state.size != 1 << circuit.num_qubits:
                raise ValueError("initial state size does not match circuit num_qubits")
        status = require_gpu(self.device)

        import torch

        from quantumvis.gpu.kernels import apply_single_qubit

        device = torch.device("cuda", status.device_index)
        if state is None:
            real = torch.zeros(1 << circuit.num_qubits, device=device, dtype=torch.float32)
            imag = torch.zeros_like(real)
            real[0] = 1
        else:
            # validate before rounding, then copy into split float32 storage.
            real = torch.tensor(state.real.copy(), device=device, dtype=torch.float32)
            imag = torch.tensor(state.imag.copy(), device=device, dtype=torch.float32)
        for gate in operations:
            apply_single_qubit(real, imag, single_qubit_matrix(gate), gate.target)
        # blocking device-to-host copies synchronize the work before returning.
        downloaded = np.empty(real.numel(), dtype=np.complex64)
        downloaded.real = real.cpu().numpy()
        downloaded.imag = imag.cpu().numpy()
        return GPUResult(downloaded)
