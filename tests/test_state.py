import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantumvis import StateVector

from .oracle import ATOL, random_state


@pytest.mark.parametrize("num_qubits", (1, 2, 5))
def test_zero_state_shape_dtype_and_normalization(num_qubits):
    state = StateVector.zero(num_qubits)
    assert state.num_qubits == num_qubits
    assert state.amplitudes.shape == (2**num_qubits,)
    assert state.amplitudes.dtype == np.complex128
    assert state.amplitudes[0] == 1
    assert np.count_nonzero(state.amplitudes) == 1


def test_constructor_accepts_arraylike_and_owns_complex128_copy():
    source = np.array([0.6, 0.8], dtype=np.float64)
    state = StateVector(source)
    assert state.amplitudes.dtype == np.complex128
    source[:] = [1, 0]
    assert_array_equal(state.amplitudes, [0.6, 0.8])
    assert not np.shares_memory(source, state.amplitudes)
    assert_array_equal(StateVector([0, 1]).amplitudes, [0, 1])


def test_amplitudes_are_read_only():
    state = StateVector.zero(2)
    assert not state.amplitudes.flags.writeable
    with pytest.raises(ValueError):
        state.amplitudes[0] = 0


@pytest.mark.parametrize(
    "amplitudes",
    (
        [],
        [1],
        [1, 0, 0],
        [[1, 0], [0, 0]],
        1,
        [0, 0],
        [1, 1],
        [np.nan, 0],
        [np.inf, 0],
        [1, complex(0, np.inf)],
    ),
)
def test_rejects_invalid_amplitude_arrays(amplitudes):
    with pytest.raises(ValueError):
        StateVector(amplitudes)


def test_does_not_silently_renormalize_input():
    with pytest.raises(ValueError):
        StateVector([np.sqrt(1 + 1e-8), 0])


def test_accepted_roundoff_is_not_silently_renormalized():
    amplitudes = np.array([np.sqrt(1 + 2e-13), 0], dtype=np.complex128)
    state = StateVector(amplitudes)
    assert_array_equal(state.amplitudes, amplitudes)
    assert_array_equal(state.probabilities(), amplitudes.real**2)
    assert state.probabilities().sum() != 1


def test_noncontiguous_input_is_copied_correctly():
    backing = np.zeros(8, dtype=np.complex128)
    backing[::2] = random_state(2, seed=232)
    source = backing[::2]
    state = StateVector(source)
    assert_array_equal(state.amplitudes, source)
    backing[:] = 0
    assert_allclose(np.vdot(state.amplitudes, state.amplitudes), 1, atol=ATOL, rtol=0)


def test_valid_complex_state_is_preserved_exactly():
    amplitudes = random_state(4, seed=349)
    state = StateVector(amplitudes)
    assert_array_equal(state.amplitudes, amplitudes)
    assert_allclose(np.vdot(state.amplitudes, state.amplitudes), 1, atol=ATOL, rtol=0)


@pytest.mark.parametrize("num_qubits", (0, -1))
def test_zero_rejects_nonpositive_qubit_count(num_qubits):
    with pytest.raises(ValueError):
        StateVector.zero(num_qubits)


@pytest.mark.parametrize("num_qubits", (True, False, 1.5, "2", None))
def test_zero_rejects_nonintegral_qubit_count(num_qubits):
    with pytest.raises(TypeError):
        StateVector.zero(num_qubits)
