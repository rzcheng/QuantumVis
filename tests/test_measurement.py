import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantumvis import Circuit, CPUSimulator, StateVector

from .oracle import ATOL


def test_probabilities_use_magnitudes_and_float64():
    amplitudes = np.array([1, 2j, -3, -4j]) / np.sqrt(30)
    state = StateVector(amplitudes)
    actual = state.probabilities()
    assert actual.dtype == np.float64
    assert_allclose(actual, np.array([1, 4, 9, 16]) / 30, atol=ATOL, rtol=0)
    assert_allclose(actual.sum(), 1, atol=ATOL, rtol=0)
    assert np.all(actual >= 0)


def test_probability_result_does_not_alias_state():
    state = StateVector([0.6, 0.8j])
    probabilities = state.probabilities()
    probabilities[:] = 0
    assert_array_equal(state.amplitudes, [0.6, 0.8j])
    assert_allclose(state.probabilities(), [0.36, 0.64], atol=ATOL, rtol=0)


@pytest.mark.parametrize("basis_index", range(8))
def test_basis_state_measurement_is_exact(basis_index):
    amplitudes = np.eye(8, dtype=np.complex128)[:, basis_index]
    samples = StateVector(amplitudes).sample(127, seed=1)
    assert samples.dtype == np.int64
    assert samples.shape == (127,)
    assert_array_equal(samples, np.full(127, basis_index))


def test_seeded_sampling_is_reproducible_and_does_not_collapse_state():
    state = CPUSimulator().run(Circuit(2).h(0).cx(0, 1))
    original = state.amplitudes.copy()
    first = state.sample(1000, seed=102)
    assert_array_equal(first, state.sample(1000, seed=102))
    assert not np.array_equal(first, state.sample(1000, seed=103))
    assert set(first) == {0, 3}
    assert_array_equal(state.amplitudes, original)


def test_sampling_matches_born_probabilities():
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    phases = np.exp(1j * np.array([0.1, -0.9, 1.7, 2.9]))
    state = StateVector(np.sqrt(probabilities) * phases)
    shots = 50_000
    counts = np.bincount(state.sample(shots, seed=719), minlength=4)
    # six standard deviations leave room for sampling noise, not systematic bias.
    tolerance = 6 * np.sqrt(probabilities * (1 - probabilities) / shots) + 1 / shots
    assert np.all(np.abs(counts / shots - probabilities) < tolerance)


def test_sampling_does_not_modify_legacy_global_rng():
    previous_rng_state = np.random.get_state()
    try:
        np.random.seed(117)
        expected = np.random.random(8)
        np.random.seed(117)
        StateVector([1, 0]).sample(100, seed=209)
        assert_array_equal(np.random.random(8), expected)
    finally:
        np.random.set_state(previous_rng_state)


def test_zero_shots_returns_empty_integer_array():
    samples = StateVector.zero(1).sample(0, seed=0)
    assert samples.shape == (0,)
    assert samples.dtype == np.int64


def test_negative_shots_rejected():
    with pytest.raises(ValueError):
        StateVector.zero(1).sample(-1)


@pytest.mark.parametrize("shots", (True, False, 1.5, "10", None))
def test_nonintegral_shots_rejected(shots):
    with pytest.raises(TypeError):
        StateVector.zero(1).sample(shots)


def test_numpy_integer_shots_supported():
    assert StateVector.zero(1).sample(np.int64(3), seed=0).shape == (3,)


def test_sampling_handles_accepted_normalization_roundoff_without_mutation():
    state = StateVector([np.sqrt(1 + 2e-13), 0])
    original = state.amplitudes.copy()
    assert_array_equal(state.sample(8, seed=0), np.zeros(8, dtype=np.int64))
    assert_array_equal(state.amplitudes, original)


@pytest.mark.parametrize("seed", (True, False, 1.5, "1", -1))
def test_invalid_seed_rejected(seed):
    expected_error = ValueError if seed == -1 else TypeError
    with pytest.raises(expected_error):
        StateVector.zero(1).sample(1, seed=seed)
