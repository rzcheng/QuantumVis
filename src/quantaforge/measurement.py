"""Computational-basis probabilities and non-collapsing seeded sampling."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantaforge._validation import integer

# Input and output validation is deliberately stricter than prospective float32 GPU checks.
NORM_ATOL = 1e-12


def _validated_amplitudes(amplitudes: ArrayLike) -> NDArray[np.complex128]:
    state = np.asarray(amplitudes, dtype=np.complex128)
    if state.ndim != 1 or state.size < 2 or state.size & (state.size - 1):
        raise ValueError("state must be one-dimensional with length 2**n for n >= 1")
    if not np.all(np.isfinite(state)):
        raise ValueError("state amplitudes must be finite")
    norm_squared = float(np.vdot(state, state).real)
    if not np.isclose(norm_squared, 1.0, atol=NORM_ATOL, rtol=0):
        raise ValueError(f"state must be normalized (squared norm was {norm_squared})")
    return state


def probabilities(amplitudes: ArrayLike) -> NDArray[np.float64]:
    """Return |amplitude|**2 after validating the input state; do not rescale it."""
    state = _validated_amplitudes(amplitudes)
    return state.real**2 + state.imag**2


def sample(amplitudes: ArrayLike, shots: int, *, seed: int | None = None) -> NDArray[np.int64]:
    """Draw basis indices with replacement; the input state does not collapse.

    Only the probability weights used by the RNG are rescaled to sum to one,
    after checking the state norm, to remove harmless floating-point drift.
    """
    shots = integer(shots, "shots")
    if seed is not None:
        seed = integer(seed, "seed")
    weights = probabilities(amplitudes)
    weights /= weights.sum()
    return np.random.default_rng(seed).choice(weights.size, size=shots, p=weights).astype(np.int64)
