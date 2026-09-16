"""host-only boundary tests, never evidence that a triton kernel executed."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from quantumvis import Circuit, StateVector
from quantumvis.gpu.kernels import (
    BLOCK_SIZE,
    MAX_NUM_QUBITS,
    _matrix_coefficients,
    _validate_num_amplitudes,
)
from quantumvis.gpu.runtime import GPUUnavailableError
from quantumvis.gpu.simulator import GPUResult, GPUSimulator


def test_public_imports_do_not_need_torch_or_triton():
    script = textwrap.dedent("""\
        import importlib.abc
        import sys

        class RejectGPUImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'triton'}:
                    raise AssertionError(f'unexpected optional import: {fullname}')

        sys.meta_path.insert(0, RejectGPUImports())
        from quantumvis import CPUSimulator, Circuit
        from quantumvis.gpu import GPUSimulator
        from quantumvis.gpu.kernels import apply_single_qubit
        from quantumvis.gpu.simulator import GPUResult

        CPUSimulator().run(Circuit(1).x(0))
        GPUSimulator()
        assert 'torch' not in sys.modules and 'triton' not in sys.modules
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.fixture
def runtime_must_not_be_queried(monkeypatch):
    def unexpected_runtime_query(device=None):
        pytest.fail("invalid input must be rejected before querying the GPU runtime")

    monkeypatch.setattr("quantumvis.gpu.simulator.require_gpu", unexpected_runtime_query)


@pytest.mark.parametrize("name", ("cx", "cz"))
def test_controlled_operations_fail_before_runtime_or_allocation(name, runtime_must_not_be_queried):
    # even a supported prefix must not run before the unsupported gate is found.
    circuit = getattr(Circuit(2).h(0), name)(0, 1)
    with pytest.raises(NotImplementedError, match="only single-qubit gates"):
        GPUSimulator().run(circuit)


def test_invalid_circuit_type_fails_before_runtime(runtime_must_not_be_queried):
    with pytest.raises(TypeError, match="circuit must be a Circuit"):
        GPUSimulator().run("H 0")


@pytest.mark.parametrize(
    ("state", "message"),
    (
        ([1, 1], "normalized"),
        ([1, 0, 0], "length"),
        ([[1, 0]], "one-dimensional"),
        ([np.nan, 0], "finite"),
        ([1, 0, 0, 0], "size does not match"),
        (np.array([1, 1], dtype=np.complex64) / np.sqrt(np.float32(2)), "normalized"),
    ),
)
def test_gpu_input_uses_unchanged_strict_cpu_validation(
    state, message, runtime_must_not_be_queried
):
    with pytest.raises(ValueError, match=message):
        GPUSimulator().run(Circuit(1), initial_state=state)


def test_oversized_launch_is_rejected_before_allocation(runtime_must_not_be_queried):
    with pytest.raises(ValueError, match="at most"):
        GPUSimulator().run(Circuit(MAX_NUM_QUBITS + 1))


@pytest.mark.parametrize("device", (-1, True, 0.5, "cuda:0"))
def test_invalid_device_is_rejected(device):
    with pytest.raises((TypeError, ValueError), match="device"):
        GPUSimulator(device)


@pytest.mark.parametrize("state", (None, [1, 0], StateVector.zero(1)))
def test_gpu_requests_propagate_unavailability_without_cpu_fallback(monkeypatch, state):
    def unavailable(device):
        assert device == 2
        raise GPUUnavailableError("test runtime has no NVIDIA GPU")

    monkeypatch.setattr("quantumvis.gpu.simulator.require_gpu", unavailable)
    with pytest.raises(GPUUnavailableError, match="no NVIDIA GPU"):
        GPUSimulator(device=2).run(Circuit(1).h(0), initial_state=state)


def test_gpu_result_owns_readonly_data_and_preserves_norm_drift():
    values = np.array([0.7, 0.7j], dtype=np.complex64)
    original = values.copy()
    result = GPUResult(values)
    values[:] = 0
    assert result.num_qubits == 1
    assert result.amplitudes.dtype == np.complex64
    assert_array_equal(result.amplitudes, original)
    with pytest.raises(ValueError):
        result.amplitudes[0] = 0
    with pytest.raises(ValueError):
        result.amplitudes.flags.writeable = True
    probabilities = result.probabilities()
    assert probabilities.dtype == np.float64
    expected = original.real.astype(np.float64) ** 2 + original.imag.astype(np.float64) ** 2
    assert_array_equal(probabilities, expected)
    assert probabilities.sum() != 1
    probabilities[:] = 0
    assert_array_equal(result.probabilities(), expected)


@pytest.mark.parametrize("values", ([1, 0], np.array([1, 0], dtype=np.complex128)))
def test_gpu_result_refuses_implicit_dtype_conversion(values):
    with pytest.raises(TypeError, match="complex64"):
        GPUResult(values)


@pytest.mark.parametrize("values", ([1], [1, 0, 0], [[1, 0]], [np.nan, 0], [np.inf, 0]))
def test_gpu_result_rejects_invalid_shape_or_nonfinite_amplitudes(values):
    with pytest.raises(ValueError):
        GPUResult(np.array(values, dtype=np.complex64))


def test_matrix_coefficients_preserve_complex_order_and_round_to_float32():
    matrix = np.array([[1 + 2j, 3 + 4j], [5 + 6j, np.sqrt(2) + 8j]], dtype=np.complex128)
    assert _matrix_coefficients(matrix) == (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        float(np.float32(np.sqrt(2))),
        8.0,
    )
    assert _matrix_coefficients(matrix.astype(np.complex64)) == _matrix_coefficients(matrix)


@pytest.mark.parametrize(
    "matrix",
    ([1, 0], np.eye(4), [[1, np.nan], [0, 1]], [[1, np.inf], [0, 1]], [[1e40, 0], [0, 1]]),
)
def test_invalid_matrix_is_rejected_without_gpu_imports(matrix):
    with pytest.raises(ValueError):
        _matrix_coefficients(matrix)


@pytest.mark.parametrize("size", (0, 1, 3, 7, 1 << 40))
def test_invalid_state_lengths_are_rejected_without_allocation(size):
    with pytest.raises(ValueError):
        _validate_num_amplitudes(size)


def test_launch_bounds_include_64_bit_indexed_sizes():
    assert _validate_num_amplitudes(1 << 32) == 32
    assert _validate_num_amplitudes(1 << MAX_NUM_QUBITS) == MAX_NUM_QUBITS
    assert ((1 << (MAX_NUM_QUBITS - 1)) + BLOCK_SIZE - 1) // BLOCK_SIZE <= 2**31 - 1
    assert ((1 << MAX_NUM_QUBITS) + BLOCK_SIZE - 1) // BLOCK_SIZE > 2**31 - 1
