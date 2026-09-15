"""These checks require actual NVIDIA execution; missing capability is an explicit skip."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantaforge import Circuit, CPUSimulator, StateVector
from quantaforge.gates import Gate
from quantaforge.gpu import gpu_status
from quantaforge.validate_gpu import (
    GATE_NAMES,
    SEED,
    analytical_cases,
    check_output,
    random_circuit,
    random_state,
    raw_unitary_check,
)

pytestmark = pytest.mark.gpu
POSITIONS = [(n, target) for n in (1, 2, 5, 9, 10) for target in range(n)]
ANALYTICAL_CASES = list(analytical_cases())


@pytest.fixture(scope="module")
def gpu_device():
    status = gpu_status()
    if not status.usable:
        pytest.skip(status.reason)
    # Compiler, imports, allocation and execution errors after this point fail.
    return status.device_index


@pytest.fixture(scope="module")
def gpu_simulator(gpu_device):
    from quantaforge.gpu import GPUSimulator

    return GPUSimulator(device=gpu_device)


@pytest.mark.parametrize("name", GATE_NAMES)
@pytest.mark.parametrize(("num_qubits", "target"), POSITIONS)
@pytest.mark.parametrize("seed_offset", (0, 101))
def test_random_single_gate_matches_cpu(gpu_simulator, name, num_qubits, target, seed_offset):
    state = random_state(num_qubits, SEED + seed_offset + num_qubits * 11 + target)
    original = state.copy()
    angle = (-0.731 if seed_offset == 0 else 2.193) if name.startswith("R") else None
    circuit = Circuit(num_qubits).add(Gate(name, target, angle=angle))
    actual = gpu_simulator.run(circuit, initial_state=state)
    expected = CPUSimulator().run(circuit, initial_state=state)

    check_output(name, actual.amplitudes, expected.amplitudes)
    assert actual.amplitudes.dtype == np.complex64
    assert actual.num_qubits == num_qubits
    assert not actual.amplitudes.flags.writeable
    assert_array_equal(state, original)
    # The result computes probabilities from its actual rounded amplitudes, with
    # no CPU StateVector normalization gate or concealed renormalization.
    exact_probabilities = np.abs(actual.amplitudes.astype(np.complex128)) ** 2
    assert_allclose(actual.probabilities(), exact_probabilities, atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize(
    ("name", "circuit", "initial", "expected"),
    ANALYTICAL_CASES,
    ids=[case[0] for case in ANALYTICAL_CASES],
)
def test_known_analytical_states(gpu_simulator, name, circuit, initial, expected):
    actual = gpu_simulator.run(circuit, initial_state=initial).amplitudes
    check_output(name, actual, expected, depth=len(circuit.operations))


@pytest.mark.parametrize(("num_qubits", "target"), POSITIONS)
def test_h_twice_restores_random_complex_state(gpu_simulator, num_qubits, target):
    initial = random_state(num_qubits, SEED + target)
    circuit = Circuit(num_qubits).h(target).h(target)
    actual = gpu_simulator.run(circuit, initial_state=initial).amplitudes
    check_output("H-H", actual, initial, depth=2)


@pytest.mark.parametrize("name", ("RX", "RY", "RZ"))
@pytest.mark.parametrize("angle", (0.0, -0.713, np.pi, 2 * np.pi, 17.2))
@pytest.mark.parametrize("target", (0, 2, 4))
def test_inverse_rotation_restores_random_state(gpu_simulator, name, angle, target):
    initial = random_state(5, SEED + target)
    circuit = Circuit(5).add(Gate(name, target, angle=angle)).add(Gate(name, target, angle=-angle))
    actual = gpu_simulator.run(circuit, initial_state=initial).amplitudes
    check_output("inverse-" + name, actual, initial, depth=2)


@pytest.mark.parametrize("num_qubits", (1, 2, 5, 10))
@pytest.mark.parametrize("depth", (1, 16, 64))
@pytest.mark.parametrize("seed_offset", (0, 101))
def test_random_circuit_matches_cpu(gpu_simulator, num_qubits, depth, seed_offset):
    seed = SEED + num_qubits + depth + seed_offset
    initial = random_state(num_qubits, seed)
    circuit = random_circuit(num_qubits, depth, seed + 1)
    actual = gpu_simulator.run(circuit, initial_state=initial).amplitudes
    expected = CPUSimulator().run(circuit, initial_state=initial).amplitudes
    check_output("random-circuit", actual, expected, depth=depth)


@pytest.mark.parametrize(("num_qubits", "target"), [(n, t) for n in (1, 2, 3, 5) for t in range(n)])
@pytest.mark.parametrize("seed_offset", (0, 101))
def test_generic_complex_unitary_against_independent_rounded_dense_oracle(
    gpu_device, num_qubits, target, seed_offset
):
    raw_unitary_check(num_qubits, target, SEED + seed_offset + target, device=gpu_device)


@pytest.mark.parametrize("num_qubits", (1, 5, 10))
def test_default_zero_and_empty_circuit(gpu_simulator, num_qubits):
    expected = StateVector.zero(num_qubits)
    actual = gpu_simulator.run(Circuit(num_qubits))
    check_output("zero-initialization", actual.amplitudes, expected.amplitudes, depth=0)


@pytest.mark.parametrize("num_qubits", (1, 5, 10))
def test_cpu_statevector_input_and_prior_result_are_preserved(gpu_simulator, num_qubits):
    initial = StateVector(random_state(num_qubits, SEED + num_qubits))
    original = initial.amplitudes.copy()
    first = gpu_simulator.run(Circuit(num_qubits).h(0), initial_state=initial)
    first_copy = first.amplitudes.copy()
    gpu_simulator.run(Circuit(num_qubits).x(0))
    assert_array_equal(initial.amplitudes, original)
    assert_array_equal(first.amplitudes, first_copy)
    check_output(
        "StateVector-input",
        first.amplitudes,
        CPUSimulator().run(Circuit(num_qubits).h(0), initial_state=initial).amplitudes,
    )


@pytest.mark.parametrize(
    ("case", "exception", "message"),
    (
        ("dtype", TypeError, "dtype torch.float32"),
        ("noncontiguous", ValueError, "contiguous one-dimensional"),
        ("rank", ValueError, "contiguous one-dimensional"),
        ("alias", ValueError, "distinct storage"),
        ("shared_views", ValueError, "distinct storage"),
        ("autograd", ValueError, "autograd"),
        ("shape", ValueError, "equal shape"),
        ("cpu", ValueError, "CUDA device"),
    ),
)
def test_raw_storage_contract_rejects_invalid_inputs(gpu_device, case, exception, message):
    import torch

    from quantaforge.gpu.kernels.single_qubit import apply_single_qubit

    device = torch.device("cuda", gpu_device)
    real = torch.zeros(8, dtype=torch.float32, device=device)
    imag = torch.zeros_like(real)
    if case == "dtype":
        real = real.to(torch.float64)
    elif case == "noncontiguous":
        real = torch.zeros(16, dtype=torch.float32, device=device)[::2]
    elif case == "rank":
        real = real.reshape(2, 4)
    elif case == "alias":
        imag = real
    elif case == "shared_views":
        storage = torch.zeros(16, dtype=torch.float32, device=device)
        real, imag = storage[:8], storage[8:]
    elif case == "autograd":
        real.requires_grad_(True)
    elif case == "shape":
        imag = torch.zeros(4, dtype=torch.float32, device=device)
    elif case == "cpu":
        real = real.cpu()
    with pytest.raises(exception, match=message):
        apply_single_qubit(real, imag, np.eye(2, dtype=np.complex64), 0)


def test_nondefault_cuda_stream_preserves_order(gpu_device, gpu_simulator):
    import torch

    stream = torch.cuda.Stream(device=gpu_device)
    initial = random_state(5, SEED)
    circuit = random_circuit(5, 16, SEED + 1)
    expected = CPUSimulator().run(circuit, initial_state=initial).amplitudes
    with torch.cuda.stream(stream):
        raw_unitary_check(5, 4, SEED + 2, device=gpu_device)
        actual = gpu_simulator.run(circuit, initial_state=initial).amplitudes
    check_output("nondefault-stream", actual, expected, depth=16)
