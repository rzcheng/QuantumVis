"""interpreter validation of the production kernel, never device acceptance."""

import hashlib
import os
import platform
from pathlib import Path

import numpy as np
import pytest

from quantumvis import Circuit, CPUSimulator
from quantumvis.gates import Gate, single_qubit_matrix
from quantumvis.gpu.kernels import BLOCK_SIZE, _matrix_coefficients
from quantumvis.validate_gpu import (
    GATE_NAMES,
    SEED,
    analytical_cases,
    check_output,
    random_circuit,
    random_state,
)

pytestmark = pytest.mark.interpreter
ANALYTICAL_CASES = list(analytical_cases())
POSITIONS = [(n, t) for n in (1, 5, 10) for t in sorted({0, n // 2, n - 1})]


@pytest.fixture(scope="module")
def interpret(record_testsuite_property):
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("interpreter validation requires a separate TRITON_INTERPRET=1 process")
    if platform.system() != "Linux":
        pytest.fail("the optional interpreter validation environment requires Linux")
    # requested validation must fail on broken imports, not silently skip.
    import torch
    import triton

    from quantumvis.gpu.kernels import single_qubit

    kernel = single_qubit._single_qubit_kernel
    assert not isinstance(kernel, triton.JITFunction), "restart with TRITON_INTERPRET=1"
    for name, value in {
        "evidence_level": "interpreter validation",
        "python_version": platform.python_version(),
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "precision": "split float32",
        "kernel_sha256": hashlib.sha256(Path(single_qubit.__file__).read_bytes()).hexdigest(),
    }.items():
        record_testsuite_property(name, value)

    def run(circuit, initial):
        real = torch.tensor(initial.real.copy(), dtype=torch.float32, device="cpu")
        imag = torch.tensor(initial.imag.copy(), dtype=torch.float32, device="cpu")
        pairs = real.numel() // 2
        for gate in circuit.operations:
            assert gate.control is None
            # bypass only the cuda storage wrapper, never the production kernel math.
            kernel[(triton.cdiv(pairs, BLOCK_SIZE),)](
                real,
                imag,
                pairs,
                *_matrix_coefficients(single_qubit_matrix(gate)),
                TARGET=gate.target,
                BLOCK=BLOCK_SIZE,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return real.numpy() + 1j * imag.numpy()

    return run


@pytest.mark.parametrize(
    ("name", "circuit", "initial", "expected"),
    ANALYTICAL_CASES,
    ids=[case[0] for case in ANALYTICAL_CASES],
)
def test_interpreter_analytical_states(interpret, name, circuit, initial, expected):
    actual = interpret(circuit, initial)
    check_output(name, actual, expected, depth=len(circuit.operations))
    oracle = CPUSimulator().run(circuit, initial_state=initial).amplitudes
    check_output(name + "-cpu", actual, oracle, depth=len(circuit.operations))


@pytest.mark.parametrize("name", GATE_NAMES)
@pytest.mark.parametrize(("num_qubits", "target"), POSITIONS)
@pytest.mark.parametrize("seed_offset", (0, 101))
def test_interpreter_random_gate(interpret, name, num_qubits, target, seed_offset):
    initial = random_state(num_qubits, SEED + seed_offset + num_qubits + target)
    angle = -0.731 if name.startswith("R") else None
    circuit = Circuit(num_qubits).add(Gate(name, target, angle=angle))
    actual = interpret(circuit, initial)
    expected = CPUSimulator().run(circuit, initial_state=initial).amplitudes
    check_output(f"interpreter-{name}-n{num_qubits}-t{target}", actual, expected)


@pytest.mark.parametrize("num_qubits", (1, 5, 10))
@pytest.mark.parametrize("depth", (8, 16))
def test_interpreter_short_circuit(interpret, num_qubits, depth):
    initial = random_state(num_qubits, SEED + num_qubits)
    circuit = random_circuit(num_qubits, depth, SEED + depth)
    actual = interpret(circuit, initial)
    expected = CPUSimulator().run(circuit, initial_state=initial).amplitudes
    check_output("interpreter-circuit", actual, expected, depth=depth)
