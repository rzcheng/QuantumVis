import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantumvis import Circuit, CPUSimulator
from scripts.export_web_demo import BENCHMARK, DEFAULT_OUTPUT, ROOT, build_export, export, record

from .oracle import dense_operator


def amplitudes(step):
    return np.array(step["real"]) + 1j * np.array(step["imag"])


@pytest.fixture(scope="module")
def payload():
    return build_export()


@pytest.mark.parametrize("committed", [False, True])
def test_every_prefix_initial_norm_and_probability(payload, committed):
    if committed:
        payload = json.loads((DEFAULT_OUTPUT / "circuits.json").read_text())
    for circuit in payload["circuits"]:
        size = 1 << circuit["num_qubits"]
        for recording in circuit["recordings"]:
            assert len(recording["steps"]) == len(recording["operations"]) + 1
            expected = np.zeros(size, dtype=complex)
            expected[0] = 1
            for index, step in enumerate(recording["steps"]):
                if index:
                    gate = recording["operations"][index - 1]
                    expected = (
                        dense_operator(
                            circuit["num_qubits"],
                            gate["name"],
                            gate["target"],
                            angle=gate["angle"],
                            control=gate["control"],
                        )
                        @ expected
                    )
                actual = amplitudes(step)
                assert_allclose(actual, expected, atol=1e-12, rtol=0)
                assert_allclose(step["probabilities"], np.abs(expected) ** 2, atol=1e-12, rtol=0)
                assert_allclose(np.vdot(actual, actual), 1, atol=1e-12, rtol=0)
                assert_allclose(sum(step["probabilities"]), 1, atol=1e-12, rtol=0)
                assert np.all(np.isfinite(actual))


@pytest.mark.parametrize("index,size", [(0, 4), (1, 8)])
def test_entangled_final_amplitudes(payload, index, size):
    expected = np.zeros(size, dtype=complex)
    expected[[0, -1]] = 1 / np.sqrt(2)
    final = payload["circuits"][index]["recordings"][0]["steps"][-1]
    assert_allclose(amplitudes(final), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("index", range(33))
def test_recorded_angles_relative_phase_and_interference(payload, index):
    recordings = payload["circuits"][2]["recordings"]
    assert len(recordings) == 33
    recording = recordings[index]
    angle = index * np.pi / 16
    assert recording["angle_radians"] == angle
    initial, after_h, after_rz, final = recording["steps"]
    assert_array_equal(amplitudes(initial), [1, 0])
    assert_allclose(after_rz["probabilities"], after_h["probabilities"], atol=1e-12, rtol=0)
    assert_allclose(
        amplitudes(after_rz),
        np.array([np.exp(-0.5j * angle), np.exp(0.5j * angle)]) / np.sqrt(2),
        atol=1e-12,
        rtol=0,
    )
    assert_allclose(
        amplitudes(final), [np.cos(angle / 2), -1j * np.sin(angle / 2)], atol=1e-12, rtol=0
    )
    if index in (0, 16, 32):
        expected = {0: [1, 0], 16: [0, -1j], 32: [-1, 0]}[index]
        assert_allclose(amplitudes(final), expected, atol=1e-12, rtol=0)
        assert_allclose(final["probabilities"], np.abs(expected) ** 2, atol=1e-12, rtol=0)


@pytest.mark.parametrize("seed", [17, 812, 20260916])
def test_record_random_angles_against_analytical_formula(seed):
    for angle in np.random.default_rng(seed).uniform(-8 * np.pi, 8 * np.pi, 20):
        result = record(Circuit(1).h(0).rz(0, angle).h(0), float(angle))
        assert_allclose(
            amplitudes(result["steps"][-1]),
            [np.cos(angle / 2), -1j * np.sin(angle / 2)],
            atol=1e-12,
            rtol=0,
        )


def test_export_runs_each_prefix_including_initial(monkeypatch):
    original = CPUSimulator.run
    calls = []

    def spy(self, circuit):
        calls.append(len(circuit.operations))
        return original(self, circuit)

    monkeypatch.setattr(CPUSimulator, "run", spy)
    record(Circuit(2).h(0).cx(0, 1))
    assert calls == [0, 1, 2]


def test_metadata_hashes_and_deterministic_finite_export(tmp_path, payload):
    metadata = payload["metadata"]
    assert payload["schema_version"] == 1
    assert metadata["backend"] == "numpy_cpu"
    assert metadata["precision"] == "complex128"
    assert "q0 is least significant" in metadata["bit_order"]
    assert metadata["numpy_version"] == np.__version__
    for path, digest in metadata["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    export(tmp_path)
    first = (tmp_path / "circuits.json").read_bytes()
    export(tmp_path)
    assert (tmp_path / "circuits.json").read_bytes() == first
    assert json.loads(first) == payload
    assert b"NaN" not in first and b"Infinity" not in first
    assert (tmp_path / "cpu-baseline.json").read_bytes() == BENCHMARK.read_bytes()
    assert (DEFAULT_OUTPUT / "cpu-baseline.json").read_bytes() == BENCHMARK.read_bytes()


def test_export_does_not_import_optional_gpu_dependencies(tmp_path):
    code = (
        "import sys; from pathlib import Path; from scripts.export_web_demo import export; "
        "export(Path(sys.argv[1])); "
        "assert not any(n.split('.')[0] in ('torch', 'triton') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, check=True)
