import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from benchmarks.benchmark_gates import MAX_QUBITS, METRIC, _reference_gate, benchmark_gates, main
from quantaforge.gates import SINGLE_QUBIT_GATES, Gate

from .oracle import dense_operator, random_state


@pytest.mark.parametrize("name", sorted(SINGLE_QUBIT_GATES))
@pytest.mark.parametrize("target", range(3))
def test_benchmark_reference_matches_independent_dense_oracle(name, target):
    initial = random_state(3, seed=914)
    angle = 0.731 if name.startswith("R") else None
    gate = Gate(name, target, angle=angle)
    expected = dense_operator(3, name, target, angle=angle) @ initial
    np.testing.assert_allclose(_reference_gate(initial, gate), expected, atol=1e-12, rtol=0)


def test_real_benchmark_schema_and_statistics():
    report = benchmark_gates(
        qubits=(1, 3), operations=tuple(sorted(SINGLE_QUBIT_GATES)), warmups=1, repetitions=3
    )
    assert report["schema_version"] == 1
    assert report["metric"] == METRIC
    assert report["backend"] == "numpy_cpu"
    assert report["dtype"] == "complex128"
    assert report["units"] == "nanoseconds"
    assert report["seed"] == 20260914
    assert report["metadata"]["numpy_version"] == np.__version__
    assert "cpu_model" in report["metadata"]
    assert "git_dirty" in report["metadata"]
    assert "thread_environment" in report["metadata"]
    assert all(len(value) == 64 for value in report["metadata"]["source_sha256"].values())
    assert len(report["cases"]) == 27
    for case in report["cases"]:
        samples = case["timings_ns"]
        assert len(samples) == case["repetitions"] == 3
        assert case["warmups"] == 1
        assert all(isinstance(sample, int) and sample > 0 for sample in samples)
        assert case["median_ns"] == float(np.median(samples))
        assert case["p95_ns"] == float(np.percentile(samples, 95, method="linear"))
        assert case["min_ns"] <= case["median_ns"] <= case["p95_ns"] <= case["max_ns"]
        assert case["state_size"] == 1 << case["num_qubits"]
        assert case["state_bytes"] == case["state_size"] * 16
        assert case["precheck"]["max_amplitude_error"] <= 1e-12
        assert case["precheck"]["squared_norm_error"] <= 1e-12
    assert {(c["num_qubits"], c["target"]) for c in report["cases"]} == {(1, 0), (3, 0), (3, 2)}
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qubits": ()},
        {"qubits": (0,)},
        {"qubits": (MAX_QUBITS + 1,)},
        {"qubits": (True,)},
        {"qubits": (2.5,)},
        {"qubits": (2, 2)},
        {"operations": ()},
        {"operations": ("CX",)},
        {"operations": ("H", "H")},
        {"warmups": 0},
        {"warmups": -1},
        {"warmups": 10001},
        {"warmups": True},
        {"repetitions": 0},
        {"repetitions": 2},
        {"repetitions": 10001},
        {"repetitions": 3.5},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_benchmark_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        benchmark_gates(**kwargs)


def test_cli_writes_json_and_refuses_overwrite(tmp_path):
    output = tmp_path / "results" / "smoke.json"
    args = [
        "--qubits",
        "2",
        "--operations",
        "H",
        "--warmups",
        "1",
        "--repetitions",
        "3",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    saved = output.read_text()
    assert len(json.loads(saved)["cases"]) == 2
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert output.read_text() == saved


def test_cli_invalid_configuration_does_not_create_artifact(tmp_path):
    output = tmp_path / "invalid.json"
    with pytest.raises(SystemExit) as error:
        main(["--qubits", "21", "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("changed_source", (False, True))
def test_benchmark_provenance_checks_the_source_actually_imported(tmp_path, changed_source):
    # check actual imports from a second copy, including a matching source tree.
    root = Path(__file__).resolve().parents[1]
    package = tmp_path / "quantaforge"
    shutil.copytree(root / "src/quantaforge", package, ignore=shutil.ignore_patterns("__pycache__"))
    simulator = package / "cpu/simulator.py"
    if changed_source:
        simulator.write_text(simulator.read_text() + "\n# A different installed source snapshot.\n")
    output = tmp_path / "result.json"
    script = textwrap.dedent("""\
        import sys
        from pathlib import Path
        import quantaforge.cpu.simulator
        from benchmarks.benchmark_gates import main

        assert Path(quantaforge.cpu.simulator.__file__).resolve() == Path(sys.argv[1]).resolve()
        raise SystemExit(main([
            '--qubits', '2', '--operations', 'H', '--warmups', '1', '--repetitions', '3',
            '--output', sys.argv[2],
        ]))
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(simulator), str(output)],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if changed_source:
        assert completed.returncode == 2, completed.stderr
        assert "loaded simulator source differs from this checkout" in completed.stderr
        assert not output.exists()
    else:
        assert completed.returncode == 0, completed.stderr
        metadata = json.loads(output.read_text())["metadata"]
        source = metadata["loaded_simulator_sources"]["quantaforge.cpu.simulator"]
        assert Path(source["path"]) == simulator.resolve()
        assert source["sha256"] == metadata["source_sha256"]["src/quantaforge/cpu/simulator.py"]
