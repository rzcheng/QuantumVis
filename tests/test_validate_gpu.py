"""cpu-only validation-command control flow; none of these tests executes a gpu."""

import json
from dataclasses import replace

import pytest

from quantumvis import validate_gpu
from quantumvis.gpu.runtime import GPUStatus


def unavailable_status():
    return GPUStatus(
        usable=False,
        reason="PyTorch unavailable: install the optional GPU dependencies",
        torch_available=False,
        triton_available=False,
        cuda_available=False,
        platform="Linux",
        python_version="3.12.11",
        torch_version=None,
        triton_version=None,
        cuda_version=None,
        gpu_name=None,
        compute_capability=None,
        device_index=None,
    )


def test_unavailable_json_has_metadata_and_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(validate_gpu, "gpu_status", lambda device: unavailable_status())
    assert validate_gpu.main(["--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "UNAVAILABLE"
    assert report["checks_passed"] == 0
    assert report["checks"] == []
    assert report["environment"]["gpu_name"] is None
    assert report["environment"]["torch_available"] is False
    assert "complex64" in report["dtype"]
    assert report["cpu_oracle_dtype"] == "complex128"
    assert report["numpy_version"] == validate_gpu.np.__version__
    assert report["timestamp_utc"].endswith("+00:00")
    assert report["tolerances"]["single_gate_atol"] == 1e-6
    assert "PyTorch unavailable" in report["failure"]


def test_unavailable_readable_report_does_not_claim_pass(monkeypatch, capsys):
    monkeypatch.setattr(validate_gpu, "gpu_status", lambda device: unavailable_status())
    assert validate_gpu.main([]) == 2
    output = capsys.readouterr().out
    assert "ENVIRONMENT" in output
    assert "UNAVAILABLE: 0 deterministic real-GPU checks passed" in output
    assert "PASS:" not in output
    assert "PyTorch unavailable" in output


def test_device_argument_is_forwarded(monkeypatch):
    devices = []

    def status(device):
        devices.append(device)
        return unavailable_status()

    monkeypatch.setattr(validate_gpu, "gpu_status", status)
    assert validate_gpu.main(["--device", "3", "--json"]) == 2
    assert devices == [3]


@pytest.mark.parametrize("arguments", (["--device", "-1"], ["--device", "abc"], ["--unknown"]))
def test_invalid_arguments_exit_two(arguments):
    with pytest.raises(SystemExit) as error:
        validate_gpu.main(arguments)
    assert error.value.code == 2


def test_help_exits_without_runtime_detection(monkeypatch, capsys):
    def unexpected_detection(device):
        raise AssertionError("help should not import or inspect optional runtime dependencies")

    monkeypatch.setattr(validate_gpu, "gpu_status", unexpected_detection)
    with pytest.raises(SystemExit) as error:
        validate_gpu.main(["--help"])
    assert error.value.code == 0
    assert "--json" in capsys.readouterr().out


def test_detection_configuration_errors_remain_debuggable(monkeypatch):
    def broken_status(device):
        raise RuntimeError("CUDA driver configuration failure")

    monkeypatch.setattr(validate_gpu, "gpu_status", broken_status)
    with pytest.raises(RuntimeError, match="CUDA driver configuration failure"):
        validate_gpu.main([])


@pytest.mark.parametrize("json_output", (False, True))
def test_unexpected_validation_errors_are_not_downgraded_to_unavailable(
    monkeypatch, capsys, json_output
):
    # this tests propagation only. no gpu output or numerical success is mocked.
    status = replace(unavailable_status(), usable=True, reason="usable", device_index=0)
    monkeypatch.setattr(validate_gpu, "gpu_status", lambda device: status)

    def broken_checks(device):
        raise RuntimeError("Triton compiler failure")

    monkeypatch.setattr(validate_gpu, "run_checks", broken_checks)
    with pytest.raises(RuntimeError, match="Triton compiler failure"):
        validate_gpu.main(["--json"] if json_output else [])
    output = capsys.readouterr().out
    if json_output:
        report = json.loads(output)
        assert report["status"] == "FAIL"
        assert report["environment"]["device_index"] == 0
        assert report["checks_passed"] == 0
        assert "RuntimeError: Triton compiler failure" in report["failure"]
    else:
        assert "ENVIRONMENT" in output
        assert "FAIL:" in output


def test_reported_validation_failure_has_nonzero_exit(monkeypatch, capsys):
    # exercise a named error pathway, not a fabricated kernel or fake gpu result.
    status = replace(unavailable_status(), usable=True, reason="usable", device_index=0)
    monkeypatch.setattr(validate_gpu, "gpu_status", lambda device: status)

    def failed_checks(device):
        raise validate_gpu.ValidationFailure("reported numerical mismatch")

    monkeypatch.setattr(validate_gpu, "run_checks", failed_checks)
    assert validate_gpu.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert report["checks_passed"] == 0
    assert report["failure"] == "reported numerical mismatch"


# these are host-side tests of the numerical acceptance checker, not kernel
# simulations or evidence of gpu numerical correctness.
def test_checker_accepts_known_float32_rounded_analytical_state():
    import numpy as np

    expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    actual = expected.astype(np.complex64)
    report = validate_gpu.check_output("rounded-H", actual, expected)
    assert 0 < report.relative_l2_error < 1e-6
    assert 0 < report.norm_drift < 1e-6


@pytest.mark.parametrize("actual", ([1j, 0], [0, 1]))
def test_checker_rejects_norm_preserving_phase_or_index_error(actual):
    import numpy as np

    assert np.vdot(actual, actual).real == 1
    with pytest.raises(validate_gpu.ValidationFailure):
        validate_gpu.check_output("wrong-unitary", actual, [1, 0])


def test_checker_rejects_distributed_error_hidden_by_componentwise_tolerance():
    import numpy as np

    expected = np.full(16384, 1 / 128, dtype=np.complex128)
    actual = expected + 5e-7j
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert abs(np.vdot(actual, actual).real - 1) < 1e-6
    with pytest.raises(validate_gpu.ValidationFailure, match="relative L2"):
        validate_gpu.check_output("distributed-error", actual, expected)


def test_checker_rejects_norm_drift_even_when_amplitude_and_l2_pass():
    import numpy as np

    expected = np.array([1, 0], dtype=np.complex128)
    actual = (1 + 7.5e-7) * expected
    assert np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert np.linalg.norm(actual - expected) < 1e-6
    with pytest.raises(validate_gpu.ValidationFailure, match="squared-norm drift"):
        validate_gpu.check_output("norm-error", actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    (
        ([float("nan"), 0], [1, 0], "nonfinite"),
        ([1, 0], [float("inf"), 0], "nonfinite"),
        ([1, 0, 0, 0], [1, 0], "output shape"),
        ([[1, 0]], [[1, 0]], "output shape"),
    ),
)
def test_checker_rejects_nonfinite_or_malformed_arrays(actual, expected, message):
    with pytest.raises(validate_gpu.ValidationFailure, match=message):
        validate_gpu.check_output("invalid-state", actual, expected)


def test_checker_rejects_zero_reference():
    with pytest.raises(ValueError, match="nonzero norm"):
        validate_gpu.check_output("invalid-reference", [0, 0], [0, 0])


def test_depth_policy_keeps_single_gate_threshold_and_scales_short_circuits():
    assert validate_gpu.error_limits(0) == (1e-6, 1e-6, 1e-6)
    assert validate_gpu.error_limits(1) == (1e-6, 1e-6, 1e-6)
    amplitude, l2, norm = validate_gpu.error_limits(64)
    assert amplitude == l2 == 1e-6 + 63 * 4 * 2**-23
    assert norm == 1e-6 + 63 * 8 * 2**-23


@pytest.mark.parametrize("depth", (True, -1, 65, 1.5, "2"))
def test_depth_policy_rejects_unsupported_depth(depth):
    with pytest.raises(ValueError, match="integer between 0 and 64"):
        validate_gpu.error_limits(depth)
