"""host-only timing boundaries and artifact guards; no gpu timings are produced."""

import json
from dataclasses import asdict

import pytest

from benchmarks import benchmark_gpu_smoke as smoke
from quantumvis.gpu.runtime import GPUStatus
from scripts.validate_nvidia import EXPECTED_GPU_TESTS, EXPECTED_VALIDATOR_CHECKS


def test_timed_call_synchronizes_on_both_sides(monkeypatch):
    events = []
    ticks = iter((10, 17))
    sentinel = object()

    def clock():
        events.append("clock")
        return next(ticks)

    def operation():
        events.append("operation")
        return sentinel

    monkeypatch.setattr(smoke.time, "perf_counter_ns", clock)
    result, elapsed = smoke.timed_call(operation, lambda: events.append("sync"))
    assert result is sentinel and elapsed == 7
    assert events == ["sync", "clock", "operation", "sync", "clock"]


@pytest.fixture
def synthetic_gpu_status():
    return GPUStatus(
        usable=True,
        reason="synthetic protocol fixture; not device evidence",
        torch_available=True,
        triton_available=True,
        cuda_available=True,
        platform="Linux",
        python_version="synthetic-python",
        torch_version="synthetic-torch",
        triton_version="synthetic-triton",
        cuda_version="synthetic-cuda",
        gpu_name="synthetic-gpu",
        compute_capability=(8, 7),
        device_index=0,
    )


@pytest.fixture
def synthetic_acceptance(tmp_path, monkeypatch, synthetic_gpu_status):
    # artifact protocol only; these reports are not device validation evidence.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    path = tmp_path / "summary.json"
    report = {
        "schema_version": 1,
        "status": "PASS",
        "gpu_tests": {
            "tests": EXPECTED_GPU_TESTS,
            "passed": EXPECTED_GPU_TESTS,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
        "source_sha256": smoke.source_hashes(smoke.ROOT),
        "commands": {"validator": {"exit_code": 0}, "pytest": {"exit_code": 0}},
        "device_metadata": {"complete": True},
        "cuda_visible_devices": "0",
    }
    validator = {
        "schema_version": 1,
        "status": "PASS",
        "environment": asdict(synthetic_gpu_status),
        "numpy_version": smoke.np.__version__,
        "checks": [{} for _ in range(EXPECTED_VALIDATOR_CHECKS)],
        "checks_passed": EXPECTED_VALIDATOR_CHECKS,
    }
    path.write_text(json.dumps(report))
    path.with_name("validator.stdout").write_text(json.dumps(validator))
    write_synthetic_junit(path.with_name("gpu-tests.xml"))
    return path, report


def write_synthetic_junit(path, tests=EXPECTED_GPU_TESTS):
    # invented testcase names exercise parsing only, never device correctness.
    cases = "".join(f'<testcase name="synthetic-{index}"/>' for index in range(tests))
    path.write_text(
        f'<testsuites><testsuite tests="{tests}" failures="0" errors="0" skipped="0">'
        f"{cases}</testsuite></testsuites>"
    )


def update_validator(path, **changes):
    validator_path = path.with_name("validator.stdout")
    validator = json.loads(validator_path.read_text())
    validator.update(changes)
    validator_path.write_text(json.dumps(validator))


def test_acceptance_requires_matching_source(synthetic_acceptance, synthetic_gpu_status):
    path, report = synthetic_acceptance
    assert smoke.require_acceptance(path, synthetic_gpu_status) == report
    report["source_sha256"] = {"stale": "source"}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="this source"):
        smoke.require_acceptance(path, synthetic_gpu_status)


@pytest.mark.parametrize(
    "condition", ("status", "empty", "partial", "failed", "skipped", "missing", "malformed")
)
def test_incomplete_acceptance_blocks_before_gpu_detection(
    synthetic_acceptance, monkeypatch, condition
):
    path, report = synthetic_acceptance
    if condition == "status":
        report["status"] = "UNAVAILABLE"
    elif condition == "empty":
        report["gpu_tests"].update(tests=0, passed=0)
    elif condition == "partial":
        report["gpu_tests"].update(tests=1, passed=1)
    elif condition == "missing":
        report.pop("gpu_tests")
    elif condition == "malformed":
        report["gpu_tests"] = [EXPECTED_GPU_TESTS]
    else:
        report["gpu_tests"]["passed"] = 0
        report["gpu_tests"]["skipped" if condition == "skipped" else "failures"] = 1
    path.write_text(json.dumps(report))
    monkeypatch.setattr(smoke, "require_gpu", lambda: pytest.fail("GPU must not be touched"))
    with pytest.raises(ValueError, match="acceptance"):
        smoke.benchmark(path)


@pytest.mark.parametrize("field", ("tests", "passed", "failures", "errors", "skipped"))
@pytest.mark.parametrize("replacement", (None, True, False, "0", 0.0))
def test_acceptance_rejects_malformed_counts(synthetic_acceptance, field, replacement):
    path, report = synthetic_acceptance
    report["gpu_tests"][field] = replacement
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="acceptance"):
        smoke.require_acceptance(path)


@pytest.mark.parametrize("name", ("summary.json", "validator.stdout"))
@pytest.mark.parametrize("contents", (None, "", "{", "[]", "null", "false", "{}"))
def test_cli_rejects_missing_or_malformed_artifacts_without_timing(
    synthetic_acceptance, monkeypatch, name, contents
):
    path, _ = synthetic_acceptance
    artifact = path.with_name(name)
    if contents is None:
        artifact.unlink()
    else:
        artifact.write_text(contents)
    monkeypatch.setattr(smoke, "require_gpu", lambda: pytest.fail("GPU must not be touched"))
    output = path.with_name("timing.json")
    assert smoke.main(["--acceptance", str(path), "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "condition",
    ("missing", "truncated", "empty", "missing_counter", "invalid_counter", "summary_mismatch"),
)
def test_invalid_junit_blocks_before_gpu_detection(
    synthetic_acceptance, monkeypatch, capsys, condition
):
    path, _ = synthetic_acceptance
    junit = path.with_name("gpu-tests.xml")
    if condition == "missing":
        junit.unlink()
    elif condition == "truncated":
        junit.write_text(junit.read_text()[:-20])
    elif condition == "empty":
        junit.write_text("<testsuites/>")
    elif condition == "missing_counter":
        junit.write_text(junit.read_text().replace(' errors="0"', ""))
    elif condition == "invalid_counter":
        junit.write_text(junit.read_text().replace(' errors="0"', ' errors="invalid"'))
    else:
        write_synthetic_junit(junit, tests=1)
    monkeypatch.setattr(smoke, "require_gpu", lambda: pytest.fail("GPU must not be touched"))
    output = path.with_name("timing.json")
    assert smoke.main(["--acceptance", str(path), "--output", str(output)]) == 2
    assert not output.exists()
    assert "gpu-tests.xml" in capsys.readouterr().err


@pytest.mark.parametrize("name", ("summary.json", "validator.stdout"))
@pytest.mark.parametrize("version", (None, True, "1", 1.0, 2))
def test_malformed_or_unsupported_schema_is_rejected(synthetic_acceptance, name, version):
    path, _ = synthetic_acceptance
    artifact = path.with_name(name)
    report = json.loads(artifact.read_text())
    report["schema_version"] = version
    artifact.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="acceptance"):
        smoke.require_acceptance(path)


@pytest.mark.parametrize(
    "changes",
    (
        {"status": "FAIL"},
        {"checks": [{}], "checks_passed": 1},
        {"checks_passed": EXPECTED_VALIDATOR_CHECKS - 1},
        {"checks": None},
        {"environment": None},
        {"environment": []},
        {"environment": {"usable": False}},
        {"numpy_version": "different-version"},
        {"numpy_version": None},
    ),
)
def test_incomplete_validator_blocks_before_gpu_detection(
    synthetic_acceptance, monkeypatch, changes
):
    path, _ = synthetic_acceptance
    update_validator(path, **changes)
    monkeypatch.setattr(smoke, "require_gpu", lambda: pytest.fail("GPU must not be touched"))
    with pytest.raises(ValueError, match="acceptance"):
        smoke.benchmark(path)


@pytest.mark.parametrize(
    "replacement",
    (
        None,
        [],
        {},
        {"validator": None},
        {"validator": {"exit_code": 1}, "pytest": {"exit_code": 0}},
        {"validator": {"exit_code": 0}, "pytest": {"exit_code": 1}},
        {"validator": {"exit_code": False}, "pytest": {"exit_code": 0}},
        {"validator": {"exit_code": 0}, "pytest": {"exit_code": "0"}},
    ),
)
def test_acceptance_requires_successful_commands(synthetic_acceptance, replacement):
    path, report = synthetic_acceptance
    report["commands"] = replacement
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="successful commands"):
        smoke.require_acceptance(path)


@pytest.mark.parametrize("replacement", (None, [], {}, {"complete": False}, {"complete": 1}))
def test_acceptance_requires_device_metadata(synthetic_acceptance, replacement):
    path, report = synthetic_acceptance
    report["device_metadata"] = replacement
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete device metadata"):
        smoke.require_acceptance(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("usable", False),
        ("reason", "different-reason"),
        ("torch_available", False),
        ("triton_available", False),
        ("cuda_available", False),
        ("platform", "different-platform"),
        ("python_version", "different-python"),
        ("torch_version", "different-torch"),
        ("triton_version", "different-triton"),
        ("cuda_version", "different-cuda"),
        ("gpu_name", "different-gpu"),
        ("compute_capability", [8, 0]),
        ("compute_capability", [8.0, 7.0]),
        ("device_index", 1),
        ("device_index", False),
    ),
)
def test_runtime_mismatch_creates_no_timing_artifact(
    synthetic_acceptance, synthetic_gpu_status, monkeypatch, capsys, field, replacement
):
    path, _ = synthetic_acceptance
    environment = asdict(synthetic_gpu_status) | {field: replacement}
    update_validator(path, environment=environment)
    monkeypatch.setattr(smoke, "require_gpu", lambda: synthetic_gpu_status)
    monkeypatch.setattr(smoke, "timed_call", lambda *_: pytest.fail("timing must not start"))
    output = path.with_name("timing.json")
    assert smoke.main(["--acceptance", str(path), "--output", str(output)]) == 2
    assert not output.exists()
    expected = "complete passing validator.stdout" if field == "usable" else field
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("field", tuple(GPUStatus.__dataclass_fields__))
def test_missing_runtime_field_is_rejected(synthetic_acceptance, synthetic_gpu_status, field):
    path, _ = synthetic_acceptance
    environment = asdict(synthetic_gpu_status)
    environment.pop(field)
    update_validator(path, environment=environment)
    with pytest.raises(ValueError, match="acceptance"):
        smoke.require_acceptance(path, synthetic_gpu_status)


@pytest.mark.parametrize("visible", (None, "", "1", "0,1"))
def test_changed_visible_devices_blocks_before_gpu_detection(
    synthetic_acceptance, monkeypatch, visible
):
    path, _ = synthetic_acceptance
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(smoke, "require_gpu", lambda: pytest.fail("GPU must not be touched"))
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        smoke.benchmark(path)


def test_missing_visible_devices_field_is_rejected(synthetic_acceptance, monkeypatch):
    path, report = synthetic_acceptance
    report.pop("cuda_visible_devices")
    path.write_text(json.dumps(report))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        smoke.require_acceptance(path)


def test_unset_visible_devices_matches_explicit_null(
    synthetic_acceptance, synthetic_gpu_status, monkeypatch
):
    path, report = synthetic_acceptance
    report["cuda_visible_devices"] = None
    path.write_text(json.dumps(report))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert smoke.require_acceptance(path, synthetic_gpu_status) == report


def test_cli_missing_acceptance_creates_no_timing_artifact(tmp_path):
    output = tmp_path / "timing.json"
    assert smoke.main(["--acceptance", str(tmp_path / "missing"), "--output", str(output)]) == 2
    assert not output.exists()


def test_cli_never_overwrites_artifact(tmp_path):
    output = tmp_path / "timing.json"
    output.write_text("original")
    with pytest.raises(SystemExit) as error:
        smoke.main(["--acceptance", "missing", "--output", str(output)])
    assert error.value.code == 2 and output.read_text() == "original"
