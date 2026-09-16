"""host-only timing boundaries and artifact guards; no gpu timings are produced."""

import json

import pytest

from benchmarks import benchmark_gpu_smoke as smoke


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
def synthetic_acceptance(tmp_path):
    # artifact protocol only; these reports are not device validation evidence.
    path = tmp_path / "summary.json"
    report = {
        "status": "PASS",
        "gpu_tests": {"tests": 1, "passed": 1, "failures": 0, "errors": 0, "skipped": 0},
        "source_sha256": smoke.source_hashes(smoke.ROOT),
    }
    path.write_text(json.dumps(report))
    return path, report


def test_acceptance_requires_matching_source(synthetic_acceptance):
    path, report = synthetic_acceptance
    assert smoke.require_acceptance(path) == report
    report["source_sha256"] = {"stale": "source"}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="this source"):
        smoke.require_acceptance(path)


@pytest.mark.parametrize("condition", ("status", "empty", "failed", "skipped", "missing"))
def test_incomplete_acceptance_blocks_before_gpu_detection(
    synthetic_acceptance, monkeypatch, condition
):
    path, report = synthetic_acceptance
    if condition == "status":
        report["status"] = "UNAVAILABLE"
    elif condition == "empty":
        report["gpu_tests"].update(tests=0, passed=0)
    elif condition == "missing":
        report.pop("gpu_tests")
    else:
        report["gpu_tests"]["passed"] = 0
        report["gpu_tests"]["skipped" if condition == "skipped" else "failures"] = 1
    path.write_text(json.dumps(report))
    monkeypatch.setattr(smoke, "require_gpu", lambda: pytest.fail("GPU must not be touched"))
    with pytest.raises(ValueError, match="acceptance"):
        smoke.benchmark(path)


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
