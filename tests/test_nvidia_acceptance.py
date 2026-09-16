"""host-only evidence/command tests; synthetic reports are not gpu validation."""

import json
import os
import sys

import pytest

from scripts import validate_nvidia as acceptance


def write_junit(tmp_path, body, *, tests=1, failures=0, errors=0, skipped=0):
    path = tmp_path / "junit.xml"
    path.write_text(
        f'<testsuites><testsuite tests="{tests}" failures="{failures}" '
        f'errors="{errors}" skipped="{skipped}">{body}</testsuite></testsuites>'
    )
    return path


def test_junit_counts_passing_evidence(tmp_path):
    path = write_junit(tmp_path, '<testcase name="one"/><testcase name="two"/>', tests=2)
    assert acceptance.junit_counts(path) == {
        "tests": 2,
        "passed": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


@pytest.mark.parametrize(
    ("tag", "field"), (("skipped", "skipped"), ("failure", "failures"), ("error", "errors"))
)
def test_nonpassing_cases_are_not_counted_as_passes(tmp_path, tag, field):
    path = write_junit(tmp_path, f'<testcase name="case"><{tag}/></testcase>', **{field: 1})
    counts = acceptance.junit_counts(path)
    assert counts["passed"] == 0
    assert counts[field] == 1


def test_junit_rejects_zero_tests(tmp_path):
    with pytest.raises(ValueError, match="no executed test cases"):
        acceptance.junit_counts(write_junit(tmp_path, "", tests=0))


@pytest.mark.parametrize("field", ("tests", "failures", "errors", "skipped"))
def test_junit_rejects_mismatched_counters(tmp_path, field):
    values = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    values[field] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        acceptance.junit_counts(write_junit(tmp_path, '<testcase name="one"/>', **values))


@pytest.fixture
def synthetic_validator_report():
    # only the artifact protocol is tested; no kernel return values are invented.
    return {"status": "PASS", "environment": {"usable": True}, "checks": [{}], "checks_passed": 1}


def test_validator_requires_process_and_report_agreement(synthetic_validator_report):
    assert acceptance.validator_passed(synthetic_validator_report, 0)
    for exit_code in (1, 2, None):
        assert not acceptance.validator_passed(synthetic_validator_report, exit_code)


@pytest.mark.parametrize(
    "replacement",
    (
        {"checks": [], "checks_passed": 0},
        {"checks_passed": 2},
        {"status": "UNAVAILABLE"},
        {"environment": {"usable": False}},
        {"environment": {"usable": "true"}},
        {"checks": None},
    ),
)
def test_validator_rejects_incomplete_artifacts(synthetic_validator_report, replacement):
    assert not acceptance.validator_passed(synthetic_validator_report | replacement, 0)


def test_command_retains_failure_stdout_stderr_and_exit_code(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import sys; print('out'); print('trace',file=sys.stderr); sys.exit(7)",
    ]
    record = acceptance.run_command("failure", command, tmp_path, os.environ.copy(), 30)
    assert record["exit_code"] == 7
    assert (tmp_path / record["stdout"]).read_text().strip() == "out"
    assert (tmp_path / record["stderr"]).read_text().strip() == "trace"
    with pytest.raises(FileExistsError):
        acceptance.run_command("failure", command, tmp_path, os.environ.copy(), 30)


def test_missing_command_is_recorded(tmp_path):
    record = acceptance.run_command("missing", [str(tmp_path / "no-command")], tmp_path, {}, 30)
    assert record["exit_code"] is None
    assert "FileNotFoundError" in (tmp_path / record["stderr"]).read_text()


def test_source_manifest_detects_test_and_source_changes(tmp_path):
    (tmp_path / "src/quantaforge").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    for name in (
        "pyproject.toml",
        "requirements-dev.txt",
        "src/quantaforge/gate.py",
        "tests/test_gate.py",
    ):
        (tmp_path / name).write_text("original")
    first = acceptance.source_hashes(tmp_path)
    (tmp_path / "tests/test_gate.py").write_text("changed")
    second = acceptance.source_hashes(tmp_path)
    assert first != second
    assert first["src/quantaforge/gate.py"] == second["src/quantaforge/gate.py"]


def test_existing_output_is_never_overwritten(tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_text("keep this evidence")
    with pytest.raises(SystemExit) as error:
        acceptance.main(["--output", str(tmp_path)])
    assert error.value.code == 2
    assert sentinel.read_text() == "keep this evidence"


@pytest.mark.parametrize("timeout", ("0", "-1"))
def test_invalid_timeout_does_not_create_output(tmp_path, timeout):
    output = tmp_path / "new"
    with pytest.raises(SystemExit) as error:
        acceptance.main(["--output", str(output), "--timeout", timeout])
    assert error.value.code == 2
    assert not output.exists()


def test_unavailable_run_saves_report_and_never_launches_pytest(tmp_path, monkeypatch):
    def unavailable_command(name, command, output, env, timeout):
        # exercise subprocess failure reporting only, never numerical gpu success.
        assert name != "pytest"
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "PYTEST_ADDOPTS" not in env
        assert "PYTEST_PLUGINS" not in env
        if name == "validator":
            (output / "validator.stdout").write_text(
                json.dumps({"status": "UNAVAILABLE", "failure": "no NVIDIA device"})
            )
        return {"exit_code": 2 if name == "validator" else 0}

    monkeypatch.setenv("PYTEST_ADDOPTS", "-k only_one_test")
    monkeypatch.setenv("PYTEST_PLUGINS", "unwanted_plugin")
    monkeypatch.setattr(acceptance, "run_command", unavailable_command)
    output = tmp_path / "evidence"
    assert acceptance.main(["--output", str(output)]) == 2
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "UNAVAILABLE"
    assert report["gpu_tests"] is None
    assert report["failure"] == "no NVIDIA device"


def test_real_pytest_skip_has_zero_exit_but_is_rejected_as_acceptance(tmp_path):
    # pytest exits zero even when every test skips.
    test = tmp_path / "test_skip.py"
    test.write_text(
        "import pytest\ndef test_requires_hardware():\n    pytest.skip('no hardware')\n"
    )
    junit = tmp_path / "skip.xml"
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(test),
        "-q",
        "-o",
        "addopts=",
        f"--junitxml={junit}",
    ]
    record = acceptance.run_command("skip", command, tmp_path, env, 30)
    assert record["exit_code"] == 0
    counts = acceptance.junit_counts(junit)
    assert counts["passed"] == 0 and counts["skipped"] == counts["tests"] == 1


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_exit"),
    (
        ("complete", "PASS", 0),
        ("skip", "FAIL", 1),
        ("failure", "FAIL", 1),
        ("process_error", "FAIL", 1),
        ("driver_error", "FAIL", 1),
        ("packages_error", "FAIL", 1),
        ("source_changed", "FAIL", 1),
        ("empty_validator", "FAIL", 1),
        ("invalid_validator_json", "FAIL", 1),
    ),
)
def test_evidence_aggregation_requires_every_condition(
    tmp_path, monkeypatch, synthetic_validator_report, scenario, expected_status, expected_exit
):
    # synthetic reports test aggregation, not gpu execution.
    output = tmp_path / "evidence"
    called = []

    def recorded_command(name, command, directory, env, timeout):
        called.append(name)
        code = 0
        if name == "validator":
            details = synthetic_validator_report.copy()
            if scenario == "empty_validator":
                details.update(checks=[], checks_passed=0)
            content = "broken JSON" if scenario == "invalid_validator_json" else json.dumps(details)
            (directory / "validator.stdout").write_text(content)
        elif name == "pytest":
            tag = {"skip": "skipped", "failure": "failure"}.get(scenario)
            kwargs = {"skipped" if tag == "skipped" else "failures": 1} if tag else {}
            body = (
                f'<testcase name="case"><{tag}/></testcase>' if tag else '<testcase name="case"/>'
            )
            path = write_junit(directory, body, **kwargs)
            path.rename(directory / "gpu-tests.xml")
            code = 1 if scenario in {"failure", "process_error"} else 0
        elif name == "nvidia-smi" and scenario == "driver_error":
            code = 1
        elif name == "packages" and scenario == "packages_error":
            code = 1
        return {"exit_code": code}

    monkeypatch.setattr(acceptance, "run_command", recorded_command)
    if scenario == "source_changed":
        snapshots = iter(({"source": "before"}, {"source": "after"}))
        monkeypatch.setattr(acceptance, "source_hashes", lambda root: next(snapshots))
    assert acceptance.main(["--output", str(output)]) == expected_exit
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == expected_status
    if scenario in {"empty_validator", "invalid_validator_json"}:
        assert "pytest" not in called
    if expected_status == "FAIL":
        assert report["failure"]
