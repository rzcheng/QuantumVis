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
    assert record["error_type"] == "FileNotFoundError"
    assert "FileNotFoundError" in (tmp_path / record["stderr"]).read_text()


@pytest.fixture
def orin_platform(tmp_path, monkeypatch):
    release = tmp_path / "nv_tegra_release"
    model = tmp_path / "model"
    release.write_text("# R36 (release), REVISION: 4.3, GCID: 000, BOARD: generic\n")
    model.write_bytes(b"NVIDIA Jetson Orin Nano Developer Kit\x00")
    monkeypatch.setattr(acceptance, "TEGRA_RELEASE", release)
    monkeypatch.setattr(acceptance, "DEVICE_MODEL", model)
    monkeypatch.setattr(acceptance.platform, "system", lambda: "Linux")
    monkeypatch.setattr(acceptance.platform, "machine", lambda: "aarch64")
    return {
        "usable": True,
        "cuda_available": True,
        "device_index": 0,
        "gpu_name": "Orin",
        "compute_capability": [8, 7],
        "torch_version": "test-build",
        "triton_version": "test-build",
        "cuda_version": "12.6",
    }


def missing_smi():
    return {"exit_code": None, "error_type": "FileNotFoundError"}


@pytest.mark.parametrize(
    "model",
    ("NVIDIA Jetson Orin Nano Developer Kit", "NVIDIA Jetson AGX Orin", "NVIDIA Orin NX"),
)
def test_orin_metadata_retains_platform_and_cuda_details(orin_platform, model):
    acceptance.DEVICE_MODEL.write_bytes(model.encode() + b"\x00")
    result = acceptance.collect_device_metadata(missing_smi(), orin_platform)
    assert result["complete"] is True
    assert result["source"] == "jetson-platform"
    assert result["device_tree_model"] == model
    assert result["nv_tegra_release"].startswith("# R36")
    assert result["cuda_device"] == orin_platform
    assert "not driver/runtime" in result["cuda_version_kind"]


@pytest.mark.parametrize(
    "replacement",
    (
        {"usable": False},
        {"cuda_available": False},
        {"device_index": None},
        {"device_index": -1},
        {"device_index": False},
        {"compute_capability": [7, 2]},
        {"compute_capability": [10, 1]},
        {"compute_capability": None},
        {"gpu_name": None},
        {"torch_version": None},
        {"triton_version": None},
        {"cuda_version": None},
    ),
)
def test_jetson_identity_cannot_replace_cuda_requirements(orin_platform, replacement):
    result = acceptance.collect_device_metadata(missing_smi(), orin_platform | replacement)
    assert result["complete"] is False


@pytest.mark.parametrize("case", ("model", "release", "missing_file", "os", "arch"))
def test_unrecognized_platform_is_not_accepted(orin_platform, monkeypatch, case):
    if case == "model":
        acceptance.DEVICE_MODEL.write_text("NVIDIA Jetson Orin unknown board")
    elif case == "release":
        acceptance.TEGRA_RELEASE.write_text("unrecognized release")
    elif case == "missing_file":
        acceptance.TEGRA_RELEASE.unlink()
    elif case == "os":
        monkeypatch.setattr(acceptance.platform, "system", lambda: "Darwin")
    elif case == "arch":
        monkeypatch.setattr(acceptance.platform, "machine", lambda: "x86_64")
    assert acceptance.collect_device_metadata(missing_smi(), orin_platform)["complete"] is False


@pytest.mark.parametrize(
    "driver", ({"exit_code": 1}, {"exit_code": None, "error_type": "TimeoutExpired"})
)
def test_jetson_does_not_hide_existing_driver_failures(orin_platform, driver):
    assert acceptance.collect_device_metadata(driver, orin_platform)["complete"] is False


def test_source_manifest_detects_test_and_source_changes(tmp_path):
    (tmp_path / "src/quantumvis").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    for name in (
        "pyproject.toml",
        "requirements-dev.txt",
        "src/quantumvis/gate.py",
        "tests/test_gate.py",
    ):
        (tmp_path / name).write_text("original")
    first = acceptance.source_hashes(tmp_path)
    (tmp_path / "tests/test_gate.py").write_text("changed")
    second = acceptance.source_hashes(tmp_path)
    assert first != second
    assert first["src/quantumvis/gate.py"] == second["src/quantumvis/gate.py"]


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
        ("orin_missing_smi", "PASS", 0),
        ("orin_skip", "FAIL", 1),
        ("orin_validation_failed", "FAIL", 1),
        ("orin_packages_error", "FAIL", 1),
    ),
)
def test_evidence_aggregation_requires_every_condition(
    tmp_path,
    monkeypatch,
    synthetic_validator_report,
    orin_platform,
    scenario,
    expected_status,
    expected_exit,
):
    # synthetic reports test aggregation, not gpu execution.
    output = tmp_path / "evidence"
    called = []

    def recorded_command(name, command, directory, env, timeout):
        called.append(name)
        code = 0
        if name == "validator":
            details = synthetic_validator_report.copy()
            if scenario.startswith("orin_"):
                details["environment"] = orin_platform
            if scenario == "orin_validation_failed":
                details["status"] = "FAIL"
                code = 1
            if scenario == "empty_validator":
                details.update(checks=[], checks_passed=0)
            content = "broken JSON" if scenario == "invalid_validator_json" else json.dumps(details)
            (directory / "validator.stdout").write_text(content)
        elif name == "pytest":
            tag = {"skip": "skipped", "orin_skip": "skipped", "failure": "failure"}.get(scenario)
            kwargs = {"skipped" if tag == "skipped" else "failures": 1} if tag else {}
            body = (
                f'<testcase name="case"><{tag}/></testcase>' if tag else '<testcase name="case"/>'
            )
            path = write_junit(directory, body, **kwargs)
            path.rename(directory / "gpu-tests.xml")
            code = 1 if scenario in {"failure", "process_error"} else 0
        elif name == "nvidia-smi" and scenario == "driver_error":
            code = 1
        elif name == "nvidia-smi" and scenario.startswith("orin_"):
            return missing_smi()
        elif name == "packages" and scenario in {"packages_error", "orin_packages_error"}:
            code = 1
        return {"exit_code": code}

    monkeypatch.setattr(acceptance, "run_command", recorded_command)
    if scenario == "source_changed":
        snapshots = iter(({"source": "before"}, {"source": "after"}))
        monkeypatch.setattr(acceptance, "source_hashes", lambda root: next(snapshots))
    assert acceptance.main(["--output", str(output)]) == expected_exit
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == expected_status
    if scenario in {"empty_validator", "invalid_validator_json", "orin_validation_failed"}:
        assert "pytest" not in called
    if expected_status == "FAIL":
        assert report["failure"]
