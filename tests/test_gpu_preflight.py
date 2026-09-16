"""host-only preflight decisions; no numerical gpu success is simulated."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from quantumvis import gpu_preflight as preflight

from .test_validate_gpu import unavailable_status


@pytest.fixture
def prerequisites(monkeypatch, tmp_path):
    paths = {name: tmp_path / name for name in ("model", "l4t_release")}
    monkeypatch.setattr(preflight, "JETSON_FILES", paths)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    status = replace(
        unavailable_status(),
        usable=True,
        reason="prerequisites only",
        cuda_available=True,
        torch_available=True,
        triton_available=True,
        torch_version="2.10.0",
        triton_version="3.6.0",
        cuda_version="12.6",
        device_index=0,
        gpu_name="synthetic metadata",
        compute_capability=(8, 6),
    )
    monkeypatch.setattr(preflight, "gpu_status", lambda: status)
    monkeypatch.setattr(preflight, "import_module", lambda name: None)
    monkeypatch.setattr(preflight, "runtime_version", lambda: (None, "unavailable in host test"))
    return paths


def test_missing_environment_never_imports_kernel_or_launches(monkeypatch, prerequisites):
    monkeypatch.setattr(preflight, "gpu_status", unavailable_status)

    def unexpected(*args):
        raise AssertionError("blocked preflight must not import or launch a kernel")

    monkeypatch.setattr(preflight, "import_module", unexpected)
    monkeypatch.setattr(preflight, "kernel_smoke", unexpected)
    report, code = preflight.preflight(smoke=True)
    assert code == 2 and report["status"] == "BLOCKED"
    assert report["kernel_import"] == "NOT_ATTEMPTED"
    assert report["smoke"] == "NOT_RUN"


def test_ready_without_smoke_does_not_claim_execution(prerequisites, monkeypatch):
    imported = []
    monkeypatch.setattr(preflight, "import_module", imported.append)
    report, code = preflight.preflight()
    assert code == 0 and report["status"] == "READY"
    assert report["minimum_requirements_satisfied"]
    assert report["kernel_import"] == "PASS" and report["smoke"] == "NOT_RUN"
    assert imported == ["quantumvis.gpu.kernels.single_qubit"]
    assert report["cuda_build_version"] == "12.6"
    assert report["cuda_runtime_version"] is None
    assert "no kernel was executed" in report["reason"]


@pytest.mark.parametrize("step", ("gpu_status", "import_module", "kernel_smoke"))
def test_broken_runtime_import_or_launch_is_error_not_blocked(
    prerequisites, monkeypatch, capsys, step
):
    def broken(*args):
        raise RuntimeError("original failure")

    monkeypatch.setattr(preflight, step, broken)
    report, code = preflight.preflight(smoke=True)
    assert code == 1 and report["status"] == "ERROR"
    assert "original failure" in report["reason"]
    assert "RuntimeError: original failure" in capsys.readouterr().err
    assert report["smoke"] != "PASS"


def test_jetson_metadata_is_recorded_without_certifying_hardware(prerequisites):
    prerequisites["model"].write_bytes(b"unknown board\x00")
    prerequisites["l4t_release"].write_text("# test release\n")
    assert preflight.jetson_info() == {"model": "unknown board", "l4t_release": "# test release"}


def test_absent_jetson_files_are_normal(prerequisites):
    assert preflight.jetson_info() == {"model": None, "l4t_release": None}


def test_unreadable_metadata_is_not_silently_hidden(prerequisites):
    prerequisites["model"].mkdir()
    report, code = preflight.preflight()
    assert code == 1 and report["status"] == "ERROR"


@pytest.mark.parametrize("architecture", ("armv7l", "riscv64"))
def test_unrecognized_architecture_is_blocked(prerequisites, monkeypatch, architecture):
    monkeypatch.setattr(preflight.platform, "machine", lambda: architecture)
    report, code = preflight.preflight()
    assert code == 2 and not report["minimum_requirements_satisfied"]


def test_unsupported_python_is_blocked(prerequisites, monkeypatch):
    monkeypatch.setattr(preflight.sys, "version_info", (3, 10))
    assert preflight.preflight()[1] == 2


@pytest.mark.parametrize(
    ("package", "version"),
    (("torch", "2.5.1"), ("torch", "3.0.0"), ("triton", "3.1.0"), ("triton", "4.0.0")),
)
def test_out_of_range_dependencies_are_blocked(prerequisites, monkeypatch, package, version):
    status = replace(preflight.gpu_status(), **{f"{package}_version": version})
    monkeypatch.setattr(preflight, "gpu_status", lambda: status)
    report, code = preflight.preflight()
    assert code == 2 and not report["minimum_requirements_satisfied"]
    assert package in report["reason"]


@pytest.mark.parametrize("json_output", (False, True))
def test_cli_emits_blocked_report(prerequisites, monkeypatch, capsys, json_output):
    monkeypatch.setattr(preflight, "gpu_status", unavailable_status)
    assert preflight.main(["--json"] if json_output else []) == 2
    output = capsys.readouterr().out
    if json_output:
        assert json.loads(output)["status"] == "BLOCKED"
    else:
        assert output.startswith("BLOCKED:")


def test_help_does_not_probe_runtime(monkeypatch):
    monkeypatch.setattr(preflight, "gpu_status", lambda: pytest.fail("unexpected runtime probe"))
    with pytest.raises(SystemExit) as error:
        preflight.main(["--help"])
    assert error.value.code == 0


@pytest.mark.parametrize("missing", ("cuda", "cuda.bindings.runtime"))
def test_absent_optional_runtime_binding_is_explicit(monkeypatch, missing):
    def absent(name):
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(preflight, "import_module", absent)
    value, note = preflight.runtime_version()
    assert value is None and "not installed" in note


def test_runtime_query_uses_library_version_not_build_version(monkeypatch):
    runtime = SimpleNamespace(getLocalRuntimeVersion=lambda: (0, 12060))
    monkeypatch.setattr(preflight, "import_module", lambda name: runtime)
    assert preflight.runtime_version() == ("12.6", "cuda.bindings.runtime.getLocalRuntimeVersion")


def test_runtime_query_failure_remains_an_error(monkeypatch):
    runtime = SimpleNamespace(getLocalRuntimeVersion=lambda: (35, 0))
    monkeypatch.setattr(preflight, "import_module", lambda name: runtime)
    with pytest.raises(RuntimeError, match="query failed"):
        preflight.runtime_version()


def test_old_binding_reports_version_unavailable(monkeypatch):
    monkeypatch.setattr(preflight, "import_module", lambda name: SimpleNamespace())
    assert preflight.runtime_version()[0] is None
