"""capability-decision tests with dependency stand-ins; no gpu execution is mocked."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from quantumvis.gpu import runtime


@pytest.fixture
def runtime_environment(monkeypatch):
    modules = {
        "torch": SimpleNamespace(
            __version__="2.10.test",
            version=SimpleNamespace(cuda="12.test"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: 0,
                device_count=lambda: 2,
                get_device_name=lambda index: f"test device {index}",
                get_device_capability=lambda index: (8, 0),
            ),
        ),
        "triton": SimpleNamespace(__version__="3.2.test"),
    }
    monkeypatch.setattr(runtime, "find_spec", lambda name: object() if name in modules else None)
    monkeypatch.setattr(runtime, "import_module", lambda name: modules[name])
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    return modules


def test_complete_prerequisites_are_not_a_claim_of_kernel_validation(runtime_environment):
    status = runtime.require_gpu()
    assert status.usable
    assert status.torch_available and status.triton_available and status.cuda_available
    assert status.gpu_name == "test device 0"
    assert status.compute_capability == (8, 0)
    assert status.torch_version == "2.10.test"
    assert status.triton_version == "3.2.test"
    assert "execution unverified" in status.reason
    assert runtime.gpu_available()


@pytest.mark.parametrize("missing", ("torch", "triton"))
def test_missing_dependency_is_reported(runtime_environment, missing):
    del runtime_environment[missing]
    status = runtime.gpu_status()
    assert not status.usable
    assert f"{'PyTorch' if missing == 'torch' else 'Triton'} is not installed" in status.reason
    with pytest.raises(runtime.GPUUnavailableError, match="not installed"):
        runtime.require_gpu()


def test_all_missing_dependencies_are_reported(runtime_environment):
    runtime_environment.clear()
    status = runtime.gpu_status()
    assert not status.torch_available
    assert not status.triton_available
    assert not status.cuda_available
    assert "PyTorch is not installed" in status.reason
    assert "Triton is not installed" in status.reason


def test_cpu_build_reported_even_when_torch_installed(runtime_environment):
    runtime_environment["torch"].version.cuda = None
    runtime_environment["torch"].cuda.is_available = lambda: False
    status = runtime.gpu_status()
    assert status.torch_available and status.triton_available
    assert not status.cuda_available
    assert not status.usable
    assert "no NVIDIA CUDA build" in status.reason
    assert "CUDA is unavailable" in status.reason


def test_cuda_build_without_visible_device_is_reported(runtime_environment):
    runtime_environment["torch"].cuda.is_available = lambda: False
    status = runtime.gpu_status()
    assert not status.usable
    assert status.cuda_version == "12.test"
    assert "CUDA is unavailable" in status.reason


def test_rocm_cuda_namespace_is_not_accepted(runtime_environment):
    runtime_environment["torch"].version.cuda = None
    status = runtime.gpu_status()
    assert status.cuda_available
    assert not status.usable
    assert "ROCm" in status.reason


def test_unsupported_platform_is_reported(runtime_environment, monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    status = runtime.gpu_status()
    assert not status.usable
    assert "host platform is Darwin" in status.reason


def test_old_compute_capability_is_reported(runtime_environment):
    runtime_environment["torch"].cuda.get_device_capability = lambda index: (7, 5)
    assert "compute capability >= 8.0" in runtime.gpu_status().reason
    assert not runtime.gpu_available()


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "y", "on", "yes"))
def test_interpreter_does_not_count_as_hardware_execution(runtime_environment, monkeypatch, value):
    monkeypatch.setenv("TRITON_INTERPRET", value)
    status = runtime.gpu_status()
    assert not status.usable
    assert "TRITON_INTERPRET" in status.reason


def test_selected_device_metadata(runtime_environment):
    status = runtime.gpu_status(1)
    assert status.device_index == 1
    assert status.gpu_name == "test device 1"


def test_in_process_interpreter_override_is_rejected(runtime_environment):
    runtime_environment["triton"].knobs = SimpleNamespace(runtime=SimpleNamespace(interpret=True))
    status = runtime.gpu_status()
    assert not status.usable
    assert "runtime interpreter is enabled" in status.reason


@pytest.mark.parametrize("value", ("0", "false", "off"))
def test_disabled_interpreter_does_not_block_runtime(runtime_environment, monkeypatch, value):
    monkeypatch.setenv("TRITON_INTERPRET", value)
    runtime_environment["triton"].knobs = SimpleNamespace(runtime=SimpleNamespace(interpret=False))
    assert runtime.gpu_status().usable


@pytest.mark.parametrize("device", (True, "cuda:0", 0.5))
def test_device_type_errors_are_explicit(runtime_environment, device):
    with pytest.raises(TypeError, match="device"):
        runtime.gpu_status(device)


@pytest.mark.parametrize("device", (-1, 2))
def test_device_range_errors_are_explicit(runtime_environment, device):
    with pytest.raises(ValueError, match="device"):
        runtime.gpu_status(device)


@pytest.mark.parametrize(
    "error", (ImportError("broken binary"), ModuleNotFoundError("internal dep"))
)
def test_broken_installed_package_is_not_silently_skipped(runtime_environment, monkeypatch, error):
    def broken_import(name):
        raise error

    monkeypatch.setattr(runtime, "import_module", broken_import)
    with pytest.raises(type(error), match=str(error)):
        runtime.gpu_status()


def test_unexpected_driver_error_is_not_silently_skipped(runtime_environment):
    def broken_driver():
        raise RuntimeError("driver query failed")

    runtime_environment["torch"].cuda.is_available = broken_driver
    with pytest.raises(RuntimeError, match="driver query failed"):
        runtime.gpu_status()


def test_cpu_and_gpu_interface_imports_do_not_import_optional_frameworks():
    # hide installed pytorch to check the missing-dependency path.
    code = """
import importlib.abc
import sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton'}:
            raise AssertionError('optional framework imported: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from quantumvis import Circuit, CPUSimulator
from quantumvis.gpu import GPUSimulator, gpu_status
assert CPUSimulator().run(Circuit(1).x(0)).amplitudes[1] == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy()
    )
    assert result.returncode == 0, result.stderr
