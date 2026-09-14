import numpy as np
import pytest

from quantaforge.gates import Gate


@pytest.mark.parametrize("name", ("", "UNKNOWN", "SWAP", "MEASURE"))
def test_unsupported_gate_name(name):
    with pytest.raises(ValueError):
        Gate(name, 0)


@pytest.mark.parametrize("name", (1, None, True))
def test_gate_name_must_be_string(name):
    with pytest.raises(TypeError):
        Gate(name, 0)


@pytest.mark.parametrize("target", (True, False, 0.5, "0", None))
def test_target_must_be_integer(target):
    with pytest.raises(TypeError):
        Gate("X", target)


def test_target_must_be_nonnegative():
    with pytest.raises(ValueError):
        Gate("X", -1)


@pytest.mark.parametrize("name", ("CX", "CZ"))
def test_controlled_gate_requires_control(name):
    with pytest.raises(ValueError):
        Gate(name, 0)


@pytest.mark.parametrize("control", (True, False, 0.5, "0"))
def test_control_must_be_integer(control):
    with pytest.raises(TypeError):
        Gate("CX", 1, control=control)


def test_control_must_be_nonnegative():
    with pytest.raises(ValueError):
        Gate("CX", 1, control=-1)


@pytest.mark.parametrize("name", ("CX", "CZ"))
def test_control_and_target_must_differ(name):
    with pytest.raises(ValueError):
        Gate(name, 1, control=1)


@pytest.mark.parametrize("name", ("RX", "RY", "RZ"))
def test_rotation_requires_angle(name):
    with pytest.raises(ValueError):
        Gate(name, 0)


@pytest.mark.parametrize("angle", (np.nan, np.inf, -np.inf))
def test_rotation_rejects_nonfinite_angle(angle):
    with pytest.raises(ValueError):
        Gate("RX", 0, angle=angle)


@pytest.mark.parametrize("angle", (True, False, "0.3", 1j, [0.3]))
def test_rotation_rejects_nonreal_angle(angle):
    with pytest.raises(TypeError):
        Gate("RX", 0, angle=angle)


@pytest.mark.parametrize("name", ("X", "H", "S", "T", "CX", "CZ"))
def test_nonrotation_rejects_unused_angle(name):
    control = 1 if name in ("CX", "CZ") else None
    with pytest.raises(ValueError):
        Gate(name, 0, control=control, angle=0.3)


@pytest.mark.parametrize("name", ("X", "H", "RX"))
def test_single_qubit_gate_rejects_unused_control(name):
    angle = 0.3 if name == "RX" else None
    with pytest.raises(ValueError):
        Gate(name, 0, angle=angle, control=1)


def test_gate_is_immutable():
    gate = Gate("X", 0)
    with pytest.raises(AttributeError):
        gate.target = 1
