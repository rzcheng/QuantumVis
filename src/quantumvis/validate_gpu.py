"""deterministic real-cuda validation; this module does not depend on pytest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from quantumvis import Circuit, CPUSimulator
from quantumvis.gates import Gate
from quantumvis.gpu.runtime import gpu_status

GATE_NAMES = ("X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ")
SEED = 20260914
ATOL = 1e-6
RTOL = 1e-5
FLOAT32_EPS = float(np.finfo(np.float32).eps)
MAX_VALIDATED_DEPTH = 64


class ValidationFailure(AssertionError):
    """a numerical/output contract failure, distinct from runtime/compiler errors."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    depth: int
    max_amplitude_error: float
    relative_l2_error: float
    norm_drift: float


def error_limits(depth: int) -> tuple[float, float, float]:
    """prespecified amplitude-atol, relative-l2 and squared-norm drift limits."""
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 64:
        raise ValueError("validation depth must be an integer between 0 and 64")
    additional_gates = max(0, depth - 1)
    l2 = ATOL + additional_gates * 4 * FLOAT32_EPS
    norm = ATOL + additional_gates * 8 * FLOAT32_EPS
    return l2, l2, norm


def check_output(
    name: str,
    actual: NDArray,
    expected: NDArray,
    *,
    depth: int = 1,
    initial_squared_norm: float = 1.0,
) -> CheckResult:
    """check amplitudes, relative l2 error, and norm in double precision.

    neither input is renormalized. gpu tolerances still need hardware validation.
    """
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    if actual.shape != expected.shape or actual.ndim != 1:
        raise ValidationFailure(f"{name}: output shape {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValidationFailure(f"{name}: nonfinite amplitudes")
    amplitude_atol, l2_limit, norm_limit = error_limits(depth)
    difference = actual - expected
    expected_norm = float(np.linalg.norm(expected))
    if expected_norm == 0:
        raise ValueError("the validation reference must have nonzero norm")
    result = CheckResult(
        name=name,
        depth=depth,
        max_amplitude_error=float(np.max(np.abs(difference))),
        relative_l2_error=float(np.linalg.norm(difference) / expected_norm),
        norm_drift=abs(float(np.vdot(actual, actual).real) - initial_squared_norm),
    )
    if (
        not np.allclose(actual, expected, atol=amplitude_atol, rtol=RTOL)
        or result.relative_l2_error > l2_limit
        or result.norm_drift > norm_limit
    ):
        raise ValidationFailure(
            f"{name}: {json.dumps(asdict(result), sort_keys=True)}; "
            f"limits: amplitude atol={amplitude_atol:g}, rtol={RTOL:g}, "
            f"relative L2<={l2_limit:g}, squared-norm drift<={norm_limit:g}"
        )
    return result


def random_state(num_qubits: int, seed: int) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    return state / np.linalg.norm(state)


def random_circuit(num_qubits: int, depth: int, seed: int) -> Circuit:
    rng = np.random.default_rng(seed)
    circuit = Circuit(num_qubits)
    for _ in range(depth):
        name = GATE_NAMES[int(rng.integers(len(GATE_NAMES)))]
        target = int(rng.integers(num_qubits))
        angle = float(rng.uniform(-2 * np.pi, 2 * np.pi)) if name.startswith("R") else None
        circuit.add(Gate(name, target, angle=angle))
    return circuit


def analytical_cases() -> Iterator[tuple[str, Circuit, NDArray, NDArray]]:
    """expected states come directly from gate identities, not production matrices."""
    cases = (
        ("X", [1, 0], [0, 1]),
        ("Y", [1, 0], [0, 1j]),
        ("Y", [0, 1], [-1j, 0]),
        ("Z", [0, 1], [0, -1]),
        ("H", [1, 0], np.array([1, 1]) / np.sqrt(2)),
        ("H", [0, 1], np.array([1, -1]) / np.sqrt(2)),
        ("S", [0, 1], [0, 1j]),
        ("T", [0, 1], [0, (1 + 1j) / np.sqrt(2)]),
    )
    for index, (name, initial, expected) in enumerate(cases):
        yield (
            f"analytical-{name}-{index}",
            Circuit(1).add(Gate(name, 0)),
            np.asarray(initial, dtype=np.complex128),
            np.asarray(expected, dtype=np.complex128),
        )
    for name, expected in (("RX", [0, -1j]), ("RY", [0, 1]), ("RZ", [-1j, 0])):
        for angle, output in ((np.pi, expected), (2 * np.pi, [-1, 0])):
            yield (
                f"analytical-{name}-{angle:g}",
                Circuit(1).add(Gate(name, 0, angle=angle)),
                np.array([1, 0], dtype=np.complex128),
                np.asarray(output, dtype=np.complex128),
            )
    yield (
        "analytical-H-H",
        Circuit(1).h(0).h(0),
        np.array([1, 0], dtype=np.complex128),
        np.array([1, 0], dtype=np.complex128),
    )


def raw_unitary_check(num_qubits: int, target: int, seed: int, device=None) -> CheckResult:
    """execute the actual kernel against an independent small dense rounded oracle."""
    import torch

    from quantumvis.gpu.kernels.single_qubit import apply_single_qubit

    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    # compare equally rounded operands without the strict cpu norm check.
    matrix = matrix.astype(np.complex64)
    state = random_state(num_qubits, seed + 1).astype(np.complex64)
    dense = np.ones((1, 1), dtype=np.complex128)
    for qubit in reversed(range(num_qubits)):
        dense = np.kron(dense, matrix if qubit == target else np.eye(2))
    expected = dense @ state.astype(np.complex128)
    cuda_device = torch.device("cuda", device) if device is not None else torch.device("cuda")
    real = torch.tensor(state.real.copy(), dtype=torch.float32, device=cuda_device)
    imag = torch.tensor(state.imag.copy(), dtype=torch.float32, device=cuda_device)
    apply_single_qubit(real, imag, matrix, target)
    # blocking cpu transfers synchronize before the reference comparison.
    actual = real.cpu().numpy() + 1j * imag.cpu().numpy()
    rounded = state.astype(np.complex128)
    return check_output(
        f"raw-unitary-n{num_qubits}-t{target}-seed{seed}",
        actual,
        expected,
        initial_squared_norm=float(np.vdot(rounded, rounded).real),
    )


def run_checks(device=None) -> Iterator[CheckResult]:
    """yield completed checks; no simulated interpreter or cpu fallback is used."""
    from quantumvis.gpu import GPUSimulator

    simulator = GPUSimulator(device=device)
    cpu = CPUSimulator()
    for name, circuit, initial, expected in analytical_cases():
        result = simulator.run(circuit, initial_state=initial)
        yield check_output(name, result.amplitudes, expected, depth=len(circuit.operations))
    for num_qubits in (1, 5, 10):
        initial = random_state(num_qubits, SEED + num_qubits)
        for target in sorted({0, num_qubits // 2, num_qubits - 1}):
            for name in GATE_NAMES:
                angle = -0.731 if name.startswith("R") else None
                circuit = Circuit(num_qubits).add(Gate(name, target, angle=angle))
                expected = cpu.run(circuit, initial_state=initial).amplitudes
                result = simulator.run(circuit, initial_state=initial)
                yield check_output(
                    f"random-{name}-n{num_qubits}-t{target}", result.amplitudes, expected
                )
            for name in ("RX", "RY", "RZ"):
                inverse = (
                    Circuit(num_qubits)
                    .add(Gate(name, target, angle=0.713))
                    .add(Gate(name, target, angle=-0.713))
                )
                actual = simulator.run(inverse, initial_state=initial).amplitudes
                yield check_output(
                    f"inverse-{name}-n{num_qubits}-t{target}", actual, initial, depth=2
                )
        for depth in (16, 64):
            circuit = random_circuit(num_qubits, depth, SEED + depth + num_qubits)
            actual = simulator.run(circuit, initial_state=initial).amplitudes
            expected = cpu.run(circuit, initial_state=initial).amplitudes
            yield check_output(f"circuit-n{num_qubits}-depth{depth}", actual, expected, depth=depth)
    for target in range(3):
        yield raw_unitary_check(3, target, SEED + target, device=device)


def _emit_report(report: dict, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    else:
        print(f"{report['status']}: {report['checks_passed']} deterministic real-GPU checks passed")
        if "failure" in report:
            print(report["failure"])
        print("No performance measurements were made.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", type=int, default=None, help="CUDA device index (default: current)"
    )
    parser.add_argument("--json", action="store_true", help="emit one structured JSON report")
    args = parser.parse_args(argv)
    if args.device is not None and args.device < 0:
        parser.error("--device must be nonnegative")
    status = gpu_status(device=args.device)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "UNAVAILABLE" if not status.usable else "RUNNING",
        "environment": asdict(status),
        "numpy_version": np.__version__,
        "dtype": "complex64 (separate float32 real/imag arrays)",
        "cpu_oracle_dtype": "complex128",
        "seed": SEED,
        "tolerances": {
            "status": "prespecified; not calibrated on this hardware",
            "single_gate_atol": ATOL,
            "rtol": RTOL,
            "single_gate_relative_l2": ATOL,
            "single_gate_squared_norm_drift": ATOL,
            "per_additional_gate_atol_and_l2": 4 * FLOAT32_EPS,
            "per_additional_gate_squared_norm_drift": 8 * FLOAT32_EPS,
            "maximum_depth": MAX_VALIDATED_DEPTH,
        },
        "checks_passed": 0,
        "checks": [],
    }
    if not args.json:
        print("ENVIRONMENT " + json.dumps(report["environment"], sort_keys=True), flush=True)
        print(f"Precision: {report['dtype']}; oracle: complex128; seed: {SEED}", flush=True)
    exit_code = 2
    if status.usable:
        try:
            for result in run_checks(device=status.device_index):
                report["checks"].append(asdict(result))
                report["checks_passed"] += 1
        except ValidationFailure as error:
            report["status"] = "FAIL"
            report["failure"] = str(error)
            exit_code = 1
        except Exception as error:
            # save diagnostics and keep the traceback; compiler failures are not skips.
            report["status"] = "FAIL"
            report["failure"] = f"{type(error).__name__}: {error}"
            _emit_report(report, json_output=args.json)
            raise
        else:
            report["status"] = "PASS"
            exit_code = 0
    else:
        report["failure"] = status.reason
    _emit_report(report, json_output=args.json)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
