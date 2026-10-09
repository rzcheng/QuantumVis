"""measure complete cpu single-gate run() latency, never kernel-only time."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quantumvis import Circuit, CPUSimulator
from quantumvis.gates import SINGLE_QUBIT_GATES, Gate

MAX_QUBITS = 20  # 16 MiB raw complex128 state; execution/checks allocate working copies.
ROTATION_ANGLE = 0.731
ATOL = 1e-12
METRIC = "complete_cpu_single_gate_run_latency"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MEASURED_SOURCES = (
    "benchmarks/benchmark_gates.py",
    "src/quantumvis/__init__.py",
    "src/quantumvis/_validation.py",
    "src/quantumvis/circuit.py",
    "src/quantumvis/gates.py",
    "src/quantumvis/measurement.py",
    "src/quantumvis/state.py",
    "src/quantumvis/cpu/__init__.py",
    "src/quantumvis/cpu/simulator.py",
)


def _integer(value: int, name: str, minimum: int, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be >= {minimum}" + (f" and <= {maximum}" if maximum else ""))


def _command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _metadata() -> dict:
    root = str(Path(__file__).resolve().parents[1])
    source_hashes = {
        name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
        for name in MEASURED_SOURCES
    }
    loaded_sources = {}
    for name in MEASURED_SOURCES:
        if not name.startswith("src/"):
            continue
        module_name = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module_name = module_name.removesuffix(".__init__")
        module = importlib.import_module(module_name)
        loaded_path = Path(module.__file__).resolve()
        loaded_hash = hashlib.sha256(loaded_path.read_bytes()).hexdigest()
        if loaded_hash != source_hashes[name]:
            raise ValueError(
                f"loaded simulator source differs from this checkout: {module_name} "
                f"at {loaded_path}; install this checkout with python -m pip install -e ."
            )
        loaded_sources[module_name] = {"path": str(loaded_path), "sha256": loaded_hash}
    git_status = _command(["git", "-C", root, "status", "--porcelain"])
    cpu = None
    if platform.system() == "Darwin":
        cpu = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    try:
        from threadpoolctl import threadpool_info
    except ModuleNotFoundError as error:
        if error.name != "threadpoolctl":
            raise
        pools = None  # optional introspection; never a benchmark dependency.
    else:
        pools = threadpool_info()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_model": cpu,
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "quantumvis_version": importlib.metadata.version("quantumvis"),
        "numpy_configuration": np.show_config(mode="dicts"),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "threadpools_if_available": pools,
        "git_revision": _command(["git", "-C", root, "rev-parse", "HEAD"]),
        "git_dirty": None if git_status is None else bool(git_status),
        "source_sha256": source_hashes,
        "loaded_simulator_sources": loaded_sources,
        "timer": "time.perf_counter_ns",
        "timer_resolution_seconds": time.get_clock_info("perf_counter").resolution,
    }


def _reference_gate(state: np.ndarray, gate: Gate) -> np.ndarray:
    """independent full-index formulas, without simulator matrices or pair reshaping."""
    indices = np.arange(state.size)
    bit = (indices & (1 << gate.target)) != 0
    partner = state[indices ^ (1 << gate.target)]
    sign = np.where(bit, -1.0, 1.0)
    match gate.name:
        case "X":
            return partner
        case "Y":
            return -1j * sign * partner
        case "Z":
            return sign * state
        case "H":
            return (sign * state + partner) / np.sqrt(2)
        case "S":
            return np.where(bit, 1j, 1) * state
        case "T":
            return np.exp(1j * np.pi / 4 * bit) * state
        case "RX":
            return np.cos(gate.angle / 2) * state - 1j * np.sin(gate.angle / 2) * partner
        case "RY":
            return np.cos(gate.angle / 2) * state - sign * np.sin(gate.angle / 2) * partner
        case "RZ":
            return np.exp(-1j * sign * gate.angle / 2) * state
        case _:
            raise ValueError(f"unsupported benchmark operation: {gate.name}")


def _measure_case(
    simulator: CPUSimulator,
    initial: np.ndarray,
    gate: Gate,
    warmups: int,
    repetitions: int,
) -> dict:
    num_qubits = initial.size.bit_length() - 1
    circuit = Circuit(num_qubits).add(gate)
    expected = _reference_gate(initial, gate)
    actual = simulator.run(circuit, initial_state=initial).amplitudes
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    norm_error = abs(float(np.vdot(actual, actual).real) - 1)
    np.testing.assert_allclose(norm_error, 0, atol=ATOL, rtol=0)
    max_error = float(np.max(np.abs(actual - expected)))
    del expected, actual

    for _ in range(warmups):
        simulator.run(circuit, initial_state=initial)

    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = simulator.run(circuit, initial_state=initial)
        stop = time.perf_counter_ns()
        elapsed.append(stop - start)
        del result  # destruction of a previous result is outside the next timed call.
    return {
        "num_qubits": num_qubits,
        "state_size": initial.size,
        "state_bytes": initial.nbytes,
        "target": gate.target,
        "operation": gate.name,
        "angle_radians": gate.angle,
        "warmups": warmups,
        "repetitions": repetitions,
        "timings_ns": elapsed,
        "median_ns": float(np.median(elapsed)),
        "p95_ns": float(np.percentile(elapsed, 95, method="linear")),
        "min_ns": min(elapsed),
        "max_ns": max(elapsed),
        "precheck": {"max_amplitude_error": max_error, "squared_norm_error": norm_error},
    }


def benchmark_gates(
    *,
    qubits: tuple[int, ...] = (8, 10, 12, 14, 16, 18),
    operations: tuple[str, ...] = ("H", "RX"),
    warmups: int = 5,
    repetitions: int = 31,
    seed: int = 20260914,
) -> dict:
    """validate all configuration before allocation, then measure independent calls."""
    _integer(warmups, "warmups", 1, 10000)
    _integer(repetitions, "repetitions", 3, 10000)
    _integer(seed, "seed", 0)
    if not qubits or not operations:
        raise ValueError("qubits and operations must be nonempty")
    for n in qubits:
        _integer(n, "num_qubits", 1, MAX_QUBITS)
    if len(set(qubits)) != len(qubits) or len(set(operations)) != len(operations):
        raise ValueError("qubits and operations must not contain duplicates")
    for operation in operations:
        if operation not in SINGLE_QUBIT_GATES:
            raise ValueError(f"unsupported benchmark operation: {operation}")

    report = {
        "schema_version": 1,
        "backend": "numpy_cpu",
        "dtype": "complex128",
        "metric": METRIC,
        "units": "nanoseconds",
        "percentile_method": "linear",
        "execution": "warmed repeated independent one-gate runs from unchanged input",
        "includes": [
            "input validation and copy",
            "gate matrix construction",
            "NumPy gate arithmetic and temporaries",
            "result validation and owned result copy",
        ],
        "excludes": [
            "Python startup and imports",
            "state generation and circuit construction",
            "correctness precheck and warmups",
            "previous result destruction",
            "metadata collection and JSON serialization",
        ],
        "seed": seed,
        "state_seed_rule": "numpy.random.default_rng(SeedSequence([seed, num_qubits]))",
        "precheck_tolerance": {"atol": ATOL, "rtol": 0},
        "metadata": _metadata(),
        "cases": [],
    }
    simulator = CPUSimulator()
    for n in qubits:
        rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
        initial = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
        initial /= np.linalg.norm(initial)
        initial.flags.writeable = False
        for operation in operations:
            for target in sorted({0, n - 1}):
                gate = Gate(
                    operation, target, angle=ROTATION_ANGLE if operation.startswith("R") else None
                )
                report["cases"].append(
                    _measure_case(simulator, initial, gate, warmups, repetitions)
                )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18])
    parser.add_argument(
        "--operations", nargs="+", choices=sorted(SINGLE_QUBIT_GATES), default=["H", "RX"]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a new artifact path")
    try:
        report = benchmark_gates(
            qubits=tuple(args.qubits),
            operations=tuple(args.operations),
            warmups=args.warmups,
            repetitions=args.repetitions,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for case in report["cases"]:
        print(
            f"{case['operation']:2s} n={case['num_qubits']:2d} target={case['target']:2d}: "
            f"median={case['median_ns'] / 1e6:.6f} ms "
            f"p95={case['p95_ns'] / 1e6:.6f} ms ({case['repetitions']} trials)"
        )
    print(f"Saved {len(report['cases'])} CPU gate-call cases to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
