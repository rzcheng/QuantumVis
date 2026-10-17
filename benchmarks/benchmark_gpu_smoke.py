"""small post-acceptance gpu timing smoke; complete gate calls, no speedup claims."""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from xml.etree.ElementTree import ParseError

import numpy as np

from benchmarks.benchmark_gates import _metadata
from quantumvis import Circuit, CPUSimulator
from quantumvis.gpu import GPUSimulator, GPUUnavailableError
from quantumvis.gpu.runtime import GPUStatus, require_gpu
from quantumvis.validate_gpu import SEED, check_output, random_state
from scripts.validate_nvidia import (
    EXPECTED_GPU_TESTS,
    ROOT,
    junit_counts,
    source_hashes,
    validator_passed,
)


def _read_acceptance(path: Path) -> dict:
    try:
        report = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid NVIDIA acceptance artifact {path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError(f"NVIDIA acceptance artifact must be a JSON object: {path}")
    return report


def require_acceptance(path: Path, status: GPUStatus | None = None) -> dict:
    """require complete acceptance for this source and available runtime metadata."""
    report = _read_acceptance(path)
    expected = {
        "tests": EXPECTED_GPU_TESTS,
        "passed": EXPECTED_GPU_TESTS,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    counts = report.get("gpu_tests")
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("status") != "PASS"
        or not isinstance(counts, dict)
        or any(
            type(counts.get(key)) is not int or counts[key] != value
            for key, value in expected.items()
        )
        or report.get("source_sha256") != source_hashes(ROOT)
    ):
        raise ValueError(
            "a complete, zero-skip NVIDIA acceptance report for this source is required"
        )
    junit = path.with_name("gpu-tests.xml")
    try:
        recorded_counts = junit_counts(junit)
    except (OSError, ParseError, KeyError, ValueError) as error:
        raise ValueError(f"invalid NVIDIA acceptance artifact {junit}: {error}") from error
    if recorded_counts != counts:
        raise ValueError("NVIDIA acceptance gpu-tests.xml counts do not match summary.json")
    commands = report.get("commands")
    metadata = report.get("device_metadata")
    if (
        not isinstance(commands, dict)
        or any(
            not isinstance(commands.get(name), dict)
            or type(commands[name].get("exit_code")) is not int
            or commands[name]["exit_code"] != 0
            for name in ("validator", "pytest")
        )
        or not isinstance(metadata, dict)
        or metadata.get("complete") is not True
    ):
        raise ValueError(
            "NVIDIA acceptance requires successful commands and complete device metadata"
        )
    validator = _read_acceptance(path.with_name("validator.stdout"))
    if (
        type(validator.get("schema_version")) is not int
        or validator["schema_version"] != 1
        or not validator_passed(validator, commands["validator"]["exit_code"])
    ):
        raise ValueError("NVIDIA acceptance requires a complete passing validator.stdout")
    if validator.get("numpy_version") != np.__version__:
        raise ValueError("NVIDIA acceptance environment mismatch: numpy_version")
    if "cuda_visible_devices" not in report or report["cuda_visible_devices"] != os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise ValueError("NVIDIA acceptance environment mismatch: CUDA_VISIBLE_DEVICES")
    if status is not None:
        environment = validator["environment"]
        capability = environment.get("compute_capability")
        if not isinstance(capability, list) or any(type(part) is not int for part in capability):
            raise ValueError("NVIDIA acceptance environment mismatch: compute_capability")
        current = asdict(status)
        if status.compute_capability is not None:
            current["compute_capability"] = list(status.compute_capability)
        for field, value in current.items():
            if (
                field not in environment
                or type(environment[field]) is not type(value)
                or environment[field] != value
            ):
                raise ValueError(f"NVIDIA acceptance environment mismatch: {field}")
    return report


def timed_call(operation, synchronize) -> tuple[object, int]:
    synchronize()
    start = time.perf_counter_ns()
    result = operation()
    synchronize()
    elapsed = time.perf_counter_ns() - start
    return result, elapsed


def benchmark(acceptance: Path) -> dict:
    require_acceptance(acceptance)
    status = require_gpu()
    require_acceptance(acceptance, status)
    import torch

    import quantumvis.gpu.kernels.single_qubit as kernel_module
    import quantumvis.gpu.runtime as runtime_module
    import quantumvis.gpu.simulator as simulator_module

    sources = [
        Path(module.__file__) for module in (kernel_module, runtime_module, simulator_module)
    ]
    if any(not path.resolve().is_relative_to(ROOT / "src") for path in sources):
        raise ValueError(
            "GPU benchmark requires this editable checkout; install with pip install -e ."
        )
    sources += [Path(__file__), Path(kernel_module.__file__).with_name("__init__.py")]
    metadata = _metadata()
    metadata["gpu"] = asdict(status)
    metadata["gpu_source_sha256"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    report = {
        "schema_version": 1,
        "metric": "complete_gpu_single_gate_run_latency_with_synchronization",
        "precision": "split float32 / complex64 output; complex128 CPU input",
        "units": "nanoseconds",
        "seed": SEED,
        "metadata": metadata,
        "acceptance_path": str(acceptance.resolve()),
        "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
        "includes": [
            "validation",
            "allocation",
            "upload",
            "gate launch",
            "download",
            "synchronization",
        ],
        "excludes": ["startup", "compilation/cache warmup", "CPU oracle", "state generation"],
        "percentile_method": "linear",
        "cases": [],
    }
    gpu, cpu = GPUSimulator(device=status.device_index), CPUSimulator()

    def synchronize():
        torch.cuda.synchronize(status.device_index)

    for qubits in (8, 12, 16):
        initial = random_state(qubits, SEED + qubits)
        circuit = Circuit(qubits).h(0)
        expected = cpu.run(circuit, initial_state=initial).amplitudes

        def operation(circuit=circuit, initial=initial):
            return gpu.run(circuit, initial_state=initial)

        check_output("benchmark-H-precheck", operation().amplitudes, expected)
        for _ in range(5):
            operation()
        samples = []
        for _ in range(31):
            result, elapsed = timed_call(operation, synchronize)
            samples.append(elapsed)
            check_output("benchmark-H-result", result.amplitudes, expected)
            del result
        report["cases"].append(
            {
                "num_qubits": qubits,
                "state_size": 1 << qubits,
                "device_state_bytes": 8 * (1 << qubits),
                "operation": "H",
                "target": 0,
                "warmups": 5,
                "repetitions": 31,
                "timings_ns": samples,
                "median_ns": float(np.median(samples)),
                "p95_ns": float(np.percentile(samples, 95, method="linear")),
            }
        )
    require_acceptance(acceptance, status)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True, type=Path, help="passing summary.json")
    parser.add_argument("--output", required=True, type=Path, help="new timing artifact")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new artifact path")
    try:
        report = benchmark(args.acceptance)
    except (GPUUnavailableError, ValueError, FileNotFoundError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved GPU timing smoke to {args.output}; no comparative performance claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
