"""small post-acceptance gpu timing smoke; complete gate calls, no speedup claims."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.benchmark_gates import _metadata
from quantumvis import Circuit, CPUSimulator
from quantumvis.gpu import GPUSimulator, GPUUnavailableError
from quantumvis.gpu.runtime import require_gpu
from quantumvis.validate_gpu import SEED, check_output, random_state
from scripts.validate_nvidia import ROOT, source_hashes


def require_acceptance(path: Path) -> dict:
    """tie the timing run to a complete acceptance report for this source snapshot."""
    report = json.loads(path.read_text())
    counts = report.get("gpu_tests") or {}
    if (
        report.get("status") != "PASS"
        or counts.get("tests", 0) <= 0
        or counts.get("passed") != counts.get("tests")
        or any(counts.get(name) != 0 for name in ("failures", "errors", "skipped"))
        or report.get("source_sha256") != source_hashes(ROOT)
    ):
        raise ValueError(
            "a passing, zero-skip NVIDIA acceptance report for this source is required"
        )
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
    require_acceptance(acceptance)
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
