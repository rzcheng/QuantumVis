"""export cpu circuit prefixes and copy the historical benchmark unchanged."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quantumvis import Circuit, CPUSimulator

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks/results/cpu-m4-pro-2026-09-14.json"
DEFAULT_OUTPUT = ROOT / "web/public/data"
CPU_MODULES = (
    "quantumvis",
    "quantumvis._validation",
    "quantumvis.circuit",
    "quantumvis.gates",
    "quantumvis.measurement",
    "quantumvis.state",
    "quantumvis.cpu",
    "quantumvis.cpu.simulator",
)


def source_hashes() -> dict[str, str]:
    """verify the imported cpu source matches the checkout before exporting."""
    hashes = {}
    for name in CPU_MODULES:
        module = importlib.import_module(name)
        loaded = Path(module.__file__).read_bytes()
        relative = "src/" + name.replace(".", "/")
        relative += "/__init__.py" if hasattr(module, "__path__") else ".py"
        if loaded != (ROOT / relative).read_bytes():
            raise RuntimeError(
                "imported source differs from checkout; install with pip install -e ."
            )
        hashes[relative] = hashlib.sha256(loaded).hexdigest()
    hashes["scripts/export_web_demo.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def record(circuit: Circuit, angle: float | None = None) -> dict:
    """run every prefix, including the empty circuit, with the cpu backend."""
    simulator = CPUSimulator()
    prefix = Circuit(circuit.num_qubits)
    steps = []
    for step in range(len(circuit.operations) + 1):
        if step:
            prefix.add(circuit.operations[step - 1])
        state = simulator.run(prefix)
        steps.append(
            {
                "real": state.amplitudes.real.tolist(),
                "imag": state.amplitudes.imag.tolist(),
                "probabilities": state.probabilities().tolist(),
            }
        )
    return {
        "angle_radians": angle,
        "operations": [asdict(gate) for gate in circuit.operations],
        "steps": steps,
    }


def build_export() -> dict:
    hashes = source_hashes()
    return {
        "schema_version": 1,
        "metadata": {
            "backend": "numpy_cpu",
            "precision": "complex128",
            "bit_order": "little-endian; q0 is least significant; labels read q[n-1]…q0",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "source_sha256": hashes,
        },
        "circuits": [
            {
                "id": "bell",
                "name": "Bell",
                "num_qubits": 2,
                "recordings": [record(Circuit(2).h(0).cx(0, 1))],
            },
            {
                "id": "ghz",
                "name": "GHZ",
                "num_qubits": 3,
                "recordings": [record(Circuit(3).h(0).cx(0, 1).cx(1, 2))],
            },
            {
                "id": "interference",
                "name": "Phase interference",
                "num_qubits": 1,
                "recordings": [
                    record(Circuit(1).h(0).rz(0, angle).h(0), angle)
                    for angle in (index * np.pi / 16 for index in range(33))
                ],
            },
        ],
    }


def export(output: Path = DEFAULT_OUTPUT) -> None:
    payload = json.dumps(build_export(), indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "circuits.json").write_text(payload, encoding="utf-8")
    shutil.copyfile(BENCHMARK, output / "cpu-baseline.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.output)
    print(f"exported cpu snapshots and unchanged benchmark to {args.output}")


if __name__ == "__main__":
    main()
