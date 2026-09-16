"""Run both real-GPU suites and preserve a complete, non-overwriting evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_hashes(root: Path) -> dict[str, str]:
    paths = [root / "pyproject.toml", root / "requirements-dev.txt"]
    for directory in ("src/quantaforge", "tests", "scripts"):
        paths.extend((root / directory).rglob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def junit_counts(path: Path) -> dict[str, int]:
    """Count actual testcase elements and reject an incomplete/malformed report."""
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    suites = list(root.iter("testsuite"))
    counts = {"tests": len(cases)}
    for plural, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
        counts[plural] = sum(1 for _ in root.iter(tag))
    counts["passed"] = sum(
        not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
        for case in cases
    )
    if not cases or not suites:
        raise ValueError("GPU pytest report contains no executed test cases")
    for key in ("tests", "failures", "errors", "skipped"):
        if sum(int(suite.attrib[key]) for suite in suites) != counts[key]:
            raise ValueError(f"inconsistent GPU pytest report: {key}")
    return counts


def validator_passed(report: dict, exit_code: int | None) -> bool:
    """An exit code alone, or a zero-check PASS, is insufficient evidence."""
    checks = report.get("checks")
    return (
        exit_code == 0
        and report.get("status") == "PASS"
        and report.get("environment", {}).get("usable") is True
        and isinstance(checks, list)
        and len(checks) > 0
        and report.get("checks_passed") == len(checks)
    )


def run_command(
    name: str, command: list[str], output: Path, env: dict[str, str], timeout: int
) -> dict:
    """Keep full child stdout/stderr, including failed compilation tracebacks."""
    record = {"command": command, "stdout": f"{name}.stdout", "stderr": f"{name}.stderr"}
    print(f"Running {name}...", flush=True)
    with (output / record["stdout"]).open("x") as stdout:
        with (output / record["stderr"]).open("x") as stderr:
            try:
                process = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=timeout
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as error:
                record.update(exit_code=None, error=str(error))
                stderr.write(f"{type(error).__name__}: {error}\n")
            else:
                record["exit_code"] = process.returncode
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new evidence directory")
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds per command (default: 900)"
    )
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not (ROOT / "tests/test_gpu_correctness.py").is_file():
        parser.error("run this script from a complete repository checkout/source archive")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output already exists: {output}; choose a new directory")

    env = os.environ.copy()
    # Ensure the recorded source is what both subprocesses execute. Disallow
    # inherited pytest selection/plugin options that could silently shrink a run.
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(name, None)
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": source_hashes(ROOT),
        "commands": {},
        "gpu_tests": None,
        "failure": None,
    }

    def save() -> None:
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def run(name: str, command: list[str]) -> dict:
        result = run_command(name, command, output, env, args.timeout)
        report["commands"][name] = result
        save()
        return result

    save()
    exit_code = 1
    try:
        # Git is useful provenance but a source archive can be validated without it.
        run("git-revision", ["git", "rev-parse", "HEAD"])
        run("git-status", ["git", "status", "--porcelain"])
        driver = run("nvidia-smi", ["nvidia-smi"])
        packages = run(
            "packages",
            [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        )
        validation = run("validator", [sys.executable, "-m", "quantaforge.validate_gpu", "--json"])
        try:
            details = json.loads((output / "validator.stdout").read_text())
        except json.JSONDecodeError as error:
            report["failure"] = f"validator did not produce valid JSON: {error}; inspect its stderr"
        else:
            if validation["exit_code"] == 2 and details.get("status") == "UNAVAILABLE":
                report["status"] = "UNAVAILABLE"
                report["failure"] = details.get("failure", "NVIDIA runtime unavailable")
                exit_code = 2
            elif not validator_passed(details, validation["exit_code"]):
                report["failure"] = "deterministic GPU validation failed or was incomplete"
            else:
                pytest = run(
                    "pytest",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_gpu_correctness.py",
                        "-q",
                        "-ra",
                        "-o",
                        "addopts=",
                        f"--junitxml={output / 'gpu-tests.xml'}",
                    ],
                )
                counts = junit_counts(output / "gpu-tests.xml")
                report["gpu_tests"] = counts
                if pytest["exit_code"] != 0 or counts["passed"] != counts["tests"]:
                    report["failure"] = (
                        "GPU pytest failed or skipped cases; full execution is required"
                    )
                elif driver["exit_code"] != 0 or packages["exit_code"] != 0:
                    report["failure"] = "driver/package metadata capture failed"
                else:
                    report["status"] = "PASS"
                    exit_code = 0
        if source_hashes(ROOT) != report["source_sha256"]:
            report["failure"] = (
                "source changed during validation; results do not describe one snapshot"
            )
            report["status"] = "FAIL"
            exit_code = 1
        if report["status"] == "RUNNING":
            report["status"] = "FAIL"
    except Exception as error:
        report["status"] = "FAIL"
        report["failure"] = f"{type(error).__name__}: {error}"
        save()
        raise
    save()
    print(f"{report['status']}: evidence saved to {output}")
    if report["failure"]:
        print(report["failure"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
