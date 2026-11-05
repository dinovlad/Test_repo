"""Run local validation and write a shareable log without installing packages.

Run from an environment containing the project and its development dependencies:
    python scripts/validate.py --log validation.log
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import tomllib
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]


def emit(message: str, log: TextIO) -> None:
    """Write the same message to the terminal and the validation log."""
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()


def environment_report(log: TextIO) -> bool:
    """Report interpreter and dependency versions without importing the estimator.

    Args:
        log:
            Open UTF-8 output log.

    Returns:
        Whether Python and all pinned runtime packages match the project settings.
        Missing development tools are reported and checked by later commands.
    """
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    emit(f"Python: {sys.version.replace(chr(10), ' ')}", log)
    emit(f"Interpreter: {sys.executable}", log)
    emit(f"Platform: {platform.platform()}", log)
    emit(f"Repository: {ROOT}", log)
    emit(f"Required Python: {config['project']['requires-python']}", log)
    matches = sys.version_info[:2] == (3, 11)
    if not matches:
        emit("MISMATCH: this run does not use the project's Python 3.11 target.", log)
    for requirement in config["project"]["dependencies"]:
        name, expected = requirement.split("==", maxsplit=1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "NOT INSTALLED"
        ok = actual == expected
        matches = matches and ok
        emit(f"{name}: {actual}; required {expected}; {'OK' if ok else 'MISMATCH'}", log)
    for requirement in config["dependency-groups"]["dev"]:
        name = requirement.split(">=", maxsplit=1)[0]
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "NOT INSTALLED"
        emit(f"Development tool {name}: {actual}", log)
    for args in (["rev-parse", "HEAD"], ["status", "--short", "--branch"]):
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10,
            )
            emit(f"git {' '.join(args)}:\n{result.stdout or result.stderr}".rstrip(), log)
        except (OSError, subprocess.TimeoutExpired) as exc:
            emit(f"Git metadata unavailable: {exc}", log)
    return matches


def run_check(label: str, command: list[str], log: TextIO, timeout: float) -> bool:
    """Run one read-only check and retain its output, even when it fails.

    Args:
        label:
            Human-readable check name.
        command:
            Argument list passed directly to subprocess without a shell.
        log:
            Open UTF-8 output log.
        timeout:
            Maximum seconds allowed for this command.

    Returns:
        Whether the command completed with exit status zero. Errors and timeouts
        are recorded as failed checks without preventing the remaining checks.
    """
    emit(f"\n=== {label} ===\n$ {shlex.join(command)}", log)
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        result = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            env=environment,
        )
        output = result.stdout
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or b""
        output = captured.decode("utf-8", errors="replace") if isinstance(captured, bytes) else captured
        output += f"\nTIMEOUT after {timeout:g} seconds."
        exit_code = 124
    except OSError as exc:
        output = f"Unable to launch command: {exc}"
        exit_code = 127
    if output:
        emit(output.rstrip(), log)
    emit(
        f"{'PASS' if exit_code == 0 else 'FAIL'}: {label}; "
        f"exit={exit_code}; elapsed={time.monotonic() - started:.2f}s",
        log,
    )
    return exit_code == 0


def main() -> int:
    """Run environment, numerical, and optional code-quality checks.

    Returns:
        Zero only when every requested check passes in the pinned environment.
        Nonzero results still leave a complete diagnostic log to share.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "validation.log")
    parser.add_argument(
        "--tests-only", action="store_true",
        help="Skip formatting, lint, type, and spelling checks; still verify environment and imports.",
    )
    parser.add_argument(
        "--timeout", type=float, default=600,
        help="Maximum seconds per check (default: 600).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    log_path = args.log.expanduser().resolve()
    if log_path.suffix.lower() != ".log":
        parser.error("--log must have a .log suffix to protect source and configuration files")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        emit("Labor cost optimization validation", log)
        emit("Each command's output is printed after it finishes or times out.", log)
        emit("No packages are installed and source files are not reformatted.", log)
        results = [("Pinned runtime environment", environment_report(log))]
        source_check = (
            "import hashlib; from pathlib import Path; import sys; "
            "import labor_cost_optimization.att as module; "
            "loaded = Path(module.__file__).resolve(); "
            "expected = Path(sys.argv[1]).resolve(); "
            "print('Loaded estimator:', loaded); "
            "sys.exit(0 if hashlib.sha256(loaded.read_bytes()).digest() == "
            "hashlib.sha256(expected.read_bytes()).digest() else "
            "'Installed estimator differs from this checkout; reinstall the project')"
        )
        checks = [
            ("Installed package and source", [
                sys.executable, "-c", source_check,
                str(ROOT / "src" / "labor_cost_optimization" / "att.py"),
            ]),
            ("Numerical and behavioral tests", [
                sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v",
            ]),
        ]
        if not args.tests_only:
            sources = ["src", "tests", "scripts"]
            spelling_files = [
                str(path.relative_to(ROOT))
                for directory in sources
                for path in sorted((ROOT / directory).rglob("*.py"))
            ]
            checks.extend([
                ("Ruff formatting", [
                    sys.executable, "-m", "ruff", "format", "--check", *sources,
                ]),
                ("Ruff lint", [sys.executable, "-m", "ruff", "check", *sources]),
                ("Mypy", [sys.executable, "-m", "mypy", *sources]),
                ("Spelling", [
                    sys.executable, "-m", "codespell", "--ignore-words-list=ure,missings",
                    *spelling_files, "README.md", "att.md",
                ]),
            ])
        for label, command in checks:
            results.append((label, run_check(label, command, log, args.timeout)))
        emit("\n=== Summary ===", log)
        for label, passed in results:
            emit(f"{'PASS' if passed else 'FAIL'}: {label}", log)
        if args.tests_only:
            emit("Quality checks were skipped with --tests-only.", log)
        success = all(passed for _, passed in results)
        emit(f"Overall result: {'PASS' if success else 'FAIL'}", log)
        emit(f"Share this log: {log_path}", log)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
