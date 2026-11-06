"""Run local validation and write a shareable log without installing packages.

Run from an environment containing the project and its development dependencies:
    python scripts/validate.py --log validation.log
"""

# Standalone command-line script, not an importable package.
# ruff: noqa: INP001

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
CHECKED_PYTHON_FILES = [
    "src/labor_cost_optimization/att.py",
    "tests/test_att.py",
    "tests/test_validate.py",
    "scripts/validate.py",
]

# Keep diagnostics runnable when even the development dependencies are missing.
try:
    Requirement = importlib.import_module("packaging.requirements").Requirement
    SpecifierSet = importlib.import_module("packaging.specifiers").SpecifierSet
except ImportError:
    Requirement = None
    SpecifierSet = None


def emit(message: str, log: TextIO) -> None:
    """Write the same message to the terminal and the validation log."""
    sys.stdout.write(message + "\n")
    sys.stdout.flush()
    log.write(message + "\n")
    log.flush()


def requirement_report(
    requirement: object,
    log: TextIO,
    *,
    development: bool = False,
) -> bool:
    """Check a distribution against a PEP 508 dependency requirement.

    Args:
        requirement:
            Requirement from project metadata, with optional extras and markers.
        log:
            Open UTF-8 output log.
        development:
            Whether to label the entry as a development tool.

    Returns:
        Whether an applicable requirement's installed distribution version
        satisfies its constraint, or its marker makes it inapplicable. Extras
        are accepted but their transitive dependencies are not checked here.
        Invalid or unverified requirements return False without stopping checks.
    """
    prefix = "Development tool " if development else ""
    if not isinstance(requirement, str):
        emit(f"{prefix}UNVERIFIED: dependency entry is not a requirement string.", log)
        return False
    if Requirement is None:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        name = match.group(1) if match else "unknown distribution"
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "NOT INSTALLED"
        emit(f"{prefix}{name}: {actual}; UNVERIFIED without packaging.", log)
        return False
    try:
        parsed = Requirement(requirement)
        if parsed.marker is not None and not parsed.marker.evaluate({"extra": ""}):
            emit(f"{prefix}{parsed.name}: SKIP (environment marker is false).", log)
            return True
    except Exception:
        # Parser exceptions can contain private URLs from the original string.
        emit(f"{prefix}INVALID dependency requirement; continuing validation.", log)
        return False
    try:
        actual = importlib.metadata.version(parsed.name)
    except importlib.metadata.PackageNotFoundError:
        actual = "NOT INSTALLED"
    if parsed.url is not None:
        emit(
            f"{prefix}{parsed.name}: {actual}; direct reference; provenance UNVERIFIED.",
            log,
        )
        return False
    expected = str(parsed.specifier) or "any installed version"
    if actual == "NOT INSTALLED":
        ok = False
    else:
        try:
            ok = not parsed.specifier or parsed.specifier.contains(actual)
        except Exception:
            emit(f"{prefix}{parsed.name}: {actual}; INVALID installed version.", log)
            return False
    emit(
        f"{prefix}{parsed.name}: {actual}; required {expected}; {'OK' if ok else 'MISMATCH'}",
        log,
    )
    return bool(ok)


def environment_report(log: TextIO) -> bool:
    """Report whether the configured runtime requirements are satisfied.

    Args:
        log:
            Open UTF-8 output log.

    Returns:
        Whether Python and applicable runtime distributions satisfy the project
        metadata. Development tools are reported separately. The optional
        packaging library interprets ranges, markers, extras, and version rules;
        without it, verification is marked incomplete and other checks continue.
    """
    emit(f"Python: {sys.version.replace(chr(10), ' ')}", log)
    emit(f"Interpreter: {sys.executable}", log)
    emit(f"Platform: {platform.platform()}", log)
    emit(f"Repository: {ROOT}", log)
    try:
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        emit(f"UNVERIFIED: cannot read project metadata ({type(exc).__name__}).", log)
        return False
    project = config.get("project", {})
    required_python = project.get("requires-python")
    matches = False
    if SpecifierSet is None:
        emit("Python requirement UNVERIFIED: packaging is not installed.", log)
    elif not isinstance(required_python, str):
        emit(
            "Python requirement UNVERIFIED: requires-python is missing or invalid.",
            log,
        )
    else:
        try:
            specifier = SpecifierSet(required_python)
            matches = specifier.contains(platform.python_version())
            emit(
                f"Required Python: {specifier}; {'OK' if matches else 'MISMATCH'}",
                log,
            )
        except Exception:
            emit("INVALID requires-python constraint; continuing validation.", log)
    for requirement in project.get("dependencies", []):
        ok = requirement_report(requirement, log)
        matches = matches and ok
    for requirement in config.get("dependency-groups", {}).get("dev", []):
        requirement_report(requirement, log, development=True)
    for args in (["rev-parse", "HEAD"], ["status", "--short", "--branch"]):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            emit(
                f"git {' '.join(args)}:\n{result.stdout or result.stderr}".rstrip(),
                log,
            )
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
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
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
        f"{'PASS' if exit_code == 0 else 'FAIL'}: {label}; exit={exit_code}; elapsed={time.monotonic() - started:.2f}s",
        log,
    )
    return exit_code == 0


def main() -> int:
    """Run environment, numerical, and optional code-quality checks.

    Returns:
        Zero only when every requested check passes and runtime requirements match.
        Nonzero results still leave a complete diagnostic log to share.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "validation.log")
    parser.add_argument(
        "--tests-only",
        action="store_true",
        help=("Skip formatting, lint, type, and spelling checks; still verify environment and imports."),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Maximum seconds per check (default: 600).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    log_path = args.log.expanduser().resolve()
    if log_path.suffix.lower() != ".log":
        parser.error(
            "--log must have a .log suffix to protect source and configuration files",
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        emit("Labor cost optimization validation", log)
        emit("Each command's output is printed after it finishes or times out.", log)
        emit("No packages are installed and source files are not reformatted.", log)
        try:
            environment_ok = environment_report(log)
        except Exception:
            emit("Environment reporting failed:\n" + traceback.format_exc(), log)
            environment_ok = False
        results = [("Runtime requirements", environment_ok)]
        required_docs = [] if args.tests_only else ["README.md", "att.md"]
        sources = [path for path in CHECKED_PYTHON_FILES if (ROOT / path).is_file()]
        documents = [path for path in required_docs if (ROOT / path).is_file()]
        missing = [path for path in [*CHECKED_PYTHON_FILES, *required_docs] if not (ROOT / path).is_file()]
        results.append(("Required validation files", not missing))
        if missing:
            emit("Missing required validation files:\n" + "\n".join(missing), log)
            emit(
                "Copy the listed files from cgbs-att into this checkout. "
                "Checks on available files will continue; overall validation will fail.",
                log,
            )
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
            (
                "Installed package and source",
                [
                    sys.executable,
                    "-c",
                    source_check,
                    str(ROOT / "src" / "labor_cost_optimization" / "att.py"),
                ],
            ),
            (
                "Numerical and behavioral tests",
                [sys.executable, "-m", "pytest", "tests", "-v"],
            ),
        ]
        if not args.tests_only and sources:
            checks.extend(
                [
                    (
                        "Ruff formatting",
                        [
                            sys.executable,
                            "-m",
                            "ruff",
                            "format",
                            "--check",
                            *sources,
                        ],
                    ),
                    ("Ruff lint", [sys.executable, "-m", "ruff", "check", *sources]),
                    ("Mypy", [sys.executable, "-m", "mypy", *sources]),
                    (
                        "Spelling",
                        [
                            sys.executable,
                            "-m",
                            "codespell_lib",
                            "--ignore-words-list=ure,missings",
                            *sources,
                            *documents,
                        ],
                    ),
                ],
            )
        elif not args.tests_only:
            emit(
                "Quality checks skipped: no expected Python files are present.",
                log,
            )
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
