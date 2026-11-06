"""Check local environment reporting without importing estimator dependencies."""

import importlib.metadata
import importlib.util
import io
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate.py"
SPEC = importlib.util.spec_from_file_location("local_validation_script", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load validation script: {SCRIPT}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
HAS_PARSER = getattr(VALIDATOR, "Requirement", None) is not None
REQUIRES_PARSER = pytest.mark.skipif(
    not HAS_PARSER,
    reason="Optional packaging parser is not installed",
)
ReportResult = tuple[bool, str, list[str]]
Report = Callable[..., ReportResult]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Create the files needed for a complete validation run.

    Args:
        tmp_path:
            Directory for the temporary checkout.

    Returns:
        The checkout root containing the expected Python files and documentation.
    """
    for relative in [*VALIDATOR.CHECKED_PYTHON_FILES, "README.md", "att.md"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Validation fixture.\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Report:
    """Create environment reports using isolated project metadata.

    Args:
        tmp_path:
            Directory for temporary project metadata.
        monkeypatch:
            Fixture for replacing metadata lookups and Git commands.

    Returns:
        A callable that reports a project's success flag, log, and queried names.
    """

    def make_report(
        requirements: list[str],
        versions: dict[str, str],
        *,
        requires_python: str = ">=3.0",
        dev_requirements: list[str] | None = None,
    ) -> ReportResult:
        """Report temporary project requirements and installed distributions."""
        normalized = {name.lower().replace("_", "-"): value for name, value in versions.items()}
        queried: list[str] = []

        def installed(name: str) -> str:
            """Return fixture metadata or the normal missing-distribution error."""
            queried.append(name)
            key = name.lower().replace("_", "-")
            if key not in normalized:
                raise importlib.metadata.PackageNotFoundError(name)
            return normalized[key]

        config = (
            f"[project]\nrequires-python = {json.dumps(requires_python)}\ndependencies = {json.dumps(requirements)}\n"
        )
        if dev_requirements is not None:
            config += f"[dependency-groups]\ndev = {json.dumps(dev_requirements)}\n"
        (tmp_path / "pyproject.toml").write_text(config, encoding="utf-8")
        log = io.StringIO()
        git = subprocess.CompletedProcess([], 0, "fixture commit\n", "")
        with monkeypatch.context() as scoped:
            scoped.setattr(VALIDATOR, "ROOT", tmp_path)
            scoped.setattr(VALIDATOR.importlib.metadata, "version", installed)
            scoped.setattr(VALIDATOR.subprocess, "run", Mock(return_value=git))
            result = VALIDATOR.environment_report(log)
        return result, log.getvalue(), queried

    return make_report


@REQUIRES_PARSER
@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("numpy==1.26.4", True),
        ("numpy>=1.20,<2", True),
        ("numpy", True),
        ("numpy~=1.26.0", True),
        ("numpy==1.26.*", True),
        ("numpy==1.26.3", False),
        ("numpy<1", False),
        ("numpy~=1.25.0", False),
    ],
)
def test_standard_specifiers_and_unversioned_requirements(
    report: Report,
    requirement: str,
    expected: bool,
) -> None:
    """Accept supported installed versions without requiring exact pins."""
    result, log, queried = report([requirement], {"numpy": "1.26.4"})
    assert result is expected, log
    assert "numpy" in queried
    assert "1.26.4" in log


@REQUIRES_PARSER
def test_mixed_project_requirements_continue_through_unpinned_entries(
    report: Report,
) -> None:
    """Reproduce a project combining exact, range, and unversioned entries."""
    result, log, queried = report(
        ["numpy>=1.20,<2", "pandas", "csdid==0.4.5"],
        {"numpy": "1.26.4", "pandas": "1.5.3", "csdid": "0.4.5"},
        dev_requirements=["codespell", "mypy>=1.19,<2"],
    )
    assert result, log
    assert "csdid" in queried
    assert "0.4.5" in log
    assert "mypy" in queried


@REQUIRES_PARSER
def test_extras_use_base_distribution_and_false_markers_skip_lookup(
    report: Report,
) -> None:
    """Evaluate markers and never request metadata for a distribution extra."""
    result, log, queried = report(
        [
            "feature[speed]>=2; python_version >= '3.0'",
            "not-installed>=1; python_version < '3.0'",
        ],
        {"feature": "2.4.0"},
    )
    assert result, log
    assert "feature" in queried
    assert "feature[speed]" not in queried
    assert "not-installed" not in queried


@REQUIRES_PARSER
def test_missing_distribution_does_not_hide_later_packages(report: Report) -> None:
    """Keep checking every runtime entry after a missing distribution."""
    result, log, queried = report(
        ["missing-package>=1", "later==2.0"],
        {"later": "2.0"},
    )
    assert not result, log
    assert "missing-package" in log
    assert "later" in queried
    assert "2.0" in log


@REQUIRES_PARSER
def test_invalid_requirement_is_reported_without_aborting(report: Report) -> None:
    """Malformed declarations must not terminate the diagnostic report."""
    result, log, queried = report(
        ["not a valid [ requirement", "later>=1"],
        {"later": "2.0"},
    )
    assert not result, log
    assert "requirement" in log.lower()
    assert "later" in queried
    assert "2.0" in log


@REQUIRES_PARSER
def test_direct_url_is_unverified_and_does_not_abort_later_checks(
    report: Report,
) -> None:
    """Installed version metadata cannot prove a direct URL's provenance."""
    result, log, queried = report(
        ["remote @ https://example.invalid/remote.whl", "later==2.0"],
        {"remote": "1.2.0", "later": "2.0"},
    )
    assert not result, log
    assert "remote" in log
    assert "UNVERIFIED" in log.upper()
    assert "later" in queried


@REQUIRES_PARSER
def test_python_constraint_comes_from_the_temporary_project(report: Report) -> None:
    """Interpreter matching must use requires-python, not a fixed version."""
    major, minor = sys.version_info[:2]
    matching = f">={major}.{minor},<{major}.{minor + 1}"
    result, log, _ = report([], {}, requires_python=matching)
    assert result, log
    result, log, queried = report(
        ["later>=1"],
        {"later": "2.0"},
        requires_python="<3.0",
    )
    assert not result, log
    assert "<3.0" in log
    assert "later" in queried


@REQUIRES_PARSER
def test_invalid_installed_version_is_reported_without_aborting(report: Report) -> None:
    """Unparsable metadata must not conceal the rest of the environment."""
    result, log, queried = report(
        ["feature>=1", "later==2.0"],
        {"feature": "not-a-version", "later": "2.0"},
    )
    assert not result, log
    assert "not-a-version" in log
    assert "later" in queried


def test_missing_optional_parser_reports_uncertainty_without_crashing(
    report: Report,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain useful installed versions when dependency parsing is unavailable."""
    monkeypatch.setattr(VALIDATOR, "Requirement", None)
    monkeypatch.setattr(VALIDATOR, "SpecifierSet", None)
    result, log, queried = report(
        ["numpy>=1.20,<2", "csdid", "later==2.0"],
        {"numpy": "1.26.4", "csdid": "0.4.5", "later": "2.0"},
    )
    assert not result, log
    assert "UNVERIFIED" in log.upper()
    assert "numpy" in queried
    assert "1.26.4" in log
    assert "later" in queried


@pytest.mark.parametrize(
    "failure",
    [None, ValueError("malformed environment configuration")],
    ids=["mismatch", "exception"],
)
def test_failed_environment_check_still_runs_tests_and_writes_summary(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception | None,
) -> None:
    """A mismatch or reporting exception must leave a complete diagnostic log."""
    for relative in ("README.md", "att.md"):
        (checkout / relative).unlink()
    log_path = checkout / "validation.log"
    monkeypatch.setattr(VALIDATOR, "ROOT", checkout)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate.py", "--tests-only", "--log", str(log_path)],
    )
    with (
        patch.object(
            VALIDATOR,
            "environment_report",
            return_value=False,
            side_effect=failure,
        ),
        patch.object(VALIDATOR, "run_check", return_value=True) as checks,
    ):
        exit_code = VALIDATOR.main()
    log = log_path.read_text(encoding="utf-8")
    commands = {call.args[0]: call.args[1] for call in checks.call_args_list}
    assert exit_code == 1
    assert commands["Numerical and behavioral tests"] == [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-v",
    ]
    assert set(commands) == {
        "Installed package and source",
        "Numerical and behavioral tests",
    }
    assert "Summary" in log
    assert "FAIL" in log
    assert "PASS: Required validation files" in log
    assert "Summary" in capsys.readouterr().out


def test_quality_checks_only_target_estimator_files_and_project_docs(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid scanning unrelated code in a larger project's source directories."""
    log_path = checkout / "validation.log"
    monkeypatch.setattr(VALIDATOR, "ROOT", checkout)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--log", str(log_path)])
    with (
        patch.object(VALIDATOR, "environment_report", return_value=True),
        patch.object(VALIDATOR, "run_check", return_value=True) as checks,
    ):
        exit_code = VALIDATOR.main()
    commands = {call.args[0]: call.args[1] for call in checks.call_args_list}
    files = {
        "src/labor_cost_optimization/att.py",
        "tests/test_att.py",
        "tests/test_validate.py",
        "scripts/validate.py",
    }
    expected = {
        "Ruff formatting": (["ruff", "format", "--check"], files),
        "Ruff lint": (["ruff", "check"], files),
        "Mypy": (["mypy"], files),
        "Spelling": (
            ["codespell_lib", "--ignore-words-list=ure,missings"],
            files | {"README.md", "att.md"},
        ),
    }
    assert exit_code == 0
    assert "PASS: Required validation files" in log_path.read_text(encoding="utf-8")
    for label, (arguments, targets) in expected.items():
        prefix = [sys.executable, "-m", *arguments]
        assert commands[label][: len(prefix)] == prefix
        assert set(commands[label][len(prefix) :]) == targets


def test_missing_validator_tests_fail_without_blocking_remaining_quality_checks(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report incomplete copies while checking the Python files that are present."""
    missing = "tests/test_validate.py"
    (checkout / missing).unlink()
    log_path = checkout / "validation.log"
    monkeypatch.setattr(VALIDATOR, "ROOT", checkout)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--log", str(log_path)])
    with (
        patch.object(VALIDATOR, "environment_report", return_value=True),
        patch.object(VALIDATOR, "run_check", return_value=True) as checks,
    ):
        exit_code = VALIDATOR.main()
    log = log_path.read_text(encoding="utf-8")
    commands = {call.args[0]: call.args[1] for call in checks.call_args_list}
    present = {
        "src/labor_cost_optimization/att.py",
        "tests/test_att.py",
        "scripts/validate.py",
    }
    assert exit_code == 1
    assert "FAIL: Required validation files" in log
    assert missing in log
    assert "Overall result: FAIL" in log
    assert set(commands["Ruff formatting"][5:]) == present
    assert set(commands["Ruff lint"][4:]) == present
    assert set(commands["Mypy"][3:]) == present
    assert set(commands["Spelling"][4:]) == present | {"README.md", "att.md"}


def test_no_expected_python_files_never_runs_unscoped_quality_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty checkout must fail without letting tools scan unrelated files."""
    log_path = tmp_path / "validation.log"
    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--log", str(log_path)])
    with (
        patch.object(VALIDATOR, "environment_report", return_value=True),
        patch.object(VALIDATOR, "run_check", return_value=True) as checks,
    ):
        exit_code = VALIDATOR.main()
    log = log_path.read_text(encoding="utf-8")
    labels = {call.args[0] for call in checks.call_args_list}
    assert exit_code == 1
    assert "FAIL: Required validation files" in log
    assert "Overall result: FAIL" in log
    assert "skip" in log.lower()
    assert labels == {
        "Installed package and source",
        "Numerical and behavioral tests",
    }
