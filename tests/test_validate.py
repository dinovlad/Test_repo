"""Check local environment reporting without importing estimator dependencies."""

import importlib.util
import sys
import types
import unittest.mock
from collections.abc import Callable
from pathlib import Path

import pytest


HAS_PARSER = importlib.util.find_spec("packaging") is not None
REQUIRES_PARSER = pytest.mark.skipif(
    not HAS_PARSER,
    reason="Optional packaging parser is not installed",
)
ReportResult = tuple[bool, str, list[str]]
Report = Callable[..., ReportResult]


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
    """Accept supported installed versions without requiring exact pins.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
        requirement:
            Dependency declaration checked against installed metadata.
        expected:
            Expected validity of the dependency declaration.
    """
    result, log, queried = report([requirement], {"numpy": "1.26.4"})
    assert result is expected, log
    assert "numpy" in queried
    assert "1.26.4" in log


@REQUIRES_PARSER
def test_mixed_project_requirements_continue_through_unpinned_entries(
    report: Report,
) -> None:
    """Reproduce a project combining exact, range, and unversioned entries.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
    """
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
    """Evaluate markers and never request metadata for a distribution extra.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
    """
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
    """Keep checking every runtime entry after a missing distribution.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
    """
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
    """Continue the diagnostic report after malformed declarations.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
    """
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
    """Report direct-reference provenance as unverified from version metadata.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
    """
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
    """Use `requires-python` when checking the current interpreter.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
    """
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
    """Continue checking the environment after unparsable version metadata.

    Args:
        report:
            Fixture producing isolated dependency-validation reports.
    """
    result, log, queried = report(
        ["feature>=1", "later==2.0"],
        {"feature": "not-a-version", "later": "2.0"},
    )
    assert not result, log
    assert "not-a-version" in log
    assert "later" in queried


def test_missing_optional_parser_reports_uncertainty_without_crashing(
    validator: types.ModuleType,
    report: Report,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain useful installed versions when dependency parsing is unavailable.

    Args:
        validator:
            Freshly loaded standalone validation module.
        report:
            Fixture producing isolated dependency-validation reports.
        monkeypatch:
            Fixture restoring temporary attribute and argument replacements.
    """
    monkeypatch.setattr(validator, "Requirement", None)
    monkeypatch.setattr(validator, "SpecifierSet", None)
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
    validator: types.ModuleType,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception | None,
) -> None:
    """Write a complete diagnostic log after a mismatch or reporting exception.

    Args:
        validator:
            Freshly loaded standalone validation module.
        checkout:
            Temporary checkout containing the required validation files.
        monkeypatch:
            Fixture restoring temporary attribute and argument replacements.
        capsys:
            Fixture capturing command-line output.
        failure:
            Backend failure or reporting error to simulate.
    """
    for relative in ("README.md", "att.md"):
        (checkout / relative).unlink()
    log_path = checkout / "validation.log"
    monkeypatch.setattr(validator, "ROOT", checkout)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate.py", "--tests-only", "--log", str(log_path)],
    )
    with (
        unittest.mock.patch.object(
            validator,
            "environment_report",
            return_value=False,
            side_effect=failure,
        ),
        unittest.mock.patch.object(validator, "run_check", return_value=True) as checks,
    ):
        exit_code = validator.main()
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
    validator: types.ModuleType,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid scanning unrelated code in a larger project's source directories.

    Args:
        validator:
            Freshly loaded standalone validation module.
        checkout:
            Temporary checkout containing the required validation files.
        monkeypatch:
            Fixture restoring temporary attribute and argument replacements.
    """
    log_path = checkout / "validation.log"
    monkeypatch.setattr(validator, "ROOT", checkout)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--log", str(log_path)])
    with (
        unittest.mock.patch.object(validator, "environment_report", return_value=True),
        unittest.mock.patch.object(validator, "run_check", return_value=True) as checks,
    ):
        exit_code = validator.main()
    commands = {call.args[0]: call.args[1] for call in checks.call_args_list}
    files = {
        "src/labor_cost_optimization/att.py",
        "tests/conftest.py",
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
    validator: types.ModuleType,
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report incomplete copies while checking the Python files that are present.

    Args:
        validator:
            Freshly loaded standalone validation module.
        checkout:
            Temporary checkout containing the required validation files.
        monkeypatch:
            Fixture restoring temporary attribute and argument replacements.
    """
    missing = "tests/test_validate.py"
    (checkout / missing).unlink()
    log_path = checkout / "validation.log"
    monkeypatch.setattr(validator, "ROOT", checkout)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--log", str(log_path)])
    with (
        unittest.mock.patch.object(validator, "environment_report", return_value=True),
        unittest.mock.patch.object(validator, "run_check", return_value=True) as checks,
    ):
        exit_code = validator.main()
    log = log_path.read_text(encoding="utf-8")
    commands = {call.args[0]: call.args[1] for call in checks.call_args_list}
    present = {
        "src/labor_cost_optimization/att.py",
        "tests/conftest.py",
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
    validator: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail an empty checkout without scanning unrelated files.

    Args:
        validator:
            Freshly loaded standalone validation module.
        tmp_path:
            Temporary directory used as an incomplete project checkout.
        monkeypatch:
            Fixture restoring temporary attribute and argument replacements.
    """
    log_path = tmp_path / "validation.log"
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["validate.py", "--log", str(log_path)])
    with (
        unittest.mock.patch.object(validator, "environment_report", return_value=True),
        unittest.mock.patch.object(validator, "run_check", return_value=True) as checks,
    ):
        exit_code = validator.main()
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
