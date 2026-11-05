"""Check local environment reporting without importing estimator dependencies."""

import contextlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate.py"
SPEC = importlib.util.spec_from_file_location("local_validation_script", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load validation script: {SCRIPT}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
HAS_PARSER = getattr(VALIDATOR, "Requirement", None) is not None


class EnvironmentReportTests(unittest.TestCase):
    """Verify requirement handling and continued diagnostic reporting."""

    def report(
        self,
        requirements: list[str],
        versions: dict[str, str],
        *,
        requires_python: str = ">=3.0",
        dev_requirements: list[str] | None = None,
    ) -> tuple[bool, str, list[str]]:
        """Report a temporary project with mocked installed distributions.

        Args:
            requirements:
                Runtime requirement strings written to the temporary project.
            versions:
                Installed distribution names and versions; absent names are missing.
            requires_python:
                Project interpreter constraint, independent of the real repository.
            dev_requirements:
                Optional development requirements; None omits that group entirely.

        Returns:
            The report's success flag, complete log, and queried distribution names.
        """
        normalized = {
            name.lower().replace("_", "-"): value for name, value in versions.items()
        }

        def installed(name: str) -> str:
            """Return fixture metadata or raise the usual missing-distribution error."""
            key = name.lower().replace("_", "-")
            if key not in normalized:
                raise importlib.metadata.PackageNotFoundError(name)
            return normalized[key]

        config = (
            "[project]\n"
            f"requires-python = {json.dumps(requires_python)}\n"
            f"dependencies = {json.dumps(requirements)}\n"
        )
        if dev_requirements is not None:
            config += f"[dependency-groups]\ndev = {json.dumps(dev_requirements)}\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(config, encoding="utf-8")
            log = io.StringIO()
            git = subprocess.CompletedProcess([], 0, "fixture commit\n", "")
            with (
                patch.object(VALIDATOR, "ROOT", root),
                patch.object(
                    VALIDATOR.importlib.metadata, "version", side_effect=installed
                ) as lookup,
                patch.object(VALIDATOR.subprocess, "run", return_value=git),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = VALIDATOR.environment_report(log)
            queried = [call.args[0] for call in lookup.call_args_list]
        return result, log.getvalue(), queried

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_standard_specifiers_and_unversioned_requirements(self):
        """Accept supported installed versions without requiring exact pins."""
        cases = [
            ("numpy==1.26.4", True),
            ("numpy>=1.20,<2", True),
            ("numpy", True),
            ("numpy~=1.26.0", True),
            ("numpy==1.26.*", True),
            ("numpy==1.26.3", False),
            ("numpy<1", False),
            ("numpy~=1.25.0", False),
        ]
        for requirement, expected in cases:
            with self.subTest(requirement=requirement):
                result, log, queried = self.report([requirement], {"numpy": "1.26.4"})
                self.assertEqual(result, expected, log)
                self.assertIn("numpy", queried)
                self.assertIn("1.26.4", log)

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_mixed_project_requirements_continue_through_unpinned_entries(self):
        """Reproduce a project combining exact, range, and unversioned entries."""
        result, log, queried = self.report(
            ["numpy>=1.20,<2", "pandas", "csdid==0.4.5"],
            {"numpy": "1.26.4", "pandas": "1.5.3", "csdid": "0.4.5"},
            dev_requirements=["codespell", "mypy>=1.19,<2"],
        )
        self.assertTrue(result, log)
        self.assertIn("csdid", queried)
        self.assertIn("0.4.5", log)
        self.assertIn("mypy", queried)

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_extras_use_base_distribution_and_false_markers_skip_lookup(self):
        """Evaluate markers and never request metadata for a distribution extra."""
        result, log, queried = self.report(
            [
                "feature[speed]>=2; python_version >= '3.0'",
                "not-installed>=1; python_version < '3.0'",
            ],
            {"feature": "2.4.0"},
        )
        self.assertTrue(result, log)
        self.assertIn("feature", queried)
        self.assertNotIn("feature[speed]", queried)
        self.assertNotIn("not-installed", queried)

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_missing_distribution_does_not_hide_later_packages(self):
        """Keep checking every runtime entry after a missing distribution."""
        result, log, queried = self.report(
            ["missing-package>=1", "later==2.0"], {"later": "2.0"}
        )
        self.assertFalse(result, log)
        self.assertIn("missing-package", log)
        self.assertIn("later", queried)
        self.assertIn("2.0", log)

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_invalid_requirement_is_reported_without_aborting(self):
        """Malformed declarations must not terminate the diagnostic report."""
        result, log, queried = self.report(
            ["not a valid [ requirement", "later>=1"], {"later": "2.0"}
        )
        self.assertFalse(result, log)
        self.assertIn("requirement", log.lower())
        self.assertIn("later", queried)
        self.assertIn("2.0", log)

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_direct_url_is_unverified_and_does_not_abort_later_checks(self):
        """Installed version metadata cannot prove a direct URL's provenance."""
        result, log, queried = self.report(
            ["remote @ https://example.invalid/remote.whl", "later==2.0"],
            {"remote": "1.2.0", "later": "2.0"},
        )
        self.assertFalse(result, log)
        self.assertIn("remote", log)
        self.assertIn("UNVERIFIED", log.upper())
        self.assertIn("later", queried)

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_python_constraint_comes_from_the_temporary_project(self):
        """Interpreter matching must use requires-python, not a fixed version."""
        major, minor = sys.version_info[:2]
        matching = f">={major}.{minor},<{major}.{minor + 1}"
        result, log, _ = self.report([], {}, requires_python=matching)
        self.assertTrue(result, log)
        result, log, queried = self.report(
            ["later>=1"], {"later": "2.0"}, requires_python="<3.0"
        )
        self.assertFalse(result, log)
        self.assertIn("<3.0", log)
        self.assertIn("later", queried)

    @unittest.skipUnless(HAS_PARSER, "Optional packaging parser is not installed")
    def test_invalid_installed_version_is_reported_without_aborting(self):
        """Unparseable metadata must not conceal the rest of the environment."""
        result, log, queried = self.report(
            ["feature>=1", "later==2.0"],
            {"feature": "not-a-version", "later": "2.0"},
        )
        self.assertFalse(result, log)
        self.assertIn("not-a-version", log)
        self.assertIn("later", queried)

    def test_missing_optional_parser_reports_uncertainty_without_crashing(self):
        """Retain useful installed versions when dependency parsing is unavailable."""
        with (
            patch.object(VALIDATOR, "Requirement", None, create=True),
            patch.object(VALIDATOR, "SpecifierSet", None, create=True),
        ):
            result, log, queried = self.report(
                ["numpy>=1.20,<2", "csdid", "later==2.0"],
                {"numpy": "1.26.4", "csdid": "0.4.5", "later": "2.0"},
            )
        self.assertFalse(result, log)
        self.assertIn("UNVERIFIED", log.upper())
        self.assertIn("numpy", queried)
        self.assertIn("1.26.4", log)
        self.assertIn("later", queried)

    def test_failed_environment_check_still_runs_tests_and_writes_summary(self):
        """A mismatch or reporting exception must leave a complete diagnostic log."""
        for failure in (None, ValueError("malformed environment configuration")):
            with self.subTest(failure=failure):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    log_path = root / "validation.log"
                    with (
                        patch.object(VALIDATOR, "ROOT", root),
                        patch.object(
                            sys,
                            "argv",
                            ["validate.py", "--tests-only", "--log", str(log_path)],
                        ),
                        patch.object(
                            VALIDATOR,
                            "environment_report",
                            return_value=False,
                            side_effect=failure,
                        ),
                        patch.object(
                            VALIDATOR, "run_check", return_value=True
                        ) as checks,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        exit_code = VALIDATOR.main()
                    log = log_path.read_text(encoding="utf-8")
                self.assertEqual(exit_code, 1)
                labels = [call.args[0] for call in checks.call_args_list]
                self.assertIn("Numerical and behavioral tests", labels)
                self.assertIn("Summary", log)
                self.assertIn("FAIL", log)


if __name__ == "__main__":
    unittest.main()
