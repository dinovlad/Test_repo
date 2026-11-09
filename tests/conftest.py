"""Provide independent estimator inputs and temporary validation checkouts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import subprocess
import types
import unittest.mock
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
import pytest


if TYPE_CHECKING:
    import pandas as pd

    import labor_cost_optimization.att as att

    Panel = tuple[pd.DataFrame, pd.DataFrame]


ReportResult = tuple[bool, str, list[str]]
Report = Callable[..., ReportResult]


def _make_panel(
    first: pd.Timestamp | None = None,
    second: pd.Timestamp | None = None,
) -> Panel:
    """Construct two dose groups and one clean comparison firm.

    Args:
        first:
            First cohort's termination period, defaulting to January 8, 2024.
        second:
            Second cohort's termination period, defaulting to January 10, 2024.

    Returns:
        Outcome and event tables with different dose supports, a common
        nonlinear calendar shock, and a dose-dependent pre-period contrast.
    """
    import pandas as pd

    first = pd.Timestamp("2024-01-08") if first is None else first
    second = pd.Timestamp("2024-01-10") if second is None else second
    events = pd.DataFrame(
        [(f"A{i}", first, 4.0 + i) for i in range(6)] + [(f"B{i}", second, 5.0 + i) for i in range(7)],
        columns=["firm", "period", "hours"],
    )
    event_lookup = events.set_index("firm")
    records = []
    for index, firm in enumerate([*events["firm"], "C0"]):
        for position, period in enumerate(pd.date_range("2024-01-01", periods=24)):
            sales = 100.0 + 11.0 * index + 3.0 * position + 0.07 * position**2
            if firm in event_lookup.index:
                cohort = event_lookup.loc[firm, "period"]
                dose = float(event_lookup.loc[firm, "hours"])
                event_time = (period - cohort).days
                if event_time == -2:
                    sales += 0.3 * dose + 0.02 * dose**2
                elif event_time >= 0:
                    sales += (event_time + 1) * (2.0 + 0.5 * dose - 0.03 * dose**2 + 0.004 * dose**3)
            records.append((firm, period, sales))
    return pd.DataFrame(records, columns=["firm", "period", "sales"]), events


@pytest.fixture
def panel_factory() -> Callable[[pd.Timestamp, pd.Timestamp], Panel]:
    """Provide a factory for synthetic panels with configurable cohort periods.

    Returns:
        A callable constructing fresh outcome and event tables.
    """
    return _make_panel


@pytest.fixture
def panel() -> Panel:
    """Provide fresh synthetic input tables for each test.

    Returns:
        Independent outcome and event tables with the default cohort periods.
    """
    return _make_panel()


@pytest.fixture
def model(panel: Panel) -> att.ATT:
    """Fit a fresh estimator without sharing mutable state between tests.

    Args:
        panel:
            This test's synthetic outcome and event tables.

    Returns:
        An estimator covering all horizons from minus two through three.

    Raises:
        ValueError:
            If the synthetic inputs fail estimator validation.
        RuntimeError:
            If a required polynomial regression cannot be fitted.
    """
    import labor_cost_optimization.att as att

    outcomes, events = panel
    return att.ATT(firm_col="firm", period_col="period", outcome_col="sales", dose_col="hours").fit(
        outcomes,
        events,
        isolation_window=(-2, 3),
    )


@pytest.fixture
def kernel_events() -> pd.DataFrame:
    """Provide canonical metadata including one deliberately ineligible firm.

    Returns:
        First-event metadata for two cohorts and four firms.
    """
    import pandas as pd

    first = pd.Timestamp("2024-01-08")
    second = pd.Timestamp("2024-01-10")
    return pd.DataFrame(
        [
            ("a", first, 4.0),
            ("b", first, 5.0),
            ("c", second, 9.0),
            ("excluded", first, 9.0),
        ],
        columns=["firm_id", "cohort", "dose"],
    )


@pytest.fixture
def bandwidth_events() -> pd.DataFrame:
    """Include repeated doses within cohorts and across different cohorts.

    Returns:
        Raw first-event rows for eight firms in three cohorts.
    """
    import pandas as pd

    first = pd.Timestamp("2024-01-08")
    second = pd.Timestamp("2024-01-10")
    return pd.DataFrame(
        {
            "firm": list("abcdefgh"),
            "period": [first] * 3 + [second] * 3 + [second + pd.Timedelta(days=3)] * 2,
            "hours": [4.0, 4.0, 6.0, 4.0, 7.0, 9.0, 6.0, 10.0],
        },
    )


@pytest.fixture
def validator() -> types.ModuleType:
    """Load an independent validation module without importing the estimator.

    Returns:
        A fresh module object containing the standalone validation runner.

    Raises:
        RuntimeError:
            If Python cannot construct a loader for the validation script.
        OSError:
            If the validation script cannot be read.
    """
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("local_validation_script", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load validation script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(tmp_path: Path, validator: types.ModuleType) -> Path:
    """Create the files needed for a complete validation run.

    Args:
        tmp_path:
            Directory for the temporary checkout.
        validator:
            This test's validation runner and required-file configuration.

    Returns:
        The checkout root containing the expected Python files and documentation.

    Raises:
        OSError:
            If a fixture directory or source file cannot be created.
    """
    for relative in [*validator.CHECKED_PYTHON_FILES, "README.md", "att.md"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Validation fixture.\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validator: types.ModuleType) -> Report:
    """Create environment reports using isolated project metadata.

    Args:
        tmp_path:
            Directory for temporary project metadata.
        monkeypatch:
            Fixture for replacing metadata lookups and Git commands.
        validator:
            This test's independent validation module.

    Returns:
        A callable reporting a project's success flag, log, and queried names.
    """

    def make_report(
        requirements: list[str],
        versions: dict[str, str],
        *,
        requires_python: str = ">=3.0",
        dev_requirements: list[str] | None = None,
    ) -> ReportResult:
        """Report temporary project requirements and installed distributions.

        Args:
            requirements:
                Runtime dependency declarations to write to project metadata.
            versions:
                Installed distribution versions to return from metadata lookups.
            requires_python:
                Python requirement for the temporary project.
            dev_requirements:
                Optional development dependency declarations.

        Returns:
            The validation result, captured report, and queried distributions.

        Raises:
            OSError:
                If temporary project metadata cannot be written.
        """
        normalized = {name.lower().replace("_", "-"): value for name, value in versions.items()}
        queried: list[str] = []

        def installed(name: str) -> str:
            """Return fixture metadata or the normal missing-distribution error.

            Args:
                name:
                    Distribution name requested by the validation runner.

            Returns:
                The configured installed version of the requested distribution.

            Raises:
                importlib.metadata.PackageNotFoundError:
                    If the fixture does not provide the requested distribution.
            """
            queried.append(name)
            key = name.lower().replace("_", "-")
            if key not in normalized:
                raise importlib.metadata.PackageNotFoundError(name)
            return normalized[key]

        python_requirement = orjson.dumps(requires_python).decode("utf-8")
        runtime_requirements = orjson.dumps(requirements).decode("utf-8")
        config = f"[project]\nrequires-python = {python_requirement}\ndependencies = {runtime_requirements}\n"
        if dev_requirements is not None:
            development_requirements = orjson.dumps(dev_requirements).decode("utf-8")
            config += f"[dependency-groups]\ndev = {development_requirements}\n"
        (tmp_path / "pyproject.toml").write_text(config, encoding="utf-8")
        file = io.StringIO()
        git = subprocess.CompletedProcess([], 0, "fixture commit\n", "")
        with monkeypatch.context() as scoped:
            scoped.setattr(validator, "ROOT", tmp_path)
            scoped.setattr(validator.importlib.metadata, "version", installed)
            scoped.setattr(validator.subprocess, "run", unittest.mock.Mock(return_value=git))
            result = validator.environment_report(file)
        return result, file.getvalue(), queried

    return make_report
