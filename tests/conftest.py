"""Provide independent synthetic inputs for estimator and bootstrap tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    import pandas as pd

    import labor_cost_optimization.att as att

    Panel = tuple[pd.DataFrame, pd.DataFrame]


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
def bootstrap_panel(panel: Panel) -> Panel:
    """Provide heterogeneous curves, shared controls, and an unused sampled firm.

    Args:
        panel:
            Fresh synthetic input tables for two cohorts.

    Returns:
        Outcome and event tables with a horizon-specific treated sample and
        three shared controls. The final firm has no usable comparison rows.
    """
    import numpy as np
    import pandas as pd

    outcomes, events = panel
    outcomes = outcomes.copy()
    origin = pd.Timestamp("2024-01-01")
    positions = (outcomes["period"] - origin).dt.days.to_numpy()
    firm_codes = pd.factorize(outcomes["firm"])[0]
    outcomes["sales"] += 0.13 * ((firm_codes * 7 + positions * 3) % 11)
    outcomes["sales"] += 0.017 * (firm_codes % 3) * positions**2
    second = pd.Timestamp("2024-01-10")
    second_post = outcomes["firm"].str.startswith("B") & outcomes["period"].ge(second)
    second_doses = outcomes.loc[second_post, "firm"].map(events.set_index("firm")["hours"])
    second_horizons = (outcomes.loc[second_post, "period"] - second).dt.days
    outcomes.loc[second_post, "sales"] += 0.25 * second_doses * (second_horizons + 1)
    rows = []
    for position, period in enumerate(pd.date_range(origin, periods=24)):
        rows.append(("C1", period, 250.0 + 2.2 * position + 0.015 * position**2 + 0.11 * np.sin(position)))
        rows.append(("C2", period, 400.0 + 4.1 * position + 0.11 * position**2 - 0.07 * np.cos(position)))
    rows.append(("unused", origin, 70.0))
    outcomes = pd.concat([outcomes, pd.DataFrame(rows, columns=outcomes.columns)], ignore_index=True)
    missing = outcomes["firm"].eq("A5") & outcomes["period"].eq(pd.Timestamp("2024-01-09"))
    return outcomes.loc[~missing].reset_index(drop=True), events.copy()


@pytest.fixture
def bootstrap_model(bootstrap_panel: Panel) -> att.ATT:
    """Fit the heterogeneous bootstrap fixture without sharing estimator state.

    Args:
        bootstrap_panel:
            This test's outcome panel and complete event table.

    Returns:
        A cubic estimator with two cohorts and horizons minus one through two.

    Raises:
        ValueError:
            If the synthetic panel violates input or comparison requirements.
        RuntimeError:
            If a required polynomial regression cannot be fitted.
    """
    import labor_cost_optimization.att as att

    outcomes, events = bootstrap_panel
    return att.ATT(firm_col="firm", period_col="period", outcome_col="sales", dose_col="hours").fit(
        outcomes,
        events,
        isolation_window=(-1, 2),
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
