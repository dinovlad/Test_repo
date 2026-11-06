"""Check continuous-dose estimates against independent numerical comparisons."""

import logging
from collections.abc import Callable
from typing import TypedDict, cast
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pandas.testing import assert_frame_equal
from scipy.interpolate import BSpline

from labor_cost_optimization import att
from labor_cost_optimization.att import ATT, estimate_h0, estimate_h_cv, estimate_pi, evaluate_theta


class ColumnMappings(TypedDict):
    """Name the four required input columns without weakening constructor types."""

    firm_col: str
    period_col: str
    outcome_col: str
    dose_col: str


class EventMappings(TypedDict):
    """Name the three columns accepted by the bandwidth selectors."""

    firm_col: str
    period_col: str
    dose_col: str


MAPPINGS: ColumnMappings = {
    "firm_col": "firm",
    "period_col": "period",
    "outcome_col": "sales",
    "dose_col": "hours",
}
EVENT_MAPPINGS: EventMappings = {
    "firm_col": "firm",
    "period_col": "period",
    "dose_col": "hours",
}
FIRST = pd.Timestamp("2024-01-08")
SECOND = pd.Timestamp("2024-01-10")
WINDOW = (-2, 3)
Panel = tuple[pd.DataFrame, pd.DataFrame]
FloatArray = NDArray[np.float64]


def make_panel(
    first: pd.Timestamp = FIRST,
    second: pd.Timestamp = SECOND,
) -> Panel:
    """Construct two dose groups and one clean comparison firm.

    Args:
        first:
            First cohort's normalized termination period.
        second:
            Second cohort's normalized termination period.

    Returns:
        Outcome and event tables with different dose supports, a common
        nonlinear calendar shock, and a dose-dependent pre-period contrast.
    """
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
def panel() -> Panel:
    """Provide fresh synthetic input tables for each test."""
    return make_panel()


@pytest.fixture
def model(panel: Panel) -> ATT:
    """Fit a fresh estimator without sharing mutable state between tests.

    Args:
        panel:
            This test's synthetic outcome and event tables.

    Returns:
        An estimator covering all horizons of the fixture's isolation window.
    """
    outcomes, events = panel
    return ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)


@pytest.fixture
def kernel_events() -> pd.DataFrame:
    """Provide canonical metadata including one deliberately ineligible firm."""
    return pd.DataFrame(
        [
            ("a", FIRST, 4.0),
            ("b", FIRST, 5.0),
            ("c", SECOND, 9.0),
            ("excluded", FIRST, 9.0),
        ],
        columns=["firm_id", "cohort", "dose"],
    )


def independent_att(
    outcomes: pd.DataFrame,
    events: pd.DataFrame,
    cohort: pd.Timestamp,
    event_time: int,
    treated_firms: tuple[str, ...],
    control_firms: tuple[str, ...],
    doses: FloatArray,
    degree: int = 3,
) -> FloatArray:
    """Compute a no-knot B-spline regression and clean-control contrast.

    Args:
        outcomes:
            Outcome rows using the fixture's input column names.
        events:
            One first-event input row per firm.
        cohort:
            Focal termination period.
        event_time:
            Signed endpoint offset from the cohort.
        treated_firms:
            Firms belonging to the comparison's observed treated sample.
        control_firms:
            Firms belonging to its observed clean comparison sample.
        doses:
            Doses at which to evaluate the independently fitted curve.
        degree:
            Polynomial degree represented by the independent B-spline basis.

    Returns:
        Predicted treated changes minus the arithmetic control change.
    """
    if event_time == -1:
        return np.zeros(len(doses))
    baseline = cohort - pd.Timedelta(days=1)
    endpoint = cohort + pd.Timedelta(days=event_time)
    values = outcomes.pivot(index="firm", columns="period", values="sales")
    changes = values[endpoint] - values[baseline]
    doses_by_firm = events.set_index("firm")["hours"]
    observed_doses = doses_by_firm.loc[list(treated_firms)].to_numpy(dtype=float)
    knot_vector = np.r_[
        np.repeat(observed_doses.min(), degree + 1),
        np.repeat(observed_doses.max(), degree + 1),
    ]
    dimension = len(knot_vector) - degree - 1
    basis = BSpline(knot_vector, np.eye(dimension), degree, extrapolate=True)
    design = np.column_stack(
        [np.ones(len(observed_doses)), basis(observed_doses)[:, 1:]],
    )
    coefficients = np.linalg.lstsq(
        design,
        changes.loc[list(treated_firms)].to_numpy(),
        rcond=None,
    )[0]
    prediction_design = np.column_stack([np.ones(len(doses)), basis(doses)[:, 1:]])
    control_mean = changes.loc[list(control_firms)].mean()
    return np.asarray(prediction_design @ coefficients - control_mean, dtype=float)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_separate_models_match_independent_spline_estimates(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
    degree: int,
) -> None:
    """Preserve ATT and theta while fitting each cell with a narrow formula."""
    outcomes, events = panel
    original_outcomes = outcomes.copy(deep=True)
    original_events = events.copy(deep=True)
    with patch.object(att.pf, "feols", wraps=att.pf.feols) as solver:
        fitted = ATT(**MAPPINGS, degree=degree).fit(outcomes, events, isolation_window=WINDOW)
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert fitted.drop_summary_ == {
        "duplicate_event_rows": 0,
        "duplicate_event_firms": 0,
        "excluded_first_events": {},
        "unavailable_treated_rows": 0,
        "dropped_treated_rows": 0,
        "dropped_cells": {},
        "dropped_cohorts": 0,
    }
    expected_cells = [(cohort, e) for cohort in (FIRST, SECOND) for e in range(-2, 4) if e != -1]
    assert solver.call_count == len(expected_cells)
    assert list(fitted.models_) == expected_cells
    assert len({id(regression) for regression in fitted.models_.values()}) == len(expected_cells)
    for key, call in zip(expected_cells, solver.call_args_list, strict=True):
        assert type(fitted.models_[key]).__name__ == "Feols"
        assert call.args[0] == f"delta_y ~ treated + treated:poly(scaled_dose, degree={degree}, raw=True)"
        assert call.kwargs["lean"]
        assert not call.kwargs["store_data"]
        assert not call.kwargs["copy_data"]
        assert "split" not in call.kwargs
        assert "fsplit" not in call.kwargs
        block = call.kwargs["data"]
        assert block["cohort"].eq(key[0]).all()
        assert block["event_time"].unique().tolist() == [key[1]]
        assert not block["firm_id"].duplicated().any()
        assert len(block) <= 8
        assert not any(column.startswith("effect_") for column in block.columns)
        treated = block["treated"] == 1
        cell_doses = events.set_index("firm")["hours"].loc[block.loc[treated, "firm_id"]].to_numpy(dtype=float)
        expected_scaled = (cell_doses - cell_doses.min()) / (cell_doses.max() - cell_doses.min())
        assert_allclose(block.loc[treated, "scaled_dose"], expected_scaled, atol=0.0, rtol=0.0)
        assert (block.loc[~treated, "scaled_dose"] == 0.0).all()
        assert fitted.bases_[key] == {"degree": degree, "lower": cell_doses.min(), "upper": cell_doses.max()}
    assert fitted.event_times_ == list(range(-2, 4))
    assert fitted.min_cohort_size_ == degree + 2
    assert not hasattr(fitted, "knots_")
    doses = np.array([2.0, 7.0, 14.0])
    results = fitted.predict(doses)
    independent_components = {}
    for key, group in results.groupby(["cohort", "event_time"]):
        cohort, event_time = cast(tuple[pd.Timestamp, int], key)
        cell = fitted.cell_diagnostics_[(cohort, event_time)]
        assert tuple(cell["control_firms"]) == ("C0",)
        assert cell["n_controls"] == 1
        expected = independent_att(
            outcomes,
            events,
            cohort,
            event_time,
            cell["treated_firms"],
            cell["control_firms"],
            group["dose"].to_numpy(dtype=float),
            degree=degree,
        )
        assert_allclose(group["att"], expected, atol=1e-8, rtol=1e-8)
        independent_components[(cohort, event_time)] = expected
    theta = evaluate_theta(fitted, doses, bandwidth=1.3)
    for event_time in range(-2, 4):
        expected_theta = []
        for dose_index, dose in enumerate(doses):
            gaussian = np.exp(-0.5 * ((events["hours"] - dose) / 1.3) ** 2)
            masses = gaussian.groupby(events["period"]).sum()
            weights = masses / masses.sum()
            expected_theta.append(
                sum(
                    weights[cohort] * independent_components[(cohort, event_time)][dose_index]
                    for cohort in (FIRST, SECOND)
                ),
            )
        actual = theta.loc[theta["event_time"] == event_time, "theta"]
        assert_allclose(actual, expected_theta, atol=1e-8, rtol=1e-8)
    assert_frame_equal(outcomes, original_outcomes)
    assert_frame_equal(events, original_events)


@pytest.mark.parametrize("identifier_kind", ["strings", "categorical", "mixed"])
def test_prepared_comparisons_match_direct_row_lookup(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
    identifier_kind: str,
) -> None:
    """Check missing-period and history selection before invoking the solver."""
    outcomes, events = panel
    clean = outcomes.loc[outcomes["firm"] == "C0"].copy()
    clean["firm"] = "C1"
    outcomes = pd.concat([outcomes, clean], ignore_index=True)
    history = pd.DataFrame(
        [("C0", pd.Timestamp("2024-01-02"), 2.0), ("C0", pd.Timestamp("2024-01-09"), 3.0)],
        columns=events.columns,
    )
    events = pd.concat([events, history], ignore_index=True)
    omitted = {
        ("A1", pd.Timestamp("2024-01-07")),
        ("A0", pd.Timestamp("2024-01-09")),
        ("B6", pd.Timestamp("2024-01-11")),
        ("C1", pd.Timestamp("2024-01-10")),
    }
    keep = [
        (row.firm, row.period) not in omitted and row.period != pd.Timestamp("2024-01-06")
        for row in outcomes.itertuples(index=False)
    ]
    outcomes = outcomes.loc[keep].sample(frac=1.0, random_state=17)
    events = events.sample(frac=1.0, random_state=23)
    outcomes.index = np.zeros(len(outcomes), dtype=int)
    events.index = np.zeros(len(events), dtype=int)
    if identifier_kind == "categorical":
        categories = [*outcomes["firm"].unique(), "unused_firm"]
        outcomes["firm"] = pd.Categorical(outcomes["firm"], categories=categories)
        events["firm"] = pd.Categorical(events["firm"], categories=categories)
    elif identifier_kind == "mixed":
        identifiers = {
            firm: index if index % 3 == 0 else ("firm", firm) if index % 3 == 1 else firm
            for index, firm in enumerate(outcomes["firm"].unique())
        }
        outcomes["firm"] = outcomes["firm"].map(identifiers)
        events["firm"] = events["firm"].map(identifiers)

    fitted = ATT(**MAPPINGS)
    with patch.object(fitted, "_fit_model") as solver, caplog.at_level(logging.WARNING):
        fitted.fit(outcomes, events, isolation_window=WINDOW)
    solver.assert_called_once()
    assert solver.call_args is not None
    blocks = solver.call_args.args[0]
    by_cell = {key: (block, design) for key, block, design in blocks}
    failed = {
        (FIRST, -2): "no_observed_treated",
        (FIRST, 1): "small_sample",
        (FIRST, 2): "no_controls",
        (SECOND, 0): "no_controls",
    }
    expected_cells = [
        (cohort, event_time)
        for cohort in (FIRST, SECOND)
        for event_time in range(-2, 4)
        if event_time != -1 and (cohort, event_time) not in failed
    ]
    assert list(by_cell) == expected_cells
    values = {(row.firm, row.period): float(row.sales) for row in outcomes.itertuples(index=False)}
    expected_availability = {}
    for cohort in (FIRST, SECOND):
        baseline = cohort - pd.Timedelta(days=1)
        targets = [row for row in events.itertuples(index=False) if row.period == cohort]
        baseline_firms = [row.firm for row in outcomes.itertuples(index=False) if row.period == baseline]
        blocked = {
            row.firm
            for row in events.itertuples(index=False)
            if cohort + pd.Timedelta(days=WINDOW[0]) <= row.period <= cohort + pd.Timedelta(days=WINDOW[1])
        }
        for event_time in range(-2, 4):
            key = (cohort, event_time)
            endpoint = cohort + pd.Timedelta(days=event_time)
            common = [firm for firm in baseline_firms if (firm, endpoint) in values]
            treated = [row.firm for row in targets if row.firm in common]
            controls = [firm for firm in common if firm not in blocked]
            for row in targets:
                missing_base = (row.firm, baseline) not in values
                missing_end = (row.firm, endpoint) not in values
                expected_availability[(row.firm, cohort, event_time)] = (
                    baseline,
                    endpoint,
                    missing_base,
                    missing_end,
                    not (missing_base or missing_end),
                )
            cell = fitted.cell_diagnostics_[key]
            assert cell["treated_firms"] == tuple(treated)
            assert cell["control_firms"] == tuple(controls)
            assert cell["n_treated"] == len(treated)
            assert cell["n_controls"] == len(controls)
            assert cell["status"] == failed.get(key, "reference" if event_time == -1 else "ready")
            if key in by_cell:
                block, design = by_cell[key]
                firms = [*treated, *controls]
                assert block["firm_id"].tolist() == firms
                expected_changes = [values[(firm, endpoint)] - values[(firm, baseline)] for firm in firms]
                assert_allclose(block["delta_y"], expected_changes, atol=0.0, rtol=0.0)
                assert block["treated"].tolist() == [1] * len(treated) + [0] * len(controls)
                assert design.shape == (len(treated), 4)
    actual_availability = {
        (row.firm_id, row.cohort, row.event_time): (
            row.baseline,
            row.period,
            row.missing_baseline,
            row.missing_endpoint,
            row.available,
        )
        for row in fitted.availability_diagnostics_.itertuples(index=False)
    }
    assert actual_availability == expected_availability
    assert len(caplog.records) == 1
    assert "drop_summary_" in caplog.text
    assert fitted.drop_summary_ == {
        "duplicate_event_rows": 1,
        "duplicate_event_firms": 1,
        "excluded_first_events": {"isolation_outside_panel": 1},
        "unavailable_treated_rows": 13,
        "dropped_treated_rows": 16,
        "dropped_cells": {"no_observed_treated": 1, "small_sample": 1, "no_controls": 2},
        "dropped_cohorts": 2,
    }


def test_placebo_sign_and_reference_normalization(model: ATT) -> None:
    """Recover a nonzero pre-trend and exact unestimated reference zeros."""
    results = model.predict([6.0, 8.0], event_times=[-2, -1, 0])
    placebo = results.loc[results["event_time"] == -2]
    expected = 0.3 * placebo["dose"] + 0.02 * placebo["dose"] ** 2
    assert_allclose(placebo["att"], expected, atol=1e-8)
    assert (placebo["estimate_type"] == "placebo").all()
    reference = results.loc[results["event_time"] == -1]
    assert (reference["att"] == 0.0).all()
    assert (reference["estimate_type"] == "reference").all()
    assert not reference["extrapolated"].any()
    for cohort in (FIRST, SECOND):
        key = (cohort, -1)
        assert key not in model.effect_columns_
        assert key not in model.models_
        assert (model.cell_coefficients_[key] == 0.0).all()
    assert str(results["cohort"].dtype) == "datetime64[ns]"
    assert str(results["period"].dtype) == "datetime64[ns]"


@pytest.mark.parametrize("slope", [0.0, 2.0], ids=["zero", "common_calendar_trend"])
def test_constant_and_zero_changes_have_zero_treatment_effects(
    panel: Panel,
    slope: float,
) -> None:
    """Allow zero changes and calendar changes shared by every firm."""
    outcomes, events = panel
    periods = (outcomes["period"] - outcomes["period"].min()).dt.days
    outcomes["sales"] = 200.0 + slope * periods
    fitted = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
    result = fitted.predict([6.0, 8.0])
    assert_allclose(result["att"], 0.0, atol=1e-10)
    theta = evaluate_theta(fitted, [6.0, 8.0], bandwidth=1.0)
    assert_allclose(theta["theta"], 0.0, atol=1e-10)


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "nonfinite_intercept"])
def test_incomplete_or_nonfinite_solver_coefficients_fail_the_entire_fit(
    panel: Panel,
    model: ATT,
    failure: str,
) -> None:
    """Never accept a backend's missing term or silently reduce a fitted curve."""
    outcomes, events = panel
    first_key, second_key = list(model.models_)[:2]
    coefficients = model.models_[second_key].coef()
    term = model.effect_columns_[second_key][-1]
    if failure == "missing":
        malformed = coefficients.drop(term)
    else:
        malformed = coefficients.copy()
        malformed.loc["Intercept" if failure == "nonfinite_intercept" else term] = np.nan
    backend_model = Mock()
    backend_model.coef.return_value = malformed
    failed = ATT(**MAPPINGS)
    with (
        patch.object(att.pf, "feols", side_effect=[model.models_[first_key], backend_model]) as solver,
        pytest.raises(RuntimeError, match="could not be fitted"),
    ):
        failed.fit(outcomes, events, isolation_window=WINDOW)
    assert solver.call_count == 2
    assert not failed.models_
    assert all(event_time == -1 for _, event_time in failed.cell_coefficients_)
    assert failed.cell_diagnostics_[first_key]["status"] == "model_failed"
    assert failed.cell_diagnostics_[second_key]["status"] == "model_failed"
    with pytest.raises(RuntimeError, match="successfully fitted"):
        failed.predict([7.0])


def test_prediction_grid_cannot_redefine_the_fitted_basis(model: ATT) -> None:
    """Keep predictions unchanged when other requested doses change."""
    saved = {key: basis.copy() for key, basis in model.bases_.items()}
    single = model.predict([7.0])
    expanded = model.predict([5.5, 7.0, 8.5])
    expanded = expanded.loc[expanded["dose"] == 7.0].reset_index(drop=True)
    assert_frame_equal(single, expanded)
    assert model.bases_ == saved
    with pytest.raises(ValueError, match="outside the fitted isolation window"):
        model.predict([7.0], event_times=[4])


def test_aggregation_matches_manual_kernel_weighted_curves(
    panel: Panel,
    model: ATT,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserve cohort size and dose composition at every requested horizon."""
    _, events = panel
    doses = [5.5, 8.5]
    bandwidth = 1.3
    theta = evaluate_theta(
        model,
        doses,
        bandwidth=bandwidth,
        event_times=[-2, -1, 0, 3],
    )
    assert not caplog.records
    components = model.predict(doses, event_times=[-2, -1, 0, 3])
    for row in theta.to_dict("records"):
        eligible = events.copy()
        distances = (eligible["hours"] - row["dose"]) / bandwidth
        eligible["kernel"] = np.exp(-0.5 * distances**2)
        weights = eligible.groupby("period")["kernel"].sum()
        weights = weights / weights.sum()
        subset = components.loc[
            (components["event_time"] == row["event_time"]) & (components["dose"] == row["dose"])
        ].set_index("cohort")
        expected = sum(weights[cohort] * subset.loc[cohort, "att"] for cohort in weights.index)
        assert row["theta"] == pytest.approx(expected, rel=0.0, abs=5e-10)
        assert row["n_cohorts_used"] == 2
        assert row["n_firms_used"] == 13
        assert not row["restricted_target"]
        assert not row["availability_restricted"]


def test_aggregation_summarizes_all_horizons_without_silencing_prediction(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Count dropped comparisons and extrapolated components in one message."""
    outcomes, events = panel
    fitted = ATT(**MAPPINGS, min_cohort_size=7).fit(outcomes, events, isolation_window=WINDOW)
    fit_summary = fitted.drop_summary_.copy()
    caplog.clear()
    doses = [14.0, 2.0, 7.0]
    horizons = [3, -1, 0, -2]
    with caplog.at_level(logging.WARNING):
        theta = evaluate_theta(fitted, doses, bandwidth=1.3, event_times=horizons)
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage().startswith("Aggregation summary:")
    assert caplog.records[0].args == (4, 24, 4, 6)
    assert fitted.drop_summary_ == fit_summary
    assert theta["event_time"].tolist() == [-2] * 3 + [-1] * 3 + [0] * 3 + [3] * 3
    assert theta["dose"].tolist() == [2.0, 7.0, 14.0] * 4
    assert (theta["n_cohorts_eligible"] == 2).all()
    assert (theta["n_cohorts_used"] == 1).all()
    assert (theta["n_firms_design"] == 13).all()
    assert (theta["n_firms_eligible"] == 13).all()
    assert (theta["n_firms_used"] == 7).all()
    assert theta["restricted_target"].all()
    assert not theta["availability_restricted"].any()
    expected_extrapolation = [row.event_time != -1 and row.dose != 7.0 for row in theta.itertuples(index=False)]
    assert theta["extrapolated"].tolist() == expected_extrapolation
    for event_time_key, group in theta.groupby("event_time", sort=False):
        event_time = cast(int, event_time_key)
        cell = fitted.cell_diagnostics_[(SECOND, event_time)]
        expected = independent_att(
            outcomes,
            events,
            SECOND,
            event_time,
            cell["treated_firms"],
            cell["control_firms"],
            group["dose"].to_numpy(dtype=float),
        )
        assert_allclose(group["theta"], expected, atol=1e-8, rtol=1e-8)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        predictions = fitted.predict(doses, event_times=horizons)
    assert len(caplog.records) == 2
    assert "unavailable comparisons" in caplog.records[0].getMessage()
    assert caplog.records[0].args == (4,)
    assert "Extrapolating" in caplog.records[1].getMessage()
    assert caplog.records[1].args == (6,)
    assert_allclose(predictions["att"], theta["theta"], atol=1e-12, rtol=0.0)


def test_reference_aggregation_outside_support_does_not_warn(
    model: ATT,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep reference zeros free of extrapolation flags and warning counts."""
    with caplog.at_level(logging.WARNING):
        theta = evaluate_theta(model, [2.0, 14.0], bandwidth=1.3, event_times=[-1])
    assert not caplog.records
    assert (theta["theta"] == 0.0).all()
    assert (theta["estimate_type"] == "reference").all()
    assert (theta["n_cohorts_used"] == 2).all()
    assert (theta["n_firms_used"] == 13).all()
    assert not theta["extrapolated"].any()
    assert not theta["restricted_target"].any()
    assert not theta["availability_restricted"].any()


@pytest.mark.parametrize("num_knots", [0, 1])
def test_polynomial_api_rejects_internal_knots(num_knots: int) -> None:
    """Reject the removed spline option instead of silently changing its model."""
    constructor: Callable[..., ATT] = ATT
    with pytest.raises(TypeError, match="num_knots"):
        constructor(**MAPPINGS, num_knots=num_knots)


def test_unused_categorical_firms_do_not_create_phantom_terminations(
    panel: Panel,
    model: ATT,
) -> None:
    """Unused categories must neither block clean controls nor add events."""
    outcomes, events = panel
    outcomes = outcomes.copy()
    events = events.copy()
    categories = [*outcomes["firm"].unique(), "unused_firm"]
    outcomes["firm"] = pd.Categorical(outcomes["firm"], categories=categories)
    events["firm"] = pd.Categorical(events["firm"], categories=categories)
    categorical = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
    assert categorical.duplicate_diagnostics_["n_dropped_events"] == 0
    assert categorical.duplicate_diagnostics_["n_affected_firms"] == 0
    assert (categorical.termination_history_["n_terminations"] > 0).all()
    assert set(categorical.termination_history_["firm_id"]) == set(events["firm"])
    assert_frame_equal(
        categorical.predict([6.0, 8.0], event_times=[-2, 0, 3]),
        model.predict([6.0, 8.0], event_times=[-2, 0, 3]),
        check_exact=False,
        atol=1e-8,
        rtol=1e-8,
    )


def test_missing_endpoints_drop_only_the_affected_comparison(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Do not require complete trajectories or leak missing firms into weights."""
    outcomes, events = panel
    omitted = {
        ("A0", pd.Timestamp("2024-01-09")),
        ("A1", pd.Timestamp("2024-01-07")),
        ("B6", pd.Timestamp("2024-01-11")),
    }
    pairs = outcomes[["firm", "period"]].itertuples(index=False, name=None)
    keep = [(firm, period) not in omitted for firm, period in pairs]
    outcomes = outcomes.loc[keep].reset_index(drop=True)
    with caplog.at_level(logging.WARNING):
        fitted = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
    assert len(caplog.records) == 1
    assert "small_sample" in caplog.text
    assert fitted.drop_summary_["unavailable_treated_rows"] == 8
    assert fitted.drop_summary_["dropped_treated_rows"] == 4
    assert fitted.drop_summary_["dropped_cells"] == {"small_sample": 1}
    assert fitted.cell_diagnostics_[(FIRST, 0)]["n_treated"] == 5
    assert fitted.cell_diagnostics_[(FIRST, 1)]["n_treated"] == 4
    assert (FIRST, 1) not in fitted.cell_coefficients_
    assert "A0" in fitted.cell_diagnostics_[(FIRST, 2)]["treated_firms"]
    assert "A1" not in fitted.cell_diagnostics_[(FIRST, 2)]["treated_firms"]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        observed_target = evaluate_theta(fitted, [7.0], bandwidth=1.0, event_times=[0])
    assert not caplog.records
    assert observed_target["availability_restricted"].all()
    assert not observed_target["restricted_target"].any()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        theta = evaluate_theta(fitted, [7.0], bandwidth=1.0, event_times=[1])
    assert caplog.records
    row = theta.iloc[0]
    assert row["n_firms_design"] == 13
    assert row["n_firms_eligible"] == 10
    assert row["n_firms_used"] == 6
    assert row["availability_restricted"]
    assert row["restricted_target"]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        prediction = fitted.predict([7.0], event_times=[1])
    assert "unavailable" in caplog.text
    assert set(prediction["cohort"]) == {SECOND}
    assert row["theta"] == pytest.approx(
        prediction.iloc[0]["att"],
        rel=0.0,
        abs=5e-11,
    )
    cell = fitted.cell_diagnostics_[(SECOND, 1)]
    expected = independent_att(
        outcomes,
        events,
        SECOND,
        1,
        cell["treated_firms"],
        cell["control_firms"],
        np.array([7.0]),
    )
    assert_allclose(prediction["att"], expected, atol=1e-8)


def test_same_period_ties_are_not_promoted_to_later_events(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log dropped records while retaining simultaneous-event multiplicity."""
    outcomes, events = panel
    earlier_input = pd.DataFrame(
        [("A0", FIRST + pd.Timedelta(hours=1), 9.0)],
        columns=events.columns,
    )
    later = pd.DataFrame(
        [
            ("A0", pd.Timestamp("2024-01-18"), 2.0),
            ("B0", pd.Timestamp("2024-01-18"), 3.0),
        ],
        columns=events.columns,
    )
    events = pd.concat([earlier_input, events, later], ignore_index=True)
    with caplog.at_level(logging.WARNING):
        fitted = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
    assert fitted.duplicate_diagnostics_["n_dropped_events"] == 3
    assert fitted.duplicate_diagnostics_["n_affected_firms"] == 2
    assert len(caplog.records) == 1
    assert fitted.drop_summary_["duplicate_event_rows"] == 3
    assert fitted.drop_summary_["duplicate_event_firms"] == 2
    assert fitted.drop_summary_["excluded_first_events"] == {"overlapping_termination": 1}
    assert fitted.drop_summary_["dropped_cells"] == {}
    selected = fitted.events_.set_index("firm_id")
    assert selected.loc["A0", "cohort"] == FIRST
    assert selected.loc["A0", "dose"] == 9.0
    assert not selected.loc["A0", "design_eligible"]
    assert selected.loc["B0", "design_eligible"]
    history = fitted.termination_history_.set_index(["firm_id", "cohort"])
    assert history.loc[("A0", FIRST), "n_terminations"] == 2
    assert ("A0", pd.Timestamp("2024-01-18")) in history.index
    assert "A0" not in fitted.cell_diagnostics_[(FIRST, 0)]["treated_firms"]


def test_later_history_still_blocks_controls_after_first_event_selection(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An excluded first event does not erase a later conflicting termination."""
    outcomes, events = panel
    clean = outcomes.loc[outcomes["firm"] == "C0"].copy()
    clean["firm"] = "C1"
    outcomes = pd.concat([outcomes, clean], ignore_index=True)
    history = pd.DataFrame(
        [
            ("C0", pd.Timestamp("2024-01-02"), 2.0),
            ("C0", pd.Timestamp("2024-01-09"), 3.0),
        ],
        columns=events.columns,
    )
    with caplog.at_level(logging.WARNING):
        fitted = ATT(**MAPPINGS).fit(
            outcomes,
            pd.concat([events, history], ignore_index=True),
            isolation_window=WINDOW,
        )
    assert caplog.records
    selected = fitted.events_.set_index("firm_id")
    assert selected.loc["C0", "cohort"] == pd.Timestamp("2024-01-02")
    assert not selected.loc["C0", "design_eligible"]
    for cohort in (FIRST, SECOND):
        assert set(fitted.cell_diagnostics_[(cohort, 0)]["control_firms"]) == {"C1"}


@pytest.mark.parametrize("event_time", [-1, 0], ids=["reference", "post"])
def test_missing_controls_cannot_be_hidden_by_aggregation(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
    event_time: int,
) -> None:
    """Keep no-control cells required even for the reference normalization."""
    outcomes, events = panel
    missing = (outcomes["firm"] == "C0") & (outcomes["period"] == FIRST - pd.Timedelta(days=1))
    with caplog.at_level(logging.WARNING):
        fitted = ATT(**MAPPINGS).fit(
            outcomes.loc[~missing],
            events,
            isolation_window=WINDOW,
        )
    assert len(caplog.records) == 1
    assert "no_controls" in caplog.text
    assert fitted.drop_summary_["dropped_cells"] == {"no_controls": 6}
    assert fitted.drop_summary_["dropped_treated_rows"] == 36
    assert fitted.cell_diagnostics_[(FIRST, event_time)]["status"] == "no_controls"
    assert FIRST in fitted.retained_cohorts_[event_time]
    with pytest.raises(ValueError, match="Required comparison is unavailable"):
        evaluate_theta(fitted, [7.0], bandwidth=1.0, event_times=[event_time])


def test_reference_only_success_is_not_a_fitted_model(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Do not invoke the regression solver when only reference zeros survive."""
    outcomes, events = panel
    chosen = outcomes["firm"].str.startswith("A") | (outcomes["firm"] == "C0")
    outcomes = outcomes.loc[chosen]
    events = events.loc[events["firm"].str.startswith("A")]
    missing = outcomes["firm"].str.startswith("A") & (outcomes["period"] == FIRST)
    failed = ATT(**MAPPINGS)
    with (
        patch.object(att.pf, "feols", wraps=att.pf.feols) as solver,
        pytest.raises(ValueError, match="No non-reference comparison"),
    ):
        failed.fit(outcomes.loc[~missing], events, isolation_window=(-1, 0))
    solver.assert_not_called()
    assert len(caplog.records) == 1
    assert failed.drop_summary_["unavailable_treated_rows"] == 6
    assert failed.drop_summary_["dropped_treated_rows"] == 0
    assert failed.drop_summary_["dropped_cells"] == {"no_observed_treated": 1}
    with pytest.raises(RuntimeError, match="successfully fitted"):
        failed.predict([7.0])


def test_explicit_minimum_cannot_be_bypassed_by_other_cells(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A larger requested sample minimum screens the smaller cohort."""
    outcomes, events = panel
    with caplog.at_level(logging.WARNING):
        fitted = ATT(**MAPPINGS, min_cohort_size=7).fit(
            outcomes,
            events,
            isolation_window=WINDOW,
        )
    assert len(caplog.records) == 1
    assert "small_sample" in caplog.text
    assert fitted.drop_summary_["dropped_cells"] == {"small_sample": 6}
    assert fitted.drop_summary_["dropped_treated_rows"] == 36
    assert fitted.drop_summary_["dropped_cohorts"] == 1
    assert fitted.min_cohort_size_ == 7
    assert FIRST not in fitted.retained_cohorts_[0]
    assert SECOND in fitted.retained_cohorts_[0]
    assert "A0" in set(fitted.events_["firm_id"])


def test_isolation_boundaries_use_overall_outcome_coverage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Accept touching windows and never promote an earlier out-of-panel event."""
    first = pd.Timestamp("2024-01-03")
    second = pd.Timestamp("2024-01-21")
    outcomes, events = make_panel(first, second)
    extra_outcomes = []
    for firm in ("left", "right", "outside"):
        copied = outcomes.loc[outcomes["firm"] == "C0"].copy()
        copied["firm"] = firm
        extra_outcomes.append(copied)
    outcomes = pd.concat([outcomes, *extra_outcomes], ignore_index=True)
    extra_events = pd.DataFrame(
        [
            ("left", pd.Timestamp("2024-01-02"), 3.0),
            ("right", pd.Timestamp("2024-01-22"), 3.0),
            ("outside", pd.Timestamp("2023-12-30"), 3.0),
            ("outside", pd.Timestamp("2024-01-12"), 4.0),
        ],
        columns=events.columns,
    )
    with caplog.at_level(logging.WARNING):
        fitted = ATT(**MAPPINGS).fit(
            outcomes,
            pd.concat([events, extra_events], ignore_index=True),
            isolation_window=WINDOW,
        )
    assert caplog.records
    selected = fitted.events_.set_index("firm_id")
    assert selected.loc["A0", "design_eligible"]
    assert selected.loc["B0", "design_eligible"]
    excluded = selected["design_eligible"].loc[["left", "right", "outside"]]
    assert not excluded.any()
    assert selected.loc["outside", "cohort"] == pd.Timestamp("2023-12-30")
    assert fitted.origin_period_ == pd.Timestamp("2024-01-01")
    assert fitted.last_period_ == pd.Timestamp("2024-01-24")


@pytest.mark.parametrize("missing", list(MAPPINGS))
def test_column_arguments_are_required(missing: str) -> None:
    """Never infer a missing input-column mapping from conventional labels."""
    constructor: Callable[..., ATT] = ATT
    incomplete = {key: value for key, value in MAPPINGS.items() if key != missing}
    with pytest.raises(TypeError, match=missing):
        constructor(**incomplete)


def test_normalization_renaming_and_input_order_preserve_estimates(
    panel: Panel,
) -> None:
    """Normalize times while respecting every configured input-column name."""
    outcomes, events = panel
    outcomes["period"] += pd.Timedelta(hours=13)
    events["period"] += pd.Timedelta(hours=23)
    renamed_outcomes = outcomes.rename(
        columns={"firm": "unit", "period": "when", "sales": "value"},
    ).sample(frac=1.0, random_state=1)
    renamed_events = events.rename(
        columns={"firm": "unit", "period": "when", "hours": "value"},
    ).sample(frac=1.0, random_state=2)
    fitted = ATT(
        firm_col="unit",
        period_col="when",
        outcome_col="value",
        dose_col="value",
        min_cohort_size=1,
    ).fit(renamed_outcomes, renamed_events, isolation_window=WINDOW)
    assert fitted.min_cohort_size_ == 5
    result = fitted.predict([6.0], event_times=[-2, 0])
    placebo = result.loc[result["event_time"] == -2, "att"]
    assert_allclose(placebo, 0.3 * 6 + 0.02 * 6**2, atol=1e-8)
    post = result.loc[result["event_time"] == 0, "att"]
    assert_allclose(post, 2 + 0.5 * 6 - 0.03 * 6**2 + 0.004 * 6**3, atol=1e-8)
    assert fitted.origin_period_ == pd.Timestamp("2024-01-01")
    assert fitted.last_period_ == pd.Timestamp("2024-01-24")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("text_periods", "datetime64"),
        ("aware_periods", "datetime64"),
        ("missing_sales", "finite"),
        ("normalized_duplicate", "unique by firm"),
        ("missing_column", "missing required columns"),
    ],
)
def test_invalid_periods_present_missing_outcomes_and_normalized_duplicates(
    panel: Panel,
    case: str,
    message: str,
) -> None:
    """Distinguish missing rows from invalid values and invalid period types."""
    outcomes, events = panel
    if case == "text_periods":
        outcomes["period"] = outcomes["period"].astype(str)
    elif case == "aware_periods":
        events["period"] = events["period"].dt.tz_localize("UTC")
    elif case == "missing_sales":
        outcomes.loc[0, "sales"] = np.nan
    elif case == "normalized_duplicate":
        duplicate = outcomes.iloc[[0]].copy()
        duplicate["period"] += pd.Timedelta(hours=1)
        outcomes = pd.concat([outcomes, duplicate], ignore_index=True)
    else:
        outcomes = outcomes.drop(columns="firm")
    with pytest.raises(ValueError, match=message):
        ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)


def test_positive_isolation_lower_bound_is_rejected(panel: Panel) -> None:
    """Reject an invalid signed window instead of silently changing its sign."""
    outcomes, events = panel
    with pytest.raises(ValueError, match="lower <= -1"):
        ATT(**MAPPINGS).fit(outcomes, events, isolation_window=(2, 3))


def test_failed_refit_invalidates_prior_predictions(panel: Panel) -> None:
    """An earlier valid model must not survive a failed replacement fit."""
    outcomes, events = panel
    fitted = ATT(**MAPPINGS)
    assert fitted.fit(outcomes, events, isolation_window=WINDOW) is fitted
    previous_regressions = fitted.models_.copy()
    fitted.fit(outcomes, events, isolation_window=(-1, 0))
    assert fitted.event_times_ == [-1, 0]
    assert set(fitted.models_) == {(FIRST, 0), (SECOND, 0)}
    assert all(regression is not previous_regressions[key] for key, regression in fitted.models_.items())
    with pytest.raises(ValueError, match="outside the fitted isolation window"):
        fitted.predict([7.0], event_times=[3])
    with pytest.raises(ValueError, match="missing required columns"):
        fitted.fit(outcomes.drop(columns="sales"), events, isolation_window=WINDOW)
    assert not fitted.models_
    with pytest.raises(RuntimeError, match="successfully fitted"):
        fitted.predict([7.0])


def test_kernel_weights_count_only_the_supplied_firms_once(
    kernel_events: pd.DataFrame,
) -> None:
    """Match direct Gaussian sums using only eligible firms, counted once each."""
    eligible = ["a", "b", "c"]
    doses = [4.5, 8.5]
    weights = estimate_pi(
        kernel_events,
        doses,
        eligible_firms=eligible,
        bandwidth=1.2,
    )
    selected = kernel_events.set_index("firm_id").loc[eligible]
    for dose in doses:
        distances = (selected["dose"].to_numpy(dtype=float) - dose) / 1.2
        gaussian = pd.Series(np.exp(-0.5 * distances**2), index=selected.index)
        expected = gaussian.groupby(selected["cohort"]).sum() / gaussian.sum()
        actual = weights.loc[weights["dose"] == dose].set_index("cohort")["weight"]
        assert_allclose(actual.reindex(expected.index), expected, atol=1e-14)
        assert actual.sum() == pytest.approx(1.0, rel=0.0, abs=5e-8)


@pytest.mark.parametrize(
    ("eligible_firms", "bandwidth", "message"),
    [
        (["a", "a"], 1.0, "unique firm identifiers"),
        (["unknown"], 1.0, "eligible firm must be present"),
        (["a", "b", "c"], 0.0, "bandwidth"),
        (["a", "b", "c"], np.nan, "bandwidth"),
    ],
    ids=["duplicate_firm", "unknown_firm", "zero_bandwidth", "nan_bandwidth"],
)
def test_kernel_rejects_invalid_eligibility_and_bandwidth(
    kernel_events: pd.DataFrame,
    eligible_firms: list[str],
    bandwidth: float,
    message: str,
) -> None:
    """Reject ambiguous eligible populations and invalid smoothing parameters."""
    with pytest.raises(ValueError, match=message):
        estimate_pi(
            kernel_events,
            [4.5, 8.5],
            eligible_firms=eligible_firms,
            bandwidth=bandwidth,
        )


@pytest.mark.parametrize("bad_dose", ["4", True, 4.0 + 1.0j])
def test_kernel_rejects_non_real_numeric_doses(
    kernel_events: pd.DataFrame,
    bad_dose: str | bool | complex,
) -> None:
    """Do not coerce text, Boolean, or complex doses into treatment amounts."""
    invalid = kernel_events.copy()
    invalid["dose"] = invalid["dose"].astype(object)
    invalid.loc[0, "dose"] = bad_dose
    with pytest.raises(ValueError, match="real numeric"):
        estimate_pi(
            invalid,
            [4.5, 8.5],
            eligible_firms=["a", "b", "c"],
            bandwidth=1.0,
        )


@pytest.fixture
def bandwidth_events() -> pd.DataFrame:
    """Include repeated doses within cohorts and across different cohorts."""
    return pd.DataFrame(
        {
            "firm": list("abcdefgh"),
            "period": [FIRST] * 3 + [SECOND] * 3 + [SECOND + pd.Timedelta(days=3)] * 2,
            "hours": [4.0, 4.0, 6.0, 4.0, 7.0, 9.0, 6.0, 10.0],
        },
    )


def independent_brier_scores(events: pd.DataFrame, bandwidths: FloatArray) -> FloatArray:
    """Calculate leave-one-firm-out scores without a self-weight subtraction.

    Args:
        events:
            Already selected first events using firm, period, and hours columns.
        bandwidths:
            Positive bandwidths in the order to evaluate.

    Returns:
        Mean multiclass Brier scores computed by explicitly removing each firm.
    """
    doses = events["hours"].to_numpy(dtype=float)
    cohorts = events["period"].to_numpy()
    levels = np.unique(cohorts)
    scores = []
    for bandwidth in bandwidths:
        losses = []
        for index, dose in enumerate(doses):
            other = np.arange(len(doses)) != index
            log_kernel = -0.5 * ((doses[other] - dose) / bandwidth) ** 2
            weights = np.exp(log_kernel - log_kernel.max())
            probabilities = np.array([weights[cohorts[other] == level].sum() / weights.sum() for level in levels])
            target = levels == cohorts[index]
            losses.append(np.sum((probabilities - target) ** 2))
        scores.append(np.mean(losses))
    return np.asarray(scores, dtype=float)


@pytest.mark.parametrize("doses", [[1.0, 2.0, 3.0, 4.0, 8.0], [4.0] * 6 + [8.0]])
def test_h0_matches_silverman_scale_with_zero_iqr_fallback(doses: list[float]) -> None:
    """Use sample standard deviation when repeated doses make the IQR zero."""
    events = pd.DataFrame({"firm": range(len(doses)), "period": FIRST, "hours": doses})
    deviation = np.std(doses, ddof=1)
    lower, upper = np.quantile(doses, [0.25, 0.75])
    scale = min(deviation, (upper - lower) / 1.349) if upper > lower else deviation
    expected = 0.9 * scale * len(doses) ** -0.2
    actual = estimate_h0(events, **EVENT_MAPPINGS)
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("unique_doses", [False, True], ids=["repeated", "unique"])
@pytest.mark.parametrize("batch_size", [1, 2, 256])
def test_bandwidth_cv_matches_stable_brute_force_leave_one_firm_out(
    bandwidth_events: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
    unique_doses: bool,
    batch_size: int,
) -> None:
    """Keep scores invariant to batching, including extremely narrow kernels."""
    if unique_doses:
        bandwidth_events["hours"] = [4.0, 4.5, 5.0, 7.0, 7.5, 8.0, 10.0, 11.0]
    candidates = np.array([2.0, 0.001, 0.7, 1.3])
    with caplog.at_level(logging.WARNING):
        selected, scores = estimate_h_cv(
            bandwidth_events,
            **EVENT_MAPPINGS,
            bandwidths=candidates,
            batch_size=batch_size,
        )
    assert len(caplog.records) <= 1
    assert list(scores.columns) == ["bandwidth", "cv_brier"]
    ordered = np.sort(candidates)
    expected = independent_brier_scores(bandwidth_events, ordered)
    assert_allclose(scores["bandwidth"], ordered, atol=0.0, rtol=0.0)
    assert_allclose(scores["cv_brier"], expected, atol=2e-14, rtol=2e-14)
    assert selected == ordered[np.argmin(expected)]
    assert np.isfinite(scores["cv_brier"]).all()


def test_bandwidth_default_grid_uses_h0(bandwidth_events: pd.DataFrame) -> None:
    """Evaluate the documented nine bandwidths without changing their order."""
    h0 = estimate_h0(bandwidth_events, **EVENT_MAPPINGS)
    selected, scores = estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS)
    expected_grid = h0 * np.geomspace(0.25, 4.0, 9)
    assert_allclose(scores["bandwidth"], expected_grid, atol=0.0, rtol=1e-14)
    expected = independent_brier_scores(bandwidth_events, expected_grid)
    assert_allclose(scores["cv_brier"], expected, atol=2e-14, rtol=2e-14)
    assert selected == expected_grid[np.argmin(expected)]


def test_bandwidth_cv_accepts_finite_scaled_distances_for_large_doses() -> None:
    """Evaluate an explicit bandwidth when the sample variance overflows."""
    events = pd.DataFrame(
        {"firm": range(7), "period": [FIRST] * 3 + [SECOND] * 4, "hours": [4.0] * 6 + [1e308]},
    )
    candidates = np.array([1e308])
    selected, scores = estimate_h_cv(events, **EVENT_MAPPINGS, bandwidths=candidates)
    expected = independent_brier_scores(events, candidates)
    assert selected == 1e308
    assert np.isfinite(scores["cv_brier"]).all()
    assert_allclose(scores["cv_brier"], expected, atol=2e-14, rtol=2e-14)


def test_bandwidth_selection_respects_first_events_mappings_and_eligibility() -> None:
    """Normalize first-event ties without filtering or mutating caller data."""
    events = pd.DataFrame(
        [
            ("a", pd.Timestamp("2024-01-08 20:00"), 4.0),
            ("b", pd.Timestamp("2024-01-11 12:00"), 50.0),
            ("a", pd.Timestamp("2024-01-08 01:00"), 9.0),
            ("c", pd.Timestamp("2024-01-01 12:00"), 8.0),
            ("b", pd.Timestamp("2024-01-09 12:00"), 6.0),
            ("d", pd.Timestamp("2024-01-09 13:00"), 6.0),
            ("e", pd.Timestamp("2023-12-01 12:00"), 10.0),
            ("a", pd.Timestamp("2024-01-20 12:00"), 100.0),
        ],
        columns=["person", "when", "assigned"],
        index=[3] * 8,
    )
    original = events.copy(deep=True)
    expected_events = pd.DataFrame(
        [("a", FIRST, 4.0), ("b", pd.Timestamp("2024-01-09"), 6.0), ("e", pd.Timestamp("2023-12-01"), 10.0)],
        columns=["firm", "period", "hours"],
    )
    mappings: EventMappings = {"firm_col": "person", "period_col": "when", "dose_col": "assigned"}
    with patch.object(att.pf, "feols", side_effect=AssertionError("Bandwidth selection must not fit ATT")) as solver:
        all_h0 = estimate_h0(events, **mappings)
        selected_h0 = estimate_h0(events, **mappings, eligible_firms=["e", "a", "b"])
        _, scores = estimate_h_cv(events, **mappings, eligible_firms=["e", "a", "b"], bandwidths=[0.5, 1.5, 4.0])
    solver.assert_not_called()
    selected_doses = np.array([4.0, 6.0, 10.0])
    expected_h0 = 0.9 * min(np.std(selected_doses, ddof=1), 3.0 / 1.349) * 3**-0.2
    assert selected_h0 == pytest.approx(expected_h0, rel=1e-14, abs=0.0)
    expected_all = 0.9 * min(np.std([4.0, 6.0, 8.0, 6.0, 10.0], ddof=1), 2.0 / 1.349) * 5**-0.2
    assert all_h0 == pytest.approx(expected_all, rel=1e-14, abs=0.0)
    expected_scores = independent_brier_scores(expected_events, np.array([0.5, 1.5, 4.0]))
    assert_allclose(scores["cv_brier"], expected_scores, atol=2e-14, rtol=2e-14)
    assert_frame_equal(events, original)


def test_bandwidth_cv_combines_singleton_boundary_and_tie_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Choose the smallest tied candidate and log all caveats only once."""
    events = pd.DataFrame({"firm": ["a", "b"], "period": [FIRST, SECOND], "hours": [4.0, 8.0]})
    with caplog.at_level(logging.WARNING):
        selected, scores = estimate_h_cv(events, **EVENT_MAPPINGS, bandwidths=[2.0, 0.001, 1.0])
    assert selected == 0.001
    assert_allclose(scores["cv_brier"], 2.0, atol=0.0, rtol=0.0)
    assert len(caplog.records) == 1
    assert "singleton" in caplog.text
    assert "boundary" in caplog.text
    assert "tie" in caplog.text


@pytest.mark.parametrize("estimator", [estimate_h0, estimate_h_cv], ids=["h0", "cv"])
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("constant", "variation"),
        ("nan_dose", "finite"),
        ("boolean_dose", "real numeric"),
        ("text_period", "datetime64"),
        ("missing_firm", "must not be missing"),
        ("unknown_firm", "eligible firm must be present"),
    ],
)
def test_bandwidth_estimators_reject_invalid_samples(
    bandwidth_events: pd.DataFrame,
    estimator: Callable[..., object],
    case: str,
    message: str,
) -> None:
    """Reject invalid populations before calculating smoothing parameters."""
    eligible = list(bandwidth_events["firm"])
    if case == "constant":
        bandwidth_events["hours"] = 4.0
    elif case == "nan_dose":
        bandwidth_events.loc[0, "hours"] = np.nan
    elif case == "boolean_dose":
        bandwidth_events["hours"] = True
    elif case == "text_period":
        bandwidth_events["period"] = bandwidth_events["period"].astype(str)
    elif case == "missing_firm":
        bandwidth_events.loc[0, "firm"] = None
    else:
        eligible = ["unknown"]
    with pytest.raises(ValueError, match=message):
        estimator(bandwidth_events, **EVENT_MAPPINGS, eligible_firms=eligible)


def test_bandwidth_cv_requires_multiple_cohorts(bandwidth_events: pd.DataFrame) -> None:
    """Allow Silverman scale estimation but reject a one-category CV target."""
    bandwidth_events["period"] = FIRST
    assert estimate_h0(bandwidth_events, **EVENT_MAPPINGS) > 0.0
    with pytest.raises(ValueError, match="cohort"):
        estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS, bandwidths=[0.5, 1.0])


@pytest.mark.parametrize("candidates", [[], [0.0, 1.0], [np.nan, 1.0], [1.0, 1.0], [True, 2.0]])
def test_bandwidth_cv_rejects_invalid_candidate_grids(
    bandwidth_events: pd.DataFrame,
    candidates: list[float],
) -> None:
    """Require distinct positive finite bandwidths instead of coercing them."""
    with pytest.raises(ValueError, match="bandwidth"):
        estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS, bandwidths=candidates)


@pytest.mark.parametrize("batch_size", [0, 1.5, True])
def test_bandwidth_cv_rejects_invalid_batch_sizes(
    bandwidth_events: pd.DataFrame,
    batch_size: int | float,
) -> None:
    """Require a positive integer batch size without accepting Boolean values."""
    estimator: Callable[..., object] = estimate_h_cv
    with pytest.raises(ValueError, match="batch_size"):
        estimator(bandwidth_events, **EVENT_MAPPINGS, bandwidths=[0.5, 1.0], batch_size=batch_size)
