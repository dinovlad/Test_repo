"""Check continuous-dose estimates against independent numerical comparisons."""

import logging
import unittest.mock
from collections.abc import Callable
from typing import TypedDict, cast

import numpy as np
import numpy.testing as np_testing
import numpy.typing as npt
import pandas as pd
import pandas.testing as pd_testing
import pydantic
import pytest
import scipy.interpolate

import labor_cost_optimization.att as att


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
FloatArray = npt.NDArray[np.float64]


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
    basis = scipy.interpolate.BSpline(knot_vector, np.eye(dimension), degree, extrapolate=True)
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
    """Preserve ATT and theta while fitting each cell with a narrow formula.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
        degree:
            Polynomial degree for the comparison regressions.
    """
    outcomes, events = panel
    original_outcomes = outcomes.copy(deep=True)
    original_events = events.copy(deep=True)
    with unittest.mock.patch.object(att.pf, "feols", wraps=att.pf.feols) as solver:
        fitted = att.ATT(**MAPPINGS, degree=degree).fit(outcomes, events, isolation_window=WINDOW)
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
        np_testing.assert_allclose(block.loc[treated, "scaled_dose"], expected_scaled, atol=0.0, rtol=0.0)
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
        np_testing.assert_allclose(group["att"], expected, atol=1e-8, rtol=1e-8)
        independent_components[(cohort, event_time)] = expected
    theta = att.evaluate_theta(fitted, doses, bandwidth=1.3)
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
        np_testing.assert_allclose(actual, expected_theta, atol=1e-8, rtol=1e-8)
    pd_testing.assert_frame_equal(outcomes, original_outcomes)
    pd_testing.assert_frame_equal(events, original_events)


@pytest.mark.parametrize("identifier_kind", ["strings", "categorical", "mixed"])
def test_prepared_comparisons_match_direct_row_lookup(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
    identifier_kind: str,
) -> None:
    """Check missing-period and history selection before invoking the solver.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
        identifier_kind:
            Firm identifier representation used in the comparison lookup.
    """
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
    outcomes.index = pd.Index(np.zeros(len(outcomes), dtype=int))
    events.index = pd.Index(np.zeros(len(events), dtype=int))
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

    fitted = att.ATT(**MAPPINGS)
    with unittest.mock.patch.object(fitted, "_fit_model") as solver, caplog.at_level(logging.WARNING):
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
                np_testing.assert_allclose(block["delta_y"], expected_changes, atol=0.0, rtol=0.0)
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


def test_placebo_sign_and_reference_normalization(model: att.ATT) -> None:
    """Recover a nonzero pre-trend and exact unestimated reference zeros.

    Args:
        model:
            Independently fitted estimator for the default synthetic panel.
    """
    results = model.predict([6.0, 8.0], event_times=[-2, -1, 0])
    placebo = results.loc[results["event_time"] == -2]
    expected = 0.3 * placebo["dose"] + 0.02 * placebo["dose"] ** 2
    np_testing.assert_allclose(placebo["att"], expected, atol=1e-8)
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
    """Allow zero changes and calendar changes shared by every firm.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        slope:
            Common outcome-change specification applied to every firm.
    """
    outcomes, events = panel
    periods = (outcomes["period"] - outcomes["period"].min()).dt.days
    outcomes["sales"] = 200.0 + slope * periods
    fitted = att.ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
    result = fitted.predict([6.0, 8.0])
    np_testing.assert_allclose(result["att"], 0.0, atol=1e-10)
    theta = att.evaluate_theta(fitted, [6.0, 8.0], bandwidth=1.0)
    np_testing.assert_allclose(theta["theta"], 0.0, atol=1e-10)


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "nonfinite_intercept"])
def test_incomplete_or_nonfinite_solver_coefficients_fail_the_entire_fit(
    panel: Panel,
    model: att.ATT,
    failure: str,
) -> None:
    """Never accept a backend's missing term or silently reduce a fitted curve.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        model:
            Independently fitted estimator for the default synthetic panel.
        failure:
            Backend failure or reporting error to simulate.
    """
    outcomes, events = panel
    first_key, second_key = list(model.models_)[:2]
    coefficients = model.models_[second_key].coef()
    term = model.effect_columns_[second_key][-1]
    if failure == "missing":
        malformed = coefficients.drop(term)
    else:
        malformed = coefficients.copy()
        malformed.loc["Intercept" if failure == "nonfinite_intercept" else term] = np.nan
    backend_model = unittest.mock.Mock()
    backend_model.coef.return_value = malformed
    failed = att.ATT(**MAPPINGS)
    with (
        unittest.mock.patch.object(att.pf, "feols", side_effect=[model.models_[first_key], backend_model]) as solver,
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


def test_prediction_grid_cannot_redefine_the_fitted_basis(model: att.ATT) -> None:
    """Keep predictions unchanged when other requested doses change.

    Args:
        model:
            Independently fitted estimator for the default synthetic panel.
    """
    saved = {key: basis.copy() for key, basis in model.bases_.items()}
    single = model.predict([7.0])
    expanded = model.predict([5.5, 7.0, 8.5])
    expanded = expanded.loc[expanded["dose"] == 7.0].reset_index(drop=True)
    pd_testing.assert_frame_equal(single, expanded)
    assert model.bases_ == saved
    with pytest.raises(ValueError, match="outside the fitted isolation window"):
        model.predict([7.0], event_times=[4])


def test_aggregation_matches_manual_kernel_weighted_curves(
    panel: Panel,
    model: att.ATT,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserve cohort size and dose composition at every requested horizon.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        model:
            Independently fitted estimator for the default synthetic panel.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    _, events = panel
    doses = [5.5, 8.5]
    bandwidth = 1.3
    theta = att.evaluate_theta(
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
    """Count dropped comparisons and extrapolated components in one message.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    outcomes, events = panel
    fitted = att.ATT(**MAPPINGS, min_cohort_size=7).fit(outcomes, events, isolation_window=WINDOW)
    fit_summary = fitted.drop_summary_.copy()
    caplog.clear()
    doses = [14.0, 2.0, 7.0]
    horizons = [3, -1, 0, -2]
    with caplog.at_level(logging.WARNING):
        theta = att.evaluate_theta(fitted, doses, bandwidth=1.3, event_times=horizons)
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
        np_testing.assert_allclose(group["theta"], expected, atol=1e-8, rtol=1e-8)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        predictions = fitted.predict(doses, event_times=horizons)
    assert len(caplog.records) == 2
    assert "unavailable comparisons" in caplog.records[0].getMessage()
    assert caplog.records[0].args == (4,)
    assert "Extrapolating" in caplog.records[1].getMessage()
    assert caplog.records[1].args == (6,)
    np_testing.assert_allclose(predictions["att"], theta["theta"], atol=1e-12, rtol=0.0)


def test_reference_aggregation_outside_support_does_not_warn(
    model: att.ATT,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep reference zeros free of extrapolation flags and warning counts.

    Args:
        model:
            Independently fitted estimator for the default synthetic panel.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    with caplog.at_level(logging.WARNING):
        theta = att.evaluate_theta(model, [2.0, 14.0], bandwidth=1.3, event_times=[-1])
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
    """Reject the removed spline option instead of silently changing its model.

    Args:
        num_knots:
            Removed spline argument that the public constructor must reject.
    """
    constructor: Callable[..., att.ATT] = att.ATT
    with pytest.raises(TypeError, match="num_knots"):
        constructor(**MAPPINGS, num_knots=num_knots)


def test_unused_categorical_firms_do_not_create_phantom_terminations(
    panel: Panel,
    model: att.ATT,
) -> None:
    """Keep unused categories from blocking controls or creating events.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        model:
            Independently fitted estimator for the default synthetic panel.
    """
    outcomes, events = panel
    outcomes = outcomes.copy()
    events = events.copy()
    categories = [*outcomes["firm"].unique(), "unused_firm"]
    outcomes["firm"] = pd.Categorical(outcomes["firm"], categories=categories)
    events["firm"] = pd.Categorical(events["firm"], categories=categories)
    categorical = att.ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
    assert categorical.duplicate_diagnostics_["n_dropped_events"] == 0
    assert categorical.duplicate_diagnostics_["n_affected_firms"] == 0
    assert (categorical.termination_history_["n_terminations"] > 0).all()
    assert set(categorical.termination_history_["firm_id"]) == set(events["firm"])
    pd_testing.assert_frame_equal(
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
    """Do not require complete trajectories or leak missing firms into weights.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
    """
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
        fitted = att.ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
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
        observed_target = att.evaluate_theta(fitted, [7.0], bandwidth=1.0, event_times=[0])
    assert not caplog.records
    assert observed_target["availability_restricted"].all()
    assert not observed_target["restricted_target"].any()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        theta = att.evaluate_theta(fitted, [7.0], bandwidth=1.0, event_times=[1])
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
    np_testing.assert_allclose(prediction["att"], expected, atol=1e-8)


def test_same_period_ties_are_not_promoted_to_later_events(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log dropped records while retaining simultaneous-event multiplicity.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
    """
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
        fitted = att.ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
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
    """Preserve later conflicting terminations after excluding a first event.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
    """
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
        fitted = att.ATT(**MAPPINGS).fit(
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
    """Keep no-control cells required even for the reference normalization.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
        event_time:
            Horizon at which to remove the required control observation.
    """
    outcomes, events = panel
    missing = (outcomes["firm"] == "C0") & (outcomes["period"] == FIRST - pd.Timedelta(days=1))
    with caplog.at_level(logging.WARNING):
        fitted = att.ATT(**MAPPINGS).fit(
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
        att.evaluate_theta(fitted, [7.0], bandwidth=1.0, event_times=[event_time])


def test_reference_only_success_is_not_a_fitted_model(
    panel: Panel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Do not invoke the regression solver when only reference zeros survive.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    outcomes, events = panel
    chosen = outcomes["firm"].str.startswith("A") | (outcomes["firm"] == "C0")
    outcomes = outcomes.loc[chosen]
    events = events.loc[events["firm"].str.startswith("A")]
    missing = outcomes["firm"].str.startswith("A") & (outcomes["period"] == FIRST)
    failed = att.ATT(**MAPPINGS)
    with (
        unittest.mock.patch.object(att.pf, "feols", wraps=att.pf.feols) as solver,
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
    """Screen the smaller cohort using the requested sample minimum.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    outcomes, events = panel
    with caplog.at_level(logging.WARNING):
        fitted = att.ATT(**MAPPINGS, min_cohort_size=7).fit(
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
    panel_factory: Callable[[pd.Timestamp, pd.Timestamp], Panel],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Accept touching windows and never promote an earlier out-of-panel event.

    Args:
        panel_factory:
            Factory creating synthetic panels with specified cohort periods.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    first = pd.Timestamp("2024-01-03")
    second = pd.Timestamp("2024-01-21")
    outcomes, events = panel_factory(first, second)
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
        fitted = att.ATT(**MAPPINGS).fit(
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
    """Never infer a missing input-column mapping from conventional labels.

    Args:
        missing:
            Required column-mapping argument omitted from construction.
    """
    constructor: Callable[..., att.ATT] = att.ATT
    incomplete = {key: value for key, value in MAPPINGS.items() if key != missing}
    with pytest.raises(TypeError, match=missing):
        constructor(**incomplete)


def test_runtime_validation_preserves_numpy_scalar_configuration(panel: Panel, model: att.ATT) -> None:
    """Accept NumPy integer and string scalars without changing fitted curves.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        model:
            Reference estimator fitted with ordinary Python configuration values.
    """
    constructor: Callable[..., att.ATT] = att.ATT
    fitted = constructor(
        firm_col=np.str_("firm"),
        period_col=np.str_("period"),
        outcome_col=np.str_("sales"),
        dose_col=np.str_("hours"),
        degree=np.int64(3),
        min_cohort_size=np.int64(5),
    )
    fit: Callable[..., att.ATT] = fitted.fit
    outcomes, events = panel
    fit(outcomes, events, isolation_window=(np.int64(-2), np.int64(3)))
    assert type(fitted.degree) is int
    assert type(fitted.min_cohort_size) is int
    pd_testing.assert_frame_equal(fitted.predict([5.0, 7.0]), model.predict([5.0, 7.0]))


def test_prediction_and_aggregation_accept_single_pass_iterables(model: att.ATT) -> None:
    """Consume each requested dose and horizon iterator only once.

    Args:
        model:
            Successfully fitted estimator using the reference configuration.
    """
    doses = np.asarray([5.0, 7.0], dtype=np.float32)
    horizons = np.asarray([-2, 0, 2], dtype=np.int64)
    aggregate: Callable[..., pd.DataFrame] = att.evaluate_theta
    expected_prediction = model.predict(doses.tolist(), event_times=horizons.tolist())
    expected_theta = att.evaluate_theta(model, doses.tolist(), event_times=horizons.tolist(), bandwidth=1.2)
    pd_testing.assert_frame_equal(model.predict(iter(doses), event_times=iter(horizons)), expected_prediction)
    pd_testing.assert_frame_equal(
        aggregate(model, iter(doses), event_times=iter(horizons), bandwidth=np.float64(1.2)),
        expected_theta,
    )


@pytest.mark.parametrize(
    "doses",
    [[], [0.0], [-1.0], [np.nan], [np.inf], [5.0, 5.0], ["5"], [True], [np.bool_(True)], [True, 5.0], [[5.0]], 5.0],
    ids=[
        "empty",
        "zero",
        "negative",
        "nan",
        "infinite",
        "duplicate",
        "text",
        "bool",
        "numpy_bool",
        "mixed_bool",
        "matrix",
        "scalar",
    ],
)
def test_invalid_prediction_grids_are_rejected_consistently(model: att.ATT, doses: object) -> None:
    """Apply the same dose restrictions to prediction and aggregation.

    Args:
        model:
            Successfully fitted estimator using the reference configuration.
        doses:
            Invalid dose collection that must not reach curve evaluation.
    """
    predict: Callable[..., pd.DataFrame] = model.predict
    aggregate: Callable[..., pd.DataFrame] = att.evaluate_theta
    with pytest.raises(ValueError, match=r"doses|validation"):
        predict(doses, event_times=[0])
    with pytest.raises(ValueError, match=r"doses|validation"):
        aggregate(model, doses, event_times=[0], bandwidth=1.2)


@pytest.mark.parametrize(
    ("argument", "value", "error_type"),
    [
        ("degree", True, "int_type"),
        ("degree", np.bool_(True), "int_type"),
        ("degree", "3", "int_type"),
        ("degree", 3.0, "int_type"),
        ("degree", 0, "greater_than_equal"),
        ("min_cohort_size", True, "int_type"),
        ("min_cohort_size", "5", "int_type"),
        ("min_cohort_size", 5.0, "int_type"),
        ("min_cohort_size", 0, "greater_than_equal"),
        ("firm_col", 1, "string_type"),
        ("dose_col", "", "string_too_short"),
    ],
)
def test_runtime_validation_does_not_coerce_configuration(argument: str, value: object, error_type: str) -> None:
    """Preserve strict constructor validation when using runtime schemas.

    Args:
        argument:
            Configuration argument to replace with an invalid value.
        value:
            Value that must not be coerced into an accepted configuration.
        error_type:
            Pydantic error identifying the violated field constraint.
    """
    constructor: Callable[..., att.ATT] = att.ATT
    arguments: dict[str, object] = {**MAPPINGS, argument: value}
    with pytest.raises(pydantic.ValidationError) as error:
        constructor(**arguments)
    assert [(item["loc"], item["type"]) for item in error.value.errors()] == [((argument,), error_type)]


def test_runtime_validation_rejects_overlapping_column_names() -> None:
    """Require distinct identifier and period columns within each input table."""
    mappings = MAPPINGS.copy()
    mappings["period_col"] = "firm"
    with pytest.raises(ValueError, match="Mapped column names must be distinct within each table"):
        att.ATT(**mappings)


def test_runtime_validation_accepts_dataframe_subclasses(panel: Panel, model: att.ATT) -> None:
    """Preserve arbitrary DataFrame subclasses when validating input tables.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        model:
            Reference estimator fitted with ordinary DataFrame inputs.
    """

    class Frame(pd.DataFrame):
        """Supply a DataFrame subtype without changing table behavior."""

    outcomes, events = panel
    fitted = att.ATT(**MAPPINGS).fit(Frame(outcomes), Frame(events), isolation_window=WINDOW)
    pd_testing.assert_frame_equal(fitted.predict([5.0, 7.0]), model.predict([5.0, 7.0]))


@pytest.mark.parametrize("argument", ["outcomes", "events"])
@pytest.mark.parametrize("value", [{}, []], ids=["mapping", "list"])
def test_runtime_validation_does_not_coerce_input_tables(
    panel: Panel,
    model: att.ATT,
    argument: str,
    value: object,
) -> None:
    """Reject mappings and row sequences instead of converting them to tables.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        model:
            Successfully fitted estimator whose state must be invalidated.
        argument:
            Input table to replace with a non-DataFrame object.
        value:
            Object that must be rejected by the DataFrame type check.
    """
    outcomes, events = panel
    inputs = {"outcomes": outcomes, "events": events, argument: value}
    fit: Callable[..., att.ATT] = model.fit
    with pytest.raises(pydantic.ValidationError) as error:
        fit(**inputs, isolation_window=WINDOW)
    assert [(item["loc"], item["type"]) for item in error.value.errors()] == [((argument,), "is_instance_of")]
    assert not model.models_
    assert model.outcome_fingerprint_ is None
    with pytest.raises(RuntimeError, match="successfully fitted"):
        model.predict([7.0])


def test_normalization_renaming_and_input_order_preserve_estimates(
    panel: Panel,
) -> None:
    """Normalize times while respecting every configured input-column name.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
    """
    outcomes, events = panel
    outcomes["period"] += pd.Timedelta(hours=13)
    events["period"] += pd.Timedelta(hours=23)
    renamed_outcomes = outcomes.rename(
        columns={"firm": "unit", "period": "when", "sales": "value"},
    ).sample(frac=1.0, random_state=1)
    renamed_events = events.rename(
        columns={"firm": "unit", "period": "when", "hours": "value"},
    ).sample(frac=1.0, random_state=2)
    fitted = att.ATT(
        firm_col="unit",
        period_col="when",
        outcome_col="value",
        dose_col="value",
        min_cohort_size=1,
    ).fit(renamed_outcomes, renamed_events, isolation_window=WINDOW)
    assert fitted.min_cohort_size_ == 5
    result = fitted.predict([6.0], event_times=[-2, 0])
    placebo = result.loc[result["event_time"] == -2, "att"]
    np_testing.assert_allclose(placebo, 0.3 * 6 + 0.02 * 6**2, atol=1e-8)
    post = result.loc[result["event_time"] == 0, "att"]
    np_testing.assert_allclose(post, 2 + 0.5 * 6 - 0.03 * 6**2 + 0.004 * 6**3, atol=1e-8)
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
    """Distinguish missing rows from invalid values and invalid period types.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
        case:
            Invalid input or sample configuration to exercise.
        message:
            Expected validation error or diagnostic message fragment.
    """
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
        att.ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)


def test_positive_isolation_lower_bound_is_rejected(panel: Panel) -> None:
    """Reject an invalid signed window instead of silently changing its sign.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
    """
    outcomes, events = panel
    with pytest.raises(pydantic.ValidationError) as error:
        att.ATT(**MAPPINGS).fit(outcomes, events, isolation_window=(2, 3))
    assert [(item["loc"], item["type"]) for item in error.value.errors()] == [
        (("isolation_window", 0), "less_than_equal"),
    ]


def test_failed_refit_invalidates_prior_predictions(panel: Panel) -> None:
    """Invalidate earlier predictions after a replacement fit fails.

    Args:
        panel:
            Fresh synthetic outcome and event tables.
    """
    outcomes, events = panel
    fitted = att.ATT(**MAPPINGS)
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


@pytest.mark.parametrize(
    "invalid_window",
    [(False, 3), (np.bool_(False), 3), (-2.0, 3), ("-2", 3), (-2,), (-2, 3, 4), None],
    ids=["bool", "numpy_bool", "float", "text", "short", "long", "none"],
)
def test_invalid_refit_window_clears_fitted_state(panel: Panel, model: att.ATT, invalid_window: object) -> None:
    """Invalidate an existing fit before rejecting malformed window inputs.

    Args:
        panel:
            Original outcome and event tables.
        model:
            Estimator with a successful fit before this replacement attempt.
        invalid_window:
            Invalid offset types or an invalid number of endpoints.
    """
    outcomes, events = panel
    fit: Callable[..., att.ATT] = model.fit
    with pytest.raises(ValueError, match=r"isolation_window|offset|validation"):
        fit(outcomes, events, isolation_window=invalid_window)
    assert not model.models_
    assert model.outcome_fingerprint_ is None
    with pytest.raises(RuntimeError, match="successfully fitted"):
        model.predict([7.0])


def test_kernel_weights_count_only_the_supplied_firms_once(
    kernel_events: pd.DataFrame,
) -> None:
    """Match direct Gaussian sums using only eligible firms, counted once each.

    Args:
        kernel_events:
            Canonical first-event metadata with an explicitly excluded firm.
    """
    eligible = ["a", "b", "c"]
    doses = [4.5, 8.5]
    weights = att.estimate_pi(
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
        np_testing.assert_allclose(actual.reindex(expected.index), expected, atol=1e-14)
        assert actual.sum() == pytest.approx(1.0, rel=0.0, abs=5e-8)


def test_kernel_accepts_numpy_scalars_and_single_pass_iterables(kernel_events: pd.DataFrame) -> None:
    """Preserve kernel weights for NumPy scalar and iterator inputs.

    Args:
        kernel_events:
            Canonical first-event metadata with an explicitly excluded firm.
    """
    expected = att.estimate_pi(kernel_events, [4.5, 8.5], eligible_firms=["a", "b", "c"], bandwidth=1.2)
    estimate: Callable[..., pd.DataFrame] = att.estimate_pi
    actual = estimate(
        kernel_events,
        iter(np.asarray([4.5, 8.5], dtype=np.float32)),
        eligible_firms=iter(["a", "b", "c"]),
        bandwidth=np.float64(1.2),
    )
    pd_testing.assert_frame_equal(actual, expected)


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
    """Reject ambiguous eligible populations and invalid smoothing parameters.

    Args:
        kernel_events:
            Canonical first-event metadata with an explicitly excluded firm.
        eligible_firms:
            Firm identifiers selected for kernel aggregation.
        bandwidth:
            Gaussian-kernel bandwidth supplied to the estimator.
        message:
            Expected validation error or diagnostic message fragment.
    """
    with pytest.raises(ValueError, match=message):
        att.estimate_pi(
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
    """Do not coerce text, Boolean, or complex doses into treatment amounts.

    Args:
        kernel_events:
            Canonical first-event metadata with an explicitly excluded firm.
        bad_dose:
            Non-real numeric or nonnumeric value that must be rejected.
    """
    invalid = kernel_events.copy()
    invalid["dose"] = invalid["dose"].astype(object)
    invalid.loc[0, "dose"] = bad_dose
    with pytest.raises(ValueError, match="real numeric"):
        att.estimate_pi(
            invalid,
            [4.5, 8.5],
            eligible_firms=["a", "b", "c"],
            bandwidth=1.0,
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
    """Use sample standard deviation when repeated doses make the IQR zero.

    Args:
        doses:
            Observed dose sample used for the independent bandwidth calculation.
    """
    events = pd.DataFrame({"firm": range(len(doses)), "period": FIRST, "hours": doses})
    deviation = np.std(doses, ddof=1)
    lower, upper = np.quantile(doses, [0.25, 0.75])
    scale = min(deviation, (upper - lower) / 1.349) if upper > lower else deviation
    expected = 0.9 * scale * len(doses) ** -0.2
    actual = att.estimate_h0(events, **EVENT_MAPPINGS)
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
    """Keep scores invariant to batching, including extremely narrow kernels.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        caplog:
            Fixture capturing diagnostic log messages.
        unique_doses:
            Whether to replace repeated doses with distinct values.
        batch_size:
            Number of dose queries processed in each computation batch.
    """
    if unique_doses:
        bandwidth_events["hours"] = [4.0, 4.5, 5.0, 7.0, 7.5, 8.0, 10.0, 11.0]
    candidates = np.array([2.0, 0.001, 0.7, 1.3])
    with caplog.at_level(logging.WARNING):
        selected, scores = att.estimate_h_cv(
            bandwidth_events,
            **EVENT_MAPPINGS,
            bandwidths=candidates,
            batch_size=batch_size,
        )
    assert len(caplog.records) <= 1
    assert list(scores.columns) == ["bandwidth", "cv_brier"]
    ordered = scores["bandwidth"].to_numpy(dtype=float)
    assert set(candidates).issubset(set(ordered))
    np_testing.assert_allclose(ordered, np.sort(ordered), atol=0.0, rtol=0.0)
    expected = independent_brier_scores(bandwidth_events, ordered)
    np_testing.assert_allclose(scores["cv_brier"], expected, atol=2e-14, rtol=2e-14)
    assert selected == ordered[np.argmin(scores["cv_brier"].to_numpy())]
    assert len(ordered) > len(candidates)
    assert np.isfinite(scores["cv_brier"]).all()


def test_bandwidth_default_grid_uses_h0(bandwidth_events: pd.DataFrame) -> None:
    """Include all nine starting bandwidths and validate every expanded score.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
    """
    h0 = att.estimate_h0(bandwidth_events, **EVENT_MAPPINGS)
    selected, scores = att.estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS)
    expected_grid = h0 * np.geomspace(0.1, 10.0, 9)
    evaluated = scores["bandwidth"].to_numpy(dtype=float)
    assert set(expected_grid).issubset(set(evaluated))
    expected = independent_brier_scores(bandwidth_events, evaluated)
    np_testing.assert_allclose(scores["cv_brier"], expected, atol=2e-14, rtol=2e-14)
    assert selected == evaluated[np.argmin(scores["cv_brier"].to_numpy())]


def test_bandwidth_cv_accepts_finite_scaled_distances_for_large_doses() -> None:
    """Evaluate an explicit bandwidth when the sample variance overflows."""
    events = pd.DataFrame(
        {"firm": range(7), "period": [FIRST] * 3 + [SECOND] * 4, "hours": [4.0] * 6 + [1e308]},
    )
    candidates = np.array([1e308])
    selected, scores = att.estimate_h_cv(events, **EVENT_MAPPINGS, bandwidths=candidates)
    evaluated = scores["bandwidth"].to_numpy(dtype=float)
    expected = independent_brier_scores(events, evaluated)
    assert 1e308 in evaluated
    assert selected == evaluated[np.argmin(scores["cv_brier"].to_numpy())]
    assert np.isfinite(scores["cv_brier"]).all()
    np_testing.assert_allclose(scores["cv_brier"], expected, atol=2e-14, rtol=2e-14)


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
    with unittest.mock.patch.object(
        att.pf,
        "feols",
        side_effect=AssertionError("Bandwidth selection must not fit ATT"),
    ) as solver:
        all_h0 = att.estimate_h0(events, **mappings)
        selected_h0 = att.estimate_h0(events, **mappings, eligible_firms=["e", "a", "b"])
        _, scores = att.estimate_h_cv(events, **mappings, eligible_firms=["e", "a", "b"], bandwidths=[0.5, 1.5, 4.0])
    solver.assert_not_called()
    selected_doses = np.array([4.0, 6.0, 10.0])
    expected_h0 = 0.9 * min(np.std(selected_doses, ddof=1), 3.0 / 1.349) * 3**-0.2
    assert selected_h0 == pytest.approx(expected_h0, rel=1e-14, abs=0.0)
    expected_all = 0.9 * min(float(np.std([4.0, 6.0, 8.0, 6.0, 10.0], ddof=1)), 2.0 / 1.349) * 5**-0.2
    assert all_h0 == pytest.approx(expected_all, rel=1e-14, abs=0.0)
    evaluated = scores["bandwidth"].to_numpy(dtype=float)
    assert {0.5, 1.5, 4.0}.issubset(set(evaluated))
    expected_scores = independent_brier_scores(expected_events, evaluated)
    np_testing.assert_allclose(scores["cv_brier"], expected_scores, atol=2e-14, rtol=2e-14)
    pd_testing.assert_frame_equal(events, original)


def test_bandwidth_cv_combines_singleton_boundary_and_tie_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Choose the smallest tied candidate and log all caveats only once.

    Args:
        caplog:
            Fixture capturing diagnostic log messages.
    """
    events = pd.DataFrame({"firm": ["a", "b"], "period": [FIRST, SECOND], "hours": [4.0, 8.0]})
    with caplog.at_level(logging.WARNING):
        selected, scores = att.estimate_h_cv(events, **EVENT_MAPPINGS, bandwidths=[2.0, 0.001, 1.0])
    assert selected == 0.001 / 8
    np_testing.assert_allclose(
        scores["bandwidth"],
        [0.001 / 8, 0.001 / 4, 0.001 / 2, 0.001, 1.0, 2.0, 4.0, 8.0, 16.0],
        atol=0.0,
        rtol=0.0,
    )
    np_testing.assert_allclose(scores["cv_brier"], 2.0, atol=0.0, rtol=0.0)
    assert len(caplog.records) == 1
    assert "singleton" in caplog.text
    assert "boundary" in caplog.text
    assert "tie" in caplog.text


@pytest.mark.parametrize(
    ("initial", "curve", "expected_bandwidth"),
    [
        ([1.0, 2.0, 4.0], {1.0: 1.3, 2.0: 1.0, 4.0: 0.8, 8.0: 0.5, 16.0: 0.6}, 8.0),
        ([1.0, 2.0, 4.0], {0.25: 0.7, 0.5: 0.4, 1.0: 0.6, 2.0: 1.0, 4.0: 1.2}, 0.5),
        ([1.0, 2.0, 4.0], {1.0: 1.0, 2.0: 0.3, 4.0: 0.8}, 2.0),
        ([1.0, 2.0, 4.0], {0.25: 0.7, 0.5: 0.6, 1.0: 0.8, 2.0: 1.0, 4.0: 0.8, 8.0: 0.4, 16.0: 0.5}, 8.0),
        ([2.0], {1.0: 0.8, 2.0: 0.5, 4.0: 0.9}, 2.0),
    ],
    ids=["upper", "lower", "interior", "both_initial_edges", "single_candidate"],
)
def test_bandwidth_search_brackets_initial_edge_minima_without_rescoring(
    bandwidth_events: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
    initial: list[float],
    curve: dict[float, float],
    expected_bandwidth: float,
) -> None:
    """Retain the first upturn and explore both initially minimizing edges.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        caplog:
            Fixture capturing diagnostic log messages.
        initial:
            Initial bandwidth candidate grid for adaptive expansion.
        curve:
            Mocked mapping from each evaluated bandwidth to its Brier score.
        expected_bandwidth:
            Bandwidth minimizing the supplied score curve.
    """
    with (
        unittest.mock.patch.object(
            att,
            "_loo_brier_scores",
            side_effect=lambda _doses, _cohorts, bandwidths, _batch: np.array([curve[float(h)] for h in bandwidths]),
        ) as scorer,
        caplog.at_level(logging.WARNING),
    ):
        selected, scores = att.estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS, bandwidths=initial)
    assert selected == expected_bandwidth
    assert not caplog.records
    assert scores["bandwidth"].tolist() == sorted(curve)
    np_testing.assert_allclose(scores["cv_brier"], [curve[h] for h in sorted(curve)], atol=0.0, rtol=0.0)
    np_testing.assert_allclose(scorer.call_args_list[0].args[2], initial, atol=0.0, rtol=0.0)
    evaluated = [float(h) for call in scorer.call_args_list for h in call.args[2]]
    assert len(evaluated) == len(set(evaluated))
    assert set(evaluated) == set(curve)
    assert all(len(call.args[2]) == 1 for call in scorer.call_args_list[1:])


def test_bandwidth_search_stops_after_three_consecutive_numerically_flat_steps(
    bandwidth_events: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reset the plateau count after a material improvement in Brier loss.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    epsilon = np.finfo(float).eps
    curve = {1.0: 1.0, 2.0: 0.8, 4.0: 0.8 - epsilon, 8.0: 0.6}
    curve.update({16.0: 0.6 - epsilon, 32.0: 0.6 - 2 * epsilon, 64.0: 0.6 - 3 * epsilon})
    with (
        unittest.mock.patch.object(
            att,
            "_loo_brier_scores",
            side_effect=lambda _doses, _cohorts, bandwidths, _batch: np.array([curve[float(h)] for h in bandwidths]),
        ),
        caplog.at_level(logging.WARNING),
    ):
        selected, scores = att.estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS, bandwidths=[1.0, 2.0])
    assert selected == 64.0
    assert scores["bandwidth"].tolist() == sorted(curve)
    assert len(caplog.records) == 1
    assert "three numerically flat" in caplog.text


def test_bandwidth_search_limits_each_direction_independently(
    bandwidth_events: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep all evaluated losses when both directions exhaust their budgets.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        caplog:
            Fixture capturing diagnostic log messages.
    """
    curve = {0.25: 0.4, 0.5: 0.6, 1.0: 0.8, 2.0: 0.8, 4.0: 0.5, 8.0: 0.3}
    with (
        unittest.mock.patch.object(
            att,
            "_loo_brier_scores",
            side_effect=lambda _doses, _cohorts, bandwidths, _batch: np.array([curve[float(h)] for h in bandwidths]),
        ),
        caplog.at_level(logging.WARNING),
    ):
        selected, scores = att.estimate_h_cv(
            bandwidth_events,
            **EVENT_MAPPINGS,
            bandwidths=[1.0, 2.0],
            max_expansions=2,
        )
    assert selected == 8.0
    assert scores["bandwidth"].tolist() == sorted(curve)
    assert len(caplog.records) == 1
    assert "lower expansion reached max_expansions=2" in caplog.text
    assert "upper expansion reached max_expansions=2" in caplog.text


@pytest.mark.parametrize(
    ("initial", "curve", "message"),
    [
        ([1e308], {5e307: 0.8, 1e308: 0.3}, "upper expansion reached floating-point"),
        (
            [np.nextafter(0.0, 1.0), 2 * np.nextafter(0.0, 1.0)],
            {np.nextafter(0.0, 1.0): 0.3, 2 * np.nextafter(0.0, 1.0): 0.8},
            "lower expansion reached floating-point",
        ),
        ([5e-154, 1e-153], {5e-154: 0.3, 1e-153: 0.8}, "lower expansion reached kernel arithmetic"),
    ],
    ids=["overflow", "underflow", "kernel_overflow"],
)
def test_bandwidth_search_checks_numerical_limits_before_scoring(
    bandwidth_events: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
    initial: list[float],
    curve: dict[float, float],
    message: str,
) -> None:
    """Return valid evaluated candidates without submitting unusable distances.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        caplog:
            Fixture capturing diagnostic log messages.
        initial:
            Initial bandwidth candidate grid for adaptive expansion.
        curve:
            Mocked mapping from each evaluated bandwidth to its Brier score.
        message:
            Expected validation error or diagnostic message fragment.
    """
    with (
        unittest.mock.patch.object(
            att,
            "_loo_brier_scores",
            side_effect=lambda _doses, _cohorts, bandwidths, _batch: np.array([curve[float(h)] for h in bandwidths]),
        ) as scorer,
        caplog.at_level(logging.WARNING),
    ):
        selected, scores = att.estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS, bandwidths=initial)
    ordered = sorted(curve)
    assert selected == ordered[np.argmin([curve[h] for h in ordered])]
    assert scores["bandwidth"].tolist() == ordered
    assert len(caplog.records) == 1
    assert message in caplog.text
    assert sum(len(call.args[2]) for call in scorer.call_args_list) == len(curve)


@pytest.mark.parametrize("estimator", [att.estimate_h0, att.estimate_h_cv], ids=["h0", "cv"])
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
    """Reject invalid populations before calculating smoothing parameters.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        estimator:
            Bandwidth-selection function under test.
        case:
            Invalid input or sample configuration to exercise.
        message:
            Expected validation error or diagnostic message fragment.
    """
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


@pytest.mark.parametrize("estimator", [att.estimate_h0, att.estimate_h_cv], ids=["h0", "cv"])
@pytest.mark.parametrize("name", [b"firm", 1, ""], ids=["bytes", "number", "empty"])
def test_bandwidth_estimators_do_not_coerce_column_names(
    bandwidth_events: pd.DataFrame,
    estimator: Callable[..., object],
    name: object,
) -> None:
    """Reject non-string or empty mappings consistently across bandwidth selectors.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        estimator:
            Bandwidth-selection function under test.
        name:
            Invalid firm-column mapping that must not be coerced into a string.
    """
    arguments = {**EVENT_MAPPINGS, "firm_col": name}
    with pytest.raises(ValueError, match=r"column|firm_col"):
        estimator(bandwidth_events, **arguments)


def test_bandwidth_cv_requires_multiple_cohorts(bandwidth_events: pd.DataFrame) -> None:
    """Allow Silverman scale estimation but reject a one-category CV target.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
    """
    bandwidth_events["period"] = FIRST
    assert att.estimate_h0(bandwidth_events, **EVENT_MAPPINGS) > 0.0
    with pytest.raises(ValueError, match="cohort"):
        att.estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS, bandwidths=[0.5, 1.0])


@pytest.mark.parametrize("candidates", [[], [0.0, 1.0], [np.nan, 1.0], [1.0, 1.0], [True, 2.0]])
def test_bandwidth_cv_rejects_invalid_candidate_grids(
    bandwidth_events: pd.DataFrame,
    candidates: list[float],
) -> None:
    """Require distinct positive finite bandwidths instead of coercing them.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        candidates:
            Invalid initial bandwidth candidate grid.
    """
    with pytest.raises(ValueError, match="bandwidth"):
        att.estimate_h_cv(bandwidth_events, **EVENT_MAPPINGS, bandwidths=candidates)


@pytest.mark.parametrize("batch_size", [0, 1.5, True])
def test_bandwidth_cv_rejects_invalid_batch_sizes(
    bandwidth_events: pd.DataFrame,
    batch_size: float,
) -> None:
    """Require a positive integer batch size without accepting Boolean values.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        batch_size:
            Number of dose queries processed in each computation batch.
    """
    estimator: Callable[..., object] = att.estimate_h_cv
    with pytest.raises(ValueError, match="batch_size"):
        estimator(bandwidth_events, **EVENT_MAPPINGS, bandwidths=[0.5, 1.0], batch_size=batch_size)


@pytest.mark.parametrize("max_expansions", [0, 1.5, True])
def test_bandwidth_cv_rejects_invalid_expansion_limits(
    bandwidth_events: pd.DataFrame,
    max_expansions: float,
) -> None:
    """Require a positive integer budget for each search direction.

    Args:
        bandwidth_events:
            Fresh raw first-event table with repeated doses.
        max_expansions:
            Invalid bound on adaptive search steps in each direction.
    """
    estimator: Callable[..., object] = att.estimate_h_cv
    with pytest.raises(ValueError, match="max_expansions"):
        estimator(bandwidth_events, **EVENT_MAPPINGS, bandwidths=[0.5, 1.0], max_expansions=max_expansions)
