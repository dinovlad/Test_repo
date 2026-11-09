"""Check firm resampling against explicit duplicated-observation calculations."""

import copy
import logging
import unittest.mock
from typing import Any, cast

import numpy as np
import numpy.testing as np_testing
import numpy.typing as npt
import pandas as pd
import pandas.testing as pd_testing
import pytest

import labor_cost_optimization.att as att


Panel = tuple[pd.DataFrame, pd.DataFrame]
Counts = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
DOSES = np.asarray([5.5, 7.5, 10.5])
BANDWIDTH = 0.9


def _counts(model: att.ATT, changes: dict[str, int] | None = None) -> Counts:
    """Create frequency counts while assigning spare copies to the unused firm.

    Args:
        model:
            Fitted fixture retaining the complete outcome-firm universe.
        changes:
            Counts overriding the initial one-copy-per-firm draw.

    Returns:
        Nonnegative counts whose total equals the original number of firms.
    """
    firms = model.bootstrap_firms_
    counts = np.ones(len(firms), dtype=np.int64)
    for firm, count in (changes or {}).items():
        counts[int(firms.get_loc(firm))] = count
    counts[int(firms.get_loc("unused"))] += len(firms) - int(counts.sum())
    assert (counts >= 0).all()
    assert counts.sum() == len(firms)
    return counts


def _force_counts(
    monkeypatch: pytest.MonkeyPatch,
    rows: Counts,
) -> list[tuple[int, int]]:
    """Supply predetermined firm draws without changing estimator arithmetic.

    Args:
        monkeypatch:
            Fixture restoring the sampling hook after the test.
        rows:
            Complete frequency-count rows for the requested replications.

    Returns:
        Recorded sampler requests containing firm count and requested batch size.
    """
    requests: list[tuple[int, int]] = []
    offset = 0

    def sample(rng: np.random.Generator, n_firms: int, size: int) -> Counts:
        """Return the next batch of predetermined multinomial counts.

        Args:
            rng:
                Generator supplied by the bootstrap.
            n_firms:
                Size of the original outcome-firm population.
            size:
                Number of requested replication rows.

        Returns:
            A fresh array containing the requested frequency rows.
        """
        nonlocal offset
        assert isinstance(rng, np.random.Generator)
        assert n_firms == rows.shape[1]
        assert offset + size <= len(rows)
        requests.append((n_firms, size))
        result = rows[offset : offset + size].copy()
        offset += size
        return result

    monkeypatch.setattr(att, "_bootstrap_counts", sample)
    return requests


def _expanded_theta(
    model: att.ATT,
    outcomes: pd.DataFrame,
    events: pd.DataFrame,
    counts: Counts,
    doses: FloatArray,
    event_time: int,
    bandwidth: float,
) -> FloatArray:
    """Expand repeated firms and independently fit raw-dose polynomial curves.

    Args:
        model:
            Fitted comparison memberships, without using its prediction methods.
        outcomes:
            Original outcome rows.
        events:
            First-event fixture with observed doses.
        counts:
            Global firm sampling multiplicities.
        doses:
            Requested dose grid.
        event_time:
            Horizon to evaluate.
        bandwidth:
            Gaussian bandwidth in dose units.

    Returns:
        An independently aggregated curve, or missing values when its required
        regression or aggregation population is undefined.
    """
    multiplicity = dict(zip(model.bootstrap_firms_, counts))
    observed = outcomes.pivot(index="firm", columns="period", values="sales")
    original_doses = events.set_index("firm")["hours"]
    groups: list[tuple[pd.Timestamp, list[str]]] = []
    for cohort in model.retained_cohorts_[event_time]:
        cell = model.cell_diagnostics_[(cohort, event_time)]
        expanded = [firm for firm in cell["treated_firms"] for _ in range(int(multiplicity[firm]))]
        if expanded:
            groups.append((cohort, expanded))
    if not groups:
        return np.full(len(doses), np.nan)
    if event_time == -1:
        return np.zeros(len(doses))
    curves: list[FloatArray] = []
    expanded_doses: list[float] = []
    cohort_labels: list[int] = []
    for group, (cohort, treated) in enumerate(groups):
        cell = model.cell_diagnostics_[(cohort, event_time)]
        controls = [firm for firm in cell["control_firms"] for _ in range(int(multiplicity[firm]))]
        if not controls:
            return np.full(len(doses), np.nan)
        baseline = cohort - pd.Timedelta(days=1)
        endpoint = cohort + pd.Timedelta(days=event_time)
        changes = observed[endpoint] - observed[baseline]
        raw_doses = original_doses.reindex(treated).to_numpy(dtype=float)
        design = np.polynomial.polynomial.polyvander(raw_doses, model.degree)
        response = changes.reindex(treated).to_numpy(dtype=float)
        coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        if rank < model.degree + 1:
            return np.full(len(doses), np.nan)
        control_mean = float(changes.reindex(controls).mean())
        curves.append(np.polynomial.polynomial.polyval(doses, coefficients) - control_mean)
        expanded_doses.extend(raw_doses)
        cohort_labels.extend([group] * len(raw_doses))
    distance = (np.asarray(expanded_doses)[:, None] - doses) / bandwidth
    log_kernel = -0.5 * distance**2
    kernel = np.exp(log_kernel - log_kernel.max(axis=0))
    denominator = kernel.sum(axis=0)
    result = np.zeros(len(doses))
    for group, curve in enumerate(curves):
        numerator = kernel[np.asarray(cohort_labels) == group].sum(axis=0)
        result += numerator / denominator * curve
    return result


def test_bootstrap_matches_expanded_firms_and_reuses_global_counts(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match duplicated rows while preserving shared controls and fixed bases.

    Args:
        bootstrap_panel:
            Heterogeneous original observations and event metadata.
        bootstrap_model:
            Fitted cubic comparison curves.
        monkeypatch:
            Fixture providing exact frequency draws and blocking refits.
    """
    outcomes, events = bootstrap_panel
    rows = np.vstack(
        [
            _counts(bootstrap_model),
            _counts(bootstrap_model, {"A0": 2, "A1": 0, "C0": 2, "C1": 0}),
            _counts(bootstrap_model, {"B0": 0, "B2": 2, "C1": 0, "C2": 2}),
        ],
    )
    requests = _force_counts(monkeypatch, rows)
    outcome_copy = outcomes.copy(deep=True)
    event_copy = events.copy(deep=True)
    coefficients = copy.deepcopy(bootstrap_model.cell_coefficients_)
    bases = copy.deepcopy(bootstrap_model.bases_)
    diagnostics_before = copy.deepcopy(bootstrap_model.cell_diagnostics_)
    original_models = bootstrap_model.models_.copy()
    original_point = att.evaluate_theta(bootstrap_model, DOSES, bandwidth=BANDWIDTH, event_times=[0, 1])
    monkeypatch.setattr(bootstrap_model, "fit", unittest.mock.Mock(side_effect=AssertionError("Unexpected refit")))
    monkeypatch.setattr(att.pf, "feols", unittest.mock.Mock(side_effect=AssertionError("Unexpected PyFixest fit")))
    attribute_names = set(vars(bootstrap_model))

    summary, draws, diagnostics = att.bootstrap_theta(
        bootstrap_model,
        DOSES,
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        event_times=[0, 1],
        n_bootstrap=len(rows),
    )

    assert sum(size for _, size in requests) == len(rows)
    assert {n_firms for n_firms, _ in requests} == {outcomes["firm"].nunique()}
    assert "unused" in bootstrap_model.bootstrap_firms_
    assert diagnostics.empty
    assert draws["success"].all()
    assert set(draws["replicate"]) == {1, 2, 3}
    pd_testing.assert_frame_equal(summary[original_point.columns], original_point)
    for replication, counts in enumerate(rows, start=1):
        for horizon in (0, 1):
            expected = _expanded_theta(bootstrap_model, outcomes, events, counts, DOSES, horizon, BANDWIDTH)
            actual = draws.loc[draws["replicate"].eq(replication) & draws["event_time"].eq(horizon), "theta"]
            np_testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
    for horizon in (0, 1):
        values = draws.loc[draws["event_time"].eq(horizon)].pivot(index="replicate", columns="dose", values="theta")
        result = summary.loc[summary["event_time"].eq(horizon)]
        np_testing.assert_allclose(result["standard_error"], values.std(axis=0, ddof=1))
        np_testing.assert_allclose(result["ci_lower"], np.quantile(values, 0.025, axis=0))
        np_testing.assert_allclose(result["ci_upper"], np.quantile(values, 0.975, axis=0))
    assert bootstrap_model.bases_ == bases
    assert bootstrap_model.cell_diagnostics_ == diagnostics_before
    assert bootstrap_model.models_ == original_models
    assert set(vars(bootstrap_model)) == attribute_names
    pd_testing.assert_frame_equal(outcomes, outcome_copy)
    pd_testing.assert_frame_equal(events, event_copy)
    for cell, original_coefficients in coefficients.items():
        np_testing.assert_array_equal(bootstrap_model.cell_coefficients_[cell], original_coefficients)


def test_all_one_counts_reproduce_original_aggregation(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce each fitted point estimate with one copy of every original firm.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Fitted comparison model.
        monkeypatch:
            Fixture replacing random draws with all-one counts.
    """
    outcomes, _ = bootstrap_panel
    _force_counts(monkeypatch, np.vstack([_counts(bootstrap_model)] * 2))
    summary, draws, diagnostics = att.bootstrap_theta(
        bootstrap_model,
        DOSES,
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        n_bootstrap=2,
    )
    for replication in (1, 2):
        actual = draws.loc[draws["replicate"].eq(replication), "theta"]
        np_testing.assert_allclose(actual, summary["theta"], atol=1e-9, rtol=1e-9)
    np_testing.assert_array_equal(summary["standard_error"], 0.0)
    assert diagnostics.empty
    assert not summary["conditional_on_success"].any()
    assert summary["inference_status"].eq("ok").all()


def test_failed_horizons_use_successful_draws_without_rescreening_or_redrawing(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retain failed draws and compute qualified intervals at each horizon.

    Args:
        bootstrap_panel:
            Original panel with one horizon-specific missing outcome.
        bootstrap_model:
            Fitted model retaining both requested comparisons.
        monkeypatch:
            Fixture supplying a draw with horizon-specific rank failure.
        caplog:
            Fixture recording aggregated bootstrap diagnostics.
    """
    outcomes, events = bootstrap_panel
    rows = np.vstack(
        [
            _counts(bootstrap_model),
            _counts(bootstrap_model, {"A0": 2, "A1": 0}),
            _counts(bootstrap_model, {"A3": 0, "A4": 0}),
        ],
    )
    requests = _force_counts(monkeypatch, rows)
    with caplog.at_level(logging.WARNING):
        summary, draws, diagnostics = att.bootstrap_theta(
            bootstrap_model,
            DOSES,
            outcomes=outcomes,
            bandwidth=BANDWIDTH,
            event_times=[-1, 0, 1],
            n_bootstrap=3,
            confidence_level=0.8,
        )
    assert sum(size for _, size in requests) == 3
    assert len(caplog.records) == 1
    failed = draws.loc[draws["replicate"].eq(3) & draws["event_time"].eq(1)]
    assert failed["theta"].isna().all()
    assert not failed["success"].any()
    assert draws.loc[draws["event_time"].isin([-1, 0]), "success"].all()
    expected = np.vstack(
        [_expanded_theta(bootstrap_model, outcomes, events, row, DOSES, 1, BANDWIDTH) for row in rows[:2]],
    )
    result = summary.loc[summary["event_time"].eq(1)]
    assert result["n_bootstrap"].eq(3).all()
    assert result["n_success"].eq(2).all()
    assert result["n_failed"].eq(1).all()
    np_testing.assert_allclose(result["success_rate"], 2 / 3)
    assert result["conditional_on_success"].all()
    assert result["inference_status"].eq("successful_draws_only").all()
    np_testing.assert_allclose(result["standard_error"], expected.std(axis=0, ddof=1), atol=1e-8)
    np_testing.assert_allclose(result["ci_lower"], np.quantile(expected, 0.1, axis=0), atol=1e-8)
    np_testing.assert_allclose(result["ci_upper"], np.quantile(expected, 0.9, axis=0), atol=1e-8)
    assert diagnostics["reason"].tolist() == ["rank_deficient"]
    assert diagnostics["event_time"].tolist() == [1]
    assert diagnostics["n_failures"].tolist() == [1]


@pytest.mark.parametrize("n_success", [0, 1])
def test_insufficient_successful_draws_keep_points_and_missing_intervals(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
    n_success: int,
) -> None:
    """Return unavailable inference without replacing failed replications.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Fitted cubic model.
        monkeypatch:
            Fixture supplying too few successful draws.
        n_success:
            Number of successful draws among the two attempted replications.
    """
    outcomes, _ = bootstrap_panel
    failure = _counts(bootstrap_model, {"C0": 0, "C1": 0, "C2": 0})
    rows = np.vstack([_counts(bootstrap_model) if n_success else failure, failure])
    _force_counts(monkeypatch, rows)
    summary, draws, diagnostics = att.bootstrap_theta(
        bootstrap_model,
        DOSES,
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        event_times=[0],
        n_bootstrap=2,
    )
    assert summary["theta"].notna().all()
    assert summary[["standard_error", "ci_lower", "ci_upper"]].isna().all().all()
    assert summary["n_success"].eq(n_success).all()
    assert summary["n_failed"].eq(2 - n_success).all()
    assert summary["inference_status"].eq("insufficient_successful_draws").all()
    assert draws["success"].sum() == n_success * len(DOSES)
    assert diagnostics["reason"].eq("no_controls").all()


def test_zero_mass_cohorts_and_empty_aggregation_are_distinct(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow a zero-mass cohort but reject an entirely empty treated population.

    Args:
        bootstrap_panel:
            Original heterogeneous outcomes and first events.
        bootstrap_model:
            Fitted model with two aggregation cohorts.
        monkeypatch:
            Fixture supplying empty-cohort and empty-population draws.
    """
    outcomes, events = bootstrap_panel
    empty_first = {f"A{index}": 0 for index in range(6)}
    empty_all = {**empty_first, **{f"B{index}": 0 for index in range(7)}}
    rows = np.vstack([_counts(bootstrap_model, empty_first), _counts(bootstrap_model, empty_all)])
    _force_counts(monkeypatch, rows)
    _, draws, diagnostics = att.bootstrap_theta(
        bootstrap_model,
        DOSES,
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        event_times=[-1, 0],
        n_bootstrap=2,
    )
    first = draws.loc[draws["replicate"].eq(1)]
    assert first["success"].all()
    expected = _expanded_theta(bootstrap_model, outcomes, events, rows[0], DOSES, 0, BANDWIDTH)
    np_testing.assert_allclose(first.loc[first["event_time"].eq(0), "theta"], expected, atol=1e-8)
    assert not draws.loc[draws["replicate"].eq(2), "success"].any()
    assert diagnostics["reason"].eq("no_treated_firms").all()
    assert diagnostics["cohort"].isna().all()
    assert diagnostics["n_failures"].eq(1).all()


def test_reference_needs_no_sampled_controls_or_polynomial_rank(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the normalization zero whenever any retained treated firm is drawn.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Validated fitted reference comparisons.
        monkeypatch:
            Fixture removing regression support and blocking linear solves.
    """
    outcomes, _ = bootstrap_panel
    changes = dict.fromkeys([f"A{i}" for i in range(1, 6)] + [f"B{i}" for i in range(7)] + ["C0", "C1", "C2"], 0)
    _force_counts(monkeypatch, np.vstack([_counts(bootstrap_model, changes)] * 2))
    monkeypatch.setattr(np.linalg, "lstsq", unittest.mock.Mock(side_effect=AssertionError("Unexpected reference fit")))
    summary, draws, diagnostics = att.bootstrap_theta(
        bootstrap_model,
        DOSES,
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        event_times=[-1],
        n_bootstrap=2,
    )
    np_testing.assert_array_equal(summary[["theta", "standard_error", "ci_lower", "ci_upper"]], 0.0)
    assert draws["success"].all()
    assert diagnostics.empty


def test_seed_and_requested_grid_do_not_change_firm_draws(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
) -> None:
    """Reuse seeded global samples across dose grids and requested horizons.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Fitted model with one common original firm universe.
    """
    outcomes, _ = bootstrap_panel
    arguments: dict[str, Any] = {
        "outcomes": outcomes,
        "bandwidth": BANDWIDTH,
        "n_bootstrap": 37,
        "random_state": 76,
    }
    full = att.bootstrap_theta(bootstrap_model, DOSES, **arguments)
    repeated = att.bootstrap_theta(bootstrap_model, DOSES, **arguments)
    selected = att.bootstrap_theta(bootstrap_model, DOSES[:2], event_times=[0], **arguments)
    for first, second in zip(full, repeated):
        pd_testing.assert_frame_equal(first, second)
    expected = full[1].loc[full[1]["event_time"].eq(0) & full[1]["dose"].isin(DOSES[:2])].reset_index(drop=True)
    pd_testing.assert_frame_equal(selected[1], expected)


def test_panel_fingerprint_accepts_reordering_and_ignored_time_components(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
) -> None:
    """Validate normalized panel content independently of row and column order.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Fitted model retaining a normalized whole-panel fingerprint.
    """
    outcomes, _ = bootstrap_panel
    transformed = outcomes.sample(frac=1, random_state=17).copy()
    transformed["period"] += pd.Timedelta(hours=13, minutes=4)
    transformed["ignored"] = "extra column"
    original = att.bootstrap_theta(
        bootstrap_model,
        DOSES,
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        n_bootstrap=5,
        random_state=8,
    )
    reordered = att.bootstrap_theta(
        bootstrap_model,
        DOSES,
        outcomes=transformed[["ignored", "sales", "period", "firm"]],
        bandwidth=BANDWIDTH,
        n_bootstrap=5,
        random_state=8,
    )
    for first, second in zip(original, reordered):
        pd_testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("change", ["balanced_controls", "unused_value", "missing_unused", "extra_firm", "missing_row"])
def test_panel_fingerprint_rejects_changes_including_unused_observations(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """Reject panel changes that an unweighted coefficient comparison can miss.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Fitted estimator retaining the full original panel identity.
        monkeypatch:
            Fixture proving validation finishes before any sampling.
        change:
            Modification to the supplied original panel.
    """
    outcomes, _ = bootstrap_panel
    changed = outcomes.copy()
    if change == "balanced_controls":
        endpoint = changed["period"].eq(pd.Timestamp("2024-01-08"))
        changed.loc[changed["firm"].eq("C1") & endpoint, "sales"] += 0.25
        changed.loc[changed["firm"].eq("C2") & endpoint, "sales"] -= 0.25
    elif change == "unused_value":
        changed.loc[changed["firm"].eq("unused"), "sales"] += 1.0
    elif change == "missing_unused":
        changed = changed.loc[~changed["firm"].eq("unused")]
    elif change == "extra_firm":
        extra = changed.iloc[[0]].copy()
        extra["firm"] = "new firm"
        changed = pd.concat([changed, extra], ignore_index=True)
    elif change == "missing_row":
        changed = changed.iloc[1:]
    else:
        raise AssertionError(f"Unknown fixture change: {change}")
    monkeypatch.setattr(att, "_bootstrap_counts", unittest.mock.Mock(side_effect=AssertionError("Unexpected sampling")))
    with pytest.raises(ValueError, match=r"outcome|panel|firm|original"):
        att.bootstrap_theta(bootstrap_model, DOSES, outcomes=changed, bandwidth=BANDWIDTH, n_bootstrap=2)


def test_numerical_solver_errors_are_not_recorded_as_sampling_failures(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise genuine numerical errors instead of silently conditioning on them.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Fully estimable fitted model.
        monkeypatch:
            Fixture injecting a genuine failure of the linear solver.
    """
    outcomes, _ = bootstrap_panel
    _force_counts(monkeypatch, np.vstack([_counts(bootstrap_model)] * 2))
    monkeypatch.setattr(np.linalg, "lstsq", unittest.mock.Mock(side_effect=np.linalg.LinAlgError("Injected failure")))
    with pytest.raises(RuntimeError):
        att.bootstrap_theta(
            bootstrap_model,
            DOSES,
            outcomes=outcomes,
            bandwidth=BANDWIDTH,
            event_times=[0],
            n_bootstrap=2,
        )


def test_kernel_preserves_frequency_counts_under_extreme_common_log_offsets(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve cohort multiplicities when the raw Gaussian logs are enormous.

    Args:
        bootstrap_panel:
            Heterogeneous cohort outcomes and doses.
        bootstrap_model:
            Fitted original cohort curves.
        monkeypatch:
            Fixture supplying equal and unequal nearest-dose multiplicities.
    """
    outcomes, events = bootstrap_panel
    rows = np.vstack([_counts(bootstrap_model), _counts(bootstrap_model, {"A0": 2, "A2": 0})])
    _force_counts(monkeypatch, rows)
    doses = np.asarray([4.5])
    bandwidth = 1e-150
    summary, draws, diagnostics = att.bootstrap_theta(
        bootstrap_model,
        doses,
        outcomes=outcomes,
        bandwidth=bandwidth,
        event_times=[0],
        n_bootstrap=2,
    )
    assert diagnostics.empty
    assert draws["success"].all()
    np_testing.assert_allclose(draws.loc[draws["replicate"].eq(1), "theta"], summary["theta"], atol=1e-9)
    for replication, counts in enumerate(rows, start=1):
        expected = _expanded_theta(bootstrap_model, outcomes, events, counts, doses, 0, bandwidth)
        np_testing.assert_allclose(draws.loc[draws["replicate"].eq(replication), "theta"], expected, atol=1e-8)


def test_numerical_kernel_underflow_cannot_hide_positive_mass_rank_failure(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep required cohorts even when their Gaussian weights underflow locally.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Fitted model whose first cohort will lose dose support.
        monkeypatch:
            Fixture supplying one rank-deficient draw and one complete draw.
    """
    outcomes, _ = bootstrap_panel
    failure = _counts(bootstrap_model, {"A3": 0, "A4": 0, "A5": 0})
    _force_counts(monkeypatch, np.vstack([failure, _counts(bootstrap_model)]))
    _, draws, diagnostics = att.bootstrap_theta(
        bootstrap_model,
        [10.5],
        outcomes=outcomes,
        bandwidth=1e-150,
        event_times=[0],
        n_bootstrap=2,
    )
    assert not draws.loc[draws["replicate"].eq(1), "success"].any()
    assert draws.loc[draws["replicate"].eq(1), "theta"].isna().all()
    assert diagnostics["reason"].tolist() == ["rank_deficient"]


def test_zero_curves_allow_large_finite_doses_without_basis_power_overflow(
    bootstrap_panel: Panel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict zero curves stably where explicit query powers would overflow.

    Args:
        bootstrap_panel:
            Original event schedule and outcome panel to replace by zeros.
        monkeypatch:
            Fixture supplying one copy of each firm in both replications.
    """
    outcomes, events = bootstrap_panel
    outcomes = outcomes.copy()
    outcomes["sales"] = 0.0
    model = att.ATT(firm_col="firm", period_col="period", outcome_col="sales", dose_col="hours").fit(
        outcomes,
        events,
        isolation_window=(-1, 2),
    )
    _force_counts(monkeypatch, np.vstack([_counts(model)] * 2))
    summary, draws, diagnostics = att.bootstrap_theta(
        model,
        [1e154],
        outcomes=outcomes,
        bandwidth=1e154,
        event_times=[0],
        n_bootstrap=2,
    )
    assert diagnostics.empty
    assert draws["success"].all()
    np_testing.assert_array_equal(summary[["theta", "standard_error", "ci_lower", "ci_upper"]], 0.0)
    np_testing.assert_array_equal(draws["theta"], 0.0)


def test_placebo_bootstrap_uses_the_same_fixed_reference_period(
    bootstrap_panel: Panel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the sign and baseline of the pre-treatment contrast.

    Args:
        bootstrap_panel:
            Original heterogeneous outcomes and first-event periods.
        monkeypatch:
            Fixture supplying nonuniform counts for an independent contrast.
    """
    outcomes, events = bootstrap_panel
    model = att.ATT(firm_col="firm", period_col="period", outcome_col="sales", dose_col="hours").fit(
        outcomes,
        events,
        isolation_window=(-2, 2),
    )
    rows = np.vstack([_counts(model), _counts(model, {"B0": 2, "B1": 0})])
    _force_counts(monkeypatch, rows)
    summary, draws, diagnostics = att.bootstrap_theta(
        model,
        DOSES,
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        event_times=[-2],
        n_bootstrap=2,
    )
    assert summary["estimate_type"].eq("placebo").all()
    assert diagnostics.empty
    for replication, counts in enumerate(rows, start=1):
        expected = _expanded_theta(model, outcomes, events, counts, DOSES, -2, BANDWIDTH)
        np_testing.assert_allclose(draws.loc[draws["replicate"].eq(replication), "theta"], expected, atol=1e-8)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_bootstrap", 1),
        ("n_bootstrap", True),
        ("n_bootstrap", np.bool_(True)),
        ("n_bootstrap", 2.5),
        ("confidence_level", 0.0),
        ("confidence_level", 1.0),
        ("confidence_level", float("nan")),
        ("confidence_level", True),
        ("confidence_level", np.bool_(True)),
        ("random_state", -1),
        ("random_state", True),
        ("random_state", np.bool_(True)),
        ("bandwidth", 0.0),
        ("bandwidth", np.bool_(True)),
        ("bandwidth", "0.9"),
    ],
)
def test_bootstrap_rejects_invalid_configuration_before_sampling(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
) -> None:
    """Reject invalid replication, confidence, bandwidth, and seed settings.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Successfully fitted estimator.
        monkeypatch:
            Fixture proving invalid requests do not start sampling.
        name:
            Invalid bootstrap argument name.
        value:
            Invalid argument value.
    """
    outcomes, _ = bootstrap_panel
    options: dict[str, Any] = {"outcomes": outcomes, "bandwidth": BANDWIDTH, "n_bootstrap": 2, name: value}
    monkeypatch.setattr(att, "_bootstrap_counts", unittest.mock.Mock(side_effect=AssertionError("Unexpected sampling")))
    with pytest.raises(ValueError, match=name):
        att.bootstrap_theta(bootstrap_model, DOSES, **options)


def test_bootstrap_accepts_numpy_options_and_single_pass_grids(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
) -> None:
    """Preserve seeded bootstrap results after validating NumPy and iterator inputs.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            Successfully fitted estimator with bootstrap metadata.
    """
    outcomes, _ = bootstrap_panel
    expected = att.bootstrap_theta(
        bootstrap_model,
        DOSES.tolist(),
        outcomes=outcomes,
        bandwidth=BANDWIDTH,
        event_times=[-1, 0],
        n_bootstrap=3,
        random_state=23,
    )
    options: dict[str, Any] = {
        "outcomes": outcomes,
        "bandwidth": np.float64(BANDWIDTH),
        "event_times": iter(np.asarray([-1, 0], dtype=np.int64)),
        "n_bootstrap": np.int64(3),
        "confidence_level": np.float64(0.95),
        "random_state": np.int64(23),
    }
    actual = att.bootstrap_theta(bootstrap_model, iter(DOSES), **options)
    for observed, reference in zip(actual, expected, strict=True):
        pd_testing.assert_frame_equal(observed, reference)


def test_unfitted_and_legacy_models_require_a_successful_current_fit(
    bootstrap_panel: Panel,
    bootstrap_model: att.ATT,
) -> None:
    """Require fitted panel metadata and invalidate it after a failed refit.

    Args:
        bootstrap_panel:
            Original outcome and event tables.
        bootstrap_model:
            A model initially carrying valid bootstrap metadata.
    """
    outcomes, events = bootstrap_panel
    unfitted = att.ATT(firm_col="firm", period_col="period", outcome_col="sales", dose_col="hours")
    with pytest.raises(RuntimeError):
        att.bootstrap_theta(unfitted, DOSES, outcomes=outcomes, bandwidth=BANDWIDTH, n_bootstrap=2)
    legacy = copy.copy(bootstrap_model)
    del legacy.outcome_fingerprint_
    with pytest.raises((ValueError, RuntimeError), match=r"[Rr]efit|fit"):
        att.bootstrap_theta(legacy, DOSES, outcomes=outcomes, bandwidth=BANDWIDTH, n_bootstrap=2)
    with pytest.raises(ValueError, match="isolation_window"):
        bootstrap_model.fit(outcomes, events, isolation_window=cast(tuple[int, int], (0, 2)))
    with pytest.raises(RuntimeError):
        att.bootstrap_theta(bootstrap_model, DOSES, outcomes=outcomes, bandwidth=BANDWIDTH, n_bootstrap=2)
