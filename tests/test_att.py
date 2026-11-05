"""Check continuous-dose estimates against independent numerical comparisons."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from scipy.interpolate import BSpline

import att
from att import ATT, estimate_pi, evaluate_theta


MAPPINGS = {
    "firm_col": "firm",
    "period_col": "period",
    "outcome_col": "sales",
    "dose_col": "hours",
}
FIRST = pd.Timestamp("2024-01-08")
SECOND = pd.Timestamp("2024-01-10")
WINDOW = (-2, 3)


def make_panel(
    first: pd.Timestamp = FIRST, second: pd.Timestamp = SECOND
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        [(f"A{i}", first, 4.0 + i) for i in range(6)]
        + [(f"B{i}", second, 5.0 + i) for i in range(7)],
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
                    sales += (event_time + 1) * (
                        2.0 + 0.5 * dose - 0.03 * dose**2 + 0.004 * dose**3
                    )
            records.append((firm, period, sales))
    return pd.DataFrame(records, columns=["firm", "period", "sales"]), events


def independent_att(
    outcomes: pd.DataFrame,
    events: pd.DataFrame,
    cohort: pd.Timestamp,
    event_time: int,
    treated_firms: tuple,
    control_firms: tuple,
    doses: np.ndarray,
    degree: int = 3,
    knots: tuple = (),
) -> np.ndarray:
    """Compute a separate treated regression and clean-control contrast.

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
            Spline polynomial degree.
        knots:
            Shared internal knot positions.

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
        knots,
        np.repeat(observed_doses.max(), degree + 1),
    ]
    dimension = len(knot_vector) - degree - 1
    basis = BSpline(knot_vector, np.eye(dimension), degree, extrapolate=True)
    design = np.column_stack([np.ones(len(observed_doses)), basis(observed_doses)[:, 1:]])
    coefficients = np.linalg.lstsq(
        design, changes.loc[list(treated_firms)].to_numpy(), rcond=None
    )[0]
    prediction_design = np.column_stack([np.ones(len(doses)), basis(doses)[:, 1:]])
    return prediction_design @ coefficients - changes.loc[list(control_firms)].mean()


class EstimationTests(unittest.TestCase):
    """Verify the stacked fit, saved basis, and dose-specific aggregation."""

    @classmethod
    def setUpClass(cls):
        """Fit the shared numerical fixture while recording the regression call."""
        cls.outcomes, cls.events = make_panel()
        cls.original_outcomes = cls.outcomes.copy(deep=True)
        cls.original_events = cls.events.copy(deep=True)
        with patch.object(att.pf, "feols", wraps=att.pf.feols) as call:
            cls.model = ATT(**MAPPINGS).fit(
                cls.outcomes, cls.events, isolation_window=WINDOW
            )
        cls.regression_count = call.call_count
        cls.regression_args = call.call_args

    def test_one_model_excludes_reference_and_matches_independent_fits(self):
        """Compare every non-reference cell to a separate numerical oracle."""
        self.assertEqual(self.regression_count, 1)
        self.assertEqual(type(self.model.model_).__name__, "Feols")
        self.assertNotIn("split", self.regression_args.kwargs)
        self.assertNotIn("fsplit", self.regression_args.kwargs)
        stack = self.regression_args.kwargs["data"]
        self.assertNotIn(-1, stack["event_time"].unique())
        self.assertFalse(stack.duplicated(["firm_id", "cell"]).any())
        self.assertEqual(self.model.event_times_, list(range(-2, 4)))
        self.assertEqual(self.model.min_cohort_size_, 5)
        doses = np.array([5.5, 7.0, 8.5])
        results = self.model.predict(doses)
        for (cohort, event_time), group in results.groupby(["cohort", "event_time"]):
            with self.subTest(cohort=cohort, event_time=event_time):
                cell = self.model.cell_diagnostics_[(cohort, event_time)]
                self.assertEqual(tuple(cell["control_firms"]), ("C0",))
                self.assertEqual(cell["n_controls"], 1)
                expected = independent_att(
                    self.outcomes,
                    self.events,
                    cohort,
                    event_time,
                    cell["treated_firms"],
                    cell["control_firms"],
                    group["dose"].to_numpy(),
                )
                assert_allclose(group["att"], expected, atol=1e-8, rtol=1e-8)
        assert_frame_equal(self.outcomes, self.original_outcomes)
        assert_frame_equal(self.events, self.original_events)

    def test_placebo_sign_and_reference_normalization(self):
        """Recover a nonzero pre-trend and exact unestimated reference zeros."""
        results = self.model.predict([6.0, 8.0], event_times=[-2, -1, 0])
        placebo = results.loc[results["event_time"] == -2]
        expected = 0.3 * placebo["dose"] + 0.02 * placebo["dose"] ** 2
        assert_allclose(placebo["att"], expected, atol=1e-8)
        self.assertTrue((placebo["estimate_type"] == "placebo").all())
        reference = results.loc[results["event_time"] == -1]
        self.assertTrue((reference["att"] == 0.0).all())
        self.assertTrue((reference["estimate_type"] == "reference").all())
        self.assertFalse(reference["extrapolated"].any())
        for cohort in (FIRST, SECOND):
            key = (cohort, -1)
            self.assertNotIn(key, self.model.effect_columns_)
            self.assertTrue((self.model.cell_coefficients_[key] == 0.0).all())
        self.assertEqual(str(results["cohort"].dtype), "datetime64[ns]")
        self.assertEqual(str(results["period"].dtype), "datetime64[ns]")

    def test_prediction_grid_cannot_redefine_the_fitted_basis(self):
        """Keep predictions unchanged when other requested doses change."""
        single = self.model.predict([7.0])
        expanded = self.model.predict([5.5, 7.0, 8.5])
        expanded = expanded.loc[expanded["dose"] == 7.0].reset_index(drop=True)
        assert_frame_equal(single, expanded)
        with self.assertRaises(ValueError):
            self.model.predict([7.0], event_times=[4])

    def test_aggregation_matches_manual_kernel_weighted_curves(self):
        """Preserve cohort size and dose composition at every requested horizon."""
        doses = [5.5, 8.5]
        bandwidth = 1.3
        theta = evaluate_theta(
            self.model, doses, bandwidth=bandwidth, event_times=[-2, -1, 0, 3]
        )
        components = self.model.predict(doses, event_times=[-2, -1, 0, 3])
        for row in theta.itertuples():
            eligible = self.events.copy()
            eligible["kernel"] = np.exp(-0.5 * ((eligible["hours"] - row.dose) / bandwidth) ** 2)
            weights = eligible.groupby("period")["kernel"].sum()
            weights = weights / weights.sum()
            subset = components.loc[
                (components["event_time"] == row.event_time)
                & (components["dose"] == row.dose)
            ].set_index("cohort")
            expected = sum(weights[cohort] * subset.loc[cohort, "att"] for cohort in weights.index)
            self.assertAlmostEqual(row.theta, expected, places=9)
            self.assertEqual(row.n_cohorts_used, 2)
            self.assertEqual(row.n_firms_used, 13)
            self.assertFalse(row.restricted_target)
            self.assertFalse(row.availability_restricted)

    def test_internal_knots_also_use_the_saved_cell_basis(self):
        """Check an actual piecewise polynomial rather than only knot-free fits."""
        model = ATT(**MAPPINGS, degree=2, num_knots=1).fit(
            self.outcomes, self.events, isolation_window=WINDOW
        )
        assert_allclose(model.knots_, [np.quantile(self.events["hours"], 0.5)])
        results = model.predict([5.5, 7.5, 8.5], event_times=[-2, 2])
        for (cohort, event_time), group in results.groupby(["cohort", "event_time"]):
            cell = model.cell_diagnostics_[(cohort, event_time)]
            expected = independent_att(
                self.outcomes, self.events, cohort, event_time,
                cell["treated_firms"], cell["control_firms"],
                group["dose"].to_numpy(), degree=2, knots=tuple(model.knots_),
            )
            assert_allclose(group["att"], expected, atol=1e-8, rtol=1e-8)

    def test_unused_categorical_firms_do_not_create_phantom_terminations(self):
        """Unused categories must neither block clean controls nor add events."""
        outcomes = self.outcomes.copy()
        events = self.events.copy()
        categories = [*outcomes["firm"].unique(), "unused_firm"]
        outcomes["firm"] = pd.Categorical(outcomes["firm"], categories=categories)
        events["firm"] = pd.Categorical(events["firm"], categories=categories)
        model = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
        self.assertEqual(model.duplicate_diagnostics_["n_dropped_events"], 0)
        self.assertEqual(model.duplicate_diagnostics_["n_affected_firms"], 0)
        self.assertTrue((model.termination_history_["n_terminations"] > 0).all())
        self.assertEqual(set(model.termination_history_["firm_id"]), set(self.events["firm"]))
        assert_frame_equal(
            model.predict([6.0, 8.0], event_times=[-2, 0, 3]),
            self.model.predict([6.0, 8.0], event_times=[-2, 0, 3]),
            check_exact=False,
            atol=1e-8,
            rtol=1e-8,
        )


class SelectionTests(unittest.TestCase):
    """Exercise endpoint-specific samples and the full termination history."""

    def test_missing_endpoints_drop_only_the_affected_comparison(self):
        """Do not require complete trajectories or leak missing firms into weights."""
        outcomes, events = make_panel()
        omitted = {("A0", pd.Timestamp("2024-01-09")),
                   ("A1", pd.Timestamp("2024-01-07")),
                   ("B6", pd.Timestamp("2024-01-11"))}
        keep = [(firm, period) not in omitted for firm, period in outcomes[["firm", "period"]].itertuples(index=False, name=None)]
        outcomes = outcomes.loc[keep].reset_index(drop=True)
        with self.assertLogs(level="WARNING"):
            model = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
        self.assertEqual(model.cell_diagnostics_[(FIRST, 0)]["n_treated"], 5)
        self.assertEqual(model.cell_diagnostics_[(FIRST, 1)]["n_treated"], 4)
        self.assertNotIn((FIRST, 1), model.cell_coefficients_)
        self.assertIn("A0", model.cell_diagnostics_[(FIRST, 2)]["treated_firms"])
        self.assertNotIn("A1", model.cell_diagnostics_[(FIRST, 2)]["treated_firms"])
        with self.assertLogs(level="WARNING"):
            theta = evaluate_theta(model, [7.0], bandwidth=1.0, event_times=[1])
        row = theta.iloc[0]
        self.assertEqual(row["n_firms_design"], 13)
        self.assertEqual(row["n_firms_eligible"], 10)
        self.assertEqual(row["n_firms_used"], 6)
        self.assertTrue(row["availability_restricted"])
        self.assertTrue(row["restricted_target"])
        with self.assertLogs(level="WARNING"):
            prediction = model.predict([7.0], event_times=[1])
        self.assertEqual(set(prediction["cohort"]), {SECOND})
        self.assertAlmostEqual(row["theta"], prediction.iloc[0]["att"], places=10)
        cell = model.cell_diagnostics_[(SECOND, 1)]
        expected = independent_att(
            outcomes, events, SECOND, 1, cell["treated_firms"],
            cell["control_firms"], np.array([7.0]),
        )
        assert_allclose(prediction["att"], expected, atol=1e-8)

    def test_same_period_ties_are_not_promoted_to_later_events(self):
        """Log dropped records while retaining simultaneous-event multiplicity."""
        outcomes, events = make_panel()
        earlier_input = pd.DataFrame([("A0", FIRST + pd.Timedelta(hours=1), 9.0)], columns=events.columns)
        later = pd.DataFrame(
            [("A0", pd.Timestamp("2024-01-18"), 2.0),
             ("B0", pd.Timestamp("2024-01-18"), 3.0)],
            columns=events.columns,
        )
        events = pd.concat([earlier_input, events, later], ignore_index=True)
        with self.assertLogs(level="WARNING") as logs:
            model = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
        self.assertEqual(model.duplicate_diagnostics_["n_dropped_events"], 3)
        self.assertEqual(model.duplicate_diagnostics_["n_affected_firms"], 2)
        self.assertTrue(logs.records)
        selected = model.events_.set_index("firm_id")
        self.assertEqual(selected.loc["A0", "cohort"], FIRST)
        self.assertEqual(selected.loc["A0", "dose"], 9.0)
        self.assertFalse(selected.loc["A0", "design_eligible"])
        self.assertTrue(selected.loc["B0", "design_eligible"])
        history = model.termination_history_.set_index(["firm_id", "cohort"])
        self.assertEqual(history.loc[("A0", FIRST), "n_terminations"], 2)
        self.assertIn(("A0", pd.Timestamp("2024-01-18")), history.index)
        self.assertNotIn("A0", model.cell_diagnostics_[(FIRST, 0)]["treated_firms"])

    def test_later_history_still_blocks_controls_after_first_event_selection(self):
        """An excluded first event does not erase a later conflicting termination."""
        outcomes, events = make_panel()
        clean = outcomes.loc[outcomes["firm"] == "C0"].copy()
        clean["firm"] = "C1"
        outcomes = pd.concat([outcomes, clean], ignore_index=True)
        history = pd.DataFrame(
            [("C0", pd.Timestamp("2024-01-02"), 2.0),
             ("C0", pd.Timestamp("2024-01-09"), 3.0)],
            columns=events.columns,
        )
        with self.assertLogs(level="WARNING"):
            model = ATT(**MAPPINGS).fit(
                outcomes, pd.concat([events, history], ignore_index=True),
                isolation_window=WINDOW,
            )
        selected = model.events_.set_index("firm_id")
        self.assertEqual(selected.loc["C0", "cohort"], pd.Timestamp("2024-01-02"))
        self.assertFalse(selected.loc["C0", "design_eligible"])
        for cohort in (FIRST, SECOND):
            self.assertEqual(set(model.cell_diagnostics_[(cohort, 0)]["control_firms"]), {"C1"})

    def test_missing_controls_cannot_be_hidden_by_aggregation(self):
        """Keep no-control cells required even for the reference normalization."""
        outcomes, events = make_panel()
        outcomes = outcomes.loc[~((outcomes["firm"] == "C0") & (outcomes["period"] == FIRST - pd.Timedelta(days=1)))]
        with self.assertLogs(level="WARNING"):
            model = ATT(**MAPPINGS).fit(outcomes, events, isolation_window=WINDOW)
        for event_time in (-1, 0):
            with self.subTest(event_time=event_time):
                self.assertEqual(model.cell_diagnostics_[(FIRST, event_time)]["status"], "no_controls")
                self.assertIn(FIRST, model.retained_cohorts_[event_time])
                with self.assertRaises(ValueError):
                    evaluate_theta(model, [7.0], bandwidth=1.0, event_times=[event_time])

    def test_reference_only_success_is_not_a_fitted_model(self):
        """Do not invoke the regression solver when only reference zeros survive."""
        outcomes, events = make_panel()
        outcomes = outcomes.loc[outcomes["firm"].str.startswith("A") | (outcomes["firm"] == "C0")]
        events = events.loc[events["firm"].str.startswith("A")]
        outcomes = outcomes.loc[~(outcomes["firm"].str.startswith("A") & (outcomes["period"] == FIRST))]
        model = ATT(**MAPPINGS)
        with patch.object(att.pf, "feols", wraps=att.pf.feols) as call:
            with self.assertRaises((ValueError, RuntimeError)):
                model.fit(outcomes, events, isolation_window=(-1, 0))
        call.assert_not_called()
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict([7.0])

    def test_explicit_minimum_cannot_be_bypassed_by_other_cells(self):
        """A larger requested sample minimum screens the smaller cohort."""
        outcomes, events = make_panel()
        with self.assertLogs(level="WARNING"):
            model = ATT(**MAPPINGS, min_cohort_size=7).fit(outcomes, events, isolation_window=WINDOW)
        self.assertEqual(model.min_cohort_size_, 7)
        self.assertNotIn(FIRST, model.retained_cohorts_[0])
        self.assertIn(SECOND, model.retained_cohorts_[0])
        self.assertIn("A0", set(model.events_["firm_id"]))

    def test_isolation_boundaries_use_overall_outcome_coverage(self):
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
            [("left", pd.Timestamp("2024-01-02"), 3.0),
             ("right", pd.Timestamp("2024-01-22"), 3.0),
             ("outside", pd.Timestamp("2023-12-30"), 3.0),
             ("outside", pd.Timestamp("2024-01-12"), 4.0)],
            columns=events.columns,
        )
        with self.assertLogs(level="WARNING"):
            model = ATT(**MAPPINGS).fit(
                outcomes, pd.concat([events, extra_events], ignore_index=True),
                isolation_window=WINDOW,
            )
        selected = model.events_.set_index("firm_id")
        self.assertTrue(selected.loc["A0", "design_eligible"])
        self.assertTrue(selected.loc["B0", "design_eligible"])
        self.assertFalse(selected.loc[["left", "right", "outside"], "design_eligible"].any())
        self.assertEqual(selected.loc["outside", "cohort"], pd.Timestamp("2023-12-30"))
        self.assertEqual(model.origin_period_, pd.Timestamp("2024-01-01"))
        self.assertEqual(model.last_period_, pd.Timestamp("2024-01-24"))


class InputAndKernelTests(unittest.TestCase):
    """Check public contracts that protect target construction and reuse."""

    def test_column_arguments_are_required(self):
        """Never infer a missing input-column mapping from conventional labels."""
        for missing in MAPPINGS:
            with self.subTest(missing=missing):
                with self.assertRaises(TypeError):
                    ATT(**{key: value for key, value in MAPPINGS.items() if key != missing})

    def test_normalization_renaming_and_input_order_preserve_estimates(self):
        """Normalize times while respecting every configured input-column name."""
        outcomes, events = make_panel()
        outcomes["period"] += pd.Timedelta(hours=13)
        events["period"] += pd.Timedelta(hours=23)
        renamed_outcomes = outcomes.rename(columns={"firm": "unit", "period": "when", "sales": "value"}).sample(frac=1.0, random_state=1)
        renamed_events = events.rename(columns={"firm": "unit", "period": "when", "hours": "value"}).sample(frac=1.0, random_state=2)
        model = ATT(firm_col="unit", period_col="when", outcome_col="value", dose_col="value", min_cohort_size=1)
        model.fit(renamed_outcomes, renamed_events, isolation_window=WINDOW)
        self.assertEqual(model.min_cohort_size_, 5)
        result = model.predict([6.0], event_times=[-2, 0])
        placebo = result.loc[result["event_time"] == -2, "att"]
        assert_allclose(placebo, 0.3 * 6 + 0.02 * 6**2, atol=1e-8)
        post = result.loc[result["event_time"] == 0, "att"]
        assert_allclose(post, 2 + 0.5 * 6 - 0.03 * 6**2 + 0.004 * 6**3, atol=1e-8)
        self.assertEqual(model.origin_period_, pd.Timestamp("2024-01-01"))
        self.assertEqual(model.last_period_, pd.Timestamp("2024-01-24"))

    def test_invalid_periods_present_missing_outcomes_and_normalized_duplicates(self):
        """Distinguish missing rows from invalid values and invalid period types."""
        outcomes, events = make_panel()
        cases = []
        text_periods = outcomes.copy()
        text_periods["period"] = text_periods["period"].astype(str)
        cases.append((text_periods, events))
        aware_periods = events.copy()
        aware_periods["period"] = aware_periods["period"].dt.tz_localize("UTC")
        cases.append((outcomes, aware_periods))
        missing_sales = outcomes.copy()
        missing_sales.loc[0, "sales"] = np.nan
        cases.append((missing_sales, events))
        duplicate = outcomes.iloc[[0]].copy()
        duplicate["period"] += pd.Timedelta(hours=1)
        cases.append((pd.concat([outcomes, duplicate], ignore_index=True), events))
        cases.append((outcomes.drop(columns="firm"), events))
        for index, (bad_outcomes, bad_events) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError)):
                    ATT(**MAPPINGS).fit(bad_outcomes, bad_events, isolation_window=WINDOW)
        with self.assertRaises(ValueError):
            ATT(**MAPPINGS).fit(outcomes, events, isolation_window=(2, 3))

    def test_failed_refit_invalidates_prior_predictions(self):
        """An earlier valid model must not survive a failed replacement fit."""
        outcomes, events = make_panel()
        model = ATT(**MAPPINGS)
        self.assertIs(model.fit(outcomes, events, isolation_window=WINDOW), model)
        previous_regression = model.model_
        model.fit(outcomes, events, isolation_window=(-1, 0))
        self.assertEqual(model.event_times_, [-1, 0])
        self.assertIsNot(model.model_, previous_regression)
        with self.assertRaises(ValueError):
            model.predict([7.0], event_times=[3])
        with self.assertRaises(ValueError):
            model.fit(outcomes.drop(columns="sales"), events, isolation_window=WINDOW)
        with self.assertRaises((ValueError, RuntimeError)):
            model.predict([7.0])

    def test_kernel_weights_count_only_the_supplied_firms_once(self):
        """Match direct Gaussian sums and reject ambiguous eligibility inputs."""
        events = pd.DataFrame(
            [("a", FIRST, 4.0), ("b", FIRST, 5.0),
             ("c", SECOND, 9.0), ("excluded", FIRST, 9.0)],
            columns=["firm_id", "cohort", "dose"],
        )
        eligible = ["a", "b", "c"]
        doses = [4.5, 8.5]
        weights = estimate_pi(events, doses, eligible_firms=eligible, bandwidth=1.2)
        selected = events.set_index("firm_id").loc[eligible]
        for dose in doses:
            gaussian = np.exp(-0.5 * ((selected["dose"] - dose) / 1.2) ** 2)
            expected = gaussian.groupby(selected["cohort"]).sum() / gaussian.sum()
            actual = weights.loc[weights["dose"] == dose].set_index("cohort")["weight"]
            assert_allclose(actual.reindex(expected.index), expected, atol=1e-14)
            self.assertAlmostEqual(actual.sum(), 1.0)
        for eligible_firms, bandwidth in [(["a", "a"], 1.0), (["unknown"], 1.0), (eligible, 0.0), (eligible, np.nan)]:
            with self.subTest(eligible=eligible_firms, bandwidth=bandwidth):
                with self.assertRaises(ValueError):
                    estimate_pi(events, doses, eligible_firms=eligible_firms, bandwidth=bandwidth)
        for bad_dose in ("4", True, 4.0 + 1.0j):
            with self.subTest(bad_dose=bad_dose):
                invalid = events.copy()
                invalid["dose"] = invalid["dose"].astype(object)
                invalid.loc[0, "dose"] = bad_dose
                with self.assertRaises(ValueError):
                    estimate_pi(invalid, doses, eligible_firms=eligible, bandwidth=1.0)


if __name__ == "__main__":
    unittest.main()
