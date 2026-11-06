"""Estimate unpooled continuous-dose ATT curves with separate polynomial models.

Periods have a fixed daily frequency. The estimator reports point estimates;
negative horizons are placebo contrasts and horizon -1 is a zero reference.
See att.md for the identifying assumptions and selected-population estimand.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from numbers import Integral
from typing import Any, Self, cast

import numpy as np
import pandas as pd
import pyfixest as pf
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.special import logsumexp


__all__ = ["ATT", "estimate_h0", "estimate_h_cv", "estimate_pi", "evaluate_theta"]

FloatArray = NDArray[np.float64]
Cell = tuple[pd.Timestamp, int]
_PERIOD = pd.Timedelta(days=1)
_PERIOD_NS = 86_400_000_000_000
_PREDICTION_COLUMNS = [
    "cohort",
    "period",
    "event_time",
    "dose",
    "att",
    "estimate_type",
    "n_treated",
    "n_controls",
    "extrapolated",
]


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    """Validate an integer, excluding Boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _validate_doses(doses: Iterable[float]) -> FloatArray:
    """Validate a nonempty collection of distinct positive finite doses."""
    if isinstance(doses, (str, bytes)):
        raise ValueError("doses must be a collection of numbers.")
    try:
        values = np.asarray(list(doses))
    except (TypeError, ValueError) as exc:
        raise ValueError("doses must be a collection of numbers.") from exc
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iuf":
        raise ValueError("doses must be a nonempty one-dimensional numeric collection.")
    result = values.astype(float)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("doses must be positive and finite.")
    if len(np.unique(result)) != len(result):
        raise ValueError("doses must be distinct.")
    return result


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Validate a DataFrame's unique labels and required mapped columns."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} must have unique column labels.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")


def _firm_ids(values: pd.Series) -> None:
    """Require nonmissing, hashable firm identifiers without coercion."""
    if values.isna().any():
        raise ValueError("Firm identifiers must not be missing.")
    try:
        for value in values:
            hash(value)
    except TypeError as exc:
        raise ValueError("Firm identifiers must be hashable.") from exc


def _periods(values: pd.Series) -> pd.Series:
    """Normalize timezone-naive datetime64[ns] period labels on a copy."""
    if values.dtype != np.dtype("datetime64[ns]") or values.isna().any():
        raise ValueError("Periods must be nonmissing timezone-naive datetime64[ns].")
    return values.dt.normalize()


def _numeric(values: pd.Series, name: str, *, positive: bool = False) -> FloatArray:
    """Require finite real numeric values on every supplied row."""
    if (
        not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
    ):
        raise ValueError(f"{name} must be real numeric values.")
    result = values.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(result).all() or (positive and (result <= 0).any()):
        qualifier = "positive and finite" if positive else "finite and nonmissing"
        raise ValueError(f"{name} must be {qualifier} on every present row.")
    return result


def _estimate_type(event_time: int) -> str:
    """Return the interpretation of a relative horizon."""
    return "reference" if event_time == -1 else "placebo" if event_time < -1 else "att"


class ATT:
    """Fit a separate polynomial OLS model for each cohort-horizon dose curve.

    Each model's intercept absorbs its control mean, so predictions use the
    treatment coefficients directly. Saved dose scaling is reused at prediction.
    No covariates or inference are implemented.

    Attributes:
        models_:
            Fitted PyFixest models keyed by non-reference cohort-horizon cells.
        events_:
            First-event metadata with firm_id, cohort, dose, design_eligible.
        termination_history_:
            Complete firm-period termination counts, including excluded events.
        bases_:
            Saved polynomial degrees and dose bounds keyed by (cohort, event_time).
        cell_coefficients_:
            Treatment coefficients from models_, with exact zeros for references.
        retained_cohorts_:
            Cohorts passing treated-design screening at each horizon, including
            those with missing controls that must cause aggregation to fail.
        cell_diagnostics_:
            Cell status, counts, and the exact contributing firm identifiers.
        availability_diagnostics_:
            Baseline and endpoint availability for every design-eligible firm
            and horizon. Diagnostics remain available after a failed fit.
        event_diagnostics_:
            First-event design eligibility, exclusion reasons, and conflicts.
        cohort_diagnostics_:
            Design-eligible firm counts and cell statuses by cohort.
        duplicate_diagnostics_:
            Counts of additional event rows and affected firms.
        drop_summary_:
            Aggregate exclusion counts, distinguishing event rows from treated
            firm-horizon rows and rejected comparisons. Logged once per fit.
        column_names_:
            The four explicit input-column mappings.
        min_cohort_size_:
            The resolved treated-sample minimum, including the design floor.
        event_times_:
            All integer horizons in the inclusive isolation window.
        origin_period_:
            Earliest normalized outcome period.
        last_period_:
            Latest normalized outcome period.
        last_period_index_:
            Inclusive elapsed-period index of last_period_ from origin_period_.
        isolation_window_:
            Signed lower and upper screening offsets.
        cell_keys_:
            Mapping from internal comparison labels to cohort-horizon keys.
        effect_columns_:
            Ordered model coefficient names for each fitted non-reference cell.
    """

    def __init__(
        self,
        *,
        firm_col: str,
        period_col: str,
        outcome_col: str,
        dose_col: str,
        degree: int = 3,
        min_cohort_size: int | None = None,
    ) -> None:
        """Configure the estimator and its required input-column mappings.

        Args:
            firm_col:
                Firm identifier column shared by both input tables.
            period_col:
                Period column shared by both input tables; datetime64[ns].
            outcome_col:
                Finite outcome column in the outcome table.
            dose_col:
                Positive finite dose column in the event table.
            degree:
                Positive polynomial degree.
            min_cohort_size:
                Optional treated-sample minimum. None resolves to degree plus
                two; a smaller explicit minimum cannot override it.

        Raises:
            ValueError:
                A mapping is empty or conflicts within a table, or a numerical
                configuration value is invalid.
        """
        names = (firm_col, period_col, outcome_col, dose_col)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("All four column-name arguments must be nonempty strings.")
        if len(set(names[:3])) != 3 or len({firm_col, period_col, dose_col}) != 3:
            raise ValueError("Mapped column names must be distinct within each table.")
        self.firm_col = firm_col
        self.period_col = period_col
        self.outcome_col = outcome_col
        self.dose_col = dose_col
        self.degree = _integer(degree, "degree", 1)
        self.min_cohort_size = None if min_cohort_size is None else _integer(min_cohort_size, "min_cohort_size", 1)
        self._reset()

    def _reset(self) -> None:
        """Invalidate fitted state before validating or attempting a new fit."""
        for name in tuple(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self._is_fitted = False
        self.models_: dict[Cell, Any] = {}
        self.column_names_ = {
            "firm_col": self.firm_col,
            "period_col": self.period_col,
            "outcome_col": self.outcome_col,
            "dose_col": self.dose_col,
        }
        self.min_cohort_size_ = max(
            self.degree + 2,
            self.min_cohort_size or 0,
        )
        self.event_times_: list[int] = []
        self.events_ = pd.DataFrame()
        self.termination_history_ = pd.DataFrame()
        self.event_diagnostics_ = pd.DataFrame()
        self.availability_diagnostics_ = pd.DataFrame()
        self.cohort_diagnostics_ = pd.DataFrame()
        self.duplicate_diagnostics_ = {"n_dropped_events": 0, "n_affected_firms": 0}
        self.bases_: dict[Cell, dict[str, Any]] = {}
        self.cell_coefficients_: dict[Cell, FloatArray] = {}
        self.cell_diagnostics_: dict[Cell, dict[str, Any]] = {}
        self.retained_cohorts_: dict[int, list[pd.Timestamp]] = {}
        self.cell_keys_: dict[int, Cell] = {}
        self.effect_columns_: dict[Cell, list[str]] = {}
        self.drop_summary_: dict[str, Any] = {}

    def _inputs(
        self,
        outcomes: pd.DataFrame,
        events: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and copy the two tables into internal column conventions."""
        outcome_columns = [self.firm_col, self.period_col, self.outcome_col]
        event_columns = [self.firm_col, self.period_col, self.dose_col]
        _require_columns(outcomes, outcome_columns, "outcomes")
        _require_columns(events, event_columns, "events")
        if outcomes.empty:
            raise ValueError("outcomes must contain at least one row.")
        outcome = outcomes[outcome_columns].copy()
        event = events[event_columns].copy()
        outcome.columns = ["firm_id", "period", "outcome"]
        event.columns = ["firm_id", "cohort", "dose"]
        _firm_ids(outcome["firm_id"])
        _firm_ids(event["firm_id"])
        outcome["period"] = _periods(outcome["period"])
        event["cohort"] = _periods(event["cohort"])
        outcome["outcome"] = _numeric(outcome["outcome"], "Outcomes")
        event["dose"] = _numeric(event["dose"], "Doses", positive=True)
        if outcome.duplicated(["firm_id", "period"]).any():
            raise ValueError(
                "Outcome rows must be unique by firm and normalized period.",
            )
        if not event["firm_id"].isin(outcome["firm_id"]).all():
            raise ValueError("Every event firm must be present in the outcome panel.")
        return outcome, event

    def _prepare_events(self, outcome: pd.DataFrame, event: pd.DataFrame) -> None:
        """Select first events and screen their complete isolation intervals."""
        self.origin_period_: pd.Timestamp = outcome["period"].min()
        self.last_period_: pd.Timestamp = outcome["period"].max()
        origin = self.origin_period_.value // _PERIOD_NS
        last = self.last_period_.value // _PERIOD_NS
        self.last_period_index_ = last - origin + 1
        counts = event.groupby("firm_id", sort=False, observed=True).size()
        self.duplicate_diagnostics_ = {
            "n_dropped_events": len(event) - len(counts),
            "n_affected_firms": int((counts > 1).sum()),
        }
        self.termination_history_ = (
            event.groupby(["firm_id", "cohort"], sort=False, observed=True)
            .size()
            .rename("n_terminations")
            .reset_index()
        )
        history = {
            firm: group
            for firm, group in self.termination_history_.groupby(
                "firm_id",
                sort=False,
                observed=True,
            )
        }
        first = event.sort_values("cohort", kind="stable").drop_duplicates("firm_id")
        diagnostics = []
        lower, upper = self.isolation_window_
        for row in first.itertuples(index=False):
            g = row.cohort.value // _PERIOD_NS
            firm_history = history[row.firm_id]
            indices = firm_history["cohort"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // _PERIOD_NS
            conflicts = firm_history.loc[(indices >= g + lower) & (indices <= g + upper)]
            n_conflicts = int(conflicts["n_terminations"].sum()) - 1
            if not origin <= g <= last:
                reason = "outside_outcome_period"
            elif g + lower < origin or g + upper > last:
                reason = "isolation_outside_panel"
            elif n_conflicts:
                reason = "overlapping_termination"
            else:
                reason = "eligible"
            diagnostics.append(
                {
                    "firm_id": row.firm_id,
                    "cohort": row.cohort,
                    "dose": row.dose,
                    "design_eligible": reason == "eligible",
                    "reason": reason,
                    "n_conflicting_terminations": n_conflicts,
                    "conflicting_periods": tuple(
                        item.cohort
                        for item in conflicts.itertuples(index=False)
                        if item.cohort != row.cohort or item.n_terminations > 1
                    ),
                },
            )
        self.event_diagnostics_ = pd.DataFrame(
            diagnostics,
            columns=[
                "firm_id",
                "cohort",
                "dose",
                "design_eligible",
                "reason",
                "n_conflicting_terminations",
                "conflicting_periods",
            ],
        )
        self.events_ = self.event_diagnostics_[["firm_id", "cohort", "dose", "design_eligible"]].copy()
        self.events_["cohort"] = pd.to_datetime(self.events_["cohort"])
        self.events_["design_eligible"] = self.events_["design_eligible"].astype(bool)
        eligible = self.events_.loc[self.events_["design_eligible"], "dose"]
        if eligible.empty:
            raise ValueError("No first events have a complete, isolated panel window.")

    def _basis(self, doses: FloatArray) -> tuple[str, dict[str, Any], FloatArray]:
        """Screen a treated polynomial design and save its degree and dose bounds.

        Rank uses NumPy's SVD tolerance: largest singular value multiplied by
        max(number of observations, number of columns) and machine precision.
        """
        p = self.degree + 1
        empty = np.empty((0, p))
        if len(doses) < self.min_cohort_size_:
            return "small_sample", {}, empty
        if len(np.unique(doses)) < p:
            return "insufficient_doses", {}, empty
        lower, upper = float(doses.min()), float(doses.max())
        scaled_doses = (doses - lower) / (upper - lower)
        design = np.polynomial.polynomial.polyvander(scaled_doses, self.degree)
        if not np.isfinite(design).all() or np.linalg.matrix_rank(design) < p:
            return "rank_deficient", {}, empty
        return "ready", {"degree": self.degree, "lower": lower, "upper": upper}, design

    def fit(
        self,
        outcomes: pd.DataFrame,
        events: pd.DataFrame,
        *,
        isolation_window: tuple[int, int],
    ) -> Self:
        """Fit separate polynomial regressions for eligible non-reference cells.

        Args:
            outcomes:
                Firm-period outcome table using the configured columns. Missing
                periods are allowed; present outcomes must be finite.
            events:
                Complete termination history using the configured columns.
                Only each firm's first event can be a target.
            isolation_window:
                Inclusive signed period offsets (lower, upper), with lower <= -1
                and upper >= 0. Every integer horizon in this window is processed.

        Returns:
            This estimator with fitted coefficients, saved bases, and diagnostics.
            Fitting clears any earlier fitted state, even if the new fit fails.
            Exclusions are recorded in drop_summary_ and logged once per fit.

        Raises:
            ValueError:
                Input values, periods, or mappings are invalid, or no
                non-reference comparison passes the design checks.
            RuntimeError:
                PyFixest fitting fails or omits a required finite coefficient.
        """
        self._reset()
        try:
            lower, upper = isolation_window
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "isolation_window must contain two signed integer offsets.",
            ) from exc
        lower = _integer(lower, "lower isolation offset")
        upper = _integer(upper, "upper isolation offset")
        if lower > -1 or upper < 0:
            raise ValueError("isolation_window requires lower <= -1 and upper >= 0.")
        self.isolation_window_ = (lower, upper)
        self.event_times_ = list(range(lower, upper + 1))
        self.retained_cohorts_ = {e: [] for e in self.event_times_}
        outcome, event = self._inputs(outcomes, events)
        try:
            self._prepare_events(outcome, event)
            blocks = self._prepare_comparisons(outcome)
            self._summarize_cohorts()
            if not blocks:
                raise ValueError("No non-reference comparison passes the required checks.")
            self._fit_model(blocks)
            self._summarize_cohorts()
            self._is_fitted = True
            return self
        finally:
            self._log_fit_exclusions()

    def _prepare_comparisons(self, outcome: pd.DataFrame) -> list[tuple[Cell, pd.DataFrame, FloatArray]]:
        """Build comparison samples with shared indexes and batched horizon lookups.

        Only treated availability uses a firm-by-horizon array. Control lookups
        process one horizon at a time over observed baseline rows. Each cell
        retains its own polynomial scaling and rank check, with no pooling.

        Args:
            outcome:
                Validated canonical firm-period outcome rows in input order.

        Returns:
            Cohort-horizon blocks and treated basis matrices, ordered by cohort
            then horizon. Also records availability, selection, and saved bases.

        Raises:
            ValueError:
                An observed outcome difference is nonfinite.
        """
        # First-event metadata is already stably sorted by cohort.
        eligible = self.events_.loc[self.events_["design_eligible"]]
        history = self.termination_history_
        firm_index = pd.Index(outcome["firm_id"].drop_duplicates(), tupleize_cols=False)
        firm_values = firm_index.to_numpy(dtype=object)
        outcome_firms = firm_index.get_indexer(outcome["firm_id"])
        outcome_periods = outcome["period"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // _PERIOD_NS
        outcome_values = outcome["outcome"].to_numpy(dtype=float)
        history_periods = history["cohort"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // _PERIOD_NS
        history_firms = firm_index.get_indexer(history["firm_id"])
        origin = min(int(outcome_periods.min()), int(history_periods.min()))
        span = max(int(outcome_periods.max()), int(history_periods.max())) - origin + 1
        outcome_keys = pd.Index(outcome_firms * span + outcome_periods - origin)
        history_keys = np.sort(history_firms * span + history_periods - origin)

        cohort_periods = pd.DatetimeIndex(eligible["cohort"].drop_duplicates().sort_values())
        cohort_days = cohort_periods.to_numpy(dtype="datetime64[ns]").astype(np.int64) // _PERIOD_NS
        n_cohorts = len(cohort_periods)
        target_cohorts = cohort_periods.get_indexer(eligible["cohort"])
        target_firms = firm_index.get_indexer(eligible["firm_id"])
        target_doses = eligible["dose"].to_numpy(dtype=float)
        target_periods = eligible["cohort"].to_numpy(dtype="datetime64[ns]")
        target_keys = target_firms * span + cohort_days[target_cohorts] - origin
        target_baselines = outcome_keys.get_indexer(target_keys - 1)
        horizons = np.asarray(self.event_times_, dtype=np.int64)
        target_endpoints = outcome_keys.get_indexer((target_keys[:, None] + horizons).ravel()).reshape(
            len(eligible),
            len(horizons),
        )
        missing_baseline = np.broadcast_to(target_baselines[:, None] < 0, target_endpoints.shape)
        missing_endpoint = target_endpoints < 0
        available = ~(missing_baseline | missing_endpoint)
        self.availability_diagnostics_ = pd.DataFrame(
            {
                "firm_id": np.repeat(eligible["firm_id"].to_numpy(), len(horizons)),
                "cohort": np.repeat(target_periods, len(horizons)),
                "event_time": np.tile(horizons, len(eligible)),
                "baseline": np.repeat(target_periods - np.timedelta64(1, "D"), len(horizons)),
                "period": (target_periods[:, None] + horizons.astype("timedelta64[D]")).ravel(),
                "missing_baseline": missing_baseline.ravel(),
                "missing_endpoint": missing_endpoint.ravel(),
                "available": available.ravel(),
            },
        ).sort_values(["cohort", "event_time"], kind="stable", ignore_index=True)

        # Group only observed baseline rows, preserving their input order within g.
        baseline_cohorts = pd.Index(cohort_days).get_indexer(outcome_periods + 1)
        baseline_rows = np.flatnonzero(baseline_cohorts >= 0)
        baseline_rows = baseline_rows[np.argsort(baseline_cohorts[baseline_rows], kind="stable")]
        baseline_cohorts = baseline_cohorts[baseline_rows]
        baseline_firms = outcome_firms[baseline_rows]
        baseline_keys = baseline_firms * span + outcome_periods[baseline_rows] - origin
        lower, upper = self.isolation_window_
        lower_keys = baseline_firms * span + cohort_days[baseline_cohorts] + lower - origin
        upper_keys = baseline_firms * span + cohort_days[baseline_cohorts] + upper - origin
        clean = np.searchsorted(history_keys, lower_keys, side="left") == np.searchsorted(
            history_keys,
            upper_keys,
            side="right",
        )
        blocks: list[tuple[Cell, pd.DataFrame, FloatArray]] = []
        for horizon_index, e in enumerate(self.event_times_):
            endpoints = outcome_keys.get_indexer(baseline_keys + e + 1)
            common = endpoints >= 0
            changes = np.empty(len(baseline_rows), dtype=float)
            with np.errstate(over="ignore", invalid="ignore"):
                changes[common] = outcome_values[endpoints[common]] - outcome_values[baseline_rows[common]]
            if not np.isfinite(changes[common]).all():
                raise ValueError("Outcome differences must be finite.")
            control_rows = np.flatnonzero(common & clean)
            control_bounds = np.r_[0, np.cumsum(np.bincount(baseline_cohorts[control_rows], minlength=n_cohorts))]
            treated_rows = np.flatnonzero(available[:, horizon_index])
            treated_bounds = np.r_[0, np.cumsum(np.bincount(target_cohorts[treated_rows], minlength=n_cohorts))]
            with np.errstate(over="ignore", invalid="ignore"):
                treated_changes = (
                    outcome_values[target_endpoints[treated_rows, horizon_index]]
                    - outcome_values[target_baselines[treated_rows]]
                )
            for cohort_index, cohort_value in enumerate(cohort_periods):
                cohort = cast(pd.Timestamp, cohort_value)
                start, stop = treated_bounds[cohort_index : cohort_index + 2]
                treated = treated_rows[start:stop]
                controls = control_rows[control_bounds[cohort_index] : control_bounds[cohort_index + 1]]
                treated_firms = tuple(firm_values[target_firms[treated]])
                control_firms = tuple(firm_values[baseline_firms[controls]])
                doses = target_doses[treated]
                key = (cohort, e)
                diagnostic = {
                    "status": "no_observed_treated",
                    "n_treated": len(treated),
                    "n_controls": len(controls),
                    "treated_firms": treated_firms,
                    "control_firms": control_firms,
                    "estimate_type": _estimate_type(e),
                    "n_distinct_doses": len(np.unique(doses)),
                    "design_dimension": self.degree + 1,
                }
                self.cell_diagnostics_[key] = diagnostic
                if not len(treated):
                    continue
                status, basis, design = self._basis(doses)
                diagnostic["status"] = status
                if status != "ready":
                    continue
                self.retained_cohorts_[e].append(cohort)
                if not len(controls):
                    diagnostic["status"] = "no_controls"
                    continue
                self.bases_[key] = basis
                if e == -1:
                    diagnostic["status"] = "reference"
                    self.cell_coefficients_[key] = np.zeros(design.shape[1])
                    continue
                block = pd.DataFrame(
                    {
                        "firm_id": [*treated_firms, *control_firms],
                        "cell": 0,
                        "cohort": cohort,
                        "period": cohort + e * _PERIOD,
                        "event_time": e,
                        "delta_y": np.r_[treated_changes[start:stop], changes[controls]],
                        "treated": np.r_[np.ones(len(treated), dtype=int), np.zeros(len(controls), dtype=int)],
                        "scaled_dose": np.r_[design[:, 1], np.zeros(len(controls))],
                    },
                )
                blocks.append((key, block, design))
        blocks.sort(key=lambda item: item[0])
        for cell, (_, block, _) in enumerate(blocks):
            block["cell"] = cell
        self.cell_diagnostics_ = dict(sorted(self.cell_diagnostics_.items()))
        self.bases_ = dict(sorted(self.bases_.items()))
        self.cell_coefficients_ = dict(sorted(self.cell_coefficients_.items()))
        return blocks

    def _log_fit_exclusions(self) -> None:
        """Record separate exclusion counts and emit at most one warning per fit."""
        excluded_events: dict[str, int] = {}
        if not self.event_diagnostics_.empty:
            reasons = self.event_diagnostics_.loc[~self.event_diagnostics_["design_eligible"], "reason"]
            excluded_events = {str(reason): int(count) for reason, count in reasons.value_counts().items()}
        unavailable = 0
        if not self.availability_diagnostics_.empty:
            unavailable = int((~self.availability_diagnostics_["available"]).sum())
        dropped_cells: dict[str, int] = {}
        dropped_rows = 0
        dropped_cohorts: set[pd.Timestamp] = set()
        for (cohort, _), cell in self.cell_diagnostics_.items():
            status = cell["status"]
            if status in {"ready", "fitted", "reference", "model_failed"}:
                continue
            dropped_cells[status] = dropped_cells.get(status, 0) + 1
            dropped_rows += cell["n_treated"]
            dropped_cohorts.add(cohort)
        self.drop_summary_ = {
            "duplicate_event_rows": self.duplicate_diagnostics_["n_dropped_events"],
            "duplicate_event_firms": self.duplicate_diagnostics_["n_affected_firms"],
            "excluded_first_events": excluded_events,
            "unavailable_treated_rows": unavailable,
            "dropped_treated_rows": dropped_rows,
            "dropped_cells": dict(sorted(dropped_cells.items())),
            "dropped_cohorts": len(dropped_cohorts),
        }
        if self.drop_summary_["duplicate_event_rows"] or excluded_events or unavailable or dropped_cells:
            logging.warning(
                "Fit exclusions: %d duplicate event rows (%d firms); excluded first events=%s; "
                "%d unavailable treated firm-horizon rows; %d observed treated firm-horizon rows "
                "in %d rejected comparisons across %d cohorts (reasons=%s). See drop_summary_ and cell_diagnostics_.",
                self.drop_summary_["duplicate_event_rows"],
                self.drop_summary_["duplicate_event_firms"],
                excluded_events,
                unavailable,
                dropped_rows,
                sum(dropped_cells.values()),
                len(dropped_cohorts),
                self.drop_summary_["dropped_cells"],
            )

    def _fit_model(self, blocks: list[tuple[Cell, pd.DataFrame, FloatArray]]) -> None:
        """Fit narrow cell models without constructing a global interaction matrix."""
        polynomial = f"poly(scaled_dose, degree={self.degree}, raw=True)"
        formula = f"delta_y ~ treated + treated:{polynomial}"
        # Formulaic labels raw ndarray columns from zero; column k is power k+1.
        names = ["treated", *[f"treated:{polynomial}[{k}]" for k in range(self.degree)]]
        required = ["Intercept", *names]
        try:
            for cell, (key, block, _) in enumerate(blocks):
                self.cell_keys_[cell] = key
                model = pf.feols(
                    formula,
                    data=block,
                    lean=True,
                    store_data=False,
                    copy_data=False,
                )
                coefficients = model.coef()
                if any(name not in coefficients.index for name in required):
                    raise RuntimeError("PyFixest omitted a required regression coefficient.")
                if not np.isfinite(coefficients.loc[required].to_numpy()).all():
                    raise RuntimeError("PyFixest returned a nonfinite regression coefficient.")
                self.models_[key] = model
                self.effect_columns_[key] = names.copy()
                self.cell_coefficients_[key] = coefficients.loc[names].to_numpy(dtype=float)
                self.cell_diagnostics_[key]["status"] = "fitted"
        except Exception as exc:
            for key, _, _ in blocks:
                self.cell_diagnostics_[key]["status"] = "model_failed"
                self.cell_coefficients_.pop(key, None)
            self.models_.clear()
            self.effect_columns_.clear()
            self._summarize_cohorts()
            raise RuntimeError(
                "The polynomial cell models could not be fitted completely.",
            ) from exc

    def _summarize_cohorts(self) -> None:
        """Summarize design eligibility and horizon statuses without dropping history."""
        records = []
        by_cohort: dict[pd.Timestamp, dict[int, dict[str, Any]]] = {}
        for (cohort, e), cell in self.cell_diagnostics_.items():
            by_cohort.setdefault(cohort, {})[e] = cell
        for cohort_key, events in self.events_.groupby("cohort", sort=True):
            cohort = cast(pd.Timestamp, cohort_key)
            cells = by_cohort.get(cohort, {})
            records.append(
                {
                    "cohort": cohort,
                    "n_firms_design": int(events["design_eligible"].sum()),
                    "n_cells_available": sum(cell["n_treated"] > 0 for cell in cells.values()),
                    "n_cells_used": sum(cell["status"] in {"fitted", "reference"} for cell in cells.values()),
                    "statuses": {e: cell["status"] for e, cell in cells.items()},
                },
            )
        self.cohort_diagnostics_ = pd.DataFrame(records)

    def _require_fitted(self) -> None:
        """Reject prediction or aggregation without a successful current fit."""
        if not self._is_fitted:
            raise RuntimeError("ATT must be successfully fitted before evaluation.")

    def _resolve_event_times(self, event_times: Iterable[int] | None) -> list[int]:
        """Validate a horizon subset without fitting or changing eligibility."""
        if event_times is None:
            return list(self.event_times_)
        try:
            values = [_integer(e, "event_time") for e in event_times]
        except TypeError as exc:
            raise ValueError(
                "event_times must be a nonempty collection of integers.",
            ) from exc
        if not values or len(set(values)) != len(values):
            raise ValueError("event_times must be nonempty and distinct.")
        if not set(values).issubset(self.event_times_):
            raise ValueError(
                "Requested event_times lie outside the fitted isolation window.",
            )
        return sorted(values)

    def predict(
        self,
        doses: Iterable[float],
        *,
        event_times: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        """Evaluate unpooled dose curves using the saved cell bases.

        Args:
            doses:
                Distinct positive finite doses at which to predict.
            event_times:
                Optional distinct integer horizons within the fitted window.
                None requests every fitted horizon, including the reference.

        Returns:
            A DataFrame sorted by cohort, event_time, and dose, with cohort,
            period, event_time, dose, att, estimate_type, n_treated, n_controls,
            and extrapolated. Cohort and period are normalized datetime64[ns].
            Failed cells are omitted with logging diagnostics. Reference cells
            are exactly zero; other extrapolated predictions are flagged.

        Raises:
            RuntimeError:
                There is no successful current fit.
            ValueError:
                Doses or horizons are invalid, or polynomial prediction is nonfinite.
        """
        self._require_fitted()
        dose_values = np.sort(_validate_doses(doses))
        horizons = self._resolve_event_times(event_times)
        failed = [
            key
            for key, cell in self.cell_diagnostics_.items()
            if key[1] in horizons and cell["status"] not in {"fitted", "reference", "no_observed_treated"}
        ]
        if failed:
            logging.warning(
                "Predictions omit %d unavailable comparisons; inspect cell_diagnostics_.",
                len(failed),
            )
        result = self._prediction_rows(dose_values, horizons)
        extrapolated_count = int(result["extrapolated"].sum())
        if extrapolated_count:
            logging.warning(
                "Extrapolating %d predictions beyond fitted dose support.",
                extrapolated_count,
            )
        return result

    def _prediction_rows(self, dose_values: FloatArray, horizons: list[int]) -> pd.DataFrame:
        """Evaluate validated requests without emitting intermediate warnings.

        Args:
            dose_values:
                Validated, sorted positive finite doses.
            horizons:
                Validated fitted horizons to evaluate.

        Returns:
            Component prediction rows with the same schema as predict().

        Raises:
            ValueError:
                A polynomial prediction is nonfinite.
        """
        records = []
        for key, coefficients in sorted(self.cell_coefficients_.items()):
            cohort, e = key
            if e not in horizons:
                continue
            cell = self.cell_diagnostics_[key]
            basis = self.bases_[key]
            if e == -1:
                predictions = np.zeros(len(dose_values))
                outside = np.zeros(len(dose_values), dtype=bool)
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    scaled_doses = (dose_values - basis["lower"]) / (basis["upper"] - basis["lower"])
                    predictions = np.polynomial.polynomial.polyval(scaled_doses, coefficients)
                if not np.isfinite(predictions).all():
                    raise ValueError(
                        "Polynomial predictions are nonfinite at the requested doses.",
                    )
                outside = (dose_values < basis["lower"]) | (dose_values > basis["upper"])
            for dose, prediction, extrapolated in zip(
                dose_values,
                predictions,
                outside,
            ):
                records.append(
                    {
                        "cohort": cohort,
                        "period": cohort + e * _PERIOD,
                        "event_time": e,
                        "dose": dose,
                        "att": prediction,
                        "estimate_type": cell["estimate_type"],
                        "n_treated": cell["n_treated"],
                        "n_controls": cell["n_controls"],
                        "extrapolated": bool(extrapolated),
                    },
                )
        return pd.DataFrame(records, columns=_PREDICTION_COLUMNS).astype(
            {
                "cohort": "datetime64[ns]",
                "period": "datetime64[ns]",
                "event_time": "int64",
                "dose": "float64",
                "att": "float64",
                "n_treated": "int64",
                "n_controls": "int64",
                "extrapolated": "bool",
            },
        )


def _validate_aggregation_firms(firms: Iterable[object], name: str) -> list[object]:
    """Validate a nonempty collection of unique, nonmissing firm identifiers.

    Args:
        firms:
            Firm identifiers to validate without converting their types.

        name:
            Argument name used in validation messages.

    Returns:
        The identifiers in their supplied order.

    Raises:
        ValueError:
            If identifiers are missing, repeated, unhashable, or not supplied
            as a nonempty collection.
    """
    if isinstance(firms, (str, bytes)):
        raise ValueError(f"{name} must be a collection of firm identifiers.")
    try:
        values = list(firms)
    except TypeError as exc:
        raise ValueError(f"{name} must be a collection of firm identifiers.") from exc
    if not values or pd.Series(values, dtype=object).isna().any():
        raise ValueError(f"{name} must contain nonmissing firm identifiers.")
    try:
        unique = set(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain hashable firm identifiers.") from exc
    if len(unique) != len(values):
        raise ValueError(f"{name} must contain unique firm identifiers.")
    return values


def _bandwidth_events(
    events: pd.DataFrame,
    firm_col: str,
    period_col: str,
    dose_col: str,
    eligible_firms: Iterable[object] | None,
) -> pd.DataFrame:
    """Validate event inputs and select one first event per requested firm."""
    columns = [firm_col, period_col, dose_col]
    if any(not isinstance(column, str) or not column for column in columns) or len(set(columns)) != 3:
        raise ValueError("Bandwidth column names must be distinct nonempty strings.")
    _require_columns(events, columns, "events")
    first = events[columns].copy()
    first.columns = ["firm_id", "cohort", "dose"]
    _firm_ids(first["firm_id"])
    first["cohort"] = _periods(first["cohort"])
    first["dose"] = _numeric(first["dose"], "Doses", positive=True)
    first = first.sort_values("cohort", kind="stable").drop_duplicates("firm_id")
    if eligible_firms is not None:
        eligible = _validate_aggregation_firms(eligible_firms, "eligible_firms")
        if not set(eligible).issubset(set(first["firm_id"])):
            raise ValueError("Every eligible firm must be present in events.")
        first = first.loc[first["firm_id"].isin(eligible)]
    if len(first) < 2:
        raise ValueError("Bandwidth estimation requires at least two selected firms.")
    return first.reset_index(drop=True)


def _silverman_h0(doses: FloatArray) -> float:
    """Compute the initial bandwidth from validated first-event doses."""
    with np.errstate(over="ignore", invalid="ignore"):
        sd = float(doses.std(ddof=1))
        q25, q75 = np.percentile(doses, [25, 75])
        iqr = float(q75 - q25)
        scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Selected doses must have positive finite variation.")
    return float(0.9 * scale * len(doses) ** (-0.2))


def estimate_h0(
    events: pd.DataFrame,
    *,
    firm_col: str,
    period_col: str,
    dose_col: str,
    eligible_firms: Iterable[object] | None = None,
) -> float:
    """Estimate Silverman's starting bandwidth from first-event doses.

    Args:
        events:
            Initial event table with positive finite doses and datetime64[ns]
            periods. Each firm's earliest normalized period is selected;
            same-period ties retain input order. Inputs are not modified.
        firm_col:
            Firm identifier column.
        period_col:
            Event period column; time components are ignored.
        dose_col:
            Dose column.
        eligible_firms:
            Optional unique firm identifiers to retain after first-event
            selection. None uses all first events. No isolation, outcome
            availability, or fitted-cell screens are applied automatically.

    Returns:
        A bandwidth in dose units, using one observation per selected firm.
        Repeated doses remain separate observations. The sample standard
        deviation is used alone when the interquartile range is zero.

    Raises:
        ValueError:
            Inputs violate the event schema or eligibility rules, fewer than
            two firms remain, or selected doses lack positive finite variation.
    """
    first = _bandwidth_events(events, firm_col, period_col, dose_col, eligible_firms)
    return _silverman_h0(first["dose"].to_numpy(dtype=float))


def _loo_brier_scores(
    doses: FloatArray,
    cohort_codes: NDArray[np.int64],
    bandwidths: FloatArray,
    batch_size: int,
) -> FloatArray:
    """Compute exact firm-level leave-one-out loss using sparse dose-cohort counts.

    Repeated doses share a query but retain their firm multiplicities. Remove
    singleton query doses before kernel stabilization; subtract one self-weight
    for repeated doses. This avoids cancellation when a tiny bandwidth would
    give an isolated held-out firm almost all the in-sample kernel mass.

    Args:
        doses:
            Positive finite doses, with one entry per selected firm.
        cohort_codes:
            Corresponding consecutive cohort codes starting from zero.
        bandwidths:
            Validated positive finite candidate bandwidths.
        batch_size:
            Maximum number of distinct query doses processed together.

    Returns:
        Mean firm-level Brier loss for each supplied bandwidth.

    Raises:
        ValueError:
            Kernel arithmetic overflows or losses are nonfinite or outside
            the Brier bounds beyond rounding tolerance.
    """
    unique_doses, dose_codes, multiplicities = np.unique(doses, return_inverse=True, return_counts=True)
    counts = csr_matrix(
        (np.ones(len(doses)), (dose_codes, cohort_codes)),
        shape=(len(unique_doses), int(cohort_codes.max()) + 1),
    )
    losses_by_dose = np.empty((len(unique_doses), len(bandwidths)))
    for start in range(0, len(unique_doses), batch_size):
        stop = min(start + batch_size, len(unique_doses))
        query_indices = np.arange(start, stop)
        singleton = multiplicities[start:stop] == 1
        self_weight = (~singleton).astype(float)
        differences = unique_doses[start:stop, None] - unique_doses[None, :]
        for index, bandwidth in enumerate(bandwidths):
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    log_kernel = -0.5 * np.square(differences / bandwidth)
            except FloatingPointError as exc:
                raise ValueError("CV kernel calculations overflowed; increase bandwidths or rescale doses.") from exc
            log_kernel[np.flatnonzero(singleton), query_indices[singleton]] = -np.inf
            log_kernel -= log_kernel.max(axis=1, keepdims=True)
            kernel = np.exp(log_kernel)
            masses = np.asarray(counts.T @ kernel.T, dtype=float).T
            denominator = masses.sum(axis=1) - self_weight
            probabilities = masses / denominator[:, None]
            own_probability = (
                np.asarray(counts[start:stop].multiply(probabilities).sum(axis=1)).ravel() / multiplicities[start:stop]
            )
            adjustment = self_weight / denominator
            # Subtract the held-out firm's probability only in its own cohort,
            # then average that cohort's contribution over same-dose firms.
            losses = (
                1.0
                + np.square(probabilities).sum(axis=1)
                - 2.0 * (1.0 + adjustment) * own_probability
                + 2.0 * adjustment
                + np.square(adjustment)
            )
            if not np.isfinite(losses).all() or (losses < -1e-12).any() or (losses > 2.0 + 1e-12).any():
                raise ValueError("CV produced invalid multiclass Brier scores.")
            losses_by_dose[start:stop, index] = np.clip(losses, 0.0, 2.0) * multiplicities[start:stop]
    return losses_by_dose.sum(axis=0) / len(doses)


def _expand_bandwidth_grid(
    doses: FloatArray,
    cohort_codes: NDArray[np.int64],
    candidates: FloatArray,
    scores: FloatArray,
    batch_size: int,
    max_expansions: int,
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Extend minimizing grid edges until loss rises or a search limit is reached.

    Args:
        doses:
            Validated first-event doses from the fixed selected population.
        cohort_codes:
            Corresponding consecutive cohort codes starting from zero.
        candidates:
            Sorted initial candidate bandwidths.
        scores:
            Already evaluated Brier scores for the initial candidates.
        batch_size:
            Number of distinct query doses processed together.
        max_expansions:
            Maximum number of new bandwidths evaluated in each direction.

    Returns:
        All evaluated bandwidths and scores in ascending bandwidth order,
        plus diagnostic messages for directions stopped without a score rise.
        Both directions are selected from the initial grid, independently.

    Raises:
        ValueError:
            An additional CV evaluation produces invalid losses.
    """
    evaluated = dict(zip(candidates.tolist(), scores.tolist(), strict=True))
    minimum = float(scores.min())
    span = float(doses.max() - doses.min())
    messages = []
    for edge, factor, direction in ((0, 0.5, "lower"), (-1, 2.0, "upper")):
        if scores[edge] != minimum:
            continue
        previous_h = float(candidates[edge])
        previous_score = float(scores[edge])
        flat_steps = 0
        for _ in range(max_expansions):
            next_h = previous_h * factor
            if not np.isfinite(next_h) or next_h <= 0 or next_h == previous_h:
                messages.append(f"{direction} expansion reached floating-point bandwidth limits before a score rise")
                break
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    np.square(np.float64(span) / next_h)
            except FloatingPointError:
                messages.append(f"{direction} expansion reached kernel arithmetic limits before a score rise")
                break
            next_score = float(_loo_brier_scores(doses, cohort_codes, np.array([next_h]), batch_size)[0])
            evaluated[next_h] = next_score
            tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(previous_score), abs(next_score))
            change = next_score - previous_score
            if change > tolerance:
                break
            flat_steps = flat_steps + 1 if abs(change) <= tolerance else 0
            if flat_steps == 3:
                messages.append(f"{direction} expansion stopped after three numerically flat steps before a score rise")
                break
            previous_h, previous_score = next_h, next_score
        else:
            messages.append(f"{direction} expansion reached max_expansions={max_expansions} before a score rise")
    expanded = np.asarray(sorted(evaluated), dtype=float)
    losses = np.asarray([evaluated[float(bandwidth)] for bandwidth in expanded], dtype=float)
    return expanded, losses, messages


def estimate_h_cv(
    events: pd.DataFrame,
    *,
    firm_col: str,
    period_col: str,
    dose_col: str,
    eligible_firms: Iterable[object] | None = None,
    bandwidths: Iterable[float] | None = None,
    batch_size: int = 256,
    max_expansions: int = 50,
) -> tuple[float, pd.DataFrame]:
    """Select a Gaussian bandwidth by leave-one-firm-out multiclass Brier loss.

    Args:
        events:
            Initial event table with positive finite doses and datetime64[ns]
            periods. Select each firm's earliest normalized event, breaking
            same-period ties by input order, without modifying the input table.
        firm_col:
            Firm identifier column.
        period_col:
            Event period column; time components are ignored.
        dose_col:
            Dose column.
        eligible_firms:
            Optional unique firm identifiers to retain after first-event
            selection. None uses all first events. Outcome availability,
            isolation, and fitted-cell eligibility are not inferred.
        bandwidths:
            Initial distinct positive finite bandwidths in dose units.
            None starts with nine geometric values from h0/10 to 10*h0.
            Minimizing edges are extended by halving or doubling until the
            score rises. Supplied grids are also starting points, not bounds.
        batch_size:
            Positive number of distinct query doses processed together.
            Batching limits kernel memory; all-distinct doses still require
            quadratic work for exact leave-one-firm-out evaluation.
        max_expansions:
            Positive maximum number of new evaluations per direction. Search
            also stops at numerical limits or after three consecutive score
            changes within 64 machine epsilons times the larger absolute
            adjacent score or one. Such stops are reported in the warning.

    Returns:
        The selected bandwidth and a DataFrame with bandwidth and cv_brier
        columns, sorted by bandwidth, including expansion and upturn points.
        Scores average over firms, not distinct doses. Exact ties choose the
        smallest bandwidth. At most one warning summarizes singleton cohorts,
        tied minima, and searches stopped without a score rise. Both edges
        are explored if they minimize the initial grid. A rise must exceed
        the same numerical tolerance used to detect flat changes.
        This tunes one selected population; no ATT models are refitted and
        no averaging of horizon-specific CV risks is performed.

    Raises:
        ValueError:
            Inputs, eligibility, initial bandwidths, or search settings are
            invalid; fewer than two cohorts or distinct doses remain; or
            kernel arithmetic overflows or produces invalid losses.
    """
    batch_size = _integer(batch_size, "batch_size", 1)
    max_expansions = _integer(max_expansions, "max_expansions", 1)
    first = _bandwidth_events(events, firm_col, period_col, dose_col, eligible_firms)
    doses = first["dose"].to_numpy(dtype=float)
    if doses.min() == doses.max():
        raise ValueError("Selected doses must have positive variation.")
    cohort_codes, cohorts = pd.factorize(first["cohort"], sort=True)
    if len(cohorts) < 2:
        raise ValueError("Bandwidth CV requires at least two selected cohorts.")
    if bandwidths is None:
        bandwidths = _silverman_h0(doses) * np.geomspace(0.1, 10.0, 9)
    try:
        if isinstance(bandwidths, (str, bytes)):
            raise ValueError("A bandwidth grid must contain numbers.")
        values = list(bandwidths)
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("Boolean bandwidths are invalid.")
        candidates = np.sort(_validate_doses(values))
    except (TypeError, ValueError) as exc:
        raise ValueError("bandwidths must contain distinct positive finite numbers.") from exc
    codes = np.asarray(cohort_codes, dtype=np.int64)
    scores = _loo_brier_scores(doses, codes, candidates, batch_size)
    candidates, scores, messages = _expand_bandwidth_grid(
        doses,
        codes,
        candidates,
        scores,
        batch_size,
        max_expansions,
    )
    best = int(np.argmin(scores))
    singletons = int(np.count_nonzero(np.bincount(cohort_codes) == 1))
    if singletons:
        messages.append(f"{singletons} singleton cohorts have zero own-cohort probability when held out")
    if best in {0, len(candidates) - 1}:
        messages.append("selected bandwidth remains at a search boundary")
    ties = int(np.count_nonzero(scores == scores[best]))
    if ties > 1:
        messages.append(f"{ties} candidates tie for the minimum; selected the smallest bandwidth")
    if messages:
        logging.warning("Bandwidth CV summary: %s.", "; ".join(messages))
    return float(candidates[best]), pd.DataFrame({"bandwidth": candidates, "cv_brier": scores})


def estimate_pi(
    events: pd.DataFrame,
    doses: Iterable[float],
    *,
    eligible_firms: Iterable[object],
    bandwidth: float,
) -> pd.DataFrame:
    """Estimate dose-specific cohort weights for eligible firms.

    Args:
        events:
            Canonical first-event metadata with firm_id, cohort, and dose
            columns, containing one row per firm. Cohort periods must be
            normalized, timezone-naive datetime64[ns] values.

        doses:
            Distinct positive finite doses at which to evaluate weights.

        eligible_firms:
            Unique firm identifiers defining the eligible sample.

        bandwidth:
            Positive finite Gaussian-kernel bandwidth in dose units.

    Returns:
        A DataFrame with cohort, dose, and weight columns, sorted by cohort
        period and dose. Weights sum to one at each requested dose.

    Raises:
        ValueError:
            If the bandwidth is nonpositive or nonfinite, the inputs violate
            the required schema or eligibility rules, or kernel calculations
            produce nonfinite intermediate values.
    """
    dose_values = _validate_doses(doses)
    if isinstance(bandwidth, (bool, np.bool_)):
        raise ValueError("bandwidth must be positive and finite.")
    try:
        kernel_bandwidth = float(bandwidth)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bandwidth must be positive and finite.") from exc
    if not np.isfinite(kernel_bandwidth) or kernel_bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite.")
    required_columns = {"firm_id", "cohort", "dose"}
    if (
        not isinstance(events, pd.DataFrame)
        or not events.columns.is_unique
        or not required_columns.issubset(events.columns)
    ):
        raise ValueError("events must have unique firm_id, cohort, and dose columns.")
    event_firms = _validate_aggregation_firms(events["firm_id"], "events.firm_id")
    eligible = _validate_aggregation_firms(eligible_firms, "eligible_firms")
    if not set(eligible).issubset(set(event_firms)):
        raise ValueError("Every eligible firm must be present in events.")
    cohorts = events["cohort"]
    if (
        cohorts.dtype != np.dtype("datetime64[ns]")
        or cohorts.isna().any()
        or not cohorts.equals(cohorts.dt.normalize())
    ):
        raise ValueError(
            "events.cohort must contain normalized timezone-naive periods.",
        )
    event_doses = _numeric(events["dose"], "events.dose", positive=True)

    selected = events["firm_id"].isin(eligible).to_numpy()
    selected_doses = event_doses[selected]
    cohort_codes, cohort_periods = pd.factorize(cohorts[selected], sort=True)
    rows: list[dict[str, Any]] = []
    for dose in dose_values:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                log_kernel = -0.5 * np.square(
                    (selected_doses - dose) / kernel_bandwidth,
                )
        except FloatingPointError as exc:
            raise ValueError("Gaussian-kernel calculations overflowed.") from exc
        if not np.isfinite(log_kernel).all():
            raise ValueError("Gaussian-kernel calculations must remain finite.")
        # Removing the common offset first preserves cohort counts even when
        # all unshifted log-kernel values have a very large magnitude.
        log_kernel -= np.max(log_kernel)
        log_mass: FloatArray = np.asarray(
            [logsumexp(log_kernel[cohort_codes == code], axis=0) for code in range(len(cohort_periods))],
            dtype=np.float64,
        )
        weights = np.exp(log_mass - logsumexp(log_mass, axis=0))
        if not np.isfinite(weights).all() or not np.isclose(
            weights.sum(),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Cohort weights must be finite and sum to one.")
        rows.extend(
            {"cohort": cohort, "dose": float(dose), "weight": float(weight)}
            for cohort, weight in zip(cohort_periods, weights, strict=True)
        )
    return pd.DataFrame(rows).sort_values(["cohort", "dose"]).reset_index(drop=True)


def evaluate_theta(
    model: ATT,
    doses: Iterable[float],
    *,
    bandwidth: float,
    event_times: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Aggregate fitted cohort effects at each dose and event period.

    Args:
        model:
            Successfully fitted ATT instance with comparison diagnostics and
            saved component coefficients. This function does not refit it.

        doses:
            Distinct positive finite doses at which to evaluate effects.

        bandwidth:
            Positive finite Gaussian-kernel bandwidth in dose units.

        event_times:
            Distinct fitted integer event periods to report. None selects the
            complete fitted interval, including its reference period.

    Returns:
        A DataFrame sorted by event_time and dose, with theta, estimate_type,
        n_cohorts_eligible, n_cohorts_used, n_firms_design, n_firms_eligible,
        n_firms_used, availability_restricted, restricted_target, and
        extrapolated columns in addition to those two identifiers. Any omitted
        comparisons, restricted horizons, and extrapolation are reported in one
        aggregate warning for the complete successful evaluation.

    Raises:
        ValueError:
            If requests are invalid, no eligible firms remain at a requested
            event period, a required comparison is unavailable, or component
            predictions and kernel weights do not match completely.

        RuntimeError:
            If the model has not been fitted successfully.
    """
    model._require_fitted()
    dose_values = _validate_doses(doses)
    resolved_event_times = model._resolve_event_times(event_times)
    design_events = model.events_.loc[model.events_["design_eligible"]]
    n_firms_design = len(design_events)
    rows: list[dict[str, Any]] = []
    restricted_horizons = 0
    excluded_treated_rows = 0
    extrapolated_predictions = 0
    prediction_doses = np.sort(dose_values)
    for event_time in resolved_event_times:
        availability = model.availability_diagnostics_
        observed = availability.loc[(availability["event_time"] == event_time) & availability["available"]]
        eligible_events = design_events.loc[design_events["firm_id"].isin(observed["firm_id"])]
        eligible_cohorts = set(eligible_events["cohort"])
        retained = model.retained_cohorts_[event_time]
        retained_cohorts = set(retained)
        if len(retained_cohorts) != len(retained) or not retained_cohorts.issubset(
            eligible_cohorts,
        ):
            raise ValueError(f"Invalid retained cohorts at event period {event_time}.")
        used_events = eligible_events.loc[eligible_events["cohort"].isin(retained_cohorts)]
        if used_events.empty:
            raise ValueError(f"No eligible firms remain at event period {event_time}.")
        estimate_type = "reference" if event_time == -1 else "att" if event_time >= 0 else "placebo"
        required_status = "reference" if event_time == -1 else "fitted"
        for cohort in retained:
            cell = model.cell_diagnostics_.get((cohort, event_time))
            if cell is None or cell["status"] != required_status or cell["n_controls"] < 1:
                raise ValueError(
                    f"Required comparison is unavailable for cohort {cohort} at event period {event_time}.",
                )
            expected_firms = set(
                used_events.loc[used_events["cohort"] == cohort, "firm_id"],
            )
            if set(cell["treated_firms"]) != expected_firms:
                raise ValueError(
                    f"Fitted firms and aggregation firms differ for cohort {cohort} at event period {event_time}.",
                )

        weights = estimate_pi(
            model.events_,
            dose_values,
            eligible_firms=used_events["firm_id"],
            bandwidth=bandwidth,
        )
        predictions = model._prediction_rows(prediction_doses, [event_time])
        components = weights.merge(
            predictions,
            on=["cohort", "dose"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if (
            not components["_merge"].eq("both").all()
            or not components["event_time"].eq(event_time).all()
            or not components["estimate_type"].eq(estimate_type).all()
            or not np.isfinite(components["att"].to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                f"Incomplete or invalid component predictions at event period {event_time}.",
            )
        restricted_target = retained_cohorts != eligible_cohorts
        if restricted_target:
            restricted_horizons += 1
        excluded_treated_rows += len(eligible_events) - len(used_events)
        extrapolated_predictions += int(predictions["extrapolated"].sum())
        for dose_key, component in components.groupby("dose", sort=True):
            # The dose column comes from the validated floating-point grid.
            dose = cast(float, dose_key)
            if not np.isclose(
                component["weight"].sum(),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError("Matched cohort weights must sum to one.")
            theta = float(np.dot(component["weight"], component["att"]))
            if not np.isfinite(theta):
                raise ValueError("Aggregated effects must be finite.")
            if event_time == -1 and not component["att"].eq(0.0).all():
                raise ValueError("Reference components must equal zero exactly.")
            rows.append(
                {
                    "event_time": event_time,
                    "dose": float(dose),
                    "theta": theta,
                    "estimate_type": estimate_type,
                    "n_cohorts_eligible": len(eligible_cohorts),
                    "n_cohorts_used": len(retained_cohorts),
                    "n_firms_design": n_firms_design,
                    "n_firms_eligible": len(eligible_events),
                    "n_firms_used": len(used_events),
                    "availability_restricted": len(eligible_events) < n_firms_design,
                    "restricted_target": restricted_target,
                    "extrapolated": False if event_time == -1 else bool(component["extrapolated"].any()),
                },
            )
    requested_horizons = set(resolved_event_times)
    omitted_comparisons = sum(
        event_time in requested_horizons and cell["status"] not in {"fitted", "reference", "no_observed_treated"}
        for (_, event_time), cell in model.cell_diagnostics_.items()
    )
    if omitted_comparisons or restricted_horizons or extrapolated_predictions:
        logging.warning(
            "Aggregation summary: %d unavailable cohort-horizon comparisons; "
            "%d excluded treated firm-horizon rows across %d restricted horizons; "
            "%d extrapolated cohort-horizon-dose predictions. Inspect returned flags and cell_diagnostics_.",
            omitted_comparisons,
            excluded_treated_rows,
            restricted_horizons,
            extrapolated_predictions,
        )
    return pd.DataFrame(rows).sort_values(["event_time", "dose"]).reset_index(drop=True)
