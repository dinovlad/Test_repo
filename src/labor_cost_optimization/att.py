"""Estimate unpooled continuous-dose ATT curves with one stacked PyFixest model.

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
from scipy.interpolate import BSpline
from scipy.special import logsumexp


__all__ = ["ATT", "estimate_pi", "evaluate_theta"]

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
    """Fit separate cohort-horizon dose curves in one stacked OLS model.

    The only fixed effect is the comparison cell. Control means are absorbed
    by those fixed effects, so predictions use the treatment coefficients
    directly. No covariates or inference are implemented.

    Attributes:
        model_:
            The single fitted PyFixest model, available after successful fit.
        events_:
            First-event metadata with firm_id, cohort, dose, design_eligible.
        termination_history_:
            Complete firm-period termination counts, including excluded events.
        bases_:
            Saved spline definitions keyed by (cohort, event_time).
        cell_coefficients_:
            Treatment coefficients from model_, with exact zeros for references.
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
        knots_:
            Shared internal knots selected before cell screening.
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
        num_knots: int = 0,
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
                Positive spline degree.
            num_knots:
                Nonnegative number of shared internal quantile knots.
            min_cohort_size:
                Optional treated-sample minimum. None resolves to degree plus
                num_knots plus two; a smaller explicit minimum cannot override it.

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
        self.num_knots = _integer(num_knots, "num_knots", 0)
        self.min_cohort_size = None if min_cohort_size is None else _integer(min_cohort_size, "min_cohort_size", 1)
        self._reset()

    def _reset(self) -> None:
        """Invalidate fitted state before validating or attempting a new fit."""
        for name in tuple(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self._is_fitted = False
        self.model_: Any = None
        self.column_names_ = {
            "firm_col": self.firm_col,
            "period_col": self.period_col,
            "outcome_col": self.outcome_col,
            "dose_col": self.dose_col,
        }
        self.min_cohort_size_ = max(
            self.degree + self.num_knots + 2,
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
        self.knots_: FloatArray = np.empty(0)

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
        if self.duplicate_diagnostics_["n_dropped_events"]:
            logging.warning(
                "Dropped %d additional termination rows from the target table for %d "
                "firms; kept the first event by period for each firm.",
                self.duplicate_diagnostics_["n_dropped_events"],
                self.duplicate_diagnostics_["n_affected_firms"],
            )
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
        if self.num_knots:
            self.knots_ = np.quantile(
                eligible.to_numpy(dtype=float),
                np.arange(1, self.num_knots + 1) / (self.num_knots + 1),
                method="linear",
            )
            if len(np.unique(self.knots_)) != self.num_knots:
                raise ValueError(
                    "Global internal knots are duplicated; change the configuration.",
                )

    def _basis(self, doses: FloatArray) -> tuple[str, dict[str, Any], FloatArray]:
        """Screen a treated design and construct its saved spline basis.

        Rank uses NumPy's SVD tolerance: largest singular value multiplied by
        max(number of observations, number of columns) and machine precision.
        """
        p = self.degree + self.num_knots + 1
        empty = np.empty((0, p))
        if len(doses) < self.min_cohort_size_:
            return "small_sample", {}, empty
        if len(np.unique(doses)) < p:
            return "insufficient_doses", {}, empty
        lower, upper = float(doses.min()), float(doses.max())
        if np.any(self.knots_ <= lower) or np.any(self.knots_ >= upper):
            return "incompatible_knots", {}, empty
        knots = np.r_[
            np.repeat(lower, self.degree + 1),
            self.knots_,
            np.repeat(upper, self.degree + 1),
        ]
        spline = BSpline(knots, np.eye(p), self.degree, extrapolate=True)
        design = np.column_stack([np.ones(len(doses)), spline(doses)[:, 1:]])
        if not np.isfinite(design).all() or np.linalg.matrix_rank(design) < p:
            return "rank_deficient", {}, empty
        return "ready", {"spline": spline, "lower": lower, "upper": upper}, design

    def fit(
        self,
        outcomes: pd.DataFrame,
        events: pd.DataFrame,
        *,
        isolation_window: tuple[int, int],
    ) -> Self:
        """Fit one stacked regression for every eligible non-reference horizon.

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

        Raises:
            ValueError:
                Input values, periods, mappings, or knots are invalid, or no
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
        self._prepare_events(outcome, event)
        by_period = {
            period: group.set_index("firm_id")["outcome"]
            for period, group in outcome.groupby("period", sort=False)
        }
        empty = pd.Series(dtype=float)
        availability: list[dict[str, Any]] = []
        blocks: list[tuple[Cell, pd.DataFrame, FloatArray]] = []
        eligible = self.events_.loc[self.events_["design_eligible"]]
        history = self.termination_history_
        for cohort_key, targets in eligible.groupby("cohort", sort=True):
            # Input validation guarantees normalized datetime64[ns] group keys.
            cohort = cast(pd.Timestamp, cohort_key)
            baseline = cohort - _PERIOD
            base = by_period.get(baseline, empty)
            conflict = history["cohort"].between(
                cohort + lower * _PERIOD,
                cohort + upper * _PERIOD,
            )
            blocked = set(history.loc[conflict, "firm_id"])
            for e in self.event_times_:
                key = (cohort, e)
                period = cohort + e * _PERIOD
                endpoint = by_period.get(period, empty)
                for row in targets.itertuples(index=False):
                    missing_base = row.firm_id not in base.index
                    missing_end = row.firm_id not in endpoint.index
                    availability.append(
                        {
                            "firm_id": row.firm_id,
                            "cohort": cohort,
                            "event_time": e,
                            "baseline": baseline,
                            "period": period,
                            "missing_baseline": missing_base,
                            "missing_endpoint": missing_end,
                            "available": not (missing_base or missing_end),
                        },
                    )
                common = base.index.intersection(endpoint.index, sort=False)
                with np.errstate(over="ignore", invalid="ignore"):
                    changes = endpoint.reindex(common) - base.reindex(common)
                if not np.isfinite(changes.to_numpy()).all():
                    raise ValueError("Outcome differences must be finite.")
                treated = targets.loc[targets["firm_id"].isin(common)]
                treated_firms = tuple(treated["firm_id"])
                control_firms = tuple(firm for firm in common if firm not in blocked)
                doses = treated["dose"].to_numpy(dtype=float)
                diagnostic = {
                    "status": "no_observed_treated",
                    "n_treated": len(treated_firms),
                    "n_controls": len(control_firms),
                    "treated_firms": treated_firms,
                    "control_firms": control_firms,
                    "estimate_type": _estimate_type(e),
                    "n_distinct_doses": len(np.unique(doses)),
                    "design_dimension": self.degree + self.num_knots + 1,
                }
                self.cell_diagnostics_[key] = diagnostic
                if not treated_firms:
                    continue
                status, basis, design = self._basis(doses)
                diagnostic["status"] = status
                if status != "ready":
                    logging.warning(
                        "Dropped cohort %s, event period %d: %s; %d treated firms, "
                        "%d distinct doses, design dimension %d.",
                        cohort,
                        e,
                        status,
                        len(treated_firms),
                        len(np.unique(doses)),
                        self.degree + self.num_knots + 1,
                    )
                    continue
                self.retained_cohorts_[e].append(cohort)
                if not control_firms:
                    diagnostic["status"] = "no_controls"
                    logging.warning(
                        "No clean controls for cohort %s, event period %d.",
                        cohort,
                        e,
                    )
                    continue
                self.bases_[key] = basis
                if e == -1:
                    diagnostic["status"] = "reference"
                    self.cell_coefficients_[key] = np.zeros(design.shape[1])
                    continue
                firms = [*treated_firms, *control_firms]
                block = pd.DataFrame(
                    {
                        "firm_id": firms,
                        "cell": len(blocks),
                        "cohort": cohort,
                        "period": period,
                        "event_time": e,
                        "delta_y": changes.reindex(firms).to_numpy(),
                        "treated": [1] * len(treated_firms) + [0] * len(control_firms),
                    },
                )
                blocks.append((key, block, design))
        self.availability_diagnostics_ = pd.DataFrame(availability)
        self._summarize_cohorts()
        if not blocks:
            raise ValueError("No non-reference comparison passes the required checks.")
        self._fit_model(blocks)
        self._summarize_cohorts()
        self._is_fitted = True
        return self

    def _fit_model(self, blocks: list[tuple[Cell, pd.DataFrame, FloatArray]]) -> None:
        """Fit every prepared cell in one model and extract treatment coefficients."""
        stacked = pd.concat([block for _, block, _ in blocks], ignore_index=True)
        p = self.degree + self.num_knots + 1
        effects = np.zeros((len(stacked), p * len(blocks)))
        names: list[str] = []
        offset = 0
        for cell, (key, block, design) in enumerate(blocks):
            cell_names = [f"effect_{cell}_{k}" for k in range(p)]
            self.cell_keys_[cell] = key
            self.effect_columns_[key] = cell_names
            names.extend(cell_names)
            effects[offset : offset + len(design), cell * p : (cell + 1) * p] = design
            offset += len(block)
        stacked = pd.concat([stacked, pd.DataFrame(effects, columns=names)], axis=1)
        try:
            model = pf.feols("delta_y ~ " + " + ".join(names) + " | cell", data=stacked)
            coefficients = model.coef()
            if any(name not in coefficients.index for name in names):
                raise RuntimeError("PyFixest omitted a required treatment coefficient.")
            if not np.isfinite(coefficients.loc[names].to_numpy()).all():
                raise RuntimeError(
                    "PyFixest returned a nonfinite treatment coefficient.",
                )
        except Exception as exc:
            for key, _, _ in blocks:
                self.cell_diagnostics_[key]["status"] = "model_failed"
            self._summarize_cohorts()
            raise RuntimeError(
                "The single PyFixest model could not be fitted completely.",
            ) from exc
        self.model_ = model
        for key, names_for_cell in self.effect_columns_.items():
            self.cell_coefficients_[key] = coefficients.loc[names_for_cell].to_numpy(
                dtype=float,
            )
            self.cell_diagnostics_[key]["status"] = "fitted"

    def _summarize_cohorts(self) -> None:
        """Summarize design eligibility and horizon statuses without dropping history."""
        records = []
        for cohort, events in self.events_.groupby("cohort", sort=True):
            cells = {e: cell for (g, e), cell in self.cell_diagnostics_.items() if g == cohort}
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
                Doses or horizons are invalid, or spline prediction is nonfinite.
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
        records = []
        extrapolated_count = 0
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
                    predictions = coefficients[0] + basis["spline"](dose_values)[:, 1:] @ coefficients[1:]
                if not np.isfinite(predictions).all():
                    raise ValueError(
                        "Spline predictions are nonfinite at the requested doses.",
                    )
                outside = (dose_values < basis["lower"]) | (dose_values > basis["upper"])
            extrapolated_count += int(outside.sum())
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
        if extrapolated_count:
            logging.warning(
                "Extrapolating %d predictions beyond fitted dose support.",
                extrapolated_count,
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
            [
                logsumexp(log_kernel[cohort_codes == code], axis=0)
                for code in range(len(cohort_periods))
            ],
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
        extrapolated columns in addition to those two identifiers.

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
    restricted_event_times = []
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
                    f"Required comparison is unavailable for cohort {cohort} "
                    f"at event period {event_time}.",
                )
            expected_firms = set(
                used_events.loc[used_events["cohort"] == cohort, "firm_id"],
            )
            if set(cell["treated_firms"]) != expected_firms:
                raise ValueError(
                    f"Fitted firms and aggregation firms differ for cohort {cohort} "
                    f"at event period {event_time}.",
                )

        weights = estimate_pi(
            model.events_,
            dose_values,
            eligible_firms=used_events["firm_id"],
            bandwidth=bandwidth,
        )
        predictions = model.predict(dose_values, event_times=[event_time])
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
            restricted_event_times.append(event_time)
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
    if restricted_event_times:
        logging.warning(
            "Aggregation is restricted to retained cohorts at event periods %s.",
            sorted(restricted_event_times),
        )
    return pd.DataFrame(rows).sort_values(["event_time", "dose"]).reset_index(drop=True)
