"""Estimate unpooled continuous-dose ATT curves with separate polynomial models.

Periods have a fixed daily frequency. Firm-level bootstrap inference accompanies
point estimates; negative horizons are placebo contrasts and horizon -1 is a zero reference.
See att.md for the identifying assumptions and selected-population estimand.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pydantic
import pyfixest as pf
import scipy.sparse
import scipy.special


__all__ = ["ATT", "bootstrap_theta", "estimate_h0", "estimate_h_cv", "estimate_pi", "evaluate_theta"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
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


_VALIDATION_CONFIG = pydantic.ConfigDict(arbitrary_types_allowed=True)
_Integer = Annotated[
    int,
    pydantic.Field(strict=True),
    pydantic.BeforeValidator(lambda value: int(value) if isinstance(value, np.integer) else value),
]
_PositiveInteger = Annotated[_Integer, pydantic.Field(ge=1)]
_ColumnName = Annotated[str, pydantic.Field(strict=True, min_length=1)]
# Instance checks prevent scalar-convertible arrays or objects from becoming grid entries.
_RealScalar = (
    pydantic.InstanceOf[int]
    | pydantic.InstanceOf[float]
    | pydantic.InstanceOf[np.integer]
    | pydantic.InstanceOf[np.floating]
)
# Older Pydantic annotations express schema inputs as type[T], which excludes union objects.
_REAL_SCALAR: pydantic.TypeAdapter[_RealScalar] = pydantic.TypeAdapter(
    cast(type[_RealScalar], _RealScalar),
    config=_VALIDATION_CONFIG,
)
_Real = Annotated[
    float,
    pydantic.Field(strict=True, allow_inf_nan=False),
    pydantic.BeforeValidator(lambda value: float(value) if isinstance(value, (np.integer, np.floating)) else value),
    pydantic.BeforeValidator(_REAL_SCALAR.validate_python),
]
_PositiveReal = Annotated[_Real, pydantic.Field(gt=0)]
_EventTimes = Annotated[list[_Integer], pydantic.Field(min_length=1)]
_FirmList = Annotated[list[object], pydantic.Field(min_length=1)]
_COLUMN_NAMES: pydantic.TypeAdapter[list[str]] = pydantic.TypeAdapter(list[_ColumnName])
_DATAFRAME: pydantic.TypeAdapter[pd.DataFrame] = pydantic.TypeAdapter(pd.DataFrame, config=_VALIDATION_CONFIG)
_EVENT_TIMES: pydantic.TypeAdapter[list[int]] = pydantic.TypeAdapter(_EventTimes)
_FIRM_LIST: pydantic.TypeAdapter[list[object]] = pydantic.TypeAdapter(_FirmList)


class _ATTOptions(pydantic.BaseModel):
    """Validate constructor configuration without coercing its scalar types.

    Attributes:
        firm_col:
            Nonempty firm identifier column name.
        period_col:
            Nonempty period column name.
        outcome_col:
            Nonempty outcome column name.
        dose_col:
            Nonempty dose column name.
        degree:
            Positive polynomial degree.
        min_cohort_size:
            Optional positive treated-sample minimum.
    """

    model_config = _VALIDATION_CONFIG

    firm_col: _ColumnName
    period_col: _ColumnName
    outcome_col: _ColumnName
    dose_col: _ColumnName
    degree: _PositiveInteger
    min_cohort_size: _PositiveInteger | None


class _FitInputs(pydantic.BaseModel):
    """Validate table types and isolation offsets after clearing fitted state.

    Attributes:
        outcomes:
            Original outcome table, retained without copying or coercion.
        events:
            Original event table, retained without copying or coercion.
        isolation_window:
            Two integer offsets spanning the reference and treatment periods.
    """

    model_config = _VALIDATION_CONFIG

    outcomes: pd.DataFrame
    events: pd.DataFrame
    isolation_window: tuple[Annotated[_Integer, pydantic.Field(le=-1)], Annotated[_Integer, pydantic.Field(ge=0)]]


def _distinct_grid(values: list[float]) -> list[float]:
    """Enforce grid uniqueness after Pydantic validates each numeric element.

    Args:
        values:
            Positive finite numeric values in their supplied order.

    Returns:
        The supplied values without sorting or deduplication.

    Raises:
        ValueError:
            The grid contains duplicate values.
    """
    if len(set(values)) != len(values):
        raise ValueError("Grid values must be distinct.")
    return values


_PositiveGrid = Annotated[list[_PositiveReal], pydantic.Field(min_length=1), pydantic.AfterValidator(_distinct_grid)]
_POSITIVE_GRID: pydantic.TypeAdapter[list[float]] = pydantic.TypeAdapter(_PositiveGrid)
# Decorated functions skip lazy Iterable validation and use these eager adapters in their bodies.


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Validate a DataFrame's unique labels and required mapped columns.

    Args:
        frame:
            Input table whose type and labels are validated without modification.
        columns:
            Required column names.
        name:
            Argument name used in validation messages.

    Raises:
        ValueError:
            The input is not a DataFrame, its labels are duplicated, or required
            columns are absent.
    """
    try:
        _DATAFRAME.validate_python(frame)
    except pydantic.ValidationError:
        raise ValueError(f"{name} must be a pandas DataFrame.") from None
    if not frame.columns.is_unique:
        raise ValueError(f"{name} must have unique column labels.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")


def _firm_ids(values: pd.Series) -> None:
    """Require nonmissing, hashable firm identifiers without coercion.

    Args:
        values:
            Firm identifiers from a validated input table.

    Raises:
        ValueError:
            At least one identifier is missing or unhashable.
    """
    if values.isna().any():
        raise ValueError("Firm identifiers must not be missing.")
    try:
        for value in values:
            hash(value)
    except TypeError as exc:
        raise ValueError("Firm identifiers must be hashable.") from exc


def _periods(values: pd.Series) -> pd.Series:
    """Normalize timezone-naive datetime64[ns] period labels on a copy.

    Args:
        values:
            Period labels whose time components are discarded.

    Returns:
        A Series of normalized periods with the original index.

    Raises:
        ValueError:
            Periods are missing or do not have timezone-naive datetime64[ns] dtype.
    """
    if values.dtype != np.dtype("datetime64[ns]") or values.isna().any():
        raise ValueError("Periods must be nonmissing timezone-naive datetime64[ns].")
    return values.dt.normalize()


def _numeric(values: pd.Series, name: str, *, positive: bool = False) -> FloatArray:
    """Require finite real numeric values on every supplied row.

    Args:
        values:
            Input values with a real numeric dtype.
        name:
            Argument name used in validation messages.
        positive:
            Whether every value must also be strictly positive.

    Returns:
        A floating-point array of the validated values.

    Raises:
        ValueError:
            The dtype is not real numeric, a value is nonfinite, or a value fails
            the requested positivity restriction.
    """
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
    """Return the interpretation of a relative horizon.

    Args:
        event_time:
            Integer number of periods relative to the first event.

    Returns:
        The reference, placebo, or treatment-effect label for the horizon.
    """
    return "reference" if event_time == -1 else "placebo" if event_time < -1 else "att"


class ATT:
    """Fit a separate polynomial OLS model for each cohort-horizon dose curve.

    Each model's intercept absorbs its control mean, so predictions use the
    treatment coefficients directly. Saved dose scaling is reused at prediction.
    The separate `bootstrap_theta()` function provides firm-level inference.
    No additional covariates are implemented.

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
            Treatment coefficients from `models_`, with exact zeros for references.
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
            Inclusive elapsed-period index of `last_period_` from `origin_period_`.
        isolation_window_:
            Signed lower and upper screening offsets.
        cell_keys_:
            Mapping from internal comparison labels to cohort-horizon keys.
        effect_columns_:
            Ordered model coefficient names for each fitted non-reference cell.
        bootstrap_firms_:
            Every original outcome firm in first-occurrence order, including
            firms that do not contribute to a fitted comparison.
        outcome_fingerprint_:
            SHA-256 digest of the normalized outcome panel, independent of row
            order, used to verify supplied bootstrap outcomes without caching them.
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
        options = _ATTOptions(
            firm_col=firm_col,
            period_col=period_col,
            outcome_col=outcome_col,
            dose_col=dose_col,
            degree=degree,
            min_cohort_size=min_cohort_size,
        )
        names = (options.firm_col, options.period_col, options.outcome_col, options.dose_col)
        if len(set(names[:3])) != 3 or len({firm_col, period_col, dose_col}) != 3:
            raise ValueError("Mapped column names must be distinct within each table.")
        self.firm_col = options.firm_col
        self.period_col = options.period_col
        self.outcome_col = options.outcome_col
        self.dose_col = options.dose_col
        self.degree = options.degree
        self.min_cohort_size = options.min_cohort_size
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
        self.bootstrap_firms_ = pd.Index([], dtype=object, tupleize_cols=False)
        self.outcome_fingerprint_: str | None = None

    def _inputs(
        self,
        outcomes: pd.DataFrame,
        events: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and copy the two tables into internal column conventions.

        Args:
            outcomes:
                Firm-period outcomes using the configured column mappings.
            events:
                Complete termination history using the configured column mappings.

        Returns:
            Separate canonical outcome and event tables without changing the inputs.

        Raises:
            ValueError:
                Tables, labels, identifiers, periods, or numeric values violate the
                required schema, outcome rows repeat, or event firms are unknown.
        """
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
        """Select first events and screen their complete isolation intervals.

        Args:
            outcome:
                Validated canonical outcome panel defining the observed period bounds.
            event:
                Validated canonical termination history, including repeated events.

        Raises:
            ValueError:
                No first event has a complete and isolated panel window.
        """
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

        Args:
            doses:
                Validated treated doses for one cohort-horizon comparison.

        Returns:
            The screening status, saved dose transformation, and polynomial matrix.
            Rejected comparisons return empty transformation and matrix objects.

        Raises:
            numpy.linalg.LinAlgError:
                Singular-value decomposition fails while checking polynomial rank.
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
            Exclusions are recorded in `drop_summary_` and logged once per fit.

        Raises:
            ValueError:
                Input values, periods, or mappings are invalid, or no
                non-reference comparison passes the design checks.
            RuntimeError:
                PyFixest fitting fails or omits a required finite coefficient.
            numpy.linalg.LinAlgError:
                Singular-value decomposition fails while screening a comparison.
        """
        self._reset()
        inputs = _FitInputs(outcomes=outcomes, events=events, isolation_window=isolation_window)
        lower, upper = inputs.isolation_window
        self.isolation_window_ = (lower, upper)
        self.event_times_ = list(range(lower, upper + 1))
        self.retained_cohorts_ = {e: [] for e in self.event_times_}
        outcome, event = self._inputs(inputs.outcomes, inputs.events)
        self.bootstrap_firms_ = pd.Index(outcome["firm_id"].drop_duplicates().to_numpy(copy=True), tupleize_cols=False)
        self.outcome_fingerprint_ = _panel_fingerprint(outcome, self.bootstrap_firms_)
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
            numpy.linalg.LinAlgError:
                Singular-value decomposition fails while screening a comparison.
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
        """Fit narrow cell models without constructing a global interaction matrix.

        Args:
            blocks:
                Prepared comparison keys, regression tables, and treated basis matrices.

        Raises:
            RuntimeError:
                A PyFixest fit fails or returns missing or nonfinite coefficients.
                All attempted non-reference fits are invalidated together.
        """
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
        """Reject prediction or aggregation without a successful current fit.

        Raises:
            RuntimeError:
                The estimator has no successful current fit.
        """
        if not self._is_fitted:
            raise RuntimeError("ATT must be successfully fitted before evaluation.")

    def _resolve_event_times(self, event_times: Iterable[int] | None) -> list[int]:
        """Validate a horizon subset without fitting or changing eligibility.

        Args:
            event_times:
                Distinct fitted integer horizons, or None to select the entire window.

        Returns:
            Requested horizons sorted in increasing order.

        Raises:
            ValueError:
                Requested horizons are not a nonempty collection of distinct fitted
                integers.
        """
        if event_times is None:
            return list(self.event_times_)
        values = _EVENT_TIMES.validate_python(event_times)
        if len(set(values)) != len(values):
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
        dose_values = np.sort(np.asarray(_POSITIVE_GRID.validate_python(doses), dtype=float))
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
            Component prediction rows with the same schema as `predict()`.

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
            copy=False,
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
    values = _FIRM_LIST.validate_python(firms)
    if pd.Series(values, dtype=object).isna().any():
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
    """Validate event inputs and select one first event per requested firm.

    Args:
        events:
            Initial termination table without an imposed outcome-panel screen.
        firm_col:
            Firm identifier column.
        period_col:
            Event period column with timezone-naive datetime64[ns] values.
        dose_col:
            Positive finite dose column.
        eligible_firms:
            Optional unique firm identifiers to retain after first-event selection.

    Returns:
        Canonical first-event rows for the selected firms, with a fresh row index.

    Raises:
        ValueError:
            Column mappings, input values, or eligibility are invalid, or fewer
            than two selected firms remain.
    """
    columns = [firm_col, period_col, dose_col]
    try:
        _COLUMN_NAMES.validate_python(columns)
    except pydantic.ValidationError:
        raise ValueError("Bandwidth column names must be distinct nonempty strings.") from None
    if len(set(columns)) != 3:
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
    """Compute the initial bandwidth from validated first-event doses.

    Args:
        doses:
            Positive finite doses for at least two selected firms.

    Returns:
        Silverman's starting bandwidth in dose units.

    Raises:
        ValueError:
            The selected doses do not have positive finite variation.
    """
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
    cohort_codes: npt.NDArray[np.int64],
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
    counts = scipy.sparse.csr_matrix(
        (np.ones(len(doses)), (dose_codes, cohort_codes)),
        shape=(len(unique_doses), int(cohort_codes.max()) + 1),
    )
    losses_by_dose = np.empty((len(unique_doses), len(bandwidths)))
    for start in range(0, len(unique_doses), batch_size):
        stop = min(start + batch_size, len(unique_doses))
        query_indices = np.arange(start, stop)
        singleton = multiplicities[start:stop] == 1
        self_weight = (~singleton).astype(float, copy=False)
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
    cohort_codes: npt.NDArray[np.int64],
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


@pydantic.validate_call(config=_VALIDATION_CONFIG)
def estimate_h_cv(
    events: pd.DataFrame,
    *,
    firm_col: _ColumnName,
    period_col: _ColumnName,
    dose_col: _ColumnName,
    eligible_firms: pydantic.SkipValidation[Iterable[object] | None] = None,
    bandwidths: pydantic.SkipValidation[Iterable[float] | None] = None,
    batch_size: _PositiveInteger = 256,
    max_expansions: _PositiveInteger = 50,
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
        candidates = np.sort(np.asarray(_POSITIVE_GRID.validate_python(bandwidths), dtype=float))
    except pydantic.ValidationError as exc:
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


@pydantic.validate_call(config=_VALIDATION_CONFIG)
def estimate_pi(
    events: pd.DataFrame,
    doses: pydantic.SkipValidation[Iterable[float]],
    *,
    eligible_firms: pydantic.SkipValidation[Iterable[object]],
    bandwidth: _PositiveReal,
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
    dose_values = np.asarray(_POSITIVE_GRID.validate_python(doses), dtype=float)
    kernel_bandwidth = bandwidth
    try:
        _require_columns(events, ["firm_id", "cohort", "dose"], "events")
    except ValueError:
        raise ValueError("events must have unique firm_id, cohort, and dose columns.") from None
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
            [scipy.special.logsumexp(log_kernel[cohort_codes == code], axis=0) for code in range(len(cohort_periods))],
            dtype=np.float64,
        )
        weights = np.exp(log_mass - scipy.special.logsumexp(log_mass, axis=0))
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
    return _evaluate_theta(model, doses, bandwidth=bandwidth, event_times=event_times, log_warning=True)


def _evaluate_theta(
    model: ATT,
    doses: Iterable[float],
    *,
    bandwidth: float,
    event_times: Iterable[int] | None,
    log_warning: bool,
) -> pd.DataFrame:
    """Evaluate the original aggregation with optional diagnostic logging.

    Args:
        model:
            Successfully fitted estimator.
        doses:
            Distinct positive finite evaluation doses.
        bandwidth:
            Positive finite kernel bandwidth.
        event_times:
            Requested fitted horizons, or None for all horizons.
        log_warning:
            Whether to log the original aggregation diagnostics.

    Returns:
        The same estimates and support flags as `evaluate_theta()`.

    Raises:
        ValueError:
            Requests, comparison metadata, or aggregation inputs are invalid.
        RuntimeError:
            The model has no successful current fit.
    """
    model._require_fitted()
    dose_values = np.asarray(_POSITIVE_GRID.validate_python(doses), dtype=float)
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
    if log_warning and (omitted_comparisons or restricted_horizons or extrapolated_predictions):
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


def _panel_fingerprint(outcome: pd.DataFrame, firms: pd.Index) -> str:
    """Fingerprint canonical outcomes independently of row order.

    Args:
        outcome:
            Validated canonical firm-period outcomes with finite numeric values.
        firms:
            Complete original firm index defining stable integer firm codes.

    Returns:
        A SHA-256 digest of ordered firm codes, normalized periods, and outcomes.

    Raises:
        ValueError:
            The panel does not contain exactly the original firm population.
    """
    codes = firms.get_indexer(outcome["firm_id"])
    if (codes < 0).any() or len(np.unique(codes)) != len(firms):
        raise ValueError("Bootstrap outcomes must contain exactly the original firms.")
    periods = outcome["period"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // _PERIOD_NS
    order = np.lexsort((periods, codes))
    fingerprint = hashlib.sha256()
    for values, dtype in (
        (codes, "<i8"),
        (periods, "<i8"),
        (outcome["outcome"].to_numpy(dtype=float), "<f8"),
    ):
        fingerprint.update(np.asarray(values[order], dtype=dtype).tobytes())
    return fingerprint.hexdigest()


def _bootstrap_panel(model: ATT, outcomes: pd.DataFrame) -> tuple[pd.Index, FloatArray, int, int]:
    """Verify supplied outcomes and construct one shared panel lookup.

    Args:
        model:
            Fitted estimator with original firm metadata and a panel fingerprint.
        outcomes:
            Original outcome table using the estimator's column mappings.

    Returns:
        Packed firm-period keys, numeric outcomes, period origin, and period span.

    Raises:
        ValueError:
            The supplied table is invalid or differs from the fitted outcome panel.
        RuntimeError:
            The model predates the required bootstrap metadata and must be refitted.
    """
    if not hasattr(model, "bootstrap_firms_") or not getattr(model, "outcome_fingerprint_", None):
        raise RuntimeError("Refit ATT before bootstrapping to retain original outcome metadata.")
    columns = [model.firm_col, model.period_col, model.outcome_col]
    _require_columns(outcomes, columns, "outcomes")
    if outcomes.empty:
        raise ValueError("outcomes must contain at least one row.")
    outcome = outcomes[columns].copy()
    outcome.columns = ["firm_id", "period", "outcome"]
    _firm_ids(outcome["firm_id"])
    outcome["period"] = _periods(outcome["period"])
    outcome["outcome"] = _numeric(outcome["outcome"], "Outcomes")
    if outcome.duplicated(["firm_id", "period"]).any():
        raise ValueError("Outcome rows must be unique by firm and normalized period.")
    if _panel_fingerprint(outcome, model.bootstrap_firms_) != model.outcome_fingerprint_:
        raise ValueError("Bootstrap outcomes must match the complete original outcome panel.")
    codes = model.bootstrap_firms_.get_indexer(outcome["firm_id"])
    periods = outcome["period"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // _PERIOD_NS
    origin = int(periods.min())
    span = int(periods.max()) - origin + 1
    keys = pd.Index(codes * span + periods - origin)
    return keys, outcome["outcome"].to_numpy(dtype=float), origin, span


def _bootstrap_counts(rng: np.random.Generator, n_firms: int, size: int) -> IntArray:
    """Draw complete-firm resampling frequencies for a bounded batch.

    Args:
        rng:
            Local random generator shared by the complete bootstrap call.
        n_firms:
            Number of original outcome firms, including control-only firms.
        size:
            Number of bootstrap replications in this batch.

    Returns:
        Integer sampling counts with shape `(size, n_firms)` and row sums `n_firms`.
    """
    return rng.multinomial(n_firms, np.full(n_firms, 1.0 / n_firms), size=size).astype(np.int64, copy=False)


def _bootstrap_changes(
    codes: IntArray,
    cohort: pd.Timestamp,
    event_time: int,
    panel: tuple[pd.Index, FloatArray, int, int],
) -> FloatArray:
    """Recover a saved comparison's exact observed outcome differences.

    Args:
        codes:
            Original firm positions for one treated or control sample.
        cohort:
            Focal cohort period determining the common baseline.
        event_time:
            Horizon whose endpoint is relative to the focal cohort.
        panel:
            Shared packed keys, numeric outcomes, origin, and period span.

    Returns:
        Outcome differences between the exact endpoint and baseline for each firm.

    Raises:
        ValueError:
            Saved membership references an unknown firm or an unavailable endpoint.
        RuntimeError:
            Outcome subtraction produces nonfinite values.
    """
    if (codes < 0).any():
        raise ValueError("Saved comparison firms must belong to the original outcome panel.")
    keys, values, origin, span = panel
    period = cohort.value // _PERIOD_NS
    baseline = keys.get_indexer(codes * span + period - 1 - origin)
    endpoint = keys.get_indexer(codes * span + period + event_time - origin)
    if (baseline < 0).any() or (endpoint < 0).any():
        raise ValueError("Saved comparison endpoints must exist in the original outcome panel.")
    try:
        with np.errstate(over="raise", invalid="raise"):
            differences = values[endpoint] - values[baseline]
    except FloatingPointError as exc:
        raise RuntimeError("Bootstrap outcome differences overflowed.") from exc
    if not np.isfinite(differences).all():
        raise RuntimeError("Bootstrap outcome differences must remain finite.")
    return differences


def _bootstrap_cell_arrays(
    model: ATT,
    key: Cell,
    panel: tuple[pd.Index, FloatArray, int, int],
    firm_doses: FloatArray,
    doses: FloatArray,
) -> tuple[IntArray, FloatArray, FloatArray, IntArray, FloatArray, FloatArray]:
    """Prepare one cell without retaining copies of every control sample.

    Args:
        model:
            Fitted estimator with saved comparison membership and polynomial bases.
        key:
            Cohort and horizon identifying the retained comparison.
        panel:
            Shared packed outcome lookup.
        firm_doses:
            Original event dose indexed by the complete original firm population.
        doses:
            Requested prediction doses in increasing order.

    Returns:
        Treated positions, polynomial matrix, and outcome differences, followed by
        control positions, their outcome differences, and the scaled prediction doses.

    Raises:
        ValueError:
            Saved comparison metadata does not match the original panel or events.
        RuntimeError:
            Polynomial or outcome calculations produce nonfinite values.
    """
    cell = model.cell_diagnostics_[key]
    treated = model.bootstrap_firms_.get_indexer(pd.Index(cell["treated_firms"], tupleize_cols=False))
    controls = model.bootstrap_firms_.get_indexer(pd.Index(cell["control_firms"], tupleize_cols=False))
    treated_changes = _bootstrap_changes(treated, key[0], key[1], panel)
    control_changes = _bootstrap_changes(controls, key[0], key[1], panel)
    treated_doses = firm_doses[treated]
    if not np.isfinite(treated_doses).all():
        raise ValueError("Saved treated firms must have original event doses.")
    basis = model.bases_[key]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            width = basis["upper"] - basis["lower"]
            design = np.polynomial.polynomial.polyvander(
                (treated_doses - basis["lower"]) / width,
                basis["degree"],
            )
            prediction = (doses - basis["lower"]) / width
    except FloatingPointError as exc:
        raise RuntimeError("Bootstrap polynomial calculations overflowed.") from exc
    if not np.isfinite(design).all() or not np.isfinite(prediction).all():
        raise RuntimeError("Bootstrap polynomial matrices must remain finite.")
    return treated, design, treated_changes, controls, control_changes, prediction


def _bootstrap_merge(
    mean: FloatArray,
    anchor: FloatArray,
    mass: FloatArray,
    local_anchor: FloatArray,
    local_mass: FloatArray,
    effect: FloatArray,
) -> None:
    """Accumulate one positive-mass cohort using stable sampled-support anchors.

    Args:
        mean:
            Running aggregate predictions to update in place.
        anchor:
            Running maximum raw log-kernel values to update in place.
        mass:
            Running kernel masses scaled to `anchor`, updated in place.
        local_anchor:
            Maximum raw log kernel of sampled treated firms at each requested dose.
        local_mass:
            This cohort's frequency-weighted kernel mass scaled to `local_anchor`.
        effect:
            This cohort's weighted polynomial treatment effects at requested doses.

    Raises:
        RuntimeError:
            Kernel accumulation produces nonfinite masses or predictions.
    """
    next_anchor = np.maximum(anchor, local_anchor)
    old_mass = mass * np.exp(anchor - next_anchor)
    added_mass = local_mass * np.exp(local_anchor - next_anchor)
    total_mass = old_mass + added_mass
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            next_mean = mean * (old_mass / total_mass) + effect * (added_mass / total_mass)
    except FloatingPointError as exc:
        raise RuntimeError("Bootstrap kernel aggregation overflowed.") from exc
    if not np.isfinite(total_mass).all() or (total_mass <= 0).any() or not np.isfinite(next_mean).all():
        raise RuntimeError("Bootstrap kernel masses and estimates must remain finite.")
    mean[:] = next_mean
    anchor[:] = next_anchor
    mass[:] = total_mass


def _bootstrap_horizon(
    model: ATT,
    event_time: int,
    counts: IntArray,
    panel: tuple[pd.Index, FloatArray, int, int],
    firm_doses: FloatArray,
    doses: FloatArray,
    bandwidth: float,
    failures: dict[tuple[pd.Timestamp | None, int, str], int],
    failure_policy: Literal["strict", "available_cohorts"],
) -> tuple[FloatArray, IntArray, IntArray, FloatArray]:
    """Re-estimate one horizon for a bounded batch of firm bootstrap draws.

    Args:
        model:
            Fitted estimator whose original retained comparisons remain fixed.
        event_time:
            Requested horizon.
        counts:
            Shared complete-firm sampling counts for this batch.
        panel:
            Shared packed outcome lookup.
        firm_doses:
            Original event dose at each original firm position.
        doses:
            Requested evaluation grid in increasing order.
        bandwidth:
            Fixed positive Gaussian-kernel bandwidth.
        failures:
            Aggregate comparison failure counts to update in place.
        failure_policy:
            Whether any positive-mass failed cell invalidates the draw or only
            estimable cohorts enter its normalized aggregation.

    Returns:
        Batch-by-dose estimates, available and failed cohort counts per draw, and
        the batch-by-dose share of full sampled kernel mass in available cohorts.
        Unavailable estimates are NaN. Shares are zero when all positive-mass
        cohorts fail and NaN when no treated firms are sampled.

    Raises:
        ValueError:
            Saved sample metadata or Gaussian-kernel calculations are invalid.
        RuntimeError:
            Weighted least squares or other numerical calculations fail unexpectedly.
    """
    batch_size = counts.shape[0]
    estimates = np.zeros((batch_size, len(doses)), dtype=float)
    anchors = np.full_like(estimates, -np.inf)
    masses = np.zeros_like(estimates)
    full_anchors = np.full_like(estimates, -np.inf)
    full_masses = np.zeros_like(estimates)
    total_treated = np.zeros(batch_size, dtype=np.int64)
    n_available = np.zeros(batch_size, dtype=np.int64)
    n_failed = np.zeros(batch_size, dtype=np.int64)
    reason: tuple[pd.Timestamp | None, int, str]
    for cohort in model.retained_cohorts_[event_time]:
        key = (cohort, event_time)
        if event_time == -1:
            treated = model.bootstrap_firms_.get_indexer(
                pd.Index(model.cell_diagnostics_[key]["treated_firms"], tupleize_cols=False),
            )
            if (treated < 0).any():
                raise ValueError("Saved treated firms must belong to the original outcome panel.")
            treated_totals = counts[:, treated].sum(axis=1)
            total_treated += treated_totals
            n_available += treated_totals > 0
            continue
        treated, design, changes, controls, control_changes, prediction = _bootstrap_cell_arrays(
            model,
            key,
            panel,
            firm_doses,
            doses,
        )
        treated_counts = counts[:, treated]
        treated_totals = treated_counts.sum(axis=1)
        total_treated += treated_totals
        control_counts = counts[:, controls]
        control_totals = control_counts.sum(axis=1)
        no_controls = (treated_totals > 0) & (control_totals == 0)
        if no_controls.any():
            reason = (cohort, event_time, "no_controls")
            failures[reason] = failures.get(reason, 0) + int(no_controls.sum())
            n_failed += no_controls
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                log_kernel = -0.5 * np.square((firm_doses[treated, None] - doses) / bandwidth)
        except FloatingPointError as exc:
            raise ValueError("Gaussian-kernel calculations overflowed.") from exc
        if not np.isfinite(log_kernel).all():
            raise ValueError("Gaussian-kernel calculations must remain finite.")
        for draw in np.flatnonzero(treated_totals > 0):
            present = treated_counts[draw] > 0
            weights = treated_counts[draw, present]
            sampled_kernel = log_kernel[present]
            local_anchor = np.max(sampled_kernel, axis=0)
            local_mass = weights @ np.exp(sampled_kernel - local_anchor)
            next_anchor = np.maximum(full_anchors[draw], local_anchor)
            full_masses[draw] = full_masses[draw] * np.exp(full_anchors[draw] - next_anchor) + local_mass * np.exp(
                local_anchor - next_anchor,
            )
            full_anchors[draw] = next_anchor
            if no_controls[draw]:
                continue
            weighted_design = design[present] * np.sqrt(weights[:, None])
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    weighted_changes = changes[present] * np.sqrt(weights)
                    coefficient, _, rank, _ = np.linalg.lstsq(
                        weighted_design,
                        weighted_changes,
                        rcond=float(np.finfo(float).eps) * max(int(treated_totals[draw]), design.shape[1]),
                    )
            except (np.linalg.LinAlgError, FloatingPointError) as exc:
                raise RuntimeError("A bootstrap weighted polynomial calculation failed.") from exc
            if rank < design.shape[1]:
                reason = (cohort, event_time, "rank_deficient")
                failures[reason] = failures.get(reason, 0) + 1
                n_failed[draw] += 1
                continue
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    control_mean = np.dot(control_counts[draw] / control_totals[draw], control_changes)
                    effect = np.polynomial.polynomial.polyval(prediction, coefficient) - control_mean
            except FloatingPointError as exc:
                raise RuntimeError("Bootstrap polynomial predictions overflowed.") from exc
            if not np.isfinite(coefficient).all() or not np.isfinite(effect).all():
                raise RuntimeError("Bootstrap regression coefficients and predictions must remain finite.")
            n_available[draw] += 1
            _bootstrap_merge(
                estimates[draw],
                anchors[draw],
                masses[draw],
                local_anchor,
                local_mass,
                effect,
            )
    no_treated = total_treated == 0
    if no_treated.any():
        reason = (None, event_time, "no_treated_firms")
        failures[reason] = failures.get(reason, 0) + int(no_treated.sum())
    unavailable = n_available == 0
    if failure_policy == "available_cohorts" and (unavailable & ~no_treated).any():
        reason = (None, event_time, "no_available_cohorts")
        failures[reason] = failures.get(reason, 0) + int((unavailable & ~no_treated).sum())
    retained_share = np.full_like(estimates, np.nan)
    retained_share[~no_treated] = 0.0
    available = ~unavailable
    if event_time == -1:
        retained_share[available] = 1.0
    elif available.any():
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # Subtract anchors first to preserve counts at extreme common log offsets.
                log_share = (
                    anchors[available]
                    - full_anchors[available]
                    + np.log(masses[available])
                    - np.log(full_masses[available])
                )
                retained_share[available] = np.exp(np.minimum(log_share, 0.0))
        except FloatingPointError as exc:
            raise RuntimeError("Bootstrap retained kernel weight calculations failed.") from exc
        if not np.isfinite(retained_share[available]).all():
            raise RuntimeError("Bootstrap retained kernel weights must remain finite.")
    if failure_policy == "strict":
        unavailable |= n_failed > 0
    estimates[unavailable] = np.nan
    return estimates, n_available, n_failed, retained_share


@pydantic.validate_call(config=_VALIDATION_CONFIG)
def bootstrap_theta(
    model: ATT,
    doses: pydantic.SkipValidation[Iterable[float]],
    *,
    outcomes: pd.DataFrame,
    bandwidth: _PositiveReal,
    event_times: pydantic.SkipValidation[Iterable[int] | None] = None,
    n_bootstrap: Annotated[_Integer, pydantic.Field(ge=2)] = 1999,
    confidence_level: Annotated[_Real, pydantic.Field(gt=0, lt=1)] = 0.95,
    random_state: Annotated[_Integer, pydantic.Field(ge=0)] | None = None,
    failure_policy: Literal["strict", "available_cohorts"] = "strict",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap dose-specific event-study effects by resampling entire firms.

    Sample the complete original firm population with replacement and reuse each
    firm's count across every comparison, horizon, dose, and kernel calculation.
    Keep the fitted comparison membership, polynomial bases, and bandwidth fixed.
    Percentile intervals and standard errors use available draws for each dose
    and horizon. The available-cohort policy renormalizes kernel weights over
    estimable cells in each draw, allowing its cohort composition to change.
    Exclusions do not establish nominal or conditional frequentist coverage.
    Kernel smoothing bias and bandwidth selection uncertainty are not corrected.
    The fitted model and supplied outcomes are not modified.

    Args:
        model:
            Successfully fitted estimator retaining original firm metadata and its
            normalized outcome fingerprint. Older fitted objects require refitting.
        doses:
            Distinct positive finite doses at which to evaluate the aggregation.
        outcomes:
            The complete original outcome table using the model's column mappings.
            Row order, ignored time components, and additional columns may differ;
            normalized firm-period observations and numeric outcomes must match.
        bandwidth:
            Fixed positive finite Gaussian-kernel bandwidth in dose units.
        event_times:
            Distinct fitted integer horizons, or None to report the full window.
        n_bootstrap:
            Number of attempted resamples, at least two. Failed draws are retained
            and are never replaced by additional resampling.
        confidence_level:
            Nominal interval coverage strictly between zero and one.
        random_state:
            Nonnegative integer seed, or None for nondeterministic local sampling.
        failure_policy:
            With "strict", any failed positive-mass cell makes that horizon draw
            unavailable. With "available_cohorts", aggregate the estimable cells
            using kernel weights normalized over those cohorts; no available
            cohorts means a missing estimate.

    Returns:
        Three DataFrames: the original `evaluate_theta()` summary with bootstrap
        standard errors, percentile bounds, attempted/successful/failed counts,
        success rates, `failure_policy`, `n_partial`, retained-weight summaries,
        and inference status; all attempted draws; and aggregated failures with
        `cohort`, `event_time`, `reason`, and `n_failures` columns. Draw columns are
        `replicate`, `event_time`, `dose`, `theta`, `success`, `n_cohorts_available`,
        `n_cohorts_failed`, and `retained_weight_share`.
        Replicate numbers start at one. Failed draws retain NaN estimates. Fewer
        than two successful draws yield missing standard errors and intervals.
        `n_partial` counts successful values omitting positive-mass failed cohorts.
        `mean_retained_weight_share` and `min_retained_weight_share` summarize the
        available share of full sampled kernel mass among successful pair draws.
        `conditional_on_success` is true when some pair draws are unavailable.
        `inference_status` is `ok`, `successful_draws_only`, `available_cohorts`,
        or `insufficient_successful_draws`, with insufficient draws taking priority.
        Horizon-wide failures use NaT for `cohort`; multiple cell failures can
        belong to one draw and are recorded even when its aggregation succeeds.
        At most one aggregated warning covers failures and original support flags.

    Raises:
        ValueError:
            Inputs, original comparison metadata, the outcome fingerprint, or
            Gaussian-kernel calculations are invalid.
        RuntimeError:
            The model is unfitted, requires refitting for bootstrap metadata, or
            unexpected regression or numerical calculations fail.
    """
    model._require_fitted()
    level = confidence_level
    dose_values = np.sort(np.asarray(_POSITIVE_GRID.validate_python(doses), dtype=float))
    horizons = model._resolve_event_times(event_times)
    panel = _bootstrap_panel(model, outcomes)
    summary = _evaluate_theta(model, dose_values, bandwidth=bandwidth, event_times=horizons, log_warning=False)
    kernel_bandwidth = float(bandwidth)
    n_firms = len(model.bootstrap_firms_)
    event_codes = model.bootstrap_firms_.get_indexer(model.events_["firm_id"])
    if (event_codes < 0).any():
        raise ValueError("Saved event firms must belong to the original outcome panel.")
    firm_doses = np.full(n_firms, np.nan)
    firm_doses[event_codes] = model.events_["dose"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    estimates = np.full((n_bootstrap, len(horizons), len(dose_values)), np.nan)
    retained_shares = np.full_like(estimates, np.nan)
    available_cohorts = np.zeros((n_bootstrap, len(horizons)), dtype=np.int64)
    failed_cohorts = np.zeros_like(available_cohorts)
    failures: dict[tuple[pd.Timestamp | None, int, str], int] = {}
    for start in range(0, n_bootstrap, 32):
        stop = min(start + 32, n_bootstrap)
        counts = _bootstrap_counts(rng, n_firms, stop - start)
        for horizon_index, event_time in enumerate(horizons):
            (
                estimates[start:stop, horizon_index],
                available_cohorts[start:stop, horizon_index],
                failed_cohorts[start:stop, horizon_index],
                retained_shares[start:stop, horizon_index],
            ) = _bootstrap_horizon(
                model,
                event_time,
                counts,
                panel,
                firm_doses,
                dose_values,
                kernel_bandwidth,
                failures,
                failure_policy,
            )
    successful = np.isfinite(estimates)
    n_success = successful.sum(axis=0)
    n_failed = n_bootstrap - n_success
    n_partial = (successful & (failed_cohorts[:, :, None] > 0)).sum(axis=0)
    standard_errors = np.full((len(horizons), len(dose_values)), np.nan)
    lower = np.full_like(standard_errors, np.nan)
    upper = np.full_like(standard_errors, np.nan)
    mean_retained = np.full_like(standard_errors, np.nan)
    min_retained = np.full_like(standard_errors, np.nan)
    tail = (1.0 - level) / 2.0
    for horizon_index in range(len(horizons)):
        for dose_index in range(len(dose_values)):
            key = (horizon_index, dose_index)
            usable = successful[:, horizon_index, dose_index]
            if n_success[key]:
                shares = retained_shares[usable, horizon_index, dose_index]
                mean_retained[key] = np.mean(shares)
                min_retained[key] = np.min(shares)
            if n_success[key] < 2:
                continue
            selected = estimates[usable, horizon_index, dose_index]
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    standard_errors[key] = np.std(selected, ddof=1)
                    lower[key], upper[key] = np.quantile(selected, [tail, 1.0 - tail])
            except FloatingPointError as exc:
                raise RuntimeError("Bootstrap interval calculations overflowed.") from exc
            if not all(np.isfinite(values[key]) for values in (standard_errors, lower, upper)):
                raise RuntimeError("Bootstrap standard errors and interval bounds must remain finite.")
    summary["standard_error"] = standard_errors.ravel()
    summary["ci_lower"] = lower.ravel()
    summary["ci_upper"] = upper.ravel()
    summary["confidence_level"] = level
    summary["n_bootstrap"] = n_bootstrap
    summary["failure_policy"] = failure_policy
    summary["n_success"] = n_success.ravel()
    summary["n_failed"] = n_failed.ravel()
    summary["n_partial"] = n_partial.ravel()
    summary["mean_retained_weight_share"] = mean_retained.ravel()
    summary["min_retained_weight_share"] = min_retained.ravel()
    summary["success_rate"] = summary["n_success"] / n_bootstrap
    summary["conditional_on_success"] = summary["n_failed"] > 0
    summary["inference_status"] = np.where(
        n_success < 2,
        "insufficient_successful_draws",
        np.where(
            n_partial > 0,
            "available_cohorts",
            np.where(n_failed > 0, "successful_draws_only", "ok"),
        ),
    ).ravel()
    draws = pd.DataFrame(
        {
            "replicate": np.repeat(np.arange(1, n_bootstrap + 1), len(horizons) * len(dose_values)),
            "event_time": np.tile(np.repeat(horizons, len(dose_values)), n_bootstrap),
            "dose": np.tile(dose_values, n_bootstrap * len(horizons)),
            "theta": estimates.ravel(),
            "success": successful.ravel(),
            "n_cohorts_available": np.repeat(available_cohorts.ravel(), len(dose_values)),
            "n_cohorts_failed": np.repeat(failed_cohorts.ravel(), len(dose_values)),
            "retained_weight_share": retained_shares.ravel(),
        },
    )
    diagnostics = pd.DataFrame(
        [
            {"cohort": cohort, "event_time": event_time, "reason": reason, "n_failures": count}
            for (cohort, event_time, reason), count in failures.items()
        ],
        columns=["cohort", "event_time", "reason", "n_failures"],
    )
    diagnostics["cohort"] = pd.to_datetime(diagnostics["cohort"])
    diagnostics = diagnostics.astype({"event_time": "int64", "reason": "object", "n_failures": "int64"}, copy=False)
    diagnostics = diagnostics.sort_values(["event_time", "cohort", "reason"], na_position="first").reset_index(
        drop=True,
    )
    requested = set(horizons)
    omitted = sum(
        event_time in requested and cell["status"] not in {"fitted", "reference", "no_observed_treated"}
        for (_, event_time), cell in model.cell_diagnostics_.items()
    )
    restricted = int(summary.loc[summary["restricted_target"], "event_time"].nunique())
    extrapolated = int(summary["extrapolated"].sum())
    if n_failed.any() or failed_cohorts.any() or omitted or restricted or extrapolated:
        logging.warning(
            "Bootstrap summary (%s): %d of %d attempted dose-horizon draws failed; "
            "%d dose-horizon pairs have fewer than two successful draws. "
            "%d positive-mass cell fits failed; %d successful dose-horizon draws use available cohorts only. "
            "Intervals do not establish nominal or conditional coverage. "
            "Original aggregation has %d unavailable comparisons, "
            "%d restricted horizons, and %d extrapolated dose-horizon estimates. Inspect summary and diagnostics.",
            failure_policy,
            int(n_failed.sum()),
            n_bootstrap * len(horizons) * len(dose_values),
            int((n_success < 2).sum()),
            int(failed_cohorts.sum()),
            int(n_partial.sum()),
            omitted,
            restricted,
            extrapolated,
        )
    return summary, draws, diagnostics
