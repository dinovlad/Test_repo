# Continuous-dose ATT

A Python 3.11 package with its estimator in the self-contained module
`src/labor_cost_optimization/att.py`, for fixed-dose event studies. `ATT` fits a
separate PyFixest polynomial model for each cohort and horizon.
`estimate_pi()` estimates dose-specific cohort weights; `evaluate_theta()`
aggregates cohort effects at each dose and horizon. `estimate_h0()` supplies a
Silverman bandwidth rule, and `estimate_h_cv()` selects a bandwidth by predicting
cohorts from doses with leave-one-firm-out cross-validation.

Install the pinned environment:

```bash
uv sync --python 3.11 --group dev
```

Supply outcome and termination DataFrames and explicitly name their columns:

```python
from labor_cost_optimization.att import ATT, estimate_h_cv, evaluate_theta

bandwidth, cv_scores = estimate_h_cv(
    events,
    firm_col=firm_col,
    period_col=period_col,
    dose_col=dose_col,
)

model = ATT(
    firm_col=firm_col,
    period_col=period_col,
    outcome_col=outcome_col,
    dose_col=dose_col,
).fit(outcomes, events, isolation_window=(-7, 14))

components = model.predict([4, 6, 8, 10, 12])
theta = evaluate_theta(model, [4, 6, 8, 10, 12], bandwidth=bandwidth)
```

The four column-name variables are caller-supplied strings. Both tables must
use the same firm and period labels. Periods are timezone-naive `datetime64[ns]`
values, normalized at a fixed daily frequency. Missing firm-period rows are
allowed; outcomes on present rows must be finite. The event table must contain
the complete termination history used for isolation and control selection.

The bandwidth helpers accept the raw event table and select the first event per
firm after normalizing periods. They need no outcome table or fitted model.
Pass `eligible_firms` to restrict their population after first-event selection;
otherwise they use all first events, without inferring isolation or outcome
availability. A bandwidth estimated on this population can be reused across
horizons, but its CV score does not average over different horizon populations.

For a quick dose-scale bandwidth, use:

```python
from labor_cost_optimization.att import estimate_h0

h0 = estimate_h0(
    events,
    firm_col=firm_col,
    period_col=period_col,
    dose_col=dose_col,
)
```

`estimate_h_cv()` evaluates nine bandwidths from `h0 / 4` to `4 * h0` by default,
or a supplied positive, distinct `bandwidths` grid. It returns the selected
bandwidth and a table with `bandwidth` and `cv_brier` columns. Its mean multiclass
Brier loss gives every firm equal weight, holding out only that firm's own
contribution, including when doses repeat. Exact ties select the smallest
bandwidth. Singleton cohorts are retained; their held-out own-cohort probability
is zero. CV requires at least two cohorts and varying doses. It emits at most one
summary warning for singleton cohorts, a boundary selection, or tied minima.
`batch_size=256` bounds the query-dose batch; grouping repeated doses reduces
work, while all-distinct doses can still require quadratic computation.
These are custom bandwidth choices, not methods prescribed by the article.
See [att.md](att.md) for the formulas and population requirements.

The implementation keeps each firm's first termination as its only target,
retains later terminations for screening, and reports exclusions in one warning
per `fit()` call. Inspect `model.drop_summary_` for additional event rows,
excluded first events, unavailable treated rows, and rejected comparisons;
detailed reasons remain in the event, availability, and cell diagnostics. Treated
row counts refer to firm–horizon pairs, so one firm can count at several horizons.
A fit with no exclusions emits no selection warning. `evaluate_theta()` also
emits at most one summary warning per call, covering omitted comparisons,
excluded treated firm–horizon rows, restricted horizons, and extrapolated
component predictions across the requested grid. Its returned flags remain
available, and evaluation does not change `model.drop_summary_`.
Horizon -1 is an exact zero reference; earlier horizons are placebo contrasts.
Estimates have no standard errors or confidence intervals. The identifying
assumptions, selected populations, diagnostics, and API are described in
[att.md](att.md).

Fitting reuses indexed baseline observations and batches endpoint and termination
history lookups across cohorts at each horizon. Each comparison still has its
own polynomial transformation and rank checks. Each regression uses a
`poly(..., raw=True)` formula with only its own comparison rows, avoiding a
global matrix of interactions with every cell. Preparation still collects the
narrow comparison frames before fitting. Fitted models are available in
`model.models_`; prediction retains each cell's fitted dose scaling. The default
cubic polynomial has the same fitted curve as a cubic B-spline with no internal
knots on the same sample, up to numerical precision. Internal-knot splines are
not supported; there is no `num_knots` parameter. Selection and aggregation
rules are unchanged by the polynomial parameterization.

Run validation from the repository root and share `validation.log` here:

```bash
uv sync --python 3.11 --group dev
uv run --no-sync python scripts/validate.py --log validation.log
```

If your environment is already activated and the project is installed, run:

```bash
python scripts/validate.py --log validation.log
```

The script records Python and dependency versions, the imported estimator's
source, pytest results, and formatting, lint, type, and spelling checks.
Quality checks target the estimator module, its two test modules, the validation
script, and project documentation, so unrelated code in a larger project is not
scanned.
When copying these files into an existing project, include both
[tests/test_att.py](tests/test_att.py) and
[tests/test_validate.py](tests/test_validate.py), along with the estimator,
runner, and documentation. Missing required files make validation fail, while
quality checks continue on the expected files that are present.
It reads Python and dependency constraints from your `pyproject.toml`, including
exact pins, ranges, unversioned dependencies, extras, and environment markers.
Version checks use `packaging` when available; missing parsing support or
unverifiable direct references are reported without aborting the other checks.
It continues after failures, saves tracebacks and command exit codes, and exits
nonzero if any requested check fails. It installs nothing and does not modify
source files. The default timeout is 600 seconds per check; override it with
`--timeout`. Use `--tests-only` to skip the code-quality checks.

The pytest tests use synthetic data and compare the real PyFixest cell models against
independent least-squares calculations, including no-internal-knot B-spline
reference curves. They cover saved polynomial transformations, cohort weights,
bandwidth rules and exact leave-one-firm-out scores, aggregation, missing periods, isolation, duplicate events, zero effects, and
failed-fit handling. A run that does not satisfy the configured Python or runtime
package requirements is marked as a mismatch even if its numerical tests pass.
The [validation workflow](.github/workflows/validate.yml) runs pytest with Python
3.11 and the pinned runtime dependencies on pushes to `cgbs-att` and pull requests.

To run individual checks directly:

```bash
uv run --no-sync python -m pytest tests -v
uv run --no-sync pre-commit run --all-files
```

With an activated environment, use `python -m pytest tests -v`. To check only
the validation runner, use `python -m pytest tests/test_validate.py -v`.
Ensure the development dependencies, including `pytest` and `codespell`, are
installed in the same environment. The validation script never installs them.

The pre-commit hooks apply formatting and lint fixes; the validation script's
quality checks are read-only.

To apply import sorting and formatting using your own project's Ruff settings,
run these commands after restoring all required files:

```bash
python -m ruff check --select I --fix src/labor_cost_optimization/att.py tests/test_att.py tests/test_validate.py scripts/validate.py
python -m ruff format src/labor_cost_optimization/att.py tests/test_att.py tests/test_validate.py scripts/validate.py
python scripts/validate.py --log validation.log
```
