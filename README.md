# Continuous-Dose ATT

A Python 3.11 package with its estimator in the self-contained module
`src/labor_cost_optimization/att.py`, for fixed-dose event studies. `ATT` fits a
separate PyFixest polynomial model for each cohort and horizon.
`estimate_pi()` estimates dose-specific cohort weights; `evaluate_theta()`
aggregates cohort effects at each dose and horizon. `estimate_h0()` supplies a
Silverman bandwidth rule, and `estimate_h_cv()` selects a bandwidth by predicting
cohorts from doses with leave-one-firm-out cross-validation. `bootstrap_theta()`
adds pointwise intervals by resampling complete firms and re-estimating both
the cohort effects and their kernel weights.

Install the pinned environment:

```bash
uv sync --python 3.11 --group dev
```

Supply outcome and termination DataFrames and explicitly name their columns:

```python
import labor_cost_optimization.att as att

bandwidth, cv_scores = att.estimate_h_cv(
    events,
    firm_col=firm_col,
    period_col=period_col,
    dose_col=dose_col,
)

model = att.ATT(
    firm_col=firm_col,
    period_col=period_col,
    outcome_col=outcome_col,
    dose_col=dose_col,
).fit(outcomes, events, isolation_window=(-7, 14))

components = model.predict([4, 6, 8, 10, 12])
theta = att.evaluate_theta(model, [4, 6, 8, 10, 12], bandwidth=bandwidth)
```

The four column-name variables are caller-supplied strings. Both tables must
use the same firm and period labels. Periods are timezone-naive `datetime64[ns]`
values, normalized at a fixed daily frequency. Missing firm-period rows are
allowed; outcomes on present rows must be finite. The event table must contain
the complete termination history used for target isolation and first-treatment status.

Pydantic v2 `Field` constraints, cached `TypeAdapter` instances, and
`validate_call` validate configuration and input types. NumPy scalar inputs are
supported; Boolean values and numeric strings are rejected for numeric
parameters and dose grids. Grid validation consumes generators once. Table
contents and sample requirements use vectorized checks. Pydantic validation
errors are subclasses of `ValueError` and expose structured error details.
The installed package includes a `py.typed` marker so type checkers can use its
inline annotations.

The bandwidth helpers accept the raw event table and select the first event per
firm after normalizing periods. They need no outcome table or fitted model.
Pass `eligible_firms` to restrict their population after first-event selection;
otherwise they use all first events, without inferring isolation or outcome
availability. A bandwidth estimated on this population can be reused across
horizons, but its CV score does not average over different horizon populations.

For a quick dose-scale bandwidth, use:

```python
import labor_cost_optimization.att as att

h0 = att.estimate_h0(
    events,
    firm_col=firm_col,
    period_col=period_col,
    dose_col=dose_col,
)
```

`estimate_h_cv()` starts with nine bandwidths from `h0 / 10` to `10 * h0`,
or a supplied positive, distinct `bandwidths` grid. If an initial minimum lies
at a boundary, it expands that boundary by factors of two until the score rises
beyond numerical tolerance. Both directions are explored independently when
both initial boundaries minimize, including a one-candidate grid. Explicit
bandwidths are an initial grid, not a restriction on the selected value.

The returned table has `bandwidth` and `cv_brier` columns, sorted by bandwidth,
and includes all evaluated points, including the first upward turn. Previously
scored candidates are cached. The selected bandwidth minimizes loss over all
visited points; exact ties select the smallest. This local boundary search does
not guarantee the global minimum over all positive bandwidths.

The search also stops a direction after three consecutive negligible score
changes, after `max_expansions=50` additional evaluations, or before a generated
bandwidth or squared scaled dose distance would leave the numerical range.
Those stops produce a warning rather than claiming an upward turn was found.
Invalid initial grids still raise an error. See [docs/att.md](docs/att.md) for the exact
floating-point tolerance and stop rules.

The mean multiclass Brier loss gives every firm equal weight, holding out only
that firm's own contribution, including when doses repeat. Singleton cohorts
remain in the score; their held-out own-cohort probability is zero. CV requires
at least two cohorts and varying doses. At most one warning combines singleton
cohorts, a selected final-grid boundary, tied minima, and expansion stop reasons.
A clean unique interior minimum emits none. `batch_size=256` bounds the
query-dose batch; grouping repeated doses reduces work, while all-distinct doses
can still require quadratic computation. Bandwidth search does not refit ATT
models or change the selected population. These are custom bandwidth choices,
not methods prescribed by the article.

The implementation keeps each firm's first termination as its only target,
retains later terminations for target isolation, and reports exclusions in one warning
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
`predict()` and `evaluate_theta()` return point estimates. Use
`bootstrap_theta()` for standard errors and pointwise intervals. The identifying
assumptions, selected populations, diagnostics, and API are described in
[docs/att.md](docs/att.md).

For cohort `g` at a post-treatment endpoint `t`, controls are firms whose first
termination is strictly after `t`, including firms with no recorded termination.
This follows Callaway et. al. (2025), Theorem C.1. A future termination inside the
target's isolation window does not exclude a control before that termination;
previously treated firms never reenter the control pool. For negative and
reference horizons, controls first terminate strictly after `g`, keeping both
comparison periods untreated and excluding the focal cohort. Every control needs
observed outcomes at the fixed baseline and current endpoint. First-treatment
status uses all first-event records, including target-ineligible and out-of-panel
events; a control need not pass its own target-isolation screen. Correct untreated
status also requires relevant pre-panel history or a credible untreated-at-start
assumption; complete within-panel records alone cannot rule out earlier treatment.

Fitting reuses indexed baseline observations and batches endpoint and first-event
lookups across cohorts at each horizon. Each comparison still has its
own polynomial transformation and rank checks. Each regression uses a
`poly(..., raw=True)` formula with only its own comparison rows, avoiding a
global matrix of interactions with every cell. Preparation still collects the
narrow comparison frames before fitting. Fitted models are available in
`model.models_`; prediction retains each cell's fitted dose scaling. The default
cubic polynomial has the same fitted curve as a cubic B-spline with no internal
knots on the same sample, up to numerical precision. Internal-knot splines are
not supported; there is no `num_knots` parameter. Selection and aggregation
rules are unchanged by the polynomial parameterization.

## Firm Bootstrap for Pointwise Intervals

```python
summary, draws, diagnostics = att.bootstrap_theta(
    model,
    doses=[4, 6, 8, 10, 12],
    outcomes=outcomes,
    bandwidth=bandwidth,
    event_times=[-2, -1, 0, 7, 14],
    n_bootstrap=1999,
    confidence_level=0.95,
    random_state=42,
    failure_policy="available_cohorts",
)
```

Pass the original outcome panel using the model's configured column names.
`fit()` retains the complete original firm index and a SHA-256 fingerprint of
the normalized panel, including firms and rows unused by the fitted comparisons.
Bootstrap verifies this fingerprint; row order, time components within a period,
and additional unmapped columns do not matter. It does not require the event
table again or retain another outcome-panel copy on the model. Models without
these fitted metadata must be fitted before calling the bootstrap.

Each draw samples as many firms as the original panel contains, with replacement.
Integer sampling counts apply to every use of a firm across periods, cohorts,
horizons, control roles, and kernel weights. Counts are processed in batches of
at most 32; comparison data are reconstructed from the supplied panel as needed.
There is no expanded copy of the panel for each draw. Weighted polynomial solves
and control means give the same estimates as fitting duplicated comparison rows;
they do not call PyFixest or `fit()` again. Each cell's coefficients are evaluated
at every requested dose.

The original retained comparisons, treated/control membership, dose scaling,
polynomial degree, and selected bandwidth remain fixed. Cohort shares vary with
the sampled firms. No minimum-size screening is repeated, and excluded
original comparisons do not enter the bootstrap aggregation.

`failure_policy="strict"` is the default. A positive-weight cohort with
insufficient polynomial rank or no sampled controls makes that draw unavailable
at the affected horizon, for every requested dose.

Use `failure_policy="available_cohorts"` to keep each estimable cohort instead.
For each draw and horizon, the kernel weights are estimated over the available
cohorts and normalized to sum to one. The aggregate is the sum of those weights
times their ATT estimates; no additional normalization is needed after defining
the weights this way. A single available cohort is sufficient. If none remain,
the corresponding estimates are missing. Failed positive-weight cohorts remain
visible in the diagnostics.

`summary` preserves every `evaluate_theta()` column and adds `standard_error`,
`ci_lower`, `ci_upper`, `confidence_level`, `n_bootstrap`, `n_success`, `n_failed`,
`success_rate`, `conditional_on_success`, `inference_status`, and `failure_policy`.
It also reports `n_partial`, counting successful draws that excluded at least one
positive-weight cohort, and `mean_retained_weight_share` and
`min_retained_weight_share` over successful draws. Standard errors use sample
standard deviations, and intervals use the successful draws' empirical
percentiles separately at each dose and horizon. At least two successful draws
are required; otherwise the standard error and bounds are missing. There is no
additional success-rate cutoff.

A cohort with no sampled treated firms contributes zero under either policy.
A draw with no sampled treated firms anywhere in the horizon is unavailable.
At the reference horizon -1, positive total treated mass gives an exact zero
without regression or control checks. With at least two successful reference
draws, its standard error and both interval bounds are zero. Polynomial cell
estimability and Gaussian support normally make availability common across
doses at a horizon, but success counts and intervals are computed separately
for each `(dose, event_time)` pair.

`draws` contains all attempted replications, including failed estimates as
missing values, with columns `replicate` (starting at 1), `event_time`, `dose`,
`theta`, `success`, `n_cohorts_available`, `n_cohorts_failed`, and
`retained_weight_share`. Cohort counts include only cohorts with sampled treated
firms. The share compares their estimable cohorts' kernel mass with the mass of
all originally retained cohorts in that draw, at that dose and horizon. It is
zero when sampled treated firms exist but no cohort is estimable, and missing
when there are no sampled treated firms. A successful reference draw has share
one. These diagnostics remain present on failed strict-policy draws too.

`diagnostics` aggregates failures by `cohort`, `event_time`, and `reason`,
reporting `n_failures`. A global lack of treated firms (`no_treated_firms`) or,
under the available-cohort policy, a complete lack of estimable positive-mass
cohorts (`no_available_cohorts`) has a missing cohort label. Cell failures remain
recorded even when the aggregate is available. Multiple comparisons can fail in
one draw, so summed comparison failure counts can exceed the number of failed
draws at a horizon. Failures are not replaced. Unexpected numerical or
implementation errors raise exceptions. Each bootstrap call emits at most one
aggregated warning, including when cohorts are excluded but every aggregate
remains available.

`inference_status` is `insufficient_successful_draws` when fewer than two draws
succeed. Otherwise it is `available_cohorts` if any successful draw excludes
positive-weight cohorts, `successful_draws_only` if some draws fail, or `ok`.
`conditional_on_success` is true whenever any draw fails at the dose–horizon
pair; `n_partial` separately identifies exclusions within successful draws.
These flags describe restrictions on the resampling distribution, not guaranteed
conditional coverage. With `available_cohorts`, each draw can average a different
cohort population. Frequent exclusions can shift that distribution away from the
original point-estimate target, especially when small cohorts have different
effects. Inspect the failure counts, reasons, and retained kernel weight when
using these intervals. The procedure assumes independent firms, holds screening
and tuning choices fixed, and ignores KDE smoothing bias. The
intervals are pointwise, not a simultaneous band over doses or horizons. A short
run can help assess failure frequency before a longer run.

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
Quality checks target the estimator module, its three test modules, shared fixtures
in `tests/conftest.py`, the validation script, and project documentation, so
unrelated code in a larger project is not scanned.
When copying these files into an existing project, include
[tests/test_att.py](tests/test_att.py),
[tests/test_bootstrap.py](tests/test_bootstrap.py),
[tests/test_validate.py](tests/test_validate.py), and
[tests/conftest.py](tests/conftest.py), along with the estimator, its `py.typed`
marker, the runner, and documentation. Missing required files make validation
fail, while quality checks continue on the expected files that are present.
It reads Python and dependency constraints from your `pyproject.toml`, including
exact pins, ranges, unversioned dependencies, extras, and environment markers.
Version checks use `packaging` when available; missing parsing support or
unverifiable direct references are reported without aborting the other checks.
It continues after failures, saves tracebacks and command exit codes, and exits
nonzero if any requested check fails. It installs nothing and does not modify
source files. The default timeout is 600 seconds per check; override it with
`--timeout`. Use `--tests-only` to skip the code-quality checks.

Reusable pytest fixtures live in `tests/conftest.py`. Unit-test assertions may
omit messages; the assertion-message rule applies to production code.

The pytest tests use synthetic data and compare the real PyFixest cell models against
independent least-squares calculations, including no-internal-knot B-spline
reference curves. They cover saved polynomial transformations, cohort weights,
bandwidth rules and exact leave-one-firm-out scores, aggregation, missing periods, isolation, duplicate events, zero effects, and
failed-fit handling. Bootstrap tests compare sampling counts with explicitly
duplicated comparison rows and cover successful-draw intervals, failures, panel
fingerprints, and shared firm weights. A run that does not satisfy the configured Python or runtime
package requirements is marked as a mismatch even if its numerical tests pass.
The [validation workflow](.github/workflows/validate.yml) runs the tests and
formatting, lint, type, and spelling checks with Python 3.11 and pinned runtime
dependencies on pushes to `cgbs-att` and pull requests.

To run individual checks directly:

```bash
uv run --no-sync python -m pytest tests -v
uv run --no-sync pre-commit run --all-files
```

With an activated environment, use `python -m pytest tests -v`. To check only
the validation runner, use `python -m pytest tests/test_validate.py -v`.
Ensure the development dependencies, including `pytest`, `codespell`, and
`orjson`, are installed in the same environment. The validation script never
installs them.

The pre-commit hooks apply formatting and lint fixes; the validation script's
quality checks are read-only.

To apply import sorting and formatting using your own project's Ruff settings,
run these commands after restoring all required files:

```bash
python -m ruff check --select I --fix src/labor_cost_optimization/att.py \
    tests/conftest.py tests/test_att.py tests/test_bootstrap.py \
    tests/test_validate.py scripts/validate.py
python -m ruff format src/labor_cost_optimization/att.py \
    tests/conftest.py tests/test_att.py tests/test_bootstrap.py \
    tests/test_validate.py scripts/validate.py
python scripts/validate.py --log validation.log
```

## Scientific References

Use Harvard author–year citations in code docstrings and project documentation.
Cite the continuous-treatment article inline as Callaway et. al. (2025), and
link to its full entry in the [references section of docs/att.md](docs/att.md#references).
Reference entries list authors, publication year, title, publication venue or
working-paper status, a stable URL or DOI, and access information when applicable.
Use title case for document headings and retain the cited work's title in each
bibliographic entry.

## Agent Rules and Skills

[AGENTS.md](AGENTS.md) is the entry point for Codex instructions in this repository.
New Codex chats opened in this repository load it automatically; it directs the
agent to read the ten shared rule files in [.agents/rules/](.agents/rules/) and
apply the relevant requirements.

Four skills are available in [.agents/skills/](.agents/skills/): `env-setup`,
`python-module`, `style-compliance`, and `commit-readiness`. Each has a `SKILL.md`
with discovery metadata. Codex selects skills when the task matches their
description, or you can request one explicitly, for example `$python-module`.
The full skill workflow is loaded when needed.

The rules and skill bodies come from the supplied `agents.zip`; skill metadata
was added for discovery, and macOS archive metadata was omitted. Edit the files
under `.agents/` to maintain them, and update `AGENTS.md` when adding or removing
rules or skills.

These instructions apply to chats working in a checkout containing these files.
Commit and merge them into the branch used for new checkouts to make them
available there as well. They do not configure chats outside this repository.
