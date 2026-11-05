# Continuous-dose ATT

A Python 3.11 package with its estimator in the self-contained module
`src/labor_cost_optimization/att.py`, for fixed-dose event studies. `ATT` fits one
PyFixest model with separate coefficients for each cohort and horizon.
`estimate_pi()` estimates dose-specific cohort weights; `evaluate_theta()`
aggregates cohort effects at each dose and horizon.

Install the pinned environment:

```bash
uv sync --python 3.11 --group dev
```

Supply outcome and termination DataFrames and explicitly name their columns:

```python
from labor_cost_optimization.att import ATT, evaluate_theta

model = ATT(
    firm_col=firm_col,
    period_col=period_col,
    outcome_col=outcome_col,
    dose_col=dose_col,
).fit(outcomes, events, isolation_window=(-7, 14))

components = model.predict([4, 6, 8, 10, 12])
theta = evaluate_theta(model, [4, 6, 8, 10, 12], bandwidth=1.0)
```

The four column-name variables are caller-supplied strings. Both tables must
use the same firm and period labels. Periods are timezone-naive `datetime64[ns]`
values, normalized at a fixed daily frequency. Missing firm-period rows are
allowed; outcomes on present rows must be finite. The event table must contain
the complete termination history used for isolation and control selection.

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
own spline and rank checks. The single PyFixest regression retains a dense
design whose memory use grows with the stacked rows and number of comparisons;
vectorized selection does not remove that constraint.

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

The pytest tests use synthetic data and compare the real PyFixest model against
independent least-squares calculations. They cover saved splines, cohort weights,
aggregation, missing periods, isolation, duplicate events, zero effects, and
failed-fit handling. A run that does not satisfy the configured Python or runtime
package requirements is marked as a mismatch even if its numerical tests pass.

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
