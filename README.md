# Continuous-dose ATT

A self-contained Python 3.11 module for fixed-dose event studies. `ATT` fits one
PyFixest model with separate coefficients for each cohort and horizon.
`estimate_pi()` estimates dose-specific cohort weights; `evaluate_theta()`
aggregates cohort effects at each dose and horizon.

Install the pinned environment:

```bash
uv sync --python 3.11 --group dev
source .venv/bin/activate
```

Supply outcome and termination DataFrames and explicitly name their columns:

```python
from att import ATT, evaluate_theta

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
retains later terminations for screening, and reports small-cell exclusions.
Horizon -1 is an exact zero reference; earlier horizons are placebo contrasts.
Estimates have no standard errors or confidence intervals. The identifying
assumptions, selected populations, diagnostics, and API are described in
[att.md](att.md).

Run the numerical checks and repository hooks:

```bash
python -m unittest discover -s tests
pre-commit run --all-files
```
