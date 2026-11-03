**Implementation specification: unpooled continuous-dose ATT and dose-specific event-study aggregation.**

This document specifies a future self-contained Python module, `att.py`. It does not implement that module. The agreed design uses independent cohort-time regressions following the parametric `contdid` estimator, retains each fitted spline basis for prediction, and estimates conditional cohort weights with a separate Gaussian-kernel function. All estimates in the first version are point estimates.

The public API consists of one class and two functions:

```python
model = ATT(
    degree=3,
    num_knots=0,
    min_cohort_size=10,
    control_group="notyettreated",
)
model.fit(panel, event_times=[0, 1, 7, 14])

unpooled = model.predict(doses=[4, 6, 8, 10, 12])

weights = estimate_pi(
    model.units_,
    doses=[4, 6, 8, 10, 12],
    eligible_cohorts=eligible_cohorts,
    bandwidth=bandwidth,
)

theta = evaluate_theta(
    model,
    doses=[4, 6, 8, 10, 12],
    bandwidth=bandwidth,
    event_times=[0, 7, 14],
)
```

The `eligible_cohorts` argument in the standalone example must contain retained cohorts eligible at one specified event day. `evaluate_theta()` constructs that set automatically for each requested event day. The bandwidth is supplied by the caller in hours.

**Statistical target and treatment coding.** For store $i$, let $G_i$ be the focal worker's termination day and $D_i>0$ the worker's previous daily hours. Realized exposure is the permanent hours reduction

$$
W_{it}=D_i\thinspace{}1\lbrace t\ge G_i\rbrace.
$$

The dose is a fixed store-level label, including on pretreatment rows. Zero exposure means that the focal termination has not occurred. The outcome is daily store sales; the counterfactual is continued employment at the original hours. Other staffing responses are included in the total effect of the focal termination.

The unpooled target is

$$
ATT(g,t,d\mid g,d)
=E[Y_t(g,d)-Y_t(0)\mid G=g,D=d].
$$

For a nonnegative event day $e=t-g$, define

$$
\mathcal G_e=\lbrace g:\ g\text{ is an observed treated cohort},\ g+e\le T\rbrace.
$$

With every treated cohort having an observed baseline, the requested aggregation is

$$
\theta(d,e)=\sum_{g\in\mathcal G_e}\pi_g(e,d)
ATT(g,g+e,d\mid g,d),
\qquad
\pi_g(e,d)=P(G=g\mid D=d,G\in\mathcal G_e).
$$

Dose and event day remain fixed; only calendar-day cohorts are averaged. Ordinary parallel trends and no anticipation identify the cohort-specific level effects. Parametric estimation additionally requires an adequate specification of each cohort-time dose curve. Strong parallel trends is not assumed, and derivatives are not interpreted as causal marginal responses. The article defines the dose-specific event-study target on printed page 41; Remark C.1 on page 42 discusses aggregating estimated components without prescribing a conditional-weight estimator. [CGBS, Appendix C](https://psantanna.com/files/CGBS_v4.pdf#page=41)

**Relationship to the R implementation.** The reference is `contdid::cont_did_acrt`, restricted to its parametric level-effect calculation: regress raw outcome changes on an intercept and dose B-splines among the focal treated cohort, then subtract the comparison group's mean change. The cohort-time comparisons use baseline $g-1$. The Python module reproduces this calculation for each retained comparison. [Estimator and subset source](https://github.com/bcallaway11/contdid/blob/main/R/cont_did.R)

Three differences are intentional: retaining the fitted basis at prediction, warning and dropping small or unidentifiable target cohorts, and adding dose-conditional kernel aggregation. Consequently, “package-compatible” refers to the estimation calculation with a consistent fitted basis, not numerical reproduction of every behavior of the current R code. In particular, the inspected R prediction path rebuilds its basis on the prediction grid. References were inspected on 2026-11-03; implementation validation should record the R version or source revision actually used.

**Input contract.** `ATT.fit()` accepts a pandas DataFrame with these standard column names:

| Column | Meaning |
| --- | --- |
| `id` | Unique store identifier; one focal termination per store |
| `day` | Integer calendar-day index from 1 through $T$ |
| `sales` | Finite numeric daily outcome |
| `cohort` | Termination day; 0 for stores untreated throughout the observed window |
| `dose` | Positive fixed hours reduction for treated stores; 0 for never-treated stores |

The panel must contain one row per store per day, with no missing outcomes or duplicate `(id, day)` pairs. `cohort` and `dose` must be constant within each store. A positive cohort must lie in $\lbrace 2,\ldots,T\rbrace$, ensuring an observed pretreatment day. Calendar dates can be converted to integer offsets before calling the module; missing calendar days must not be compressed into adjacent indices. The input panel is validated rather than silently balanced or imputed.

This specification uses the theoretical convention $G=\infty$ for never-treated units in formulas and `cohort=0` in data. Code must explicitly handle that sentinel. Requested `event_times` are distinct nonnegative integers; requested doses are distinct, finite, positive numbers. Inputs are copied as needed; fitting must not alter the caller's DataFrame.

Runtime dependencies are NumPy, pandas, PyFixest, SciPy, and the Python standard library. Exact runtime pins and development-tool requirements are declared in [pyproject.toml](pyproject.toml). The project requires Python 3.11; [.python-version](.python-version) selects that interpreter for tools such as uv. The first version covers balanced daily panels, equal unit weights, no anticipation, and post-treatment level estimates. Covariates, inference, plotting, treatment slopes, and changing-dose paths are outside this initial scope.

Use the supplied [.pre-commit-config.yaml](.pre-commit-config.yaml) for spelling, formatting, lint, and type checks on Python, Markdown, and notebooks. The hooks use system executables, so activate the development environment before installing or running them:

```bash
uv sync --python 3.11 --group dev
source .venv/bin/activate
pre-commit install
pre-commit run --all-files
```

**`ATT` owns independent cohort-time fits.** The class is panel-level for convenience, but never pools regression coefficients across cohorts. Its signatures are:

```python
ATT(
    *,
    degree=3,
    num_knots=0,
    min_cohort_size=10,
    control_group="notyettreated",
)

ATT.fit(panel, *, event_times) -> ATT
ATT.predict(doses, *, event_times=None) -> pandas.DataFrame
```

Construction requires a positive integer `degree`, a nonnegative integer `num_knots`, a positive integer `min_cohort_size`, and one of the two specified control-group options. `fit()` returns `self` and replaces any previously fitted state; a failed refit must not leave an earlier model available for prediction. `predict()` before successful fitting raises an error. Omitting `event_times` in prediction requests all fitted horizons; explicitly requesting an unfitted horizon raises an error rather than fitting additional models implicitly. Event-time and dose requests must be nonempty.

For a cohort $g$ and an in-window horizon $e$, set $t=g+e$ and form the comparison sample from the focal cohort and one of these control sets:

$$
\mathcal C_{gt}=
\begin{cases}
\lbrace j:G_j=\infty\text{ or }G_j>t\rbrace,&\texttt{notyettreated},\\
\lbrace j:G_j=\infty\rbrace,&\texttt{nevertreated}.
\end{cases}
$$

Compute $\Delta Y_j^{g,t}=Y_{jt}-Y_{j,g-1}$ on those exact calendar dates. Future-treated controls retain their original cohort and dose in stored metadata. Their local treatment indicator is zero in this comparison; their positive eventual dose must not put them in the focal treated regression.

Let $b_g(d)$ denote the saved B-spline columns, excluding the first column of the full B-spline basis. The fitted design includes a separate intercept:

$$
x_g(d)=(1,b_g(d)')',\qquad
\widehat\gamma_{gt}
=\arg\min_\gamma\sum_{i:G_i=g}
\left(\Delta Y_i^{g,t}-x_g(D_i)'\gamma\right)^2.
$$

The control correction is calculated separately and stored:

$$
\widehat m_{0,gt}
=\frac{1}{|\mathcal C_{gt}|}\sum_{j\in\mathcal C_{gt}}\Delta Y_j^{g,t}.
$$

Prediction is

$$
\widehat{ATT}(g,t,d\mid g,d)
=x_g(d)'\widehat\gamma_{gt}-\widehat m_{0,gt}.
$$

The intercept in $\widehat\gamma_{gt}$ belongs to the raw treated-outcome regression. The stored control mean is subtracted exactly once. Neither the controls nor the kernel weights enter the treated OLS fit. Use a numerically stable least-squares solver, not an explicit inverse of $X'X$.

**Fitted basis and knots.** Let $q$ be the degree and $K$ the number of internal knots. The regression has $p=q+K+1$ columns, including the intercept. Defaults $q=3,K=0$ give a cubic polynomial expressed in a B-spline basis.

For $K>0$, choose shared internal knots once using the positive doses from one row per initially valid treated store, before the small-cohort screen. Use quantiles at $k/(K+1)$, $k=1,\ldots,K$, with linear interpolation matching R's default quantile convention. This follows the package's global knot-selection structure. Keep the selected knots fixed after screening. [Setup source](https://github.com/bcallaway11/contdid/blob/main/R/setup_pte_cont.R), [quantile helper](https://github.com/bcallaway11/contdid/blob/main/R/helper_functions.R)

For each retained cohort, use its observed treated-dose minimum and maximum as boundary knots. Build and store the resulting basis definition: degree, internal knots, repeated boundary knots, and retained-column convention. A SciPy `BSpline` representation can evaluate that saved basis. Construct the full basis, omit its first column, and add an explicit intercept, consistent with the R parameterization. [B-spline conventions](https://wwenjie.org/splines2/reference/bspline)

Reject duplicate global internal knots with a configuration error; do not silently change $K$. If a shared internal knot is not strictly inside a cohort's boundary knots, warn and drop that target cohort with an `incompatible_knots` reason. Degenerate dose ranges and rank-deficient designs are handled by the cohort screen below.

`predict()` must evaluate the fitted basis without recalculating knots or boundaries from the requested doses. In particular, a prediction at dose 8 must be identical whether requested alone or alongside other doses. With a balanced panel, the same cohort basis can be reused across its horizons; coefficients and control means remain specific to each comparison.

To stay close to the parametric routine, allow polynomial continuation outside a cohort's observed dose range, issue an extrapolation warning, and flag those rows. Do not clip doses, discard cohorts at particular doses, or renormalize weights because of extrapolation. Such values rely on the fitted functional form beyond observed support.

**Small-cohort screening and dropping.** The proposed configurable default is `min_cohort_size=10` distinct stores. This is an implementation choice, not a threshold supplied by the article or a guarantee of precision. Let $n_{\mathrm{user}}$ denote the configured `min_cohort_size`. The effective minimum is

$$
n_{\min}=\max(n_{\mathrm{user}},p+1),
$$

which also requires positive residual degrees of freedom. Count stores once, not daily rows. A cohort is retained only if its sample size meets that minimum, it has at least $p$ distinct dose values, and its actual regression matrix has full column rank. Use the least-squares solver's documented numerical rank tolerance consistently. The actual rank check is necessary even when the distinct-dose check passes.

For every dropped target cohort, emit one warning identifying its date, number of stores, number of distinct doses, required design dimension, and reason. Record the same information in `cohort_diagnostics_`. Never silently lower the degree, remove knots, or pool adjacent calendar cohorts to rescue a fit.

Screening depends on the cohort's dose design, so drop that cohort from target estimation at every event day. **Keep its stores in the underlying panel and preserve their actual treatment dates.** They remain valid controls for earlier comparisons while they are untreated. Dropping a target cohort must not turn its stores into never-treated stores or remove valid control observations.

Ordinary horizon exclusions, $g+e>T$, are not small-cohort failures. If a retained in-window cohort-time comparison has zero controls, warn and record `no_controls` in `cell_diagnostics_`; no model is fitted for that cell. `evaluate_theta()` must raise a clear error for an affected requested horizon rather than silently remove that comparison and renormalize. A singleton control group is computable for point estimation; its count is reported.

If all target cohorts fail screening, or no requested comparison can be fitted, `fit()` raises an error after recording the reasons. These failure rules are deliberately explicit; predictions from failed cells are not filled with zero.

**Stored state and unpooled output.** Keep only the information needed for prediction, weights, and diagnostics:

- `units_`: one row per original store, with `id`, `cohort`, and `dose`; includes controls and dropped target cohorts.
- `event_times_`, `last_day_`, `knots_`, and `retained_cohorts_`.
- `bases_`: saved basis definitions keyed by cohort.
- `models_`: separate coefficient vectors, control means, and sample counts keyed by `(cohort, event_time)`.
- `cohort_diagnostics_`: retained/dropped status and design information for every treated cohort.
- `cell_diagnostics_`: in-window eligibility, control counts, and fit status for every requested cohort/horizon pair.

`ATT.predict()` returns one row for each available `(cohort, event_time, dose)` combination, with columns `cohort`, `day`, `event_time`, `dose`, `att`, `n_treated`, `n_controls`, and `extrapolated`. Sort deterministically by cohort, event time, and dose. If unavailable in-window cells exist, warn that the returned table is incomplete and direct the caller to `cell_diagnostics_`. Out-of-window combinations are simply absent.

**Separate kernel-weight function.** This function is independent of fitted outcomes and of `ATT` internals:

```python
estimate_pi(
    units,
    doses,
    *,
    eligible_cohorts,
    bandwidth,
) -> pandas.DataFrame
```

`units` must have exactly one row per store, with unique `id` and finite, valid cohort/dose labels. Validate uniqueness instead of deduplicating ambiguous input. `eligible_cohorts` is an explicit nonempty set of positive cohort labels, all present in `units`. Filter to those cohorts before any kernel normalization. Never-treated stores and excluded target cohorts do not enter these weights.

For $g$ in the supplied set $\mathcal R_e$, use a common bandwidth $h>0$ and

$$
q_g(d;h)=\sum_{i:G_i=g}
\exp\left[-\frac12\left(\frac{D_i-d}{h}\right)^2\right],
\qquad
\widehat\pi_g(e,d)
=\frac{q_g(d;h)}{\sum_{k\in\mathcal R_e}q_k(d;h)}.
$$

The Gaussian normalization constant and common $1/h$ cancel. Summing over stores, rather than averaging within each cohort, preserves differences in cohort size. All weights are nonnegative and sum to one at each requested dose. Compute normalization with log-sum-exp for numerical stability; invalid bandwidths or nonfinite intermediate results raise errors rather than triggering a fallback to uniform weights.

Return a table with `cohort`, `dose`, and `weight`, sorted by cohort and dose. The function does not need an event-time argument: the caller supplies the appropriate eligible set. Reusing this function with different sets supplies the horizon dependence. It can also be used directly to inspect how the weights change with dose or bandwidth.

This is a custom estimator of conditional cohort probabilities. The article's definition does not determine a kernel or bandwidth, and the package's existing dose aggregation does not supply these conditional weights. A fixed positive bandwidth smooths cohort composition over nearby doses; consistency for the exact conditional probabilities requires suitable smoothness, bandwidth, and sample-size conditions. No automatic bandwidth selection is included in the first version.

**Thin extension for evaluating the event study.** A standalone function is sufficient; no second estimator class or inheritance hierarchy is needed:

```python
evaluate_theta(
    model,
    doses,
    *,
    bandwidth,
    event_times=None,
) -> pandas.DataFrame
```

For each requested fitted horizon $e$:

1. Construct $\mathcal G_e$ from all original treated cohorts with an observed endpoint.
2. Let $\mathcal R$ be the set stored in `retained_cohorts_`, and set $\mathcal R_e=\mathcal G_e\cap\mathcal R$. If empty, raise an error.
3. Require every cohort in $\mathcal R_e$ to have a fitted cell at this horizon. A missing control group is an error here, not an additional implicit cohort-selection rule.
4. Obtain unpooled predictions from `ATT.predict()` and weights from `estimate_pi(model.units_, ..., eligible_cohorts=R_e)`.
5. Join them by the exact cohort and dose labels. Validate a complete one-to-one match and weights summing to one; do not let an inner join silently discard unmatched components.
6. Calculate

$$
\widehat\theta_{\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
\widehat\pi_g(e,d)\thinspace{}
\widehat{ATT}(g,g+e,d\mid g,d).
$$

The function only aggregates existing estimates. It neither refits the dose curves nor changes the sample screening. Changing bandwidth changes the weights and aggregation, not the underlying ATT models.

Return columns `event_time`, `dose`, `theta`, `n_cohorts_eligible`, `n_cohorts_used`, `n_treated_eligible`, `n_treated_used`, `restricted_target`, and `extrapolated`. Counts refer to the original and retained horizon-eligible target populations, respectively. `restricted_target` is true when any eligible target cohort was screened out. `extrapolated` is true when any contributing cohort prediction is outside that cohort's observed dose range. Return rows sorted by event time and dose. The component estimates and weights remain independently inspectable through the existing APIs.

**Dropping cohorts changes the target.** After screening, the reported aggregation pertains to retained cohorts, with weights estimating

$$
P(G=g\mid D=d,G\in\mathcal R_e).
$$

For a given retained set, its population counterpart is

$$
\theta_{\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
P(G=g\mid D=d,G\in\mathcal R_e)
ATT(g,g+e,d\mid g,d).
$$

This equals the originally requested target when every eligible cohort is retained, or under additional special conditions; equality is not assumed. The reported table uses the short column name `theta`, but its diagnostics and documentation must preserve this distinction. Emit an aggregation warning when the target is restricted, listing affected horizons and directing the caller to the cohort diagnostics. Do not retain excluded cohorts in the kernel denominator, replace their unknown effects with zero, or imply that renormalization recovers their effects.

The screen is sample-dependent, so this is a description of the retained-cohort target, not an inferential result about selection. Cohort composition also changes naturally with the horizon. The first version follows that horizon-specific definition and does not impose a common balanced cohort set across event days.

**Implementation boundaries and validation plan.** Put the class, both public functions, and a few private validation/basis helpers in one `att.py` file. Use ordinary warning categories with informative messages and small dictionaries or records for fitted cells. External R code is not a runtime dependency. No generic estimator framework, serialization layer, or extra classes are needed.

When the Python implementation is written, verify behavior with targeted numerical examples:

- A single two-period comparison with a known polynomial dose effect and a nonzero common control trend recovers the expected ATT curve, subtracting the control mean exactly once.
- Predictions at a fixed dose do not depend on which other doses appear in the prediction request; permuting input rows leaves results unchanged.
- Where the R training and prediction bases coincide, full-rank unpooled point estimates agree numerically. For other grids, compare with an R reference that explicitly reuses training boundary knots; document that prediction correction.
- A future-treated store contributes to earlier valid controls but never after its own treatment starts. This also holds when its target cohort was dropped.
- Small and rank-deficient cohorts produce warnings and diagnostic entries, disappear from target predictions and kernel denominators, and remain correctly classified in the panel.
- Kernel weights sum to one, reflect unequal cohort sizes, change with dose when dose composition differs across cohorts, and agree with the explicit formula on a small example.
- The aggregation equals a manually computed weighted sum, reports its retained population, and refuses to silently aggregate a horizon with an unavailable required cell.

Only the specification is delivered at this stage. Subsequent inference would need to account for shared controls, estimated cohort weights, smoothing choices, and the cohort screen; ordinary regression standard errors alone would not provide that inference.
