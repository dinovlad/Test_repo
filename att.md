**Implementation specification: unpooled continuous-dose ATT and dose-specific event-study aggregation.**

This document specifies a future self-contained Python module, `att.py`. It does not implement that module. The design uses independent cohort-time regressions following the parametric `contdid` calculation, retains each fitted spline basis for prediction, and estimates conditional cohort weights with a separate Gaussian-kernel function. It accommodates multiple isolated employee terminations at the same store. Event selection and control selection are extensions to the article's design; the regression calculation remains the reference calculation. All estimates in the first version are point estimates.

The public API consists of one class and two functions:

```python
model = ATT(degree=3, num_knots=0, min_cohort_size=10)
model.fit(
    outcomes,
    events,
    event_times=[0, 1, 7, 14],
    isolation_window=(7, 14),
    history_window=(history_start, history_end),
)

unpooled = model.predict(doses=[4, 6, 8, 10, 12])

weights = estimate_pi(
    model.events_,
    doses=[4, 6, 8, 10, 12],
    eligible_event_ids=eligible_event_ids,
    bandwidth=bandwidth,
)

theta = evaluate_theta(
    model,
    doses=[4, 6, 8, 10, 12],
    bandwidth=bandwidth,
    event_times=[0, 7, 14],
)
```

The standalone weight example requires the precise retained event IDs eligible at one event day. `evaluate_theta()` constructs that set automatically. The bandwidth is supplied in hours. The illustrated isolation window is a configuration example, not a recommended empirical choice; `history_start` and `history_end` declare the dates over which termination recording is complete for all outcome stores.

**Event identifiers and store outcomes.** Let $i=(s,j)$ identify an employee employment spell $j$ at store $s$, with termination day $G_i$ and previous daily scheduled hours $D_i>0$. Use a separate spell identifier if an employee is rehired. Several sufficiently separated events from the same store can enter the analysis. There is no single permanent focal employee or first termination selected for each store: each eligible termination is focal in its own comparison.

The event's treatment label and associated outcome are

$$
W_{it}=D_i\thinspace{}1\lbrace t\ge G_i\rbrace,
\qquad Y_{it}=Y_{s(i),t}.
$$

The dose is the fixed number of daily hours removed by that employee's termination, not the store's total staffing reduction. The event label is absorbing even though the store can experience other events. A replacement or redistribution of work can offset the staffing loss without changing this label. Such staffing responses belong to the total effect of the termination.

Store sales are shared outcomes. Copying the sales series across employees does not create independent observations or valid untreated coworkers. If employee A leaves a store, employee B at that same store cannot serve as a control merely because B has not left. Keep the outcomes and events in separate tables and construct comparison samples explicitly.

**Isolation and event eligibility.** Prespecify an isolation window $(L,H)$, where $L\ge1$ and $H\ge\max(e)$ across fitted event days. For an event at $g$, screen the closed calendar interval $[g-L,g+H]$. An event is eligible only if it has an observed baseline, lies within the outcome period, has complete termination-history coverage for that entire interval, and no other employee termination at its store occurs inside it. Simultaneous terminations therefore exclude every affected event. Apply this rule to each event against the complete original history, before any filtering; do not remove one event and thereby make another artificially eligible.

Here, an overlapping event means another termination falls within the specified screening interval. Reuse of a store's sales observations in separate comparisons is still possible and creates dependence. Window lengths must address the plausible duration of anticipation and changing carryover effects; spacing alone does not establish identification.

Write $S_i=1$ for an event passing these design restrictions and $\mathcal E$ for the resulting event population, before small-cohort screening. Keep $(L,H)$ fixed when comparing different event days or prediction requests. Refitting with a different window can change even the event-day-zero target. Do not infer complete history from the first and last recorded termination dates. If history ends at $T$, events whose screening interval extends beyond $T$ are ineligible even for shorter horizons. Complete termination records beyond the sales panel can avoid that exclusion without providing additional outcome observations.

**Statistical target and identifying restriction.** For an eligible event, let $Y^{\mathrm{term}}_{i,t}$ be store sales under the observed termination and $Y^{\mathrm{keep}}_{i,t}$ sales if that employee instead continues at their original daily hours through the comparison endpoint. The regimes share the actual history before the current event; subsequent staffing responses follow their respective regimes. The contrast does not remove every earlier termination from the store's history.

The event-specific analogue of the article's unpooled target is

$$
ATT_{\mathcal E}(g,t,d\mid g,d)
=E[Y^{\mathrm{term}}_{i,t}-Y^{\mathrm{keep}}_{i,t}
\mid G_i=g,D_i=d,S_i=1].
$$

For $e=t-g\ge0$, define

$$
\mathcal G_e=\lbrace g:\ g\text{ has eligible events},\ g+e\le T\rbrace.
$$

The requested aggregation, adapted to eligible events, is

$$
\theta_{\mathcal E}(d,e)=\sum_{g\in\mathcal G_e}\pi_g(e,d)
ATT_{\mathcal E}(g,g+e,d\mid g,d),
\qquad
\pi_g(e,d)=P(G_i=g\mid D_i=d,S_i=1,G_i\in\mathcal G_e).
$$

Dose and event day remain fixed; only calendar-day cohorts are averaged. These probabilities sample termination events, not stores with equal total weight. A store with more eligible terminations contributes more events. Throughout the API, the short names `att` and `theta` refer to these event-population quantities, subject to the additional cohort restriction described below.

Let $\mathcal C_g$ be the clean control stores defined below. In addition to consistency, no anticipation of the focal termination, and no relevant spillovers between stores, the required parallel-trends restriction is

$$
E[Y^{\mathrm{keep}}_{i,t}-Y^{\mathrm{keep}}_{i,g-1}
\mid G_i=g,D_i=d,S_i=1]
=E[Y_{s,t}-Y_{s,g-1}\mid s\in\mathcal C_g].
$$

The right-hand expectation weights control stores equally. This is an explicit identifying assumption for the selected events and their comparison stores, not a result implied by distance between terminations. Earlier terminations may leave persistent level effects that difference out; changing carryover effects must still be compatible with this restriction. The first version has no covariate or treatment-history adjustment to repair a failure of that assumption.

Selection on the absence of later terminations uses post-event information and may itself be affected by the focal event. Identification is therefore asserted only under the stated restriction for the selected population. It does not recover effects for excluded events or identify a population that would remain isolated under both counterfactual regimes. Parametric estimation also requires an adequate specification of each cohort-time dose curve and relevant dose support. Strong parallel trends is not assumed, and slopes across doses are not interpreted as causal marginal responses.

**Differences from the article.** Callaway, Goodman-Bacon and Sant'Anna (2025), abbreviated CGBS, provide the starting point. The distinctions below are substantive design changes, not merely a relabeling of the unit.

| Feature | Article's staggered-adoption setting | This plan |
| --- | --- | --- |
| Treatment histories | One adoption date and a fixed subsequent dose per unit, Assumption 3-MP, printed p. 39 | One absorbing label per termination spell, but potentially several events affecting a store |
| Observation structure | Independent sampled units, Assumption 1-MP, p. 39 | Events within a store share outcomes and histories; independent event sampling is not assumed |
| Counterfactual and PT | Outcomes without treatment; PT-MP and Theorem C.1, pp. 39–40 | No current focal termination, retaining prior history; a separate PT restriction for selected events and clean comparison stores |
| Controls | Not-yet-treated or never-treated units, Theorem C.1, p. 40 | Stores without a termination in the specified window; sufficiently earlier terminations are allowed only under the stated identifying restriction |
| Aggregation population | Dose-specific event-study aggregation over treatment cohorts, p. 41 | The same aggregation structure applied to eligible termination events, with an additional retained-cohort restriction |

The article's dose-specific event-study aggregation on printed page 41 is unnumbered; it is the source of the aggregation structure, not an equation proving this repeated-event extension. Remark C.1 on page 42 discusses estimating components and aggregating them. [CGBS, Appendix C](https://psantanna.com/files/CGBS_v4.pdf#page=40)

**Relationship to the R implementation.** The reference calculation is the parametric level-effect part of `contdid::cont_did_acrt`: regress raw outcome changes on an intercept and dose B-splines among the focal treated cohort, then subtract the comparison group's mean change. Use baseline $g-1$ and reproduce that calculation for each constructed comparison. [Estimator and subset source](https://github.com/bcallaway11/contdid/blob/main/R/cont_did.R)

The event-based input tables, isolation screen, and store-history-based control rule are custom additions. Passing a duplicated store–employee panel directly to the package would not implement them. The earlier `control_group` choices are replaced by the single clean-window rule below; a coworker's own future termination date is never sufficient for control eligibility.

Other intentional implementation differences are retaining the fitted basis at prediction, warning and dropping small or unidentifiable target cohorts, and adding dose-conditional kernel aggregation. Thus compatibility refers to the regression calculation on the same comparison sample with a consistent fitted basis. The inspected R prediction path rebuilds its basis on the prediction grid. Sources were inspected on 2026-11-03; implementation validation should record the R version or source revision actually used.

**Input contract.** `ATT.fit()` accepts two pandas DataFrames. The outcome table has these standard column names:

| Column | Meaning |
| --- | --- |
| `store_id` | Store identifier |
| `day` | Integer calendar-day index from 1 through $T$ |
| `sales` | Finite numeric daily store sales |

The event table contains the complete termination history within the declared history window for those stores:

| Column | Meaning |
| --- | --- |
| `event_id` | Unique store–employee employment-spell identifier |
| `store_id` | Store whose sales are affected |
| `employee_id` | Employee identifier; may repeat across distinct employment spells |
| `cohort` | Integer termination day on the same calendar as `day` |
| `dose` | Positive finite daily hours removed at termination |

The outcome panel is balanced, with exactly one row per `(store_id, day)` and no missing days or outcomes. The events table has one row per termination and unique `event_id`; stores with no terminations need no event row. Event dates may lie outside the outcome panel but must lie within the declared history window. A target event must have $2\le G_i\le T$. Do not encode never-treated stores as artificial zero-dose events. Validate identifiers, foreign keys, integer dates, finite values, and nonmissing required columns; reject duplicate event IDs rather than silently deduplicating them. Different event IDs at the same store/date are simultaneous events and are excluded by the isolation rule.

`history_window=(a,b)` is a caller declaration of complete termination recording for every outcome store, including stores with no recorded events, with $a\le1\le T\le b$. Coverage cannot be verified from event records alone. Preserve all history rows, including events outside the sales panel, overlapping events, and events in dropped cohorts, for subsequent control screening. Calendar dates can be converted to integer offsets before fitting; missing calendar days must not be compressed. Validate the inputs rather than balancing or imputing them silently. Fitting must not alter the caller's DataFrames.

Runtime dependencies are NumPy, pandas, PyFixest, SciPy, and the Python standard library. Exact runtime pins and development-tool requirements are declared in [pyproject.toml](pyproject.toml). The project requires Python 3.11; [.python-version](.python-version) selects that interpreter for tools such as uv. The first version covers balanced daily outcomes, equal treated-event weights within each regression, equal control-store weights, no anticipation, and post-treatment level estimates. Covariates, inference, plotting, treatment slopes, and changing the dose within an event are outside this initial scope.

Use the supplied [.pre-commit-config.yaml](.pre-commit-config.yaml) for spelling, formatting, lint, and type checks on Python, Markdown, and notebooks. The hooks use system executables, so activate the development environment before installing or running them:

```bash
uv sync --python 3.11 --group dev
source .venv/bin/activate
pre-commit install
pre-commit run --all-files
```

**`ATT` owns independent cohort-time fits.** The class manages the two tables but never pools regression coefficients across cohorts. Its signatures are:

```python
ATT(*, degree=3, num_knots=0, min_cohort_size=10)

ATT.fit(
    outcomes,
    events,
    *,
    event_times,
    isolation_window,
    history_window,
) -> ATT

ATT.predict(doses, *, event_times=None) -> pandas.DataFrame
```

Construction requires a positive integer `degree`, a nonnegative integer `num_knots`, and a positive integer `min_cohort_size`. Require integer window bounds with $L\ge1$ and $H\ge0$, and distinct nonnegative fitted event days no greater than $H$. Dose requests must be distinct, finite, and positive. Event-time and dose requests must be nonempty.

`fit()` returns `self` and replaces any previously fitted state; a failed refit must not leave an earlier model available for prediction. `predict()` before successful fitting raises an error. Omitting `event_times` in prediction requests all fitted horizons; explicitly requesting an unfitted horizon raises an error rather than fitting additional models implicitly.

For a retained cohort $g$ and an in-window horizon $e$, set $t=g+e$. Use one treated row per eligible event at $g$. Under the isolation rule these events necessarily occur at distinct stores; assert that fact when constructing the sample. Select control stores using every recorded termination, not only retained target events:

$$
\mathcal C_g=\lbrace s:\quad
\text{no termination at store }s\text{ lies in }[g-L,g+H]\rbrace.
$$

Count each control store once, regardless of its number of employees or recorded events. A treated store cannot also enter its own comparison as a control. A store may serve as a control for another comparison outside the window of its termination, including after an earlier termination, only under the stated PT and carryover conditions. The same fixed screening window applies at all fitted horizons for $g$; control means still depend on the endpoint. This rule deliberately screens through $g+H$ even when estimating a shorter horizon.

Compute $\Delta Y_s^{g,t}=Y_{s,t}-Y_{s,g-1}$ on those exact calendar dates. Let $b_g(d)$ denote the saved B-spline columns, excluding the first column of the full B-spline basis. The fitted design includes a separate intercept:

$$
x_g(d)=(1,b_g(d)')',\qquad
\widehat\gamma_{gt}
=\arg\min_\gamma\sum_{i:G_i=g,S_i=1}
\left(\Delta Y_{s(i)}^{g,t}-x_g(D_i)'\gamma\right)^2.
$$

The control correction is calculated separately and stored:

$$
\widehat m_{0,gt}
=\frac{1}{|\mathcal C_g|}\sum_{s\in\mathcal C_g}\Delta Y_s^{g,t}.
$$

Prediction is

$$
\widehat{ATT}_{\mathcal E}(g,t,d\mid g,d)
=x_g(d)'\widehat\gamma_{gt}-\widehat m_{0,gt}.
$$

The intercept belongs to the raw treated-outcome regression. The stored control mean is subtracted exactly once. Neither controls nor kernel weights enter the treated OLS fit. Use a numerically stable least-squares solver, not an explicit inverse of $X'X$.

**Fitted basis and knots.** Let $q$ be the degree and $K$ the number of internal knots. The regression has $p=q+K+1$ columns, including the intercept. Defaults $q=3,K=0$ give a cubic polynomial expressed in a B-spline basis.

For $K>0$, choose shared internal knots once using the positive doses from one row per design-eligible termination event, before the small-cohort screen. Use quantiles at $k/(K+1)$, $k=1,\ldots,K$, with linear interpolation matching R's default quantile convention. This retains the package's global knot-selection structure, now applied to the eligible event population. Keep the selected knots fixed after screening. [Setup source](https://github.com/bcallaway11/contdid/blob/main/R/setup_pte_cont.R), [quantile helper](https://github.com/bcallaway11/contdid/blob/main/R/helper_functions.R)

For each retained cohort, use its observed treated-dose minimum and maximum as boundary knots. Build and store the resulting basis definition: degree, internal knots, repeated boundary knots, and retained-column convention. A SciPy `BSpline` representation can evaluate that saved basis. Construct the full basis, omit its first column, and add an explicit intercept, consistent with the R parameterization. [B-spline conventions](https://wwenjie.org/splines2/reference/bspline)

Reject duplicate global internal knots with a configuration error; do not silently change $K$. If a shared internal knot is not strictly inside a cohort's boundary knots, warn and drop that target cohort with an `incompatible_knots` reason. Degenerate dose ranges and rank-deficient designs are handled by the cohort screen below.

`predict()` must evaluate the fitted basis without recalculating knots or boundaries from the requested doses. In particular, a prediction at dose 8 must be identical whether requested alone or alongside other doses. Because event selection uses a fixed isolation window and the outcome panel is balanced, the same cohort basis can be reused across its horizons; coefficients and control means remain specific to each comparison.

To stay close to the parametric routine, allow polynomial continuation outside a cohort's observed dose range, issue an extrapolation warning, and flag those rows. Do not clip doses, discard cohorts at particular doses, or renormalize weights because of extrapolation. Such values rely on the fitted functional form beyond observed support.

**Small-cohort screening and dropping.** The proposed configurable default is `min_cohort_size=10` distinct stores. This is an implementation choice, not a threshold supplied by the article or a guarantee of precision. Let $n_{\mathrm{user}}$ denote the configured `min_cohort_size`. The effective minimum is

$$
n_{\min}=\max(n_{\mathrm{user}},p+1),
$$

which also requires positive residual degrees of freedom. Count distinct treated stores after the isolation screen, not employee records or daily rows. This equals the number of eligible events in that cohort; a store may still contribute separate events to other cohorts. A cohort is retained only if its sample size meets that minimum, it has at least $p$ distinct dose values, and its actual regression matrix has full column rank. Use the least-squares solver's documented numerical rank tolerance consistently. The actual rank check is necessary even when the distinct-dose check passes.

For every dropped target cohort, emit one warning identifying its date, number of stores, number of distinct doses, required design dimension, and reason. Record the same information in `cohort_diagnostics_`. Never silently lower the degree, remove knots, or pool adjacent calendar cohorts to rescue a fit.

Screening depends on the cohort's dose design, so drop that cohort from target estimation at every event day. **Keep all its events in the complete termination history and all its stores in the outcome panel.** Those stores may still serve as controls in other clean windows. Dropping an event or cohort must never erase a termination from control screening or make its store appear untreated during that termination.

Ordinary horizon exclusions, $g+e>T$, are not small-cohort failures. If a retained in-window cohort-time comparison has zero controls, warn and record `no_controls` in `cell_diagnostics_`; no model is fitted for that cell. `evaluate_theta()` must raise a clear error for an affected requested horizon rather than silently remove that comparison and renormalize. A singleton control group is computable for point estimation; its count is reported.

If all target cohorts fail screening, or no requested comparison can be fitted, `fit()` raises an error after recording the reasons. These failure rules are deliberately explicit; predictions from failed cells are not filled with zero.

**Stored state and unpooled output.** Keep only the information needed for prediction, weights, and diagnostics:

- `events_`: one row per original termination, with original identifiers, date, dose, and a `design_eligible` flag; includes excluded events and dropped cohorts.
- `event_diagnostics_`: eligibility and reasons such as `outside_outcome_period`, `incomplete_history`, and `overlapping_termination`, with conflicting event IDs where relevant.
- `event_times_`, `last_day_`, `isolation_window_`, `history_window_`, `knots_`, and `retained_cohorts_`.
- `bases_`: saved basis definitions keyed by cohort.
- `models_`: separate coefficient vectors, control means, and sample counts keyed by `(cohort, event_time)`.
- `cohort_diagnostics_`: retained/dropped status and design information for every cohort containing design-eligible events.
- `cell_diagnostics_`: in-window eligibility, control counts, and fit status for every requested cohort/horizon pair, together with the contributing event IDs and unique control-store IDs.

`ATT.predict()` returns one row for each available `(cohort, event_time, dose)` combination, with columns `cohort`, `day`, `event_time`, `dose`, `att`, `n_treated`, `n_controls`, and `extrapolated`. Sort deterministically by cohort, event time, and dose. If unavailable in-window cells exist, warn that the returned table is incomplete and direct the caller to `cell_diagnostics_`. Out-of-window combinations are simply absent.

**Separate kernel-weight function.** This function is independent of fitted outcomes and of `ATT` internals:

```python
estimate_pi(
    events,
    doses,
    *,
    eligible_event_ids,
    bandwidth,
) -> pandas.DataFrame
```

`events` must have one row per termination with unique `event_id` and valid cohort/dose labels. Repeated `store_id` values are permitted. `eligible_event_ids` is an explicit nonempty collection of unique IDs, all present in `events`. The caller must supply only events eligible for the intended target and horizon; the function does not decide isolation or fit regressions. Validate IDs and filter to that precise set before any kernel normalization. A cohort-only argument is insufficient because the isolation screen can exclude individual events within a retained cohort.

Let $\mathcal I_e$ be the supplied set of event IDs and $\mathcal R_e$ the represented cohorts. For a common bandwidth $h>0$, use

$$
q_g(d;h)=\sum_{i\in\mathcal I_e:G_i=g}
\exp\left[-\frac12\left(\frac{D_i-d}{h}\right)^2\right],
\qquad
\widehat\pi_g(e,d)
=\frac{q_g(d;h)}{\sum_{k\in\mathcal R_e}q_k(d;h)}.
$$

Each supplied event contributes exactly once. Do not count daily rows, duplicate sales by workforce size, deduplicate stores across different events, or average kernel contributions within each cohort. Summing event contributions preserves cohort sizes and the event-weighted target. The Gaussian normalization constant and common $1/h$ cancel. All weights are nonnegative and sum to one at each requested dose. Compute normalization with log-sum-exp; invalid bandwidths or nonfinite intermediate results raise errors rather than triggering uniform weights.

Return `cohort`, `dose`, and `weight`, sorted by cohort and dose. No event-time argument is needed: supplying a different eligible-event set supplies the horizon dependence. The function can also inspect sensitivity to dose and bandwidth.

This is a custom estimator of conditional cohort probabilities. The article does not prescribe this kernel or bandwidth, and the package's existing dose aggregation does not supply these conditional weights. A positive bandwidth smooths cohort composition over nearby doses; consistency for exact conditional probabilities requires suitable smoothness, bandwidth, and sample-size conditions. Repeated events from one store do not create independent information. Automatic bandwidth selection is outside the first version.

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

1. Select all design-eligible events with $G_i+e\le T$. Their represented cohorts form $\mathcal G_e$.
2. Let $\mathcal R$ be `retained_cohorts_`. Retain only events in those cohorts, obtaining $\mathcal I_e$ and $\mathcal R_e=\mathcal G_e\cap\mathcal R$. If empty, raise an error.
3. Require a fitted cell for every cohort in $\mathcal R_e$. A missing control group is an error here, not an additional implicit selection rule.
4. Obtain unpooled predictions from `ATT.predict()` and weights from `estimate_pi(model.events_, ..., eligible_event_ids=I_e)`.
5. Join by exact cohort and dose labels. Require a complete one-to-one match and weights summing to one; do not let an inner join discard unmatched components.
6. Calculate

$$
\widehat\theta_{\mathcal E,\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
\widehat\pi_g(e,d)\thinspace{}
\widehat{ATT}_{\mathcal E}(g,g+e,d\mid g,d).
$$

The function only aggregates existing estimates. It neither refits the curves nor changes the window or screening. Changing bandwidth changes the weights, not the underlying ATT models.

Return `event_time`, `dose`, `theta`, `n_cohorts_eligible`, `n_cohorts_used`, `n_events_eligible`, `n_events_used`, `n_stores_eligible`, `n_stores_used`, `restricted_target`, and `extrapolated`. Eligible counts refer to the design-eligible population at that horizon before the cohort screen; used counts refer to the retained population. Store counts deduplicate across events for reporting only, not weighting. `restricted_target` is true whenever any cohort-screen rule excludes an eligible cohort, including small samples, rank failures, and incompatible knots. This flag does not negate the isolation restriction, which always defines the target and is recorded in the model's window and event diagnostics. `extrapolated` is true if any contributing prediction lies outside its cohort's observed dose range. Sort by event time and dose. Component estimates and weights remain inspectable through the separate APIs.

**Dropping events or cohorts changes the target.** Isolation and history coverage first restrict the population to $S_i=1$. After the cohort screen, weights estimate

$$
P(G_i=g\mid D_i=d,S_i=1,G_i\in\mathcal R_e).
$$

For a given retained set, the population counterpart is

$$
\theta_{\mathcal E,\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
P(G_i=g\mid D_i=d,S_i=1,G_i\in\mathcal R_e)
ATT_{\mathcal E}(g,g+e,d\mid g,d).
$$

This equals the pre-cohort-screen target only if every eligible cohort is retained or under additional special conditions. It is not generally the effect over all terminations or an equal-store average. The short output name `theta` must retain this interpretation in documentation and diagnostics. Warn when the cohort screen restricts an aggregation, listing affected horizons. Never retain excluded events in the kernel denominator, replace unknown effects with zero, or imply that renormalization recovers excluded effects.

The cohort screen is sample-dependent; this target description is not an inferential result about selection. Cohort composition may also change with outcome availability at each horizon. No additional common balanced cohort set is imposed across event days, although the fixed isolation window and available termination-history coverage can themselves restrict the set to cohorts observed at every fitted horizon.

**Implementation boundaries and validation plan.** Put the class, both public functions, and a few private validation, window-selection, and basis helpers in one `att.py` file. Use ordinary warning categories and small dictionaries or records for fitted cells. External R code is not a runtime dependency. No generic estimator framework, serialization layer, or extra classes are needed.

When the Python implementation is written, verify these behaviors with targeted examples:

- A two-period comparison with a known polynomial dose effect and a nonzero control trend recovers the curve, subtracting the control mean once.
- Predictions at a fixed dose are invariant to other requested doses and input row order. Where R training and prediction bases coincide, unpooled estimates on the same constructed sample agree numerically; otherwise compare against an R reference that explicitly reuses the training basis.
- Two sufficiently separated events at one store both enter eligible cohorts and kernel weights. Coworkers never become extra control rows, and every control store contributes once per comparison.
- Simultaneous and overlapping events are excluded using the original complete history. An excluded event or a dropped cohort still disqualifies its store from controls in every conflicting window.
- Incomplete screening coverage excludes an event without silently assuming no future termination. Complete extended history permits late events to contribute at observed shorter horizons. Changing only the prediction horizons leaves the fitted isolation screen unchanged.
- Small, rank-deficient, and incompatible-knot cohorts warn, have diagnostics, and disappear from target predictions and kernel denominators while remaining in the termination history.
- Weights count each selected event once, sum to one, reflect unequal cohort sizes, and match the explicit formula. A within-cohort excluded event cannot leak back into the weights.
- Aggregation equals a manual weighted sum, reports distinct event and store counts, identifies the retained population, and rejects a horizon with a missing required cell.

Only the specification is delivered now. Future inference must cluster or resample by store, carrying all its events and outcomes together, and account for shared controls, estimated weights, smoothing, and selection. Clustering handles dependence; it does not fix contaminated controls or justify the identifying assumption. Store-level independence and an adequate number of independent stores would still require assessment.

## References

Callaway, B., Goodman-Bacon, A. and Sant'Anna, P.H.C. (2025) *Difference-in-Differences with a Continuous Treatment*. Working paper, draft dated 31 December. Available at: [https://psantanna.com/files/CGBS_v4.pdf](https://psantanna.com/files/CGBS_v4.pdf) (Accessed: 3 November 2026).
