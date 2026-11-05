**Implementation specification: unpooled continuous-dose ATT and dose-specific event-study aggregation.**

This document specifies a future self-contained Python module, `att.py`. It does not implement that module. The design uses independent cohort-time regressions following the parametric `contdid` calculation, retains each fitted spline basis for prediction, and estimates conditional cohort weights with a separate Gaussian-kernel function. It estimates the effect of each store's first recorded termination, with at most one target event per store. Event selection and control selection are extensions to the article's design; the regression calculation remains the reference calculation. All estimates in the first version are point estimates.

The public API consists of one class and two functions:

```python
model = ATT(
    degree=3,
    num_knots=0,
    min_cohort_size=None,
    firm_col="firm_id",
    date_col="date",
    outcome_col="sales",
    dose_col="dose",
)
model.fit(
    outcomes,
    events,
    event_times=[0, 1, 7, 14],
    isolation_window=(-7, 14),
)

unpooled = model.predict(doses=[4, 6, 8, 10, 12])

weights = estimate_pi(
    model.events_,
    doses=[4, 6, 8, 10, 12],
    eligible_firms=eligible_firms,
    bandwidth=bandwidth,
)

theta = evaluate_theta(
    model,
    doses=[4, 6, 8, 10, 12],
    bandwidth=bandwidth,
    event_times=[0, 7, 14],
)
```

The standalone weight example requires the precise retained store identifiers eligible at one event day. `evaluate_theta()` constructs that set automatically. The bandwidth is supplied in hours. The illustrated isolation window is a configuration example, not a recommended empirical choice. Screening bounds are inferred from the overall minimum and maximum normalized outcome dates across all stores.

**Store units and first terminations.** Store $i$ is the unit. Let $G_i$ be its earliest recorded termination date in the supplied history, and let $D_i>0$ be the daily hours removed by that termination. The store identifier is the sole estimation-unit key. Employee identifiers are neither required nor used. Keep only this first termination as the store's estimation candidate; later terminations never become additional targets or enter its aggregation weights.

The fixed treatment label is

$$
W_{it}=D_i\thinspace{}1\lbrace t\ge G_i\rbrace.
$$

The outcome $Y_{it}$ is daily store sales. The dose is the hours removed by the selected termination, not the store's total subsequent staffing reduction. Replacement hiring or redistribution of work does not change this label; those responses belong to the total effect of the selected termination. Keep store-day outcomes and termination records in separate tables, with at most one treated observation per store in each comparison.

**Isolation and event eligibility.** Prespecify an isolation window $(L,H)$, where $L\le-1$ and $H\ge\max(e)$ across fitted event days. The tuple contains signed offsets in calendar days: `(-7, 14)` means seven days before through fourteen days after termination, with both boundaries included. For an event at $g$, screen the closed calendar interval $[g+L,g+H]$. An event is design-eligible only if its entire isolation interval lies within the overall outcome panel bounds and no other termination at its store occurs inside it, including any later termination that is excluded from target estimation. In internal day indices, require $g+L\ge1$ and $g+H\le T$, with equality allowed at either boundary. Assume termination recording is complete for every outcome store throughout these panel bounds, including days with no outcome row. Simultaneous terminations therefore exclude every affected event. Apply this rule to each first-event candidate using the complete distinct-date termination history described below. Exclude only the one selected termination from its store's overlap count, not all records at that store or on that date. The same-day rule below specifies how multiple input records are counted. Dropping rows from target estimation must not make another event artificially eligible.

Here, an overlapping event means another termination falls within the specified screening interval. Reuse of a store's sales observations in separate comparisons is still possible and creates dependence. Window lengths must address the plausible duration of anticipation and changing carryover effects; spacing alone does not establish identification.

Write $S_i=1$ for an event passing these design restrictions and $\mathcal E$ for the resulting event population, before endpoint-availability and small-sample screening. Keep $(L,H)$ fixed when comparing different event days or prediction requests. Refitting with a different window can change even the event-day-zero target. Do not derive screening bounds from termination dates or from an individual store's first and last observed sales dates. An event whose isolation interval extends beyond the overall panel bounds is ineligible at every requested horizon, including shorter horizons with observed endpoints. Additional termination records outside the panel do not extend these bounds. Completeness of termination recording within the bounds is a data assumption, not something inferred from sales observations.

**Statistical target and identifying restriction.** For each comparison let $e=t-g\ge0$. Define $O_{s,u}=1$ when the outcome table contains a valid observation for store $s$ on calendar date $u$, and zero otherwise. Event endpoint availability is

$$
A_i(e)=O_{i,G_i-1}\thinspace{}O_{i,G_i+e}.
$$

Both dates must be observed exactly. Missing intervening days do not prevent that particular long-difference comparison. Let $\mathcal E_e$ contain the design-eligible events with $A_i(e)=1$.

For an eligible event, let $`Y^{\mathrm{term}}_{i,t}`$ be store sales under the observed termination and $`Y^{\mathrm{keep}}_{i,t}`$ sales if that employee instead continues at their original daily hours through the comparison endpoint. The regimes share the actual history before the current event; subsequent staffing responses follow their respective regimes. The contrast does not remove every earlier termination from the store's history.

The event-specific analogue of the article's unpooled target, restricted to observed endpoints, is

$$
ATT_{\mathcal E_e}(g,t,d\mid g,d)
=E[Y^{\mathrm{term}}_{i,t}-Y^{\mathrm{keep}}_{i,t}
\mid G_i=g,D_i=d,S_i=1,A_i(e)=1].
$$

Define $\mathcal G_e$ as the cohorts represented among $\mathcal E_e$. The requested aggregation is

$$
\theta_{\mathcal E}(d,e)=\sum_{g\in\mathcal G_e}\pi_g(e,d)
ATT_{\mathcal E_e}(g,g+e,d\mid g,d),
\qquad
\pi_g(e,d)=P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1).
$$

Dose and event day remain fixed; only calendar-day cohorts are averaged. Each eligible store contributes its first termination exactly once when estimating these cohort probabilities. A store with additional termination records receives no extra weight. Kernel contributions depend on distance from dose $d$, so weights need not be equal across stores at a particular requested dose. Throughout the API, `att` and `theta` refer to these quantities, subject to the additional comparison-specific cohort restriction below.

Let $\mathcal C_{g,e}$ be the clean control stores with observed endpoints, defined below. In addition to consistency, no anticipation of the focal termination, and no relevant spillovers between stores, the identifying parallel-trends restriction is

$$
E[Y^{\mathrm{keep}}_{i,t}-Y^{\mathrm{keep}}_{i,g-1}
\mid G_i=g,D_i=d,S_i=1,A_i(e)=1]
=E[Y_{s,t}-Y_{s,g-1}\mid s\in\mathcal C_{g,e}].
$$

The right-hand expectation weights control stores equally. This is an assumption for the selected events and their observed comparison stores, not a result implied by distance between terminations. Earlier terminations may leave persistent level effects that difference out; changing carryover effects must still be compatible with this restriction. The first version has no covariate or treatment-history adjustment to repair a failure of that assumption.

Selection on later terminations and on observed outcomes uses post-event information and may itself be affected by the focal event. Identification is asserted only under the stated restriction for the selected population. In particular, excluding stores after closure does not recover a treatment effect that includes their subsequent sales or closure consequences. Generalizing to all isolated events requires additional assumptions or a different outcome definition. The target is not a population guaranteed to remain isolated and observed under both counterfactual regimes. Parametric estimation also requires an adequate specification of each comparison's dose curve and relevant dose support. Strong parallel trends is not assumed, and slopes across doses are not interpreted as causal marginal responses.

**Relationship to the article.** Callaway et. al. (2025) provide the starting point. With stores as units and one selected termination per store, the treatment-label structure now matches one adoption date and fixed dose per unit. The isolation screen, observed-outcome selection, and comparison-store rule still require the distinctions below.

| Feature | Article's staggered-adoption setting | This plan |
| --- | --- | --- |
| Treatment histories | One adoption date and a fixed subsequent dose per unit, Assumption 3-MP, printed p. 39 | One first-termination label and dose per store; later terminations are used only to screen overlap and control eligibility |
| Observation structure | Independent sampled units, Assumption 1-MP, p. 39 | One outcome trajectory per store; independence across stores remains an assumption, and a store may be reused across comparisons |
| Outcome availability | The stated panel contains outcomes in every modeled period | Store-day gaps are allowed; each comparison uses its own observed endpoints and therefore its own selected population |
| Counterfactual and PT | Outcomes without treatment; PT-MP and Theorem C.1, pp. 39–40 | No current focal termination, retaining prior history; a separate PT restriction for selected events and clean comparison stores with observed endpoints |
| Controls | Not-yet-treated or never-treated units, Theorem C.1, p. 40 | Stores without a termination in the specified window; sufficiently earlier terminations are allowed only under the stated identifying restriction |
| Aggregation population | Dose-specific event-study aggregation over treatment cohorts, p. 41 | The same structure applied to stores with eligible first terminations and observed endpoints, with a horizon-specific retained-cohort restriction |

The article's dose-specific event-study aggregation on printed page 41 is unnumbered; it is the source of the aggregation structure, not an equation establishing identification for these selected store comparisons. Remark C.1 on page 42 discusses estimating components and aggregating them. [Callaway et. al. (2025), Appendix C](https://psantanna.com/files/CGBS_v4.pdf#page=40)

**Relationship to the R implementation.** The reference calculation is the parametric level-effect part of `contdid::cont_did_acrt`: regress raw outcome changes on an intercept and dose B-splines among the focal treated cohort, then subtract the comparison group's mean change. Use baseline $g-1$ and reproduce that calculation for each constructed comparison. [Estimator and subset source](https://github.com/bcallaway11/contdid/blob/main/R/cont_did.R)

The separate input tables, first-event selection, isolation screen, and explicit handling of missing store-days are custom additions. The clean-window control rule can include stores whose first termination occurred earlier, provided their history and counterfactual trends satisfy the stated assumption. Keeping one target event per store does not make such stores untreated in the article's absorbing-treatment sense; this control rule remains an extension to its comparison design.

Other intentional implementation differences are retaining the fitted basis at prediction, warning and dropping small or unidentifiable cohort–event-day comparisons, and adding dose-conditional kernel aggregation. Thus compatibility refers to the regression calculation on the same comparison sample with a consistent fitted basis. The inspected R prediction path rebuilds its basis on the prediction grid. Sources were inspected on 2026-11-03; implementation validation should record the R version or source revision actually used.

**Input contract and configurable columns.** `ATT.fit()` accepts two pandas DataFrames. All four input-column parameters are required keyword-only constructor arguments with no defaults. The names supplied in the introductory example are explicit caller choices. The outcome table uses:

| `ATT` parameter | Meaning |
| --- | --- |
| `firm_col` | Store identifier |
| `date_col` | Outcome date; dtype `datetime64[ns]`, with time ignored |
| `outcome_col` | Finite numeric daily store sales; no missing values on present rows |

The event table must record all terminations within the overall outcome panel date range for those stores:

| `ATT` parameter | Meaning |
| --- | --- |
| `firm_col` | Store whose sales are affected |
| `date_col` | Termination date; dtype `datetime64[ns]`, with time ignored |
| `dose_col` | Positive finite daily hours removed at termination |

The shared `firm_col` and `date_col` parameters name columns present in both input tables. Their labels must match across the tables; `date_col` identifies outcome dates in the outcome table and termination dates in the event table. If the source tables use different labels, rename them before fitting. Store identifier values must refer to the same stores across tables. Each parameter must be a nonempty string naming an existing column. Require distinct mapped columns within each table and unique DataFrame column labels. `outcome_col` and `dose_col` may have the same name because they belong to different tables. Extra columns are ignored. Use the supplied store identifier directly as the estimation key. No employee, spell, or event identifier column is required or generated.

Validate both date columns as timezone-naive `datetime64[ns]` with no `NaT`. Do not silently parse strings, accept object-date columns, or convert timezone-aware datetimes. Normalize copies with `.dt.normalize()` so only the calendar date is used while retaining `datetime64[ns]`. Validate uniqueness of `(firm_id, normalized date)` in the outcome table after normalization. Two intraday rows that collapse to the same store-day are an error, not an instruction to sum or select one. Multiple event rows per store are allowed and processed by the first-event and same-day rules below.

**First-event selection and duplicate-event logging.** Validate required input values and normalize dates before selection. Group only by the column named by `firm_col`, sort by the normalized `date_col` using a stable sort, and retain that entire first row for each store. If several rows share the earliest date, retain the first in original input order as the candidate. Ignore time of day and retain that row's dose without averaging or summing across rows.

Select the first event across the full supplied history, before isolation, outcome availability, or sample-size screening. If it is subsequently ineligible, do not promote a later event from the same store. The target is the effect of each eligible store's earliest recorded termination in the supplied history, not necessarily its first-ever termination outside that history.

Let `n_dropped_events` be the input row count minus the number of distinct stores, and `n_affected_firms` the number of stores with more than one input row. When `n_dropped_events > 0`, emit one aggregate message per `fit()` call:

```python
logging.warning(
    "Dropped %d additional termination rows from the target table for %d "
    "firms; kept the first event by date for each firm.",
    n_dropped_events,
    n_affected_firms,
)
```

For example, three rows for one store and two for another mean three dropped rows for two affected stores. Include both same-day and later-dated records in the count. With no repeated stores, do not emit this message. Record both counts in `duplicate_diagnostics_`.

Only first-event rows populate `events_`, fitted targets, global knot selection, and kernel inputs. Build `termination_history_` separately from all input records, grouped by `(firm_id, normalized termination date)` with the number of records on that date. Later dates remain solely for isolation and control-exposure screening. No employee or new event identifier is stored.

**Same-day records.** Treat each supplied event row as one actual termination. Without employee identifiers, multiple rows at the same store and normalized date cannot be distinguished automatically from duplicate copies of one record. The working convention is that they are simultaneous terminations, including when their doses match. Preserve their count in `termination_history_`. If the first termination date has more than one record, mark its candidate `overlapping_termination` and exclude it; do not promote a later date or interpret the first row's dose as the combined reduction. Duplicate copies of the same underlying termination must be removed upstream if that is how the source data are recorded.

**Logging convention.** Every nonfatal warning in this specification must use the standard-library function `logging.warning()`, including duplicate-event summaries, failed comparison screens, missing controls, extrapolation, incomplete prediction tables, and restricted aggregations. Do not import or use the `warnings` module, warning categories, or `warnings.warn()`. Do not explicitly configure logging handlers or levels in the module; leave logging setup to the caller. Validation and required-model failures remain exceptions.

The outcome panel may have missing store-days, including temporary inactivity and permanent closure. Every row that is present must have a finite, nonmissing outcome. An absent row is not a zero outcome. Do not fill missing dates, impute outcomes, carry observations forward, or replace a required endpoint with the nearest observed date. A real observed zero is allowed. A store or event contributes to a comparison only when its exact baseline and endpoint rows exist. Missing intermediate dates alone do not invalidate that comparison.

Use the earliest and latest normalized outcome dates as the overall study range. An optional internal integer day index may be constructed as the number of calendar days since the earliest date plus one, with the same origin for outcome dates and termination dates. Set $T$ to the last such index. Preserve gaps: neither within-store row numbers nor ranks of observed dates are calendar-day indices. The formulas in this document use these internal day indices for convenience; users supply datetime columns. Event days and isolation lengths are integer numbers of calendar days.

A design-eligible target event has $1-L\le G_i\le T-H$; actual endpoint observation is checked separately for each comparison. Events outside the outcome range are allowed in the history table and still enter first-event selection. If a supplied earlier event lies outside the panel, do not replace it with a later in-panel event. Such extra records do not extend the isolation-screen bounds. Stores with no terminations need no event row; do not create zero-dose events. Validate nonmissing identifiers, event-store foreign keys, finite doses and outcomes, and all date and uniqueness requirements. Preserve the caller's identifier values without string coercion. Fitting must not alter either input DataFrame.

The inferred coverage interval runs from the earliest to the latest normalized outcome date over the entire panel. Assume the event table is complete for every outcome store throughout that interval, including stores with no recorded terminations and days when a store is closed or inactive. Completeness cannot be verified from the outcome panel or the presence of individual event records. Preserve all supplied terminations for first-event selection and history screening, including those during missing outcome periods, outside the panel, overlapping other events, or excluded from estimation. Extra event records do not relax the requirement that the full isolation interval lies inside the outcome panel bounds.

Runtime dependencies are NumPy, pandas, PyFixest, SciPy, and the Python standard library. Exact runtime pins and development-tool requirements are declared in [pyproject.toml](pyproject.toml). The project requires Python 3.11; [.python-version](.python-version) selects that interpreter for tools such as uv. The first version covers daily outcomes with missing store-days allowed, equal treated-event weights within each regression, equal control-store weights, no anticipation, and post-treatment level estimates. Covariates, inference, plotting, treatment slopes, and changing the dose within an event are outside this initial scope.

Use the supplied [.pre-commit-config.yaml](.pre-commit-config.yaml) for spelling, formatting, lint, and type checks on Python, Markdown, and notebooks. The hooks use system executables, so activate the development environment before installing or running them:

```bash
uv sync --python 3.11 --group dev
source .venv/bin/activate
pre-commit install
pre-commit run --all-files
```

**`ATT` owns independent cohort-time fits.** The class manages the two tables but never pools regression coefficients across cohorts. Its signatures are:

```python
ATT(
    *,
    firm_col,
    date_col,
    outcome_col,
    dose_col,
    degree=3,
    num_knots=0,
    min_cohort_size=None,
)

ATT.fit(
    outcomes,
    events,
    *,
    event_times,
    isolation_window,
) -> ATT

ATT.predict(doses, *, event_times=None) -> pandas.DataFrame
```

Construction requires all four column-name arguments, a positive integer `degree`, a nonnegative integer `num_knots`, and `min_cohort_size` either `None` or a positive integer. Require a two-element `isolation_window` of integer signed offsets with $L\le-1$ and $H\ge0$, and distinct nonnegative fitted event days no greater than $H$. The lower bound must include the baseline at event day $-1$; reject a positive lower offset rather than converting its sign implicitly. Dose requests must be distinct, finite, and positive. Event-time and dose requests must be nonempty.

`fit()` returns `self` and replaces any previously fitted state; a failed refit must not leave an earlier model available for prediction. `predict()` before successful fitting raises an error. Omitting `event_times` in prediction requests all fitted horizons; explicitly requesting an unfitted horizon raises an error rather than fitting additional models implicitly.

For a cohort $g$ and a horizon $e$, set $t=g+e$. Use one treated row per store with a design-eligible first termination at $g$ and $A_i(e)=1$. Assert uniqueness of the store identifier. The baseline is the preceding calendar day, even if that store was inactive then. Select controls using the complete termination history and exact endpoint availability:

$$
\mathcal C_{g,e}=\lbrace s:\quad
\text{no termination at store }s\text{ lies in }[g+L,g+H],\quad
O_{s,g-1}=O_{s,g+e}=1\rbrace.
$$

Count each control store once, regardless of workforce size. A treated store cannot also be its own control. A store may control for another event outside its own termination window, including after an earlier termination, only under the stated PT and carryover conditions. The history screen remains fixed through $g+H$ at all horizons; the observed-endpoint restriction can change both treated and control samples across horizons.

Compute $\Delta Y_s^{g,t}=Y_{s,t}-Y_{s,g-1}$. Let $b_{g,e}(d)$ denote the saved B-spline columns for this comparison, excluding the first full-basis column. The fitted design includes a separate intercept:

$$
x_{g,e}(d)=(1,b_{g,e}(d)')',\qquad
\widehat\gamma_{g,e}
=\arg\min_\gamma\sum_{i:G_i=g,S_i=1,A_i(e)=1}
\left(\Delta Y_i^{g,t}-x_{g,e}(D_i)'\gamma\right)^2.
$$

Store the control correction separately:

$$
\widehat m_{0,g,e}
=\frac{1}{|\mathcal C_{g,e}|}\sum_{s\in\mathcal C_{g,e}}\Delta Y_s^{g,t}.
$$

Prediction is

$$
\widehat{ATT}_{\mathcal E_e}(g,t,d\mid g,d)
=x_{g,e}(d)'\widehat\gamma_{g,e}-\widehat m_{0,g,e}.
$$

The intercept belongs to the raw treated-outcome regression. Subtract the stored control mean exactly once. Neither controls nor kernel weights enter the treated OLS fit. Use a numerically stable least-squares solver, not an explicit inverse of $X'X$.

**Fitted basis and knots.** Let $q$ be the degree and $K$ the number of internal knots. The regression has $p=q+K+1$ columns, including the intercept. Defaults $q=3,K=0$ give a cubic polynomial expressed in a B-spline basis.

For $K>0$, choose shared internal knots once using the positive doses from one row per design-eligible store, before endpoint-availability and small-sample screening. A store contributes once even if it is observed at several requested horizons. Use quantiles at $k/(K+1)$, $k=1,\ldots,K$, with linear interpolation matching R's default quantile convention. This retains the package's global knot-selection structure, now applied to the eligible event population. Keep the selected knots fixed after screening. [Setup source](https://github.com/bcallaway11/contdid/blob/main/R/setup_pte_cont.R), [quantile helper](https://github.com/bcallaway11/contdid/blob/main/R/helper_functions.R)

For each `(cohort, event_time)` comparison, use the minimum and maximum doses among its endpoint-observed treated stores as boundary knots. Build and store the resulting basis definition: degree, internal knots, repeated boundary knots, and retained-column convention. A SciPy `BSpline` representation can evaluate that saved basis. Construct the full basis, omit its first column, and add an explicit intercept, consistent with the R parameterization. [B-spline conventions](https://wwenjie.org/splines2/reference/bspline)

Reject duplicate global internal knots with a configuration error; do not silently change $K$. If a shared internal knot is not strictly inside a comparison's boundary knots, warn and drop that comparison with an `incompatible_knots` reason. Degenerate dose ranges and rank-deficient designs are handled by the comparison-specific screen below.

`predict()` must evaluate the fitted basis without recalculating knots or boundaries from the requested doses. In particular, a prediction at dose 8 must be identical whether requested alone or alongside other doses. Save a separate basis definition for each `(cohort, event_time)` because missing endpoints can change its treated sample and dose support. Do not assume that one cohort has the same basis at every horizon.

To stay close to the parametric routine, allow polynomial continuation outside a comparison's observed dose range, issue an extrapolation warning, and flag those rows. Do not clip doses, discard cohorts at particular doses, or renormalize weights because of extrapolation. Such values rely on the fitted functional form beyond observed support.

**Small-sample screening and dropping.** `min_cohort_size=None` resolves to the minimum sample size permitted by this design. Let $p=q+K+1$ be the number of regression columns, including the intercept. The design retains its requirement of positive residual degrees of freedom, so its sample-size floor is $p+1$. Full-column-rank OLS alone algebraically requires only $p$ observations; the extra observation is this module's design rule, not an identification threshold supplied by the article.

If the caller supplies a positive integer minimum, denote it by $n_{\mathrm{user}}$. Resolve the effective threshold as

$$
n_{\min}=
\begin{cases}
p+1,&\text{if the minimum is unspecified},\\
\max(n_{\mathrm{user}},p+1),&\text{otherwise}.
\end{cases}
$$

Thus `degree=3`, `num_knots=0`, and `min_cohort_size=None` give $p=4$ and an effective minimum of five stores. A user-specified minimum below five does not override the design floor; a larger minimum increases it. Resolve this threshold once from the configured basis dimension and store it in `min_cohort_size_`. Apply it separately to each `(cohort, event_time)` treated sample after endpoint selection. `None` does not disable screening, and meeting this minimum does not guarantee precision.

Count each observed treated store once; daily outcome rows and additional termination records do not increase the sample size. Require at least $p$ distinct doses and full column rank under a consistent documented numerical tolerance. The rank check is necessary even if the distinct-dose check passes.

For each nonempty treated comparison failing sample size, dose support, rank, or knot compatibility, warn with its cohort date, event day, store count, distinct-dose count, design dimension, and reason. Record the failure in `cell_diagnostics_`. Drop that cohort only at the affected horizon. This replaces the previous rule that dropped a cohort at every horizon: missing endpoints can make one comparison fail while another remains estimable. Do not reduce degree, remove knots, or pool cohorts to rescue a fit.

Keep every distinct termination date of every excluded store in `termination_history_`. That store can still be a control in other clean comparisons with observed endpoints. Exclusion must never erase a termination from history screening.

Distinguish design and availability exclusions from estimation failures. Mark an event dated outside the overall outcome range as `outside_outcome_period`, and an otherwise in-range event whose isolation interval crosses a panel boundary as `isolation_outside_panel`. For every design-eligible event, all requested endpoints lie within the overall panel range because $e\le H$. Store-specific rows may still be missing. If a comparison has no treated stores with both endpoints, mark `no_observed_treated`; it has no events in that horizon's observed-endpoint population. Record per-store `missing_baseline` and `missing_endpoint` reasons in the availability diagnostics. If a nonempty treated design passes screening but has zero endpoint-observed clean controls, warn and mark `no_controls`; do not fit that cell. `evaluate_theta()` must error for an affected requested horizon rather than drop the cell and renormalize silently. A singleton control group is computable for point estimation and its count is reported.

If no requested comparison can be fitted, `fit()` raises an error after recording reasons. Never fill a failed cell with a zero effect.

**Stored state and unpooled output.** Keep the information needed for prediction, weights, and diagnostics:

- `column_names_`: the four supplied input-column mappings; the `firm_col` and `date_col` mappings are shared by both tables.
- `min_cohort_size_`: the resolved effective sample-size minimum, including the design floor.
- `events_`: canonical columns `firm_id`, `cohort`, `dose`, and `design_eligible`, with one selected first-event row per store represented in the input events. `cohort` is a normalized `datetime64[ns]` date. Includes excluded events; these canonical names are internal/output conventions, not required names in the user's input.
- `termination_history_`: canonical `firm_id`, normalized `cohort` date, and `n_terminations`, unique by store and date; includes later dates excluded from target estimation.
- `duplicate_diagnostics_`: `n_dropped_events` and `n_affected_firms` from first-event selection.
- `event_diagnostics_`: design eligibility and reasons such as `outside_outcome_period`, `isolation_outside_panel`, and `overlapping_termination`, keyed by store; report conflicting dates and same-day multiplicity where relevant.
- `availability_diagnostics_`: endpoint availability and missing-date reasons keyed by store and event day, including the exact baseline and endpoint dates.
- `event_times_`, `origin_date_`, `last_date_`, `last_day_`, `isolation_window_`, and `knots_`. `origin_date_` and `last_date_` are the inferred screening bounds; no separate coverage interval is stored.
- `retained_cohorts_`: a mapping from event day to the cohort dates whose nonempty treated designs pass that horizon's screen. A missing control group does not remove a cohort from this set.
- `bases_` and `models_`: saved basis definitions and separate coefficients, control means, and sample counts keyed by `(cohort, event_time)`.
- `cohort_diagnostics_`: summary of design-eligible events and available, retained, or failed comparisons for each cohort.
- `cell_diagnostics_`: horizon status, treated-design checks, control counts, the exact contributing treated stores, and the unique control-store identifiers.

`ATT.predict()` returns one row per available `(cohort, event_time, dose)`, with `cohort`, `day`, `event_time`, `dose`, `att`, `n_treated`, `n_controls`, and `extrapolated`. Both `cohort` and endpoint `day` are normalized `datetime64[ns]` dates, not internal integer indices. Sort by cohort, event time, and dose. Warn when requested comparisons were dropped or have no controls, directing the caller to diagnostics. Ordinary outcome-availability exclusions are recorded separately; do not manufacture predictions for them.

**Separate kernel-weight function.** This function is independent of fitted outcomes and of `ATT` internals:

```python
estimate_pi(
    events,
    doses,
    *,
    eligible_firms,
    bandwidth,
) -> pandas.DataFrame
```

`events` uses canonical columns `firm_id`, normalized datetime `cohort`, and positive `dose`. It is independent of the fitted model and can be constructed directly; `ATT.fit()` maps custom input columns to this schema. Require one already-selected first-event row per store. This helper does not repeat preprocessing. `eligible_firms` is an explicit nonempty collection of unique store identifiers, all present in `events`. The caller supplies only stores with observed endpoints in retained comparisons at the intended horizon. Validate identifiers and filter to that exact set before normalization. A cohort-only argument is insufficient because isolation or missing endpoints can exclude individual stores within a cohort.

Let $\mathcal I_e$ be the supplied set of store identifiers and $\mathcal R_e$ the represented cohorts. For a common bandwidth $h>0$, use

$$
q_g(d;h)=\sum_{i\in\mathcal I_e:G_i=g}
\exp\left[-\frac12\left(\frac{D_i-d}{h}\right)^2\right],
\qquad
\widehat\pi_g(e,d)
=\frac{q_g(d;h)}{\sum_{k\in\mathcal R_e}q_k(d;h)}.
$$

Each supplied store contributes exactly once. Do not count daily rows, additional termination records, or workforce size, and do not average kernel contributions within each cohort. Summing store contributions preserves differences in cohort size. The Gaussian normalization constant and common $1/h$ cancel. All weights are nonnegative and sum to one at each requested dose. Compute normalization with log-sum-exp; invalid bandwidths or nonfinite intermediate results raise errors rather than triggering uniform weights.

Return `cohort`, `dose`, and `weight`, sorted by normalized cohort date and dose. No event-time argument is needed: supplying a different eligible-event set supplies the horizon dependence. The function can also inspect sensitivity to dose and bandwidth.

This is a custom estimator of conditional cohort probabilities. The article does not prescribe this kernel or bandwidth, and the package's existing dose aggregation does not supply these conditional weights. A positive bandwidth smooths cohort composition over nearby doses; consistency for exact conditional probabilities requires suitable smoothness, bandwidth, and sample-size conditions. Each store supplies only one dose/cohort observation to this calculation. Automatic bandwidth selection is outside the first version.

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

1. Select all design-eligible stores with both exact outcome endpoints observed, $A_i(e)=1$. Their represented cohorts form $\mathcal G_e$.
2. Retrieve $\mathcal R_e$ from `retained_cohorts_[e]`, then keep only those observed stores in those cohorts to form $\mathcal I_e$. If empty, raise an error.
3. Require a fitted cell for every cohort in $\mathcal R_e$. A missing control group is an error here, not an additional implicit selection rule.
4. Obtain unpooled predictions from `ATT.predict()` and weights from `estimate_pi(model.events_, ..., eligible_firms=I_e)`.
5. Join by exact cohort and dose labels. Require a complete one-to-one match and weights summing to one; do not let an inner join discard unmatched components.
6. Calculate

$$
\widehat\theta_{\mathcal E,\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
\widehat\pi_g(e,d)\thinspace{}
\widehat{ATT}_{\mathcal E_e}(g,g+e,d\mid g,d).
$$

The function only aggregates existing estimates. It neither refits the curves nor changes the window, endpoint samples, or screening. Changing bandwidth changes the weights, not the underlying ATT models.

Return `event_time`, `dose`, `theta`, `n_cohorts_eligible`, `n_cohorts_used`, `n_firms_design`, `n_firms_eligible`, `n_firms_used`, `availability_restricted`, `restricted_target`, and `extrapolated`. `n_firms_design` counts stores with design-eligible first terminations, including those without observed endpoints at this horizon. Eligible counts refer to the endpoint-observed population before comparison screening; used counts refer to retained comparisons. Store and selected-event counts are identical, so separate event-count columns are unnecessary. `availability_restricted` is true when some design-eligible stores lack a baseline or endpoint row at the requested horizon. Overall panel-boundary exclusions have already occurred in the design screen. `restricted_target` is true when any nonempty eligible cohort is excluded at this horizon by a sample-size, rank, dose-support, or knot screen. Isolation restrictions always define the target and are recorded separately. `extrapolated` is true if any contributing prediction is outside that comparison's observed dose range. Sort by event time and dose; component estimates and weights remain separately inspectable.

**Observation availability and screening change the target.** First-event selection limits candidates to one termination per store. The isolation design then restricts that population to $S_i=1$. Outcome availability then restricts it to $A_i(e)=1$ at each horizon. After comparison screening, weights estimate

$$
P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1,G_i\in\mathcal R_e).
$$

For a given retained set, the population counterpart is

$$
\theta_{\mathcal E,\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1,G_i\in\mathcal R_e)
ATT_{\mathcal E_e}(g,g+e,d\mid g,d).
$$

This equals the pre-screen observed-endpoint target only when every eligible cohort is retained or under additional special conditions. It does not generally equal the effect over all terminations or all stores before the isolation, availability, and comparison screens. Keep this interpretation despite the short output name `theta`. Warn when comparison screening restricts an aggregation, listing affected horizons. Availability exclusions are separately visible through counts, flags, and missing-date diagnostics. Never retain excluded stores in the kernel denominator, replace unknown effects with zero, or imply that renormalization recovers missing effects.

The screen is sample-dependent; this description is not an inferential result about selection. Both within-cohort store composition and retained cohorts may change across horizons. The fixed isolation window ensures calendar coverage through $H$ for all design-eligible events. No common set of stores with complete outcomes is imposed across horizons; inactivity or closure can still change the observed sample.

**Implementation boundaries and validation plan.** Put the class, both public functions, and a few private validation, window-selection, and basis helpers in one `att.py` file. Use `firm` consistently in all Python identifiers, canonical column names, docstrings, and error/log messages, including private helpers. This plan keeps the application term “store” in explanatory prose; code examples and public schema names follow the general `firm` terminology. Use `logging.warning()` for nonfatal diagnostics and small dictionaries or records for fitted cells. External R code is not a runtime dependency. No generic estimator framework, serialization layer, or extra classes are needed.

When the Python implementation is written, verify these behaviors with targeted examples:

- All four column-name arguments are required; omitting one fails construction. Renaming every input column through the constructor leaves results unchanged, using the same `firm_col` and `date_col` labels in both tables. Inputs missing either shared column fail validation; `outcome_col` and `dose_col` may share a label across the two tables.
- The signed window `(-7, 14)` screens exactly from $g-7$ through $g+14$, including both boundaries; `(7, 14)` is rejected. Check the panel-bound conditions $g+L\ge1$ and $g+H\le T$.
- `min_cohort_size=None` resolves to $p+1$ for the configured degree and knot count; a smaller explicit minimum cannot bypass that floor, while a larger one is enforced. The default cubic design resolves to five.
- Time-of-day differences do not affect estimates; same-store intraday duplicates in the outcome table are rejected after normalization, as are `NaT` and timezone-aware inputs. Missing calendar days are never compressed.
- Multiple termination rows for a store retain only its earliest normalized-date row and its dose. Same-day ties use original row order; an ineligible first event is not replaced. Check the aggregate log counts, including three dropped rows for two stores in the example, and no duplicate warning when each store has one event row.
- Later distinct dates dropped from targets still block conflicting controls or isolation windows. Multiple records on the first store-date retain their multiplicity in history and exclude the candidate as overlapping; no later date is promoted.
- Capture warning-level log records to verify every specified warning uses `logging.warning()`; no warnings-module emissions are produced.
- A missing baseline or endpoint excludes that store only from affected comparisons; missing intervening days alone do not. Present rows with missing or infinite outcomes fail validation. Temporary inactivity can cause a store to contribute at one horizon, disappear at another, and reappear later.
- A two-period comparison with a known polynomial dose effect and a nonzero control trend recovers the curve, subtracting the control mean once.
- Predictions at a fixed dose are invariant to other requested doses and input row order. For same-day ties, the selected candidate row follows input order, but the candidate is excluded as overlapping, so its dose cannot change fitted effects. Where R training and prediction bases coincide, unpooled estimates on the same constructed sample agree numerically; otherwise compare against an R reference that explicitly reuses the training basis.
- Two sufficiently separated termination dates at one store produce only one target event and one kernel contribution. Its later date remains in the history screen, and every control store contributes once per comparison.
- Simultaneous and overlapping events are excluded using the original complete history. An excluded event or a dropped cohort still disqualifies its store from controls in every conflicting window.
- Screening bounds equal the overall normalized outcome-date minimum and maximum, regardless of individual-store gaps or additional event dates. An isolation interval crossing either boundary excludes the event at every horizon; intervals touching the boundaries are allowed. Extra event records outside the panel cannot rescue an excluded event. Changing only prediction horizons leaves the fitted screen unchanged.
- Small, rank-deficient, and incompatible-knot comparisons warn and disappear from the corresponding horizon only, while retaining the complete history. Verify that an estimable short horizon survives a sparse later horizon, with separate fitted bases and matching store-specific kernel inputs.
- Weights count each selected store once, sum to one, reflect unequal cohort sizes, and match the explicit formula. A within-cohort excluded event cannot leak back into the weights.
- Aggregation equals a manual weighted sum, reports store counts, identifies the retained population, and rejects a horizon with a missing required cell.

Only the specification is delivered now. Future inference must cluster or resample by store, carrying its selected event, full termination history, and outcomes together, and account for shared controls, estimated weights, smoothing, and selection. Clustering handles dependence; it does not fix contaminated controls or justify the identifying assumption. Store-level independence and an adequate number of independent stores would still require assessment.

## References

Callaway, B., Goodman-Bacon, A. and Sant'Anna, P.H.C. (2025) *Difference-in-Differences with a Continuous Treatment*. Working paper, draft dated 31 December. Available at: [https://psantanna.com/files/CGBS_v4.pdf](https://psantanna.com/files/CGBS_v4.pdf) (Accessed: 3 November 2026).
