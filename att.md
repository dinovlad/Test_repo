**Implementation specification: unpooled continuous-dose ATT and dose-specific event-study aggregation.**

This document specifies a future self-contained Python module, `att.py`. It does not implement that module. The design uses independent cohort-time regressions following the parametric `contdid` calculation, retains each fitted spline basis for prediction, and estimates conditional cohort weights with a separate Gaussian-kernel function. It accommodates multiple isolated employee terminations at the same store. Event selection and control selection are extensions to the article's design; the regression calculation remains the reference calculation. All estimates in the first version are point estimates.

The public API consists of one class and two functions:

```python
model = ATT(
    degree=3,
    num_knots=0,
    min_cohort_size=10,
    outcome_store_col="store_id",
    outcome_date_col="date",
    outcome_col="sales",
    event_store_col="store_id",
    employee_col="employee_id",
    event_date_col="termination_date",
    dose_col="dose",
)
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
    eligible_pairs=eligible_pairs,
    bandwidth=bandwidth,
)

theta = evaluate_theta(
    model,
    doses=[4, 6, 8, 10, 12],
    bandwidth=bandwidth,
    event_times=[0, 7, 14],
)
```

The standalone weight example requires the precise retained `(store_id, employee_id)` pairs eligible at one event day. `evaluate_theta()` constructs that set automatically. The bandwidth is supplied in hours. The illustrated isolation window is a configuration example, not a recommended empirical choice; `history_start` and `history_end` declare the dates over which termination recording is complete for all outcome stores.

**Event identifiers and store outcomes.** Let $i=(s,j)$ identify employee $j$ at store $s$, with termination day $G_i$ and previous daily scheduled hours $D_i>0$. The pair `(store_id, employee_id)` is the sole estimation-unit key. The input may contain several events for that pair; keep its first event by normalized termination date as the only candidate for estimation. Later events do not become additional targets or enter the aggregation weights. The duplicate-event rule below logs how many rows were dropped for how many pairs. Several sufficiently separated events from the same store can enter the analysis. There is no single permanent focal employee or first termination selected for each store: each eligible termination is focal in its own comparison.

The event's treatment label and associated outcome are

$$
W_{it}=D_i\thinspace{}1\lbrace t\ge G_i\rbrace,
\qquad Y_{it}=Y_{s(i),t}.
$$

The dose is the fixed number of daily hours removed by that employee's termination, not the store's total staffing reduction. The event label is absorbing even though the store can experience other events. A replacement or redistribution of work can offset the staffing loss without changing this label. Such staffing responses belong to the total effect of the termination.

Store sales are shared outcomes. Copying the sales series across employees does not create independent observations or valid untreated coworkers. If employee A leaves a store, employee B at that same store cannot serve as a control merely because B has not left. Keep the outcomes and events in separate tables and construct comparison samples explicitly.

**Isolation and event eligibility.** Prespecify an isolation window $(L,H)$, where $L\ge1$ and $H\ge\max(e)$ across fitted event days. For an event at $g$, screen the closed calendar interval $[g-L,g+H]$. An event is design-eligible only if its termination and preceding calendar date lie within the overall outcome date range, it has complete termination-history coverage for that entire interval, and no other distinct termination at its store occurs inside it, including a later termination of the same employee. Simultaneous terminations therefore exclude every affected event. Apply this rule to each first-event candidate using the complete distinct-date termination history described below. Exclude only that candidate's exact `(store, employee, date)` from its own overlap check, not every termination of its pair. Repeated input rows for the same pair and date are one historical termination. Dropping rows from target estimation must not make another event artificially eligible.

Here, an overlapping event means another termination falls within the specified screening interval. Reuse of a store's sales observations in separate comparisons is still possible and creates dependence. Window lengths must address the plausible duration of anticipation and changing carryover effects; spacing alone does not establish identification.

Write $S_i=1$ for an event passing these design restrictions and $\mathcal E$ for the resulting event population, before endpoint-availability and small-sample screening. Keep $(L,H)$ fixed when comparing different event days or prediction requests. Refitting with a different window can change even the event-day-zero target. Do not infer complete history from the first and last recorded termination dates. If history ends at $T$, events whose screening interval extends beyond $T$ are ineligible even for shorter horizons. Complete termination records beyond the sales panel can avoid that exclusion without providing additional outcome observations.

**Statistical target and identifying restriction.** For each comparison let $e=t-g\ge0$. Define $O_{s,u}=1$ when the outcome table contains a valid observation for store $s$ on calendar date $u$, and zero otherwise. Event endpoint availability is

$$
A_i(e)=O_{s(i),G_i-1}\thinspace{}O_{s(i),G_i+e}.
$$

Both dates must be observed exactly. Missing intervening days do not prevent that particular long-difference comparison. Let $\mathcal E_e$ contain the design-eligible events with $A_i(e)=1$.

For an eligible event, let $Y^{\mathrm{term}}_{i,t}$ be store sales under the observed termination and $Y^{\mathrm{keep}}_{i,t}$ sales if that employee instead continues at their original daily hours through the comparison endpoint. The regimes share the actual history before the current event; subsequent staffing responses follow their respective regimes. The contrast does not remove every earlier termination from the store's history.

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

Dose and event day remain fixed; only calendar-day cohorts are averaged. These probabilities sample termination events, not stores with equal total weight. A store with more eligible, endpoint-observed terminations contributes more events. Throughout the API, `att` and `theta` refer to these quantities, subject to the additional comparison-specific cohort restriction below.

Let $\mathcal C_{g,e}$ be the clean control stores with observed endpoints, defined below. In addition to consistency, no anticipation of the focal termination, and no relevant spillovers between stores, the identifying parallel-trends restriction is

$$
E[Y^{\mathrm{keep}}_{i,t}-Y^{\mathrm{keep}}_{i,g-1}
\mid G_i=g,D_i=d,S_i=1,A_i(e)=1]
=E[Y_{s,t}-Y_{s,g-1}\mid s\in\mathcal C_{g,e}].
$$

The right-hand expectation weights control stores equally. This is an assumption for the selected events and their observed comparison stores, not a result implied by distance between terminations. Earlier terminations may leave persistent level effects that difference out; changing carryover effects must still be compatible with this restriction. The first version has no covariate or treatment-history adjustment to repair a failure of that assumption.

Selection on later terminations and on observed outcomes uses post-event information and may itself be affected by the focal event. Identification is asserted only under the stated restriction for the selected population. In particular, excluding stores after closure does not recover a treatment effect that includes their subsequent sales or closure consequences. Generalizing to all isolated events requires additional assumptions or a different outcome definition. The target is not a population guaranteed to remain isolated and observed under both counterfactual regimes. Parametric estimation also requires an adequate specification of each comparison's dose curve and relevant dose support. Strong parallel trends is not assumed, and slopes across doses are not interpreted as causal marginal responses.

**Differences from the article.** Callaway et. al. (2025) provide the starting point. The distinctions below are substantive design changes, not merely a relabeling of the unit.

| Feature | Article's staggered-adoption setting | This plan |
| --- | --- | --- |
| Treatment histories | One adoption date and a fixed subsequent dose per unit, Assumption 3-MP, printed p. 39 | One absorbing label per store–employee pair, but potentially several events affecting a store |
| Observation structure | Independent sampled units, Assumption 1-MP, p. 39 | Events within a store share outcomes and histories; independent event sampling is not assumed |
| Outcome availability | The stated panel contains outcomes in every modeled period | Store-day gaps are allowed; each comparison uses its own observed endpoints and therefore its own selected population |
| Counterfactual and PT | Outcomes without treatment; PT-MP and Theorem C.1, pp. 39–40 | No current focal termination, retaining prior history; a separate PT restriction for selected events and clean comparison stores with observed endpoints |
| Controls | Not-yet-treated or never-treated units, Theorem C.1, p. 40 | Stores without a termination in the specified window; sufficiently earlier terminations are allowed only under the stated identifying restriction |
| Aggregation population | Dose-specific event-study aggregation over treatment cohorts, p. 41 | The same structure applied to endpoint-observed eligible terminations, with a horizon-specific retained-cohort restriction |

The article's dose-specific event-study aggregation on printed page 41 is unnumbered; it is the source of the aggregation structure, not an equation proving this repeated-event extension. Remark C.1 on page 42 discusses estimating components and aggregating them. [Callaway et. al. (2025), Appendix C](https://psantanna.com/files/CGBS_v4.pdf#page=40)

**Relationship to the R implementation.** The reference calculation is the parametric level-effect part of `contdid::cont_did_acrt`: regress raw outcome changes on an intercept and dose B-splines among the focal treated cohort, then subtract the comparison group's mean change. Use baseline $g-1$ and reproduce that calculation for each constructed comparison. [Estimator and subset source](https://github.com/bcallaway11/contdid/blob/main/R/cont_did.R)

The event-based input tables, isolation screen, store-history-based control rule, and explicit handling of missing store-days are custom additions. Passing a duplicated store–employee panel directly to the package would not implement them. The earlier `control_group` choices are replaced by the single clean-window rule below; a coworker's own future termination date is never sufficient for control eligibility.

Other intentional implementation differences are retaining the fitted basis at prediction, warning and dropping small or unidentifiable cohort–event-day comparisons, and adding dose-conditional kernel aggregation. Thus compatibility refers to the regression calculation on the same comparison sample with a consistent fitted basis. The inspected R prediction path rebuilds its basis on the prediction grid. Sources were inspected on 2026-11-03; implementation validation should record the R version or source revision actually used.

**Input contract and configurable columns.** `ATT.fit()` accepts two pandas DataFrames. All seven input-column parameters are required keyword-only constructor arguments with no defaults. The names supplied in the introductory example are explicit caller choices. The outcome table uses:

| `ATT` parameter | Meaning |
| --- | --- |
| `outcome_store_col` | Store identifier |
| `outcome_date_col` | Outcome date; dtype `datetime64[ns]`, with time ignored |
| `outcome_col` | Finite numeric daily store sales; no missing values on present rows |

The event table contains the complete termination history within the declared history window for those stores:

| `ATT` parameter | Meaning |
| --- | --- |
| `event_store_col` | Store whose sales are affected |
| `employee_col` | Employee identifier; together with the store, identifies the estimation unit |
| `event_date_col` | Termination date; dtype `datetime64[ns]`, with time ignored |
| `dose_col` | Positive finite daily hours removed at termination |

The two store-column parameters allow different names in the two tables; their identifier values must still refer to the same stores. Each parameter must be a nonempty string naming an existing column. Require distinct mapped columns within each table and unique DataFrame column labels. The same name may be used in different tables. Extra columns are ignored. No new identifier column is required or generated for events: represent each key as a `(store_id, employee_id)` tuple or a two-level index, never by concatenating strings.

Validate both date columns as timezone-naive `datetime64[ns]` with no `NaT`. Do not silently parse strings, accept object-date columns, or convert timezone-aware datetimes. Normalize copies with `.dt.normalize()` so only the calendar date is used while retaining `datetime64[ns]`. Validate uniqueness of `(store, normalized date)` in the outcome table after normalization. Two intraday rows that collapse to the same store-day are an error, not an instruction to sum or select one. Repeated store–employee pairs in the event input are allowed and processed by the first-event rule below. Different employees terminated at the same store on the same normalized date are simultaneous events and fail the isolation screen.

**First-event selection and duplicate-event logging.** Validate required input values and normalize dates before selecting the first event. Group by the columns named by `event_store_col` and `employee_col`, sort by the normalized `event_date_col` using a stable sort, and keep that entire first row for each pair. If several rows share the earliest normalized date, retain the first in the original input row order; time of day is ignored even for ties. Retain that row's dose without averaging across duplicates.

Select the first event across the full supplied history, before isolation, outcome availability, or sample-size screening. If that event is subsequently ineligible, do not promote a later event of the same pair. The target is therefore restricted to eligible first events per store–employee pair within the supplied history, not necessarily each employee's first-ever termination outside that history.

Let `n_dropped_events` be the input row count minus the number of distinct store–employee pairs. Let `n_affected_units` be the number of those pairs with more than one input row. When `n_dropped_events > 0`, emit one aggregate message per `fit()` call:

```python
logging.warning(
    "Dropped %d duplicate event rows from the target table for %d "
    "store-employee units; kept the first event by date for each unit.",
    n_dropped_events,
    n_affected_units,
)
```

For example, three rows for one pair and two for another mean three dropped events for two affected units. Count both repeated same-day records and later dated records in this message. With no duplicates, do not emit this message. Record both counts in `duplicate_diagnostics_`.

Only the first-event rows populate `events_`, fitted targets, global knot selection, and kernel-weight inputs. Separately build `termination_history_` from all distinct `(store, employee, normalized termination date)` combinations in the original input. This table needs no dose or new event identifier. Later distinct dates remain solely to screen isolation and control exposure; repeated records on the same pair/date do not create false overlapping terminations. This preserves the existing requirement that target exclusions cannot erase treatment history.

**Logging convention.** Every nonfatal warning in this specification must use the standard-library function `logging.warning()`, including duplicate-event summaries, failed comparison screens, missing controls, extrapolation, incomplete prediction tables, and restricted aggregations. Do not import or use the `warnings` module, warning categories, or `warnings.warn()`. Do not explicitly configure logging handlers or levels in the module; leave logging setup to the caller. Validation and required-model failures remain exceptions.

The outcome panel may have missing store-days, including temporary inactivity and permanent closure. Every row that is present must have a finite, nonmissing outcome. An absent row is not a zero outcome. Do not fill missing dates, impute outcomes, carry observations forward, or replace a required endpoint with the nearest observed date. A real observed zero is allowed. A store or event contributes to a comparison only when its exact baseline and endpoint rows exist. Missing intermediate dates alone do not invalidate that comparison.

Use the earliest and latest normalized outcome dates as the overall study range. An optional internal integer day index may be constructed as the number of calendar days since the earliest date plus one, with the same origin for outcome dates, termination dates, and history bounds. Set $T$ to the last such index. Preserve gaps: neither within-store row numbers nor ranks of observed dates are calendar-day indices. The formulas in this document use these internal day indices for convenience; users supply datetime columns. Event days and isolation lengths are integer numbers of calendar days.

A design-eligible target event has $2\le G_i\le T$; actual endpoint observation is checked separately for each comparison. Events outside the outcome range are allowed in the history table. Stores with no terminations need no event row; do not create zero-dose events. Validate nonmissing identifiers, event-store foreign keys, finite doses and outcomes, and all date and uniqueness requirements. Preserve the caller's identifier values without string coercion. Fitting must not alter either input DataFrame.

`history_window=(a,b)` accepts date or timezone-naive pandas Timestamp bounds, normalized to dates. It declares complete termination recording for every outcome store from $a$ through $b$, including stores with no recorded events. The history window must contain the overall outcome date range, and every supplied termination must lie within it. Completeness cannot be verified from event records or inferred from observed sales dates. Preserve all terminations for screening, including those during missing outcome periods, outside the sales panel, overlapping other events, or excluded from estimation.

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
    outcome_store_col,
    outcome_date_col,
    outcome_col,
    event_store_col,
    employee_col,
    event_date_col,
    dose_col,
    degree=3,
    num_knots=0,
    min_cohort_size=10,
)

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

Construction requires all seven column-name arguments, a positive integer `degree`, a nonnegative integer `num_knots`, and a positive integer `min_cohort_size`. Require integer window bounds with $L\ge1$ and $H\ge0$, and distinct nonnegative fitted event days no greater than $H$. Dose requests must be distinct, finite, and positive. Event-time and dose requests must be nonempty.

`fit()` returns `self` and replaces any previously fitted state; a failed refit must not leave an earlier model available for prediction. `predict()` before successful fitting raises an error. Omitting `event_times` in prediction requests all fitted horizons; explicitly requesting an unfitted horizon raises an error rather than fitting additional models implicitly.

For a cohort $g$ and a horizon $e$, set $t=g+e$. Use one treated row per design-eligible event at $g$ with $A_i(e)=1$. These events necessarily occur at distinct stores under the isolation rule; assert that fact. The baseline is the preceding calendar day, even if that store was inactive then. Select controls using the complete termination history and exact endpoint availability:

$$
\mathcal C_{g,e}=\lbrace s:\quad
\text{no termination at store }s\text{ lies in }[g-L,g+H],\quad
O_{s,g-1}=O_{s,g+e}=1\rbrace.
$$

Count each control store once, regardless of workforce size. A treated store cannot also be its own control. A store may control for another event outside its own termination window, including after an earlier termination, only under the stated PT and carryover conditions. The history screen remains fixed through $g+H$ at all horizons; the observed-endpoint restriction can change both treated and control samples across horizons.

Compute $\Delta Y_s^{g,t}=Y_{s,t}-Y_{s,g-1}$. Let $b_{g,e}(d)$ denote the saved B-spline columns for this comparison, excluding the first full-basis column. The fitted design includes a separate intercept:

$$
x_{g,e}(d)=(1,b_{g,e}(d)')',\qquad
\widehat\gamma_{g,e}
=\arg\min_\gamma\sum_{i:G_i=g,S_i=1,A_i(e)=1}
\left(\Delta Y_{s(i)}^{g,t}-x_{g,e}(D_i)'\gamma\right)^2.
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

For $K>0$, choose shared internal knots once using the positive doses from one row per design-eligible store–employee pair, before endpoint-availability and small-sample screening. A pair contributes once even if it is observed at several requested horizons. Use quantiles at $k/(K+1)$, $k=1,\ldots,K$, with linear interpolation matching R's default quantile convention. This retains the package's global knot-selection structure, now applied to the eligible event population. Keep the selected knots fixed after screening. [Setup source](https://github.com/bcallaway11/contdid/blob/main/R/setup_pte_cont.R), [quantile helper](https://github.com/bcallaway11/contdid/blob/main/R/helper_functions.R)

For each `(cohort, event_time)` comparison, use the minimum and maximum doses among its endpoint-observed treated pairs as boundary knots. Build and store the resulting basis definition: degree, internal knots, repeated boundary knots, and retained-column convention. A SciPy `BSpline` representation can evaluate that saved basis. Construct the full basis, omit its first column, and add an explicit intercept, consistent with the R parameterization. [B-spline conventions](https://wwenjie.org/splines2/reference/bspline)

Reject duplicate global internal knots with a configuration error; do not silently change $K$. If a shared internal knot is not strictly inside a comparison's boundary knots, warn and drop that comparison with an `incompatible_knots` reason. Degenerate dose ranges and rank-deficient designs are handled by the comparison-specific screen below.

`predict()` must evaluate the fitted basis without recalculating knots or boundaries from the requested doses. In particular, a prediction at dose 8 must be identical whether requested alone or alongside other doses. Save a separate basis definition for each `(cohort, event_time)` because missing endpoints can change its treated sample and dose support. Do not assume that one cohort has the same basis at every horizon.

To stay close to the parametric routine, allow polynomial continuation outside a comparison's observed dose range, issue an extrapolation warning, and flag those rows. Do not clip doses, discard cohorts at particular doses, or renormalize weights because of extrapolation. Such values rely on the fitted functional form beyond observed support.

**Small-sample screening and dropping.** The configurable default `min_cohort_size=10` now means ten distinct treated stores in a specific `(cohort, event_time)` comparison after endpoint selection. This is an implementation choice, not an article threshold or a guarantee of precision. With $n_{\mathrm{user}}$ denoting that parameter and $p$ the design dimension, require

$$
n_{\min}=\max(n_{\mathrm{user}},p+1).
$$

Count each observed treated pair once; the isolation rule makes this the distinct-store count within the comparison. Require at least $p$ distinct doses and full column rank under a consistent documented numerical tolerance. The rank check is necessary even if the distinct-dose check passes.

For each nonempty treated comparison failing sample size, dose support, rank, or knot compatibility, warn with its cohort date, event day, store count, distinct-dose count, design dimension, and reason. Record the failure in `cell_diagnostics_`. Drop that cohort only at the affected horizon. This replaces the previous rule that dropped a cohort at every horizon: missing endpoints can make one comparison fail while another remains estimable. Do not reduce degree, remove knots, or pool cohorts to rescue a fit.

Keep every distinct termination date of every excluded pair in `termination_history_`. Its store can still be a control in other clean comparisons with observed endpoints. Exclusion must never erase a termination from history screening.

Distinguish availability exclusions from estimation failures. An endpoint outside the overall outcome range is `outside_outcome_period`. If an in-range comparison has no treated pairs with both endpoints, mark `no_observed_treated`; it has no events in that horizon's observed-endpoint population. Record per-pair `missing_baseline` and `missing_endpoint` reasons in the availability diagnostics. If a nonempty treated design passes screening but has zero endpoint-observed clean controls, warn and mark `no_controls`; do not fit that cell. `evaluate_theta()` must error for an affected requested horizon rather than drop the cell and renormalize silently. A singleton control group is computable for point estimation and its count is reported.

If no requested comparison can be fitted, `fit()` raises an error after recording reasons. Never fill a failed cell with a zero effect.

**Stored state and unpooled output.** Keep the information needed for prediction, weights, and diagnostics:

- `column_names_`: the seven supplied input-column mappings.
- `events_`: canonical columns `store_id`, `employee_id`, `cohort`, `dose`, and `design_eligible`, with the selected first-event row per original pair. `cohort` is a normalized `datetime64[ns]` date. Includes excluded events; these canonical names are internal/output conventions, not required names in the user's input.
- `termination_history_`: canonical `store_id`, `employee_id`, and normalized `cohort` date, unique by those three columns; includes later distinct dates dropped from estimation.
- `duplicate_diagnostics_`: `n_dropped_events` and `n_affected_units` from first-event selection.
- `event_diagnostics_`: design eligibility and reasons such as `outside_outcome_period`, `incomplete_history`, and `overlapping_termination`, keyed by the pair; report conflicting pairs and dates where relevant.
- `availability_diagnostics_`: endpoint availability and missing-date reasons keyed by pair and event day, including the exact baseline and endpoint dates.
- `event_times_`, `origin_date_`, `last_date_`, `last_day_`, `isolation_window_`, `history_window_`, and `knots_`.
- `retained_cohorts_`: a mapping from event day to the cohort dates whose nonempty treated designs pass that horizon's screen. A missing control group does not remove a cohort from this set.
- `bases_` and `models_`: saved basis definitions and separate coefficients, control means, and sample counts keyed by `(cohort, event_time)`.
- `cohort_diagnostics_`: summary of design-eligible events and available, retained, or failed comparisons for each cohort.
- `cell_diagnostics_`: horizon status, treated-design checks, control counts, the exact contributing treated pairs, and the unique control-store identifiers.

`ATT.predict()` returns one row per available `(cohort, event_time, dose)`, with `cohort`, `day`, `event_time`, `dose`, `att`, `n_treated`, `n_controls`, and `extrapolated`. Both `cohort` and endpoint `day` are normalized `datetime64[ns]` dates, not internal integer indices. Sort by cohort, event time, and dose. Warn when requested comparisons were dropped or have no controls, directing the caller to diagnostics. Ordinary outcome-availability exclusions are recorded separately; do not manufacture predictions for them.

**Separate kernel-weight function.** This function is independent of fitted outcomes and of `ATT` internals:

```python
estimate_pi(
    events,
    doses,
    *,
    eligible_pairs,
    bandwidth,
) -> pandas.DataFrame
```

`events` uses the canonical metadata schema above: `store_id`, `employee_id`, normalized datetime `cohort`, and positive `dose`. It is independent of the fitted model and can be constructed directly; custom input columns are mapped to this schema by `ATT.fit()`. Require the already-selected first-event metadata with unique store–employee pairs; this helper does not repeat `fit()` preprocessing. `eligible_pairs` is an explicit nonempty collection of unique two-element tuples, all present in `events`. Repeated stores with different employees are permitted. The caller supplies only pairs with observed endpoints in retained comparisons at the intended horizon; the function does not decide availability, isolation, or fit regressions. Validate pairs and filter to that exact set before normalization. A cohort-only argument is insufficient because the isolation screen can exclude individual events within a retained cohort.

Let $\mathcal I_e$ be the supplied set of store–employee pairs and $\mathcal R_e$ the represented cohorts. For a common bandwidth $h>0$, use

$$
q_g(d;h)=\sum_{i\in\mathcal I_e:G_i=g}
\exp\left[-\frac12\left(\frac{D_i-d}{h}\right)^2\right],
\qquad
\widehat\pi_g(e,d)
=\frac{q_g(d;h)}{\sum_{k\in\mathcal R_e}q_k(d;h)}.
$$

Each supplied pair contributes exactly once. Do not count daily rows, duplicate sales by workforce size, deduplicate stores across different events, or average kernel contributions within each cohort. Summing event contributions preserves cohort sizes and the event-weighted target. The Gaussian normalization constant and common $1/h$ cancel. All weights are nonnegative and sum to one at each requested dose. Compute normalization with log-sum-exp; invalid bandwidths or nonfinite intermediate results raise errors rather than triggering uniform weights.

Return `cohort`, `dose`, and `weight`, sorted by normalized cohort date and dose. No event-time argument is needed: supplying a different eligible-event set supplies the horizon dependence. The function can also inspect sensitivity to dose and bandwidth.

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

1. Select all design-eligible pairs with both exact outcome endpoints observed, $A_i(e)=1$. Their represented cohorts form $\mathcal G_e$.
2. Retrieve $\mathcal R_e$ from `retained_cohorts_[e]`, then keep only those observed pairs in those cohorts to form $\mathcal I_e$. If empty, raise an error.
3. Require a fitted cell for every cohort in $\mathcal R_e$. A missing control group is an error here, not an additional implicit selection rule.
4. Obtain unpooled predictions from `ATT.predict()` and weights from `estimate_pi(model.events_, ..., eligible_pairs=I_e)`.
5. Join by exact cohort and dose labels. Require a complete one-to-one match and weights summing to one; do not let an inner join discard unmatched components.
6. Calculate

$$
\widehat\theta_{\mathcal E,\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
\widehat\pi_g(e,d)\thinspace{}
\widehat{ATT}_{\mathcal E_e}(g,g+e,d\mid g,d).
$$

The function only aggregates existing estimates. It neither refits the curves nor changes the window, endpoint samples, or screening. Changing bandwidth changes the weights, not the underlying ATT models.

Return `event_time`, `dose`, `theta`, `n_cohorts_eligible`, `n_cohorts_used`, `n_events_design`, `n_events_eligible`, `n_events_used`, `n_stores_eligible`, `n_stores_used`, `availability_restricted`, `restricted_target`, and `extrapolated`. `n_events_design` counts all design-eligible pairs, including pairs without observed endpoints at this horizon. Eligible counts refer to the endpoint-observed population before comparison screening; used counts refer to retained comparisons. Store counts deduplicate across pairs for reporting only. `availability_restricted` is true when some design-eligible pairs lack either endpoint, including endpoints beyond the study range. `restricted_target` is true when any nonempty eligible cohort is excluded at this horizon by a sample-size, rank, dose-support, or knot screen. Isolation restrictions always define the target and are recorded separately. `extrapolated` is true if any contributing prediction is outside that comparison's observed dose range. Sort by event time and dose; component estimates and weights remain separately inspectable.

**Observation availability and screening change the target.** First-event selection first limits candidates to one event per store–employee pair. The isolation design then restricts that population to $S_i=1$. Outcome availability then restricts it to $A_i(e)=1$ at each horizon. After comparison screening, weights estimate

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

This equals the pre-screen observed-endpoint target only when every eligible cohort is retained or under additional special conditions. It does not generally equal the effect over all isolated events, all terminations, or an equal-store average. Keep this interpretation despite the short output name `theta`. Warn when comparison screening restricts an aggregation, listing affected horizons. Availability exclusions are separately visible through counts, flags, and missing-date diagnostics. Never retain excluded pairs in the kernel denominator, replace unknown effects with zero, or imply that renormalization recovers missing effects.

The screen is sample-dependent; this description is not an inferential result about selection. Both within-cohort event composition and retained cohorts may change across horizons. No common balanced sample is imposed. A fixed isolation window does not prevent horizon-specific loss of outcomes from inactivity or closure.

**Implementation boundaries and validation plan.** Put the class, both public functions, and a few private validation, window-selection, and basis helpers in one `att.py` file. Use `logging.warning()` for nonfatal diagnostics and small dictionaries or records for fitted cells. External R code is not a runtime dependency. No generic estimator framework, serialization layer, or extra classes are needed.

When the Python implementation is written, verify these behaviors with targeted examples:

- All seven column-name arguments are required; omitting one fails construction. Renaming every input column through the constructor leaves results unchanged, including different store-column names across tables.
- Time-of-day differences do not affect estimates; same-store intraday duplicates in the outcome table are rejected after normalization, as are `NaT` and timezone-aware inputs. Missing calendar days are never compressed.
- Repeated pairs retain only their earliest normalized-date row and its dose. Same-day ties use original row order; an ineligible first event is not replaced. Check the aggregate log counts, including three dropped rows for two units in the example, and no duplicate warning for unique pairs.
- Later distinct dates dropped from targets still block conflicting controls or isolation windows. Repeated rows for the same pair/date create only one historical termination and do not create false overlaps.
- Capture warning-level log records to verify every specified warning uses `logging.warning()`; no warnings-module emissions are produced.
- A missing baseline or endpoint excludes that pair only from affected comparisons; missing intervening days alone do not. Present rows with missing or infinite outcomes fail validation. Temporary inactivity can cause a pair to contribute at one horizon, disappear at another, and reappear later.
- A two-period comparison with a known polynomial dose effect and a nonzero control trend recovers the curve, subtracting the control mean once.
- Predictions at a fixed dose are invariant to other requested doses and to input row order when there are no conflicting ties at a pair's earliest normalized date. For conflicting same-day ties, the documented first-row selection intentionally depends on input order. Where R training and prediction bases coincide, unpooled estimates on the same constructed sample agree numerically; otherwise compare against an R reference that explicitly reuses the training basis.
- Two sufficiently separated first events for different employees at one store both enter eligible cohorts and kernel weights. Coworkers never become extra control rows, and every control store contributes once per comparison.
- Simultaneous and overlapping events are excluded using the original complete history. An excluded event or a dropped cohort still disqualifies its store from controls in every conflicting window.
- Incomplete screening coverage excludes an event without silently assuming no future termination. Complete extended history permits late events to contribute at observed shorter horizons. Changing only the prediction horizons leaves the fitted isolation screen unchanged.
- Small, rank-deficient, and incompatible-knot comparisons warn and disappear from the corresponding horizon only, while retaining the complete history. Verify that an estimable short horizon survives a sparse later horizon, with separate fitted bases and matching pair-specific kernel inputs.
- Weights count each selected event once, sum to one, reflect unequal cohort sizes, and match the explicit formula. A within-cohort excluded event cannot leak back into the weights.
- Aggregation equals a manual weighted sum, reports distinct event and store counts, identifies the retained population, and rejects a horizon with a missing required cell.

Only the specification is delivered now. Future inference must cluster or resample by store, carrying all its events and outcomes together, and account for shared controls, estimated weights, smoothing, and selection. Clustering handles dependence; it does not fix contaminated controls or justify the identifying assumption. Store-level independence and an adequate number of independent stores would still require assessment.

## References

Callaway, B., Goodman-Bacon, A. and Sant'Anna, P.H.C. (2025) *Difference-in-Differences with a Continuous Treatment*. Working paper, draft dated 31 December. Available at: [https://psantanna.com/files/CGBS_v4.pdf](https://psantanna.com/files/CGBS_v4.pdf) (Accessed: 3 November 2026).
