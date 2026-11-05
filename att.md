**Implementation specification: unpooled continuous-dose ATT and dose-specific event-study aggregation.**

This document specifies the self-contained Python module [att.py](src/labor_cost_optimization/att.py). The design uses one fully interacted PyFixest OLS model on stacked cohort-time comparisons, with separate coefficients and a saved spline basis for every comparison. Its point estimates reproduce the parametric `contdid` calculation on the same samples and bases, and a separate Gaussian-kernel function estimates conditional cohort weights. It estimates the effect of each store's first recorded termination, with at most one target event per store. Event selection and control selection are extensions to the article's design; the stacked regression is an algebraically equivalent parameterization of the reference point-estimation calculation. The module evaluates every integer horizon in the isolation window: post-treatment effects, pre-treatment placebo contrasts, and the reference period. All estimates are point estimates.

**Periods and frequency.** The frequency is hardcoded to one calendar day. A period means one step at that fixed frequency. There is no frequency parameter or support for monthly or annual inputs. Input and output period labels are timezone-naive `datetime64[ns]` values normalized to the start of each period. All event-time offsets, isolation bounds, baseline comparisons, and dose units have a daily interpretation.

The public API consists of one class and two functions. In this example, `firm_col`, `period_col`, `outcome_col`, and `dose_col` are caller-supplied strings naming columns in the input tables; all four must be provided:

```python
model = ATT(
    firm_col=firm_col,
    period_col=period_col,
    outcome_col=outcome_col,
    dose_col=dose_col,
    degree=3,
    num_knots=0,
    min_cohort_size=None,
)
model.fit(
    outcomes,
    events,
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
    event_times=[-7, -2, -1, 0, 7, 14],
)
```

For `isolation_window=(-7, 14)`, `fit()` processes all 22 integer horizons from -7 through 14. It takes no `event_times` argument. Prediction and aggregation may select subsets of that range.

The standalone weight example requires the precise retained store identifiers eligible at one event period. `evaluate_theta()` constructs that set automatically. The bandwidth is supplied in hours. The illustrated isolation window is a configuration example, not a recommended empirical choice. Screening bounds are inferred from the overall minimum and maximum normalized outcome periods across all stores.

**Store units and first terminations.** Store $i$ is the unit. Let $G_i$ be its earliest recorded termination period in the supplied history, and let $D_i>0$ be the scheduled hours per period removed by that termination. The store identifier is the sole estimation-unit key. Employee identifiers are neither required nor used. Keep only this first termination as the store's estimation candidate; later terminations never become additional targets or enter its aggregation weights.

The fixed treatment label is

$$
W_{it}=D_i\thinspace{}1\lbrace t\ge G_i\rbrace.
$$

The outcome $Y_{it}$ is store sales per period. The dose is the hours removed by the selected termination, not the store's total subsequent staffing reduction. Replacement hiring or redistribution of work does not change this label; those responses belong to the total effect of the selected termination. Keep store-period outcomes and termination records in separate tables, with at most one treated observation per store in each comparison.

**Isolation and event eligibility.** Prespecify an isolation window $(L,H)$, where $L\le-1$ and $H\ge0$. These bounds define all fitted event periods, $e\in\lbrace L,L+1,\ldots,H\rbrace$. The tuple contains signed offsets in periods: `(-7, 14)` means seven periods before through fourteen periods after termination, with both boundaries included. For an event at $g$, screen the closed period interval $[g+L,g+H]$. An event is design-eligible only if its entire isolation interval lies within the overall outcome panel bounds and no other termination at its store occurs inside it, including any later termination that is excluded from target estimation. In internal period indices, require $g+L\ge1$ and $g+H\le T$, with equality allowed at either boundary. Assume termination recording is complete for every outcome store throughout these panel bounds, including periods with no outcome row. Simultaneous terminations therefore exclude every affected event. Apply this rule to each first-event candidate using the complete distinct-period termination history described below. Exclude only the one selected termination from its store's overlap count, not all records at that store or in that period. The same-period rule below specifies how multiple input records are counted. Dropping rows from target estimation must not make another event artificially eligible.

Here, an overlapping event means another termination falls within the specified screening interval. Reuse of a store's sales observations in separate comparisons is still possible and creates dependence. Window lengths must address the plausible duration of anticipation and changing carryover effects; spacing alone does not establish identification.

Write $S_i=1$ for an event passing these design restrictions and $\mathcal E$ for the resulting event population, before endpoint-availability and small-sample screening. Keep $(L,H)$ fixed when comparing different event periods or prediction requests. Refitting with a different window can change even the event-period-zero target. Do not derive screening bounds from termination periods or from an individual store's first and last observed sales periods. An event whose isolation interval extends beyond the overall panel bounds is ineligible at every requested horizon, including shorter horizons with observed endpoints. Additional termination records outside the panel do not extend these bounds. Completeness of termination recording within the bounds is a data assumption, not something inferred from sales observations.

**Statistical target and identifying restriction.** For each comparison let $e=t-g$ be any integer from $L$ through $H$. Define $O_{s,u}=1$ when the outcome table contains a valid observation for store $s$ in period $u$, and zero otherwise. Event endpoint availability is

$$
A_i(e)=O_{i,G_i-1}\thinspace{}O_{i,G_i+e}.
$$

Both periods must be observed exactly. Missing intervening periods do not prevent that particular long-difference comparison. Let $\mathcal E_e$ contain the design-eligible events with $A_i(e)=1$.

For an eligible event, let $`Y^{\mathrm{term}}_{i,t}`$ be store sales under the observed termination and $`Y^{\mathrm{keep}}_{i,t}`$ sales if that employee instead continues at their original scheduled hours per period through the comparison endpoint. The regimes share the actual history before the current event; subsequent staffing responses follow their respective regimes. The contrast does not remove every earlier termination from the store's history.

For post-treatment horizons $e\ge0$, the event-specific analogue of the article's unpooled target, restricted to observed endpoints, is

$$
ATT_{\mathcal E_e}(g,t,d\mid g,d)
=E[Y^{\mathrm{term}}_{i,t}-Y^{\mathrm{keep}}_{i,t}
\mid G_i=g,D_i=d,S_i=1,A_i(e)=1].
$$

Define $\mathcal G_e$ as the cohorts represented among $\mathcal E_e$. For $e\ge0$, the requested aggregation is

$$
\theta_{\mathcal E}(d,e)=\sum_{g\in\mathcal G_e}\pi_g(e,d)
ATT_{\mathcal E_e}(g,g+e,d\mid g,d),
\qquad
\pi_g(e,d)=P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1).
$$

Dose and event period remain fixed; only treatment-period cohorts are averaged. Each eligible store contributes its first termination exactly once when estimating these cohort probabilities. A store with additional termination records receives no extra weight. Kernel contributions depend on distance from dose $d$, so weights need not be equal across stores at a particular requested dose. For nonnegative horizons, the API fields `att` and `theta` refer to these quantities, subject to the additional comparison-specific cohort restriction below. Negative horizons use the diagnostic interpretation defined next.

Let $\mathcal C_{g,e}$ be the clean control stores with observed endpoints, defined below. In addition to consistency, no anticipation of the focal termination, and no relevant spillovers between stores, the identifying parallel-trends restriction is

$$
E[Y^{\mathrm{keep}}_{i,t}-Y^{\mathrm{keep}}_{i,g-1}
\mid G_i=g,D_i=d,S_i=1,A_i(e)=1]
=E[Y_{s,t}-Y_{s,g-1}\mid s\in\mathcal C_{g,e}].
$$

The right-hand expectation weights control stores equally. This is an assumption for the selected events and their observed comparison stores, not a result implied by distance between terminations. Earlier terminations may leave persistent level effects that difference out; changing carryover effects must still be compatible with this restriction. The module has no covariate or treatment-history adjustment to repair a failure of that assumption.

Selection on later terminations and on observed outcomes uses post-event information and may itself be affected by the focal event. Identification is asserted only under the stated restriction for the selected population. In particular, excluding stores after closure does not recover a treatment effect that includes their subsequent sales or closure consequences. Generalizing to all isolated events requires additional assumptions or a different outcome definition. The target is not a population guaranteed to remain isolated and observed under both counterfactual regimes. Parametric estimation also requires an adequate specification of each comparison's dose curve and relevant dose support. Strong parallel trends is not assumed, and slopes across doses are not interpreted as causal marginal responses.

**Negative horizons and the reference period.** Use the same baseline $g-1$ for every horizon. For $e\le-2$, estimate the dose-specific pre-treatment contrast

$$
P_{g,e}(d)
=E[Y_{i,g+e}-Y_{i,g-1}\mid G_i=g,D_i=d,S_i=1,A_i(e)=1]
-E[Y_{s,g+e}-Y_{s,g-1}\mid s\in\mathcal C_{g,e}].
$$

At every horizon, subtract sales in period $g-1$ from sales in period $g+e$. Thus $e=-7$ uses $Y_{g-7}-Y_{g-1}$. Here, the treated sample means membership in the future termination cohort; those stores have not yet experienced the focal termination.

These pre-treatment contrasts are placebo diagnostics. Under no anticipation and the pre-period counterpart of the selected-sample PT restriction above, their population values are zero. Post-treatment PT alone does not establish that implication. Estimate them freely rather than forcing them to zero; finite-sample nonzero values are not themselves causal effects of a termination that has not occurred. Formal testing and inference are outside the scope of this module.

At $e=-1$, both periods are $g-1$, so the outcome changes and their contrast are exactly zero. This is the event study's reference normalization, not an independently estimated pre-treatment effect. It must not be interpreted as evidence for PT.

Define a common reported component for all horizons:

$$
\tau_{g,e}(d)=
\begin{cases}
ATT_{\mathcal E_e}(g,g+e,d\mid g,d),&e\ge0,\\
P_{g,e}(d),&e\le-2,\\
0,&e=-1.
\end{cases}
$$

For every horizon, aggregate these components using the same dose-conditional weights and the appropriate observed-endpoint population:

$$
\theta_{\mathcal E}(d,e)
=\sum_{g\in\mathcal G_e}\pi_g(e,d)\thinspace{}\tau_{g,e}(d).
$$

The reported `estimate_type` is `att` for $e\ge0$, `placebo` for $e\le-2$, and `reference` for $e=-1$. For negative horizons, the numeric columns `att` and `theta` contain these diagnostics or the zero reference, not post-treatment causal ATT estimates. Provided the reference population passes the bookkeeping checks below, its aggregation is exactly zero at every requested dose.

**Relationship to the article.** Callaway et. al. (2025) provide the starting point. With stores as units and one selected termination per store, the treatment-label structure matches one adoption period and fixed dose per unit. The isolation screen, observed-outcome selection, and comparison-store rule require the distinctions below.

| Feature | Article's staggered-adoption setting | This plan |
| --- | --- | --- |
| Treatment histories | One adoption period and a fixed subsequent dose per unit, Assumption 3-MP, printed p. 39 | One first-termination label and dose per store; later terminations are used only to screen overlap and control eligibility |
| Observation structure | Independent sampled units, Assumption 1-MP, p. 39 | One outcome trajectory per store; independence across stores remains an assumption, and a store may be reused across comparisons |
| Outcome availability | The stated panel contains outcomes in every modeled period | Store-period gaps are allowed; each comparison uses its own observed endpoints and therefore its own selected population |
| Counterfactual and PT | Outcomes without treatment; PT-MP and Theorem C.1, pp. 39–40 | No current focal termination, retaining prior history; a separate PT restriction for selected events and clean comparison stores with observed endpoints |
| Controls | Not-yet-treated or never-treated units, Theorem C.1, p. 40 | Stores without a termination in the specified window; sufficiently earlier terminations are allowed only under the stated identifying restriction |
| Aggregation population | Dose-specific event-study aggregation over treatment cohorts, p. 41 | The same structure applied to stores with eligible first terminations and observed endpoints, with a horizon-specific retained-cohort restriction |
| Pre-treatment output | The cited post-treatment ATT aggregation has a causal level-effect interpretation | Negative horizons are fixed-baseline placebo contrasts, with event period -1 normalized to zero |

The article's dose-specific event-study aggregation on printed page 41 is unnumbered; it is the source of the aggregation structure, not an equation establishing identification for these selected store comparisons. Remark C.1 on page 42 discusses estimating components and aggregating them. [Callaway et. al. (2025), Appendix C](https://psantanna.com/files/CGBS_v4.pdf#page=40)

**Relationship to the R implementation.** The reference calculation is the parametric level-effect part of `contdid::cont_did_acrt`: regress raw outcome changes on an intercept and dose B-splines among the focal treated cohort, then subtract the comparison group's mean change. Use baseline $g-1$ and reproduce those point estimates through one stacked PyFixest regression, with a cell fixed effect and cell-specific treatment-basis coefficients. The equivalence below holds for the same samples and fitted bases without additional covariates. Extending this fixed baseline to pre-treatment horizons is an explicit convention of this plan; numerical comparisons with the package must use the same pre-treatment baseline convention rather than assuming its defaults match. [Estimator and subset source](https://github.com/bcallaway11/contdid/blob/main/R/cont_did.R)

The separate input tables, first-event selection, isolation screen, and explicit handling of missing store-periods are custom additions. The clean-window control rule can include stores whose first termination occurred earlier, provided their history and counterfactual trends satisfy the stated assumption. Keeping one target event per store does not make such stores untreated in the article's absorbing-treatment sense; this control rule remains an extension to its comparison design.

Other intentional implementation differences are retaining the fitted basis at prediction, warning and dropping small or unidentifiable cohort–event-period comparisons, and adding dose-conditional kernel aggregation. Thus compatibility refers to the regression calculation on the same comparison sample with a consistent fitted basis. The inspected R prediction path rebuilds its basis on the prediction grid. Sources were inspected on 2026-11-03; implementation validation should record the R version or source revision actually used.

**Input contract and configurable columns.** `ATT.fit()` accepts two pandas DataFrames. All four input-column parameters are required keyword-only constructor arguments with no defaults. Omitting any one raises `TypeError` at construction. No input-column names are inferred or supplied as fallbacks. The outcome table uses:

| `ATT` parameter | Meaning |
| --- | --- |
| `firm_col` | Store identifier |
| `period_col` | Outcome period; dtype `datetime64[ns]`, with time ignored |
| `outcome_col` | Finite numeric store sales per period; no missing values on present rows |

The event table must record all terminations within the overall outcome panel period range for those stores:

| `ATT` parameter | Meaning |
| --- | --- |
| `firm_col` | Store whose sales are affected |
| `period_col` | Termination period; dtype `datetime64[ns]`, with time ignored |
| `dose_col` | Positive finite scheduled hours per period removed at termination |

The shared `firm_col` and `period_col` parameters name columns present in both input tables. Their labels must match across the tables; `period_col` identifies outcome periods in the outcome table and termination periods in the event table. If the source tables use different labels, rename them before fitting. Store identifier values must refer to the same stores across tables. Each parameter must be a nonempty string naming an existing column. Require distinct mapped columns within each table and unique DataFrame column labels. `outcome_col` and `dose_col` may have the same name because they belong to different tables. Extra columns are ignored. Use the supplied store identifier directly as the estimation key. No employee, spell, or event identifier column is required or generated.

Validate both period columns as timezone-naive `datetime64[ns]` with no `NaT`. Do not silently parse strings, accept columns with object dtype, or convert timezone-aware datetimes. Normalize copies with `.dt.normalize()` under the fixed daily frequency, ignoring within-period time components while retaining `datetime64[ns]`. Do not infer frequency from observed gaps or convert inputs to pandas Period dtype. Validate uniqueness using the column named by `firm_col` and the normalized column named by `period_col` in the outcome table. Two within-period rows that collapse to the same store-period are an error, not an instruction to sum or select one. Multiple event rows per store are allowed and processed by the first-event and same-period rules below.

**First-event selection and duplicate-event logging.** Validate required input values and normalize periods before selection. Group only by the column named by `firm_col`, sort by the normalized `period_col` using a stable sort, and retain that entire first row for each store. If several rows share the earliest period, retain the first in original input order as the candidate. Ignore within-period time components and retain that row's dose without averaging or summing across rows.

Select the first event across the full supplied history, before isolation, outcome availability, or sample-size screening. If it is subsequently ineligible, do not promote a later event from the same store. The target is the effect of each eligible store's earliest recorded termination in the supplied history, not necessarily its first-ever termination outside that history.

Let `n_dropped_events` be the input row count minus the number of distinct stores, and `n_affected_firms` the number of stores with more than one input row. When `n_dropped_events > 0`, emit one aggregate message per `fit()` call:

```python
logging.warning(
    "Dropped %d additional termination rows from the target table for %d "
    "firms; kept the first event by period for each firm.",
    n_dropped_events,
    n_affected_firms,
)
```

For example, three rows for one store and two for another mean three dropped rows for two affected stores. Include both same-period and later-period records in the count. With no repeated stores, do not emit this message. Record both counts in `duplicate_diagnostics_`.

Only first-event rows populate `events_`, fitted targets, global knot selection, and kernel inputs. Build `termination_history_` separately from all input records, grouped by the column named by `firm_col` and the normalized column named by `period_col`, with the number of records in that period. Later periods remain solely for isolation and control-exposure screening. No employee or new event identifier is stored.

**Same-period records.** Treat each supplied event row as one actual termination. Without employee identifiers, multiple rows at the same store and normalized period cannot be distinguished automatically from duplicate copies of one record. The working convention is that they are simultaneous terminations, including when their doses match. Preserve their count in `termination_history_`. If the first termination period has more than one record, mark its candidate `overlapping_termination` and exclude it; do not promote a later period or interpret the first row's dose as the combined reduction. Duplicate copies of the same underlying termination must be removed upstream if that is how the source data are recorded.

**Logging convention.** Every nonfatal warning in this specification must use the standard-library function `logging.warning()`, including duplicate-event summaries, failed comparison screens, missing controls, extrapolation, incomplete prediction tables, and restricted aggregations. Do not import or use the `warnings` module, warning categories, or `warnings.warn()`. Do not explicitly configure logging handlers or levels in the module; leave logging setup to the caller. Validation and required-model failures remain exceptions.

The outcome panel may have missing store-periods, including temporary inactivity and permanent closure. Every row that is present must have a finite, nonmissing outcome. An absent row is not a zero outcome. Do not fill missing periods, impute outcomes, carry observations forward, or replace a required endpoint with the nearest observed period. A real observed zero is allowed. A store or event contributes to a comparison only when its exact baseline and endpoint rows exist. Missing intermediate periods alone do not invalidate that comparison.

Use the earliest and latest normalized outcome periods as the overall study range. An optional internal integer period index may be constructed as the number of fixed-frequency steps since the earliest period plus one, with the same origin for outcome periods and termination periods. For the hardcoded daily frequency, each step is exactly one calendar day; compute elapsed steps rather than ranks of observed labels. Set $T$ to the last such index. Preserve gaps: neither within-store row numbers nor ranks of observed periods are period indices. The formulas in this document use these internal period indices for convenience; users supply datetime columns. Event periods and isolation lengths are integer numbers of periods.

A design-eligible target event has $1-L\le G_i\le T-H$; actual endpoint observation is checked separately for each comparison. Events outside the outcome range are allowed in the history table and still enter first-event selection. If a supplied earlier event lies outside the panel, do not replace it with a later in-panel event. Such extra records do not extend the isolation-screen bounds. Stores with no terminations need no event row; do not create zero-dose events. Validate nonmissing identifiers, event-store foreign keys, finite doses and outcomes, and all period and uniqueness requirements. Preserve the caller's identifier values without string coercion. Fitting must not alter either input DataFrame.

The inferred coverage interval runs from the earliest to the latest normalized outcome period over the entire panel. Assume the event table is complete for every outcome store throughout that interval, including stores with no recorded terminations and periods when a store is closed or inactive. Completeness cannot be verified from the outcome panel or the presence of individual event records. Preserve all supplied terminations for first-event selection and history screening, including those during missing outcome periods, outside the panel, overlapping other events, or excluded from estimation. Extra event records do not relax the requirement that the full isolation interval lies inside the outcome panel bounds.

Runtime dependencies are NumPy, pandas, PyFixest, SciPy, and the Python standard library. Exact runtime pins and development-tool requirements are declared in [pyproject.toml](pyproject.toml). The project requires Python 3.11; [.python-version](.python-version) selects that interpreter for tools such as uv. The module covers outcomes per period with missing store-periods allowed, equal weights for treated and control rows within each stacked comparison, no anticipation, post-treatment level estimates, and pre-treatment placebo diagnostics. Covariates, inference, plotting, treatment slopes, and changing the dose within an event are outside this scope.

Use the supplied [.pre-commit-config.yaml](.pre-commit-config.yaml) for spelling, formatting, lint, and type checks on Python, Markdown, and notebooks. The hooks use system executables, so activate the development environment before installing or running them:

```bash
uv sync --python 3.11 --group dev
source .venv/bin/activate
pre-commit install
pre-commit run --all-files
```

**`ATT` owns one fully interacted PyFixest model.** The class manages the two tables, constructs all eligible cohort-time comparisons, and estimates one `pyfixest.feols()` model. Every comparison has its own coefficients; none are shared across cohorts or horizons. The formula construction provides a place for future covariate extensions, but this specification adds no covariates or control-column parameters. Its signatures are:

```python
ATT(
    *,
    firm_col,
    period_col,
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
    isolation_window,
) -> ATT

ATT.predict(doses, *, event_times=None) -> pandas.DataFrame
```

Construction requires all four column-name arguments, a positive integer `degree`, a nonnegative integer `num_knots`, and `min_cohort_size` either `None` or a positive integer. Require a two-element `isolation_window` of integer signed offsets with $L\le-1$ and $H\ge0$, and generate all integer horizons in that interval. Set `event_times_ = list(range(L, H + 1))` internally; callers do not choose fitted horizons separately. The lower bound must include the baseline at event period $-1$; reject a positive lower offset rather than converting its sign implicitly. Dose requests must be distinct, finite, and positive. An explicit `event_times` request to `predict()` or `evaluate_theta()` must be a nonempty collection of distinct integers in the fitted interval, including negative values when requested. Dose requests must be nonempty.

`fit()` returns `self` and replaces any previously fitted state; a failed refit must not leave an earlier model available for prediction. `predict()` before successful fitting raises an error. Omitting `event_times` in either prediction or aggregation requests the complete fitted interval, including negative periods and the reference period. An out-of-window request raises an error rather than fitting additional models implicitly. `fit()` constructs and screens every `(cohort, event_time)` comparison, then stacks all eligible non-reference comparisons for one `feols()` call. Unavailable or screened cells remain explicitly recorded in diagnostics.

For a cohort $g$ and each horizon $e=L,\ldots,H$, set $t=g+e$. Use one treated row per store with a design-eligible first termination at $g$ and $A_i(e)=1$. Assert uniqueness of the store identifier. The baseline is always $g-1$, the period immediately before termination, even when the comparison endpoint precedes it or the store was inactive on that baseline period. Select controls using the complete termination history and exact endpoint availability:

$$
\mathcal C_{g,e}=\lbrace s:\quad
\text{no termination at store }s\text{ lies in }[g+L,g+H],\quad
O_{s,g-1}=O_{s,g+e}=1\rbrace.
$$

Count each control store once, regardless of workforce size. A treated store cannot also be its own control. A store may control for another event outside its own termination window, including after an earlier termination, only under the stated PT and carryover conditions. The history screen remains fixed through $g+H$ at all horizons; the observed-endpoint restriction can change both treated and control samples across horizons.

**Single stacked regression.** After the cell-specific checks below, let $\mathcal J$ contain the non-reference cells $c=(g,e)$ with $e\ne-1$ that pass both treated-design and clean-control availability checks. For each cell, stack one row per eligible treated store and one row per clean control store. The row outcome is

$$
\Delta Y_{ic}=Y_{i,g+e}-Y_{i,g-1}.
$$

Keep this order of subtraction at negative horizons. Let $Z_{ic}=1$ for the cell's eligible termination cohort and $Z_{ic}=0$ for its clean controls. At negative horizons, $Z_{ic}$ still identifies the future termination cohort. A store can appear in multiple cells and can have different roles across cells, but it appears at most once within any cell. Keep every such comparison row with weight one; reuse does not create another target event or another contribution to the kernel weights.

Let $p=q+K+1$ be the per-cell treatment-curve dimension defined by the degree and knot count below. Let $B_{ck}(d)$, $k=1,\ldots,p-1$, be the saved B-spline columns for cell $c$, excluding the first full-basis column. Each cell has a separate treatment intercept and $p-1$ spline coefficients. Estimate the following equation by unweighted OLS in one PyFixest model:

$$
\Delta Y_{ic}
=\alpha_c
+Z_{ic}\left[
\beta_{c0}+\sum_{k=1}^{p-1}\beta_{ck}B_{ck}(D_i)
\right]
+\varepsilon_{ic},
\qquad c=(g,e)\in\mathcal J.
$$

Here, $\alpha_c$ is a cell fixed effect. Every $\beta_{ck}$ is specific to the cell, including $\beta_{c0}$. For control rows, all treatment terms are defined as zero; their dose need not be defined or evaluated. There are no additional covariates, shared treatment coefficients, or regression weights. The kernel probabilities are used only in subsequent aggregation. One model does not borrow information across cells or resolve small treated cohorts.

Construct the stacked DataFrame with internal `firm_id`, `cell`, `cohort`, `period`, `event_time`, `delta_y`, and `treated` columns, where `period` is the endpoint and `cell` is an internal label for `(cohort, event_time)`. Assert uniqueness of `(firm_id, cell)`. For every cell, precompute $p$ uniquely named numeric effect columns: its treated indicator and that indicator times each of its saved spline columns. Fill them with zero for control rows and rows belonging to other cells. Evaluate the basis only on that cell's treated doses. Use deterministic safe column names and retain their ordered mapping to the cell and basis coefficients; do not generate a spline from all stacked doses or from the prediction grid.

With `effect_columns` containing all those precomputed column names, the estimation call is:

```python
import pyfixest as pf

self.model_ = pf.feols(
    "delta_y ~ " + " + ".join(effect_columns) + " | cell",
    data=stacked,
)
```

Call `feols()` exactly once after constructing and screening the stack, returning one `Feols` object. Do not use `split`, `fsplit`, multiple-model syntax, or separate per-cell estimation calls. The fixed-effect part is only `cell`. The absolute endpoint period $g+e$ and baseline period $g-1$ are fixed within each cell, so a common calendar effect contributes $\lambda_{g+e}-\lambda_{g-1}$ and is absorbed by $\alpha_c$. No separate calendar-period fixed effect is included. Use PyFixest's regression solver without constructing an explicit matrix inverse. [PyFixest 0.60.0 API and formula syntax](https://github.com/py-econometrics/pyfixest/blob/v0.60.0/pyfixest/estimation/api/feols.py)

**Equivalence and component prediction.** Let $\mathcal C_c=\mathcal C_{g,e}$ and define the clean-control mean change as $\widehat m_{0,c}$. Let $\widehat\gamma_{c0},\ldots,\widehat\gamma_{c,p-1}$ denote the coefficients from the reference treated-only regression on an intercept and the same cell basis. For identical samples and bases, the stacked OLS estimates satisfy, up to numerical precision,

$$
\widehat\alpha_c
=\widehat m_{0,c}
=\frac{1}{|\mathcal C_c|}\sum_{i\in\mathcal C_c}\Delta Y_{ic},
\qquad
\widehat\beta_{c0}
=\widehat\gamma_{c0}-\widehat m_{0,c},
\qquad
\widehat\beta_{ck}=\widehat\gamma_{ck},\quad k=1,\ldots,p-1.
$$

The identities follow by writing the treated intercept as $\gamma_{c0}=\alpha_c+\beta_{c0}$: each cell's OLS objective separates into a control intercept fit and a treated spline fit. These reference coefficients and control means establish equivalence; implementation obtains the treatment coefficients from the single PyFixest model.

For $c=(g,e)$, the reported component prediction is

$$
\widehat\tau_{g,e}(d)
=\widehat\beta_{c0}
+\sum_{k=1}^{p-1}\widehat\beta_{ck}B_{ck}(d).
$$

For $e\ge0$ this predicts ATT; for $e\le-2$ it predicts the placebo contrast. `ATT.predict()` evaluates the saved cell basis and this treatment component. It does not add $\widehat\alpha_c$ or subtract $\widehat m_{0,c}$ again. PyFixest's full outcome prediction includes the cell fixed effect and must not be returned directly as ATT. Separate control-mean estimation, fixed-effect extraction, and reference treated-only regression fits are not needed for prediction.

**Reference-cell handling.** At $e=-1$, availability requires only the observed baseline row: $A_i(-1)=O_{i,G_i-1}$. Apply the same treated-sample minimum, dose-support, rank, knot-compatibility, and clean-control availability checks as at other horizons. These are bookkeeping rules for a consistent reported population; they are not needed to identify the zero normalization itself.

For a reference cell passing those checks, save its basis definition, sample counts, and an ATT coefficient vector of exact zeros. Reference cells contribute no rows, effect columns, or fixed-effect levels to the PyFixest model. Record the cell status as `reference`, and return `att=0` with `estimate_type="reference"` for every requested dose. Reference rows use `extrapolated=False` and do not trigger extrapolation warnings because the value is a normalization, not an estimated continuation of a dose curve. A failed reference cell follows the usual screening or missing-control rules; do not supply a zero for an unavailable comparison.

**Fitted basis and knots.** Let $q$ be the degree and $K$ the number of internal knots. Each cell's treatment curve has $p=q+K+1$ coefficients, including its treatment intercept; the stacked regression also has one cell fixed effect per non-reference cell. Defaults $q=3,K=0$ give a cubic polynomial expressed in a B-spline basis.

For $K>0$, choose shared internal knots once using the positive doses from one row per design-eligible store, before endpoint-availability and small-sample screening. A store contributes once even if it is observed at several requested horizons. Use quantiles at $k/(K+1)$, $k=1,\ldots,K$, with linear interpolation matching R's default quantile convention. This uses the package's global knot-selection structure, applied to the eligible event population. Keep the selected knots fixed after screening. [Setup source](https://github.com/bcallaway11/contdid/blob/main/R/setup_pte_cont.R), [quantile helper](https://github.com/bcallaway11/contdid/blob/main/R/helper_functions.R)

For each `(cohort, event_time)` comparison, use the minimum and maximum doses among its endpoint-observed treated stores as boundary knots. Build and store the resulting basis definition: degree, internal knots, repeated boundary knots, and retained-column convention. A SciPy `BSpline` representation can evaluate that saved basis. Construct the full basis, omit its first column, and use a separate treatment intercept, consistent with the R parameterization. Multiply these terms by the treated indicator and the appropriate cell indicator when constructing the stacked design. [B-spline conventions](https://wwenjie.org/splines2/reference/bspline)

Reject duplicate global internal knots with a configuration error; do not silently change $K$. If a shared internal knot is not strictly inside a comparison's boundary knots, warn and drop that comparison with an `incompatible_knots` reason. Degenerate dose ranges and rank-deficient designs are handled by the comparison-specific screen below.

For non-reference comparisons, `predict()` must evaluate the fitted basis without recalculating knots or boundaries from the requested doses. In particular, a prediction at dose 8 must be identical whether requested alone or alongside other doses. Save a separate basis definition for each `(cohort, event_time)` because missing endpoints can change its treated sample and dose support. Do not assume that one cohort has the same basis at every horizon.

For non-reference comparisons, allow polynomial continuation outside the observed dose range, issue an extrapolation warning, and flag those rows. Do not clip doses, discard cohorts at particular doses, or renormalize weights because of extrapolation. Such values rely on the fitted functional form beyond observed support.

**Small-sample screening and dropping.** `min_cohort_size=None` resolves to the minimum treated sample size permitted by this design. Let $p=q+K+1$ be the dimension of the treated-only intercept-plus-basis design within a cell. Require positive residual degrees of freedom for that treated component, giving a treated-sample floor of $p+1$. Full-column-rank estimation of that component algebraically requires only $p$ treated observations; the extra observation is this module's design rule, not an identification threshold supplied by the article. The cell fixed effect additionally requires at least one clean control. Additional control rows or observations in other cells do not relax the treated-sample floor.

If the caller supplies a positive integer minimum, denote it by $n_{\mathrm{user}}$. Resolve the effective threshold as

$$
n_{\min}=
\begin{cases}
p+1,&\text{if the minimum is unspecified},\\
\max(n_{\mathrm{user}},p+1),&\text{otherwise}.
\end{cases}
$$

Thus `degree=3`, `num_knots=0`, and `min_cohort_size=None` give $p=4$ and an effective minimum of five stores. A user-specified minimum below five does not override the design floor; a larger minimum increases it. Resolve this threshold once from the configured basis dimension and store it in `min_cohort_size_`. Apply it separately to each `(cohort, event_time)` treated sample after endpoint selection. `None` does not disable screening, and meeting this minimum does not guarantee precision.

Count each observed treated store once; outcome rows and additional termination records do not increase the sample size. Require at least $p$ distinct doses and full column rank of the treated intercept-plus-basis matrix under a consistent documented numerical tolerance. The rank check is necessary even if the distinct-dose check passes.

For each nonempty treated comparison failing sample size, dose support, rank, or knot compatibility, warn with its cohort period, event period, store count, distinct-dose count, design dimension, and reason. Record the failure in `cell_diagnostics_`. Drop that cohort only at the affected horizon. Missing endpoints can make one comparison fail while another remains estimable. Do not reduce degree, remove knots, or pool cohorts to rescue a fit.

Keep every distinct termination period of every excluded store in `termination_history_`. That store can still be a control in other clean comparisons with observed endpoints. Exclusion must never erase a termination from history screening.

Distinguish design and availability exclusions from estimation failures. Mark an event whose period lies outside the overall outcome range as `outside_outcome_period`, and an otherwise in-range event whose isolation interval crosses a panel boundary as `isolation_outside_panel`. For every design-eligible event, all generated endpoints and the baseline lie within the overall panel range because $L\le e\le H$ and $L\le-1$. Store-specific rows may still be missing. If a comparison has no treated stores with both endpoints, mark `no_observed_treated`; it has no events in that horizon's observed-endpoint population. Record per-store `missing_baseline` and `missing_endpoint` reasons in the availability diagnostics. If a nonempty treated design passes screening but has zero endpoint-observed clean controls, warn and mark `no_controls`; do not include that cell in the stacked regression. `evaluate_theta()` must error for an affected requested horizon rather than drop the cell and renormalize silently. A singleton control group is computable for point estimation and its count is reported.

If no non-reference comparison passes the required checks, `fit()` raises an error after recording reasons and does not call `feols()`; reference zeros alone do not constitute a successful fit. Otherwise, fit all cells in $\mathcal J$ in one model and verify that every expected treatment coefficient is present and finite. A model-fitting error, nonfinite coefficient, or unexpected coefficient dropped by PyFixest makes `fit()` fail; do not silently accept a lower-dimensional curve or fall back to separate fits. At least one post-treatment or placebo cell and the single PyFixest model must be successfully fitted. Never fill a failed cell with a zero effect.

**Stored state and unpooled output.** Keep the information needed for prediction, weights, and diagnostics:

- `column_names_`: the four supplied input-column mappings; the `firm_col` and `period_col` mappings are shared by both tables.
- `min_cohort_size_`: the resolved effective sample-size minimum, including the design floor.
- `events_`: canonical columns `firm_id`, `cohort`, `dose`, and `design_eligible`, with one selected first-event row per store represented in the input events. `cohort` is a normalized `datetime64[ns]` period label. Includes excluded events; these canonical names are internal/output conventions, not required names in the user's input.
- `termination_history_`: canonical `firm_id`, normalized `cohort` period, and `n_terminations`, unique by store and period; includes later periods excluded from target estimation.
- `duplicate_diagnostics_`: `n_dropped_events` and `n_affected_firms` from first-event selection.
- `event_diagnostics_`: design eligibility and reasons such as `outside_outcome_period`, `isolation_outside_panel`, and `overlapping_termination`, keyed by store; report conflicting periods and same-period multiplicity where relevant.
- `availability_diagnostics_`: endpoint availability and missing-period reasons keyed by store and event period, including the exact baseline and endpoint periods.
- `event_times_`: every integer from the fitted lower offset through the upper offset, including -1.
- `origin_period_`, `last_period_`, `last_period_index_`, `isolation_window_`, and `knots_`. `origin_period_` and `last_period_` are the inferred screening bounds, and `last_period_index_` is $T$; no separate coverage interval is stored.
- `retained_cohorts_`: a mapping from event period to the cohort periods whose nonempty treated designs pass that horizon's screen. A missing control group does not remove a cohort from this set.
- `model_`: the single fitted PyFixest `Feols` object containing all successfully estimable non-reference cells in $\mathcal J$.
- `bases_`: saved basis definitions keyed by `(cohort, event_time)`, including successful reference cells.
- `cell_keys_`: a mapping from internal `cell` labels to `(cohort, event_time)` keys. These labels identify comparisons, not additional estimation units or termination events.
- `effect_columns_`: the ordered treatment-intercept and spline column names for each fitted non-reference cell, used to extract its coefficients from `model_`.
- `cell_coefficients_`: the extracted ATT coefficient vectors keyed by `(cohort, event_time)`; successful reference cells store exact zero vectors. These are coefficients from one model, not separate model objects. Counts and estimate types are recorded in `cell_diagnostics_`.
- `cohort_diagnostics_`: summary of design-eligible events and available, retained, or failed comparisons for each cohort.
- `cell_diagnostics_`: horizon status, treated-design checks, control counts, the exact contributing treated stores, and the unique control-store identifiers.

`ATT.predict()` returns one row per available `(cohort, event_time, dose)`, with `cohort`, `period`, `event_time`, `dose`, `att`, `estimate_type`, `n_treated`, `n_controls`, and `extrapolated`. Both `cohort` and endpoint `period` are normalized `datetime64[ns]` period labels, not internal integer indices. `estimate_type` distinguishes `att`, `placebo`, and `reference`; `n_treated` counts the termination cohort even at negative horizons. Sort by cohort, event time, and dose. Warn when requested comparisons were dropped or have no controls, directing the caller to diagnostics. Ordinary outcome-availability exclusions are recorded separately; do not manufacture predictions for them.

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

Each supplied store contributes exactly once. Do not count outcome rows, additional termination records, or workforce size, and do not average kernel contributions within each cohort. Summing store contributions preserves differences in cohort size. The Gaussian normalization constant and common $1/h$ cancel. All weights are nonnegative and sum to one at each requested dose. Compute normalization with log-sum-exp; invalid bandwidths or nonfinite intermediate results raise errors rather than triggering uniform weights.

Return `cohort`, `dose`, and `weight`, sorted by normalized cohort period and dose. No event-time argument is needed: supplying a different eligible-event set supplies the horizon dependence. The function can also inspect sensitivity to dose and bandwidth.

This is a custom estimator of conditional cohort probabilities. The article does not prescribe this kernel or bandwidth, and the package's existing dose aggregation does not supply these conditional weights. A positive bandwidth smooths cohort composition over nearby doses; consistency for exact conditional probabilities requires suitable smoothness, bandwidth, and sample-size conditions. Each store supplies only one dose/cohort observation to this calculation. Automatic bandwidth selection is outside the scope of this module.

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

For each requested horizon $e$ in the fitted interval, including negative horizons and the reference period:

1. Select all design-eligible stores with both exact outcome endpoints observed, $A_i(e)=1$. Their represented cohorts form $\mathcal G_e$.
2. Retrieve $\mathcal R_e$ from `retained_cohorts_[e]`, then keep only those observed stores in those cohorts to form $\mathcal I_e$. If empty, raise an error.
3. Require a fitted or successfully normalized reference cell for every cohort in $\mathcal R_e$. A missing control group is an error here, not an additional implicit selection rule.
4. Obtain unpooled predictions from `ATT.predict()` and weights from `estimate_pi(model.events_, ..., eligible_firms=I_e)`.
5. Join by exact cohort and dose labels. Require a complete one-to-one match and weights summing to one; do not let an inner join discard unmatched components.
6. Calculate

$$
\widehat\theta_{\mathcal E,\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
\widehat\pi_g(e,d)\thinspace{}
\widehat\tau_{g,e}(d).
$$

At $e\ge0$, this aggregates causal ATT estimates under the stated assumptions; at $e\le-2$, it aggregates placebo estimates; at $e=-1$, every retained component and their weighted sum are exactly zero. The same eligibility, kernel-weight, and required-cell checks apply to all three types. The function only aggregates existing estimates. It neither refits the curves nor changes the window, endpoint samples, or screening. Changing bandwidth changes the weights, not the underlying ATT models.

Return `event_time`, `dose`, `theta`, `estimate_type`, `n_cohorts_eligible`, `n_cohorts_used`, `n_firms_design`, `n_firms_eligible`, `n_firms_used`, `availability_restricted`, `restricted_target`, and `extrapolated`. `n_firms_design` counts stores with design-eligible first terminations, including those without observed endpoints at this horizon. Eligible counts refer to the endpoint-observed population before comparison screening; used counts refer to retained comparisons. Store and selected-event counts are identical, so separate event-count columns are unnecessary. `availability_restricted` is true when some design-eligible stores lack a baseline or endpoint row at the requested horizon. Overall panel-boundary exclusions have already occurred in the design screen. `restricted_target` is true when any nonempty eligible cohort is excluded at this horizon by a sample-size, rank, dose-support, or knot screen. Isolation restrictions always define the target and are recorded separately. For non-reference horizons, `extrapolated` is true if any contributing prediction is outside that comparison's observed dose range. The `estimate_type` values follow the same horizon-based rule as `ATT.predict()`; reference rows have `extrapolated=False`. Sort by event time and dose; component estimates and weights remain separately inspectable.

**Observation availability and screening change the target.** First-event selection limits candidates to one termination per store. The isolation design then restricts that population to $S_i=1$. Outcome availability then restricts it to $A_i(e)=1$ at each horizon. After comparison screening, weights estimate

$$
P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1,G_i\in\mathcal R_e).
$$

For a given retained set, the population counterpart is

$$
\theta_{\mathcal E,\mathcal R}(d,e)
=\sum_{g\in\mathcal R_e}
P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1,G_i\in\mathcal R_e)
\tau_{g,e}(d).
$$

This equals the pre-screen observed-endpoint target only when every eligible cohort is retained or under additional special conditions. It does not generally equal the effect over all terminations or all stores before the isolation, availability, and comparison screens. Keep this population restriction and the ATT/placebo/reference distinction despite the short output name `theta`. Warn when comparison screening restricts an aggregation, listing affected horizons. Availability exclusions are separately visible through counts, flags, and missing-period diagnostics. Never retain excluded stores in the kernel denominator, replace unknown effects with zero, or imply that renormalization recovers missing effects.

The screen is sample-dependent; this description is not an inferential result about selection. Both within-cohort store composition and retained cohorts may change across horizons. The fixed isolation window ensures period coverage from $L$ through $H$, including the baseline, for all design-eligible events. No common set of stores with complete outcomes is imposed across horizons; inactivity or closure can still change the observed sample.

**Implementation boundaries and validation plan.** Put the class, both public functions, and a few private validation, window-selection, basis, and stacked-design helpers in the single `src/labor_cost_optimization/att.py` file. Use the `labor_cost_optimization` package under the `src` directory; keep estimator logic in that module. Use `firm` and `period` consistently in all Python identifiers, canonical column names, docstrings, and error/log messages, including private helpers. Use `period_col`, `origin_period_`, `last_period_`, `last_period_index_`, and output column `period`; preserve literal third-party API/type names such as `datetime64[ns]` and `.dt.normalize()`. This plan keeps the application term “store” in explanatory prose; code examples and public schema names follow the general `firm` and `period` terminology. Use `logging.warning()` for nonfatal diagnostics and small dictionaries or records for fitted cells. External R code is not a runtime dependency. No generic estimator framework, serialization layer, or extra classes are needed.

**Docstring convention.** Use Google-style docstrings throughout `src/labor_cost_optimization/att.py`, including the module, `ATT`, its constructor and public methods, both public functions, and private helpers. Start with a concise summary followed by a blank line before any extended description or sections. Use `Args:`, `Returns:`, and `Raises:` where applicable; omit empty or inapplicable sections. Document constructor arguments in `__init__` and fitted public attributes in the class docstring's `Attributes:` section. Simple helpers may use a one-line docstring when no further contract needs explaining.

Indent argument names, exception classes, and attribute names four spaces relative to the section heading, each followed by a colon on its own line. Start each description on the next line, indented eight spaces, and keep wrapped description lines at that indentation. Plain prose in `Returns:` stays at four spaces. Preserve blank lines between the summary and sections and between successive sections. Document every argument except `self` and `cls`, using its exact signature name; type annotations belong in the signature. Explain returned objects and relevant DataFrame schemas, and describe the conditions for exceptions the caller needs to handle. Describe state changes such as `fit()` replacing earlier fitted state and when fitted attributes become available. Keep docstrings consistent with the implemented behavior and use `firm` and `period` throughout. Nonfatal diagnostics emitted by `logging.warning()` are not exceptions and do not belong in `Raises:`.

For example, the `estimate_pi()` docstring follows this layout:

```python
"""Estimate dose-specific cohort weights for eligible firms.

Args:
    events:
        Canonical first-event metadata with firm_id, cohort, and dose
        columns, containing one row per firm.
    doses:
        Distinct positive finite doses at which to evaluate weights.
    eligible_firms:
        Unique firm identifiers defining the eligible sample.
    bandwidth:
        Positive finite Gaussian-kernel bandwidth in dose units.

Returns:
    A DataFrame with cohort, dose, and weight columns, sorted by cohort
    and dose. Weights sum to one at each requested dose.

Raises:
    ValueError:
        If the bandwidth is nonpositive or nonfinite, the inputs
        violate the required schema or eligibility rules, or weight
        normalization produces nonfinite results.
"""
```

The pytest suite in [tests/test_att.py](tests/test_att.py) checks the implementation against independent numerical calculations and targeted examples. Run `uv sync --python 3.11 --group dev`, then `uv run --no-sync python scripts/validate.py --log validation.log` from the repository root. The runner [scripts/validate.py](scripts/validate.py) records environment versions, package-source verification, test tracebacks, and read-only quality checks in a shareable log, continuing after individual failures. It performs no installation and returns a nonzero exit status for any failed check or mismatch with the configured runtime requirements. It uses `packaging` when available to interpret Python constraints and dependency requirements, including ranges and environment markers; missing parsing support is reported without preventing the remaining checks. Run the tests directly with `python -m pytest tests -v`; [tests/test_validate.py](tests/test_validate.py) also checks the diagnostic runner without importing the estimator. Quality commands target only the estimator module, the two test modules, the runner, and project documentation. Missing required files produce a failed validation result; quality checks continue on the expected files that are present, with no broad directory scan. Development dependencies include `pytest` and `codespell`. Required behaviors include:

- All four column-name arguments are required; omitting any one raises `TypeError` at construction. No conventional input labels or internal schema names supply defaults. Renaming every input column through the constructor leaves results unchanged, using the same `firm_col` and `period_col` labels in both tables. Inputs missing either shared column fail validation; `outcome_col` and `dose_col` may share a label across the two tables.
- `fit()` has no `event_times` argument and generates `list(range(-7, 15))` for `(-7, 14)`. Default prediction and aggregation include all 22 horizons; explicit subsets may include negative periods, and out-of-window requests fail.
- The signed window `(-7, 14)` screens exactly from $g-7$ through $g+14$, including both boundaries; `(7, 14)` is rejected. Check the panel-bound conditions $g+L\ge1$ and $g+H\le T$.
- `min_cohort_size=None` resolves to $p+1$ for the configured degree and knot count; a smaller explicit minimum cannot bypass that floor, while a larger one is enforced. The default cubic design resolves to five.
- Frequency is hardcoded daily, with `datetime64[ns]` input/output types. There is no frequency inference, configurable frequency, or additional frequency support.
- Within-period time differences do not affect estimates; same-store within-period duplicates in the outcome table are rejected after normalization, as are `NaT` and timezone-aware inputs. Missing periods are never compressed.
- Multiple termination rows for a store retain only its earliest normalized-period row and its dose. Same-period ties use original row order; an ineligible first event is not replaced. Check the aggregate log counts, including three dropped rows for two stores in the example, and no duplicate warning when each store has one event row.
- Later distinct periods dropped from targets still block conflicting controls or isolation windows. Multiple records on the first store-period retain their multiplicity in history and exclude the candidate as overlapping; no later period is promoted.
- Capture warning-level log records to verify every specified warning uses `logging.warning()`; no warnings-module emissions are produced.
- A missing baseline or endpoint excludes that store only from affected comparisons; missing intervening periods alone do not. Present rows with missing or infinite outcomes fail validation. Temporary inactivity can cause a store to contribute at one horizon, disappear at another, and reappear later.
- Pre-treatment placebo estimates use $Y_{g+e}-Y_{g-1}$ with the correct sign and a fixed baseline. A known dose-related pre-trend is recovered without being forced to zero or labeled as a causal ATT.
- An eligible reference cell returns exact zero ATT coefficients, predictions, and aggregation at $e=-1$, with the correct estimate type and counts. It contributes no regression rows or columns. Reference-only successes cannot hide failure of every non-reference comparison.
- One `feols()` call produces one `Feols` object for all successfully estimable non-reference cells in $\mathcal J$, without `split` or `fsplit`. On a constructed sample with multiple cohorts and horizons, its ATT curves match separate treated-only spline OLS predictions minus each cell's control mean, including a cell with a single control.
- A two-period comparison with a known polynomial dose effect and a nonzero control trend recovers the curve from the treatment coefficients. Prediction neither adds the cell fixed effect nor subtracts a control mean again.
- Stacked design columns use each cell's saved basis and are zero for controls and rows belonging to other cells. All stacked rows receive unit weight; reused stores remain present in every eligible comparison.
- A common calendar-period shock cancels through each cell's control comparison without a separate calendar-period fixed effect.
- Missing or nonfinite expected treatment coefficients after `feols()` cause fitting to fail. No automatically dropped spline term may silently redefine a retained curve.
- Predictions at a fixed dose are invariant to other requested doses and input row order. For same-period ties, the selected candidate row follows input order, but the candidate is excluded as overlapping, so its dose cannot change fitted effects. Where R training and prediction bases coincide, unpooled estimates on the same constructed sample agree numerically; otherwise compare against an R reference that explicitly reuses the training basis.
- Two sufficiently separated termination periods at one store produce only one target event and one kernel contribution. Its later period remains in the history screen, and every control store contributes once per comparison.
- Simultaneous and overlapping events are excluded using the original complete history. An excluded event or a dropped cohort still disqualifies its store from controls in every conflicting window.
- Screening bounds equal the overall normalized outcome-period minimum and maximum, regardless of individual-store gaps or additional event periods. An isolation interval crossing either boundary excludes the event at every horizon; intervals touching the boundaries are allowed. Extra event records outside the panel cannot rescue an excluded event. Changing only prediction horizons leaves the fitted screen unchanged.
- Small, rank-deficient, and incompatible-knot comparisons warn and disappear from the corresponding horizon only, while retaining the complete history. Verify that an estimable short horizon survives a sparse later horizon, with separate fitted bases and matching store-specific kernel inputs.
- Weights count each selected store once, sum to one, reflect unequal cohort sizes, and match the explicit formula. A within-cohort excluded event cannot leak back into the weights.
- Aggregation equals a manual weighted sum for both post-treatment and placebo horizons, reports store counts and estimate types, identifies the retained population, and rejects any requested horizon with a missing required cell, including a reference cell with no controls.

The implementation provides point estimates through `ATT`, `estimate_pi()`, and `evaluate_theta()`. Future inference must cluster or resample by store, carrying its selected event, full termination history, and outcomes together, and account for shared controls, estimated weights, smoothing, and selection. Clustering handles dependence; it does not fix contaminated controls or justify the identifying assumption. Store-level independence and an adequate number of independent stores would still require assessment.

## References

Callaway, B., Goodman-Bacon, A. and Sant'Anna, P.H.C. (2025) *Difference-in-Differences with a Continuous Treatment*. Working paper, draft dated 31 December. Available at: [https://psantanna.com/files/CGBS_v4.pdf](https://psantanna.com/files/CGBS_v4.pdf) (Accessed: 3 November 2026).
