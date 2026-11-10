# Continuous-Dose ATT and Dose-Specific Event-Study Aggregation

The `att.py` module estimates continuous-dose treatment effects separately by treatment cohort and event period, then averages over cohorts at each fixed dose and horizon. The application below uses the scheduled hours removed by a store's first recorded employee termination as its fixed dose and store sales as the outcome. Polynomial comparison models, conditional cohort weights, bandwidth selection, and pointwise bootstrap intervals are documented together with their assumptions and limitations.

## Theoretical Background

### Units, Treatment, and Isolation

Callaway et. al. (2025) provide the identification and aggregation starting point for a staggered-adoption design with a fixed continuous dose. In the application described here, store $`i`$ is the unit, $`G_i`$ is its earliest recorded termination period, and $`D_i>0`$ is the scheduled hours per period removed by that termination. Only the first termination is a target event. Later terminations remain relevant to target-event isolation, but do not create additional target events or aggregation contributions. Control eligibility is determined by the first termination period.

The fixed treatment label is

```math
W_{it}=D_i\thinspace{}1\lbrace t\ge G_i\rbrace.
```

The outcome $`Y_{it}`$ is store sales per period. The dose measures hours removed by the selected termination, rather than the store's total subsequent staffing reduction. Replacement hiring and redistribution of work belong to the total effect of that termination; they do not redefine its dose. A period is one calendar day in the implementation.

The isolation window $`(L,H)`$ satisfies $`L\le-1`$ and $`H\ge0`$ and includes every integer event period $`e\in\lbrace L,L+1,\ldots,H\rbrace`$. For a termination at $`g`$, the closed interval $`[g+L,g+H]`$ must lie within the overall outcome panel bounds and contain no other termination at the same store. With the first and last panel periods indexed by 1 and $`T`$, eligibility requires $`g+L\ge1`$ and $`g+H\le T`$. Equality at a boundary is allowed. Simultaneous terminations exclude the affected target because only one selected termination is exempted from the overlap count.

Termination recording is assumed complete for every outcome store throughout the overall panel bounds, including periods without sales observations. Additional termination history outside those bounds does not extend the observed panel. Dropping an event from target estimation does not remove it from termination history or erase its store's first-treatment status. A target whose full isolation interval crosses a panel boundary is excluded at every horizon, even if a shorter comparison has observed endpoints.

Write $`S_i=1`$ for a target passing these design restrictions and $`\mathcal{E}`$ for the resulting event population before endpoint and small-sample screening. The same isolation window applies at all horizons. Changing it can change the population even at event period zero. Window lengths must reflect plausible anticipation and disruption by additional terminations; spacing between events alone does not establish identification. Reuse of store outcomes across comparisons also creates dependence.

### Observed Comparisons and Parallel Trends

For cohort $`g`$ and horizon $`e`$, the outcome period is $`t=g+e`$, with baseline $`g-1`$ at every horizon. Define $`O_{s,u}=1`$ when store $`s`$ has a valid observed outcome in period $`u`$, and zero otherwise. Target endpoint availability is

```math
A_i(e)=O_{i,G_i-1}\thinspace{}O_{i,G_i+e}.
```

Both endpoints must be observed exactly; missing intervening periods do not prevent this comparison. Let $`\mathcal{E}_e`$ contain the design-eligible events satisfying $`A_i(e)=1`$. For control selection, $`G_s`$ is the first termination in the full supplied history of store $`s`$, with $`G_s=\infty`$ for a store with no recorded termination. The control set is

```math
\mathcal{C}_{g,e}=\lbrace s:\quad
G_s>\max(g,g+e),\quad
O_{s,g-1}=O_{s,g+e}=1\rbrace.
```

For post-treatment horizons $`e\ge0`$, this is $`G_s>t`$ with $`t=g+e`$: the control has not yet been treated at the comparison endpoint, including stores with no recorded termination. This is the control-timing rule in Callaway et. al. (2025), Theorem C.1, printed p. 40. A store first treated exactly at $`t`$ is excluded. A store first treated after $`t`$ can be a control even when that termination lies before the end $`g+H`$ of the target's isolation window. Previously treated stores never reenter the control pool.

For negative horizons, including the reference at $`e=-1`$, the rule becomes $`G_s>g`$. This keeps both the endpoint and the fixed baseline $`g-1`$ untreated and excludes the entire focal cohort, including its ineligible target candidates. It is the module's fixed-baseline placebo convention; Theorem C.1 states the post-treatment identification result.

Each control store contributes once, regardless of workforce size. Its own target event need not pass isolation, endpoint-availability, or sample-size screening. Control timing uses every selected first-event record, including excluded candidates and events outside the outcome panel. The post-treatment timing-eligible pool shrinks as the endpoint advances and future stores become treated; missing endpoint outcomes can also change the actual sample. A future termination immediately after the endpoint is allowed, so no anticipation remains an identifying assumption.

The article-style untreated interpretation requires correctly observed treatment-entry periods. The module can identify only the earliest supplied event, not a store's lifetime first termination. Relevant earlier history must be supplied, or an untreated-at-panel-start assumption must be credible. Complete recording within the outcome panel alone cannot rule out earlier unrecorded treatment, including for a store assigned an infinite cutoff because it has no event rows.

For an eligible target, $`Y^{\mathrm{term}}_{i,t}`$ denotes store sales under the observed termination and $`Y^{\mathrm{keep}}_{i,t}`$ sales if that employee continues at the original scheduled hours through the comparison endpoint. The regimes share the actual history before this first recorded termination, with subsequent staffing responses following their respective regimes. The scope of the supplied history determines which event is considered first.

For post-treatment horizons $`e\ge0`$, the event-specific analogue of the article's unpooled target is

```math
ATT_{\mathcal{E}_e}(g,t,d\mid g,d)
=E[Y^{\mathrm{term}}_{i,t}-Y^{\mathrm{keep}}_{i,t}
\mid G_i=g,D_i=d,S_i=1,A_i(e)=1].
```

In addition to consistency, no anticipation of the focal termination, and no relevant spillovers between stores, identification uses the selected-sample parallel-trends restriction

```math
E[Y^{\mathrm{keep}}_{i,t}-Y^{\mathrm{keep}}_{i,g-1}
\mid G_i=g,D_i=d,S_i=1,A_i(e)=1]
=E[Y_{s,t}-Y_{s,g-1}\mid s\in\mathcal{C}_{g,e}].
```

The right-hand expectation weights eligible not-yet-treated control stores equally. The first-treatment cutoff keeps controls untreated at both comparison periods. Parallel trends must still hold for the isolated, endpoint-observed target population and its observed controls; matching the article's timing rule alone does not establish this selected-sample restriction. There is no covariate or treatment-history adjustment to repair a failure of it. Strong parallel trends is not assumed, and slopes across doses are not interpreted as causal marginal responses.

Isolation based on later terminations and selection on observed outcomes use post-event information and may themselves be affected by the focal termination. Identification is asserted for this selected population under the stated restriction. Excluding stores after closure does not recover an effect that includes their subsequent sales or closure consequences. Generalization to all isolated events needs additional assumptions or a different outcome definition. The selected population is not guaranteed to remain isolated and observed under both counterfactual regimes. Parametric estimation also requires an adequate dose-curve specification and relevant dose support.

### Dose-Specific Event-Study Target

Let $`\mathcal{G}_e`$ be the cohorts represented among $`\mathcal{E}_e`$. For $`e\ge0`$, the requested aggregation is

```math
\theta_{\mathcal{E}}(d,e)=\sum_{g\in\mathcal{G}_e}\pi_g(e,d)
ATT_{\mathcal{E}_e}(g,g+e,d\mid g,d),
\qquad
\pi_g(e,d)=P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1).
```

Dose and event period remain fixed; only treatment-period cohorts are averaged. Each eligible store contributes its first termination once to the cohort probabilities. Additional termination records create no extra weight. The conditional composition of cohorts can vary with dose, so the weights need not be equal across stores at a requested dose.

### Negative Horizons and the Reference Period

For $`e\le-2`$, the dose-specific pre-treatment contrast is

```math
P_{g,e}(d)
=E[Y_{i,g+e}-Y_{i,g-1}\mid G_i=g,D_i=d,S_i=1,A_i(e)=1]
-E[Y_{s,g+e}-Y_{s,g-1}\mid s\in\mathcal{C}_{g,e}].
```

The subtraction order remains outcome at $`g+e`$ minus outcome at $`g-1`$; for example, $`e=-7`$ uses $`Y_{g-7}-Y_{g-1}`$. Membership in the treated sample identifies the future termination cohort at negative horizons.

These contrasts are placebo diagnostics. No anticipation and the pre-period counterpart of the selected-sample parallel-trends restriction imply zero population values. Post-treatment parallel trends alone does not imply this result. Finite-sample nonzero values are not causal effects of a termination that has not occurred. Formal joint pre-trend testing is not supplied.

At $`e=-1`$, both endpoints are $`g-1`$, so outcome changes and their contrast are exactly zero. This is a reference normalization, not an independently estimated treatment effect or evidence for parallel trends. A common component notation is

```math
\tau_{g,e}(d)=
\begin{cases}
ATT_{\mathcal{E}_e}(g,g+e,d\mid g,d),&e\ge0,\\
P_{g,e}(d),&e\le-2,\\
0,&e=-1.
\end{cases}
```

At every horizon the aggregation is

```math
\theta_{\mathcal{E}}(d,e)
=\sum_{g\in\mathcal{G}_e}\pi_g(e,d)\thinspace{}\tau_{g,e}(d).
```

Negative-horizon values therefore represent the corresponding placebo or reference quantity. For every available reference population, the aggregation is exactly zero at every dose.

### Relationship to Callaway et. al. (2025)

One adoption period and a fixed dose per store match the article's treatment-label structure, and post-treatment controls follow its not-yet-treated timing rule. Target-event isolation, observed-outcome selection, and pre-treatment conventions retain the distinctions below.

| Feature | Article's Staggered-Adoption Setting | Store-Termination Design |
| --- | --- | --- |
| Treatment histories | One adoption period and fixed subsequent dose per unit, Assumption 3-MP, printed p. 39 | One first-termination label and dose per store; later terminations screen target-event overlap |
| Observation structure | Independent sampled units, Assumption 1-MP, p. 39 | One outcome trajectory per store; independence across stores remains an assumption, and stores may recur across comparisons |
| Outcome availability | Outcomes in every modeled panel period | Store-period gaps are allowed; each comparison has its own observed-endpoint population |
| Counterfactual and parallel trends | Outcomes without treatment; PT-MP and Theorem C.1, pp. 39–40 | No first recorded termination; parallel trends for isolated events and not-yet-treated controls with observed endpoints |
| Controls | Not-yet-treated or never-treated units at the outcome period, Theorem C.1, p. 40 | The same post-treatment cutoff, with exact observed endpoints; no previously treated controls |
| Aggregation population | Dose-specific event-study aggregation over treatment cohorts, p. 41 | That structure applied to eligible first terminations with observed endpoints and horizon-specific retained cohorts |
| Pre-treatment output | The cited post-treatment ATT aggregation has a causal level-effect interpretation | Negative horizons are fixed-baseline placebo contrasts using cohorts treated after the focal cohort; event period -1 is normalized to zero |

The dose-specific event-study aggregation on printed page 41 is unnumbered. It supplies the aggregation structure; the additional target isolation and observed-outcome restrictions still require the stated selected-sample identifying assumptions. Remark C.1 on page 42 discusses component estimation and aggregation. The kernel weighting, bandwidth criteria, and bootstrap below are custom additions, rather than the article's prescribed estimators or inference routine.

### Polynomial Comparison Estimation

The parametric level-effect calculation in `contdid::cont_did_acrt` fits raw outcome changes on an intercept and dose B-splines within the focal treated cohort, then subtracts the control group's mean change. Without internal knots, degree-$`q`$ B-splines span the same functions as ordinary polynomials of degree at most $`q`$. For identical treated and control samples, degree, baseline convention, and equal observation weights, the polynomial regression below is algebraically equivalent to that calculation when predictions consistently reuse the training basis. This equivalence does not extend to specifications with internal spline knots or additional covariates. It is not a claim of identical numerical output from an unmodified R package run; sample selection, pre-treatment comparisons, and prediction-basis handling differ as described in [Comparison With contdid Before Aggregation](#comparison-with-contdid-before-aggregation).

Let $`\mathcal{J}`$ contain the non-reference cells $`c=(g,e)`$ with $`e\ne-1`$ that pass treated-design screening and have eligible controls. Each cell contains one row per eligible treated store and one row per eligible control store. Its row outcome is

```math
\Delta Y_{ic}=Y_{i,g+e}-Y_{i,g-1}.
```

Let $`Z_{ic}=1`$ for the cell's eligible termination cohort and $`Z_{ic}=0`$ for its controls. A store appears at most once within a cell, but may recur across cells or change roles. Every comparison row has weight one. Reuse creates neither another target event nor another kernel contribution.

For polynomial degree $`q`$ and treatment-curve dimension $`p=q+1`$, let $`a_c`$ and $`b_c`$ be the minimum and maximum endpoint-observed treated doses in that cell. The saved scaling is

```math
x_c(d)=\frac{d-a_c}{b_c-a_c}.
```

This changes the coefficient parameterization without changing the polynomial function space. The unweighted OLS equation, estimated separately for every cell, is

```math
\Delta Y_{ic}
=\alpha_c
+Z_{ic}\left[
\beta_{c0}+\sum_{k=1}^{q}\beta_{ck}x_c(D_i)^k
\right]
+\varepsilon_{ic},
\qquad c=(g,e)\in\mathcal{J}.
```

All coefficients are specific to the cell. Treatment terms are zero for controls. There are no additional covariates, shared coefficients, or regression weights; kernel probabilities enter only subsequent aggregation. Separate estimation does not pool information or resolve small daily cohorts.

The absolute outcome period $`g+e`$ and baseline period $`g-1`$ are fixed within a cell. A common calendar effect contributes $`\lambda_{g+e}-\lambda_{g-1}`$ and is absorbed by $`\alpha_c`$; no separate calendar-period fixed effect is included.

Let $`\mathcal{C}_c=\mathcal{C}_{g,e}`$, let $`\widehat{m}_{0,c}`$ be the control mean change, and let $`\widehat{\gamma}_{c0},\ldots,\widehat{\gamma}_{cq}`$ be the coefficients from the reference treated-only polynomial regression. Then

```math
\widehat{\alpha}_c
=\widehat{m}_{0,c}
=\frac{1}{|\mathcal{C}_c|}\sum_{i\in\mathcal{C}_c}\Delta Y_{ic},
\qquad
\widehat{\beta}_{c0}
=\widehat{\gamma}_{c0}-\widehat{m}_{0,c},
\qquad
\widehat{\beta}_{ck}=\widehat{\gamma}_{ck},\quad k=1,\ldots,q.
```

Writing the treated intercept as $`\gamma_{c0}=\alpha_c+\beta_{c0}`$ separates the OLS objective into a control-intercept fit and a treated-polynomial fit. A joint regression with wholly cell-specific coefficients separates into the same objectives, so separate cell fitting preserves its point estimates. Basis coefficients may differ from the R parameterization even though fitted curves agree.

For $`c=(g,e)`$, the component prediction is

```math
\widehat{\tau}_{g,e}(d)
=\widehat{\beta}_{c0}
+\sum_{k=1}^{q}\widehat{\beta}_{ck}x_c(d)^k.
```

This is the treatment contrast, rather than the full predicted outcome. The control intercept is not added, and the control mean is not subtracted again. The saved transformation is reused for every prediction; changing the prediction grid does not change the curve. Each cell may have a different transformation because endpoint availability changes its treated sample. Predictions outside observed dose support continue the polynomial and rely on that functional-form assumption; extrapolation does not remove or reweight cohorts.

At the reference horizon, $`A_i(-1)=O_{i,G_i-1}`$. The same treated-sample, dose-support, rank, and control-availability rules determine whether its population is available, but its effect is exactly zero without regression estimation. These screens are bookkeeping rules for a consistent reported population, not conditions needed to identify zero itself.

### Comparison Screening and the Restricted Target

The treatment component has $`p=q+1`$ coefficients. Full rank algebraically requires at least $`p`$ treated observations, while this design uses a floor of $`p+1`$ to retain positive residual degrees of freedom in the treated component. At least one eligible control is also required. Neither additional control rows nor observations in other cells relax the treated floor. If an additional user-selected minimum is $`n_{\mathrm{user}}`$, the threshold is

```math
n_{\min}=
\begin{cases}
p+1,&\text{if the minimum is unspecified},\\
\max(n_{\mathrm{user}},p+1),&\text{otherwise}.
\end{cases}
```

For a cubic curve, $`p=4`$ and the default minimum is five treated stores. Meeting the threshold does not guarantee precision. Each cell also needs at least $`p`$ distinct treated doses and a full-rank intercept-plus-power design. Numerical rank decisions can depend on scaling and floating-point tolerance even when polynomial parameterizations are mathematically equivalent.

A cohort failing sample size, support, or rank is excluded only at the affected horizon. It may remain estimable at others, and its stores' first termination periods still determine when they can serve as controls. A comparison with no observed treated endpoints is absent from that horizon's observed-event population. A retained treated comparison with no controls is an estimation failure, not an extra population screen: aggregation does not silently discard it or supply a zero effect.

Let $`\mathcal{R}_e\subseteq\mathcal{G}_e`$ be the retained cohort set and $`\mathcal{I}_e`$ the stores in those cohorts with eligible first events and both endpoints observed. After screening, the cohort probabilities concern

```math
P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1,G_i\in\mathcal{R}_e).
```

For a fixed retained set, the population counterpart is

```math
\theta_{\mathcal{E},\mathcal{R}}(d,e)
=\sum_{g\in\mathcal{R}_e}
P(G_i=g\mid D_i=d,S_i=1,A_i(e)=1,G_i\in\mathcal{R}_e)
\tau_{g,e}(d).
```

This equals the pre-screen observed-endpoint target only when all eligible cohorts are retained or under additional special conditions. It does not generally equal the effect over all terminations or all stores before the isolation, availability, and sample screens. Renormalization does not recover missing effects.

Both the within-cohort store composition and retained cohorts can change with horizon. A fixed isolation window ensures calendar coverage in the overall panel, but does not impose complete observations for every store. Inactivity and closure can still change the sample. The screen is sample-dependent; these definitions do not establish inference conditional on a selection event.

### Conditional Cohort Weights and Aggregation

For the exact store set $`\mathcal{I}_e`$ and a common bandwidth $`h>0`$, the Gaussian-kernel estimator is

```math
q_g(d;h)=\sum_{i\in\mathcal{I}_e:G_i=g}
\exp\left[-\frac{1}{2}\left(\frac{D_i-d}{h}\right)^2\right],
\qquad
\widehat{\pi}_g(e,d)
=\frac{q_g(d;h)}{\sum_{k\in\mathcal{R}_e}q_k(d;h)}.
```

Every supplied store contributes once. Summing contributions rather than taking a mean within each cohort preserves differences in cohort size. Outcome rows, workforce size, and additional termination records do not enter the count. The Gaussian normalization constant and common $`1/h`$ cancel, and the weights are nonnegative and sum to one at each requested dose.

The conditional event-study estimate is

```math
\widehat{\theta}_{\mathcal{E},\mathcal{R}}(d,e)
=\sum_{g\in\mathcal{R}_e}
\widehat{\pi}_g(e,d)\thinspace{}
\widehat{\tau}_{g,e}(d).
```

At nonnegative horizons it aggregates ATT estimates under the stated assumptions; at earlier horizons it aggregates placebo contrasts or the zero reference. Changing bandwidth changes the cohort weights, not the fitted comparison curves. The store set must match the contributing comparisons exactly; cohort labels alone do not identify it when isolation or missing endpoints exclude individual stores.

A positive bandwidth smooths cohort composition over nearby doses. Consistency for exact conditional probabilities requires suitable smoothness, bandwidth, and sample-size conditions. Each store supplies one dose/cohort observation. This conditional-probability estimator is not prescribed by Callaway et. al. (2025), and the R package's existing dose aggregation does not supply these conditional weights.

### Bandwidth Criteria

Bandwidth selection uses one selected first event per store in a specified population, independently of outcomes. It does not automatically impose the isolation, endpoint, or regression screens. Reusing a selected bandwidth across horizons does not make its validation risk an average over the different horizon-specific populations.

Let $`\mathcal{I}`$ be this store set, $`n=|\mathcal{I}|`$, and $`s_D`$ its dose sample standard deviation with divisor $`n-1`$. Let $`Q_D`$ be its interquartile range based on linearly interpolated sample quartiles. The dose-scale starting rule is

```math
\widehat{h}_0
=0.9\thinspace{}s_\star\thinspace{}n^{-1/5},
\qquad
s_\star=
\begin{cases}
\min(s_D,Q_D/1.349),&Q_D>0,\\
s_D,&Q_D=0.
\end{cases}
```

This positive bandwidth is measured in dose units, hours here. It requires at least two stores and positive dose variation. The standard-deviation fallback accommodates repeated doses when the interquartile range is zero; it cannot assign a bandwidth to constant doses. This Silverman rule is a scale-based candidate, not an optimum derived for conditional cohort probabilities, and does not require multiple cohorts.

Cross-validation predicts the categorical cohort $`G_i`$ from dose $`D_i`$. The exact leave-one-store-out Gaussian probabilities are

```math
\widehat{p}_{g,-i}(D_i;h)
=
\frac{\displaystyle\sum_{j\in\mathcal{I}:\,j\ne i,\,G_j=g}
\exp\left[-\frac{1}{2}\left(\frac{D_j-D_i}{h}\right)^2\right]}
{\displaystyle\sum_{j\in\mathcal{I}:\,j\ne i}
\exp\left[-\frac{1}{2}\left(\frac{D_j-D_i}{h}\right)^2\right]}.
```

Only the held-out store's own kernel contribution is removed. Other stores at the same dose remain in training. With $`\mathcal{G}`$ denoting represented cohorts, the mean multiclass Brier loss is

```math
CV(h)
=\frac{1}{n}\sum_{i\in\mathcal{I}}
\sum_{g\in\mathcal{G}}
\left[1\lbrace G_i=g\rbrace-\widehat{p}_{g,-i}(D_i;h)\right]^2.
```

Every store has weight $`1/n`$. The loss neither averages equally over unique doses or cohorts nor divides by the number of cohorts. It uses only doses and cohort labels; sales and ATT estimates do not enter. It tunes cohort-composition prediction rather than the outcome regression or mean squared error of $`\theta`$.

Cross-validation requires at least two stores, two cohorts, and positive dose variation. A singleton cohort remains a class in the loss; holding out its sole store gives probability zero for that store's own cohort. Stores or classes are not discarded or given pseudocounts for this reason.

The bandwidth search compares a geometric candidate grid and expands initially minimizing boundaries by repeated factors of two. Its numerical stopping criterion compares successive losses $`L_{\mathrm{prev}}`$ and $`L_{\mathrm{new}}`$ using

```math
\delta
=64\thinspace{}\epsilon\thinspace{}
\max(1,|L_{\mathrm{prev}}|,|L_{\mathrm{new}}|),
```

Here $`\epsilon`$ is machine precision for the score type. A meaningful rise means $`L_{\mathrm{new}}-L_{\mathrm{prev}}>\delta`$; three consecutive changes with $`|L_{\mathrm{new}}-L_{\mathrm{prev}}|\le\delta`$ indicate a numerical plateau. This tolerance governs stopping, not exact minimizer ties. A plateau, expansion cap, or numerical-range stop does not establish that the risk has turned upward. The procedure selects the minimum over visited candidates and is not a global optimizer over all positive bandwidths. It does not search between candidates or beyond a detected rise.

Both bandwidth rules are custom additions. They do not supply identification or inference for the event-study target.

### Store Bootstrap and Pointwise Intervals

The bootstrap resamples independent stores, retaining each store's complete outcome trajectory and all its appearances in the comparisons. It is a custom inference procedure for the selected-store target, not a reproduction of the article's or R package's inference routine.

The original horizon-specific retained sets $`\mathcal{R}_e`$, exact treated and control memberships, polynomial degrees and saved transformations, dose grid, and bandwidth remain fixed. First-event, isolation, endpoint, and original minimum-size screens are not rerun; bandwidth is not reselected. The weighted polynomial regressions must nevertheless remain identifiable in each draw. Holding this specification fixed does not establish inference conditional on data-dependent selection or account for changes in screening and tuning across samples.

#### Sampling and Comparison Estimates

Let $`N`$ count every distinct store in the original outcome panel, including stores unused by the requested comparisons. Each of the $`B`$ attempted replications draws $`N`$ stores with replacement, giving frequency counts

```math
(M_1^{(b)},\ldots,M_N^{(b)})
\sim\mathrm{Multinomial}
\left(N;\frac{1}{N},\ldots,\frac{1}{N}\right).
```

The same $`M_i^{(b)}`$ applies to every period, treated/control appearance, and kernel contribution of store $`i`$. This preserves within-store dependence and covariance across comparisons, including shared controls and estimated cohort composition. Stores without selected events can still contribute as controls. Calendar periods, individual outcome rows, and fixed numbers within cohorts are not resampled separately. Integer weights are algebraically equivalent to duplicating entire store histories with distinct copy identifiers.

For $`c=(g,e)`$, write $`T_{g,e}=\mathcal{I}_e\cap\{i:G_i=g\}`$ and let $`C_{g,e}`$ denote the original controls. Let $`p_{g,e}(d)`$ contain the intercept and powers of the saved scaled dose through the configured degree; the prime denotes transpose. For a non-reference comparison with positive sampled treated mass, the weighted estimates are

```math
\widehat{\gamma}_{g,e}^{*(b)}
=\arg\min_\gamma
\sum_{i\in T_{g,e}}M_i^{(b)}
\left[\Delta Y_{ic}-p_{g,e}(D_i)'\gamma\right]^2,
```

```math
\widehat{m}_{0,g,e}^{*(b)}
=\frac{
\sum_{i\in C_{g,e}}M_i^{(b)}\Delta Y_{ic}
}{\sum_{i\in C_{g,e}}M_i^{(b)}},
\qquad
\widehat{\tau}_{g,e}^{*(b)}(d)
=p_{g,e}(d)'\widehat{\gamma}_{g,e}^{*(b)}
-\widehat{m}_{0,g,e}^{*(b)}.
```

This decomposition is equivalent to the original treatment-interacted regression applied to duplicated comparison rows. The control mean is subtracted once. Both outcome effects and cohort probabilities use the same sampled stores, retaining their covariance and their full nonlinear product without a delta-method or influence-function approximation.

#### Full and Available-Cohort Aggregation

Write $`K_h(D_i-d)`$ for the Gaussian kernel contribution at the fixed bandwidth; its common normalization cancels in the probabilities. Under the full retained-cohort policy, the draw's aggregation is

```math
\widehat{\pi}_{g,e}^{*(b)}(d)
=\frac{
\sum_{i\in T_{g,e}}M_i^{(b)}K_h(D_i-d)
}{
\sum_{i\in\mathcal{I}_e}M_i^{(b)}K_h(D_i-d)
},
\qquad
\widehat{\theta}^{*(b)}(d,e)
=\sum_{g\in\mathcal{R}_e}
\widehat{\pi}_{g,e}^{*(b)}(d)
\widehat{\tau}_{g,e}^{*(b)}(d).
```

Under the available-cohort policy, let $`\mathcal{A}_e^{(b)}\subseteq\mathcal{R}_e`$ contain the positive-treated-mass cohorts with estimable comparisons in draw $`b`$. Their kernel masses and probabilities are

```math
q_{g,e}^{(b)}(d)
=\sum_{i\in T_{g,e}}M_i^{(b)}K_h(D_i-d),
\qquad
\widehat{\pi}_{g,e}^{*(b),\mathrm{available}}(d)
=\frac{q_{g,e}^{(b)}(d)}{
\sum_{h\in\mathcal{A}_e^{(b)}}q_{h,e}^{(b)}(d)
}.
```

The aggregate is

```math
\widehat{\theta}_{\mathrm{available}}^{*(b)}(d,e)
=\sum_{g\in\mathcal{A}_e^{(b)}}
\widehat{\pi}_{g,e}^{*(b),\mathrm{available}}(d)
\widehat{\tau}_{g,e}^{*(b)}(d).
```

These probabilities already sum to one, so there is no additional denominator in the aggregate. A failed positive-weight cohort is removed from both the numerator and denominator of the kernel probabilities; its missing effect is not replaced with zero. With one available cohort, that cohort's ATT is the aggregate. With none, the draw is missing.

A retained cohort receiving no sampled treated stores has zero aggregate contribution and requires no comparison fit. If no cohort receives treated stores, the horizon is unavailable under either policy. If a positive-treated-mass non-reference cohort has no sampled controls or insufficient weighted polynomial rank, the full policy makes the horizon unavailable; the available-cohort policy excludes that comparison and retains other estimable cohorts. Failure at one horizon does not invalidate another horizon.

At the reference horizon, positive overall treated mass gives exact zero under either policy without sampled-control or rank checks. The original sample-size floor is not reapplied: four distinct sampled doses may identify a cubic even when the original design required five stores. Neither policy reduces the polynomial degree, replaces missing effects with zero, or redraws until estimation succeeds. Unexpected numerical failures are not ordinary sample-support failures.

Cell estimability depends on cohort and horizon, and the Gaussian kernel is positive at every finite requested dose. Availability normally agrees across doses within a horizon, while recorded draw validity and intervals remain specific to each dose–horizon pair. Mathematical kernel mass is zero only when the cohort has no sampled treated stores; numerical underflow does not justify treating a sampled cohort as absent.

The available-cohort policy changes the cohort mixture across draws. If weakly identified cohorts fail more often and have different effects, its distribution may shift away from the original point-estimate target. The fraction of original bootstrap kernel mass retained is

```math
\rho_b(d,e)
=\frac{
\sum_{g\in\mathcal{A}_e^{(b)}}q_{g,e}^{(b)}(d)
}{
\sum_{g\in\mathcal{R}_e}q_{g,e}^{(b)}(d)
}.
```

This share distinguishes negligible exclusions from exclusions carrying most of the weight. It is undefined without sampled treated mass, zero when there is treated mass but no estimable cohort, and one for a successful reference draw. It can vary with dose even when the available cohort set is unchanged. A very small positive share may underflow to zero numerically without invalidating an otherwise estimable aggregate.

#### Standard Errors, Intervals, and Limitations

Let $`\mathcal{B}_{d,e}`$ be replication indices with a finite aggregate at dose $`d`$ and horizon $`e`$, and let $`B_{d,e}=|\mathcal{B}_{d,e}|`$. All successful values are used, without an additional minimum success-rate threshold. Let $`\widehat{\theta}^{*(b)}`$ denote the aggregate under the selected policy. For $`B_{d,e}\ge2`$, its bootstrap mean and standard error are

```math
\overline{\theta}_e^*(d)
=\frac{1}{B_{d,e}}\sum_{b\in\mathcal{B}_{d,e}}\widehat{\theta}^{*(b)}(d,e),
\qquad
\widehat{\mathrm{se}}(d,e)
=\left[
\frac{1}{B_{d,e}-1}
\sum_{b\in\mathcal{B}_{d,e}}
\left(\widehat{\theta}^{*(b)}(d,e)-\overline{\theta}_e^*(d)\right)^2
\right]^{1/2}.
```

For confidence level $`\ell`$, set $`\alpha=1-\ell`$. Let $`\widehat{Q}_{d,e}(a)`$ be the empirical $`a`$-quantile of successful estimates, using linear interpolation. The percentile interval is

```math
\left[
\widehat{Q}_{d,e}(\alpha/2),
\widehat{Q}_{d,e}(1-\alpha/2)
\right].
```

With fewer than two successful draws, the standard error and interval bounds are undefined, including at the reference horizon. With at least two successful reference draws, they are all exactly zero. The original point estimate remains the reported estimate; it is not replaced by the successful-draw mean.

Discarding failed replications restricts the bootstrap distribution to successful estimation. Available-cohort aggregation additionally permits different mixtures across successful draws. Neither restriction establishes nominal or conditional frequentist coverage, and a high success rate is not a coverage guarantee. Independent stores, adequate information in retained comparisons, and the identifying assumptions remain necessary. KDE smoothing bias is ignored; bandwidth and polynomial specification are fixed. Intervals are pointwise, not simultaneous across doses and horizons.

## Implementation Details

### Module Scope and Usage

The self-contained `att.py` module exposes one class and five functions: `ATT`, `estimate_h0()`, `estimate_h_cv()`, `estimate_pi()`, `evaluate_theta()`, and `bootstrap_theta()`. The module supports separate polynomial dose curves, cohort aggregation, and pointwise bootstrap intervals. It does not provide covariates, spline knots, treatment-slope estimates, changing-dose treatment paths, simultaneous confidence bands, or plotting.

The frequency is fixed at one calendar day. Input and output period labels are timezone-naive `datetime64[ns]`, normalized to the start of each period. Time-of-day components are ignored. Event-time offsets, isolation bounds, and hours removed per period consequently have a daily interpretation. There is no frequency parameter or support for monthly or annual inputs.

`outcomes` and `events` are the input DataFrames, and the four column-name variables identify their columns. They have no default names.

```python
from labor_cost_optimization import att

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
    degree=3,
    min_cohort_size=None,
)
model.fit(outcomes, events, isolation_window=(-7, 14))

unpooled = model.predict(doses=[4, 6, 8, 10, 12])

theta = att.evaluate_theta(
    model,
    doses=[4, 6, 8, 10, 12],
    bandwidth=bandwidth,
    event_times=[-7, -2, -1, 0, 7, 14],
)

summary, draws, diagnostics = att.bootstrap_theta(
    model,
    doses=[4, 6, 8, 10, 12],
    outcomes=outcomes,
    bandwidth=bandwidth,
    event_times=[-7, -2, -1, 0, 7, 14],
    n_bootstrap=100,
    confidence_level=0.95,
    random_state=42,
    failure_policy="available_cohorts",
)
```

The isolation window is an example, not a recommended empirical choice. It fits all 22 integer horizons from -7 through 14, including the reference period -1. Prediction and aggregation can request subsets afterward.

The bandwidth example tunes on every selected first event in the raw event table. It does not automatically tune on the isolated and endpoint-observed sample retained by the fitted model. Use the bandwidth helpers' `eligible_firms` argument when a particular estimation population is intended. Bandwidths have the same units as doses, hours in this application.

### Input Tables and Column Mappings

`ATT.fit()` accepts two pandas DataFrames. The constructor's four column-name arguments are required keyword-only strings, without inferred names or fallback defaults.

| Table | Column Parameter | Required Values |
| --- | --- | --- |
| Outcomes | `firm_col` | Nonmissing, hashable store identifiers |
| Outcomes | `period_col` | Timezone-naive `datetime64[ns]` values without `NaT` |
| Outcomes | `outcome_col` | Finite numeric sales on every present row; observed zero is allowed |
| Events | `firm_col` | Identifiers of stores present in the outcome table |
| Events | `period_col` | Termination periods with the same dtype requirements as outcome periods |
| Events | `dose_col` | Positive finite numeric scheduled hours removed per period |

`firm_col` and `period_col` name columns in both tables. Rename columns before calling the module if the source tables use different labels. Identifier values refer to the same stores across tables and retain their original types; the module does not coerce them to strings. No employee, spell, or event identifier is required or generated.

Mapped names must be nonempty and distinct within each table, and DataFrame column labels must be unique. `outcome_col` and `dose_col` may share a name because they belong to different tables. Extra columns are ignored. Missing column-name arguments raise `TypeError`; invalid names, schemas, or values raise `ValueError`.

Period strings, object-dtype period columns, timezone-aware datetimes, and pandas Period values are rejected rather than converted silently. The module normalizes copies of the datetime columns and leaves both input DataFrames unchanged. The outcome table must be unique by store and normalized period. Two rows differing only by time of day therefore produce a duplicate-store-period error; they are not summed or reduced to one row.

Missing store-period rows are allowed, including inactivity and permanent closure. Every present row needs a finite, nonmissing outcome. An absent row is not zero. The module neither fills gaps nor substitutes a nearby observation for a required endpoint. A comparison needs its exact baseline and endpoint rows; missing intermediate periods alone do not invalidate it.

The overall screening interval is inferred from the earliest and latest normalized outcome periods across the complete panel. Calendar gaps are preserved. It is not inferred separately per store, from termination periods, or by ranking observed labels. All supplied events enter first-event selection and history screening, including events outside the panel, during missing-outcome periods, or subsequently excluded from estimation. Extra event records do not extend the screening interval.

The event table must record every termination of every outcome store during the overall panel interval, including periods of inactivity or closure. Completeness is a data assumption, not something the module can establish from observed records. Stores without terminations need no event row; zero-dose placeholder events are invalid. First-treatment control eligibility also requires enough earlier history to identify treatment entry, or a justified untreated-at-panel-start assumption. A store with no supplied event is treated as having no observed treatment entry; the module cannot verify that it was never treated before the panel.

### First Events and Termination History

Only the first termination of each store is a target candidate. Selection uses the full supplied event history after validation and period normalization, before isolation, endpoint-availability, or sample-size screening. A later event is never promoted when the first candidate is excluded.

Selection uses a stable sort by normalized period and retains the entire first row, including its dose. Ties at the earliest period use original input order. Time-of-day components do not break ties. Doses are neither averaged nor summed across event rows.

Every input row is treated as one actual termination. Multiple rows at the same store and normalized period count as simultaneous terminations, even when their doses match. If the earliest period has several records, the selected candidate is marked `overlapping_termination` and excluded. If those records are duplicate copies of the same underlying termination, remove the copies upstream; employee identifiers are unavailable to distinguish them automatically.

`events_` contains one first-event row per represented store, including target-ineligible candidates; control timing uses all of those rows. `termination_history_` retains all distinct store-period combinations and their record counts for target isolation. Once a store has its first termination, it cannot serve as a control at that period or later, regardless of later terminations or target exclusions.

`duplicate_diagnostics_` reports `n_dropped_events`, the input event-row count minus the number of represented stores, and `n_affected_firms`, the number of stores with more than one event row. For example, three rows for one store and two for another mean three additional target rows removed across two stores. These counts include later and simultaneous records; they do not imply erroneous source duplicates. They also appear in `drop_summary_` as `duplicate_event_rows` and `duplicate_event_firms`.

### Comparison With contdid Before Aggregation

The module implements the parametric dose-specific ATT calculation with additional sample restrictions and prediction conventions. It is not a strict reproduction of `contdid`, even when each store has only one termination. A single event removes duplicate-selection and overlap issues, but does not remove the full-window panel-boundary screen, endpoint-availability restrictions, or comparison-level design screens.

This comparison refers to the R package source at [commit `fe6f2cf`](https://github.com/bcallaway11/contdid/tree/fe6f2cffdf6624e11d9cf2bbed62c3a3a0e3f3f2), inspected on 2026-11-10. The relevant routine is [`cont_did_acrt`](https://github.com/bcallaway11/contdid/blob/fe6f2cffdf6624e11d9cf2bbed62c3a3a0e3f3f2/R/cont_did.R#L410), which computes dose-specific level effects as well as slopes. The package's level event-study path uses a separate routine for cohort effects averaged over doses; it is not the reference for this module's dose-specific component curves.

| Aspect | This Module | Inspected contdid Implementation |
| --- | --- | --- |
| Cell regression | Separate equal-weight PyFixest regression with a control intercept and treatment-interacted polynomial | Treated-only B-spline OLS followed by subtraction of the control mean; algebraically equivalent on matched samples with the same degree, no internal knots, and a consistently reused basis |
| Target panel boundaries | The entire `isolation_window` must lie within the overall outcome panel bounds | No corresponding requirement that every target have this module's complete isolation window within the panel bounds |
| Missing periods | A store can enter a comparison whenever its exact baseline and endpoint outcomes are observed; missing intervening periods are allowed | The public routine does not support `allow_unbalanced_panel=TRUE`; it does not offer this module's endpoint-specific unbalanced-panel design |
| Small samples and dose support | Requires at least `degree + 2` treated stores, at least `degree + 1` distinct doses, and full rank; cubic fits require five treated stores | The cell regression does not implement these explicit screening rules; its fitting and preprocessing behavior should not be assumed identical |
| Post-treatment controls | Not-yet-treated stores at the outcome period, including never-treated stores, with both outcomes observed | The same treatment-timing cutoff under `control_group="notyettreated"`; actual samples agree only when outcome availability and preprocessing also agree |
| Pre-treatment baseline | Fixed at the period immediately before the focal cohort's event; horizon -1 is normalized to zero | Defaults to `base_period="varying"`, using the period immediately before each pre-treatment endpoint |
| Pre-treatment controls | Stores first treated after the focal cohort, including never-treated stores | The not-yet-treated subset uses stores first treated after the comparison endpoint, excluding the focal cohort from controls, and includes never-treated stores |
| Prediction basis | Reuses each cell's fitted dose bounds and degree for every prediction request | Constructs a fresh B-spline basis from `dvals` without explicitly reusing the training boundary knots; different grid boundaries can change the reported dose-specific curve |

For example, with `isolation_window=(-7, 14)`, a termination two periods before the panel ends is excluded by this module even if its event-period outcome and baseline are observed. Having only one termination at that store does not change the exclusion. Missing outcomes can change both comparison samples, while treated-design screens determine which cells are estimated. Screened stores can still serve as controls when treatment timing and observed endpoints permit. Identical regression formulas alone do not establish equal ATT estimates.

The [R subset function](https://github.com/bcallaway11/contdid/blob/fe6f2cffdf6624e11d9cf2bbed62c3a3a0e3f3f2/R/cont_did.R#L544) defines its baseline and not-yet-treated comparisons. The [R prediction code](https://github.com/bcallaway11/contdid/blob/fe6f2cffdf6624e11d9cf2bbed62c3a3a0e3f3f2/R/cont_did.R#L433) constructs the training and prediction bases in separate calls. This module's saved transformation makes prediction at dose 8 identical whether 8 is requested alone or with other doses. The regression equivalence in [Polynomial Comparison Estimation](#polynomial-comparison-estimation) concerns predictions from a consistently evaluated fitted curve, not a freshly reparameterized prediction grid. These distinctions apply before cohort aggregation or bootstrap inference.

### ATT Construction, Fitting, and Prediction

```python
ATT(
    *,
    firm_col,
    period_col,
    outcome_col,
    dose_col,
    degree=3,
    min_cohort_size=None,
)

ATT.fit(outcomes, events, *, isolation_window) -> ATT

ATT.predict(doses, *, event_times=None) -> pandas.DataFrame
```

| Parameter | Meaning and Validation |
| --- | --- |
| `firm_col`, `period_col`, `outcome_col`, `dose_col` | Required column mappings described above |
| `degree` | Positive integer polynomial degree; default 3 gives cubic curves |
| `min_cohort_size` | `None` or a positive integer; the effective treated-sample minimum is at least `degree + 2` |
| `isolation_window` | Required pair of integer signed offsets; lower bound at most -1, upper bound at least 0; both endpoints included |
| `doses` | Nonempty iterable of distinct positive finite real doses |
| `event_times` | `None` for every fitted horizon, or a nonempty iterable of distinct integer horizons within the fitted window |

The lower isolation bound includes the baseline. Positive lower bounds are rejected rather than negated implicitly. `fit()` has no `event_times` argument: every integer horizon in the isolation window is screened and estimated. A later prediction request outside that interval raises an error; it does not fit additional comparisons.

NumPy integer and floating-point scalars are accepted where appropriate. Numeric strings, complex values, and Booleans are rejected, including Boolean elements mixed into numeric grids. Iterables such as generators are consumed once into validated collections. Pydantic validation errors are subclasses of `ValueError` and include field or element locations.

`fit()` returns `self` and replaces prior fitted state. Any failed refit invalidates previous predictions. Calling `predict()` without a successful current fit raises `RuntimeError`. Invalid prediction requests or nonfinite polynomial predictions raise `ValueError`.

Each non-reference cohort–horizon comparison has a separate PyFixest OLS model and its own polynomial coefficients. There is no pooling across cohorts or horizons. Treated and control rows have equal weights within an original comparison. Target samples follow the isolation and exact-endpoint rules in the theoretical section. Controls require their first termination strictly after the later of the focal cohort period and comparison endpoint, plus both observed outcomes; their own target isolation eligibility is not required. One store contributes at most once as treated and at most once as control within a comparison, and cannot be both in that comparison. It can appear in other comparisons when eligible.

Every comparison saves the polynomial transformation estimated from its own endpoint-observed treated doses. Predictions reuse its `lower`, `upper`, and `degree` values. Requesting dose 8 alone or alongside other doses produces the same prediction at 8. Different horizons may save different transformations because their observed samples differ.

Non-reference predictions outside a comparison's observed dose range use polynomial continuation and are flagged as extrapolated. The module does not clip doses, omit cohorts at particular doses, or renormalize weights because of extrapolation. Reference values are exact zero normalizations, with `extrapolated=False`.

#### Comparison Screening and Fit Failures

With `degree=3` and `min_cohort_size=None`, each treated comparison needs at least five observed stores. A smaller explicit minimum cannot reduce the design floor; a larger one increases it. The resolved value is stored in `min_cohort_size_`. `None` does not disable screening, and passing the minimum does not guarantee precision.

A treated comparison also needs at least `degree + 1` distinct doses and full column rank of its saved polynomial design. The numerical rank criterion uses NumPy's standard SVD tolerance. Each store counts once; extra outcome rows, other cohorts, control observations, and further terminations cannot satisfy a treated-sample requirement. At least one eligible control is needed; a singleton control group can supply a point estimate.

Sample-size, dose-support, and rank screens drop only the affected cohort–horizon comparison. A cohort can remain usable at other horizons. Screened stores retain their first-event status and may serve as controls for earlier cohorts only while satisfying the first-treatment cutoff and endpoint requirements. The polynomial degree is not reduced to rescue a comparison.

| Status or Reason | Meaning |
| --- | --- |
| `eligible` | The selected first event passes design eligibility |
| `outside_outcome_period` | The selected first event lies outside the overall outcome period range |
| `isolation_outside_panel` | The event is in range, but its complete isolation interval extends outside the panel bounds |
| `overlapping_termination` | Another termination, including a simultaneous record, lies in the isolation interval |
| `missing_baseline`, `missing_endpoint` | A design-eligible store lacks a required exact outcome row at this horizon |
| `no_observed_treated` | No treated store has both required outcome rows in the comparison |
| `small_sample` | The observed treated sample is below `min_cohort_size_` |
| `insufficient_doses` | The treated sample has fewer than `degree + 1` distinct doses |
| `rank_deficient` | The saved treated polynomial design is not full column rank |
| `no_controls` | A nonempty treated design passes screening but lacks endpoint-observed eligible controls |
| `fitted` | A non-reference comparison was fitted successfully |
| `reference` | The reference comparison passed bookkeeping checks and has exact zero coefficients |
| `model_failed` | A backend fitting or coefficient-validation failure invalidated the fit |

Reference cells undergo the same original treated-sample, dose-support, rank, and control-availability checks. A successful reference cell saves its basis, counts, and zero coefficients without fitting a PyFixest model. An unavailable reference cell does not receive a manufactured zero.

If no non-reference cell passes all required checks, fitting fails; reference cells alone do not constitute a successful fit. A backend fitting error, nonfinite coefficient, or unexpected dropped coefficient invalidates the entire fit rather than silently removing the failed cell or accepting a lower-dimensional curve. Diagnostic information already computed remains available, but failed cells never receive zero effects. Counts for processing stages not reached are not completed-screening results.

#### Prediction Output

`predict()` returns one row per available `(cohort, event_time, dose)`, sorted by those columns.

| Column | Meaning |
| --- | --- |
| `cohort` | Normalized termination period, dtype `datetime64[ns]` |
| `period` | Normalized comparison endpoint period, dtype `datetime64[ns]` |
| `event_time` | Integer periods since termination |
| `dose` | Requested dose |
| `att` | Component estimate at that dose and horizon |
| `estimate_type` | `att` for nonnegative horizons, `placebo` below -1, or `reference` at -1 |
| `n_treated`, `n_controls` | Numbers of stores in the original comparison; `n_treated` refers to the future termination cohort even before treatment, and `n_controls` can change across horizons as controls become treated or endpoint availability changes |
| `extrapolated` | Whether a non-reference prediction lies outside that cell's observed dose support |

Unavailable or screened comparisons are omitted, with diagnostics and aggregate warnings where applicable. Ordinary outcome-availability exclusions are recorded separately. Every successful reference prediction has `att=0`.

### Fitted State and Diagnostics

Public fitted attributes provide the selected samples, saved transformations, models, and exclusion information. Canonical output names are independent of the caller's input-column names.

| Attribute | Contents |
| --- | --- |
| `column_names_` | The four supplied input-column mappings |
| `min_cohort_size_` | Resolved treated-sample minimum including the design floor |
| `events_` | `firm_id`, normalized datetime `cohort`, `dose`, and `design_eligible`; one selected first-event row per store, including excluded candidates |
| `termination_history_` | `firm_id`, normalized datetime `cohort`, and `n_terminations`; unique by store and period, retaining all supplied history |
| `duplicate_diagnostics_` | `n_dropped_events` and `n_affected_firms` from first-event selection |
| `drop_summary_` | Aggregate counts and reason dictionaries for the latest fit attempt |
| `event_diagnostics_` | Store-level design eligibility, exclusion reasons, and relevant overlap information |
| `availability_diagnostics_` | Store–horizon endpoint availability, missing-row reasons, and exact baseline/endpoint periods |
| `cohort_diagnostics_` | Cohort summaries of design-eligible events and available, retained, or failed comparisons |
| `cell_diagnostics_` | Comparison statuses, treated-design checks, control counts, and exact contributing treated/control store identifiers |
| `event_times_` | Every integer horizon in the fitted isolation interval, including -1 |
| `origin_period_`, `last_period_`, `last_period_index_` | Overall normalized panel bounds and final internal elapsed-period index |
| `isolation_window_` | Fitted signed offset pair |
| `retained_cohorts_` | Horizon-to-cohort mapping for nonempty treated designs passing that horizon's screen; missing controls do not remove a cohort from this mapping |
| `models_` | PyFixest `Feols` objects keyed by `(cohort, event_time)` for fitted non-reference cells |
| `bases_` | Saved `lower`, `upper`, and `degree` per successful comparison, including reference cells |
| `cell_keys_` | Mapping from internal comparison labels to `(cohort, event_time)`; these labels are not additional event identifiers |
| `effect_columns_` | Ordered treatment-intercept and polynomial coefficient names for each fitted non-reference cell |
| `cell_coefficients_` | ATT coefficient vectors by comparison; successful reference cells contain exact zero vectors |
| `bootstrap_firms_` | Complete original outcome-store index, including control-only and unused stores |
| `outcome_fingerprint_` | SHA-256 fingerprint of the canonical complete outcome panel, used to verify bootstrap inputs without storing another panel copy |

`drop_summary_` separates different stages and counting units:

| Field | Meaning |
| --- | --- |
| `duplicate_event_rows` | Additional event rows removed from targets while remaining in history |
| `duplicate_event_firms` | Stores with more than one input event row |
| `excluded_first_events` | Design-exclusion reason-to-count mapping; each excluded store contributes once |
| `unavailable_treated_rows` | Design-eligible store–horizon pairs missing at least one endpoint; each pair counts once even when both rows are absent |
| `dropped_treated_rows` | Observed treated-store counts summed over rejected cells, including `no_controls`; control rows are not counted |
| `dropped_cells` | Exclusion reason-to-count mapping for rejected cohort–horizon comparisons |
| `dropped_cohorts` | Number of distinct cohort periods represented by rejected cells; they can remain estimable at other horizons |

The two treated-row fields include all fitted horizons, including the reference. They count comparison rows, not distinct stores or raw outcome rows. An empty treated cell adds to `dropped_cells` but contributes no `dropped_treated_rows`. A missing baseline may make one store unavailable at several horizons. Counts from these different stages cannot be summed into a single number of excluded stores. Prediction, aggregation, and bootstrap calls leave this fit summary unchanged.

### Conditional Cohort Weights

```python
estimate_pi(
    events,
    doses,
    *,
    eligible_firms,
    bandwidth,
) -> pandas.DataFrame
```

This standalone function accepts canonical event metadata with columns `firm_id`, normalized datetime `cohort`, and positive `dose`, containing one already-selected first-event row per store. It can receive `model.events_` or a separately constructed table with the same schema. It does not perform first-event selection or other preprocessing again.

`eligible_firms` is required: a nonempty collection of unique store identifiers, all present in `events`. It identifies the exact endpoint-observed stores in retained comparisons at the intended horizon. Cohort labels alone are insufficient because isolation or missing endpoints can exclude individual stores within a cohort. The function restricts kernel contributions and normalization to the supplied stores, counting each once.

`doses` is a nonempty collection of distinct positive finite values. `bandwidth` is positive and finite in dose units. Invalid schemas, identifiers, doses, bandwidths, or nonfinite normalization results raise `ValueError`; invalid normalization does not trigger uniform fallback weights.

The result contains `cohort`, `dose`, and `weight`, sorted by cohort and dose. Weights are nonnegative and sum to one at each requested dose. No event-time argument is needed: horizon dependence enters through `eligible_firms`. For example, after constructing the intended retained-store set:

```python
weights = att.estimate_pi(
    model.events_,
    doses=[4, 6, 8, 10, 12],
    eligible_firms=eligible_firms,
    bandwidth=bandwidth,
)
```

`evaluate_theta()` constructs the appropriate retained-store set automatically when evaluating the fitted event study.

### Bandwidth Selection

```python
estimate_h0(
    events,
    *,
    firm_col,
    period_col,
    dose_col,
    eligible_firms=None,
) -> float

estimate_h_cv(
    events,
    *,
    firm_col,
    period_col,
    dose_col,
    eligible_firms=None,
    bandwidths=None,
    batch_size=256,
    max_expansions=50,
) -> tuple[float, pandas.DataFrame]
```

Both helpers accept raw event records and required column mappings, independently of `ATT` and the outcome table. They require nonempty distinct mapped names, unique DataFrame labels, nonmissing hashable identifiers, positive finite numeric doses, and timezone-naive `datetime64[ns]` periods. Input tables are unchanged.

Each helper normalizes periods and selects the first entire event row per store by stable period order, using original order for same-period ties. An optional `eligible_firms` filter is applied only after this selection. A supplied subset must be nonempty, contain unique valid identifiers, and refer only to stores present in the event table. `None` uses every selected first event. Neither filtering nor repeated rows can promote a later event or give a store several contributions.

These helpers do not apply isolation, outcome-availability, or regression-design screens. In particular, they select a stable first row among simultaneous events without applying `ATT`'s overlap exclusion. A supplied subset is therefore necessary when tuning on an externally selected population. The resulting bandwidth may be reused across horizons, but its score describes only that supplied population; it does not average validation risks across horizons or refit outcome models.

`estimate_h0()` returns the positive finite dose-scale rule given in the theoretical section. It requires at least two selected stores and positive dose variation, but does not require multiple cohorts. Repeated doses are allowed. If the interquartile range is zero, it uses the sample-standard-deviation fallback; constant doses still raise `ValueError`.

`estimate_h_cv()` minimizes the leave-one-store-out multiclass Brier criterion described above. It requires at least two selected stores, at least two represented cohorts, and positive dose variation, even when an explicit candidate grid is supplied. Singleton cohorts remain in the score. Store outcomes and fitted ATT estimates are not inputs.

| Option | Behavior |
| --- | --- |
| `bandwidths=None` | Nine initial candidates from `h0 * numpy.geomspace(0.1, 10.0, 9)` on the selected population |
| Explicit `bandwidths` | Nonempty iterable of distinct positive finite initial candidates; these are not hard search limits |
| `batch_size` | Positive integer, default 256; controls the number of unique query-dose rows processed together without changing the criterion |
| `max_expansions` | Positive integer, default 50; limits extra evaluations in each search direction separately |

If an initial minimizing candidate lies on the lower boundary, the search repeatedly halves that boundary. If a minimizer lies on the upper boundary, it repeatedly doubles that boundary. The directions are fixed by the initial minimizers. A tie including both boundaries, including a single-candidate grid, starts both directions independently; a unique interior minimizer starts neither.

Each direction stops at its first meaningful score increase, retaining that candidate in the returned table. The numerical tolerance and plateau criterion are defined in the theoretical section. Three consecutive changes within tolerance stop a direction as a numerical plateau; a meaningful decrease resets that count. Search also stops at `max_expansions`, before a nonpositive or nonfinite candidate, or before squared scaled dose distances would overflow. A plateau, expansion limit, or numerical-range limit does not prove the score has turned upward.

The function returns `(bandwidth, scores)`. `scores` has `bandwidth` and `cv_brier` columns, sorted by bandwidth, with each evaluated candidate once, including expansion points and the first meaningful rise. The returned bandwidth is the smallest exact minimizer among all evaluated candidates. The search does not fill gaps between candidates or continue past a rise; it is not a global optimizer over all positive bandwidths.

The criterion is exact despite batching, including repeated-dose multiplicities and removal of only the held-out store's own contribution. Runtime can remain quadratic when all doses are distinct; memory also depends on the numbers of distinct doses and cohorts. A different `batch_size` is a computational choice, not an approximation or a change to the statistical target.

### Dose-Specific Event-Study Aggregation

```python
evaluate_theta(
    model,
    doses,
    *,
    bandwidth,
    event_times=None,
) -> pandas.DataFrame
```

`model` must be successfully fitted. `doses` and `event_times` follow the prediction rules, and `bandwidth` must be positive and finite. `None` requests the entire fitted horizon interval, including negative horizons and -1.

The function uses exact endpoint-observed stores in the horizon's retained cohorts and the conditional weights specified in the theoretical section. It only aggregates existing estimates: it does not refit curves, change the isolation window, alter endpoint samples, or rerun screening. Changing bandwidth changes cohort weights without changing ATT models. Dose remains fixed within each aggregate; only cohorts are averaged.

An empty eligible retained-store set raises an error. Every retained cohort needs a fitted component or successfully normalized reference cell. In particular, a missing control group raises an error rather than being silently discarded and followed by renormalization. Predictions and weights must match completely by cohort and dose; unknown effects are not filled with zero. These checks apply to post-treatment, placebo, and reference horizons alike.

The result is sorted by `event_time` and `dose`:

| Column | Meaning |
| --- | --- |
| `event_time`, `dose` | Requested horizon and dose |
| `theta` | Cohort-aggregated component estimate |
| `estimate_type` | `att`, `placebo`, or `reference`, following prediction conventions |
| `n_cohorts_eligible`, `n_cohorts_used` | Endpoint-observed cohorts before comparison screening, and retained cohorts used in the aggregate |
| `n_firms_design` | Stores with design-eligible first events, including those missing endpoints at this horizon |
| `n_firms_eligible`, `n_firms_used` | Endpoint-observed stores before comparison screening, and stores in retained comparisons |
| `availability_restricted` | Some design-eligible stores lack a required baseline or endpoint row |
| `restricted_target` | At least one nonempty eligible cohort was excluded by a sample-size, dose-support, or rank screen |
| `extrapolated` | At least one contributing non-reference prediction is outside its cell's observed dose support |

Isolation restrictions always define the target and remain visible in fit diagnostics. Panel-boundary exclusions have already occurred before these availability and comparison-screening counts. Store counts and selected-event counts coincide. Every successful reference aggregate is exactly zero and has `extrapolated=False`.

### Pointwise Bootstrap Intervals

```python
bootstrap_theta(
    model,
    doses,
    *,
    outcomes,
    bandwidth,
    event_times=None,
    n_bootstrap=100,
    confidence_level=0.95,
    random_state=None,
    failure_policy="available_cohorts",
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
```

The function returns `(summary, draws, diagnostics)` without changing the fitted model or supplied outcomes. It resamples the complete original store population and recomputes component estimates and cohort weights using the shared sampling counts, as described in the theoretical section. The original point estimates in `summary` equal `evaluate_theta()` at the supplied grid and bandwidth under either failure policy.

| Parameter | Meaning and Validation |
| --- | --- |
| `model` | Successfully fitted `ATT` with original-store metadata and outcome fingerprint |
| `doses` | Nonempty iterable of distinct positive finite doses |
| `outcomes` | Complete original outcome table, using the fitted model's column mappings |
| `bandwidth` | Fixed positive finite kernel bandwidth in dose units |
| `event_times` | Distinct fitted integer horizons, or `None` for the full fitted interval |
| `n_bootstrap` | Integer number of attempted draws, at least 2; default 100 |
| `confidence_level` | Finite nominal interval level strictly between zero and one; default 0.95 |
| `random_state` | Nonnegative integer seed or `None` for nondeterministic local sampling |
| `failure_policy` | `"available_cohorts"` by default, or `"strict"` |

The complete normalized store-period observations and numeric outcomes must match the original fit, including stores or rows unused by the requested comparisons. Row order, ignored time-of-day components, and extra unmapped columns may differ. Matching original coefficients alone is insufficient. `outcome_fingerprint_` and `bootstrap_firms_` enforce this contract; models lacking that metadata require refitting. The event table is not supplied again.

Invalid inputs or an unavailable required original comparison raise before resampling. The original retained cohorts, exact treated/control memberships, polynomial specifications and saved transformations, requested grid, and bandwidth remain fixed. First-event selection, isolation, endpoint screening, bandwidth tuning, and the original `min_cohort_size` screen are not repeated inside draws. The bootstrap still requires an identifiable weighted polynomial fit when a non-reference cell has positive sampled treated mass.

#### Failure Policies

Draw availability is evaluated by horizon and ultimately retained separately for every requested dose–horizon pair. A failure elsewhere does not remove a finite available value.

| Condition | `strict` | `available_cohorts` |
| --- | --- | --- |
| One cohort has no sampled treated stores, but another has positive treated mass | Zero contribution from that cohort; no ATT or sampled-control requirement | Same |
| No retained cohort has sampled treated stores | Missing horizon estimate, reason `no_treated_firms` | Same |
| Positive treated mass but no sampled controls in a non-reference cell | Missing horizon estimate; record `no_controls` for the failed cohort | Exclude that cohort and record `no_controls`; use other estimable cohorts |
| Positive treated mass but insufficient weighted polynomial rank | Missing horizon estimate; record `rank_deficient` | Exclude that cohort and record `rank_deficient`; use other estimable cohorts |
| Reference horizon -1 with positive overall treated mass | Exact zero without sampled-control or rank checks | Same |

Under `strict`, one failed positive-mass non-reference cell invalidates that horizon's draw at every requested dose. Under `available_cohorts`, weights are normalized over the remaining estimable positive-mass cohorts. If none remain despite positive treated mass, the horizon is missing with reason `no_available_cohorts`. Failed positive-weight effects are never replaced by zero. Neither policy lowers the degree or draws replacements for failures.

A cubic polynomial with four distinct sampled doses can be estimable even when original fitting required five stores. Unexpected numerical failures, nonfinite arithmetic, and implementation errors raise exceptions rather than being reported as ordinary support failures.

Intervals and standard errors use the available finite values for each dose–horizon pair. There is no minimum success-rate threshold. Fewer than two successful draws leave both interval bounds and the standard error missing, including at -1. With at least two successful reference draws, all three values are exactly zero. Original point estimates are always retained rather than replaced by bootstrap means. The theoretical section describes the limitations of success-restricted intervals and changing available-cohort mixtures.

#### Bootstrap Outputs

`summary` is sorted by `event_time` and `dose`. It contains every original `evaluate_theta()` column, preserving point estimates and sample/support flags, together with:

| Column | Meaning |
| --- | --- |
| `standard_error` | Sample standard deviation of successful draws, using `ddof=1` |
| `ci_lower`, `ci_upper` | Successful-draw percentile interval bounds using linear quantile interpolation |
| `confidence_level` | Requested nominal interval level |
| `failure_policy` | Selected policy |
| `n_bootstrap` | Total attempted replications, including failures |
| `n_success`, `n_failed` | Successful and failed values at this dose–horizon pair; their sum is `n_bootstrap` |
| `success_rate` | `n_success / n_bootstrap` |
| `n_partial` | Successful draws excluding at least one failed positive-mass cohort |
| `mean_retained_weight_share`, `min_retained_weight_share` | Mean and minimum original bootstrap kernel-mass share retained across successful draws; missing when none succeed |
| `conditional_on_success` | True if at least one replication fails at this pair |
| `inference_status` | `insufficient_successful_draws` if fewer than two succeed; otherwise `available_cohorts` if `n_partial > 0`, `successful_draws_only` if some draws fail, or `ok` |

`draws` has columns `replicate`, `event_time`, `dose`, `theta`, `success`, `n_cohorts_available`, `n_cohorts_failed`, and `retained_weight_share`. It is sorted by replication, horizon, and dose. Replications run from 1 through `n_bootstrap`. Every requested row is retained, including missing estimates as `NaN` with `success=False`.

Available and failed cohort counts concern originally retained cohorts with positive sampled treated mass. A zero-mass cohort counts as neither available nor failed. `retained_weight_share` is the available cohorts' share of the original bootstrap kernel mass: zero when positive treated mass exists but no cohort is available, and `NaN` when there is no treated mass. A successful reference draw has share one. Extremely small shares can underflow to zero without making an otherwise estimable aggregate unavailable. These diagnostics remain recorded even when `strict` rejects the aggregate. `n_partial` describes cohort exclusions in successful draws; `conditional_on_success` describes missing whole draw values.

`diagnostics` has columns `cohort`, `event_time`, `reason`, and `n_failures`, aggregated by comparison and failure reason. `cohort` has dtype `datetime64[ns]`. Horizon-level `no_treated_firms` and `no_available_cohorts` use `NaT`; the latter occurs only under `available_cohorts`. Failures of excluded cells are recorded even if other cohorts produce a successful aggregate. Several comparisons may fail in one replication, so comparison failure counts do not generally sum to `n_failed`. A clean run returns an empty table with this schema.

### Logging, Errors, and Computational Limits

Nonfatal diagnostics use `logging.warning()`, with logging configuration left to the caller. Validation errors and required-model failures remain exceptions.

| Operation | Warning Behavior |
| --- | --- |
| `fit()` | One aggregate warning for event selection, design exclusions, missing endpoints, and dropped comparisons; no per-cohort or per-cell warnings |
| `predict()` | Aggregate warnings for omitted unavailable comparisons and for extrapolated predictions |
| `estimate_h_cv()` | At most one warning combining singleton-cohort counts, a boundary optimum, exact ties, and plateau/expansion/numerical-range stops |
| `evaluate_theta()` | At most one warning across the requested grid for omitted comparisons, restricted targets, and extrapolated predictions; internal prediction is quiet |
| `bootstrap_theta()` | At most one warning combining original aggregation restrictions/extrapolation, failed draw values, excluded positive-mass cohorts, and insufficient successful draws; internal point aggregation is quiet |

Fit warnings report aggregate counts and reasons and direct callers to the stored diagnostics. A clean fit emits no exclusion warning. A clean CV search has an interior unique optimum, no singleton cohorts, and no exceptional stopping condition. Expansion ending at a meaningful rise is not itself exceptional.

Aggregation warnings count omitted comparisons and excluded observed treated store–horizon rows once per horizon, independently of dose-grid size. Extrapolation counts refer to contributing cohort–horizon–dose predictions. Reference rows never extrapolate. These warnings do not overwrite `drop_summary_`. Available-cohort bootstrap can warn even when every requested aggregate is finite, because successful draws may exclude positive-mass cohorts. Clean calls emit no warning.

Comparison selection uses shared indexed data and separate cell models, but fitting is not a streaming operation. Comparison frames, lookups, diagnostics, and retained models consume memory, and observations can be reused across comparisons. Bootstrap avoids duplicating entire panels for every draw, but its returned draw table necessarily grows with replications, horizons, and requested doses. These implementation choices preserve the estimator's mathematics; they do not establish a universal runtime or memory bound.

## References

Callaway, B., Goodman-Bacon, A. and Sant'Anna, P.H.C. (2025) *Difference-in-Differences with a Continuous Treatment*. Working paper, draft dated 31 December. Available at: [https://psantanna.com/files/CGBS_v4.pdf](https://psantanna.com/files/CGBS_v4.pdf) (Accessed: 3 November 2026).
