# ATT Codex Maintenance Notes

This file is repository-specific guidance for Codex when maintaining `src/labor_cost_optimization/att.py`. The portable mathematical and public API documentation is [att.md](att.md). Copying the estimator documentation to another project does not require this file. Read [AGENTS.md](../AGENTS.md) and its referenced rules before changing the module; the project-specific exceptions there take precedence over generic setup conventions.

## Repository Layout and Coding Conventions

Put the class, all five public functions, and a few private validation, window-selection, polynomial, bandwidth, and cell-regression helpers in the single `src/labor_cost_optimization/att.py` file. Use the `labor_cost_optimization` package under the `src` directory; keep estimator logic in that module. Use `firm` and `period` consistently in all Python identifiers, canonical column names, docstrings, and error/log messages, including private helpers. Use `period_col`, `origin_period_`, `last_period_`, `last_period_index_`, and output column `period`; preserve literal third-party API/type names such as `datetime64[ns]` and `.dt.normalize()`. The portable documentation keeps the application term “store” in explanatory prose; code examples and public schema names follow the general `firm` and `period` terminology. Use `logging.warning()` for nonfatal diagnostics and small dictionaries or records for fitted cells. External R code is not a runtime dependency. Pydantic v2 `Field` constraints, cached `TypeAdapter` instances, and `validate_call` enforce scalar and collection types and bounds; private input models use `ConfigDict(arbitrary_types_allowed=True)`. Named schema aliases and explicit adapter annotations support older Pydantic typing signatures. Preserve accepted NumPy scalar values and the public `ValueError` contract, including its `ValidationError` subclass; numerical, period, isolation, and sample checks remain explicit domain validation. These schemas stay in the same module and do not extend the public API. No generic estimator framework or serialization layer is needed. Include the package's `py.typed` marker through `[tool.setuptools.package-data]` so installed consumers can use its inline annotations.

Follow the repository rules for imports, type annotations, serialization, and American English. Keep all estimator logic in the single module. Unit-test assertions may omit messages; production assertions require messages. Do not add controls, alternative treatment histories, a generic estimator framework, or a serialization layer without a requested scope change.

## Development Environment and Commands

The project targets Python 3.11 (`>=3.11,<3.12`) and records dependencies in [pyproject.toml](../pyproject.toml); [.python-version](../.python-version) selects the interpreter. Runtime pins are `numpy==1.26.4`, `pandas==1.5.3`, `pyfixest==0.60.0`, and `scipy==1.15.2`, with `pydantic>=2,<3`. Development dependencies remain in `[dependency-groups].dev`; keep the existing uv workflow. The user has authorized Python 3.12 for local testing only, without changing project settings. Report such checks as compatibility checks, not validation of the pinned Python 3.11 environment.

Use the supplied [.pre-commit-config.yaml](../.pre-commit-config.yaml) for spelling, formatting, lint, and type checks on Python, Markdown, and notebooks. The hooks use system executables, so activate the development environment before installing or running them:

```bash
uv sync --python 3.11 --group dev
source .venv/bin/activate
pre-commit install
pre-commit run --all-files
```

## Docstring Convention

Use Google-style docstrings throughout `src/labor_cost_optimization/att.py`, including the module, `ATT`, its constructor and public methods, all five public functions, and private helpers. Start with a concise summary followed by a blank line before any extended description or sections. Use `Args:`, `Returns:`, and `Raises:` where applicable; omit empty or inapplicable sections. Document constructor arguments in `__init__` and fitted public attributes in the class docstring's `Attributes:` section. Include `Args:` whenever a function or method accepts arguments other than `self` or `cls`, `Returns:` whenever its return annotation is not `None`, and `Raises:` for explicit or relevant unhandled exceptions. A one-line helper docstring is appropriate only when none of these sections is required. Wrap references to functions, methods, attributes, and schema columns in backticks.

Indent argument names, exception classes, and attribute names four spaces relative to the section heading, each followed by a colon on its own line. Start each description on the next line, indented eight spaces, and keep wrapped description lines at that indentation. Plain prose in `Returns:` stays at four spaces. Preserve blank lines between the summary and sections and between successive sections. Do not insert a blank line between a section heading and its first entry. Exception descriptions may use plain prose for multiple related conditions, as in the example below. Document every argument except `self` and `cls`, using its exact signature name; type annotations belong in the signature. Explain returned objects and relevant DataFrame schemas, and describe the conditions for exceptions the caller needs to handle. Describe state changes such as `fit()` replacing earlier fitted state and when fitted attributes become available. Keep docstrings consistent with the implemented behavior and use `firm` and `period` throughout. Nonfatal diagnostics emitted by `logging.warning()` are not exceptions and do not belong in `Raises:`.

For example, the `estimate_pi()` docstring follows this layout:

```python
"""Estimate dose-specific cohort weights for eligible firms.

Args:
    events:
        Canonical first-event metadata with `firm_id`, `cohort`, and `dose`
        columns, containing one row per firm.
    doses:
        Distinct positive finite doses at which to evaluate weights.
    eligible_firms:
        Unique firm identifiers defining the eligible sample.
    bandwidth:
        Positive finite Gaussian-kernel bandwidth in dose units.

Returns:
    A DataFrame with `cohort`, `dose`, and `weight` columns, sorted by `cohort`
    and `dose`. Weights sum to one at each requested dose.

Raises:
    ValueError:
        If the bandwidth is nonpositive or nonfinite, the inputs
        violate the required schema or eligibility rules, or weight
        normalization produces nonfinite results.
"""
```

Do not insert blank lines between successive arguments or attributes. Keep the blank lines separating docstring sections.

## Internal Implementation Recipes

Preserve the estimands and public behavior in [att.md](att.md) when changing these implementation details. The recipes below describe the current computational choices, not additional identifying assumptions.

### Validation and State Lifecycle

Use Pydantic v2 `Field` constraints for scalar bounds and types, cached `TypeAdapter.validate_python()` calls for eager collection validation, and `validate_call` for stateless function options. NumPy integer and floating-point scalars remain supported. Numeric strings and Boolean values, including Boolean elements mixed into numeric grids, are rejected. Generators are consumed once into validated lists. Keep vectorized checks for table contents, period normalization, and sample rules; grid uniqueness requires a separate Pydantic validator. Validate fit inputs only after clearing fitted state, so any invalid refit leaves predictions unavailable. Pydantic raises `ValidationError`, a subclass of `ValueError`, with structured field or element locations. Missing constructor column arguments retain Python's `TypeError` behavior.

Use the public `ValueError` contract for invalid values, including Pydantic validation errors. Clear old fitted state before validating a refit. Retain diagnostic counts computed before failure, but never leave stale model predictions available. Use `logging.warning()` for nonfatal diagnostics, aggregate them across each public call, and leave logging handlers and levels to the caller. Internal prediction and aggregation work should be quiet when its caller emits the combined warning.

### Period Indexing and Comparison Construction

Use the earliest and latest normalized outcome periods as the overall study range. An optional internal integer period index may be constructed as the number of fixed-frequency steps since the earliest period plus one, with the same origin for outcome periods and termination periods. For the hardcoded daily frequency, each step is exactly one calendar day; compute elapsed steps rather than ranks of observed labels. Set $`T`$ to the last such index. Preserve gaps: neither within-store row numbers nor ranks of observed periods are period indices. The formulas in [att.md](att.md#theoretical-background) use these internal period indices for convenience; users supply datetime columns. Event periods and isolation lengths are integer numbers of periods.

Use internal integer store codes and elapsed-period indices for exact outcome lookups. Prepare shared baseline observations once and map every outcome store to its first termination period using all canonical first-event rows, not only design-eligible targets. Stores without events have an infinite first-treatment cutoff. At each horizon, match endpoints across cohorts in batches and select controls whose first event is strictly after the later of the focal cohort period and endpoint. This gives the article's endpoint cutoff after treatment and excludes the focal cohort at placebo/reference horizons. Never let excluded targets or events outside the panel disappear from this timing lookup. Keep the full termination-history window lookup for target isolation only. Preserve original identifier values and normalized datetime labels in public outputs and diagnostics; integer indices must preserve gaps in the calendar.

Keep the cell-specific loop needed to save each polynomial transformation, screen its support and rank, record its diagnostics, and assemble accepted comparison rows. Vectorized selection must preserve treated and control membership, endpoint availability, saved transformations, and the cell-specific coefficient interpretation. Do not construct a dense store-by-period-by-horizon panel to perform these lookups.

Preparation collects narrow DataFrames for the accepted non-reference comparisons, then fits those cells in order. Each regression contains only one cell's rows and $`q+2`$ columns, counting its control intercept, treatment intercept, and $`q`$ polynomial terms. Do not concatenate all comparisons into a global regression or create columns for interactions with every cell. This is not a streaming implementation: the collected comparison frames, lookup arrays, diagnostics, and retained models still consume memory, and comparison rows can reuse the same store at several cohorts or horizons. Performance claims require measurement on a specified dataset; changing the regression architecture does not establish a universal speed bound.

### Cell Regressions and Prediction

Construct each comparison DataFrame with internal `firm_id`, `cell`, `cohort`, `period`, `event_time`, `delta_y`, `treated`, and `scaled_dose` columns, where `period` is the endpoint and `cell` is an internal label for `(cohort, event_time)`. Assert uniqueness of `(firm_id, cell)`. `treated` is numeric 0/1. Set `scaled_dose` to $`x_c(D_i)`$ for treated rows and to finite zero for controls. A missing placeholder would cause formula processing to drop control rows even though their treatment interactions are zero. Let PyFixest's formula machinery construct the polynomial terms; no expanded treatment-design columns for other cells are needed.

The cell estimation call is:

```python
import pyfixest as pf

self.models_[key] = pf.feols(
    "delta_y ~ treated"
    f" + treated:poly(scaled_dose, degree={self.degree}, raw=True)",
    data=block,
    lean=True,
    store_data=False,
    copy_data=False,
)
```

Use separate `feols()` calls and store their `Feols` objects by `(cohort, event_time)` in `models_`. The ordinary intercept supplies $`\alpha_c`$; no fixed-effect formula is required within a single cell. The absolute endpoint period $`g+e`$ and baseline period $`g-1`$ are fixed within each cell, so a common calendar effect contributes $`\lambda_{g+e}-\lambda_{g-1}`$ and is absorbed by $`\alpha_c`$. No separate calendar-period fixed effect is included. Use PyFixest's regression solver without constructing an explicit matrix inverse. `lean=True` and `store_data=False` avoid retaining large regression data in the fitted objects. `copy_data=False` avoids an extra copy of the internally constructed cell frame; callers' input tables remain untouched. Module predictions use the saved treatment coefficients and transformation instead of PyFixest's full-outcome prediction API. [PyFixest 0.60.0 API and formula syntax](https://github.com/py-econometrics/pyfixest/blob/v0.60.0/pyfixest/estimation/api/feols.py)

Keep the saved treatment coefficient vector and dose transformation for each cell. Validate every expected treatment coefficient after fitting. Compute component predictions from that vector; do not return full-outcome PyFixest predictions or subtract the control mean twice. Successful reference cells store exact zero vectors without a fitted regression. They cannot make an otherwise unsuccessful fit count as fitted.

### Kernel Aggregation and Bandwidth Search

For each requested horizon, derive the exact endpoint-observed treated identifiers in the retained cohorts. Require a fitted or valid reference cell for every such cohort. Join component predictions and weights by cohort and dose with complete one-to-one coverage. Missing required cells must raise rather than silently disappearing through an inner join. Normalize only over the documented eligible stores, count each store once, and evaluate the weighted sum in [Theoretical Background](att.md#theoretical-background).

Use log-sum-exp for Gaussian cohort-weight normalization. Reject nonfinite numerical results rather than substituting uniform weights. Cache the bandwidth scores already computed, fix expansion directions using the initial minimizers, and explore tied lower and upper boundaries independently. Keep the first rising point in the score table and distinguish numerical plateau, expansion cap, and range-limit stops from a verified rise.

Compute the exact criterion using batches of unique query doses against unique training doses and sparse dose-by-cohort count matrices. Account for multiplicities when subtracting a held-out store's self contribution and averaging the loss, so grouped computation agrees with the explicit store-by-store definition. `batch_size` must be a positive integer and limits the number of query-dose rows processed together; changing it must not change the selected bandwidth or scores beyond numerical precision. Avoid allocating the full store-by-store kernel matrix. Runtime can still be quadratic when every dose is distinct, and memory also depends on the number of distinct doses and cohorts; batching is not an approximation or a guarantee of subquadratic computation.

### Bootstrap Computation

Use a stable weighted least-squares solve, not an explicit matrix inverse. The decomposition in [Sampling and Comparison Estimates](att.md#sampling-and-comparison-estimates) is algebraically equivalent to the original treatment-interacted regression fitted to duplicated comparison rows. Subtract the control mean once. Bootstrap does not call `pyfixest.feols()` or `ATT.fit()` again, and it does not change any stored models, coefficients, bases, or diagnostics. Evaluate each solved polynomial at all requested doses.

Compute kernel masses stably in the log domain, excluding zero sampling counts before taking logarithms. A Gaussian cohort mass is mathematically zero only when the cohort has no sampled treated stores; numerical underflow must not create an apparent zero-mass cohort. Re-estimating both effects and weights in the same draw retains their covariance and the full nonlinear product; there is no delta-method or influence-function approximation in this procedure.

Process sampling counts in batches of at most 32 replications and reconstruct comparison arrays from a shared indexed outcome panel as needed. Do not retain a full $`B\times N`$ count matrix, duplicate the outcome panel for every draw, or cache all repeated control panels on the fitted model. Temporary kernel work is comparison-specific. The returned draw table necessarily retains the requested replication-by-horizon-by-dose results.

Keep full-retained and available-cohort kernel accumulators separate and stable. A very small retained kernel share may underflow to zero without invalidating an otherwise estimable available-cohort aggregate. Preserve failed-cell diagnostics even when other cells yield a successful draw. Use the same store counts for every comparison and kernel contribution in a replication; do not reapply original selection or minimum-size screens.

### Reference Comparisons

The parametric R reference is the no-internal-knot polynomial case of `contdid::cont_did_acrt`. Compare identical samples, polynomial degree, and baseline convention, and explicitly reuse the training basis in reference predictions. Basis coefficients need not agree across parameterizations. The inspected R prediction path rebuilds its basis on the prediction grid; sources were inspected on 2026-11-03. Record the actual R version or source revision used for future numerical comparisons. See the [R reference source](https://github.com/bcallaway11/contdid/blob/main/R/cont_did.R). External R code is not a runtime dependency.

## Validation Commands and Required Coverage

The pytest suites in [tests/test_att.py](../tests/test_att.py) and [tests/test_bootstrap.py](../tests/test_bootstrap.py) check the implementation against independent numerical calculations and targeted examples. Run `uv sync --python 3.11 --group dev`, then `uv run --no-sync python scripts/validate.py --log validation.log` from the repository root. The runner [scripts/validate.py](../scripts/validate.py) records environment versions, package-source verification, test tracebacks, and read-only quality checks in a shareable log, continuing after individual failures. It performs no installation and returns a nonzero exit status for any failed check or mismatch with the configured runtime requirements. It uses `packaging` when available to interpret Python constraints and dependency requirements, including ranges and environment markers; missing parsing support is reported without preventing the remaining checks. Run the tests directly with `python -m pytest tests -v`; [tests/test_validate.py](../tests/test_validate.py) also checks the diagnostic runner without importing the estimator. Shared, independently initialized fixtures live in [tests/conftest.py](../tests/conftest.py); include that file when copying the validation suite. Unit-test assertions may omit messages. Quality commands target only the estimator module, the three test modules, the shared fixture module, the runner, and project documentation. Missing required files produce a failed validation result; quality checks continue on the expected files that are present, with no broad directory scan. Development dependencies include `pytest`, `codespell`, and `orjson`; test fixtures use `orjson` for serialization. Required behaviors include:

- All four column-name arguments are required; omitting any one raises `TypeError` at construction. No conventional input labels or internal schema names supply defaults. Renaming every input column through the constructor leaves results unchanged, using the same `firm_col` and `period_col` labels in both tables. Inputs missing either shared column fail validation; `outcome_col` and `dose_col` may share a label across the two tables.
- `fit()` has no `event_times` argument and generates `list(range(-7, 15))` for `(-7, 14)`. Default prediction and aggregation include all 22 horizons; explicit subsets may include negative periods, and out-of-window requests fail.
- The signed window `(-7, 14)` screens exactly from $`g-7`$ through $`g+14`$, including both boundaries; `(7, 14)` is rejected. Check the panel-bound conditions $`g+L\ge1`$ and $`g+H\le T`$.
- `min_cohort_size=None` resolves to $`p+1`$ for the configured polynomial degree; a smaller explicit minimum cannot bypass that floor, while a larger one is enforced. The default cubic design resolves to five.
- Frequency is hardcoded daily, with `datetime64[ns]` input/output types. There is no frequency inference, configurable frequency, or additional frequency support.
- Within-period time differences do not affect estimates; same-store within-period duplicates in the outcome table are rejected after normalization, as are `NaT` and timezone-aware inputs. Missing periods are never compressed.
- Multiple termination rows for a store retain only its earliest normalized-period row and its dose. Same-period ties use original row order; an ineligible first event is not replaced. Check `duplicate_event_rows=3` and `duplicate_event_firms=2` for the example, with no separate duplicate-event warning.
- Later distinct periods remain available to target-isolation checks after first-event selection. Multiple records on the first store-period retain their multiplicity in history and exclude the candidate as overlapping; no later period is promoted. Their earliest period still determines the store's control-treatment status.
- Capture warning-level log records to verify that `evaluate_theta()` emits at most one warning for the full requested grid when omissions, target restrictions, or extrapolation apply, with correct comparison, row, horizon, and component-prediction counts. Clean and reference-only evaluations must not produce spurious extrapolation warnings. Direct `ATT.predict()` warnings remain functional after aggregation.
- Capture warning-level log records to verify that `fit()` emits one aggregate exclusion warning when exclusions occur and none for a clean fit, rather than one record per cohort or cell. Verify `drop_summary_` reason totals and counting units, including repeated store–horizon rows, empty treated cells, and observed treated rows in `no_controls` cells. On a failed fit, preserve the counts computed before failure. Every module warning uses `logging.warning()`; the module does not use the warnings module.
- A missing baseline or endpoint excludes that store only from affected comparisons; missing intervening periods alone do not. Present rows with missing or infinite outcomes fail validation. Temporary inactivity can cause a store to contribute at one horizon, disappear at another, and reappear later.
- Pre-treatment placebo estimates use $`Y_{g+e}-Y_{g-1}`$ with the correct sign and a fixed baseline. A known dose-related pre-trend is recovered without being forced to zero or labeled as a causal ATT.
- An eligible reference cell returns exact zero ATT coefficients, predictions, and aggregation at $`e=-1`$, with the correct estimate type and counts. It contributes no regression rows or columns. Reference-only successes cannot hide failure of every non-reference comparison.
- Separate `feols()` calls produce one `Feols` object per successfully estimable non-reference cell in $`\mathcal{J}`$, without a global expanded design. On constructed samples with multiple cohorts and horizons, their ATT curves match independent treated-only OLS predictions in the package's no-internal-knot B-spline basis minus each cell's control mean, including a cell with a single control. Check several degrees, observed and extrapolated doses, and missing endpoints; basis coefficients themselves need not match.
- Batched baseline, endpoint, and first-treatment lookups preserve comparison membership and availability diagnostics across several cohorts, missing periods, and excluded first events. Verify target isolation separately using complete termination history. Confirm that reference, placebo, and post-treatment point estimates retain the independent least-squares equivalence; do not replace numerical checks with a claimed speed improvement.
- A two-period comparison with a known polynomial dose effect and a nonzero control trend recovers the curve from the treatment coefficients. Prediction neither adds the control intercept nor subtracts a control mean again.
- Each regression uses its own saved dose transformation and polynomial formula, with a finite zero placeholder for control doses. All comparison rows receive unit weight; reused stores remain present in every eligible comparison. No cell regression contains rows or interaction columns for other cells.
- A common calendar-period shock cancels through each cell's control comparison without a separate calendar-period fixed effect.
- Missing or nonfinite expected treatment coefficients after `feols()` cause fitting to fail. No automatically dropped polynomial term may silently redefine a retained curve.
- Predictions at a fixed dose are invariant to other requested doses and input row order. For same-period ties, the selected candidate row follows input order, but the candidate is excluded as overlapping, so its dose cannot change fitted effects. Where R training and prediction bases coincide, unpooled estimates on the same constructed sample agree numerically; otherwise compare against an R reference that explicitly reuses the training basis.
- Two sufficiently separated termination periods at one store produce only one target event and one kernel contribution. Its later period remains in the target-isolation history; its first period determines control eligibility. Every control store contributes once per comparison.
- Simultaneous and overlapping target events are excluded using the original complete history. Excluded candidates and dropped cohorts remain in the control first-treatment lookup, as do first events outside the panel; their stores cannot reenter the control pool after treatment.
- For post-treatment comparisons, a store with its first termination inside the target's isolation window but after the current endpoint remains a control. It is excluded starting exactly at its first termination period. Previously treated stores are excluded even when they have no event inside the target's window. A future control need not qualify as an isolated target itself.
- For negative and reference horizons, controls must first be treated strictly after the focal cohort. Exclude the focal cohort, including ineligible candidates, and any store treated by the fixed baseline. Missing baseline or endpoint rows still exclude an otherwise eligible control at that comparison.
- Screening bounds equal the overall normalized outcome-period minimum and maximum, regardless of individual-store gaps or additional event periods. An isolation interval crossing either boundary excludes the event at every horizon; intervals touching the boundaries are allowed. Extra event records outside the panel cannot rescue an excluded event. Changing only prediction horizons leaves the fitted screen unchanged.
- Small or rank-deficient comparisons contribute to the single fit warning and disappear from the corresponding horizon only, while retaining the complete history. Verify that an estimable short horizon survives a sparse later horizon, with separate fitted bases and matching store-specific kernel inputs.
- Weights count each selected store once, sum to one, reflect unequal cohort sizes, and match the explicit formula. A within-cohort excluded event cannot leak back into the weights.
- Both bandwidth helpers select the first event per store after period normalization, apply optional eligible-store filtering afterward, preserve inputs, and work without outcomes or an `ATT` fit. Verify renamed columns, repeated terminations, same-period ties, invalid identifiers, and unknown or duplicated subset identifiers.
- `estimate_h0()` matches the sample-standard-deviation and interquartile-range formula, including the zero-interquartile-range fallback. One cohort remains valid when doses vary; insufficient observations and constant doses fail.
- `estimate_h_cv()` matches a direct store-by-store leave-one-out Gaussian calculation, including repeated doses, unequal cohort sizes, and singleton cohorts. Its score averages over stores and sums over every cohort, with the held-out store excluded but same-dose peers retained.
- Verify the default nine-bandwidth initial grid from `h0 / 10` through `10 * h0`, sorted score schema, smallest exact-minimum tie rule, positive unique initial-candidate validation, positive integer batch size, and positive integer `max_expansions`. Different batch sizes must preserve the score and selection within numerical precision. One cohort or constant doses fail even with explicit candidates.
- Check factor-two expansion below and above initially minimizing boundaries, including explicit initial grids. Ties at both boundaries and one-candidate grids must explore both directions independently. Retain the first meaningful rise, cache initial and visited scores, and select the minimum across the complete sorted evaluated grid.
- Verify that meaningful rises use the stated machine-precision tolerance, three consecutive negligible changes stop a direction as a plateau, and meaningful decreases reset that count. Cap additional evaluations separately per direction and stop safely before generated bandwidths or squared scaled dose distances leave the numerical range. Plateau, cap, and range-limit warnings must not claim that a rise was observed.
- A CV search emits at most one aggregate warning combining singleton cohorts, a selected final-grid boundary, tied minima, and expansion stop reasons, with correct counts. A clean interior unique-minimum search emits none, including an expansion terminated by a meaningful rise. Bandwidth selection never refits component models or mutates their samples, coefficients, diagnostics, or weights.
- Aggregation equals a manual weighted sum for both post-treatment and placebo horizons, reports store counts and estimate types, identifies the retained population, and rejects any requested horizon with a missing required cell, including a reference cell with no controls.

- Bootstrap sampling counts produce the same cohort curves, kernel weights, and aggregates as explicitly duplicated comparison rows using the original horizon-specific not-yet-treated controls; all-one counts reproduce the original point estimates.
- One count per original outcome store is shared across every treated/control role and all requested doses and horizons, including stores unused by the requested comparisons. A fixed seed reproduces draws without mutating the fitted model.
- Zero-mass cohorts contribute zero under both policies. Under `strict`, positive-mass rank or control failures invalidate every dose at the affected horizon. Under `available_cohorts`, retain estimable cohorts, normalize their kernel weights to one, and report missing estimates only when no cohort remains. No treated mass makes the whole horizon unavailable; reference draws with positive treated mass remain zero without regression checks.
- Bootstrap does not reapply the original minimum-size screen, change the saved dose basis, refit PyFixest models, or omit failed-cell diagnostics. Unexpected numerical errors raise instead of producing ordinary failure diagnostics.
- Successful-only quantiles, sample standard deviations, attempted/successful/failed counts, and status flags match independent calculations separately for each dose–horizon pair. Zero or one successful draw produces missing inference values without discarding raw draws or the original point estimate. Check available-cohort estimates against an independent calculation on explicitly duplicated rows after excluding failed cohorts; normalized weights must prevent missing effects from being treated as zero.
- Full-panel fingerprint checks accept reordered rows, ignored time components, and extra unmapped columns, but reject changed normalized observations, missing rows, or a changed store population, including changes outside fitted comparison samples.
- Bootstrap failures, cohort exclusions, and original aggregation warnings produce a single combined warning, with aggregated diagnostics that count failed comparisons independently of the requested dose count. Verify `n_partial`, available and failed cohort counts, and retained kernel mass shares; underflow of a very small retained share must not discard an otherwise valid available-cohort estimate.

## Documentation Maintenance

Keep [att.md](att.md) portable: its top-level sections are Theoretical Background, Implementation Details, and References. Put statistical definitions and equations in the first section and the public API, input contracts, outputs, and user-visible behavior in the second. Keep repository paths, setup commands, coding instructions, test-running guidance, and internal construction recipes in this file. Do not link the portable document to these maintenance notes or require this file when copying the module documentation elsewhere.

Use the title `Continuous-Dose ATT and Dose-Specific Event-Study Aggregation`. Cite the single article inline as Callaway et. al. (2025) and retain its Harvard-style entry at the end of the portable document. Do not describe earlier drafts or abandoned implementation versions in the portable documentation.

For GitHub math, use fenced `math` blocks for display equations and protected dollar/backtick delimiters for inline expressions. Use explicit braces around accent and font arguments. Avoid `operatorname`; use a supported upright text macro where appropriate. A bare equals-sign line outside a fenced math block can become a Markdown heading, and unprotected asterisks or underscores can be interpreted as Markdown. When refactoring, check all formulas, prose links, and heading anchors, not just the reported example.

Run spelling checks on both documentation files. For structural changes, verify that the mathematical expressions survive Markdown parsing unchanged and check them with MathJax when available. Do not claim a browser check when only source or local rendering was checked. No estimator regression tests need to be added for a documentation-only reorganization.
