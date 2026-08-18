# Phase 8 — Target Weight Construction

**Generated from:** config SHA-256 · data snapshot `manifest_2026-08-06.json` ·
git commit per [phase8_manifest.json](phase8_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py --phase 8`
**Scope:** target weights only. **No portfolio return, transaction cost,
turnover, Sharpe ratio, or cumulative wealth was computed or examined.**
A test asserts no output column contains any performance term.

Six strategies × 200 origins = **6,000 target weights**, 1,200 audit rows.

---

## 1. The ladder as implemented

| # | Strategy | Covariance source | Cap |
|---|---|---|---|
| 1 | Equal weight | none | n/a (20% each) |
| 2 | 60/40 SPY/IEF | none | **exempt** (holds 60% SPY by design) |
| 3 | Static minimum variance | Ledoit-Wolf through 2009-12-31, then frozen | 40% |
| 4 | Rolling Ledoit-Wolf min-var | current unconditional LW | 40% |
| 5 | EWMA-scaled min-var | EWMA vol forecasts × **LW correlations** | 40% |
| 6 | Regime-aware min-var | A2-consumed matrix | 40% |

Strategy 5 deliberately combines EWMA volatility forecasts with the
Ledoit-Wolf **correlation** matrix rather than using the raw EWMA
covariance, whose weight-based effective sample size is only ≈32
observations at λ = 0.94 and which Phase 7 measured as the worst-
conditioned estimator (mean condition number 116, max 505).

## 2. Optimizer results

`optimizer_summary.csv`, 800 optimized solves (4 strategies × 200 origins):

| Strategy | Success rate | Mean iterations | Max constraint violation | Mean assets at cap | Fallback requests |
|---|---|---|---|---|---|
| static_minvar | 100% | 8.0 | 0.0 | 1.0 | 0 |
| rolling_lw_minvar | 100% | 8.9 | 3.3e-16 | 1.0 | 0 |
| ewma_scaled_minvar | 100% | 6.7 | 4.4e-16 | 1.5 | 0 |
| regime_minvar | 100% | 7.5 | 4.4e-16 | 1.5 | 0 |

**Every solve converged and every solution passed independent
validation.** Constraint violations are at machine precision (≤4.4e-16),
verified by recomputing the weight sum and bounds directly rather than
trusting the solver's success flag. Zero fallback requests were emitted,
so Phase 9 will face no "no trade" rebalances from optimization failure.

Covariance matrices are scaled by their trace before SLSQP (a raw daily
covariance makes the objective ~1e-4 and starves the convergence test);
positive scaling leaves the argmin unchanged, which is unit-tested
across factors from 1e-4 to 1e4. Reported objective values are computed
on the **unscaled** matrix so they remain comparable across dates.

## 3. Amendment A2 verified at the weight level

On the four A2 fallback origins the regime-aware strategy consumes the
unconditional Ledoit-Wolf matrix, so its targets must equal the
rolling-LW targets exactly:

| Date | max abs weight difference vs rolling LW |
|---|---|
| 2009-12-31 | **0.00e+00** |
| 2012-03-30 | **0.00e+00** |
| 2012-04-30 | **0.00e+00** |
| 2012-05-31 | **0.00e+00** |

And the two strategies differ at **196 of 200 origins** — exactly
200 − 4 — with typical differences near 0.10 in the largest weight.

**A window inconsistency was found and fixed to achieve this.** The
first Phase 8 pass produced a 3e-3 difference on those dates because
Phase 7 estimates Ledoit-Wolf on the feature window (which starts ~22
trading days later, since realized volatility needs 21 observations)
while Phase 8 was recomputing it on the full return history. Two
windows, two slightly different matrices. Since the A2 fallback is
*defined* as "use the unconditional Ledoit-Wolf estimate" and the
preregistered primary hypothesis compares regime-aware against rolling
Ledoit-Wolf, that mismatch would have introduced a small
apples-to-oranges component into the headline comparison on exactly the
dates where the two should coincide. The fix was architectural rather
than a loosened tolerance: **Phase 7 now exports its Ledoit-Wolf and
regime-mixture matrices, and Phase 8 consumes them directly instead of
re-estimating anything.** One window, one set of matrices, exact
equality by construction. The static target likewise now comes from the
first origin's matrix.

## 4. What the targets look like

Mean target weight by strategy (descriptive; no performance implied):

| Strategy | SPY | QQQ | IWM | IEF | GLD |
|---|---|---|---|---|---|
| Equal weight | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |
| 60/40 | 0.600 | 0.000 | 0.000 | 0.400 | 0.000 |
| Static min-var | 0.142 | 0.173 | 0.000 | 0.400 | 0.286 |
| Rolling LW min-var | 0.249 | 0.077 | 0.000 | 0.400 | 0.273 |
| EWMA-scaled min-var | 0.314 | 0.038 | 0.010 | 0.400 | 0.238 |
| Regime-aware min-var | 0.293 | 0.061 | 0.000 | 0.400 | 0.246 |

Two structural observations:

**IEF sits at the 40% cap in every optimized strategy at every origin.**
The cap binds continuously for the lowest-volatility asset, so the
optimizers are constrained rather than interior. This is a direct
consequence of the weight limit and should be remembered when
interpreting later results: these are capped minimum-variance
portfolios, not unconstrained ones.

**IWM is excluded almost everywhere.** Phase 4 measured SPY-IWM
correlation at 0.89 with IWM the more volatile of the pair, so a
variance minimizer has no reason to hold it — the collinear equity
block resolves toward the least volatile member.

Target dispersion across dates (mean standard deviation per asset)
separates the rungs cleanly: 0.0000 for the three static strategies,
0.0259 for rolling LW, 0.0604 for EWMA-scaled, and 0.0605 for
regime-aware. Each ladder rung adds responsiveness, with the
regime-aware and EWMA-scaled strategies moving about 2.3× as much as
rolling Ledoit-Wolf. Whether that extra movement earns its turnover is
a Phase 9-10 question and is not addressed here.

## 5. Traceability

Every optimized target carries a `covariance_hash` (SHA-256 prefix of
the exact input matrix), so any weight can be traced to the matrix that
produced it. The audit also records `solver_success`, `solver_status`,
`iterations`, `objective_value`, `sum_weights`, `minimum_weight`,
`maximum_weight`, `constraint_violation`, `cap_binding_count`,
`fallback_requested`, and `covariance_source`.

## 6. Failure policy (deferred to Phase 9 by design)

Phase 8 cannot implement "hold the previous drifted portfolio" without
using subsequent asset returns, which would violate the phase boundary.
A failed optimization therefore emits **no weight rows** and sets
`fallback_requested = True`; Phase 9 reconstructs pre-trade drifted
holdings and interprets the request as "no trade". Phase 8 never
substitutes previous target weights. The path is unit-tested (a
degenerate covariance yields `None` weights plus a fallback request)
even though it was never triggered on this data.

## 7. Artifacts

`outputs/strategies/target_weights.parquet` (6,000 rows: date,
strategy, asset, weight), `optimizer_audit.csv` (1,200 rows),
`optimizer_summary.csv`, [phase8_manifest.json](phase8_manifest.json).
Reproduce with `python scripts/run_analysis.py --phase 8` after Phase 7.
Tests: 18 Phase 8 unit tests covering exact rule-based weights, the
cap exemption for 60/40, constraint satisfaction, scale invariance,
static-target constancy, per-strategy covariance sourcing, the
EWMA-vol × LW-correlation construction, A2 equivalence, the fallback
request path, hash traceability, and the absence of any performance
quantity.
