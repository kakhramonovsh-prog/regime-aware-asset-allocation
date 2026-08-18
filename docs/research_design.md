# Preregistered Research Design

**Project:** Regime-Aware Volatility Forecasting and Dynamic Asset Allocation
**Author:** Shakhrukh Kakhramonov
**Design frozen:** 2026-08-15, git tag `v0.2.0-preregistered`
**Status at freeze:** the data pipeline and unit tests exist; **no volatility
model, regime model, portfolio optimization, or backtest has been run.**
No out-of-sample result of any kind existed when this document was committed.

This document fixes the hypothesis, the primary specification, the
information timeline, and the robustness grid *before* any result is
generated. The machine-readable version of every frozen parameter is
[`config/analysis_plan.yaml`](../config/analysis_plan.yaml); a unit test
(`tests/test_analysis_plan.py`) asserts that the runtime configuration
matches the frozen plan. Any later deviation must be documented in a
"Deviations" section appended below, with the git history showing when
and why.

---

## 1. Hypothesis and primary outcome

**Primary hypothesis.**

- H0: `SR_net(regime-aware min-var) − SR_net(rolling Ledoit-Wolf min-var) ≤ 0`
- H1: the difference is positive.

The primary comparison is against the **closest non-regime strategy**
(rolling Ledoit-Wolf minimum variance), not against equal weight or
60/40. If the regime-aware portfolio only beats static benchmarks, the
improvement could come from dynamic covariance estimation alone; the
paired comparison isolates the contribution of regime conditioning.

**Primary outcome.** Difference in annualized out-of-sample Sharpe
ratio between the regime-aware minimum-variance strategy and the
rolling Ledoit-Wolf minimum-variance strategy, **net of 10 bps
proportional transaction costs**, over the common evaluation period,
with a stationary-bootstrap 95% confidence interval.

**Secondary outcomes** (reported for all strategies, never promoted to
headline): annualized return, annualized volatility, maximum drawdown,
Sortino ratio, Calmar ratio, one-way turnover, total cost drag, and
certainty-equivalent return under CRRA utility with risk aversion 5
computed from monthly net returns.

Equal weight and 60/40 remain in every table as economic benchmarks.

**Language rule.** Results are reported with confidence intervals. If
an interval includes zero, the paper says so plainly (e.g. "the
estimated net Sharpe difference is positive but the interval is wide
and does not exclude zero"). The word "outperformed" is reserved for
differences whose interval excludes zero.

## 2. Sample and evaluation period

| Item | Frozen value |
|---|---|
| Data snapshot | `data/snapshots/manifest_2026-08-06.json` (SHA-256 per file) |
| Full sample | 2004-11-18 to 2026-08-06 (5,462 trading days) |
| Initial training window | 2004-11-18 to 2009-12-31 |
| Out-of-sample period | 2010-01-01 to 2026-08-06 (~199 months) |
| Rebalancing | Monthly, signal on the last trading day of each month |
| Estimation scheme | Expanding window anchored at 2004-11-18 |

The first out-of-sample rebalance uses the signal computed on the last
trading day of December 2009 and executes on the first trading day of
January 2010. All strategies share the same return panel, rebalance
dates, execution timeline, and cost convention, and are evaluated over
the identical out-of-sample window.

## 3. Information and execution timeline

The backtest simulates this sequence at every rebalance:

1. **Observe** information through the close of signal date `t` (the
   month's last trading day).
2. **Estimate** all models (volatility, HMM, covariance) using data
   through `t` only.
3. **Compute** target weights from those estimates.
4. **Execute** at the close of the next trading day `t+1`. Because the
   dataset contains adjusted closes only, assuming execution at the
   same close where the signal was observed would be look-ahead; the
   conservative convention sacrifices one day.
5. **Earn** the new weights' returns starting with the `t+1 → t+2`
   close-to-close return. The `t → t+1` return accrues to the old
   (drifted) weights.
6. **Charge** transaction costs on the execution date `t+1`.

**Macro availability lag.** All FRED-sourced series (VIXCLS, DGS2,
DGS10, T10Y2Y, DFF) are lagged **one additional trading day** before
entering any signal, because H.15 yield data are published with a lag
and same-close availability of the VIX print is not guaranteed for a
close-of-day decision process. Price-derived features (returns,
realized volatility) use data through the close of `t` without extra
lag. This policy is implemented by `apply_signal_lag()` in
`src/preprocessing.py` and is fixed for all specifications. Real-time
ALFRED vintages are unnecessary for these series (daily financial
series are not revised the way monthly macro releases are) and are
listed as future work.

**Pipeline-level causality requirement.** The preprocessing-level
truncation test (`assert_causal`) extends to the full pipeline: for any
date `t`, perturbing or deleting raw prices, macro observations, or
returns dated after `t` must not change any signal, regime probability,
covariance estimate, target weight, trade, cost, or portfolio return
dated on or before `t`. Implemented today for the raw-to-processed
stage (`tests/test_pipeline_causality.py`); the same test wraps the
backtest engine when Phase 9 is built, and Phase 9 is not complete
until it passes.

## 4. Volatility models (Phase 5)

Compared models, all estimated on expanding windows with a minimum of
252 daily observations:

| Model | Frozen specification |
|---|---|
| Rolling historical | 63-trading-day window standard deviation |
| EWMA | RiskMetrics recursion, λ = 0.94, seeded with the first 63 days' sample variance |
| GARCH(1,1) | Normal innovations, refit at each rebalance date via `arch` |

**Horizon alignment.** The portfolio rebalances monthly, so the
evaluation target is the **integrated variance over the next holding
period** (sum of daily squared log returns from execution to the next
rebalance), not one-day-ahead variance. Each model produces a
holding-period variance forecast at each rebalance date; all models are
evaluated on identical dates.

**Losses.** QLIKE is the primary loss (robust to noisy volatility
proxies in the sense of Patton 2011); MAE and RMSE are secondary.
Pairwise Diebold-Mariano tests with HAC long-run variance are reported
with the caveat of ~199 monthly observations.

**Failure policy.** If a GARCH refit fails to converge, the EWMA
forecast substitutes for that date and the event is logged in an audit
column. Daily variance forecasts are floored at 1e-8. Annualization
uses 252 trading days.

**Model feeding the portfolio.** The volatility-scaled strategy uses
**EWMA (λ = 0.94)**, chosen ex ante because it is deterministic, has no
convergence failures, and is the industry-standard filter. GARCH-fed
portfolios are robustness only. The out-of-sample loss comparison does
NOT get to promote a different model into the main portfolio after the
fact.

## 5. Regime model (Phase 6)

**Main specification: 2-state Gaussian HMM**, estimated by EM
(`hmmlearn`), refit at every rebalance date on the expanding window.

**Features** (daily, standardized):

| Feature | Construction | Signal lag |
|---|---|---|
| Market return | SPY daily log return | none (price-derived) |
| Realized volatility | 21-day rolling std of SPY log returns | none (price-derived) |
| VIX level | log(VIXCLS) | 1 trading day |
| Yield slope | DGS10 − DGS2 | 1 trading day |

Standardization is **expanding-window**: at refit date `t`, features
are z-scored with means and standard deviations computed from data
through `t` only. Full-sample standardization is look-ahead and is
caught by the causality tests.

**Real-time filtering (the critical constraint).**
`predict_proba`-style forward-backward smoothing uses observations
after each interior date. Therefore:

- The trading signal at rebalance `t` is the **filtered** probability
  `p_t = P(S_t = k | F_t)`: fit on data through `t`, take the forward
  probabilities at the final observation of the truncated sample.
- Full-sample smoothed probabilities are computed once, stored in a
  separate file (`outputs/regimes/ex_post_probabilities`), and appear
  only in explicitly labeled ex-post descriptive figures. They never
  touch a trading decision.
- Stored real-time probabilities must be reproducible: the value stored
  for date `t` must equal the last filtered probability from a fresh
  fit on the sample truncated at `t`, and appending future observations
  must leave stored historical values unchanged. Both are unit tests.

**Label switching.** State indices from EM are arbitrary and can flip
across refits. After every fit, states are relabeled using training
information only: ascending order of state-conditional mean realized
volatility (state 0 = lower volatility). The 3-state robustness model
orders low → medium → high the same way. No code path may assume raw
state 0 means "calm".

**Initialization.** EM is a local optimizer, so each refit runs 16
predetermined initializations (seeds 42 through 57), keeps the fit with
the highest training log-likelihood, then relabels canonically. Each
refit logs convergence status, iterations, log-likelihood, transition
matrix, and state occupancy. Seed sensitivity is a robustness check.

**Degenerate-fit guard.** If any state's occupancy over the training
window falls below 5%, the refit is flagged; the regime-aware strategy
falls back to the unconditional (Ledoit-Wolf) covariance for that
rebalance, and the fallback is recorded in an audit column.

**Interpretation discipline.** States are **latent market states** in
the sense of Hamilton (1989), not directly observed economic facts. The
paper labels them "low-volatility state" / "high-volatility state" only
if the estimated state-conditional moments justify those names.

**Feature redundancy.** Realized volatility and VIX are expected to be
highly correlated, which risks the HMM collapsing into a pure
volatility classifier. The 4-feature set above is frozen as the main
specification; dropping VIX and dropping realized volatility are both
preregistered robustness runs. Reported diagnostics: feature
correlations, state-conditional feature means, occupancy, mean state
duration, transition matrices, number of state changes, and stability
of state paths across expanding refits. The feature set is **not**
selected on backtest Sharpe.

## 6. Regime-aware covariance and optimization (Phases 7-8)

At rebalance date `t` with holding horizon `H = 21` trading days:

1. **Filtered state probability** `p_t` (Section 5).
2. **Horizon-averaged state probabilities** using the transition matrix
   `P_t` estimated through `t`:
   `p̄_t(H) = (1/H) Σ_{h=1..H} p_t P_t^h`.
3. **State-conditional covariances** `Σ_{k,t}` (wording tightened by
   Amendment A1): weighted covariance of daily returns with weights
   `γ_{s,k} = P(S_s = k | F_t)`, the **smoothed responsibilities from
   the model fit on data through `t`**. Smoothing at a historical date
   `s < t` uses observations between `s` and `t`, all of which are
   known at the rebalance date, so the estimate conditions only on
   `F_t` and is causal *at the decision date*. The trading signal for
   the current state remains the filtered endpoint probability `p_t`
   (identical to the smoothed value at `s = t`). Full-sample smoothed
   paths (conditioning on data after `t`) never enter any estimate.
   Each `Σ_{k,t}` is shrunk toward the unconditional Ledoit-Wolf
   estimate. Effective sample size per state:
   `n_eff,k = (Σ_s γ_{s,k})² / Σ_s γ_{s,k}²`.
   If `n_eff,k < 60`, shrinkage toward the unconditional estimate
   increases with mixing weight `n_eff/(n_eff+60)`; if a state is
   degenerate (Section 5), the strategy falls back to unconditional.
   **Why 60:** roughly one quarter of daily data (63 trading days,
   rounded), matching the project's 63-day estimation windows; a
   five-asset covariance matrix has 15 free parameters, so 60
   observations is about four per parameter, a floor below which
   sample covariance estimates are fragile. The value is a
   regularization constant, not an estimated quantity; thresholds
   **30 and 120** are predetermined robustness checks (Amendment A1).
4. **Forecast covariance** `Σ_RA,t = Σ_k p̄_{t,k}(H) Σ_{k,t}`,
   symmetrized and checked positive semidefinite (eigenvalue floor
   1e-10 with logged clipping).
5. **Optimize** `min_w wᵀ Σ_RA,t w` subject to `Σ w_i = 1`,
   `0 ≤ w_i ≤ 0.40` (SLSQP). The optimizer's convergence status is
   stored; on failure the portfolio **holds its drifted previous
   weights** and the failure is recorded in an audit column. Weights
   are never silently substituted.

## 7. Strategy menu (ablation ladder)

Six strategies, ordered so each rung isolates one design ingredient:

| # | Strategy | Adds |
|---|---|---|
| 1 | Equal weight (1/5 each) | baseline |
| 2 | 60/40 SPY/IEF, monthly rebalanced | static policy portfolio |
| 3 | Static minimum variance (Ledoit-Wolf on the initial training window, never re-estimated) | optimization |
| 4 | Rolling Ledoit-Wolf minimum variance | dynamic covariance (**primary comparator**) |
| 5 | Volatility-scaled minimum variance (EWMA λ=0.94 forecast variances + Ledoit-Wolf correlation) | explicit vol forecasting |
| 6 | Regime-aware minimum variance (Section 6) | latent-regime conditioning (**primary strategy**) |

Reading performance down the ladder attributes any improvement to
dynamic covariance, volatility forecasting, or regime conditioning
specifically. Inverse-volatility weighting is a robustness benchmark
only. A buy-and-hold 60/40 variant is reported as a sensitivity.

**Volatility-targeting conflict, resolved.** A portfolio cannot be
simultaneously fully invested in risky ETFs and scaled to a fixed
absolute volatility target without cash or leverage. The main analysis
therefore contains **no absolute-volatility-targeting strategy**; rung
5 is a fully invested, risk-based reweighting. An absolute vol-target
strategy with a cash sleeve earning DFF (risky weights ≤ 1, no
leverage) is a possible robustness extension, clearly separated from
the main ladder.

**Constraints (all optimized strategies):** long-only, fully invested
(`Σw = 1`), max 40% per ETF, no expected-return forecasting anywhere in
the main analysis.

## 8. Transaction costs and turnover

- Pre-trade weights are the previous targets **drifted** by realized
  returns since the last rebalance; turnover compares new targets with
  drifted holdings, not with previous targets.
- Absolute trading volume: `V_t = Σ_i |w_i,target − w_i,pretrade|`.
- Reported one-way turnover: `V_t / 2`.
- Cost charged on the execution date: `cost_t = c × V_t`, with
  `c = 10 bps` in the primary specification. **The cost multiplies the
  full trade sum `V_t`, never the halved turnover figure**, so costs
  are not understated; the factor of two is never applied twice. The
  convention is fixed here and unit-tested in Phase 9.
- Worked example: targets rise 10 percentage points in each of two
  assets and fall 20 in a third. Then `V_t = 0.10 + 0.10 + 0.20 =
  0.40`, reported turnover is `0.20`, and the cost at 10 bps is
  `0.0010 × 0.40 = 4.0` bps of portfolio value.
- 0, 5, and 20 bps are **sensitivity analyses**, not alternative
  headline results.
- Gross returns, cost series, and net returns are stored separately;
  the pipeline never stores only the net series.
- Every strategy, including 60/40 and equal weight, faces the same
  execution timeline and cost convention.

## 9. Statistical inference (Phase 11)

With roughly 199 monthly out-of-sample observations spanning few
independent crisis episodes, interval estimates carry more weight than
point p-values.

- **Primary:** stationary bootstrap (Politis-Romano) confidence
  intervals for the net Sharpe difference of the primary comparison;
  mean block length chosen by the standard automatic rule, seed 12345,
  10,000 replications, resampling the paired monthly net return series
  jointly to preserve cross-strategy dependence.
- HAC (Newey-West) standard errors for mean-return differences.
- Diebold-Mariano tests for volatility-forecast loss differences.
- If factor regressions are added, factor data source is documented
  first and no alpha claim is made without its test.
- Because one primary comparison is preregistered, no multiple-testing
  correction applies to the headline. If the robustness grid is ever
  summarized as a formal joint test, Hansen's SPA framework is the
  named tool.

## 10. Robustness grid (Phase 12, predefined)

All runs below are labeled robustness; none can be promoted to primary:

- 3-state HMM (vs 2-state main).
- Feature ablations: drop VIX; drop realized volatility.
- Covariance estimators: sample, EWMA covariance, Ledoit-Wolf.
- Estimation windows: rolling 5-year vs expanding.
- Weight caps: 30%, 40%, 50%.
- Costs: 0, 5, 20 bps.
- Universe: core SPY/IEF/GLD vs all five ETFs.
- HMM seed sensitivity (alternative predetermined seed block).
- Subperiods: pre-2020, COVID (2020-02 to 2020-12), 2022 tightening
  (2022-01 to 2023-10), post-2020 full. Subperiod findings are
  **descriptive** (samples are small) and labeled as such.

## 11. Deliverables per phase

Paths use the repository's `outputs/` tree (equivalent to the
`results/` naming some style guides use).

- **Phase 4 (EDA):** `outputs/tables/summary_statistics.csv`,
  `correlation_matrix.csv`, `missingness_audit.csv`;
  `outputs/figures/normalized_prices`, `rolling_volatility`,
  `rolling_correlations`, `drawdowns`, `macro_features`. EDA exists to
  find data problems; it does not redesign the primary model.
- **Phase 5:** `outputs/forecasts/volatility_forecasts.parquet`,
  `outputs/tables/volatility_loss_by_asset.csv`,
  `forecast_comparisons.csv`.
- **Phase 6:** `outputs/regimes/realtime_probabilities.parquet` and
  `ex_post_probabilities.parquet` (kept strictly separate),
  `outputs/tables/regime_characteristics.csv`,
  `transition_matrices.csv`, `hmm_stability.csv`.
- **Phase 7:** one audit row per estimator and rebalance date: date,
  estimator, min/max eigenvalue, condition number, shrinkage intensity,
  effective sample size, fallback flag.
- **Phases 8-10:** `outputs/backtests/weights.parquet`,
  `trades.parquet`, `returns_gross.parquet`, `returns_net.parquet`;
  `outputs/tables/performance_summary.csv`, `turnover_and_costs.csv`.
- **Phases 11-12:** interval tables for the primary comparison and the
  robustness grid.
- **Paper:** `outputs/results_manifest.json` recording git commit, data
  hashes, config hash, environment versions, timestamp, and every
  generated table/figure; LaTeX pulls numbers via `\input{}` macros
  only.

## 12. Reproducibility freeze

- Data snapshot manifest with SHA-256 hashes:
  `data/snapshots/manifest_2026-08-06.json` (generated by
  `scripts/freeze_snapshot.py`). Raw vendor files are not redistributed
  until Yahoo's terms are checked; the downloader, metadata, and hashes
  are public instead.
- Environment: `requirements.txt` is the authoritative pinned source;
  `environment.yml` wraps it; Python 3.12.
- CI: GitHub Actions runs the unit suite on Python 3.12 on every push.
- Seeds: global 42; HMM initializations 42-57; bootstrap 12345. Fixed
  here, before any result.
- Repository remote is private until the design is frozen, EDA is
  complete, and the README accurately reflects finished work only.

## 13. References

Standard results this design relies on. Bibliographic details were
verified against publisher or index pages on 2026-08-15 (sources:
Taylor & Francis, ScienceDirect/ACM DL, JSTOR, RePEc, Duke Scholars);
they will be re-checked once more when `references.bib` is built.

- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle. *Econometrica*
  57(2), 357-384. JSTOR: https://www.jstor.org/stable/1912559
  (DOI 10.2307/1912559).
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate
  Analysis* 88(2), 365-411. DOI 10.1016/S0047-259X(03)00096-4.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies. *Journal of Econometrics* 160(1), 246-256.
  DOI 10.1016/j.jeconom.2010.03.034.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive
  Accuracy. *Journal of Business & Economic Statistics* 13(3), 253-263.
  DOI 10.1080/07350015.1995.10524599.
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap.
  *Journal of the American Statistical Association* 89(428), 1303-1313.
  DOI 10.1080/01621459.1994.10476870.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability.
  *Journal of Business & Economic Statistics* 23(4), 365-380.
  DOI 10.1198/073500105000000063.

## Amendments and deviations

Any change to a frozen item after `v0.2.0-preregistered` must be
recorded here with date, reason, and whether any results had been seen
when the change was made. The tagged version remains immutable in git
history, so every amendment is diffable.

### Amendment A1 — 2026-08-15 (no results of any kind existed)

Made in response to external review of the preregistration, before
Phase 4 began and before any model estimation:

1. **Responsibilities clarified (§6 step 3).** State-conditional
   covariances use smoothed responsibilities `P(S_s = k | F_t)` from
   the fit on data through `t`. This conditions only on information
   available at the rebalance date; the current-state trading signal
   remains the filtered endpoint probability. The original wording
   ("filtered state probabilities available at t") was ambiguous
   between this and re-using the stored real-time filtered series.
2. **`n_eff` threshold rationale documented (§6)** and thresholds
   30/120 added to the predetermined robustness grid. The value 60 is
   a regularization constant chosen for consistency with the project's
   63-day quarterly windows (~4 observations per free covariance
   parameter for 5 assets); it was not tuned on any result.
3. **Cost convention restated with a worked example (§8)** confirming
   the cost multiplies the full trade sum `V_t`, not the halved
   turnover figure.
4. **References verified (§13)**: `[VERIFY]` markers replaced with
   publisher-checked volumes, pages, and DOIs.

### Amendment A2 — 2026-08-15 (no covariance or portfolio results existed)

**Absorbing transition rows are treated as degenerate fits.**

If either HMM state has a transition persistence meeting the previously
recorded absorbing-state threshold (`diag(P) >= 1 - 1e-12`, fixed in
`implementation_parameters.csv` before this amendment), the fit remains
in the regime diagnostics, but the **regime-conditioned covariance is
not consumed for that rebalance**. The portfolio-construction pipeline
instead uses the unconditional Ledoit-Wolf covariance estimated through
that date. The real-time probability is preserved unchanged, and the
fallback date and reason are audit-logged. The rule applies
symmetrically to either state.

Recorded outputs per rebalance:

```
hmm_probability_used_for_reporting = original probability (unchanged)
covariance_used_for_allocation     = unconditional Ledoit-Wolf
fallback_used                      = True
fallback_reason                    = "absorbing_transition"
```

**Rationale.** The transition estimate sits on the parameter boundary,
and horizon-averaging `p_t P_t^h` would otherwise mechanically assign
100% probability to the high-volatility state for the entire holding
period. Unconditional shrinkage is conservative and already consistent
with the degeneracy policy the plan applies to low occupancy. Four of
200 origins are affected (2009-12-31, 2012-03-30, 2012-04-30,
2012-05-31).

**State of knowledge when adopted:** the four affected dates, their
flags, and their probabilities were known. **No covariance estimate,
portfolio weight, return, or performance statistic had been computed or
examined**, so no performance information was available to motivate the
rule.

Explicitly excluded (none of these is done): replacing the probability,
interpolating from neighboring months, deleting the affected rebalance,
refitting until a preferred transition matrix appears, or changing the
absorbing threshold after seeing portfolio results.

The same fallback treatment covers degenerate occupancy (the
preregistered rule, reason `degenerate_occupancy`) and singular state
covariance (reason `singular_covariance`); neither occurred in this
sample, so neither affects any result.

**Robustness requirement:** an `accept_as_estimated` case is added to
the frozen robustness grid, using the original regime-conditioned
covariance at those four dates, to show whether the conservative
fallback materially affects conclusions.

### Amendment A3 — 2026-08-15 (no covariance result had been computed)

**The regime mixture covariance is a within-state mixture; the
between-state mean term is a robustness case.**

The frozen plan writes the mixture as `Σ_RA,t = Σ_k p̄_{t,k} Σ_{k,t}`
without stating whether `Σ_{k,t}` is a centered covariance or an
uncentered second moment. This amendment resolves the ambiguity before
any covariance was estimated.

**Main specification.** `Σ_{k,t}` is the **centered** responsibility-
weighted covariance around its own state mean `μ_{k,t}`, and the mixture
is the plan's formula as written:

```
Σ_RA,t = Σ_k p̄_{t,k}(H) · C_{k,t}
```

This omits the between-state mean-dispersion term and therefore
implicitly treats the conditional means as common across states. That
is a deliberate simplification, consistent with the frozen plan's
prohibition on expected-return forecasting: it keeps estimated
conditional means — the noisiest quantities in the problem — out of the
allocation entirely. It is documented here as an assumption, not
presented as the complete total covariance.

**Robustness case `total_covariance_mixture`.** The complete law of
total covariance, added to the frozen robustness grid:

```
μ̄_t     = Σ_k p̄_{t,k} μ_{k,t}
Σ_RA,t  = Σ_k p̄_{t,k} [ C_{k,t} + (μ_{k,t} − μ̄_t)(μ_{k,t} − μ̄_t)ᵀ ]
```

**Reporting requirement.** Every rebalance records the Frobenius norm of
the omitted between-state term relative to the within-state mixture, so
the magnitude of the assumption is visible in the outputs rather than
argued in prose. At daily frequency the term is expected to be small
(state mean differences are order basis points while daily variances are
order 1e-4), but that is an empirical question the audit answers rather
than an assertion.

Adopted before any covariance matrix, eigenvalue, weight, or return had
been computed.

### Amendment A4 — 2026-08-15 (no portfolio return had been computed)

**Initial entry and multiplicative cost accounting.**

Two accounting conventions the frozen plan left implicit, fixed before
Phase 9 generated any return.

**(a) Initial portfolio entry.** At the first execution date every
strategy begins from 100% cash immediately before trading:

```
w_0^-      = (0, 0, 0, 0, 0)
TradeSum_0 = sum_i |w_0,i^target - 0| = 1
```

At 10 bps every fully invested strategy therefore pays a **10 bps entry
cost**. This is identical across strategies and prevents silently
granting cost-free initial positions. No return accrues before the
first execution. Audit fields `initial_entry`, `pretrade_cash_weight`,
and `full_trade_sum` record it.

**(b) Multiplicative cost and return timing.** For an execution at the
close of date `t`:

1. old holdings earn the close-to-close return ending at `t`;
2. pre-trade weights are computed **after** that return;
3. trades occur at that close;
4. costs are deducted from pre-trade wealth;
5. new weights begin earning from `t -> t+1`.

The accounting identity is multiplicative, never additive:

```
V_t^-       = V_{t-1}^+ (1 + r_t^gross)
TC_t        = c * sum_i |w_{t,i}^target - w_{t,i}^-|
V_t^+       = V_t^- (1 - TC_t)
1 + r_t^net = (1 + r_t^gross)(1 - TC_t)
```

`net = gross - cost` is **not** used even though the difference is
numerically small; the audit records `wealth_identity_error` so the
identity is verified rather than assumed.

**(c) Daily drift.** Pre-trade weights come from applying the drift
recurrence **every trading day**, not once per month from cumulative
returns:

```
w_{i,t}^- = w_{i,t-1} (1 + r_{i,t}) / sum_j w_{j,t-1} (1 + r_{j,t})
```

so stored daily holdings reconcile exactly with the return series.

**(d) Returns convention.** Phase 9 accounting uses **simple** returns,
because weight drift and wealth compounding are multiplicative in
simple returns. Phases 5-8 use log returns for volatility and
covariance estimation, which is standard and unaffected.

**(e) Cost independence of holdings.** Costs are applied as a wealth
multiplier and never alter the weight vector or the targets, so the
holdings path and gross returns are identical across all cost
scenarios. Phase 9 computes the weight path once and applies each cost
level to it, making this property structural rather than merely tested.

Adopted before any portfolio return, cost, turnover, or wealth path had
been computed.
