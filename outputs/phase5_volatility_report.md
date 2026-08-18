# Phase 5 — Volatility Forecasting Report

**Generated from:** config SHA-256 `54ccd33d…855d6c` · data snapshot
`manifest_2026-08-06.json` · git commit per
[phase5_manifest.json](phase5_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py --phase 5`
**Scope:** volatility models only, estimated through each rebalance
date. **No portfolio returns, regime model, or strategy selection was
computed or examined.** All numbers below are read from the generated
tables; per-artifact hashes are in the manifest.

---

## Setup (as frozen)

Three models — 63-day rolling historical, EWMA (λ = 0.94, seeded with
the first 63 days' sample variance), GARCH(1,1) with normal innovations
refit at every origin — each estimated on expanding windows of daily
log returns **through the forecast origin only**. Forecast object:
integrated variance of the next holding period (trading days after the
month-end origin through the next month-end), evaluated against
realized integrated variance (sum of squared daily log returns).
249 month-end origins per asset, 2005-11-30 to 2026-06-30 (the final
partial month is excluded); identical dates for all models, enforced by
an alignment assertion. Losses: QLIKE primary, MAE/RMSE (variance
units) secondary. Daily variance floor 1e-8.

### Implementation decisions and their provenance (stated honestly)

Four choices are **not** part of the design frozen at
`v0.2.0-preregistered`:

1. zero-mean Gaussian GARCH specification (no conditional mean model),
2. returns scaled by 100 for optimizer stability,
3. HAC lag 3 for Diebold-Mariano standard errors,
4. excluding the sample's final partial month (not a full holding
   period).

They were written into `src/volatility.py` and its tests before the
estimation run executed, but they were first committed to git in the
same commit as the results, so git alone cannot prove they preceded the
numbers. They are therefore reported as **pre-estimation implementation
decisions, not preregistered commitments**. None was revised after
seeing output. The frozen plan fixed what mattered for the portfolio:
the model set, the estimation scheme, the loss functions, and the
identity of the portfolio-feeding model.

## Estimation diagnostics

All **1,245 GARCH refits converged (0 substitutions, 0 floored
forecasts)** — `volatility_estimation_log.csv`. The EWMA fallback path
exists and is unit-tested, but was never triggered on this data.

## Results (`volatility_loss_by_asset.csv`)

Mean QLIKE, full evaluation window (249 months; OOS-2010+ subset in the
table, same ordering everywhere):

| Asset | Hist63 | EWMA | GARCH(1,1) | Lowest |
|---|---|---|---|---|
| SPY | 0.544 | 0.424 | **0.326** | GARCH |
| QQQ | 0.421 | 0.338 | **0.271** | GARCH |
| IWM | 0.384 | 0.284 | **0.235** | GARCH |
| GLD | 0.287 | 0.270 | **0.224** | GARCH |
| IEF | 0.173 | 0.177 | **0.158** | GARCH |

**GARCH(1,1) attains the lowest QLIKE for all five assets**, in both
the full window and the 2010+ subset, and it also leads MAE/RMSE for
most assets.

### Effect sizes before significance

`forecast_comparisons.csv` reports the mean QLIKE differential with its
HAC(3) standard error and 95% confidence interval; negative favors the
first model. Selected rows (n = 249 each):

| Comparison | Asset | Mean diff | 95% CI | p (HAC) | p (Holm, 15 tests) |
|---|---|---|---|---|---|
| GARCH vs EWMA | QQQ | −0.067 | [−0.107, −0.027] | 0.001 | **0.016** |
| GARCH vs EWMA | SPY | −0.098 | [−0.172, −0.023] | 0.011 | 0.138 |
| GARCH vs EWMA | IWM | −0.049 | [−0.093, −0.005] | 0.031 | 0.336 |
| GARCH vs hist63 | GLD | −0.063 | [−0.110, −0.017] | 0.008 | 0.109 |
| GARCH vs hist63 | SPY | −0.217 | [−0.419, −0.016] | 0.035 | 0.345 |
| EWMA vs hist63 | SPY | −0.120 | [−0.278, +0.038] | 0.138 | 0.790 |

Full family: **all ten GARCH comparisons have negative point
estimates** (GARCH favored), and seven of ten have unadjusted p < 0.10.
EWMA beats hist63 on point estimates for four of five assets but every
one of those confidence intervals contains zero.

### Multiplicity

Holm-adjusted p-values across the family of 15 comparisons appear as a
secondary column; the unadjusted HAC(3) results are unchanged. **After
Holm adjustment only one comparison remains significant at 5%: GARCH
vs EWMA for QQQ (p_holm = 0.016).** SPY's GARCH-vs-EWMA result
(p_holm = 0.138) and every GARCH-vs-hist63 result (p_holm ≥ 0.109) do
not survive.

The defensible statement is therefore: **GARCH's point-estimate
advantage is consistent in sign across all five assets and both
comparisons, but individually the differences are mostly not
statistically distinguishable from zero once multiplicity is taken into
account.** Consistency of sign across assets is the stronger evidence
here; no single test carries the claim. With ~249 monthly observations
and volatility-clustered losses, this is the expected power situation
rather than a surprise.

None of these 15 tests is the preregistered primary hypothesis (that is
the Phase 9-11 portfolio comparison), so the Holm family is confined to
this forecast-evaluation exercise.

## The preregistration consequence, stated plainly

The frozen plan fixed **EWMA as the portfolio-feeding model regardless
of this comparison** (chosen ex ante for determinism and zero
convergence risk). The comparison now says GARCH forecasts holding-
period variance better. Both statements stand:

1. The volatility-scaled strategy in the main analysis uses EWMA,
   exactly as preregistered. This result does not and cannot promote
   GARCH into the main specification.
2. A GARCH-fed variant belongs in the robustness analysis (the frozen
   grid already lists "different volatility models"), where it will be
   labeled robustness, not headline.

This is the designed behavior of the preregistration: a better-looking
alternative discovered after the freeze changes the robustness section,
never the main specification.

## Reading the figure

`forecasts_vs_realized_spy.png` (annualized-vol transform for
readability; losses are computed on variances): all three forecasts
lag realized volatility at crisis onsets — necessarily, being formed
before the period they forecast — and the 63-day historical model both
arrives latest at spikes and decays slowest afterward (visible after
2008-09, 2011, 2020), which is the mechanical source of its QLIKE
deficit. GARCH tracks the post-spike decay best; that pattern, not
level accuracy in calm periods, is where its edge comes from.

## Artifacts

`outputs/forecasts/volatility_forecasts.parquet` (3,735 rows: 249
origins x 5 assets x 3 models, forecasts + realized targets + flags),
`outputs/tables/volatility_loss_by_asset.csv`,
`outputs/tables/forecast_comparisons.csv`,
`outputs/tables/volatility_estimation_log.csv`,
`outputs/figures/forecasts_vs_realized_spy.png`,
[phase5_manifest.json](phase5_manifest.json) (SHA-256 of each artifact,
git commit, config and data hashes). Reproduce:
`python scripts/run_analysis.py --phase 5` (refuses to run off-snapshot).
Tests: 12 Phase 5 unit tests cover the EWMA recursion, QLIKE formula,
holding-period alignment, causality under future perturbation, the
GARCH failure/substitution path, and the variance floor.
