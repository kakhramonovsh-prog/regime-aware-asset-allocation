# Phase 10 — Performance Metrics

**Generated from:** config SHA-256 · data snapshot `manifest_2026-08-06.json` ·
git commit per [phase10_manifest.json](phase10_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py --phase 10`
**Scope:** descriptive metrics computed from the **stored Phase 9 return
series**; the backtest was not re-run. Every difference below is a
**point estimate**. Nothing here is described as statistically
significant — whether any of it survives inference is Phase 11's
question, and the answer is not yet known.

Sample: 2010-01-04 to 2026-08-06, 4,173 trading days (~16.6 years).

---

## 1. Primary result (net of 10 bps costs)

| Strategy | CAGR | Ann. vol | **Sharpe** | Sortino | Max DD | Calmar | Terminal wealth |
|---|---|---|---|---|---|---|---|
| Equal weight | 11.61% | 11.90% | 0.857 | 1.214 | −22.80% | 0.509 | 6.18× |
| 60/40 | 9.82% | 9.78% | 0.852 | 1.207 | −21.20% | 0.463 | 4.73× |
| Static min-var | 9.04% | 8.23% | 0.910 | 1.301 | −19.17% | 0.471 | 4.20× |
| **Rolling LW min-var** (comparator) | 8.72% | 7.88% | **0.908** | 1.293 | −17.41% | 0.500 | 4.00× |
| EWMA-scaled min-var | 8.70% | 7.54% | 0.945 | 1.356 | −17.30% | 0.503 | 3.99× |
| **Regime-aware min-var** (primary) | 8.73% | 7.71% | **0.929** | 1.331 | −17.44% | 0.501 | 4.01× |

**The preregistered primary comparison:**

```
Sharpe(regime-aware, net 10bps)  = 0.9292
Sharpe(rolling LW,   net 10bps)  = 0.9083
Difference                       = +0.0210
```

The regime-aware strategy's Sharpe ratio is **higher than its
comparator's by 0.021**, achieved through a **0.175 pp lower annualized
volatility** (7.709% vs 7.884%) at essentially identical CAGR (+0.018 pp)
and essentially identical maximum drawdown (−17.44% vs −17.41%).

**This is a point estimate and nothing more.** A difference of 0.021 in
Sharpe over ~16.6 years is small in absolute terms, and no confidence
interval has been computed. It must not be called an improvement,
outperformance, or evidence of anything until Phase 11 attaches an
interval to it. The honest present statement is: *the regime-aware
strategy's realized Sharpe ratio was higher than the comparator's by
0.021 in this sample.*

Read down the ablation ladder, the pattern is that **most of the
risk-adjusted gain over the naive benchmarks comes from optimization
itself** (equal weight 0.857 → static min-var 0.910), with dynamic
covariance adding little (0.908) and volatility/regime conditioning
adding modest further amounts (0.945 and 0.929). Whether those last
increments are distinguishable from noise is exactly what Phase 11 must
determine.

Note also that the **highest Sharpe belongs to the EWMA-scaled strategy
(0.945), not the regime-aware one** — a result the preregistration did
not predict and which is reported as found. EWMA-scaled is a ladder rung,
not the primary strategy, so this does not change the preregistered
hypothesis; it is recorded because suppressing it would be selective
reporting.

## 2. Cost sensitivity

Sharpe by cost scenario (0/5/20 bps are sensitivity, 10 bps is primary):

| Strategy | 0 bps | 5 bps | **10 bps** | 20 bps |
|---|---|---|---|---|
| Equal weight | 0.8598 | 0.8582 | 0.8566 | 0.8534 |
| 60/40 | 0.8553 | 0.8539 | 0.8524 | 0.8494 |
| Static min-var | 0.9140 | 0.9119 | 0.9098 | 0.9055 |
| Rolling LW min-var | 0.9126 | 0.9104 | **0.9083** | 0.9040 |
| EWMA-scaled min-var | 0.9705 | 0.9575 | 0.9445 | 0.9184 |
| Regime-aware min-var | 0.9432 | 0.9362 | **0.9292** | 0.9152 |

The regime-aware advantage over the comparator **shrinks monotonically
as costs rise**: +0.031 at 0 bps, +0.026 at 5 bps, +0.021 at 10 bps,
+0.011 at 20 bps. This is the direct consequence of the Phase 9 finding
that it trades 3.2× as much (54.2% vs 16.9% annual half-turnover). At
20 bps roughly two-thirds of the gross advantage has been consumed by
trading. The EWMA-scaled strategy, which trades most of all, decays
fastest in absolute terms (0.971 → 0.918).

## 3. Tail risk (losses reported as positive numbers)

Historical daily VaR and Expected Shortfall, net of 10 bps:

| Strategy | VaR 95% | VaR 99% | ES 95% | ES 99% |
|---|---|---|---|---|
| Equal weight | 1.148% | 2.030% | 1.759% | 2.855% |
| 60/40 | 0.937% | 1.667% | 1.471% | 2.450% |
| Static min-var | 0.792% | 1.379% | 1.210% | 1.929% |
| Rolling LW min-var | 0.772% | 1.348% | 1.161% | 1.874% |
| EWMA-scaled min-var | 0.732% | 1.310% | 1.099% | 1.710% |
| Regime-aware min-var | 0.751% | 1.353% | 1.128% | 1.781% |

The regime-aware portfolio has slightly thinner tails than the
comparator at the 95% level (ES 1.128% vs 1.161%) and essentially the
same at 99% (1.781% vs 1.874%). Again: differences, not verdicts.

## 4. Metric definitions (frozen before computation)

- **CAGR** `(V_T/V_0)^(365.25/D) − 1` over elapsed calendar days, with
  the initial entry cost included in the wealth path.
- **Annualized volatility** `std(r) × √252`.
- **Risk-free** `r_f,t = (DFF_{t−1}/100) × (calendar days/360)`, using
  the last rate known at the start of each interval; zero on the
  cost-only entry row.
- **Sharpe** `mean(r − r_f)/std(r − r_f) × √252`.
- **Sortino** same numerator, denominator `√(mean(min(r − r_f, 0)²))`.
- **Drawdown** from the net wealth path; **Calmar** `CAGR/|max DD|`.
- **VaR/ES** historical, 95% and 99%, **losses as positive numbers**.

## 5. Safeguards (checked, not assumed)

`safeguard_checks.csv`, all passed:

| Check | Result |
|---|---|
| Zero-cost net metrics equal gross metrics | max gap **5.9e-15** |
| Higher costs weakly lower terminal wealth | holds for all 6 strategies |
| CAGR reconciles with first/last wealth | max error **6.7e-15** |
| All strategies share identical dates | true in all 4 scenarios |

Metrics consume the stored Phase 9 files and never recreate the
backtest. Rolling Sharpe uses a trailing window, verified by a test that
truncates the sample and confirms earlier values are unchanged. Tables
are stored at full precision; rounding appears only in this document.

## 6. Artifacts

`outputs/performance/performance_summary.csv` (30 rows: 6 strategies ×
4 cost scenarios plus gross, full precision), `safeguard_checks.csv`,
`wealth_paths_10bps.parquet`, `drawdown_paths_10bps.parquet`,
`rolling_sharpe_10bps.parquet`; figures `cumulative_growth.png`,
`strategy_drawdowns.png`, `rolling_sharpe.png`,
`weights_regime_minvar.png`, `weights_rolling_lw_minvar.png`,
`turnover.png`; [phase10_manifest.json](phase10_manifest.json).

Reproduce with `python scripts/run_analysis.py --phase 10` after Phase 9.
Tests: 20 Phase 10 unit tests covering every frozen definition, the
CAGR/wealth reconciliation, the manual drawdown check, the VaR/ES sign
convention, and rolling-Sharpe causality.

## 7. What Phase 11 must decide

The descriptive answer to the research question is: **in this sample,
regime conditioning produced a Sharpe ratio 0.021 higher than the
closest non-regime strategy, net of 10 bps costs, by reducing volatility
rather than raising return.** Phase 11 attaches a stationary-bootstrap
confidence interval to that difference. Given ~199 monthly observations,
a wide interval that includes zero is a plausible outcome, and if that
is what the data say, that is what will be reported.
