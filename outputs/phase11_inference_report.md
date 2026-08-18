# Phase 11 — Statistical Inference

**Generated from:** config SHA-256 · data snapshot `manifest_2026-08-06.json` ·
git commit per [phase11_manifest.json](phase11_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py --phase 11`

Inference on the preregistered primary hypothesis, using the **same
estimand Phase 10 reported**: annualized Sharpe computed from **paired
daily excess returns** with √252 annualization. Monthly returns are not
resampled, because doing so would change the estimand.

---

## 1. The preregistered primary result

**H0:** `SR_net(regime-aware) − SR_net(rolling Ledoit-Wolf) ≤ 0` at 10 bps.

| Quantity | Value |
|---|---|
| Observed Sharpe difference | **+0.0210** |
| 95% percentile confidence interval | **[−0.0748, +0.1147]** |
| **Interval excludes zero** | **No** |
| Bootstrap standard deviation | 0.0480 |
| One-sided p-value (centered null) | **0.327** |
| Replications / seed / mean block | 10,000 / 12345 / 21 days |
| Daily observations | 4,173 |

**H0 is not rejected.** The point estimate is positive, but the
confidence interval is roughly nine times as wide as the estimate and
comfortably contains zero. The estimated difference is about 0.44
bootstrap standard deviations from zero.

The defensible statement is:

> Over 2010-2026, the regime-aware minimum-variance strategy achieved a
> Sharpe ratio 0.021 higher than the closest non-regime strategy, net of
> 10 bps costs. That difference is not statistically distinguishable
> from zero: the 95% interval runs from −0.075 to +0.115.

It must **not** be described as outperformance, improvement, or evidence
that regime conditioning works.

## 2. Block-length sensitivity

Sensitivity checks, never a menu to select from — the 21-day result
above stands regardless:

| Mean block | Role | 95% interval | Excludes zero |
|---|---|---|---|
| **21 days** | **PRIMARY** | [−0.0748, +0.1147] | No |
| 10 days | sensitivity | [−0.0754, +0.1149] | No |
| 42 days | sensitivity | [−0.0715, +0.1150] | No |
| 63 days | sensitivity | [−0.0713, +0.1193] | No |

The conclusion is insensitive to block length: every interval is nearly
identical and every one contains zero. The dependence structure is not
driving the result.

## 3. HAC inference on mean returns

A different question from the Sharpe comparison: do average returns
differ? Paired daily net return differences, Newey-West standard errors:

| HAC lags | Role | Annualized difference | 95% interval | p (two-sided) |
|---|---|---|---|---|
| **21** | **PRIMARY** | **+0.314 bps** | [−70.7, +71.3] bps | 0.993 |
| 5 | sensitivity | +0.314 bps | [−71.3, +71.9] bps | 0.993 |
| 42 | sensitivity | +0.314 bps | [−70.5, +71.1] bps | 0.993 |

The annualized arithmetic return difference is **0.314 basis points**
against an annualized HAC standard error of **36.2 bps** — the estimate
sits **0.0087 standard errors from zero**, which is why the p-value is
0.993. Regime conditioning did not change returns in any detectable way,
consistent with the Phase 10 finding that the two CAGRs differ by
0.018 pp.

**Scale verification.** Because a p-value that close to one invites the
suspicion of a scaling error, the quantities are shown explicitly and
the equivalence is unit-tested:

| Quantity | Value |
|---|---|
| Daily mean difference | 1.246e-07 |
| Daily HAC standard error | 1.437e-05 |
| Annualized mean difference (×252) | 3.140e-05 (0.314 bps) |
| Annualized HAC standard error (×252) | 3.622e-03 (36.2 bps) |
| t on the daily scale | 0.0086713 |
| t on the annualized scale | **0.0086713** |
| Two-sided p (statsmodels / manual normal) | 0.993081 / 0.993081 |

Both the estimate and its standard error are annualized by the same
linear factor of 252, so the t-statistic and p-value are invariant to
scale; a test asserts this equality and would fail if an annualized
mean were ever compared against a daily or √252-scaled standard error.
The p-value of 0.993 is retained as correct.

**Diebold-Mariano is deliberately not applied here.** DM compares
forecast losses, which is the Phase 5 exercise; it is not a valid test
of whether one investment strategy outperformed another. A test asserts
no DM function exists in the inference module.

## 4. Secondary family A — other metrics, same comparison (unadjusted)

**This is a distinct family from §5.** Here the *comparison* is fixed
(regime-aware vs rolling Ledoit-Wolf at 10 bps) and the *metric* varies.
These intervals are **unadjusted for multiplicity**. Section 5 holds the
metric fixed (Sharpe) and varies the comparison, with Holm adjustment
applied there. Readers should not read the two sections as
contradictory: they answer different questions and carry different
adjustment regimes.

Paired bootstrap differences, regime-aware minus rolling Ledoit-Wolf:

| Metric | Difference | 95% interval | Excludes zero | Role |
|---|---|---|---|---|
| Sharpe | +0.0210 | [−0.0748, +0.1147] | No | **CONFIRMATORY** |
| Sortino | +0.0379 | [−0.1068, +0.1785] | No | secondary |
| **Annualized volatility** | **−0.00175** | **[−0.00327, −0.00033]** | **Yes** | secondary |
| CAGR | +0.00018 | [−0.0075, +0.0079] | No | secondary |
| Max drawdown † | −0.00030 | [−0.0152, +0.0308] | No | secondary |
| Calmar † | +0.00020 | [−0.1062, +0.1827] | No | secondary |

† Path-dependent. Resampling reorders history, so bootstrap intervals
for drawdown and Calmar describe chronologies other than the realized
one and should be read with particular caution.

The only interval in this family that excludes zero is the volatility
difference: annualized volatility 0.175 pp lower (95% interval 0.033 to
0.327 pp lower). The correct characterization is:

> **The annualized-volatility difference is suggestive and consistent
> with the model's mechanism, but it is a secondary, unadjusted interval
> among multiple examined metrics and should not be treated as
> confirmatory evidence.**

It is **not** described as statistically established, because it has not
survived any explicitly defined multiplicity adjustment. Six metrics
were examined without adjustment; at 95% confidence roughly 0.3 false
exclusions are expected by chance alone. What raises it above an
incidental finding is that it is the outcome the mechanism *predicts* —
a variance minimizer given additional information about variance should
reduce variance — not that its interval happened to exclude zero.

The economically coherent reading is that **regime conditioning appears
to have done what it was designed to do (reduce risk), but the effect
was too small relative to sampling noise to establish an improvement in
risk-adjusted return once trading costs were paid.**

## 5. Secondary family B — other comparisons, same metric (Holm-adjusted)

**Distinct from §4.** Here the *metric* is fixed (Sharpe) and the
*comparison* varies across strategies and cost levels. Holm adjustment
is applied across this family of seven.

| Comparison | Cost | Difference | 95% interval | p (centered) | p (Holm) |
|---|---|---|---|---|---|
| EWMA-scaled − rolling LW | 10 bps | +0.0362 | [−0.112, +0.186] | 0.312 | **1.000** |
| Static min-var − rolling LW | 10 bps | +0.0015 | [−0.058, +0.062] | 0.478 | 1.000 |
| Equal weight − rolling LW | 10 bps | −0.0517 | [−0.312, +0.219] | 0.650 | 1.000 |
| 60/40 − rolling LW | 10 bps | −0.0559 | [−0.382, +0.289] | 0.623 | 1.000 |
| Regime − rolling LW | 0 bps | +0.0306 | [−0.064, +0.124] | 0.257 | 1.000 |
| Regime − rolling LW | 5 bps | +0.0258 | [−0.069, +0.119] | 0.289 | 1.000 |
| Regime − rolling LW | 20 bps | +0.0113 | [−0.085, +0.106] | 0.407 | 1.000 |

**Nothing in the secondary family survives Holm adjustment; every
adjusted p-value is 1.000, and every interval contains zero.** Notably
the EWMA-scaled strategy, which had the highest realized Sharpe in
Phase 10 (0.945), is also not statistically distinguishable from the
comparator. The Phase 10 ranking is therefore a descriptive ordering,
not a demonstrated hierarchy.

The regime-aware advantage shrinks monotonically with costs at every
level (+0.031 → +0.026 → +0.021 → +0.011), and no cost scenario
produces an interval excluding zero.

## 6. p-value discipline

The one-sided p-values above come from the **centered bootstrap null**,
`ΔSR_b⁰ = ΔSR_b − ΔŜR`, which imposes a zero difference while retaining
the bootstrap's shape and dependence. The fraction of *ordinary*
bootstrap draws below zero is a different quantity and is never reported
as a p-value; a test asserts the two differ.

Both columns are resampled with **identical bootstrap indices** in every
replication, preserving contemporaneous pairing and therefore the strong
correlation between the two strategies. A test confirms that feeding the
same series twice produces exactly zero difference with zero variance —
which would be impossible under independent resampling.

## 7. Answer to the research question

> **Can regime identification and volatility-aware portfolio
> construction improve out-of-sample risk-adjusted performance relative
> to conventional static allocation?**

On this evidence: **not demonstrably.** The confirmatory Sharpe
difference of +0.021 is not distinguishable from zero (interval −0.075
to +0.115), and the return difference is essentially exactly zero
(0.314 bps annualized, p = 0.993). A suggestive but unadjusted secondary
interval indicates lower volatility (−0.175 pp annualized), consistent
with the mechanism but not confirmatory on its own. Higher turnover
consumed a substantial share of the gross advantage: the point estimate
falls from +0.031 at zero cost to +0.011 at 20 bps.

This is a negative result on the preregistered hypothesis, and it is
reported as such. The power limitation is real and was anticipated in
the design: ~16.6 years containing few independent volatility cycles
cannot resolve a Sharpe difference of this magnitude.

## 8. Artifacts

`outputs/inference/`: `primary_bootstrap_draws.parquet` (10,000 draws ×
6 metrics), `primary_inference_summary.csv`,
`block_length_sensitivity.csv`, `hac_mean_difference.csv`,
`secondary_metric_intervals.csv`,
`multiple_comparison_adjustments.csv`;
[phase11_manifest.json](phase11_manifest.json).

Reproduce with `python scripts/run_analysis.py --phase 11` after
Phase 10. Tests: 23 Phase 11 unit tests covering paired-index identity,
seed reproducibility, the degenerate identical-series case, block
construction, the estimand's daily/√252 alignment, the centered null,
the ban on mislabeling ordinary bootstrap probabilities, HAC on paired
differences, and the absence of Diebold-Mariano.
