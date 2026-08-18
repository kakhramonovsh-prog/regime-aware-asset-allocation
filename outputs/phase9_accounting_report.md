# Phase 9 — Walk-Forward Portfolio Accounting

**Generated from:** config SHA-256 · data snapshot `manifest_2026-08-06.json` ·
git commit per [phase9_manifest.json](phase9_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py --phase 9`
**Scope:** accounting only — drifted holdings, trades, turnover, costs,
gross and net returns. **No Sharpe ratio, drawdown, cumulative-return
ranking, or strategy selection was computed or examined.** A test
asserts no output column carries such a term. Cumulative wealth appears
solely as the accounting quantity the cost identity is checked against.

6 strategies × 4,173 trading days = **25,038 daily holding records**;
100,152 audit rows (one per day, strategy, and cost scenario).

---

## 1. Timeline as executed

| | |
|---|---|
| First signal | 2009-12-31 (preregistered) |
| First execution | **2010-01-04**, the next trading day |
| Executions | 200, one per month-end signal |
| Accounting period | 2010-01-04 to 2026-08-06 |

New target weights never earn the return ending on their own execution
date; that return belongs to the old holdings. This is verified by a
test that switches a portfolio entirely from one asset to another and
asserts the execution-day return equals the *outgoing* asset's return.

## 2. Amendment A4 conventions, verified in output

**(a) Initial entry from 100% cash.** Every strategy begins flat and
pays an identical entry cost:

| Strategy | Trade sum | Cost at 10 bps | Gross return | Net return |
|---|---|---|---|---|
| all six | **1.000** | **0.0010** | 0.0000 | **−0.0010** |

No strategy receives a cost-free initial position, and no return accrues
before the first execution.

**(b) Multiplicative identity.** `1 + r_net = (1 + r_gross)(1 − TC)`,
with the additive shortcut never used. The audit stores
`wealth_identity_error` for every row:

**Maximum wealth identity error across all 100,152 rows: 0.000e+00.**

**(c) Daily drift.** The recurrence
`w_i(1+r_i) / Σ_j w_j(1+r_j)` is applied on every trading day, not once
per month. A test reconstructs each day's pre-trade weights from the
previous day's post-trade weights and the day's returns, matching to
1e-12 across the whole path.

**(d) Simple returns** for accounting, log returns retained for
estimation in Phases 5-8.

**(e) Cost independence.** The weight path is simulated once and each
cost level applied to it, so holdings and gross returns are identical
across scenarios **by construction**. Confirmed in output: gross returns
are bit-identical across all four cost levels, and at 0 bps net returns
equal gross returns to 1.1e-16.

## 3. Trading activity

Annualized trading activity. **Half-turnover** is the conventional
reporting figure (`Σ|Δw| / 2`); **full traded notional** is what costs
are actually charged on (`Σ|Δw|`); the last column is the annualized
**cost expenditure**, i.e. the sum of charged cost fractions per year:

| Strategy | Half-turnover | Full traded notional | Cost expenditure at 10 bps |
|---|---|---|---|
| 60/40 | 14.4% | 28.8% | 2.88 bps |
| Rolling LW min-var | 16.9% | 33.8% | 3.39 bps |
| Static min-var | 17.6% | 35.2% | 3.53 bps |
| Equal weight | 18.9% | 37.8% | 3.78 bps |
| **Regime-aware min-var** | **54.2%** | **108.4%** | **10.84 bps** |
| **EWMA-scaled min-var** | **98.2%** | **196.4%** | **19.64 bps** |

The last column is deliberately labeled *cost expenditure*, not "cost
drag": the realized drag on CAGR depends on compounding and on when the
costs are charged, and is computed in Phase 10 rather than inferred
here.

Two observations, both descriptive:

**The ladder's responsive rungs trade far more.** Regime-aware
half-turnover is 3.2× the rolling Ledoit-Wolf comparator, and
EWMA-scaled is 5.8×. This is the mechanical counterpart of the Phase 8
finding that target dispersion rises across the ladder (0.026 for
rolling LW, 0.060 for both responsive strategies).

**Cost expenditure scales with it.** At the 10 bps primary assumption
the regime-aware strategy spends about 10.8 bps per year against the
comparator's 3.4 bps — a **hurdle of roughly 7.5 bps per year** that its
risk reduction must clear before it can improve net performance.
Whether it does is a Phase 10-11 question and is deliberately not
answered here.

Even the nominally static strategies trade: fixed targets must be
re-established each month after drift, which is why 60/40 shows 14.4%
annual half-turnover rather than zero. A test asserts this scheduled
rebalancing occurs.

## 4. Other integrity checks

- **Zero fallback requests** — Phase 8 produced no optimization
  failures, so no rebalance was skipped. The no-trade path is
  nonetheless implemented and unit-tested (no trade, zero cost,
  continued drift).
- **Zero trades on non-rebalance days**: holdings only change through
  drift between executions.
- **All post-entry pre-trade weight sums equal 1** to 1e-12; weights
  are finite and non-negative throughout.
- **Causality**: perturbing returns after any cutoff leaves every
  earlier weight, trade, and return bit-identical (tested).

## 5. A note carried forward for robustness

Phase 8 found **IEF pinned at the 40% cap at every origin in every
optimized strategy**. The cap is therefore economically binding rather
than incidental, and the preregistered 30% / 50% cap robustness cases
are **essential to interpreting the results, not optional decoration**.
The robustness section will report them as first-class evidence.

## 6. Artifacts

`outputs/backtests/`: `pretrade_weights.parquet`,
`posttrade_weights.parquet`, `trades.parquet`, `turnover.parquet`,
`gross_returns.parquet`, `transaction_costs.parquet`,
`net_returns_0bps.parquet`, `net_returns_5bps.parquet`,
`net_returns_10bps.parquet`, `net_returns_20bps.parquet`,
`accounting_audit.parquet` (date, strategy, signal_date,
execution_date, rebalance, initial_entry, fallback_requested,
trade_executed, full_trade_sum, half_turnover_reporting, cost_bps,
cost_fraction, gross_return, net_return, pretrade_weight_sum,
posttrade_weight_sum, wealth_identity_error);
[phase9_manifest.json](phase9_manifest.json).

Reproduce with `python scripts/run_analysis.py --phase 9` after Phase 8.
Tests: 20 Phase 9 unit tests covering the signal-to-execution mapping,
the ban on new weights earning their execution-date return, the entry
trade sum, the full-trade-sum cost convention, daily drift, cost
independence, zero-cost equality, no-trade cases, the multiplicative
identity, weight validity, causality, scheduled rebalancing, and the
absence of performance statistics.
