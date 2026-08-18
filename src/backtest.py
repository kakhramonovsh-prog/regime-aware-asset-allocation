"""Walk-forward portfolio accounting (Phase 9).

Implements the frozen timeline and the Amendment A4 conventions:

* signal at the close of month-end ``t_signal``, execution at the close
  of the **next trading day** ``t_exec``; new weights first earn the
  ``t_exec -> t_exec+1`` return,
* holdings drift by the daily recurrence
  ``w_i (1+r_i) / sum_j w_j (1+r_j)`` applied **every trading day**,
* the first execution starts from 100% cash, so its trade sum is 1 and
  every strategy pays the same entry cost,
* costs are charged on the **full** trade sum
  ``sum_i |w_target - w_pretrade|`` (the halved figure is reporting
  only) and applied multiplicatively:
  ``1 + r_net = (1 + r_gross)(1 - TC)``,
* a ``fallback_requested`` rebalance is a **no trade**: holdings keep
  drifting and no cost is charged.

Returns here are **simple** returns, because drift and wealth
compounding are multiplicative in simple returns. Phases 5-8 use log
returns for estimation, which is unaffected.

Costs never alter the weight path or the targets, so the holdings and
gross returns are computed once and each cost level is applied to that
single path. Cost-independence of holdings is therefore structural, not
merely asserted.

This module computes no Sharpe ratio, drawdown, cumulative-return
ranking, or strategy selection: those belong to Phase 10 onward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps_scenarios: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    main_cost_bps: float = 10.0


def execution_dates_for(
    signal_dates: pd.DatetimeIndex, trading_days: pd.DatetimeIndex
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each signal date to the following trading day's close.

    A signal observed at the close of ``t`` cannot be traded at that
    same close, so execution is the next available trading day. Signals
    with no subsequent trading day are dropped.
    """
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal in signal_dates:
        later = trading_days[trading_days > signal]
        if len(later):
            mapping[pd.Timestamp(signal)] = pd.Timestamp(later[0])
    return mapping


def drift(weights: np.ndarray, day_returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Advance holdings one trading day.

    Returns ``(drifted_weights, portfolio_gross_return)``. With weights
    summing to one, ``sum_i w_i (1+r_i) = 1 + r_portfolio``, so the
    growth factor yields the gross return directly. An empty (all-cash)
    portfolio earns nothing and stays empty.
    """
    grown = weights * (1.0 + day_returns)
    total = grown.sum()
    if total <= 0 or weights.sum() <= 0:
        return weights.copy(), 0.0
    return grown / total, float(total - weights.sum())


def simulate_path(
    daily_returns: pd.DataFrame,
    targets: dict[pd.Timestamp, np.ndarray | None],
    start: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Simulate the cost-independent holdings path and gross returns.

    ``targets`` maps **execution dates** to target weight vectors;
    ``None`` marks a fallback request (no trade). The simulation starts
    at the first execution date from 100% cash and runs to the end of
    ``daily_returns``.

    Returns frames for pre-trade weights, post-trade weights, trades,
    turnover, gross returns, and a per-day event record. No cost is
    applied here: costs multiply wealth and never change these
    quantities.
    """
    assets = list(daily_returns.columns)
    execution_dates = sorted(targets)
    if not execution_dates:
        raise ValueError("no execution dates supplied")
    first_execution = pd.Timestamp(start or execution_dates[0])
    dates = daily_returns.index[daily_returns.index >= first_execution]

    holdings = np.zeros(len(assets))
    pretrade_rows, posttrade_rows, trade_rows, event_rows = [], [], [], []

    for date in dates:
        is_execution = date in targets

        if date == first_execution:
            # A4(a): begin from 100% cash; no return accrues before the
            # first execution.
            gross_return = 0.0
            pretrade = np.zeros(len(assets))
        else:
            pretrade, gross_return = drift(holdings, daily_returns.loc[date].to_numpy())

        target = targets.get(date) if is_execution else None
        trade_requested = is_execution
        fallback = is_execution and target is None
        trade_executed = bool(is_execution and target is not None)

        if trade_executed:
            trades = target - pretrade
            posttrade = target.copy()
        else:
            # No trade: holdings simply continue drifting.
            trades = np.zeros(len(assets))
            posttrade = pretrade.copy()

        full_trade_sum = float(np.abs(trades).sum())
        holdings = posttrade

        pretrade_rows.append({"date": date, **dict(zip(assets, pretrade))})
        posttrade_rows.append({"date": date, **dict(zip(assets, posttrade))})
        trade_rows.append({"date": date, **dict(zip(assets, trades))})
        event_rows.append(
            {
                "date": date,
                "rebalance": bool(is_execution),
                "initial_entry": bool(date == first_execution),
                "pretrade_cash_weight": float(1.0 - pretrade.sum()),
                "fallback_requested": bool(fallback),
                "trade_executed": trade_executed,
                "trade_requested": trade_requested,
                "full_trade_sum": full_trade_sum,
                "half_turnover_reporting": full_trade_sum / 2.0,
                "gross_return": float(gross_return),
                "pretrade_weight_sum": float(pretrade.sum()),
                "posttrade_weight_sum": float(posttrade.sum()),
            }
        )

    def _frame(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows).set_index("date")

    return {
        "pretrade_weights": _frame(pretrade_rows),
        "posttrade_weights": _frame(posttrade_rows),
        "trades": _frame(trade_rows),
        "events": _frame(event_rows),
    }


def apply_costs(events: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Apply one cost level to a simulated path.

    Cost fraction is ``c * full_trade_sum`` with ``c = cost_bps / 1e4``,
    charged on the execution date. Net returns follow the multiplicative
    identity; ``wealth_identity_error`` verifies it rather than assuming
    it.
    """
    c = cost_bps / 1e4
    cost_fraction = c * events["full_trade_sum"].to_numpy()
    gross = events["gross_return"].to_numpy()
    net = (1.0 + gross) * (1.0 - cost_fraction) - 1.0

    wealth = np.cumprod(1.0 + net)
    identity = np.cumprod((1.0 + gross) * (1.0 - cost_fraction))

    return pd.DataFrame(
        {
            "cost_bps": cost_bps,
            "cost_fraction": cost_fraction,
            "gross_return": gross,
            "net_return": net,
            "wealth": wealth,
            "wealth_identity_error": np.abs(wealth - identity),
        },
        index=events.index,
    )


def run_backtest(
    daily_returns: pd.DataFrame,
    targets_by_strategy: dict[str, dict[pd.Timestamp, np.ndarray | None]],
    cfg: BacktestConfig = BacktestConfig(),
    signal_dates: dict[pd.Timestamp, pd.Timestamp] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the accounting layer for every strategy and cost scenario.

    Returns long-format frames keyed by output name, each carrying a
    ``strategy`` column, plus a combined accounting audit.
    """
    signal_lookup = {v: k for k, v in (signal_dates or {}).items()}

    collected: dict[str, list[pd.DataFrame]] = {
        "pretrade_weights": [], "posttrade_weights": [], "trades": [],
        "turnover": [], "gross_returns": [], "audit": [],
    }
    net_by_scenario: dict[float, list[pd.DataFrame]] = {
        bps: [] for bps in cfg.cost_bps_scenarios
    }

    for strategy, targets in targets_by_strategy.items():
        path = simulate_path(daily_returns, targets)
        events = path["events"]

        for name in ("pretrade_weights", "posttrade_weights", "trades"):
            frame = path[name].reset_index()
            frame.insert(1, "strategy", strategy)
            collected[name].append(frame)

        turnover = events[["full_trade_sum", "half_turnover_reporting",
                           "rebalance", "trade_executed"]].reset_index()
        turnover.insert(1, "strategy", strategy)
        collected["turnover"].append(turnover)

        gross = events[["gross_return"]].reset_index()
        gross.insert(1, "strategy", strategy)
        collected["gross_returns"].append(gross)

        for bps in cfg.cost_bps_scenarios:
            costed = apply_costs(events, bps).reset_index()
            costed.insert(1, "strategy", strategy)
            net_by_scenario[bps].append(costed)

            audit = costed.merge(
                events.reset_index()[
                    ["date", "rebalance", "initial_entry", "fallback_requested",
                     "trade_executed", "full_trade_sum", "half_turnover_reporting",
                     "pretrade_weight_sum", "posttrade_weight_sum"]
                ],
                on="date",
            )
            audit["execution_date"] = audit["date"].where(audit["rebalance"])
            audit["signal_date"] = audit["execution_date"].map(signal_lookup)
            collected["audit"].append(audit)

    out = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in collected.items() if frames
    }
    for bps, frames in net_by_scenario.items():
        out[f"net_returns_{int(bps)}bps"] = pd.concat(frames, ignore_index=True)
    return out
