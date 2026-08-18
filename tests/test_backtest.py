"""Tests for Phase 9 portfolio accounting.

Covers every required property: signal-to-execution mapping, the ban on
new weights earning their own execution-date return, the initial entry
trade sum, the full-trade-sum cost convention, daily drift, cost
independence of holdings and gross returns, zero-cost equality, no-trade
cases, the multiplicative wealth identity, weight validity, causality,
scheduled rebalancing of fixed strategies, and the absence of any
performance statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest as bt

ASSETS = ["A", "B", "C"]


@pytest.fixture
def daily_returns() -> pd.DataFrame:
    """Deterministic small return panel (simple returns)."""
    idx = pd.bdate_range("2020-01-01", periods=40, name="date")
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        rng.normal(0.0004, 0.01, (40, 3)), index=idx, columns=ASSETS
    )


@pytest.fixture
def simple_targets(daily_returns):
    """Execute on day 5 and day 25 with an equal-weight target."""
    dates = daily_returns.index
    equal = np.full(3, 1 / 3)
    return {dates[5]: equal.copy(), dates[25]: equal.copy()}


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def test_signal_maps_to_next_trading_day(daily_returns):
    trading_days = daily_returns.index
    signals = pd.DatetimeIndex([trading_days[5], trading_days[20]])
    mapping = bt.execution_dates_for(signals, trading_days)
    assert mapping[trading_days[5]] == trading_days[6]
    assert mapping[trading_days[20]] == trading_days[21]


def test_signal_without_following_day_is_dropped(daily_returns):
    trading_days = daily_returns.index
    mapping = bt.execution_dates_for(
        pd.DatetimeIndex([trading_days[-1]]), trading_days
    )
    assert mapping == {}


def test_new_weights_do_not_earn_their_execution_day_return(daily_returns):
    """The return ending at the execution close belongs to the OLD
    holdings, never to the newly established target."""
    dates = daily_returns.index
    # Enter at day 0 into asset A only, then switch entirely to B at day 3.
    targets = {
        dates[0]: np.array([1.0, 0.0, 0.0]),
        dates[3]: np.array([0.0, 1.0, 0.0]),
    }
    path = bt.simulate_path(daily_returns, targets)
    events = path["events"]
    # On the switch date the gross return must equal asset A's return
    # (old holding), not asset B's.
    switch_return = events.loc[dates[3], "gross_return"]
    assert switch_return == pytest.approx(daily_returns.loc[dates[3], "A"], rel=1e-12)
    assert switch_return != pytest.approx(daily_returns.loc[dates[3], "B"], rel=1e-6)
    # The day AFTER the switch is earned by B.
    assert events.loc[dates[4], "gross_return"] == pytest.approx(
        daily_returns.loc[dates[4], "B"], rel=1e-12
    )


# ---------------------------------------------------------------------------
# Initial entry
# ---------------------------------------------------------------------------

def test_initial_entry_trade_sum_is_one(daily_returns, simple_targets):
    path = bt.simulate_path(daily_returns, simple_targets)
    events = path["events"]
    first = events.index[0]
    assert events.loc[first, "initial_entry"] is np.True_ or events.loc[first, "initial_entry"]
    assert events.loc[first, "full_trade_sum"] == pytest.approx(1.0)
    assert events.loc[first, "pretrade_weight_sum"] == pytest.approx(0.0)
    assert events.loc[first, "pretrade_cash_weight"] == pytest.approx(1.0)
    assert events.loc[first, "gross_return"] == 0.0   # no return before entry


def test_initial_entry_costs_the_same_for_every_fully_invested_strategy(daily_returns):
    dates = daily_returns.index
    for target in (np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.6, 0.4, 0.0]),
                   np.array([1.0, 0.0, 0.0])):
        path = bt.simulate_path(daily_returns, {dates[2]: target})
        costed = bt.apply_costs(path["events"], 10.0)
        assert costed["cost_fraction"].iloc[0] == pytest.approx(10.0 / 1e4)


# ---------------------------------------------------------------------------
# Trade sums and costs
# ---------------------------------------------------------------------------

def test_ten_percent_transfer_creates_twenty_percent_trade_sum(daily_returns):
    dates = daily_returns.index
    flat = pd.DataFrame(0.0, index=dates, columns=ASSETS)  # no drift
    start = np.array([0.5, 0.5, 0.0])
    moved = np.array([0.4, 0.6, 0.0])                      # 10% from A to B
    path = bt.simulate_path(flat, {dates[0]: start, dates[1]: moved})
    events = path["events"]
    assert events.loc[dates[1], "full_trade_sum"] == pytest.approx(0.20)
    assert events.loc[dates[1], "half_turnover_reporting"] == pytest.approx(0.10)


def test_cost_uses_full_trade_sum_not_half(daily_returns):
    dates = daily_returns.index
    flat = pd.DataFrame(0.0, index=dates, columns=ASSETS)
    path = bt.simulate_path(
        flat, {dates[0]: np.array([0.5, 0.5, 0.0]),
               dates[1]: np.array([0.4, 0.6, 0.0])}
    )
    costed = bt.apply_costs(path["events"], 10.0)
    # Trade sum 0.20 at 10bps costs 0.20 * 0.0010 = 2.0 bps of portfolio
    # value. Charging the halved turnover (0.10) would give 1.0 bp, so
    # this pins the convention to the FULL trade sum.
    full_sum_cost = 0.20 * 10.0 / 1e4
    halved_cost = 0.10 * 10.0 / 1e4
    assert costed.loc[dates[1], "cost_fraction"] == pytest.approx(full_sum_cost)
    assert costed.loc[dates[1], "cost_fraction"] == pytest.approx(2.0 / 1e4)
    assert costed.loc[dates[1], "cost_fraction"] != pytest.approx(halved_cost)


def test_target_equal_to_pretrade_produces_no_trade_and_no_cost(daily_returns):
    dates = daily_returns.index
    flat = pd.DataFrame(0.0, index=dates, columns=ASSETS)   # weights never drift
    equal = np.full(3, 1 / 3)
    path = bt.simulate_path(flat, {dates[0]: equal.copy(), dates[3]: equal.copy()})
    events = path["events"]
    assert events.loc[dates[3], "full_trade_sum"] == pytest.approx(0.0, abs=1e-15)
    costed = bt.apply_costs(events, 20.0)
    assert costed.loc[dates[3], "cost_fraction"] == pytest.approx(0.0, abs=1e-18)


def test_fallback_request_means_no_trade_no_cost_continued_drift(daily_returns):
    dates = daily_returns.index
    targets = {dates[0]: np.array([1.0, 0.0, 0.0]), dates[5]: None}
    path = bt.simulate_path(daily_returns, targets)
    events = path["events"]
    assert events.loc[dates[5], "fallback_requested"]
    assert not events.loc[dates[5], "trade_executed"]
    assert events.loc[dates[5], "full_trade_sum"] == 0.0
    costed = bt.apply_costs(events, 20.0)
    assert costed.loc[dates[5], "cost_fraction"] == 0.0
    # Holdings continue drifting: post-trade equals the drifted pre-trade.
    pre = path["pretrade_weights"].loc[dates[5]].to_numpy()
    post = path["posttrade_weights"].loc[dates[5]].to_numpy()
    np.testing.assert_allclose(pre, post)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def test_daily_pretrade_weights_equal_mechanical_drift(daily_returns, simple_targets):
    path = bt.simulate_path(daily_returns, simple_targets)
    pre = path["pretrade_weights"]
    post = path["posttrade_weights"]
    dates = pre.index
    for i in range(1, len(dates)):
        previous = post.iloc[i - 1].to_numpy()
        r = daily_returns.loc[dates[i]].to_numpy()
        grown = previous * (1 + r)
        expected = grown / grown.sum()
        np.testing.assert_allclose(pre.iloc[i].to_numpy(), expected, rtol=1e-12)


def test_drift_function_returns_portfolio_return():
    weights = np.array([0.5, 0.5])
    returns = np.array([0.10, -0.02])
    drifted, gross = bt.drift(weights, returns)
    assert gross == pytest.approx(0.5 * 0.10 + 0.5 * -0.02)
    assert drifted.sum() == pytest.approx(1.0)
    assert drifted[0] > 0.5   # the winner's share grows


def test_empty_portfolio_earns_nothing():
    drifted, gross = bt.drift(np.zeros(3), np.array([0.05, 0.05, 0.05]))
    assert gross == 0.0
    np.testing.assert_allclose(drifted, 0.0)


# ---------------------------------------------------------------------------
# Cost independence and the wealth identity
# ---------------------------------------------------------------------------

def test_gross_returns_identical_across_cost_scenarios(daily_returns, simple_targets):
    path = bt.simulate_path(daily_returns, simple_targets)
    baseline = bt.apply_costs(path["events"], 0.0)["gross_return"]
    for bps in (5.0, 10.0, 20.0, 100.0):
        other = bt.apply_costs(path["events"], bps)["gross_return"]
        pd.testing.assert_series_equal(baseline, other)


def test_zero_cost_net_equals_gross_exactly(daily_returns, simple_targets):
    path = bt.simulate_path(daily_returns, simple_targets)
    costed = bt.apply_costs(path["events"], 0.0)
    np.testing.assert_allclose(
        costed["net_return"].to_numpy(), costed["gross_return"].to_numpy(), atol=1e-18
    )


def test_higher_costs_do_not_change_holdings(daily_returns, simple_targets):
    """Holdings are computed once; costs are a wealth multiplier only."""
    path = bt.simulate_path(daily_returns, simple_targets)
    weights_a = path["posttrade_weights"].copy()
    # Applying any cost level must not touch the stored path.
    for bps in (0.0, 10.0, 50.0):
        bt.apply_costs(path["events"], bps)
    pd.testing.assert_frame_equal(weights_a, path["posttrade_weights"])


def test_wealth_identity_is_multiplicative(daily_returns, simple_targets):
    path = bt.simulate_path(daily_returns, simple_targets)
    costed = bt.apply_costs(path["events"], 10.0)
    assert costed["wealth_identity_error"].max() < 1e-15
    # Explicit check against the additive shortcut on the entry day,
    # where the difference is largest.
    first = costed.index[0]
    additive = costed.loc[first, "gross_return"] - costed.loc[first, "cost_fraction"]
    multiplicative = costed.loc[first, "net_return"]
    assert multiplicative == pytest.approx(
        (1 + costed.loc[first, "gross_return"])
        * (1 - costed.loc[first, "cost_fraction"]) - 1
    )
    # With zero gross return on entry day the two coincide; on a day with
    # both a return and a trade they must not.
    rebalance_days = costed.index[path["events"]["full_trade_sum"] > 0]
    later = [d for d in rebalance_days if abs(costed.loc[d, "gross_return"]) > 1e-6]
    if later:
        d = later[0]
        add = costed.loc[d, "gross_return"] - costed.loc[d, "cost_fraction"]
        assert costed.loc[d, "net_return"] != pytest.approx(add, abs=1e-18)


# ---------------------------------------------------------------------------
# Weight validity and causality
# ---------------------------------------------------------------------------

def test_weights_finite_nonnegative_and_sum_to_one(daily_returns, simple_targets):
    path = bt.simulate_path(daily_returns, simple_targets)
    for name in ("pretrade_weights", "posttrade_weights"):
        frame = path[name]
        assert np.all(np.isfinite(frame.to_numpy()))
        assert frame.to_numpy().min() >= -1e-15
        sums = frame.sum(axis=1)
        # The entry row is all cash by construction; every later row is
        # fully invested.
        np.testing.assert_allclose(sums.iloc[1:].to_numpy(), 1.0, atol=1e-12)


def test_future_returns_cannot_change_earlier_accounting(daily_returns, simple_targets):
    cutoff = daily_returns.index[20]
    perturbed = daily_returns.copy()
    rng = np.random.default_rng(7)
    future = perturbed.index > cutoff
    perturbed.loc[future] += rng.uniform(0.05, 0.20, (int(future.sum()), 3))

    base = bt.simulate_path(daily_returns, simple_targets)
    alt = bt.simulate_path(perturbed, simple_targets)
    for name in ("pretrade_weights", "posttrade_weights", "trades", "events"):
        pd.testing.assert_frame_equal(
            base[name].loc[:cutoff], alt[name].loc[:cutoff]
        )


def test_fixed_strategies_still_rebalance_on_schedule(daily_returns):
    """A constant target must be re-established at each execution date
    after drift, producing a non-zero trade."""
    dates = daily_returns.index
    fixed = np.array([0.6, 0.4, 0.0])
    targets = {dates[0]: fixed.copy(), dates[10]: fixed.copy()}
    path = bt.simulate_path(daily_returns, targets)
    events = path["events"]
    assert events.loc[dates[10], "trade_executed"]
    assert events.loc[dates[10], "full_trade_sum"] > 0   # drift was undone
    np.testing.assert_allclose(
        path["posttrade_weights"].loc[dates[10]].to_numpy(), fixed, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

def test_phase9_computes_no_performance_statistics(daily_returns, simple_targets):
    """Wealth is an accounting quantity; no Sharpe, drawdown, ranking or
    selection may appear."""
    path = bt.simulate_path(daily_returns, simple_targets)
    costed = bt.apply_costs(path["events"], 10.0)
    forbidden = {"sharpe", "sortino", "drawdown", "calmar", "rank",
                 "best", "winner", "selected", "annualized"}
    for frame in (*path.values(), costed):
        for column in frame.columns:
            assert not any(token in column.lower() for token in forbidden), column
