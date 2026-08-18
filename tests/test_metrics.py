"""Tests for Phase 10 performance metrics.

Covers the frozen definitions and every required safeguard: CAGR
reconciling with the wealth path, drawdown matching a manual
calculation, Sharpe/Sortino on excess returns, the risk-free
convention, VaR/ES sign convention, and the absence of look-ahead in
rolling Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics as mt


@pytest.fixture
def returns() -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=500, name="date")
    rng = np.random.default_rng(4)
    return pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)


@pytest.fixture
def zero_rf(returns) -> pd.Series:
    return pd.Series(0.0, index=returns.index)


# ---------------------------------------------------------------------------
# Wealth, CAGR, drawdown
# ---------------------------------------------------------------------------

def test_wealth_path_compounds_multiplicatively():
    r = pd.Series([0.10, -0.05, 0.02], index=pd.date_range("2020-01-01", periods=3))
    w = mt.wealth_path(r)
    assert w.iloc[0] == pytest.approx(1.10)
    assert w.iloc[1] == pytest.approx(1.10 * 0.95)
    assert w.iloc[2] == pytest.approx(1.10 * 0.95 * 1.02)


def test_cagr_reconciles_with_first_and_last_wealth(returns):
    w = mt.wealth_path(returns)
    value = mt.cagr(w)
    elapsed = (w.index[-1] - w.index[0]).days
    expected = (w.iloc[-1] / 1.0) ** (365.25 / elapsed) - 1
    assert value == pytest.approx(expected, rel=1e-12)
    # Round trip: compounding the CAGR over the period recovers terminal wealth.
    assert (1 + value) ** (elapsed / 365.25) == pytest.approx(w.iloc[-1], rel=1e-9)


def test_cagr_includes_entry_cost():
    """A cost-only first day must lower the CAGR versus a costless path."""
    idx = pd.date_range("2020-01-01", periods=366)
    costless = pd.Series(0.0, index=idx)
    costless.iloc[1:] = 0.0004
    with_entry = costless.copy()
    with_entry.iloc[0] = -0.0010          # 10 bps entry cost
    assert mt.cagr(mt.wealth_path(with_entry)) < mt.cagr(mt.wealth_path(costless))


def test_max_drawdown_matches_manual_calculation():
    prices = pd.Series(
        [1.0, 1.2, 0.9, 1.1, 0.6, 0.8],
        index=pd.date_range("2020-01-01", periods=6),
    )
    r = prices.pct_change().fillna(prices.iloc[0] - 1)
    w = mt.wealth_path(r.iloc[1:])
    manual = (0.6 / 1.2) - 1             # trough 0.6 against peak 1.2
    assert mt.max_drawdown(prices) == pytest.approx(manual)
    dd = mt.drawdown_path(prices)
    assert dd.iloc[1] == 0.0             # new high
    assert dd.min() == pytest.approx(manual)


def test_drawdown_is_zero_for_monotone_growth():
    w = pd.Series([1.0, 1.1, 1.2, 1.3], index=pd.date_range("2020-01-01", periods=4))
    np.testing.assert_allclose(mt.drawdown_path(w).to_numpy(), 0.0)
    assert mt.max_drawdown(w) == 0.0


# ---------------------------------------------------------------------------
# Risk-free convention
# ---------------------------------------------------------------------------

def test_risk_free_uses_previous_rate_and_act360():
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    dff = pd.Series([1.80, 1.80, 1.80], index=dates)
    rf = mt.risk_free_daily(dates, dff)
    assert rf.iloc[0] == 0.0                       # entry row earns nothing
    # 2020-01-03 is one calendar day after 01-02.
    assert rf.iloc[1] == pytest.approx((1.80 / 100) * (1 / 360))
    # 2020-01-06 is three calendar days after 01-03 (weekend).
    assert rf.iloc[2] == pytest.approx((1.80 / 100) * (3 / 360))


def test_risk_free_is_zero_when_rate_is_zero(returns):
    dff = pd.Series(0.0, index=returns.index)
    rf = mt.risk_free_daily(returns.index, dff)
    np.testing.assert_allclose(rf.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------

def test_sharpe_matches_manual_formula(returns, zero_rf):
    value = mt.sharpe_ratio(returns, zero_rf)
    manual = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert value == pytest.approx(manual, rel=1e-12)


def test_sharpe_uses_excess_returns(returns):
    flat_rf = pd.Series(0.0002, index=returns.index)
    assert mt.sharpe_ratio(returns, flat_rf) < mt.sharpe_ratio(
        returns, pd.Series(0.0, index=returns.index)
    )


def test_sortino_penalizes_only_downside(returns, zero_rf):
    value = mt.sortino_ratio(returns, zero_rf)
    downside = np.minimum(returns, 0.0)
    manual = returns.mean() / np.sqrt((downside**2).mean()) * np.sqrt(252)
    assert value == pytest.approx(manual, rel=1e-12)
    # With no negative returns the downside deviation is zero -> undefined.
    positive = pd.Series(0.01, index=returns.index)
    assert np.isnan(mt.sortino_ratio(positive, pd.Series(0.0, index=returns.index)))


def test_sortino_exceeds_sharpe_for_right_skewed_series():
    idx = pd.date_range("2020-01-01", periods=200)
    r = pd.Series(np.r_[np.full(180, -0.001), np.full(20, 0.02)], index=idx)
    rf = pd.Series(0.0, index=idx)
    assert mt.sortino_ratio(r, rf) > mt.sharpe_ratio(r, rf)


def test_calmar_is_cagr_over_absolute_drawdown():
    assert mt.calmar_ratio(0.08, -0.20) == pytest.approx(0.40)
    assert np.isnan(mt.calmar_ratio(0.08, 0.0))


# ---------------------------------------------------------------------------
# VaR / ES sign convention
# ---------------------------------------------------------------------------

def test_var_and_es_report_losses_as_positive(returns):
    var95 = mt.historical_var(returns, 0.95)
    es95 = mt.expected_shortfall(returns, 0.95)
    assert var95 > 0, "losses must be reported as positive numbers"
    assert es95 > 0
    assert es95 >= var95, "ES is at least as large as VaR"
    assert mt.historical_var(returns, 0.99) >= var95


def test_var_matches_empirical_quantile(returns):
    assert mt.historical_var(returns, 0.95) == pytest.approx(
        -np.quantile(returns.to_numpy(), 0.05)
    )


# ---------------------------------------------------------------------------
# Rolling Sharpe causality
# ---------------------------------------------------------------------------

def test_rolling_sharpe_has_no_lookahead(returns, zero_rf):
    full = mt.rolling_sharpe(returns, zero_rf, window=60)
    cutoff = returns.index[300]
    truncated = mt.rolling_sharpe(returns.loc[:cutoff], zero_rf.loc[:cutoff], window=60)
    pd.testing.assert_series_equal(
        full.loc[:cutoff].dropna(), truncated.dropna(), check_freq=False
    )


def test_rolling_sharpe_requires_full_window(returns, zero_rf):
    rolling = mt.rolling_sharpe(returns, zero_rf, window=60)
    assert rolling.iloc[:59].isna().all()
    assert rolling.iloc[59:].notna().all()


# ---------------------------------------------------------------------------
# Summary assembly
# ---------------------------------------------------------------------------

def test_performance_summary_is_self_consistent(returns, zero_rf):
    summary = mt.performance_summary(returns, zero_rf, label="test")
    w = mt.wealth_path(returns)
    assert summary["terminal_wealth"] == pytest.approx(w.iloc[-1])
    assert summary["cagr"] == pytest.approx(mt.cagr(w))
    assert summary["max_drawdown"] == pytest.approx(mt.max_drawdown(w))
    assert summary["calmar"] == pytest.approx(
        summary["cagr"] / abs(summary["max_drawdown"])
    )
    assert summary["n_days"] == len(returns)
    assert summary["max_drawdown"] <= 0
    assert summary["var_95"] > 0


def test_summary_reports_turnover_in_both_conventions(returns, zero_rf):
    summary = mt.performance_summary(
        returns, zero_rf, turnover_half=0.169, cost_expenditure=3.39
    )
    assert summary["ann_half_turnover"] == pytest.approx(0.169)
    assert summary["ann_full_traded_notional"] == pytest.approx(0.338)
    assert summary["ann_cost_expenditure_bps"] == pytest.approx(3.39)


def test_zero_cost_metrics_equal_gross_metrics(returns, zero_rf):
    """Net returns at zero cost are the gross returns, so every metric
    must coincide."""
    gross = mt.performance_summary(returns, zero_rf, label="gross")
    net0 = mt.performance_summary(returns.copy(), zero_rf, label="net0")
    for key in ("cagr", "ann_volatility", "sharpe", "sortino",
                "max_drawdown", "calmar", "var_95", "es_99"):
        assert gross[key] == pytest.approx(net0[key], rel=1e-15)


def test_higher_costs_weakly_lower_terminal_wealth(returns):
    """Applying a larger uniform cost cannot raise terminal wealth."""
    rf = pd.Series(0.0, index=returns.index)
    previous = np.inf
    for bps in (0.0, 5.0, 10.0, 20.0):
        costed = (1 + returns) * (1 - bps / 1e4 * 0.1) - 1
        terminal = mt.performance_summary(costed, rf)["terminal_wealth"]
        assert terminal <= previous + 1e-12
        previous = terminal
