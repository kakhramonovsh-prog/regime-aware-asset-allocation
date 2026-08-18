"""Performance metrics (Phase 10).

Every metric is computed from the **stored Phase 9 return series**; this
module never re-runs the backtest, re-derives weights, or touches price
data beyond the risk-free series. Definitions are frozen in
``config/analysis_plan.yaml`` under ``metrics``:

* CAGR ``(V_T/V_0)^(365.25/D) - 1`` over elapsed calendar days, with the
  initial entry cost included in the wealth path,
* annualized volatility ``std(r) * sqrt(252)``,
* risk-free ``r_f,t = (DFF_{t-1}/100) * (calendar days / 360)``, using
  the last DFF known at the start of the interval and zero on the
  entry row (which is cost-only and earns nothing),
* Sharpe and Sortino on daily **excess** returns, the latter with a
  zero minimum acceptable excess return,
* drawdown from the net wealth path, Calmar as ``CAGR / |max DD|``,
* historical 95%/99% VaR and Expected Shortfall, reported with
  **losses as positive numbers**.

Phase 10 is descriptive. Differences between strategies are reported as
point estimates; nothing here may be described as statistically
significant, which is Phase 11's question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25
ACT_360 = 360.0


# ---------------------------------------------------------------------------
# Risk-free series
# ---------------------------------------------------------------------------

def risk_free_daily(
    dates: pd.DatetimeIndex, dff: pd.Series, entry_date: pd.Timestamp | None = None
) -> pd.Series:
    """Daily risk-free return over each interval, ACT/360.

    Uses the last DFF observation known at the **start** of the interval
    (the previous trading day's rate), scaled by the calendar days
    elapsed. The entry row is cost-only and earns nothing, so its
    risk-free return is zero.
    """
    dff = dff.reindex(dates.union(dff.index)).ffill().reindex(dates)
    previous_rate = dff.shift(1)
    calendar_days = pd.Series(dates, index=dates).diff().dt.days
    rf = (previous_rate / 100.0) * (calendar_days / ACT_360)
    rf.iloc[0] = 0.0
    if entry_date is not None and entry_date in rf.index:
        rf.loc[entry_date] = 0.0
    return rf.fillna(0.0)


# ---------------------------------------------------------------------------
# Wealth and drawdown
# ---------------------------------------------------------------------------

def wealth_path(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Cumulative wealth from a return series."""
    return initial * (1.0 + returns).cumprod()


def drawdown_path(wealth: pd.Series) -> pd.Series:
    """Drawdown relative to the running maximum of the wealth path."""
    return wealth / wealth.cummax() - 1.0


def max_drawdown(wealth: pd.Series) -> float:
    """Most negative drawdown (returned as a negative number)."""
    return float(drawdown_path(wealth).min())


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def cagr(wealth: pd.Series, initial: float = 1.0) -> float:
    """Compound annual growth rate over elapsed calendar days.

    ``V_0`` is the wealth before the first return (the initial capital),
    so the entry cost embedded in the first return is included.
    """
    elapsed = (wealth.index[-1] - wealth.index[0]).days
    if elapsed <= 0:
        return np.nan
    return float((wealth.iloc[-1] / initial) ** (DAYS_PER_YEAR / elapsed) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series) -> float:
    """Annualized Sharpe ratio on daily excess returns."""
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    sigma = excess.std(ddof=1)
    if sigma == 0:
        return np.nan
    return float(excess.mean() / sigma * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: pd.Series) -> float:
    """Annualized Sortino ratio, minimum acceptable excess return zero.

    The denominator is the root-mean-square of the *negative* excess
    returns, averaged over all observations (not only the negative
    ones), which is the standard downside-deviation definition.
    """
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    downside = np.minimum(excess, 0.0)
    denominator = np.sqrt((downside**2).mean())
    if denominator == 0:
        return np.nan
    return float(excess.mean() / denominator * np.sqrt(TRADING_DAYS))


def calmar_ratio(cagr_value: float, maximum_drawdown: float) -> float:
    if maximum_drawdown == 0:
        return np.nan
    return float(cagr_value / abs(maximum_drawdown))


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Historical VaR at ``level``. **Losses are positive numbers.**"""
    return float(-np.quantile(returns.to_numpy(), 1.0 - level))


def expected_shortfall(returns: pd.Series, level: float = 0.95) -> float:
    """Historical Expected Shortfall. **Losses are positive numbers.**"""
    threshold = np.quantile(returns.to_numpy(), 1.0 - level)
    tail = returns[returns <= threshold]
    if tail.empty:
        return np.nan
    return float(-tail.mean())


def rolling_sharpe(
    returns: pd.Series, risk_free: pd.Series, window: int = TRADING_DAYS
) -> pd.Series:
    """Backward-looking rolling annualized Sharpe ratio.

    Uses pandas' trailing window, so the value at ``t`` depends only on
    observations up to and including ``t``; no future data enters.
    """
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    mean = excess.rolling(window, min_periods=window).mean()
    sigma = excess.rolling(window, min_periods=window).std(ddof=1)
    return (mean / sigma) * np.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def performance_summary(
    returns: pd.Series,
    risk_free: pd.Series,
    turnover_half: float | None = None,
    cost_expenditure: float | None = None,
    label: str = "",
) -> dict:
    """Full metric set for one return series. Full precision, no rounding."""
    wealth = wealth_path(returns)
    cagr_value = cagr(wealth)
    mdd = max_drawdown(wealth)
    years = (returns.index[-1] - returns.index[0]).days / DAYS_PER_YEAR

    summary = {
        "label": label,
        "n_days": int(len(returns)),
        "start": returns.index[0],
        "end": returns.index[-1],
        "years": years,
        "terminal_wealth": float(wealth.iloc[-1]),
        "cagr": cagr_value,
        "ann_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free),
        "sortino": sortino_ratio(returns, risk_free),
        "max_drawdown": mdd,
        "calmar": calmar_ratio(cagr_value, mdd),
        "var_95": historical_var(returns, 0.95),
        "var_99": historical_var(returns, 0.99),
        "es_95": expected_shortfall(returns, 0.95),
        "es_99": expected_shortfall(returns, 0.99),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "positive_days_pct": float((returns > 0).mean() * 100),
    }
    if turnover_half is not None:
        summary["ann_half_turnover"] = turnover_half
        summary["ann_full_traded_notional"] = 2.0 * turnover_half
    if cost_expenditure is not None:
        summary["ann_cost_expenditure_bps"] = cost_expenditure
    return summary
