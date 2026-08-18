"""Tests for panel construction, return math, rolling stats, and the
look-ahead guard. The causality tests are the most important tests in
the project: every backward-looking feature must pass them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as prep


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def test_price_panel_shape_and_columns(prices, price_frames):
    assert list(prices.columns) == list(price_frames)
    assert prices.notna().all().all()
    assert prices.index.is_monotonic_increasing


def test_price_panel_inner_join_drops_partial_dates(price_frames):
    frames = {t: df.copy() for t, df in price_frames.items()}
    # Knock out one ticker's price on one date: that date must disappear.
    victim_date = frames["BBB"].index[100]
    frames["BBB"].loc[victim_date, "Adj Close"] = np.nan
    panel = prep.build_price_panel(frames)
    assert victim_date not in panel.index
    assert len(panel) == len(frames["AAA"]) - 1


def test_macro_panel_aligns_and_fills_gaps(macro_frames, trading_days):
    macro = prep.build_macro_panel(macro_frames, trading_days, ffill_limit=5)
    assert list(macro.columns) == ["MACRO1", "MACRO2"]
    pd.testing.assert_index_equal(macro.index, trading_days)
    # The two single-day holes in MACRO2 must be filled with the prior value.
    hole = trading_days[50]
    prior = trading_days[49]
    assert macro.loc[hole, "MACRO2"] == macro.loc[prior, "MACRO2"]
    assert macro.notna().all().all()


def test_macro_panel_weekend_observation_carries_forward(trading_days):
    # A series observed only on Saturdays must still populate trading days.
    saturdays = pd.date_range(trading_days.min(), trading_days.max(), freq="W-SAT")
    weekly = pd.DataFrame({"WKLY": np.arange(len(saturdays), dtype=float)}, index=saturdays)
    macro = prep.build_macro_panel({"WKLY": weekly}, trading_days, ffill_limit=5)
    after_first_sat = macro.loc[macro.index > saturdays[0], "WKLY"]
    assert after_first_sat.notna().all()


def test_macro_panel_long_gap_raises(macro_frames, trading_days):
    frames = {k: v.copy() for k, v in macro_frames.items()}
    frames["MACRO2"].iloc[100:120, 0] = np.nan  # 20-day hole > ffill limit
    with pytest.raises(ValueError, match="forward-fill"):
        prep.build_macro_panel(frames, trading_days, ffill_limit=5)


def test_macro_panel_never_backfills(trading_days):
    # A series that starts late must have NaN before its first observation.
    late_start = pd.DataFrame(
        {"LATE": np.ones(len(trading_days) - 100)}, index=trading_days[100:]
    )
    macro = prep.build_macro_panel({"LATE": late_start}, trading_days, ffill_limit=5)
    assert macro["LATE"].iloc[:100].isna().all()
    assert macro["LATE"].iloc[100:].notna().all()


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def test_simple_returns_match_manual_calculation(prices):
    returns = prep.simple_returns(prices)
    manual = prices.iloc[1]["AAA"] / prices.iloc[0]["AAA"] - 1.0
    assert returns.iloc[0]["AAA"] == pytest.approx(manual, rel=1e-12)
    assert len(returns) == len(prices) - 1


def test_log_and_simple_returns_are_consistent(prices):
    simple = prep.simple_returns(prices)
    log = prep.log_returns(prices)
    np.testing.assert_allclose(np.log1p(simple), log, rtol=1e-10)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def test_rolling_volatility_matches_manual_window(prices):
    returns = prep.log_returns(prices)
    vol = prep.rolling_volatility(returns["AAA"], window=21, annualize=False)
    manual = returns["AAA"].iloc[100 - 20 : 101].std(ddof=1)
    assert vol.iloc[100] == pytest.approx(manual, rel=1e-12)
    assert vol.iloc[: 20].isna().all()


def test_rolling_volatility_annualization(prices):
    returns = prep.log_returns(prices)
    daily = prep.rolling_volatility(returns, window=21, annualize=False)
    annual = prep.rolling_volatility(returns, window=21, annualize=True)
    np.testing.assert_allclose(annual, daily * np.sqrt(252), rtol=1e-12)


def test_rolling_covariance_matches_numpy(prices):
    returns = prep.log_returns(prices)
    window = 63
    mats = prep.rolling_covariance_matrices(returns, window=window)
    last_date = returns.index[-1]
    expected = np.cov(returns.iloc[-window:].to_numpy(), rowvar=False, ddof=1)
    np.testing.assert_allclose(mats[last_date].to_numpy(), expected, rtol=1e-12)
    # First window-1 dates have no matrix.
    assert returns.index[window - 2] not in mats
    assert returns.index[window - 1] in mats


def test_drawdown_on_crafted_series():
    prices = pd.Series([100.0, 110.0, 99.0, 104.5, 121.0, 90.75])
    dd = prep.drawdown_series(prices)
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == 0.0                      # new high
    assert dd.iloc[2] == pytest.approx(99.0 / 110.0 - 1)
    assert dd.iloc[4] == 0.0                      # recovered to new high
    assert prep.max_drawdown(prices) == pytest.approx(90.75 / 121.0 - 1)


# ---------------------------------------------------------------------------
# Macro features
# ---------------------------------------------------------------------------

def test_yield_curve_slope():
    macro = pd.DataFrame({"DGS10": [4.0, 4.1], "DGS2": [4.5, 4.0]})
    slope = prep.yield_curve_slope(macro)
    np.testing.assert_allclose(slope, [-0.5, 0.1])


def test_vix_log_changes():
    macro = pd.DataFrame({"VIXCLS": [20.0, 22.0]})
    change = prep.vix_changes(macro, log=True)
    assert change.iloc[1] == pytest.approx(np.log(22 / 20))


# ---------------------------------------------------------------------------
# Look-ahead guard
# ---------------------------------------------------------------------------

def test_rolling_features_are_causal(prices):
    """Truncating the future must not change past feature values."""
    returns = prep.log_returns(prices)
    prep.assert_causal(lambda df: prep.rolling_volatility(df, window=21), returns)
    prep.assert_causal(prep.simple_returns, prices)
    prep.assert_causal(prep.drawdown_series, prices)


def test_assert_causal_catches_lookahead(prices):
    """A deliberately look-ahead transform (centered window) must fail."""
    returns = prep.log_returns(prices)

    def centered_vol(df):
        return df.rolling(window=21, center=True).std()

    with pytest.raises(AssertionError, match="look-ahead"):
        prep.assert_causal(centered_vol, returns)


def test_assert_causal_catches_full_sample_zscore(prices):
    """Full-sample standardization leaks future means/stds and must fail."""
    returns = prep.log_returns(prices)

    def full_sample_zscore(df):
        return (df - df.mean()) / df.std()

    with pytest.raises(AssertionError, match="look-ahead"):
        prep.assert_causal(full_sample_zscore, returns)
