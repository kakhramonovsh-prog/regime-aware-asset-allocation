"""Pipeline-level causality: perturbing or deleting RAW data after date t
must not change any processed value or derived feature dated on or
before t.

This extends the transform-level ``assert_causal`` guard to the
raw-to-processed stage. The same requirement wraps the backtest engine
when Phase 9 is built (docs/research_design.md §3): signals, weights,
trades, costs, and portfolio returns join this test then.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import preprocessing as prep


def _build(price_frames, macro_frames):
    prices = prep.build_price_panel(price_frames)
    macro = prep.build_macro_panel(macro_frames, prices.index, ffill_limit=5)
    return prices, macro


def test_truncating_future_raw_data_preserves_past_panels(
    price_frames, macro_frames, trading_days
):
    prices_full, macro_full = _build(price_frames, macro_frames)
    cutoff = trading_days[250]

    truncated_prices = {t: df.loc[:cutoff] for t, df in price_frames.items()}
    truncated_macro = {k: df.loc[:cutoff] for k, df in macro_frames.items()}
    prices_trunc, macro_trunc = _build(truncated_prices, truncated_macro)

    pd.testing.assert_frame_equal(
        prices_full.loc[:cutoff], prices_trunc, check_freq=False
    )
    pd.testing.assert_frame_equal(
        macro_full.loc[:cutoff], macro_trunc, check_freq=False
    )


def test_perturbing_future_raw_data_preserves_past_features(
    price_frames, macro_frames, trading_days
):
    prices_full, macro_full = _build(price_frames, macro_frames)
    cutoff = trading_days[250]

    # Corrupt every raw observation after the cutoff with large noise.
    rng = np.random.default_rng(0)
    perturbed_prices = {}
    for ticker, df in price_frames.items():
        df = df.copy()
        future = df.index > cutoff
        df.loc[future, "Adj Close"] *= 1 + rng.uniform(0.5, 2.0, future.sum())
        perturbed_prices[ticker] = df
    perturbed_macro = {}
    for series_id, df in macro_frames.items():
        df = df.copy()
        future = df.index > cutoff
        df.loc[future, series_id] += rng.uniform(10, 100, future.sum())
        perturbed_macro[series_id] = df

    prices_pert, macro_pert = _build(perturbed_prices, perturbed_macro)

    # Processed panels unchanged before the cutoff.
    pd.testing.assert_frame_equal(
        prices_full.loc[:cutoff], prices_pert.loc[:cutoff], check_freq=False
    )
    pd.testing.assert_frame_equal(
        macro_full.loc[:cutoff], macro_pert.loc[:cutoff], check_freq=False
    )

    # Derived features unchanged before the cutoff.
    for derive in (
        lambda p: prep.log_returns(p),
        lambda p: prep.rolling_volatility(prep.log_returns(p), window=21),
        lambda p: prep.drawdown_series(p),
    ):
        full_feat = derive(prices_full)
        pert_feat = derive(prices_pert)
        pd.testing.assert_frame_equal(
            full_feat.loc[:cutoff], pert_feat.loc[:cutoff], check_freq=False
        )


def test_signal_lag_shifts_by_trading_days(trading_days):
    series = pd.Series(np.arange(len(trading_days), dtype=float), index=trading_days)
    lagged = prep.apply_signal_lag(series, lag_days=1)
    # Value dated t was observed at t-1.
    assert np.isnan(lagged.iloc[0])
    assert lagged.iloc[1] == series.iloc[0]
    assert lagged.iloc[-1] == series.iloc[-2]


def test_signal_lag_rejects_negative_lag(trading_days):
    import pytest

    series = pd.Series(1.0, index=trading_days)
    with pytest.raises(ValueError, match="non-negative"):
        prep.apply_signal_lag(series, lag_days=-1)
