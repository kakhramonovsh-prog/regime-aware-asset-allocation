"""Shared fixtures: synthetic market data with a fixed seed.

Tests never touch the network. Synthetic geometric-Brownian prices are
enough to verify the mechanics (alignment, return math, causality);
correctness on real data is a separate concern handled by the
validation checks in ``src/data_loader.py`` at download time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TICKERS = ["AAA", "BBB", "CCC"]
N_DAYS = 400
SEED = 42


@pytest.fixture(scope="session")
def trading_days() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=N_DAYS, name="Date")


@pytest.fixture(scope="session")
def price_frames(trading_days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Per-ticker OHLCV frames mimicking the raw vendor format."""
    rng = np.random.default_rng(SEED)
    frames: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        log_ret = rng.normal(loc=0.0003, scale=0.012, size=N_DAYS)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        frames[ticker] = pd.DataFrame(
            {
                "Open": close * (1 + rng.normal(0, 0.002, N_DAYS)),
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Adj Close": close * 0.98,  # offset mimics dividend adjustment
                "Volume": rng.integers(1e5, 1e6, N_DAYS),
            },
            index=trading_days,
        )
    return frames


@pytest.fixture(scope="session")
def prices(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from src.preprocessing import build_price_panel

    return build_price_panel(price_frames)


@pytest.fixture(scope="session")
def macro_frames(trading_days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Synthetic macro series with realistic irregularities:

    * MACRO1 observed every calendar day (like DFF),
    * MACRO2 observed on trading days but with two missing single days
      (like a bond-market holiday in DGS10).
    """
    rng = np.random.default_rng(SEED + 1)
    calendar_days = pd.date_range(
        trading_days.min(), trading_days.max(), freq="D", name="Date"
    )
    macro1 = pd.DataFrame(
        {"MACRO1": 2.0 + rng.normal(0, 0.01, len(calendar_days)).cumsum()},
        index=calendar_days,
    )
    macro2_values = 15.0 + rng.normal(0, 0.3, N_DAYS).cumsum()
    macro2 = pd.DataFrame({"MACRO2": macro2_values}, index=trading_days)
    macro2.iloc[[50, 200], 0] = np.nan
    return {"MACRO1": macro1, "MACRO2": macro2}
