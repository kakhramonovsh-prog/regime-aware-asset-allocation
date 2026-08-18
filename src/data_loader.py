"""Download and load raw market and macro data.

Responsibilities of this module are strictly I/O:

* download adjusted ETF prices from Yahoo Finance (via ``yfinance``),
* download macro series from FRED (via ``pandas-datareader``),
* persist raw observations to ``data/raw`` exactly as received,
* record download metadata (source, timestamp, date range, row counts),
* reload raw files into DataFrames for downstream processing.

No transformation beyond parsing happens here. Alignment, filling, and
feature construction live in :mod:`src.preprocessing` so that every
modelling choice is visible in one place and covered by tests.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the project YAML configuration.

    Parameters
    ----------
    path:
        Location of ``config.yaml``. Defaults to ``config/config.yaml``
        at the repository root.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Fail fast if required configuration keys are missing."""
    required_data_keys = {"start_date", "tickers", "fred_series", "raw_dir", "processed_dir"}
    if "data" not in config:
        raise KeyError("config missing top-level 'data' section")
    missing = required_data_keys - set(config["data"])
    if missing:
        raise KeyError(f"config['data'] missing keys: {sorted(missing)}")
    if not config["data"]["tickers"]:
        raise ValueError("config['data']['tickers'] is empty")


# ---------------------------------------------------------------------------
# Validation of received data
# ---------------------------------------------------------------------------

def validate_price_frame(df: pd.DataFrame, ticker: str) -> None:
    """Sanity-check a raw single-ticker OHLCV frame from the vendor.

    Raises ``ValueError`` on: empty frame, missing required columns,
    non-monotonic or duplicated dates, or non-positive adjusted closes.
    These checks catch silent vendor failures (renamed columns, empty
    responses) before bad data reaches the pipeline.
    """
    if df.empty:
        raise ValueError(f"{ticker}: received empty price frame")
    missing_cols = [c for c in ("Close", "Adj Close") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker}: missing columns {missing_cols}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: date index is not sorted ascending")
    if df.index.has_duplicates:
        raise ValueError(f"{ticker}: duplicate dates in index")
    adj = df["Adj Close"].dropna()
    if adj.empty:
        raise ValueError(f"{ticker}: 'Adj Close' is entirely missing")
    if (adj <= 0).any():
        bad = adj[adj <= 0].index[0].date()
        raise ValueError(f"{ticker}: non-positive adjusted close on {bad}")


def validate_macro_frame(df: pd.DataFrame, series_id: str) -> None:
    """Sanity-check a raw FRED series."""
    if df.empty:
        raise ValueError(f"{series_id}: received empty frame from FRED")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{series_id}: date index is not sorted ascending")
    if df.index.has_duplicates:
        raise ValueError(f"{series_id}: duplicate dates in index")
    if df[series_id].dropna().empty:
        raise ValueError(f"{series_id}: series is entirely missing")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV history for each ticker from Yahoo Finance.

    Prices are requested with ``auto_adjust=False`` so both the raw
    ``Close`` and the dividend/split-adjusted ``Adj Close`` are stored.
    Returns are later computed from ``Adj Close``, which proxies total
    return (price appreciation plus reinvested distributions).

    Returns
    -------
    dict
        Mapping ticker -> raw OHLCV DataFrame indexed by date.
    """
    import yfinance as yf  # imported here so tests never need network

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        df.index.name = "Date"
        validate_price_frame(df, ticker)
        frames[ticker] = df[[c for c in PRICE_FIELDS if c in df.columns]]
    return frames


def download_fred(
    series_ids: list[str],
    start: str,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download macro series from FRED (no API key required).

    Returns
    -------
    dict
        Mapping series_id -> single-column DataFrame indexed by date.
    """
    import pandas_datareader.data as web  # imported here to keep tests offline

    frames: dict[str, pd.DataFrame] = {}
    for series_id in series_ids:
        df = web.DataReader(series_id, "fred", start=start, end=end)
        df.index.name = "Date"
        validate_macro_frame(df, series_id)
        frames[series_id] = df
    return frames


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw(
    price_frames: dict[str, pd.DataFrame],
    macro_frames: dict[str, pd.DataFrame],
    raw_dir: str | Path,
) -> Path:
    """Write raw frames to CSV and a metadata JSON alongside them.

    Files written::

        data/raw/prices_<TICKER>.csv
        data/raw/fred_<SERIES>.csv
        data/raw/download_metadata.json

    The metadata file records the download timestamp (UTC), source,
    date coverage, and row count for every file, which is what makes a
    later sample-period statement in the paper auditable.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "price_source": "Yahoo Finance via yfinance",
        "macro_source": "FRED via pandas-datareader",
        "files": {},
    }

    for ticker, df in price_frames.items():
        path = raw_dir / f"prices_{ticker}.csv"
        df.to_csv(path)
        metadata["files"][path.name] = _describe(df)

    for series_id, df in macro_frames.items():
        path = raw_dir / f"fred_{series_id}.csv"
        df.to_csv(path)
        metadata["files"][path.name] = _describe(df)

    meta_path = raw_dir / "download_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return meta_path


def _describe(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize a frame for the metadata record."""
    return {
        "rows": int(len(df)),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
    }


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

def load_raw_prices(raw_dir: str | Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Reload the per-ticker raw OHLCV CSVs written by :func:`save_raw`."""
    raw_dir = Path(raw_dir)
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = raw_dir / f"prices_{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python scripts/download_data.py' first."
            )
        frames[ticker] = pd.read_csv(path, index_col="Date", parse_dates=True)
    return frames


def load_raw_macro(raw_dir: str | Path, series_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Reload the raw FRED CSVs written by :func:`save_raw`."""
    raw_dir = Path(raw_dir)
    frames: dict[str, pd.DataFrame] = {}
    for series_id in series_ids:
        path = raw_dir / f"fred_{series_id}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python scripts/download_data.py' first."
            )
        frames[series_id] = pd.read_csv(path, index_col="Date", parse_dates=True)
    return frames
