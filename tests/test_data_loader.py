"""Tests for config loading, vendor-data validation, and raw persistence."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import data_loader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_project_config_loads_and_validates():
    config = data_loader.load_config()
    assert set(config["data"]["tickers"]) == {"SPY", "QQQ", "IWM", "IEF", "GLD"}
    assert "VIXCLS" in config["data"]["fred_series"]
    assert config["project"]["random_seed"] == 42


def test_config_missing_section_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project:\n  name: x\n", encoding="utf-8")
    with pytest.raises(KeyError, match="data"):
        data_loader.load_config(bad)


# ---------------------------------------------------------------------------
# Price-frame validation
# ---------------------------------------------------------------------------

def test_valid_price_frame_passes(price_frames):
    for ticker, df in price_frames.items():
        data_loader.validate_price_frame(df, ticker)


def test_empty_price_frame_raises():
    with pytest.raises(ValueError, match="empty"):
        data_loader.validate_price_frame(pd.DataFrame(), "XXX")


def test_missing_adj_close_raises(price_frames):
    df = price_frames["AAA"].drop(columns=["Adj Close"])
    with pytest.raises(ValueError, match="Adj Close"):
        data_loader.validate_price_frame(df, "AAA")


def test_unsorted_index_raises(price_frames):
    df = price_frames["AAA"].iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        data_loader.validate_price_frame(df, "AAA")


def test_duplicate_dates_raise(price_frames):
    df = pd.concat([price_frames["AAA"], price_frames["AAA"].iloc[[0]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        data_loader.validate_price_frame(df, "AAA")


def test_negative_price_raises(price_frames):
    df = price_frames["AAA"].copy()
    df.iloc[10, df.columns.get_loc("Adj Close")] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        data_loader.validate_price_frame(df, "AAA")


# ---------------------------------------------------------------------------
# Raw persistence round-trip
# ---------------------------------------------------------------------------

def test_save_and_reload_round_trip(tmp_path, price_frames, macro_frames):
    meta_path = data_loader.save_raw(price_frames, macro_frames, tmp_path)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(metadata["files"]) == len(price_frames) + len(macro_frames)
    assert "downloaded_at_utc" in metadata

    reloaded = data_loader.load_raw_prices(tmp_path, list(price_frames))
    for ticker, original in price_frames.items():
        # check_freq=False: a CSV round-trip cannot preserve the index's
        # freq attribute; dates and values are what matter.
        pd.testing.assert_index_equal(
            reloaded[ticker].index, original.index, exact="equiv"
        )
        pd.testing.assert_series_equal(
            reloaded[ticker]["Adj Close"],
            original["Adj Close"],
            check_exact=False,
            check_freq=False,
            rtol=1e-10,
        )

    reloaded_macro = data_loader.load_raw_macro(tmp_path, list(macro_frames))
    assert set(reloaded_macro) == set(macro_frames)


def test_load_missing_raw_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        data_loader.load_raw_prices(tmp_path, ["SPY"])
