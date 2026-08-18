"""Tests for the Phase 4 EDA computations and integrity gates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import eda
from src import preprocessing as prep


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def test_summary_statistics_formulas(prices):
    stats = eda.summary_statistics(prices)
    asset = prices.columns[0]
    returns = prep.log_returns(prices)[asset]

    expected_vol = returns.std(ddof=1) * np.sqrt(252) * 100
    assert stats.loc[asset, "ann_vol_pct"] == pytest.approx(expected_vol, rel=1e-10)

    years = len(returns) / 252
    expected_cagr = ((prices[asset].iloc[-1] / prices[asset].iloc[0]) ** (1 / years) - 1) * 100
    assert stats.loc[asset, "ann_return_cagr_pct"] == pytest.approx(expected_cagr, rel=1e-10)

    expected_dd = prep.max_drawdown(prices[asset]) * 100
    assert stats.loc[asset, "max_drawdown_pct"] == pytest.approx(expected_dd, rel=1e-10)

    assert (stats["q01_daily_pct"] < stats["q05_daily_pct"]).all()
    assert (stats["max_drawdown_pct"] <= 0).all()


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def test_max_identical_run_detects_stale_stretch():
    s = pd.Series([1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    assert eda._max_identical_run(s) == 4


def test_staleness_audit_counts_zero_returns(prices):
    p = prices.copy()
    p.iloc[10, 0] = p.iloc[9, 0]  # force one unchanged day in asset 1
    audit = eda.staleness_audit(p, pd.DataFrame({"M": p.iloc[:, 0]}, index=p.index))
    assert audit.loc[p.columns[0], "n_zero_return_days"] >= 1
    assert audit.loc[p.columns[0], "max_identical_run_days"] >= 2


def test_calendar_audit_clean_business_days(trading_days):
    audit = eda.calendar_audit(trading_days)
    assert audit.loc["n_duplicate_dates", "value"] == 0
    assert audit.loc["n_weekend_rows", "value"] == 0
    # bdate_range has only weekend gaps (3 calendar days max)
    assert audit.loc["max_calendar_gap_days", "value"] <= 4


def test_calendar_audit_flags_long_gap(trading_days):
    with_hole = trading_days.delete(slice(100, 110))  # ~2-week hole
    audit = eda.calendar_audit(with_hole)
    assert audit.loc["n_gaps_over_4d", "value"] >= 1
    assert audit.loc["max_calendar_gap_days", "value"] > 4


def test_missingness_audit_shape(price_frames, macro_frames, prices, trading_days):
    macro = prep.build_macro_panel(macro_frames, prices.index, ffill_limit=5)
    audit = eda.missingness_audit(price_frames, macro_frames, prices, macro)
    assert set(audit.index) == set(price_frames) | set(macro_frames)
    assert (audit.loc[list(price_frames), "n_missing_aligned"] == 0).all()


# ---------------------------------------------------------------------------
# Correlation diagnostics
# ---------------------------------------------------------------------------

def test_conditioning_diagnostics(prices):
    returns = prep.log_returns(prices)
    corr, diag = eda.correlation_and_conditioning(returns)
    n = len(returns.columns)
    assert corr.shape == (n, n)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    eigvals = [diag.loc[f"eigenvalue_{i + 1}", "value"] for i in range(n)]
    assert eigvals == sorted(eigvals, reverse=True)
    assert diag.loc["condition_number", "value"] == pytest.approx(
        eigvals[0] / eigvals[-1], rel=1e-10
    )
    # Eigenvalues of a correlation matrix sum to its dimension.
    assert sum(eigvals) == pytest.approx(n, rel=1e-9)


# ---------------------------------------------------------------------------
# Subperiods
# ---------------------------------------------------------------------------

def test_subperiod_summaries_respect_bounds(prices, trading_days):
    macro = pd.DataFrame({"VIXCLS": 20.0}, index=prices.index)
    # Rename columns so SPY/IEF exist for the pairwise correlation.
    p = prices.copy()
    p.columns = ["SPY", "IEF", "GLD"]
    start, end = str(trading_days[50].date()), str(trading_days[250].date())
    out = eda.subperiod_summaries(p, macro, {"window": (start, end)})
    assert (out["start"] >= start).all()
    assert (out["end"] <= end).all()
    assert set(out["asset"]) == {"SPY", "IEF", "GLD"}


def test_subperiod_skips_tiny_windows(prices):
    p = prices.copy()
    p.columns = ["SPY", "IEF", "GLD"]
    macro = pd.DataFrame({"VIXCLS": 20.0}, index=p.index)
    start = str(p.index[0].date())
    end = str(p.index[10].date())  # < 42 obs
    out = eda.subperiod_summaries(p, macro, {"tiny": (start, end)})
    assert out.empty


# ---------------------------------------------------------------------------
# Snapshot integrity gate
# ---------------------------------------------------------------------------

def test_verify_snapshot_passes_and_fails(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": {"data.csv": {"sha256": eda.sha256_of(data_file)}}}),
        encoding="utf-8",
    )
    eda.verify_snapshot(manifest, tmp_path)  # must not raise

    data_file.write_text("a,b\n1,999\n", encoding="utf-8")  # tamper
    with pytest.raises(RuntimeError, match="frozen snapshot"):
        eda.verify_snapshot(manifest, tmp_path)
