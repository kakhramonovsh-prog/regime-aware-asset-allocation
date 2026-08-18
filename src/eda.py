"""Phase 4 exploratory data analysis and data-quality diagnostics.

Everything in this module is descriptive. It computes summary
statistics, data-quality audits, and correlation diagnostics from the
frozen processed panels; it fits no model, evaluates no strategy, and
produces nothing a trading rule could consume. Look-ahead is not a
concern for full-sample *description*, and no output here feeds a
signal.

Integrity gates:

* :func:`verify_snapshot` refuses to run the analysis if any data file
  on disk differs from the frozen SHA-256 manifest
  (``data/snapshots/manifest_2026-08-06.json``).
* :func:`write_eda_manifest` records the git commit, config hash, and
  the hash of every produced artifact, so each table and figure is
  traceable to the exact code and data that generated it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from src import preprocessing as prep

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Integrity gates
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(manifest_path: Path, project_root: Path) -> dict:
    """Fail hard unless every data file matches the frozen manifest.

    Returns the parsed manifest so callers can reference its hashes.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for rel_path, info in manifest["files"].items():
        path = project_root / rel_path
        if not path.exists():
            mismatches.append(f"{rel_path}: missing")
        elif sha256_of(path) != info["sha256"]:
            mismatches.append(f"{rel_path}: hash differs from frozen snapshot")
    if mismatches:
        raise RuntimeError(
            "Data on disk does not match the frozen snapshot "
            f"({manifest_path.name}); refusing to run EDA on unfrozen data.\n  "
            + "\n  ".join(mismatches)
        )
    return manifest


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_statistics(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset descriptive statistics of daily log returns.

    Annualized return is the geometric CAGR from first to last price;
    annualized volatility is daily log-return std scaled by sqrt(252).
    Tail statistics (1%/5% quantiles), Jarque-Bera normality test, and
    maximum drawdown are included. Full-sample, descriptive only.
    """
    returns = prep.log_returns(prices)
    n = len(returns)
    years = n / TRADING_DAYS
    rows = {}
    for asset in prices.columns:
        r = returns[asset]
        jb_stat, jb_p = sps.jarque_bera(r)
        rows[asset] = {
            "n_obs": n,
            "mean_daily_bps": r.mean() * 1e4,
            "median_daily_bps": r.median() * 1e4,
            "std_daily_pct": r.std(ddof=1) * 100,
            "skewness": sps.skew(r),
            "excess_kurtosis": sps.kurtosis(r, fisher=True),
            "min_daily_pct": r.min() * 100,
            "max_daily_pct": r.max() * 100,
            "q01_daily_pct": r.quantile(0.01) * 100,
            "q05_daily_pct": r.quantile(0.05) * 100,
            "pct_positive_days": (r > 0).mean() * 100,
            "ann_return_cagr_pct": ((prices[asset].iloc[-1] / prices[asset].iloc[0]) ** (1 / years) - 1) * 100,
            "ann_vol_pct": r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100,
            "max_drawdown_pct": prep.max_drawdown(prices[asset]) * 100,
            "jarque_bera_stat": jb_stat,
            "jarque_bera_pvalue": jb_p,
        }
    return pd.DataFrame(rows).T.rename_axis("asset")


def macro_summary(macro: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for macro level series."""
    out = {}
    for col in macro.columns:
        s = macro[col].dropna()
        out[col] = {
            "n_obs": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(ddof=1),
            "min": s.min(),
            "max": s.max(),
            "first_value": s.iloc[0],
            "last_value": s.iloc[-1],
            "n_missing": int(macro[col].isna().sum()),
        }
    return pd.DataFrame(out).T.rename_axis("series")


# ---------------------------------------------------------------------------
# Data-quality audits
# ---------------------------------------------------------------------------

def missingness_audit(
    raw_prices: dict[str, pd.DataFrame],
    raw_macro: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Missing values before and after alignment, per series."""
    rows = {}
    for ticker, df in raw_prices.items():
        rows[ticker] = {
            "kind": "price",
            "n_raw": len(df),
            "n_missing_raw": int(df["Adj Close"].isna().sum()),
            "n_aligned": int(prices[ticker].notna().sum()),
            "n_missing_aligned": int(prices[ticker].isna().sum()),
        }
    for series_id, df in raw_macro.items():
        aligned = macro[series_id]
        first_valid = aligned.first_valid_index()
        leading = int(aligned.loc[:first_valid].isna().sum()) if first_valid is not None else len(aligned)
        rows[series_id] = {
            "kind": "macro",
            "n_raw": len(df),
            "n_missing_raw": int(df[series_id].isna().sum()),
            "n_aligned": int(aligned.notna().sum()),
            "n_missing_aligned": int(aligned.isna().sum()),
            "n_leading_nan_aligned": leading,
        }
    return pd.DataFrame(rows).T.rename_axis("series")


def _max_identical_run(series: pd.Series) -> int:
    """Length of the longest run of consecutive identical values."""
    s = series.dropna()
    if s.empty:
        return 0
    change = s.ne(s.shift()).cumsum()
    return int(change.value_counts().max())


def staleness_audit(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Stale-value diagnostics: identical-value runs and zero returns.

    A long identical-value run in a *price* suggests a stale quote or a
    data hole papered over; in a *macro policy rate* (DFF) it is
    economically real. The audit records both; interpretation happens
    in the report, and nothing is silently corrected.
    """
    rows = {}
    for col in prices.columns:
        s = prices[col]
        zero_ret = int((s.pct_change() == 0).sum())
        rows[col] = {
            "kind": "price",
            "max_identical_run_days": _max_identical_run(s),
            "n_zero_return_days": zero_ret,
            "pct_unchanged_days": zero_ret / max(len(s) - 1, 1) * 100,
        }
    for col in macro.columns:
        s = macro[col]
        unchanged = int((s.diff() == 0).sum())
        rows[col] = {
            "kind": "macro",
            "max_identical_run_days": _max_identical_run(s),
            "n_zero_change_days": unchanged,
            "pct_unchanged_days": unchanged / max(len(s) - 1, 1) * 100,
        }
    return pd.DataFrame(rows).T.rename_axis("series")


def calendar_audit(index: pd.DatetimeIndex, gap_threshold_days: int = 4) -> pd.DataFrame:
    """Trading-calendar diagnostics on the aligned panel index.

    Flags duplicate dates, weekend rows, and calendar gaps longer than
    ``gap_threshold_days`` (ordinary weekends plus single holidays are
    at most 4 calendar days between closes).
    """
    gaps = pd.Series(index[1:] - index[:-1], index=index[1:]).dt.days
    long_gaps = gaps[gaps > gap_threshold_days].sort_values(ascending=False)
    rows = {
        "n_days": len(index),
        "first_date": str(index.min().date()),
        "last_date": str(index.max().date()),
        "n_duplicate_dates": int(index.duplicated().sum()),
        "n_weekend_rows": int(index.dayofweek.isin([5, 6]).sum()),
        "max_calendar_gap_days": int(gaps.max()),
        f"n_gaps_over_{gap_threshold_days}d": int(len(long_gaps)),
        "long_gap_dates": "; ".join(
            f"{d.date()} ({g}d)" for d, g in long_gaps.head(10).items()
        ),
    }
    return pd.DataFrame({"value": rows}).rename_axis("check")


# ---------------------------------------------------------------------------
# Correlation structure and conditioning
# ---------------------------------------------------------------------------

def correlation_and_conditioning(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-sample correlation matrix plus eigenvalue diagnostics.

    The condition number of the correlation matrix quantifies how close
    the asset universe is to collinearity, previewing the Phase 7
    discussion of why sample covariance inversion is fragile.
    """
    corr = returns.corr()
    eigvals = np.linalg.eigvalsh(corr.to_numpy())[::-1]
    diag = pd.DataFrame(
        {
            "value": {
                **{f"eigenvalue_{i + 1}": ev for i, ev in enumerate(eigvals)},
                "condition_number": eigvals[0] / eigvals[-1],
                "variance_share_pc1_pct": eigvals[0] / eigvals.sum() * 100,
            }
        }
    ).rename_axis("diagnostic")
    return corr, diag


def realized_vol_series(prices: pd.DataFrame, window: int = 21) -> pd.Series:
    """Annualized rolling realized volatility of SPY, in percent."""
    rv = prep.rolling_volatility(prep.log_returns(prices)["SPY"], window=window) * 100
    rv.name = f"SPY_realized_vol_{window}d_pct"
    return rv


def macro_feature_correlations(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Correlations among candidate HMM features, levels and changes.

    The VIX / realized-volatility redundancy question is answered here:
    the table reports level correlation, 21-day change correlation, and
    the correlation of VIX today with realized volatility over the
    *next* 21 days (descriptive; no model input).
    """
    returns = prep.log_returns(prices)
    rv = realized_vol_series(prices, window=21)
    vix = macro["VIXCLS"]
    slope = prep.yield_curve_slope(macro)

    features = pd.DataFrame(
        {
            "mkt_log_return": returns["SPY"],
            "realized_vol_21d": rv,
            "log_vix": np.log(vix),
            "yield_slope": slope,
            "dff": macro["DFF"],
        }
    ).dropna()

    level_corr = features.corr()
    level_corr.index = [f"level:{c}" for c in level_corr.index]

    changes = features.diff().dropna()
    change_corr = changes.corr()
    change_corr.index = [f"change:{c}" for c in change_corr.index]

    fwd_rv = rv.shift(-21)  # realized vol over the NEXT 21 days
    extra = pd.DataFrame(
        {
            "value": {
                "corr(VIX_level, RV21_level)": vix.corr(rv),
                "corr(dVIX_21d, dRV21_21d)": vix.diff(21).corr(rv.diff(21)),
                "corr(VIX_t, RV21_next21d)": vix.corr(fwd_rv),
            }
        }
    ).rename_axis("statistic")

    combined = pd.concat([level_corr, change_corr])
    combined.attrs["vix_rv_stats"] = extra
    return combined, extra


# ---------------------------------------------------------------------------
# Subperiods
# ---------------------------------------------------------------------------

def subperiod_summaries(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Descriptive per-subperiod statistics (labeled descriptive only).

    Per asset: annualized CAGR, annualized volatility, max drawdown
    (measured within the subperiod). Per period: SPY-IEF daily return
    correlation and mean VIX.
    """
    rows = []
    for name, (start, end) in periods.items():
        p = prices.loc[start:end]
        m = macro.loc[start:end]
        if len(p) < 42:
            continue
        r = prep.log_returns(p)
        years = len(r) / TRADING_DAYS
        for asset in p.columns:
            rows.append(
                {
                    "period": name,
                    "start": str(p.index.min().date()),
                    "end": str(p.index.max().date()),
                    "asset": asset,
                    "ann_return_cagr_pct": ((p[asset].iloc[-1] / p[asset].iloc[0]) ** (1 / years) - 1) * 100,
                    "ann_vol_pct": r[asset].std(ddof=1) * np.sqrt(TRADING_DAYS) * 100,
                    "max_drawdown_pct": prep.max_drawdown(p[asset]) * 100,
                    "spy_ief_corr": r["SPY"].corr(r["IEF"]),
                    "mean_vix": m["VIXCLS"].mean(),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def write_phase_manifest(
    output_paths: list[Path],
    project_root: Path,
    snapshot_manifest_path: Path,
    out_path: Path,
    phase: str = "4-eda",
    note: str = (
        "Descriptive analysis only. No volatility model, HMM, portfolio "
        "optimization, or backtest was estimated or examined in Phase 4."
    ),
) -> Path:
    """Record git commit, config hash, snapshot hash, and artifact hashes."""
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    config_hash = hashlib.sha256()
    for name in ("config.yaml", "analysis_plan.yaml"):
        config_hash.update((project_root / "config" / name).read_bytes())
    manifest = {
        "phase": phase,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "config_sha256": config_hash.hexdigest(),
        "data_snapshot_manifest": snapshot_manifest_path.name,
        "data_snapshot_sha256": sha256_of(snapshot_manifest_path),
        "note": note,
        "artifacts": {
            str(p.relative_to(project_root)).replace("\\", "/"): sha256_of(p)
            for p in output_paths
        },
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


# Backward-compatible alias (Phase 4 call sites and tests)
write_eda_manifest = write_phase_manifest
