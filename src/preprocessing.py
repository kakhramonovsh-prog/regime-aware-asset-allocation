"""Build processed datasets and point-in-time features from raw data.

Every function in this module obeys one rule: the value of any derived
series at date ``t`` may depend only on observations dated ``t`` or
earlier. :func:`assert_causal` provides a direct test of that property
and is exercised in the unit tests for each rolling transformation.

Alignment policy (documented in ``data/README.md``):

* The asset panel keeps only dates on which **all** tickers have an
  adjusted close (inner join on trading days).
* Macro series are reindexed to the asset panel's trading days and
  forward-filled up to ``macro_ffill_limit`` days. Forward-filling a
  yield or an index close carries the last *observed* value forward,
  which is exactly the information an investor held on that day; it
  never uses future data. Backward-filling is never used.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_price_panel(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the adjusted-close panel (dates x tickers).

    Keeps only dates where every ticker has an adjusted close, so the
    panel starts at the youngest asset's first trading day and skips
    any date on which one exchange-traded product did not print.
    """
    panel = pd.DataFrame(
        {ticker: df["Adj Close"] for ticker, df in price_frames.items()}
    ).sort_index()
    panel = panel.dropna(how="any")
    if panel.empty:
        raise ValueError("price panel is empty after alignment")
    panel.index.name = "Date"
    return panel


def build_macro_panel(
    macro_frames: dict[str, pd.DataFrame],
    trading_days: pd.DatetimeIndex,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Align macro series to the asset panel's trading days.

    Each series is reindexed to the union of its own dates and the
    target trading days, forward-filled up to ``ffill_limit`` days, then
    subset to the trading days. The two-step reindex means a macro
    observation printed on a non-trading day (e.g. DFF on a weekend)
    still carries into the next trading day.

    Raises
    ------
    ValueError
        If any series still has an internal gap after the limited
        forward-fill, which would signal a data problem rather than a
        routine holiday mismatch. Missing values *before* a series
        begins are left as NaN and documented, never filled.
    """
    columns = {}
    for series_id, df in macro_frames.items():
        series = df[series_id]
        union_index = series.index.union(trading_days)
        aligned = series.reindex(union_index).ffill(limit=ffill_limit)
        aligned = aligned.reindex(trading_days)
        first_valid = aligned.first_valid_index()
        if first_valid is not None:
            internal = aligned.loc[first_valid:]
            if internal.isna().any():
                n_gaps = int(internal.isna().sum())
                raise ValueError(
                    f"{series_id}: {n_gaps} missing values remain after "
                    f"forward-fill limit of {ffill_limit} days; inspect the raw file"
                )
        columns[series_id] = aligned
    panel = pd.DataFrame(columns, index=trading_days)
    panel.index.name = "Date"
    return panel


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic returns ``P_t / P_{t-1} - 1``. First row is dropped."""
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns ``ln(P_t / P_{t-1})``. First row is dropped."""
    return np.log(prices).diff().iloc[1:]


# ---------------------------------------------------------------------------
# Rolling statistics (all backward-looking)
# ---------------------------------------------------------------------------

def rolling_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int,
    annualize: bool = True,
) -> pd.DataFrame | pd.Series:
    """Rolling standard deviation of returns over ``window`` days.

    The window at date ``t`` covers ``t - window + 1`` through ``t``
    (pandas default), so no future observation enters the estimate.
    """
    vol = returns.rolling(window=window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    return returns_a.rolling(window=window, min_periods=window).corr(returns_b)


def rolling_covariance_matrices(
    returns: pd.DataFrame,
    window: int,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Backward-looking sample covariance matrix at each date.

    Returns a dict keyed by date; the matrix at date ``t`` uses returns
    from ``t - window + 1`` through ``t``. Dates with fewer than
    ``window`` prior observations are omitted.
    """
    matrices: dict[pd.Timestamp, pd.DataFrame] = {}
    values = returns.to_numpy()
    for i in range(window - 1, len(returns)):
        window_slice = values[i - window + 1 : i + 1]
        cov = np.cov(window_slice, rowvar=False, ddof=1)
        matrices[returns.index[i]] = pd.DataFrame(
            cov, index=returns.columns, columns=returns.columns
        )
    return matrices


def annualized_volatility(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Full-sample annualized volatility (descriptive use only)."""
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------

def drawdown_series(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Drawdown at each date: price relative to its running maximum.

    Values are in [-1, 0]; 0 means the series is at a new high. The
    running maximum uses only past and current prices.
    """
    running_max = prices.cummax()
    return prices / running_max - 1.0


def max_drawdown(prices: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Most negative drawdown over the sample."""
    return drawdown_series(prices).min()


# ---------------------------------------------------------------------------
# Macro features
# ---------------------------------------------------------------------------

def yield_curve_slope(macro: pd.DataFrame) -> pd.Series:
    """10y minus 2y Treasury slope in percentage points.

    Computed from DGS10 and DGS2 rather than taken from T10Y2Y so the
    construction is explicit; T10Y2Y is retained in the raw data as a
    cross-check.
    """
    if not {"DGS10", "DGS2"}.issubset(macro.columns):
        raise KeyError("macro panel must contain DGS10 and DGS2")
    slope = macro["DGS10"] - macro["DGS2"]
    slope.name = "yield_slope"
    return slope


def vix_changes(macro: pd.DataFrame, log: bool = True) -> pd.Series:
    """Daily VIX changes; log differences by default (VIX is positive
    and right-skewed, so log changes are closer to symmetric)."""
    if "VIXCLS" not in macro.columns:
        raise KeyError("macro panel must contain VIXCLS")
    vix = macro["VIXCLS"]
    change = np.log(vix).diff() if log else vix.diff()
    change.name = "vix_log_change" if log else "vix_change"
    return change


def apply_signal_lag(
    data: pd.DataFrame | pd.Series,
    lag_days: int = 1,
) -> pd.DataFrame | pd.Series:
    """Lag a series by ``lag_days`` trading days for use in trading signals.

    Frozen policy (docs/research_design.md §3): FRED-sourced series are
    lagged one extra trading day before entering any signal, because
    H.15 yields are published with a lag and same-close availability of
    the VIX print is not guaranteed for a close-of-day decision.
    Price-derived features carry lag 0.

    The value dated ``t`` in the returned frame was observed on trading
    day ``t - lag_days``.
    """
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative")
    return data.shift(lag_days)


# ---------------------------------------------------------------------------
# Look-ahead guard
# ---------------------------------------------------------------------------

def assert_causal(
    transform: Callable[[pd.DataFrame], pd.DataFrame | pd.Series],
    data: pd.DataFrame,
    n_checks: int = 5,
    rtol: float = 1e-12,
) -> None:
    """Verify that a transformation never uses future information.

    For ``n_checks`` cutoff dates spread through the sample, the
    transform is applied to (a) the full dataset and (b) the dataset
    truncated at the cutoff. If the outputs up to the cutoff differ,
    the transform looked ahead, and an ``AssertionError`` is raised.

    This is the project's primary defence against accidental look-ahead
    bias in feature construction; every rolling feature used by a
    strategy must pass it in the test suite.
    """
    full = transform(data)
    cutoffs = np.linspace(len(data) // 2, len(data) - 2, n_checks, dtype=int)
    for i in cutoffs:
        cutoff_date = data.index[i]
        truncated = transform(data.iloc[: i + 1])
        full_part = full.loc[:cutoff_date]
        trunc_part = truncated.loc[:cutoff_date]
        if isinstance(full_part, pd.Series):
            full_part = full_part.to_frame()
            trunc_part = trunc_part.to_frame()
        if not np.allclose(
            full_part.to_numpy(dtype=float),
            trunc_part.to_numpy(dtype=float),
            rtol=rtol,
            equal_nan=True,
        ):
            raise AssertionError(
                f"transform output before {cutoff_date.date()} changes when "
                "future data is removed: look-ahead bias detected"
            )
