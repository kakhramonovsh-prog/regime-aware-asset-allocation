"""Volatility forecasting and out-of-sample evaluation (Phase 5).

Frozen specification (config/analysis_plan.yaml, docs/research_design.md §4):

* Models: 63-day rolling historical variance, EWMA (RiskMetrics,
  lambda = 0.94, seeded with the first 63 days' sample variance), and
  GARCH(1,1) with normal innovations refit at every rebalance date.
* Estimation uses returns through the rebalance date ``t`` only.
* The forecast object is the **integrated variance of the next holding
  period** (the trading days after ``t`` up to and including the next
  rebalance date), compared against realized integrated variance (sum
  of squared daily log returns over the same days).
* Losses: QLIKE primary, MAE and RMSE secondary; identical evaluation
  dates for every model; Diebold-Mariano pairwise tests with HAC
  standard errors.
* Daily variance forecasts are floored at ``variance_floor_daily``;
  GARCH non-convergence substitutes the EWMA forecast and is logged.
* EWMA feeds the portfolio in later phases regardless of which model
  wins this comparison (fixed ex ante).

Implementation decisions, with honest provenance: the zero-mean GARCH
specification, the x100 return scaling, and HAC lag 3 for DM tests were
written into this module and its tests before the estimation run
executed, but they are NOT part of the plan frozen at
v0.2.0-preregistered and were first committed to git in the same commit
as the Phase 5 results. They are pre-estimation implementation choices,
not preregistered commitments. The same applies to excluding the final
partial month (not a full holding period).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

GARCH_SCALE = 100.0  # returns are scaled to percent for the optimizer


# ---------------------------------------------------------------------------
# Rebalance calendar
# ---------------------------------------------------------------------------

def month_end_rebalance_dates(
    index: pd.DatetimeIndex,
    min_observations: int = 252,
) -> pd.DatetimeIndex:
    """Last trading day of each month, usable as a forecast origin.

    A date qualifies only if at least ``min_observations`` return
    observations precede it (inclusive) and a subsequent month-end
    exists, so every forecast has a full holding period. The sample's
    final partial month is therefore excluded.
    """
    s = pd.Series(index, index=index)
    month_ends = pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())
    positions = index.get_indexer(month_ends)
    eligible = month_ends[(positions + 1) >= min_observations]
    return eligible[:-1]  # last month-end has no full holding period after it


def holding_period(
    index: pd.DatetimeIndex, t: pd.Timestamp, t_next: pd.Timestamp
) -> pd.DatetimeIndex:
    """Trading days strictly after ``t`` through ``t_next`` inclusive."""
    return index[(index > t) & (index <= t_next)]


# ---------------------------------------------------------------------------
# Per-day conditional variance estimators (causal by construction)
# ---------------------------------------------------------------------------

def hist_variance_series(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling sample variance through each date (ddof=1)."""
    return returns.rolling(window=window, min_periods=window).var(ddof=1)


def ewma_variance_series(
    returns: pd.Series, lam: float = 0.94, seed_window: int = 63
) -> pd.Series:
    """RiskMetrics next-day conditional variance, indexed at date t.

    The value at date ``t`` is the one-step-ahead variance forecast
    formed from returns through ``t``:
    ``sigma2_{t+1|t} = lam * sigma2_{t|t-1} + (1 - lam) * r_t^2``,
    seeded at the ``seed_window``-th observation with the sample
    variance of the first ``seed_window`` returns. Earlier dates are NaN.
    """
    r = returns.to_numpy()
    out = np.full(len(r), np.nan)
    if len(r) < seed_window:
        return pd.Series(out, index=returns.index, name=returns.name)
    sigma2 = np.var(r[:seed_window], ddof=1)
    out[seed_window - 1] = lam * sigma2 + (1 - lam) * r[seed_window - 1] ** 2
    for i in range(seed_window, len(r)):
        out[i] = lam * out[i - 1] + (1 - lam) * r[i] ** 2
    return pd.Series(out, index=returns.index, name=returns.name)


def garch_integrated_forecast(
    returns_through_t: pd.Series, horizon: int
) -> tuple[float, bool]:
    """GARCH(1,1)-normal integrated variance over ``horizon`` days.

    Fits on returns through the forecast origin only (zero-mean spec,
    returns scaled by 100), sums the per-step variance forecasts, and
    rescales to raw-return units. Returns ``(integrated_var, converged)``.
    Raises only on structural errors; optimizer non-convergence is
    reported through the flag.
    """
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            returns_through_t.to_numpy() * GARCH_SCALE,
            mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False,
        )
        result = model.fit(disp="off", show_warning=False, options={"maxiter": 500})
        forecast = result.forecast(horizon=horizon, reindex=False)
    ivar_scaled = float(forecast.variance.to_numpy()[-1].sum())
    converged = int(getattr(result, "convergence_flag", 1)) == 0
    return ivar_scaled / GARCH_SCALE**2, converged


# ---------------------------------------------------------------------------
# Forecast construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolatilityConfig:
    hist_window: int = 63
    ewma_lambda: float = 0.94
    ewma_seed_window: int = 63
    min_observations: int = 252
    variance_floor_daily: float = 1e-8


def build_volatility_forecasts(
    returns: pd.DataFrame,
    cfg: VolatilityConfig = VolatilityConfig(),
    garch_fitter=garch_integrated_forecast,
) -> pd.DataFrame:
    """Integrated-variance forecasts and realized targets, long format.

    One row per (rebalance date, asset, model). Columns: ``date``,
    ``next_date``, ``horizon_days``, ``asset``, ``model``,
    ``forecast_ivar``, ``realized_ivar``, ``floored``, ``converged``,
    ``substituted``. Every model appears on identical dates by
    construction. ``garch_fitter`` is injectable for tests.

    Causality: the hist/EWMA per-day series at ``t`` use returns
    through ``t``; the GARCH fit receives ``returns.loc[:t]`` only.
    The realized target uses days strictly after ``t``.
    """
    rebalance_dates = month_end_rebalance_dates(returns.index, cfg.min_observations)
    all_month_ends = pd.DatetimeIndex(
        pd.Series(returns.index, index=returns.index)
        .groupby(returns.index.to_period("M")).max().to_numpy()
    )

    hist = {a: hist_variance_series(returns[a], cfg.hist_window) for a in returns.columns}
    ewma = {
        a: ewma_variance_series(returns[a], cfg.ewma_lambda, cfg.ewma_seed_window)
        for a in returns.columns
    }

    rows: list[dict] = []
    for t in rebalance_dates:
        t_next = all_month_ends[all_month_ends > t][0]
        days = holding_period(returns.index, t, t_next)
        horizon = len(days)
        floor = cfg.variance_floor_daily * horizon
        for asset in returns.columns:
            realized = float((returns.loc[days, asset] ** 2).sum())
            ewma_ivar = float(ewma[asset].loc[t]) * horizon

            try:
                garch_ivar, converged = garch_fitter(returns.loc[:t, asset], horizon)
                substituted = False
            except Exception:
                garch_ivar, converged, substituted = ewma_ivar, False, True
            if not converged and not substituted:
                garch_ivar, substituted = ewma_ivar, True

            for model, ivar in (
                ("hist63", float(hist[asset].loc[t]) * horizon),
                ("ewma", ewma_ivar),
                ("garch11", garch_ivar),
            ):
                rows.append(
                    {
                        "date": t,
                        "next_date": t_next,
                        "horizon_days": horizon,
                        "asset": asset,
                        "model": model,
                        "forecast_ivar": max(ivar, floor),
                        "realized_ivar": realized,
                        "floored": ivar < floor,
                        "converged": converged if model == "garch11" else True,
                        "substituted": substituted if model == "garch11" else False,
                    }
                )
    out = pd.DataFrame(rows)
    _assert_common_dates(out)
    return out


def _assert_common_dates(forecasts: pd.DataFrame) -> None:
    """Every model must be evaluated on identical (date, asset) pairs."""
    pivot = forecasts.pivot_table(
        index=["date", "asset"], columns="model", values="forecast_ivar", aggfunc="size"
    )
    if pivot.isna().any().any() or (pivot != 1).any().any():
        raise AssertionError("models are not aligned on identical evaluation dates")


# ---------------------------------------------------------------------------
# Losses and comparison
# ---------------------------------------------------------------------------

def qlike(realized: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series):
    """QLIKE loss, robust to noisy volatility proxies (Patton 2011):
    ``L = rv/f - ln(rv/f) - 1``; zero when the forecast equals realized."""
    ratio = np.asarray(realized, dtype=float) / np.asarray(forecast, dtype=float)
    return ratio - np.log(ratio) - 1.0


def loss_table(forecasts: pd.DataFrame, oos_start: str | None = None) -> pd.DataFrame:
    """Mean losses per (asset, model); optionally also for dates >= oos_start."""
    def _block(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        grouped = df.groupby(["asset", "model"])
        out = pd.DataFrame(
            {
                f"qlike{suffix}": grouped.apply(
                    lambda g: float(np.mean(qlike(g["realized_ivar"], g["forecast_ivar"]))),
                    include_groups=False,
                ),
                f"mae_var_1e4{suffix}": grouped.apply(
                    lambda g: float(np.mean(np.abs(g["forecast_ivar"] - g["realized_ivar"]))) * 1e4,
                    include_groups=False,
                ),
                f"rmse_var_1e4{suffix}": grouped.apply(
                    lambda g: float(np.sqrt(np.mean((g["forecast_ivar"] - g["realized_ivar"]) ** 2))) * 1e4,
                    include_groups=False,
                ),
                f"n{suffix}": grouped.size(),
            }
        )
        return out

    table = _block(forecasts, "")
    if oos_start is not None:
        oos = _block(forecasts[forecasts["date"] >= pd.Timestamp(oos_start)], "_oos2010")
        table = table.join(oos)
    return table.reset_index()


def diebold_mariano(
    forecasts: pd.DataFrame,
    model_a: str,
    model_b: str,
    hac_lags: int = 3,
) -> pd.DataFrame:
    """DM test on QLIKE loss differentials per asset (a minus b).

    Negative mean differential favors ``model_a``. Reports the mean
    differential with its HAC (Newey-West, ``hac_lags`` lags) standard
    error and 95% confidence interval, not only the p-value. The ~250
    monthly observations caveat belongs in the report, not hidden here.
    """
    import statsmodels.api as sm

    rows = []
    for asset, group in forecasts.groupby("asset"):
        wide = group.pivot(index="date", columns="model", values="forecast_ivar")
        realized = group.pivot(index="date", columns="model", values="realized_ivar")[model_a]
        d = qlike(realized, wide[model_a]) - qlike(realized, wide[model_b])
        d = pd.Series(d, index=wide.index).dropna()
        ols = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}
        )
        mean_diff = float(d.mean())
        hac_se = float(ols.bse[0])
        rows.append(
            {
                "asset": asset,
                "comparison": f"{model_a} vs {model_b}",
                "mean_qlike_diff": mean_diff,
                "hac_se": hac_se,
                "ci95_lower": mean_diff - 1.96 * hac_se,
                "ci95_upper": mean_diff + 1.96 * hac_se,
                "dm_stat": float(ols.tvalues[0]),
                "p_value": float(ols.pvalues[0]),
                "n": len(d),
                "favors": model_a if mean_diff < 0 else model_b,
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: pd.Series | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values for a family of tests.

    Sort ascending, multiply p_(i) by (m - i), enforce monotonicity via
    a running maximum, cap at 1, and return in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum((m - np.arange(m)) * p[order], 1.0)
    )
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
