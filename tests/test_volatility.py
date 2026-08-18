"""Tests for Phase 5 volatility forecasting: formulas, causality,
alignment, failure handling, and the variance floor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as prep
from src import volatility as vol


@pytest.fixture(scope="module")
def returns(prices) -> pd.DataFrame:
    r = prep.log_returns(prices)
    r.columns = ["SPY", "IEF", "GLD"]
    return r


FAST_CFG = vol.VolatilityConfig(min_observations=252)


def _fake_garch(returns_through_t: pd.Series, horizon: int) -> tuple[float, bool]:
    """Deterministic stand-in: last-63-day variance times horizon."""
    return float(returns_through_t.iloc[-63:].var(ddof=1)) * horizon, True


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def test_rebalance_dates_are_month_ends_with_min_obs(returns):
    dates = vol.month_end_rebalance_dates(returns.index, min_observations=252)
    assert len(dates) > 0
    # Every date is its month's last trading day.
    for t in dates:
        month_days = returns.index[returns.index.to_period("M") == t.to_period("M")]
        assert t == month_days.max()
    # Minimum observations respected for the first origin.
    first_pos = returns.index.get_loc(dates[0])
    assert first_pos + 1 >= 252
    # The final month-end has no full holding period and must be excluded.
    all_month_ends = returns.index.to_series().groupby(
        returns.index.to_period("M")).max()
    assert dates[-1] < all_month_ends.iloc[-1]


def test_holding_period_excludes_origin_includes_endpoint(returns):
    dates = vol.month_end_rebalance_dates(returns.index, 252)
    t = dates[0]
    t_next = returns.index[returns.index > t][20]
    days = vol.holding_period(returns.index, t, t_next)
    assert t not in days
    assert days[-1] == t_next
    assert (days > t).all()


# ---------------------------------------------------------------------------
# Estimator formulas
# ---------------------------------------------------------------------------

def test_ewma_recursion_matches_manual_loop():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.01, 100))
    lam, seed = 0.94, 63
    series = vol.ewma_variance_series(r, lam=lam, seed_window=seed)

    sigma2 = r.iloc[:seed].var(ddof=1)
    manual = lam * sigma2 + (1 - lam) * r.iloc[seed - 1] ** 2
    assert series.iloc[seed - 1] == pytest.approx(manual, rel=1e-12)
    for i in range(seed, 100):
        manual = lam * manual + (1 - lam) * r.iloc[i] ** 2
    assert series.iloc[-1] == pytest.approx(manual, rel=1e-12)
    assert series.iloc[: seed - 1].isna().all()


def test_hist_variance_matches_pandas(returns):
    hist = vol.hist_variance_series(returns["SPY"], window=63)
    expected = returns["SPY"].iloc[100 - 62 : 101].var(ddof=1)
    assert hist.iloc[100] == pytest.approx(expected, rel=1e-12)


def test_qlike_formula():
    assert vol.qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(0.0)
    assert vol.qlike(np.array([2.0]), np.array([1.0]))[0] == pytest.approx(2 - np.log(2) - 1)
    # Wrong forecasts in either direction are penalized.
    assert (vol.qlike(np.array([1.0]), np.array([3.0])) > 0).all()


# ---------------------------------------------------------------------------
# Forecast construction
# ---------------------------------------------------------------------------

def test_forecasts_positive_aligned_and_realized_matches_sum(returns):
    fc = vol.build_volatility_forecasts(returns, FAST_CFG, garch_fitter=_fake_garch)
    assert (fc["forecast_ivar"] > 0).all()
    # Identical evaluation dates per model.
    by_model = fc.groupby("model")["date"].apply(lambda s: tuple(sorted(s.unique())))
    assert len(set(by_model)) == 1
    # Realized integrated variance equals the sum of squared returns
    # over days strictly after t through t_next.
    row = fc.iloc[0]
    days = vol.holding_period(returns.index, row["date"], row["next_date"])
    expected = float((returns.loc[days, row["asset"]] ** 2).sum())
    assert row["realized_ivar"] == pytest.approx(expected, rel=1e-12)
    assert len(days) == row["horizon_days"]


def test_variance_floor_applied():
    idx = pd.bdate_range("2020-01-01", periods=400)
    flat = pd.DataFrame({"AAA": np.zeros(400)}, index=idx)  # zero returns
    cfg = vol.VolatilityConfig(min_observations=252, variance_floor_daily=1e-8)
    fc = vol.build_volatility_forecasts(flat, cfg, garch_fitter=_fake_garch)
    floors = fc["horizon_days"] * 1e-8
    assert (fc["forecast_ivar"] >= floors - 1e-18).all()
    assert fc["floored"].any()


def test_garch_failure_substitutes_ewma_and_logs(returns):
    def failing_garch(series, horizon):
        raise RuntimeError("optimizer exploded")

    fc = vol.build_volatility_forecasts(returns, FAST_CFG, garch_fitter=failing_garch)
    garch = fc[fc["model"] == "garch11"]
    ewma = fc[fc["model"] == "ewma"]
    assert garch["substituted"].all()
    assert not garch["converged"].any()
    np.testing.assert_allclose(
        garch["forecast_ivar"].to_numpy(), ewma["forecast_ivar"].to_numpy()
    )


def test_nonconvergence_flag_substitutes_ewma(returns):
    def nonconverged_garch(series, horizon):
        return 99.0, False  # returns a value but reports non-convergence

    fc = vol.build_volatility_forecasts(returns, FAST_CFG, garch_fitter=nonconverged_garch)
    garch = fc[fc["model"] == "garch11"]
    ewma = fc[fc["model"] == "ewma"]
    assert garch["substituted"].all()
    np.testing.assert_allclose(
        garch["forecast_ivar"].to_numpy(), ewma["forecast_ivar"].to_numpy()
    )


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def test_forecasts_causal_under_future_perturbation(returns):
    fc_full = vol.build_volatility_forecasts(returns, FAST_CFG, garch_fitter=_fake_garch)
    cutoff = fc_full["date"].unique()[2]  # a mid-sample rebalance origin

    perturbed = returns.copy()
    future = perturbed.index > fc_full[fc_full["date"] == cutoff]["next_date"].iloc[0]
    rng = np.random.default_rng(0)
    perturbed.loc[future] += rng.uniform(0.05, 0.10, (int(future.sum()), perturbed.shape[1]))

    fc_pert = vol.build_volatility_forecasts(perturbed, FAST_CFG, garch_fitter=_fake_garch)
    keep = fc_full["date"] <= cutoff
    pd.testing.assert_frame_equal(
        fc_full[keep].reset_index(drop=True),
        fc_pert[keep.to_numpy()].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Real GARCH smoke test (small, one fit)
# ---------------------------------------------------------------------------

def test_real_garch_fit_smoke():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0, 0.012, 300))
    ivar, converged = vol.garch_integrated_forecast(r, horizon=21)
    assert ivar > 0
    assert isinstance(converged, bool)
    # A 21-day integrated variance should be near 21x the daily variance.
    assert ivar == pytest.approx(21 * r.var(ddof=1), rel=0.5)


# ---------------------------------------------------------------------------
# Losses and comparison
# ---------------------------------------------------------------------------

def test_loss_table_and_dm_shapes(returns):
    fc = vol.build_volatility_forecasts(returns, FAST_CFG, garch_fitter=_fake_garch)
    losses = vol.loss_table(fc)
    assert set(losses["model"]) == {"hist63", "ewma", "garch11"}
    assert len(losses) == 3 * len(returns.columns)
    assert losses["qlike"].notna().all()

    dm = vol.diebold_mariano(fc, "ewma", "hist63")
    assert len(dm) == len(returns.columns)
    assert dm["p_value"].between(0, 1).all()
    # CI is centered on the mean differential with HAC half-width.
    row = dm.iloc[0]
    assert row["ci95_lower"] == pytest.approx(row["mean_qlike_diff"] - 1.96 * row["hac_se"])
    assert row["ci95_upper"] == pytest.approx(row["mean_qlike_diff"] + 1.96 * row["hac_se"])
    assert row["ci95_lower"] <= row["mean_qlike_diff"] <= row["ci95_upper"]


def test_holm_adjustment_known_case():
    # Classic textbook check: p = (0.01, 0.04, 0.03, 0.005), m = 4.
    # Sorted: 0.005*4=0.02; 0.01*3=0.03; 0.03*2=0.06; 0.04*1=0.04 -> cummax 0.06.
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = vol.holm_adjust(p)
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    # Monotone in the original ordering of sorted p-values and capped at 1.
    assert (adj <= 1).all()
    big = vol.holm_adjust(np.array([0.5, 0.9, 0.7]))
    assert (big <= 1).all()
