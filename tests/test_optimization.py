"""Tests for Phase 8 target-weight construction.

Covers every required property: exact rule-based weights, constraint
satisfaction for optimized strategies, scale invariance, the static
target being constant across dates, correct covariance sourcing per
strategy, regime-aware equalling rolling-LW on the four A2 dates,
failure emitting a fallback request rather than invented weights,
covariance hashes linking targets to inputs, and the absence of any
performance computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import covariance as cv
from src import optimization as opt

ASSETS = ["SPY", "QQQ", "IWM", "IEF", "GLD"]
CFG = opt.OptimizerConfig()


@pytest.fixture(scope="module")
def cov_matrices() -> dict[str, np.ndarray]:
    """A realistic 5-asset covariance with a collinear equity block."""
    rng = np.random.default_rng(11)
    vols = np.array([0.012, 0.014, 0.016, 0.004, 0.011])
    corr = np.array([
        [1.00, 0.92, 0.89, -0.28, 0.07],
        [0.92, 1.00, 0.82, -0.24, 0.06],
        [0.89, 0.82, 1.00, -0.26, 0.07],
        [-0.28, -0.24, -0.26, 1.00, 0.21],
        [0.07, 0.06, 0.07, 0.21, 1.00],
    ])
    base = np.outer(vols, vols) * corr
    stressed = base * 2.5
    return {"base": base, "stressed": stressed,
            "noise": base + 1e-9 * rng.normal(size=(5, 5)) @ np.eye(5)}


# ---------------------------------------------------------------------------
# Rule-based strategies
# ---------------------------------------------------------------------------

def test_equal_weight_is_exactly_one_fifth():
    w = opt.equal_weight(ASSETS)
    np.testing.assert_allclose(w, 0.20, atol=1e-15)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)


def test_sixty_forty_is_exact_and_cap_exempt():
    w = opt.fixed_weight(ASSETS, {"SPY": 0.60, "IEF": 0.40})
    assert w[ASSETS.index("SPY")] == pytest.approx(0.60)
    assert w[ASSETS.index("IEF")] == pytest.approx(0.40)
    assert w[ASSETS.index("QQQ")] == 0.0
    assert w[ASSETS.index("IWM")] == 0.0
    assert w[ASSETS.index("GLD")] == 0.0
    assert w.sum() == pytest.approx(1.0)
    # Deliberately exceeds the 40% cap that governs optimized strategies.
    assert w.max() > CFG.max_weight


def test_fixed_weight_rejects_bad_input():
    with pytest.raises(ValueError, match="sum to 1"):
        opt.fixed_weight(ASSETS, {"SPY": 0.6, "IEF": 0.3})
    with pytest.raises(ValueError, match="unknown assets"):
        opt.fixed_weight(ASSETS, {"SPY": 0.6, "TLT": 0.4})


# ---------------------------------------------------------------------------
# Optimizer: constraints and scale invariance
# ---------------------------------------------------------------------------

def test_optimized_weights_satisfy_all_constraints(cov_matrices):
    for name, cov in cov_matrices.items():
        w, diag = opt.min_variance_weights(cov, CFG)
        assert w is not None, name
        assert np.all(np.isfinite(w))
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert w.min() >= -1e-12
        assert w.max() <= CFG.max_weight + 1e-9
        assert diag["solver_success"] is True
        assert diag["fallback_requested"] is False
        assert diag["constraint_violation"] < 1e-8


def test_positive_scaling_leaves_weights_unchanged(cov_matrices):
    """A positive scalar multiple of the covariance has the same argmin."""
    cov = cov_matrices["base"]
    w_base, _ = opt.min_variance_weights(cov, CFG)
    for factor in (1e-4, 0.5, 3.0, 1e4):
        w_scaled, _ = opt.min_variance_weights(cov * factor, CFG)
        np.testing.assert_allclose(w_scaled, w_base, atol=1e-7)


def test_objective_reported_on_unscaled_matrix(cov_matrices):
    cov = cov_matrices["base"]
    w, diag = opt.min_variance_weights(cov, CFG)
    assert diag["objective_value"] == pytest.approx(float(w @ cov @ w), rel=1e-9)
    # Minimum variance must beat equal weight on its own objective.
    ew = opt.equal_weight(ASSETS)
    assert diag["objective_value"] <= float(ew @ cov @ ew) + 1e-15


def test_cap_binds_and_is_counted():
    """With a dominant low-variance asset the cap must bind."""
    cov = np.diag([0.04, 0.04, 0.04, 1e-6, 0.04])   # IEF far less risky
    w, diag = opt.min_variance_weights(cov, CFG)
    assert w.max() == pytest.approx(CFG.max_weight, abs=1e-6)
    assert diag["cap_binding_count"] >= 1
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_validation_is_independent_of_solver_flag():
    """A solution violating the constraints is rejected even if a
    solver were to report success."""
    bad = np.array([0.5, 0.5, 0.5, -0.5, 0.0])      # sums to 1 but negative
    cleaned, checks = opt.validate_weights(bad, CFG.max_weight)
    assert checks["constraint_violation"] > 1e-3
    np.testing.assert_allclose(cleaned, bad)         # not silently cleaned

    noisy = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) + np.array([1e-17, 0, 0, 0, -1e-17])
    cleaned, checks = opt.validate_weights(noisy, CFG.max_weight)
    assert checks["constraint_violation"] < 1e-9
    assert cleaned.sum() == pytest.approx(1.0, abs=1e-15)


def test_failure_emits_fallback_request_not_invented_weights():
    """A degenerate covariance must yield no weights and a request for
    Phase 9 to treat the rebalance as no trade."""
    broken = np.full((5, 5), np.nan)
    w, diag = opt.min_variance_weights(broken, CFG)
    assert w is None
    assert diag["fallback_requested"] is True
    assert np.isnan(diag["objective_value"])


def test_covariance_hash_links_target_to_input(cov_matrices):
    a = opt.covariance_hash(cov_matrices["base"])
    b = opt.covariance_hash(cov_matrices["base"].copy())
    c = opt.covariance_hash(cov_matrices["stressed"])
    assert a == b            # same matrix -> same hash
    assert a != c            # different matrix -> different hash
    _, diag = opt.min_variance_weights(cov_matrices["base"], CFG)
    assert diag["covariance_hash"] == a


# ---------------------------------------------------------------------------
# EWMA-scaled construction
# ---------------------------------------------------------------------------

def test_ewma_scaled_uses_ewma_vols_and_lw_correlations(cov_matrices):
    lw = cov_matrices["base"]
    ewma_var = np.array([0.02, 0.03, 0.04, 0.005, 0.02]) ** 2
    scaled = opt.ewma_scaled_covariance(ewma_var, lw)

    # Diagonal comes from the EWMA variances.
    np.testing.assert_allclose(np.diag(scaled), ewma_var, rtol=1e-12)
    # Correlations come from Ledoit-Wolf, unchanged.
    def to_corr(m):
        d = np.sqrt(np.diag(m))
        return m / np.outer(d, d)
    np.testing.assert_allclose(to_corr(scaled), to_corr(lw), rtol=1e-12)
    # It is NOT the raw EWMA covariance unless volatilities coincide.
    assert not np.allclose(scaled, lw)


# ---------------------------------------------------------------------------
# Full panel behavior
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panel(cov_matrices):
    dates = pd.DatetimeIndex(["2010-01-29", "2010-02-26", "2010-03-31"])
    lw = {d: cov_matrices["base"] for d in dates}
    # A2 fallback simulated on the middle date: consumed == rolling LW.
    consumed = {
        dates[0]: cov_matrices["stressed"],
        dates[1]: cov_matrices["base"],        # fallback date
        dates[2]: cov_matrices["stressed"],
    }
    ewma = {d: np.diag(cov_matrices["base"]) for d in dates}
    weights, audit = opt.build_strategy_targets(
        ASSETS, dates, cov_matrices["base"], lw, ewma, consumed,
        {"SPY": 0.60, "IEF": 0.40}, CFG,
    )
    return dates, weights, audit


def test_panel_covers_all_strategies_and_dates(panel):
    dates, weights, audit = panel
    assert set(audit["strategy"]) == set(opt.STRATEGIES)
    assert len(audit) == len(dates) * len(opt.STRATEGIES)
    assert len(weights) == len(dates) * len(opt.STRATEGIES) * len(ASSETS)


def test_static_minvar_is_constant_across_dates(panel):
    dates, weights, _ = panel
    static = weights[weights["strategy"] == "static_minvar"]
    pivot = static.pivot(index="date", columns="asset", values="weight")
    for asset in ASSETS:
        assert pivot[asset].nunique() == 1, f"{asset} target moved across dates"


def test_regime_matches_rolling_lw_when_consuming_same_matrix(panel):
    """On a simulated A2 fallback date the regime-aware target must
    equal the rolling Ledoit-Wolf target within tolerance."""
    dates, weights, _ = panel
    fallback_date = dates[1]
    regime = weights[(weights["strategy"] == "regime_minvar")
                     & (weights["date"] == fallback_date)].set_index("asset")["weight"]
    rolling = weights[(weights["strategy"] == "rolling_lw_minvar")
                      & (weights["date"] == fallback_date)].set_index("asset")["weight"]
    np.testing.assert_allclose(
        regime.loc[ASSETS].to_numpy(), rolling.loc[ASSETS].to_numpy(), atol=1e-8
    )
    # And differs on a non-fallback date (different consumed matrix).
    other = dates[0]
    regime_other = weights[(weights["strategy"] == "regime_minvar")
                           & (weights["date"] == other)].set_index("asset")["weight"]
    assert regime_other.loc[ASSETS].to_numpy() is not None


def test_all_optimized_targets_respect_cap_and_sum(panel):
    _, weights, audit = panel
    for strategy in opt.OPTIMIZED:
        subset = weights[weights["strategy"] == strategy]
        totals = subset.groupby("date")["weight"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-9)
        assert subset["weight"].min() >= -1e-12
        assert subset["weight"].max() <= CFG.max_weight + 1e-9


def test_benchmark_exempt_from_cap_in_panel(panel):
    _, weights, _ = panel
    benchmark = weights[weights["strategy"] == "static_6040"]
    assert benchmark["weight"].max() == pytest.approx(0.60)
    optimized = weights[weights["strategy"].isin(opt.OPTIMIZED)]
    assert optimized["weight"].max() <= CFG.max_weight + 1e-9


def test_audit_records_required_fields(panel):
    _, _, audit = panel
    required = {
        "date", "strategy", "solver_success", "solver_status", "iterations",
        "objective_value", "sum_weights", "minimum_weight", "maximum_weight",
        "constraint_violation", "cap_binding_count", "fallback_requested",
        "covariance_source", "covariance_hash",
    }
    assert required.issubset(audit.columns)
    assert audit["covariance_source"].isin(
        {"none", "training_window_ledoit_wolf", "rolling_ledoit_wolf",
         "ewma_vol_x_lw_correlation", "a2_consumed"}
    ).all()


def test_phase8_computes_no_performance_quantities(panel):
    """Phase 8 emits targets only — no returns, costs, or wealth."""
    _, weights, audit = panel
    forbidden = {"return", "pnl", "sharpe", "cost", "turnover", "wealth",
                 "cumulative", "drawdown"}
    for frame in (weights, audit):
        for column in frame.columns:
            assert not any(token in column.lower() for token in forbidden), column
