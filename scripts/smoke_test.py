"""End-to-end smoke test on synthetic data. No market data, no network.

For a reviewer who wants to confirm the pipeline runs before deciding
whether to obtain the frozen snapshot. Exercises every stage — features,
HMM, covariance, optimization, accounting, metrics, bootstrap — on
generated data with a known embedded volatility regime, and checks the
invariants the real pipeline relies on.

It verifies that the machinery works. It says nothing about whether the
paper's numbers are right; only the frozen snapshot can establish that.

    python scripts/smoke_test.py

Runs in roughly 30 seconds and exits non-zero on any failure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import backtest as bt  # noqa: E402
from src import covariance as cv  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import optimization as opt  # noqa: E402
from src import preprocessing as prep  # noqa: E402
from src import regimes as rg  # noqa: E402
from src import statistical_tests as st  # noqa: E402
from src import units as un  # noqa: E402
from src import volatility as vol  # noqa: E402

ASSETS = ["SPY", "QQQ", "IWM", "IEF", "GLD"]
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def synthetic_market(n_days: int = 1600, seed: int = 7):
    """Prices and macro with a deliberate high-volatility episode.

    Days 700-850 carry triple volatility and a correlation spike, so a
    two-state HMM has something real to find.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n_days, name="Date")

    stressed = np.zeros(n_days, dtype=bool)
    stressed[700:850] = True
    market = rng.normal(0.0004, np.where(stressed, 0.030, 0.008))

    betas = {"SPY": 1.0, "QQQ": 1.15, "IWM": 1.25, "IEF": -0.25, "GLD": 0.05}
    idio = {"SPY": 0.003, "QQQ": 0.005, "IWM": 0.007, "IEF": 0.002, "GLD": 0.008}
    returns = pd.DataFrame(
        {a: betas[a] * market + rng.normal(0, idio[a], n_days) for a in ASSETS},
        index=idx,
    )
    prices = 100 * (1 + returns).cumprod()

    realized = pd.Series(market, index=idx).rolling(21).std()
    macro = pd.DataFrame(
        {
            "VIXCLS": (12 + 900 * realized.fillna(realized.mean())).clip(9, 80),
            "DGS10": 3.0 + np.cumsum(rng.normal(0, 0.01, n_days)),
            "DGS2": 2.5 + np.cumsum(rng.normal(0, 0.012, n_days)),
            "DFF": np.clip(2.0 + np.cumsum(rng.normal(0, 0.005, n_days)), 0, None),
        },
        index=idx,
    )
    return prices, macro, stressed


def main() -> None:
    started = time.time()
    print("Smoke test: synthetic data, no network, no market data.\n")

    prices, macro, stressed = synthetic_market()
    log_returns = prep.log_returns(prices)
    print(f"Generated {len(prices)} days x {len(ASSETS)} assets "
          f"with a {int(stressed.sum())}-day stressed episode.\n")

    # --- preprocessing -----------------------------------------------
    print("Preprocessing")
    prep.assert_causal(lambda df: prep.rolling_volatility(prep.log_returns(df), 21),
                       prices)
    check("rolling features are causal", True, "assert_causal passed")
    lagged = prep.apply_signal_lag(macro["VIXCLS"], 1)
    check("signal lag shifts by one day",
          lagged.iloc[5] == macro["VIXCLS"].iloc[4])

    # --- volatility ---------------------------------------------------
    print("\nVolatility")
    ewma = vol.ewma_variance_series(log_returns["SPY"], lam=0.94)
    check("EWMA variances positive", bool((ewma.dropna() > 0).all()))
    check("EWMA responds to the stressed episode",
          ewma.iloc[700:850].mean() > 3 * ewma.iloc[100:600].mean(),
          f"ratio {ewma.iloc[700:850].mean() / ewma.iloc[100:600].mean():.1f}x")
    q = vol.qlike(np.array([4e-4]), np.array([4e-4]))
    check("QLIKE is zero at a perfect forecast", abs(q[0]) < 1e-12)

    # --- regimes ------------------------------------------------------
    print("\nRegimes (HMM)")
    features = rg.build_features(prices, macro, macro_lag_days=1)
    check("feature panel built", list(features.columns) == rg.FEATURE_NAMES,
          f"{len(features)} rows")
    cfg = rg.HMMConfig(n_init=4, n_iter=60, min_observations=252)
    origins = pd.DatetimeIndex([features.index[900], features.index[1100]])
    out = rg.run_expanding_hmm(features, origins, cfg)
    realtime = out["realtime"]
    check("probabilities sum to one",
          bool(np.allclose(realtime["prob_s0"] + realtime["prob_s1"], 1.0)))
    check("transition rows sum to one",
          bool(np.allclose(
              out["transitions"].groupby(["date", "from_state"])["probability"].sum(), 1.0)))
    check("A2 routing is flag-driven, not date-driven",
          rg.covariance_consumption({"absorbing_state": True,
                                     "degenerate_occupancy": False,
                                     "singular_covariance": False})["fallback_used"])
    check("no initialization failures",
          int(out["initializations"].eval("~usable").sum()) == 0)

    # --- covariance ---------------------------------------------------
    print("\nCovariance")
    window = log_returns.loc[: origins[0]]
    lw, intensity = cv.ledoit_wolf_covariance(window)
    sample = cv.sample_covariance(window)
    lw_cond, _ = cv.condition_numbers(lw)
    sample_cond, _ = cv.condition_numbers(sample)
    check("Ledoit-Wolf improves conditioning", lw_cond <= sample_cond,
          f"{sample_cond:.1f} -> {lw_cond:.1f}")
    _, psd = cv.enforce_psd(lw, "smoke")
    check("estimator is PSD without correction", not psd["psd_correction_used"])
    try:
        bad = np.array([[1.0, 0.0], [0.0, -0.3]])
        cv.enforce_psd(bad, "broken")
        check("materially non-PSD matrix is rejected", False, "no exception raised")
    except cv.MateriallyNonPSDError:
        check("materially non-PSD matrix is rejected", True)
    p_bar = cv.horizon_average_probabilities(
        np.array([0.3, 0.7]), np.array([[0.95, 0.05], [0.1, 0.9]]), 21)
    check("horizon probabilities sum to one", abs(p_bar.sum() - 1.0) < 1e-12)

    # --- optimization -------------------------------------------------
    print("\nPortfolio construction")
    opt_cfg = opt.OptimizerConfig(max_weight=0.40)
    weights, diagnostics = opt.min_variance_weights(lw, opt_cfg)
    check("optimizer converged", diagnostics["solver_success"])
    check("weights sum to one", abs(weights.sum() - 1.0) < 1e-9)
    check("weights respect the 40% cap and long-only",
          weights.max() <= 0.4 + 1e-9 and weights.min() >= -1e-12,
          f"max {weights.max():.3f}")
    scaled, _ = opt.min_variance_weights(lw * 1e4, opt_cfg)
    check("weights invariant to covariance scaling",
          bool(np.allclose(weights, scaled, atol=1e-7)))
    check("60/40 benchmark is exempt from the cap",
          opt.fixed_weight(ASSETS, {"SPY": 0.6, "IEF": 0.4}).max() == 0.6)

    # --- accounting ---------------------------------------------------
    print("\nAccounting")
    simple = prices.pct_change().iloc[1:]
    execution_dates = simple.index[[300, 320, 340, 360]]
    targets = {d: weights.copy() for d in execution_dates}
    path = bt.simulate_path(simple, targets)
    events = path["events"]
    check("entry trade sum is one",
          abs(events.iloc[0]["full_trade_sum"] - 1.0) < 1e-12)
    check("no return accrues before the first execution",
          events.iloc[0]["gross_return"] == 0.0)
    costed = bt.apply_costs(events, 10.0)
    check("wealth identity holds exactly",
          float(costed["wealth_identity_error"].max()) < 1e-15)
    zero = bt.apply_costs(events, 0.0)
    check("zero-cost net equals gross",
          bool(np.allclose(zero["net_return"], zero["gross_return"], atol=1e-18)))
    check("gross returns are cost-independent",
          bool(np.allclose(costed["gross_return"], zero["gross_return"])))

    # --- metrics and inference ---------------------------------------
    print("\nMetrics and inference")
    net = costed["net_return"]
    rf = pd.Series(0.0, index=net.index)
    wealth = mt.wealth_path(net)
    check("CAGR reconciles with the wealth path",
          abs((1 + mt.cagr(wealth)) ** ((wealth.index[-1] - wealth.index[0]).days / 365.25)
              - wealth.iloc[-1]) < 1e-9)
    check("VaR and ES report losses as positive",
          mt.historical_var(net, 0.95) > 0 and
          mt.expected_shortfall(net, 0.95) >= mt.historical_var(net, 0.95))

    boot = st.BootstrapConfig(n_replications=300, batch_size=150)
    draws = st.paired_bootstrap_differences(net, net.copy(), net, net.copy(),
                                            ("sharpe",), boot)["sharpe"]
    check("paired resampling cancels identical series",
          bool(np.allclose(draws, 0.0, atol=1e-12)))
    a = st.stationary_bootstrap_indices(400, 21, 20, np.random.default_rng(12345))
    b = st.stationary_bootstrap_indices(400, 21, 20, np.random.default_rng(12345))
    check("bootstrap indices reproduce from the seed",
          bool(np.array_equal(a, b)))

    # --- units --------------------------------------------------------
    print("\nUnit conversion")
    check("0.0000314 -> 0.314 bps", abs(un.decimal_to_bps(0.0000314) - 0.314) < 1e-9)
    check("0.00175 -> 0.175 pp",
          abs(un.decimal_to_percentage_points(0.00175) - 0.175) < 1e-9)
    frame = un.build_display_table({"sharpe_difference": 0.020966,
                                    "vol_difference": -0.00175})
    un.verify_display_table(frame)
    check("display table verifies against raw values", True)
    frame.loc[0, "display_value"] = 99.0
    try:
        un.verify_display_table(frame)
        check("tampered display value is caught", False, "no exception")
    except ValueError:
        check("tampered display value is caught", True)

    # --- summary ------------------------------------------------------
    failed = [name for name, ok, _ in CHECKS if not ok]
    elapsed = time.time() - started
    print(f"\n{'-' * 62}")
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed in {elapsed:.1f}s")
    if failed:
        print("\nFAILED:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    print("\nPipeline executes end to end on synthetic data.")
    print("This does NOT reproduce the paper's results, which require the")
    print("frozen 2026-08-06 snapshot. See docs/REPRODUCTION.md.")


if __name__ == "__main__":
    main()
