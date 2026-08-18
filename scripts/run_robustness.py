"""Phase 12 robustness grid.

Each specification varies **one factor at a time** from the primary
model. No Cartesian product is constructed: crossing every state count,
feature set, cap, universe, seed, and shrinkage value would produce
hundreds of specifications and invite selection.

Feasibility constraint: a 30% cap is infeasible for a three-asset
universe (3 x 0.30 = 0.90 < 1), so the 30% cap runs on the five-asset
universe only. The three-asset core universe uses the primary 40% cap
(3 x 0.40 = 1.20, feasible).

Every specification reports the same block: net Sharpe difference at
10 bps with a paired stationary-bootstrap interval, volatility, CAGR and
drawdown differences, half-turnover and full traded notional, annualized
cost expenditure, event counts, the sign relative to the primary
estimate, and whether the interval contains zero.

Subperiods slice the existing primary strategy paths and are explicitly
descriptive. No robustness result may be promoted into the headline
conclusion.

Usage::

    python scripts/run_robustness.py [--only NAME ...] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src import backtest as bt  # noqa: E402
from src import covariance as cv  # noqa: E402
from src import data_loader, eda  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import optimization as opt  # noqa: E402
from src import preprocessing as prep  # noqa: E402
from src import regimes as rg  # noqa: E402
from src import statistical_tests as st  # noqa: E402

SNAPSHOT = PROJECT_ROOT / "data" / "snapshots" / "manifest_2026-08-06.json"
FULL_UNIVERSE = ["SPY", "QQQ", "IWM", "IEF", "GLD"]
CORE_UNIVERSE = ["SPY", "IEF", "GLD"]


@dataclass(frozen=True)
class Spec:
    """One robustness specification: exactly one factor differs."""

    name: str
    factor: str
    universe: tuple[str, ...] = tuple(FULL_UNIVERSE)
    n_states: int = 2
    drop_feature: str | None = None
    max_weight: float = 0.40
    neff_threshold: float = 60.0
    seed_start: int = 42
    accept_as_estimated: bool = False        # A2 robustness
    total_covariance_mixture: bool = False   # A3 robustness
    covariance_window: str = "expanding"

    @property
    def needs_hmm_refit(self) -> bool:
        """Does this spec change the HMM fits themselves?"""
        return (
            self.n_states != 2
            or self.drop_feature is not None
            or self.seed_start != 42
            or tuple(self.universe) != tuple(FULL_UNIVERSE)
        )


def build_specifications() -> list[Spec]:
    """The preregistered one-at-a-time robustness cases."""
    return [
        Spec("primary", "-"),
        Spec("hmm_3_states", "HMM states", n_states=3),
        Spec("drop_vix", "feature set", drop_feature="log_vix_lag1"),
        Spec("drop_realized_vol", "feature set", drop_feature="realized_vol_21d"),
        Spec("rolling_5y_window", "estimation window", covariance_window="rolling_5y"),
        # 30% cap is infeasible on 3 assets; five-asset universe only.
        Spec("cap_30pct", "weight cap", max_weight=0.30),
        Spec("cap_50pct", "weight cap", max_weight=0.50),
        Spec("neff_kappa_30", "shrinkage threshold", neff_threshold=30.0),
        Spec("neff_kappa_120", "shrinkage threshold", neff_threshold=120.0),
        Spec("core_universe", "universe", universe=tuple(CORE_UNIVERSE)),
        Spec("alt_seeds_100", "HMM seeds", seed_start=100),
        Spec("a2_accept_as_estimated", "A2 degeneracy rule", accept_as_estimated=True),
        Spec("a3_total_covariance", "A3 mixture formula", total_covariance_mixture=True),
    ]


# ---------------------------------------------------------------------------
# Pipeline for one specification
# ---------------------------------------------------------------------------

def covariances_for_spec(
    spec: Spec,
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    origins: pd.DatetimeIndex,
    hmm_cache: dict,
    config: dict,
    plan: dict,
) -> tuple[dict, dict, dict]:
    """Per-origin (consumed, rolling LW) matrices plus event counts."""
    assets = list(spec.universe)
    returns = prep.log_returns(prices[assets])
    features = rg.build_features(
        prices[assets], macro,
        macro_lag_days=config["data"]["macro_signal_lag_days"],
    )
    if spec.drop_feature:
        features = features.drop(columns=[spec.drop_feature])

    hmm_cfg = rg.HMMConfig(
        n_states=spec.n_states,
        n_init=config["regimes"]["n_initializations"],
        seed_start=spec.seed_start,
        min_occupancy=config["regimes"]["min_state_occupancy"],
        min_observations=plan["volatility"]["min_observations"],
    )

    # Canonical labeling always orders states by training-only expected
    # realized volatility. When that feature is dropped from the model,
    # the raw series is still supplied for labeling (A1/A2 policy).
    label_series = prep.rolling_volatility(
        prep.log_returns(prices[assets])["SPY"], window=21
    )

    # The cache key must contain EVERY model-determining setting. A
    # portfolio-only change (weight cap, shrinkage threshold, A2/A3 rule)
    # legitimately reuses the primary HMM; anything touching the model
    # must miss the cache.
    cache_key = (
        tuple(assets),                       # universe
        spec.n_states,                       # state count
        tuple(features.columns),             # feature set (post-drop)
        spec.seed_start, hmm_cfg.n_init,     # seed set
        hmm_cfg.min_observations,            # training-window rule
        "expanding_zscore_window_only",      # standardization rule
        _environment_hash(),                 # data + config hash
    )
    if cache_key not in hmm_cache:
        hmm_cache[cache_key] = _fit_hmm_path(
            features, origins, hmm_cfg, returns, label_series
        )
    fits = hmm_cache[cache_key]

    consumed: dict[pd.Timestamp, np.ndarray] = {}
    rolling_lw: dict[pd.Timestamp, np.ndarray] = {}
    events = {
        "fallbacks": 0, "guard_triggered": 0, "psd_corrections": 0,
        "failed_inits": int(sum(f["n_failed_inits"] for f in fits.values())),
    }

    for t in origins:
        fit = fits[t]
        window = fit["returns_window"]
        if spec.covariance_window == "rolling_5y":
            cutoff = t - pd.DateOffset(years=5)
            window = window.loc[window.index >= cutoff]
        lw, _ = cv.ledoit_wolf_covariance(window)
        lw = cv.symmetrize(lw)
        rolling_lw[t] = lw

        responsibilities = fit["responsibilities"][-len(window):]
        state_covs, state_means, n_eff = cv.state_conditional_moments(
            window, responsibilities
        )
        shrunk = [
            cv.shrink_state_covariance(c, lw, n, spec.neff_threshold)[0]
            for c, n in zip(state_covs, n_eff)
        ]
        p_bar = cv.horizon_average_probabilities(
            fit["p_t"], fit["transmat"], int(plan["regime_covariance"]["horizon_days"])
        )
        mixture, _ = cv.regime_mixture(
            p_bar, shrunk, state_means, spec.total_covariance_mixture
        )

        degenerate = fit["guard_triggered"]
        events["guard_triggered"] += int(degenerate)
        use_fallback = degenerate and not spec.accept_as_estimated
        events["fallbacks"] += int(use_fallback)
        chosen = lw if use_fallback else mixture
        corrected, psd = cv.enforce_psd(chosen, f"{t.date()}:{spec.name}")
        events["psd_corrections"] += int(psd["psd_correction_used"])
        consumed[t] = corrected

    return consumed, rolling_lw, events, fits


_ENVIRONMENT_HASH: str | None = None


def _environment_hash() -> str:
    """SHA-256 prefix over the data snapshot and both config files.

    Included in the HMM cache key so a cache entry can never survive a
    change to the data or configuration.
    """
    global _ENVIRONMENT_HASH
    if _ENVIRONMENT_HASH is None:
        import hashlib

        digest = hashlib.sha256()
        digest.update(SNAPSHOT.read_bytes())
        for name in ("config.yaml", "analysis_plan.yaml"):
            digest.update((PROJECT_ROOT / "config" / name).read_bytes())
        _ENVIRONMENT_HASH = digest.hexdigest()[:16]
    return _ENVIRONMENT_HASH


def _fit_hmm_path(features, origins, hmm_cfg, returns, label_series) -> dict:
    """Fit the HMM at every origin, caching what the covariances need."""
    fits: dict[pd.Timestamp, dict] = {}
    for t in origins:
        window = features.loc[:t]
        X = rg.standardize_window(window)
        model, records = rg.fit_multistart(X, hmm_cfg)
        if rg.REALIZED_VOL_FEATURE in window.columns:
            perm = rg.canonical_permutation(model, list(window.columns))
            vol_window = window
        else:
            aligned = label_series.reindex(window.index)
            perm = rg.canonical_permutation_by_series(model, X, aligned.to_numpy())
            vol_window = window.assign(**{rg.REALIZED_VOL_FEATURE: aligned})
        responsibilities = model.predict_proba(X)[:, perm]
        diagnostics = rg.state_diagnostics(model, X, vol_window, perm, hmm_cfg)
        fits[t] = {
            "responsibilities": responsibilities,
            "transmat": model.transmat_[np.ix_(perm, perm)],
            "p_t": responsibilities[-1],
            "guard_triggered": bool(diagnostics["guard_triggered"]),
            "diagnostics": diagnostics,
            "n_failed_inits": int(sum(1 for r in records if not r["usable"])),
            "returns_window": returns.loc[window.index[0] : t].reindex(window.index).dropna(),
        }
    return fits


def hmm_diagnostics_summary(fits: dict, spec: Spec) -> dict:
    """Per-specification HMM health, including canonical state ordering.

    Reports occupancy, effective sample size, transition degeneracy,
    fallback counts, and failed initializations for every state — the
    safeguards a three-state model needs most, recorded for all specs so
    zeros are visible rather than absent.
    """
    frame = pd.DataFrame([f["diagnostics"] for f in fits.values()])
    summary = {
        "specification": spec.name,
        "n_states": spec.n_states,
        "n_origins": len(fits),
        "n_failed_initializations": int(sum(f["n_failed_inits"] for f in fits.values())),
        "n_absorbing_transition": int(frame["absorbing_state"].sum()),
        "n_degenerate_occupancy": int(frame["degenerate_occupancy"].sum()),
        "n_singular_covariance": int(frame["singular_covariance"].sum()),
        "n_guard_triggered": int(frame["guard_triggered"].sum()),
        "min_occupancy_any_state": float(frame["min_occupancy"].min()),
    }
    for k in range(spec.n_states):
        summary[f"mean_occupancy_s{k}"] = float(frame[f"occupancy_s{k}"].mean())
        summary[f"min_n_eff_s{k}"] = float(frame[f"n_eff_s{k}"].min())
        summary[f"mean_realized_vol_s{k}"] = float(frame[f"mean_realized_vol_s{k}"].mean())
        summary[f"mean_persistence_s{k}"] = float(frame[f"persistence_s{k}"].mean())
    # Canonical labeling must order states by ascending realized vol.
    ordered = [summary[f"mean_realized_vol_s{k}"] for k in range(spec.n_states)]
    summary["canonical_ordering_holds"] = bool(
        all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1))
    )
    summary["state_vol_ordering"] = " < ".join(f"{v:.4f}" for v in ordered)
    return summary


def evaluate_spec(
    spec: Spec,
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    origins: pd.DatetimeIndex,
    hmm_cache: dict,
    config: dict,
    plan: dict,
    boot_cfg: st.BootstrapConfig,
) -> tuple[dict, dict]:
    """Run one specification end to end; return (summary, hmm_summary)."""
    assets = list(spec.universe)
    consumed, rolling_lw, events, fits = covariances_for_spec(
        spec, prices, macro, origins, hmm_cache, config, plan
    )

    opt_cfg = opt.OptimizerConfig(max_weight=spec.max_weight)
    daily = prices[assets].pct_change().iloc[1:]
    execution_map = bt.execution_dates_for(origins, daily.index)

    targets = {"regime": {}, "rolling": {}}
    optimizer_failures = 0
    for signal, execution in execution_map.items():
        for key, matrix in (("regime", consumed[signal]), ("rolling", rolling_lw[signal])):
            weights, diagnostics = opt.min_variance_weights(matrix, opt_cfg)
            optimizer_failures += int(diagnostics["fallback_requested"])
            targets[key][execution] = weights

    cost_bps = float(config["backtest"]["main_cost_bps"])
    paths = {k: bt.simulate_path(daily, v) for k, v in targets.items()}
    costed = {k: bt.apply_costs(v["events"], cost_bps) for k, v in paths.items()}

    dates = costed["regime"].index
    risk_free = mt.risk_free_daily(dates, macro["DFF"], entry_date=dates[0])
    net_a, net_b = costed["regime"]["net_return"], costed["rolling"]["net_return"]
    excess_a = net_a - risk_free.reindex(net_a.index).fillna(0.0)
    excess_b = net_b - risk_free.reindex(net_b.index).fillna(0.0)

    draws = st.paired_bootstrap_differences(
        excess_a, excess_b, net_a, net_b, ("sharpe",), boot_cfg
    )["sharpe"]
    low, high = st.percentile_interval(draws, boot_cfg.confidence)
    sharpe_difference = mt.sharpe_ratio(net_a, risk_free) - mt.sharpe_ratio(net_b, risk_free)

    years = len(dates) / 252
    half_turnover = float(paths["regime"]["events"]["half_turnover_reporting"].sum() / years)

    half_turnover_rolling = float(
        paths["rolling"]["events"]["half_turnover_reporting"].sum() / years
    )
    # Decomposition inputs for interpreting any specification whose
    # difference departs from the primary: BOTH legs are reported, so a
    # change can be attributed to the regime strategy, to the
    # comparator, or to concentration, rather than assumed to reflect
    # better regime identification.
    weights_regime = paths["regime"]["posttrade_weights"]
    weights_rolling = paths["rolling"]["posttrade_weights"]
    p_value = st.centered_bootstrap_pvalue(draws, sharpe_difference, "greater")

    return {
        "_hmm_summary": hmm_diagnostics_summary(fits, spec),
        "specification": spec.name,
        "factor_varied": spec.factor,
        "n_assets": len(assets),
        "max_weight": spec.max_weight,
        "n_evaluation_days": len(dates),
        "eval_start": str(dates[0].date()),
        "eval_end": str(dates[-1].date()),
        "sharpe_difference": sharpe_difference,
        "ci95_lower": low,
        "ci95_upper": high,
        "interval_contains_zero": bool(low <= 0 <= high),
        "p_one_sided_centered": p_value,
        "sign_matches_primary": None,     # filled in after the primary runs
        "vol_difference": mt.annualized_volatility(net_a) - mt.annualized_volatility(net_b),
        "vol_regime": mt.annualized_volatility(net_a),
        "vol_rolling": mt.annualized_volatility(net_b),
        "cagr_difference": mt.cagr(mt.wealth_path(net_a)) - mt.cagr(mt.wealth_path(net_b)),
        "maxdd_difference": (mt.max_drawdown(mt.wealth_path(net_a))
                             - mt.max_drawdown(mt.wealth_path(net_b))),
        "half_turnover_pct": half_turnover * 100,
        "full_traded_notional_pct": half_turnover * 200,
        "half_turnover_rolling_pct": half_turnover_rolling * 100,
        "ann_cost_expenditure_bps": float(
            costed["regime"]["cost_fraction"].sum() / years * 1e4
        ),
        "mean_ief_weight_regime": float(weights_regime["IEF"].mean())
        if "IEF" in weights_regime else np.nan,
        "mean_ief_weight_rolling": float(weights_rolling["IEF"].mean())
        if "IEF" in weights_rolling else np.nan,
        "mean_max_weight_regime": float(weights_regime.max(axis=1).mean()),
        "effective_n_assets_regime": float(
            (1.0 / (weights_regime**2).sum(axis=1)).mean()
        ),
        "effective_n_assets_rolling": float(
            (1.0 / (weights_rolling**2).sum(axis=1)).mean()
        ),
        "n_hmm_guard_events": events["guard_triggered"],
        "n_hmm_failed_initializations": events["failed_inits"],
        "n_a2_fallbacks": events["fallbacks"],
        "n_psd_corrections": events["psd_corrections"],
        "n_optimizer_failures": optimizer_failures,
    }


# ---------------------------------------------------------------------------
# Subperiods (descriptive slices of the primary paths)
# ---------------------------------------------------------------------------

def subperiod_table(plan: dict, macro: pd.DataFrame) -> pd.DataFrame:
    """Descriptive subperiod slices of the existing primary paths."""
    net = pd.read_parquet(PROJECT_ROOT / "outputs" / "backtests" / "net_returns_10bps.parquet")
    net["date"] = pd.to_datetime(net["date"])
    wide = net.pivot(index="date", columns="strategy", values="net_return")
    risk_free = mt.risk_free_daily(wide.index, macro["DFF"], entry_date=wide.index[0])

    rows = []
    periods = {"full_oos": (str(wide.index[0].date()), str(wide.index[-1].date()))}
    periods.update({k: tuple(v) for k, v in plan["robustness_grid"]["subperiods"].items()})
    for name, (start, end) in periods.items():
        a = wide.loc[start:end, "regime_minvar"]
        b = wide.loc[start:end, "rolling_lw_minvar"]
        if len(a) < 60:
            continue
        rf = risk_free.loc[a.index]
        rows.append({
            "subperiod": name, "start": str(a.index[0].date()), "end": str(a.index[-1].date()),
            "n_days": len(a),
            "sharpe_difference": mt.sharpe_ratio(a, rf) - mt.sharpe_ratio(b, rf),
            "vol_difference": mt.annualized_volatility(a) - mt.annualized_volatility(b),
            "cagr_difference": mt.cagr(mt.wealth_path(a)) - mt.cagr(mt.wealth_path(b)),
            "status": "DESCRIPTIVE ONLY - small sample, no inference",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="fewer bootstrap replications, for smoke testing")
    parser.add_argument("--force", action="store_true",
                        help="recompute specifications that already have checkpoints")
    parser.add_argument(
        "--allow-unfrozen", action="store_true",
        help="proceed when data does not match the frozen snapshot; results "
             "will NOT reproduce the paper (live-rebuild mode)",
    )
    args = parser.parse_args()

    config = data_loader.load_config()
    with open(PROJECT_ROOT / "config" / "analysis_plan.yaml", encoding="utf-8") as fh:
        plan = yaml.safe_load(fh)
    try:
        eda.verify_snapshot(SNAPSHOT, PROJECT_ROOT)
        print(f"Snapshot verified against {SNAPSHOT.name}.")
    except RuntimeError:
        if not args.allow_unfrozen:
            raise
        print("WARNING: LIVE-REBUILD mode; results will NOT reproduce the paper.")

    processed = PROJECT_ROOT / config["data"]["processed_dir"]
    prices = pd.read_csv(processed / "prices.csv", index_col="Date", parse_dates=True)
    macro = pd.read_csv(processed / "macro.csv", index_col="Date", parse_dates=True)
    origins = rg.rebalance_origins(
        prep.log_returns(prices).index,
        first_signal_after=plan["sample"]["training_end"],
        min_observations=plan["volatility"]["min_observations"],
    )

    boot_cfg = st.BootstrapConfig(
        n_replications=1000 if args.quick else int(plan["inference"]["bootstrap_replications"]),
        seed=int(plan["inference"]["bootstrap_seed"]),
        mean_block=21,
    )

    specs = build_specifications()
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]

    out_dir = PROJECT_ROOT / "outputs" / "robustness"
    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(message: str) -> None:
        """Write progress to disk immediately.

        A long run must leave a durable trail: buffered console output is
        lost if the process is interrupted, and an earlier attempt stalled
        overnight with nothing to show for it.
        """
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")
            fh.flush()

    hmm_cache: dict = {}
    results: list[dict] = []
    hmm_summaries: list[dict] = []
    log(f"START {len(specs)} specifications, "
        f"{boot_cfg.n_replications} bootstrap replications")

    for i, spec in enumerate(specs, 1):
        started = time.time()
        # Resume support: a completed specification is never recomputed.
        checkpoint = partial_dir / f"{spec.name}.json"
        if checkpoint.exists() and not args.force:
            import json

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            hmm_summaries.append(saved.pop("_hmm_summary"))
            results.append(saved)
            log(f"[{i}/{len(specs)}] {spec.name}: loaded from checkpoint")
            continue

        log(f"[{i}/{len(specs)}] {spec.name}  (factor: {spec.factor}) ...")
        result = evaluate_spec(
            spec, prices, macro, origins, hmm_cache, config, plan, boot_cfg
        )
        result["runtime_seconds"] = round(time.time() - started, 1)

        import json

        checkpoint.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        hmm_summaries.append(result.pop("_hmm_summary"))
        results.append(result)
        log(f"[{i}/{len(specs)}] {spec.name}: dSharpe "
            f"{result['sharpe_difference']:+.4f}  "
            f"CI [{result['ci95_lower']:+.4f}, {result['ci95_upper']:+.4f}]  "
            f"contains zero: {result['interval_contains_zero']}  "
            f"({result['runtime_seconds']}s)")

    frame = pd.DataFrame(results)
    if "primary" in set(frame["specification"]):
        primary_sign = np.sign(
            frame.loc[frame["specification"] == "primary", "sharpe_difference"].iloc[0]
        )
        frame["sign_matches_primary"] = np.sign(frame["sharpe_difference"]) == primary_sign

    # Item 1: specifications sharing a date range must share the same
    # paired bootstrap index matrix, so cross-specification differences
    # reflect returns rather than Monte Carlo noise.
    common_dates = frame["n_evaluation_days"].nunique() == 1
    frame["shares_common_evaluation_window"] = common_dates
    if not common_dates:
        print("\nWARNING: specifications do not share an identical evaluation "
              "window; sample differences are labeled in the grid.")

    # Item 2: Holm across the formal robustness Sharpe tests ONLY. The
    # Phase 11 primary test is deliberately excluded from this family.
    secondary = frame["specification"] != "primary"
    frame["p_holm_robustness_family"] = np.nan
    frame.loc[secondary, "p_holm_robustness_family"] = st.holm_adjust(
        frame.loc[secondary, "p_one_sided_centered"].to_numpy()
    )
    frame["family"] = np.where(
        secondary, "robustness (Holm-adjusted)", "PRIMARY (Phase 11, outside family)"
    )

    artifacts = []

    path = out_dir / "robustness_grid.csv"
    frame.to_csv(path, index=False)
    artifacts.append(path)
    print(f"\nWrote {path.relative_to(PROJECT_ROOT)}  ({len(frame)} specifications)")

    hpath = out_dir / "hmm_diagnostics_by_spec.csv"
    pd.DataFrame(hmm_summaries).to_csv(hpath, index=False)
    artifacts.append(hpath)
    print(f"Wrote {hpath.relative_to(PROJECT_ROOT)}")

    subperiods = subperiod_table(plan, macro)
    spath = out_dir / "subperiod_descriptive.csv"
    subperiods.to_csv(spath, index=False)
    artifacts.append(spath)
    print(f"Wrote {spath.relative_to(PROJECT_ROOT)}  ({len(subperiods)} subperiods)")

    from src import visualization as viz
    figure = viz.fig_robustness_forest(
        frame, PROJECT_ROOT / "outputs" / "figures" / "robustness_forest.png"
    )
    artifacts.append(figure)
    print(f"Wrote {figure.relative_to(PROJECT_ROOT)}")

    manifest = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT,
        PROJECT_ROOT / "outputs" / "phase12_manifest.json",
        phase="12-robustness",
        note=(
            "One-at-a-time robustness grid; no Cartesian product. The 30% cap "
            "runs on the five-asset universe only (infeasible for three "
            "assets). Subperiods are descriptive slices of the primary paths. "
            "No robustness result may be promoted into the headline conclusion."
        ),
    )
    print(f"Provenance manifest: {manifest.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
