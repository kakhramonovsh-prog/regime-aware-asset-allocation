"""Run the analysis pipeline. Implemented: Phase 4 (EDA), Phase 5
(volatility forecasting).

Usage::

    python scripts/run_analysis.py [--phase {4,5,all}] [--config ...]

Every phase refuses to run unless the data on disk matches the frozen
SHA-256 snapshot manifest, and writes a provenance manifest tying every
artifact to the git commit, config hash, and data hashes.

Phase 5 estimates volatility models only (through each rebalance date);
no portfolio returns, regime model, or strategy selection is computed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src import data_loader, eda, preprocessing as prep, visualization as viz  # noqa: E402
from src import volatility as vol  # noqa: E402

SNAPSHOT_MANIFEST = PROJECT_ROOT / "data" / "snapshots" / "manifest_2026-08-06.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument(
        "--phase",
        choices=["4", "5", "6", "7", "8", "9", "10", "11", "all"],
        default="all",
    )
    parser.add_argument(
        "--allow-unfrozen", action="store_true",
        help="proceed when data does not match the frozen snapshot; results "
             "will NOT reproduce the paper (live-rebuild mode)",
    )
    args = parser.parse_args()

    config = data_loader.load_config(args.config)
    with open(PROJECT_ROOT / "config" / "analysis_plan.yaml", encoding="utf-8") as fh:
        plan = yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # Integrity gate: only the frozen snapshot may be analyzed.
    # ------------------------------------------------------------------
    try:
        eda.verify_snapshot(SNAPSHOT_MANIFEST, PROJECT_ROOT)
        print(f"Snapshot verified against {SNAPSHOT_MANIFEST.name}: all hashes match.")
    except RuntimeError:
        if not args.allow_unfrozen:
            raise
        print("WARNING: data does not match the frozen snapshot. Running in "
              "LIVE-REBUILD mode; results will NOT reproduce the paper.")

    processed = PROJECT_ROOT / config["data"]["processed_dir"]
    prices = pd.read_csv(processed / "prices.csv", index_col="Date", parse_dates=True)
    macro = pd.read_csv(processed / "macro.csv", index_col="Date", parse_dates=True)

    if args.phase in ("4", "all"):
        run_phase4(config, plan, prices, macro)
    if args.phase in ("5", "all"):
        run_phase5(config, plan, prices)
    if args.phase in ("6", "all"):
        run_phase6(config, plan, prices, macro)
    if args.phase in ("7", "all"):
        run_phase7(config, plan, prices, macro)
    if args.phase in ("8", "all"):
        run_phase8(config, plan, prices, macro)
    if args.phase in ("9", "all"):
        run_phase9(config, plan, prices)
    if args.phase in ("10", "all"):
        run_phase10(config, plan, macro)
    if args.phase in ("11", "all"):
        run_phase11(config, plan, macro)


def run_phase4(config: dict, plan: dict, prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    raw_prices = data_loader.load_raw_prices(raw_dir, list(config["data"]["tickers"]))
    raw_macro = data_loader.load_raw_macro(raw_dir, list(config["data"]["fred_series"]))

    returns = prep.log_returns(prices)
    slope = prep.yield_curve_slope(macro)

    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def save(df: pd.DataFrame, name: str) -> None:
        path = tables_dir / name
        df.to_csv(path, float_format="%.6f")
        artifacts.append(path)
        print(f"  table  {name}")

    print("\nGenerating tables:")
    save(eda.summary_statistics(prices), "summary_statistics.csv")
    save(eda.macro_summary(macro), "macro_summary.csv")
    save(eda.missingness_audit(raw_prices, raw_macro, prices, macro), "missingness_audit.csv")
    save(eda.staleness_audit(prices, macro), "staleness_audit.csv")
    save(eda.calendar_audit(prices.index), "calendar_audit.csv")

    corr, conditioning = eda.correlation_and_conditioning(returns)
    save(corr, "correlation_matrix.csv")
    save(conditioning, "correlation_conditioning.csv")

    feature_corr, vix_rv = eda.macro_feature_correlations(prices, macro)
    save(feature_corr, "macro_feature_correlations.csv")
    save(vix_rv, "vix_vs_realized_vol_stats.csv")

    periods = {"full_sample": (str(prices.index.min().date()), str(prices.index.max().date())),
               "training_window": (plan["sample"]["full_start"], plan["sample"]["training_end"])}
    periods.update({k: tuple(v) for k, v in plan["robustness_grid"]["subperiods"].items()})
    save(eda.subperiod_summaries(prices, macro, periods), "subperiod_summaries.csv")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("Generating figures:")
    rv = eda.realized_vol_series(prices, window=21)
    figures = [
        viz.fig_normalized_prices(prices, figures_dir / "normalized_prices.png"),
        viz.fig_return_distributions(returns, figures_dir / "return_distributions.png"),
        viz.fig_rolling_volatility(returns, figures_dir / "rolling_volatility.png"),
        viz.fig_rolling_correlations(returns, figures_dir / "rolling_correlations.png"),
        viz.fig_drawdowns(prices, figures_dir / "drawdowns.png"),
        viz.fig_macro_features(macro, slope, figures_dir / "macro_features.png"),
        viz.fig_vix_vs_realized(macro["VIXCLS"], rv, figures_dir / "vix_vs_realized_vol.png"),
    ]
    for f in figures:
        artifacts.append(f)
        print(f"  figure {f.name}")

    # ------------------------------------------------------------------
    # Provenance manifest
    # ------------------------------------------------------------------
    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "eda_manifest.json",
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 4 complete. Descriptive analysis only: no volatility model, "
          "HMM, portfolio optimization, or backtest was estimated or examined.")


def run_phase5(config: dict, plan: dict, prices: pd.DataFrame) -> None:
    """Volatility forecasting per the frozen Phase 5 specification."""
    returns = prep.log_returns(prices)
    cfg = vol.VolatilityConfig(
        hist_window=config["volatility_models"]["hist_window_days"],
        ewma_lambda=config["volatility_models"]["ewma_lambda"],
        min_observations=plan["volatility"]["min_observations"],
        variance_floor_daily=float(plan["volatility"]["variance_floor_daily"]),
    )
    n_dates = len(vol.month_end_rebalance_dates(returns.index, cfg.min_observations))
    print(f"\nPhase 5: fitting 3 models x {len(returns.columns)} assets x "
          f"{n_dates} month-end origins (GARCH refits take a few minutes)...")

    forecasts = vol.build_volatility_forecasts(returns, cfg)

    forecasts_dir = PROJECT_ROOT / "outputs" / "forecasts"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    fpath = forecasts_dir / "volatility_forecasts.parquet"
    forecasts.to_parquet(fpath, index=False)
    artifacts.append(fpath)
    print(f"  forecasts {fpath.name}  ({len(forecasts)} rows)")

    losses = vol.loss_table(forecasts, oos_start=plan["sample"]["oos_start"])
    lpath = tables_dir / "volatility_loss_by_asset.csv"
    losses.to_csv(lpath, index=False, float_format="%.6g")
    artifacts.append(lpath)
    print(f"  table  {lpath.name}")

    comparisons = pd.concat(
        [
            vol.diebold_mariano(forecasts, a, b)
            for a, b in [("ewma", "hist63"), ("garch11", "hist63"), ("garch11", "ewma")]
        ],
        ignore_index=True,
    )
    # Holm adjustment across the full family of 15 comparisons
    # (secondary column; unadjusted HAC(3) p-values are unchanged).
    comparisons["p_holm"] = vol.holm_adjust(comparisons["p_value"])
    cpath = tables_dir / "forecast_comparisons.csv"
    comparisons.to_csv(cpath, index=False, float_format="%.6g")
    artifacts.append(cpath)
    print(f"  table  {cpath.name}")

    garch_log = forecasts[forecasts["model"] == "garch11"][
        ["date", "asset", "horizon_days", "converged", "substituted", "floored"]
    ]
    gpath = tables_dir / "volatility_estimation_log.csv"
    garch_log.to_csv(gpath, index=False)
    artifacts.append(gpath)
    n_sub = int(garch_log["substituted"].sum())
    print(f"  table  {gpath.name}  (GARCH substitutions: {n_sub}/{len(garch_log)})")

    fig = viz.fig_forecasts_vs_realized(
        forecasts, "SPY", figures_dir / "forecasts_vs_realized_spy.png"
    )
    artifacts.append(fig)
    print(f"  figure {fig.name}")

    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "phase5_manifest.json",
        phase="5-volatility",
        note=(
            "Volatility forecasting only, estimated through each rebalance "
            "date. No portfolio returns, regime model, or strategy selection "
            "was computed or examined. EWMA remains the portfolio-feeding "
            "model regardless of this comparison (fixed ex ante)."
        ),
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 5 complete. No portfolio returns or strategy selection examined.")


def run_phase6(config: dict, plan: dict, prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    """Expanding-window HMM regime identification (no allocation work)."""
    from src import regimes as rg

    cfg = rg.HMMConfig(
        n_states=config["regimes"]["n_states_main"],
        n_init=config["regimes"]["n_initializations"],
        seed_start=config["regimes"]["init_seeds_start"],
        min_occupancy=config["regimes"]["min_state_occupancy"],
        min_observations=plan["volatility"]["min_observations"],
    )
    features = rg.build_features(
        prices, macro, macro_lag_days=config["data"]["macro_signal_lag_days"]
    )
    returns_index = prep.log_returns(prices).index
    # Primary sample starts at the preregistered first signal origin
    # (December 2009 month-end -> first execution 2010-01-04), NOT at
    # the earliest date satisfying the minimum-observation rule.
    rebalance_dates = rg.rebalance_origins(
        returns_index,
        first_signal_after=plan["sample"]["training_end"],
        min_observations=cfg.min_observations,
    )

    regimes_dir = PROJECT_ROOT / "outputs" / "regimes"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    regimes_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    print(f"\nPhase 6: {cfg.n_states}-state HMM, {len(rebalance_dates)} monthly refits "
          f"x {cfg.n_init} starts (seeds {cfg.seed_start}-{cfg.seed_start + cfg.n_init - 1})")
    print(f"  first signal origin {rebalance_dates[0].date()} "
          f"(preregistered), last {rebalance_dates[-1].date()}")
    out = rg.run_expanding_hmm(features, rebalance_dates, cfg, verbose=True)

    params = rg.implementation_parameters(cfg, features, str(rebalance_dates[0].date()))

    realtime = out["realtime"]
    rt_path = regimes_dir / "realtime_probabilities.parquet"
    realtime.to_parquet(rt_path, index=False)
    artifacts.append(rt_path)
    print(f"  regimes {rt_path.name}  ({len(realtime)} dates, "
          f"{int(realtime['classified_high_vol'].sum())} classified high-vol)")

    for name, frame in (
        ("all_initializations.csv", out["initializations"]),
        ("selected_fit_diagnostics.csv", out["diagnostics"]),
        ("model_revision_stability.csv", out["revisions"]),
        ("implementation_parameters.csv", params),
    ):
        path = regimes_dir / name
        frame.to_csv(path, index=False, float_format="%.6g")
        artifacts.append(path)
        print(f"  regimes {name}")

    tpath = regimes_dir / "transition_matrices.parquet"
    out["transitions"].to_parquet(tpath, index=False)
    artifacts.append(tpath)
    print(f"  regimes {tpath.name}")

    # State characteristics: feature means conditional on the real-time
    # classification (descriptive; uses stored real-time labels only).
    labels = realtime.set_index("date")["classified_high_vol"]
    joined = features.reindex(labels.index).join(labels)
    characteristics = joined.groupby("classified_high_vol").agg(["mean", "std", "count"])
    characteristics.columns = ["_".join(c) for c in characteristics.columns]
    cpath = regimes_dir / "state_characteristics.csv"
    characteristics.to_csv(cpath, float_format="%.6g")
    artifacts.append(cpath)
    print(f"  regimes {cpath.name}")

    # Stability diagnostics, split into two distinct concepts:
    #   (a) parameter dispersion + model revision  -> model stability
    #   (b) state evolution (endpoint prob movement) -> market conditions
    diagnostics = out["diagnostics"]
    revisions = out["revisions"]
    stability_rows = []
    for column in [c for c in diagnostics.columns if c.startswith(
            ("occupancy_s", "persistence_s", "n_eff_s", "expected_duration_s",
             "mean_realized_vol_s"))]:
        # Expected duration is infinite when a state is absorbing
        # (persistence = 1). Summarize finite values and count the rest
        # rather than letting inf swallow the whole row.
        series = diagnostics[column]
        finite = series[np.isfinite(series)]
        stability_rows.append({
            "concept": "parameter dispersion across refits", "quantity": column,
            "mean": finite.mean(), "std": finite.std(ddof=1),
            "min": finite.min(), "max": finite.max(),
            "n_non_finite": int((~np.isfinite(series)).sum()),
        })
    for column in ("mean_abs_prob_revision", "max_abs_prob_revision",
                   "prob_revision_corr", "classification_agreement",
                   "mean_drift_l2", "transmat_drift_l1"):
        series = revisions[column]
        stability_rows.append({
            "concept": "model revision (same overlapping window)", "quantity": column,
            "mean": series.mean(), "std": series.std(ddof=1),
            "min": series.min(), "max": series.max(),
        })
    evolution = realtime["prob_change_vs_prev_month"].dropna().abs()
    stability_rows.append({
        "concept": "state evolution (market conditions, NOT instability)",
        "quantity": "abs_monthly_change_in_endpoint_prob",
        "mean": evolution.mean(), "std": evolution.std(ddof=1),
        "min": evolution.min(), "max": evolution.max(),
    })
    counts = [
        ("n_refits", len(realtime)),
        ("n_classification_changes", int(realtime["classification_changed"].sum())),
        ("n_selected_seed_changes", int(revisions["selected_seed_changed"].sum())),
        ("n_guard_triggered", int(realtime["guard_triggered"].sum())),
        ("n_failed_initializations", int((~out["initializations"]["usable"]).sum())),
        ("n_refits_with_multiple_fits_in_selection_tol",
         int(out["initializations"].groupby("date")["within_selection_tol"].sum().gt(1).sum())),
    ]
    stability = pd.concat([
        pd.DataFrame(stability_rows),
        pd.DataFrame([{"concept": "counts", "quantity": q, "mean": v} for q, v in counts]),
    ], ignore_index=True)
    spath = regimes_dir / "stability_diagnostics.csv"
    stability.to_csv(spath, index=False, float_format="%.6g")
    artifacts.append(spath)
    print(f"  regimes {spath.name}")

    # Distinctiveness ONLY - not evidence of predictive or economic value.
    distinct = rg.distinctiveness_diagnostic(realtime, features)
    tcpath = regimes_dir / "distinctiveness_diagnostic.csv"
    distinct.to_csv(tcpath, index=False, float_format="%.6g")
    artifacts.append(tcpath)
    print(f"  regimes {tcpath.name}")

    # Feature ablations (preregistered robustness): drop VIX, drop RV.
    print("  running feature ablations (drop VIX, drop realized vol)...")
    ablation_rows = []
    for label, drop in (("drop_vix", "log_vix_lag1"),
                        ("drop_realized_vol", "realized_vol_21d")):
        alt = rg.run_ablation(features, drop, rebalance_dates, cfg)
        ablation_rows.append(rg.classification_agreement(realtime, alt, label))
    ablations = pd.DataFrame(ablation_rows)
    apath = regimes_dir / "feature_ablations.csv"
    ablations.to_csv(apath, index=False, float_format="%.6g")
    artifacts.append(apath)
    print(f"  regimes {apath.name}")

    # Ex-post smoothed path (descriptive only, stored separately).
    expost = rg.fit_expost_smoothed(features, cfg)
    epath = regimes_dir / "expost_smoothed_probabilities.parquet"
    expost.to_parquet(epath, index=False)
    artifacts.append(epath)
    print(f"  regimes {epath.name}  (EX-POST, never a trading signal)")

    fig = viz.fig_regime_probabilities(
        realtime, expost, features, figures_dir / "regime_probabilities.png"
    )
    artifacts.append(fig)
    print(f"  figure {fig.name}")

    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "phase6_manifest.json",
        phase="6-regimes",
        note=(
            "HMM regime identification only. Real-time signals are filtered "
            "probabilities from models fit through each rebalance date; "
            "ex-post smoothed probabilities are stored separately and are "
            "descriptive only. No covariance estimation, portfolio "
            "optimization, or strategy performance was computed or examined."
        ),
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 6 complete. No covariance, optimization, or portfolio results examined.")


def run_phase7(config: dict, plan: dict, prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    """Covariance estimation and conditioning diagnostics (no weights)."""
    from src import covariance as cv
    from src import regimes as rg

    hmm_cfg = rg.HMMConfig(
        n_states=config["regimes"]["n_states_main"],
        n_init=config["regimes"]["n_initializations"],
        seed_start=config["regimes"]["init_seeds_start"],
        min_occupancy=config["regimes"]["min_state_occupancy"],
        min_observations=plan["volatility"]["min_observations"],
    )
    cov_cfg = cv.CovarianceConfig(
        ewma_lambda=config["volatility_models"]["ewma_lambda"],
        neff_threshold=float(plan["regime_covariance"]["neff_threshold"]),
        horizon_days=int(plan["regime_covariance"]["horizon_days"]),
        include_between_state_term=False,   # A3: main = within-state only
    )
    features = rg.build_features(
        prices, macro, macro_lag_days=config["data"]["macro_signal_lag_days"]
    )
    returns = prep.log_returns(prices)
    rebalance_dates = rg.rebalance_origins(
        returns.index,
        first_signal_after=plan["sample"]["training_end"],
        min_observations=hmm_cfg.min_observations,
    )

    cov_dir = PROJECT_ROOT / "outputs" / "covariance"
    cov_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    print(f"\nPhase 7: covariance estimation at {len(rebalance_dates)} origins "
          f"({rebalance_dates[0].date()} to {rebalance_dates[-1].date()})")
    audit, consumed = cv.build_covariance_panel(
        returns, features, rebalance_dates, hmm_cfg, cov_cfg, verbose=True
    )

    apath = cov_dir / "covariance_audit.csv"
    audit.to_csv(apath, index=False, float_format="%.8g")
    artifacts.append(apath)
    print(f"  covariance {apath.name}  ({len(audit)} rows)")

    # Matrices exported for Phase 8, long format. Strategies consume
    # these directly so no downstream step re-estimates a covariance on
    # a different window.
    rows = []
    assets = list(prices.columns)
    for date, by_name in consumed.items():
        for name, matrix in by_name.items():
            for i, a in enumerate(assets):
                for j, b in enumerate(assets):
                    rows.append({"date": date, "matrix": name, "asset_i": a,
                                 "asset_j": b, "covariance": float(matrix[i, j])})
    cpath = cov_dir / "consumed_covariances.parquet"
    pd.DataFrame(rows).to_parquet(cpath, index=False)
    artifacts.append(cpath)
    print(f"  covariance {cpath.name}  ({len(consumed)} dates x 3 matrices)")

    # Conditioning summary by estimator.
    summary = audit.groupby("estimator").agg(
        n=("date", "size"),
        mean_cov_condition=("covariance_condition_number", "mean"),
        max_cov_condition=("covariance_condition_number", "max"),
        mean_corr_condition=("correlation_condition_number", "mean"),
        max_corr_condition=("correlation_condition_number", "max"),
        min_eigenvalue_before=("min_eigenvalue_before", "min"),
        n_psd_corrections=("psd_correction_used", "sum"),
        max_psd_correction=("psd_correction_magnitude", "max"),
    ).reset_index()
    spath = cov_dir / "conditioning_summary.csv"
    summary.to_csv(spath, index=False, float_format="%.8g")
    artifacts.append(spath)
    print(f"  covariance {spath.name}")

    fallbacks = audit[audit["fallback_used"]]["date"].drop_duplicates()
    print(f"  A2 fallbacks: {len(fallbacks)} origins "
          f"({', '.join(str(d.date()) for d in fallbacks)})")

    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "phase7_manifest.json",
        phase="7-covariance",
        note=(
            "Covariance estimation and conditioning diagnostics only. "
            "All estimators use returns through the rebalance origin; "
            "state-conditioned estimates use smoothed responsibilities "
            "from the HMM fit through that origin. No optimizer weights "
            "or portfolio returns were computed or examined."
        ),
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 7 complete. No weights or portfolio returns examined.")


def run_phase8(config: dict, plan: dict, prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    """Target-weight construction for the six-strategy ladder.

    Emits target weights only: no return, cost, turnover, or wealth is
    computed here. Failed optimizations emit no weights and set
    fallback_requested, which Phase 9 interprets as "no trade".
    """
    from src import covariance as cv
    from src import optimization as opt
    from src import regimes as rg

    returns = prep.log_returns(prices)
    assets = list(prices.columns)
    rebalance_dates = rg.rebalance_origins(
        returns.index,
        first_signal_after=plan["sample"]["training_end"],
        min_observations=plan["volatility"]["min_observations"],
    )

    # All covariance matrices come from the Phase 7 artifact, estimated
    # on one window. Phase 8 never re-estimates a covariance: doing so on
    # a different window would make the A2 fallback subtly different from
    # the rolling-LW comparator it is defined to equal.
    consumed_path = PROJECT_ROOT / "outputs" / "covariance" / "consumed_covariances.parquet"
    if not consumed_path.exists():
        raise SystemExit("Run Phase 7 first: consumed_covariances.parquet is missing.")
    long = pd.read_parquet(consumed_path)

    def _matrices(name: str) -> dict[pd.Timestamp, np.ndarray]:
        subset = long[long["matrix"] == name]
        out: dict[pd.Timestamp, np.ndarray] = {}
        for date, group in subset.groupby("date"):
            pivot = group.pivot(index="asset_i", columns="asset_j", values="covariance")
            out[pd.Timestamp(date)] = pivot.loc[assets, assets].to_numpy()
        return out

    rolling_lw = _matrices("ledoit_wolf")
    consumed = _matrices("consumed")

    # Static target: the Ledoit-Wolf matrix at the first origin, i.e.
    # estimated through the preregistered training end and then frozen.
    training_origin = rebalance_dates[0]
    static_cov = rolling_lw[training_origin]

    # EWMA volatility forecasts from the Phase 5 model (lambda fixed ex
    # ante). The recursion is seeded at the start of the return series;
    # by the first origin the seed's weight is lambda^~1270 (~1e-34), so
    # the window-start difference is numerically irrelevant.
    lam = config["volatility_models"]["ewma_lambda"]
    ewma_series = {
        a: vol.ewma_variance_series(returns[a], lam=lam) for a in assets
    }
    ewma_variances = {
        t: np.array([float(ewma_series[a].loc[t]) for a in assets])
        for t in rebalance_dates
    }

    cfg = opt.OptimizerConfig(max_weight=config["portfolio"]["max_weight"])
    print(f"\nPhase 8: target weights for {len(opt.STRATEGIES)} strategies "
          f"x {len(rebalance_dates)} origins (cap {cfg.max_weight:.0%}, "
          "60/40 benchmark cap-exempt)")

    weights, audit = opt.build_strategy_targets(
        assets, rebalance_dates, static_cov, rolling_lw, ewma_variances,
        consumed, config["benchmark"]["weights"], cfg,
    )

    strat_dir = PROJECT_ROOT / "outputs" / "strategies"
    strat_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    wpath = strat_dir / "target_weights.parquet"
    weights.to_parquet(wpath, index=False)
    artifacts.append(wpath)
    print(f"  strategies {wpath.name}  ({len(weights)} weight rows)")

    apath = strat_dir / "optimizer_audit.csv"
    audit.to_csv(apath, index=False, float_format="%.8g")
    artifacts.append(apath)
    n_fallback = int(audit["fallback_requested"].sum())
    print(f"  strategies {apath.name}  ({len(audit)} rows, "
          f"{n_fallback} fallback requests)")

    summary = audit[audit["strategy"].isin(opt.OPTIMIZED)].groupby("strategy").agg(
        n=("date", "size"),
        solver_success_rate=("solver_success", "mean"),
        mean_iterations=("iterations", "mean"),
        max_constraint_violation=("constraint_violation", "max"),
        mean_cap_binding=("cap_binding_count", "mean"),
        fallback_requests=("fallback_requested", "sum"),
    ).reset_index()
    spath = strat_dir / "optimizer_summary.csv"
    summary.to_csv(spath, index=False, float_format="%.8g")
    artifacts.append(spath)
    print(f"  strategies {spath.name}")

    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "phase8_manifest.json",
        phase="8-strategies",
        note=(
            "Target weights only. No portfolio return, transaction cost, "
            "turnover, Sharpe ratio, or cumulative wealth was computed or "
            "examined. Failed optimizations emit no weights and set "
            "fallback_requested for Phase 9 to treat as no trade."
        ),
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 8 complete. Targets only: no returns, costs, or performance examined.")


def run_phase9(config: dict, plan: dict, prices: pd.DataFrame) -> None:
    """Walk-forward accounting: drift, trades, costs, gross and net returns.

    Amendment A4 conventions: entry from 100% cash, daily drift, cost on
    the full trade sum, multiplicative net-return identity. No Sharpe,
    drawdown, ranking, or selection is computed here.
    """
    from src import backtest as bt
    from src import optimization as opt

    # A4(d): accounting uses SIMPLE returns.
    daily_returns = prices.pct_change().iloc[1:]
    assets = list(prices.columns)

    weights_path = PROJECT_ROOT / "outputs" / "strategies" / "target_weights.parquet"
    audit_path = PROJECT_ROOT / "outputs" / "strategies" / "optimizer_audit.csv"
    if not weights_path.exists():
        raise SystemExit("Run Phase 8 first: target_weights.parquet is missing.")
    targets_long = pd.read_parquet(weights_path)
    targets_long["date"] = pd.to_datetime(targets_long["date"])
    optimizer_audit = pd.read_csv(audit_path, parse_dates=["date"])

    signal_dates = pd.DatetimeIndex(sorted(targets_long["date"].unique()))
    execution_map = bt.execution_dates_for(signal_dates, daily_returns.index)
    print(f"\nPhase 9: accounting for {len(opt.STRATEGIES)} strategies, "
          f"{len(execution_map)} executions")
    print(f"  first signal {signal_dates[0].date()} -> first execution "
          f"{execution_map[signal_dates[0]].date()} (entry from 100% cash)")

    targets_by_strategy: dict[str, dict[pd.Timestamp, np.ndarray | None]] = {}
    for strategy in opt.STRATEGIES:
        subset = targets_long[targets_long["strategy"] == strategy]
        wide = subset.pivot(index="date", columns="asset", values="weight")
        failures = set(
            optimizer_audit.loc[
                (optimizer_audit["strategy"] == strategy)
                & optimizer_audit["fallback_requested"], "date"
            ]
        )
        mapping: dict[pd.Timestamp, np.ndarray | None] = {}
        for signal, execution in execution_map.items():
            if signal in failures or signal not in wide.index:
                mapping[execution] = None          # no trade (A4 / Phase 8 policy)
            else:
                mapping[execution] = wide.loc[signal, assets].to_numpy()
        targets_by_strategy[strategy] = mapping

    cfg = bt.BacktestConfig(
        cost_bps_scenarios=tuple(float(b) for b in config["backtest"]["transaction_cost_bps"]),
        main_cost_bps=float(config["backtest"]["main_cost_bps"]),
    )
    results = bt.run_backtest(
        daily_returns, targets_by_strategy, cfg,
        signal_dates={v: k for k, v in {s: e for s, e in execution_map.items()}.items()},
    )

    out_dir = PROJECT_ROOT / "outputs" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    file_map = {
        "pretrade_weights": "pretrade_weights.parquet",
        "posttrade_weights": "posttrade_weights.parquet",
        "trades": "trades.parquet",
        "turnover": "turnover.parquet",
        "gross_returns": "gross_returns.parquet",
        "audit": "accounting_audit.parquet",
    }
    for key, filename in file_map.items():
        path = out_dir / filename
        results[key].to_parquet(path, index=False)
        artifacts.append(path)
        print(f"  backtests {filename}  ({len(results[key])} rows)")

    costs_only = results["audit"][
        ["date", "strategy", "cost_bps", "cost_fraction", "full_trade_sum"]
    ]
    cpath = out_dir / "transaction_costs.parquet"
    costs_only.to_parquet(cpath, index=False)
    artifacts.append(cpath)
    print(f"  backtests {cpath.name}  ({len(costs_only)} rows)")

    for bps in cfg.cost_bps_scenarios:
        key = f"net_returns_{int(bps)}bps"
        path = out_dir / f"{key}.parquet"
        results[key].to_parquet(path, index=False)
        artifacts.append(path)
        print(f"  backtests {path.name}")

    max_identity_error = float(results["audit"]["wealth_identity_error"].max())
    print(f"  max wealth identity error: {max_identity_error:.3e}")

    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "phase9_manifest.json",
        phase="9-accounting",
        note=(
            "Portfolio accounting only: drifted holdings, trades, turnover, "
            "costs, gross and net returns. No Sharpe ratio, drawdown, "
            "cumulative-return ranking, or strategy selection was computed "
            "or examined."
        ),
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 9 complete. Accounting only: no performance statistics examined.")


def run_phase10(config: dict, plan: dict, macro: pd.DataFrame) -> None:
    """Performance metrics from the STORED Phase 9 return series.

    Never re-runs the backtest. Descriptive only: differences are point
    estimates and nothing is described as statistically significant,
    which is Phase 11's question.
    """
    from src import metrics as mt
    from src import optimization as opt
    from src import visualization as viz

    bt_dir = PROJECT_ROOT / "outputs" / "backtests"
    if not (bt_dir / "gross_returns.parquet").exists():
        raise SystemExit("Run Phase 9 first: backtest outputs are missing.")

    gross = pd.read_parquet(bt_dir / "gross_returns.parquet")
    gross["date"] = pd.to_datetime(gross["date"])
    turnover = pd.read_parquet(bt_dir / "turnover.parquet")
    turnover["date"] = pd.to_datetime(turnover["date"])
    costs = pd.read_parquet(bt_dir / "transaction_costs.parquet")
    costs["date"] = pd.to_datetime(costs["date"])

    scenarios = [int(b) for b in config["backtest"]["transaction_cost_bps"]]
    main_bps = int(config["backtest"]["main_cost_bps"])
    net_frames: dict[int, pd.DataFrame] = {}
    for bps in scenarios:
        frame = pd.read_parquet(bt_dir / f"net_returns_{bps}bps.parquet")
        frame["date"] = pd.to_datetime(frame["date"])
        net_frames[bps] = frame

    dates = pd.DatetimeIndex(sorted(gross["date"].unique()))
    risk_free = mt.risk_free_daily(dates, macro["DFF"], entry_date=dates[0])
    print(f"\nPhase 10: metrics from stored returns, {len(dates)} days, "
          f"{len(opt.STRATEGIES)} strategies, {len(scenarios)} cost scenarios")

    rows: list[dict] = []
    for bps in scenarios:
        wide = net_frames[bps].pivot(index="date", columns="strategy", values="net_return")
        for strategy in opt.STRATEGIES:
            half_turnover = float(
                turnover.loc[turnover["strategy"] == strategy, "half_turnover_reporting"].sum()
                / (len(dates) / 252)
            )
            expenditure = float(
                costs.loc[(costs["strategy"] == strategy) & (costs["cost_bps"] == bps),
                          "cost_fraction"].sum() / (len(dates) / 252) * 1e4
            )
            summary = mt.performance_summary(
                wide[strategy], risk_free, half_turnover, expenditure,
                label=strategy,
            )
            summary.update(series="net", cost_bps=bps, strategy=strategy)
            rows.append(summary)

    gross_wide = gross.pivot(index="date", columns="strategy", values="gross_return")
    for strategy in opt.STRATEGIES:
        summary = mt.performance_summary(gross_wide[strategy], risk_free, label=strategy)
        summary.update(series="gross", cost_bps=np.nan, strategy=strategy)
        rows.append(summary)

    performance = pd.DataFrame(rows)

    # ---------------- safeguards, checked not assumed ----------------
    checks: list[dict] = []

    gross_rows = performance[performance["series"] == "gross"].set_index("strategy")
    zero_rows = performance[
        (performance["series"] == "net") & (performance["cost_bps"] == 0)
    ].set_index("strategy")
    max_gap = max(
        abs(gross_rows.loc[s, m] - zero_rows.loc[s, m])
        for s in opt.STRATEGIES for m in ("cagr", "sharpe", "max_drawdown")
    )
    checks.append({"check": "zero_cost_net_equals_gross", "value": max_gap,
                   "passed": bool(max_gap < 1e-12)})

    monotone = True
    for strategy in opt.STRATEGIES:
        subset = performance[
            (performance["series"] == "net") & (performance["strategy"] == strategy)
        ].sort_values("cost_bps")
        wealth_values = subset["terminal_wealth"].to_numpy()
        monotone &= bool(np.all(np.diff(wealth_values) <= 1e-12))
    checks.append({"check": "higher_costs_weakly_lower_wealth", "value": float(monotone),
                   "passed": monotone})

    main = performance[
        (performance["series"] == "net") & (performance["cost_bps"] == main_bps)
    ]
    reconcile = 0.0
    main_wide = net_frames[main_bps].pivot(index="date", columns="strategy", values="net_return")
    for strategy in opt.STRATEGIES:
        wealth = mt.wealth_path(main_wide[strategy])
        elapsed = (wealth.index[-1] - wealth.index[0]).days
        implied = (1 + float(main.loc[main["strategy"] == strategy, "cagr"].iloc[0])) ** (
            elapsed / 365.25
        )
        reconcile = max(reconcile, abs(implied - wealth.iloc[-1]))
    checks.append({"check": "cagr_reconciles_with_wealth", "value": reconcile,
                   "passed": bool(reconcile < 1e-8)})

    same_dates = all(
        net_frames[bps].groupby("strategy")["date"].apply(
            lambda s: tuple(sorted(s))
        ).nunique() == 1
        for bps in scenarios
    )
    checks.append({"check": "all_strategies_same_dates", "value": float(same_dates),
                   "passed": same_dates})

    check_frame = pd.DataFrame(checks)
    if not check_frame["passed"].all():
        raise RuntimeError(f"Phase 10 safeguards failed:\n{check_frame}")
    print("  safeguards: all passed "
          f"(max zero-cost gap {max_gap:.2e}, CAGR reconciliation {reconcile:.2e})")

    # ---------------- outputs ----------------
    perf_dir = PROJECT_ROOT / "outputs" / "performance"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    perf_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    ppath = perf_dir / "performance_summary.csv"
    performance.to_csv(ppath, index=False)      # full precision, no rounding
    artifacts.append(ppath)
    print(f"  performance {ppath.name}  ({len(performance)} rows)")

    cpath = perf_dir / "safeguard_checks.csv"
    check_frame.to_csv(cpath, index=False)
    artifacts.append(cpath)

    wealth_main = pd.DataFrame(
        {s: mt.wealth_path(main_wide[s]) for s in opt.STRATEGIES}
    )
    drawdowns = pd.DataFrame(
        {s: mt.drawdown_path(wealth_main[s]) for s in opt.STRATEGIES}
    )
    rolling = pd.DataFrame(
        {s: mt.rolling_sharpe(main_wide[s], risk_free) for s in opt.STRATEGIES}
    )
    for frame, name in ((wealth_main, "wealth_paths"), (drawdowns, "drawdown_paths"),
                        (rolling, "rolling_sharpe")):
        path = perf_dir / f"{name}_{main_bps}bps.parquet"
        frame.reset_index().to_parquet(path, index=False)
        artifacts.append(path)

    print("  figures:")
    figures = [
        viz.fig_cumulative_growth(wealth_main, figures_dir / "cumulative_growth.png", main_bps),
        viz.fig_strategy_drawdowns(drawdowns, figures_dir / "strategy_drawdowns.png", main_bps),
        viz.fig_rolling_sharpe(rolling, figures_dir / "rolling_sharpe.png"),
    ]
    posttrade = pd.read_parquet(bt_dir / "posttrade_weights.parquet")
    posttrade["date"] = pd.to_datetime(posttrade["date"])
    for strategy in ("regime_minvar", "rolling_lw_minvar"):
        subset = posttrade[posttrade["strategy"] == strategy].set_index("date")
        subset = subset.drop(columns=["strategy"])
        figures.append(viz.fig_weights_through_time(
            subset, figures_dir / f"weights_{strategy}.png",
            viz.STRATEGY_LABELS[strategy],
        ))
    rolling_turnover = pd.DataFrame({
        s: turnover[turnover["strategy"] == s].set_index("date")[
            "half_turnover_reporting"].rolling(252).sum()
        for s in opt.STRATEGIES
    })
    figures.append(viz.fig_turnover(rolling_turnover, figures_dir / "turnover.png"))
    for f in figures:
        artifacts.append(f)
        print(f"    {f.name}")

    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "phase10_manifest.json",
        phase="10-performance",
        note=(
            "Descriptive performance metrics computed from the stored Phase 9 "
            "return series; the backtest was not re-run. Differences between "
            "strategies are point estimates only. No statistical significance "
            "is claimed or tested here; that is Phase 11."
        ),
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 10 complete. Descriptive only: no significance tested.")


def run_phase11(config: dict, plan: dict, macro: pd.DataFrame) -> None:
    """Inference on the preregistered primary comparison.

    Bootstraps the SAME estimand Phase 10 reported: paired daily excess
    returns annualized by sqrt(252). Only regime-aware vs rolling
    Ledoit-Wolf at 10 bps is confirmatory.
    """
    from src import metrics as mt
    from src import statistical_tests as st

    bt_dir = PROJECT_ROOT / "outputs" / "backtests"
    main_bps = int(config["backtest"]["main_cost_bps"])
    scenarios = [int(b) for b in config["backtest"]["transaction_cost_bps"]]

    net_frames: dict[int, pd.DataFrame] = {}
    for bps in scenarios:
        frame = pd.read_parquet(bt_dir / f"net_returns_{bps}bps.parquet")
        frame["date"] = pd.to_datetime(frame["date"])
        net_frames[bps] = frame.pivot(index="date", columns="strategy", values="net_return")

    dates = net_frames[main_bps].index
    risk_free = mt.risk_free_daily(dates, macro["DFF"], entry_date=dates[0])

    PRIMARY, COMPARATOR = "regime_minvar", "rolling_lw_minvar"
    cfg = st.BootstrapConfig(
        n_replications=int(plan["inference"]["bootstrap_replications"]),
        seed=int(plan["inference"]["bootstrap_seed"]),
        mean_block=21,
    )
    print(f"\nPhase 11: inference on {PRIMARY} vs {COMPARATOR} at {main_bps} bps")
    print(f"  paired stationary bootstrap, {cfg.n_replications} reps, seed "
          f"{cfg.seed}, mean block {cfg.mean_block} days, daily excess returns")

    def series(bps: int, strategy: str) -> tuple[pd.Series, pd.Series]:
        net = net_frames[bps][strategy]
        return net, net - risk_free.reindex(net.index).fillna(0.0)

    net_a, excess_a = series(main_bps, PRIMARY)
    net_b, excess_b = series(main_bps, COMPARATOR)

    observed = {
        "sharpe": mt.sharpe_ratio(net_a, risk_free) - mt.sharpe_ratio(net_b, risk_free),
        "sortino": mt.sortino_ratio(net_a, risk_free) - mt.sortino_ratio(net_b, risk_free),
        "ann_volatility": mt.annualized_volatility(net_a) - mt.annualized_volatility(net_b),
        "cagr": mt.cagr(mt.wealth_path(net_a)) - mt.cagr(mt.wealth_path(net_b)),
        "max_drawdown": mt.max_drawdown(mt.wealth_path(net_a)) - mt.max_drawdown(mt.wealth_path(net_b)),
        "calmar": (mt.calmar_ratio(mt.cagr(mt.wealth_path(net_a)), mt.max_drawdown(mt.wealth_path(net_a)))
                   - mt.calmar_ratio(mt.cagr(mt.wealth_path(net_b)), mt.max_drawdown(mt.wealth_path(net_b)))),
    }

    metrics = tuple(observed)
    draws = st.paired_bootstrap_differences(
        excess_a, excess_b, net_a, net_b, metrics, cfg
    )

    inference_dir = PROJECT_ROOT / "outputs" / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    draws_frame = pd.DataFrame({f"d_{m}": draws[m] for m in metrics})
    dpath = inference_dir / "primary_bootstrap_draws.parquet"
    draws_frame.to_parquet(dpath, index=False)
    artifacts.append(dpath)
    print(f"  inference {dpath.name}  ({len(draws_frame)} draws x {len(metrics)} metrics)")

    low, high = st.percentile_interval(draws["sharpe"], cfg.confidence)
    p_one_sided = st.centered_bootstrap_pvalue(
        draws["sharpe"], observed["sharpe"], "greater"
    )
    primary = pd.DataFrame([{
        "comparison": f"{PRIMARY} minus {COMPARATOR}",
        "cost_bps": main_bps,
        "estimand": "annualized Sharpe on daily excess returns (sqrt 252)",
        "observed_difference": observed["sharpe"],
        "ci95_lower": low,
        "ci95_upper": high,
        "interval_excludes_zero": bool(low > 0 or high < 0),
        "bootstrap_mean": float(np.mean(draws["sharpe"])),
        "bootstrap_sd": float(np.std(draws["sharpe"], ddof=1)),
        "p_one_sided_centered_null": p_one_sided,
        "n_replications": cfg.n_replications,
        "mean_block_days": cfg.mean_block,
        "seed": cfg.seed,
        "n_daily_observations": len(net_a),
    }])
    ppath = inference_dir / "primary_inference_summary.csv"
    primary.to_csv(ppath, index=False)
    artifacts.append(ppath)
    print(f"  PRIMARY: dSharpe {observed['sharpe']:+.4f}, 95% CI "
          f"[{low:+.4f}, {high:+.4f}], excludes zero: {bool(low > 0 or high < 0)}")

    # Block-length sensitivity (never replaces the 21-day primary).
    rows = []
    for block in (cfg.mean_block, *cfg.sensitivity_blocks):
        block_draws = st.paired_bootstrap_differences(
            excess_a, excess_b, net_a, net_b, ("sharpe",), cfg, mean_block=block
        )["sharpe"]
        bl, bh = st.percentile_interval(block_draws, cfg.confidence)
        rows.append({
            "mean_block_days": block,
            "role": "PRIMARY" if block == cfg.mean_block else "sensitivity",
            "observed_difference": observed["sharpe"],
            "ci95_lower": bl, "ci95_upper": bh,
            "interval_excludes_zero": bool(bl > 0 or bh < 0),
            "bootstrap_sd": float(np.std(block_draws, ddof=1)),
        })
    sensitivity = pd.DataFrame(rows)
    spath = inference_dir / "block_length_sensitivity.csv"
    sensitivity.to_csv(spath, index=False)
    artifacts.append(spath)
    print(f"  inference {spath.name}")

    # HAC on paired daily net return differences.
    hac_rows = []
    for lags in (21, 5, 42):
        result = st.hac_mean_difference(net_a, net_b, lags=lags)
        result["role"] = "PRIMARY" if lags == 21 else "sensitivity"
        hac_rows.append(result)
    hac = pd.DataFrame(hac_rows)
    hpath = inference_dir / "hac_mean_difference.csv"
    hac.to_csv(hpath, index=False)
    artifacts.append(hpath)
    print(f"  inference {hpath.name}")

    # Secondary metric intervals.
    secondary_rows = []
    for metric in metrics:
        ml, mh = st.percentile_interval(draws[metric], cfg.confidence)
        secondary_rows.append({
            "metric": metric,
            "observed_difference": observed[metric],
            "ci95_lower": ml, "ci95_upper": mh,
            "interval_excludes_zero": bool(ml > 0 or mh < 0),
            "path_dependent": metric in st.PATH_DEPENDENT,
            "role": "CONFIRMATORY" if metric == "sharpe" else "secondary",
        })
    secondary = pd.DataFrame(secondary_rows)
    spath2 = inference_dir / "secondary_metric_intervals.csv"
    secondary.to_csv(spath2, index=False)
    artifacts.append(spath2)
    print(f"  inference {spath2.name}")

    # Secondary comparison family with Holm adjustment.
    family_rows = []
    other_strategies = ["ewma_scaled_minvar", "static_minvar", "equal_weight", "static_6040"]
    for strategy in other_strategies:
        net_x, excess_x = series(main_bps, strategy)
        d = st.paired_bootstrap_differences(
            excess_x, excess_b, net_x, net_b, ("sharpe",), cfg
        )["sharpe"]
        obs = mt.sharpe_ratio(net_x, risk_free) - mt.sharpe_ratio(net_b, risk_free)
        cl, ch = st.percentile_interval(d, cfg.confidence)
        family_rows.append({
            "comparison": f"{strategy} minus {COMPARATOR}", "cost_bps": main_bps,
            "observed_difference": obs, "ci95_lower": cl, "ci95_upper": ch,
            "p_one_sided_centered_null": st.centered_bootstrap_pvalue(d, obs, "greater"),
        })
    for bps in [b for b in scenarios if b != main_bps]:
        na, ea = series(bps, PRIMARY)
        nb, eb = series(bps, COMPARATOR)
        d = st.paired_bootstrap_differences(ea, eb, na, nb, ("sharpe",), cfg)["sharpe"]
        obs = mt.sharpe_ratio(na, risk_free) - mt.sharpe_ratio(nb, risk_free)
        cl, ch = st.percentile_interval(d, cfg.confidence)
        family_rows.append({
            "comparison": f"{PRIMARY} minus {COMPARATOR}", "cost_bps": bps,
            "observed_difference": obs, "ci95_lower": cl, "ci95_upper": ch,
            "p_one_sided_centered_null": st.centered_bootstrap_pvalue(d, obs, "greater"),
        })
    family = pd.DataFrame(family_rows)
    family["p_holm"] = st.holm_adjust(family["p_one_sided_centered_null"])
    family["role"] = "secondary / robustness"
    family["note"] = "not confirmatory; the preregistered primary is unaffected"
    mpath = inference_dir / "multiple_comparison_adjustments.csv"
    family.to_csv(mpath, index=False)
    artifacts.append(mpath)
    print(f"  inference {mpath.name}  ({len(family)} secondary comparisons)")

    manifest_path = eda.write_phase_manifest(
        artifacts, PROJECT_ROOT, SNAPSHOT_MANIFEST,
        PROJECT_ROOT / "outputs" / "phase11_manifest.json",
        phase="11-inference",
        note=(
            "Paired stationary-bootstrap inference on daily excess returns, "
            "matching the Phase 10 estimand. Only regime-aware vs rolling "
            "Ledoit-Wolf at 10 bps is confirmatory; all other comparisons are "
            "secondary or robustness and carry Holm-adjusted p-values. "
            "Diebold-Mariano is deliberately not applied to portfolio returns."
        ),
    )
    print(f"\nProvenance manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("Phase 11 complete.")


if __name__ == "__main__":
    main()
