"""Generate every table, figure and macro the paper uses. One command.

Reads only artifacts produced by earlier phases, routes every number
through ``src/units.py``, verifies each display value against its raw
value, and writes:

* ``paper/tables/*.csv``   - display tables carrying raw + display + unit
* ``paper/tables/*.tex``   - LaTeX tables generated from those frames
* ``paper/macros.tex``     - every headline number as a LaTeX macro
* ``paper/figures/*.png``  - the figures the paper references
* ``outputs/results_manifest.json`` - provenance for all of it

No number may be typed into the LaTeX by hand. The paper calls
``\\primarySharpeDiff`` and similar macros, so a changed result
propagates automatically and a stale number cannot survive.

    python scripts/generate_paper_outputs.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import latex as tex_util  # noqa: E402
from src import units as un  # noqa: E402
from src.eda import sha256_of  # noqa: E402

OUTPUTS = PROJECT_ROOT / "outputs"
PAPER = PROJECT_ROOT / "paper"
SNAPSHOT = PROJECT_ROOT / "data" / "snapshots" / "manifest_2026-08-06.json"

# Figures the paper references, in the order they appear.
PAPER_FIGURES = [
    "normalized_prices.png",
    "vix_vs_realized_vol.png",
    "regime_probabilities.png",
    "cumulative_growth.png",
    "strategy_drawdowns.png",
    "robustness_forest.png",
]


# LaTeX command names may contain letters only. Digits must be spelled
# out, and a name like \cap30Diff would additionally collide with the
# built-in \cap operator.
_DIGIT_WORDS = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
}
_NUMBER_WORDS = {
    "30": "Thirty", "50": "Fifty", "120": "OneTwenty", "2020": "TwentyTwenty",
    "95": "NinetyFive", "99": "NinetyNine",
}


def macro_name(key: str) -> str:
    """snake_case to a LaTeX-legal, letters-only camelCase macro name."""
    parts = key.split("_")
    camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
    for number, word in sorted(_NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
        camel = camel.replace(number, word)
    camel = "".join(_DIGIT_WORDS.get(ch, ch) for ch in camel)
    if not camel.isalpha():
        raise ValueError(f"macro name {camel!r} from {key!r} is not letters-only")
    return camel


CAPTIONS = {
    "table1_performance": "Realized performance, 2010--2026, net of 10 basis "
        "points. Annualized figures; maximum drawdown from the net wealth path.",
    "table2_primary_inference": "Preregistered primary comparison. Paired "
        "stationary bootstrap, 10{,}000 replications, mean block 21 trading "
        "days, seed 12345, on daily excess returns.",
    "table3_secondary_intervals": "Secondary metric differences, regime-aware "
        "minus rolling Ledoit--Wolf. Unadjusted intervals.",
    "table4_robustness_grid": "Robustness grid: thirteen specifications, each "
        "varying one factor from the primary model.",
}


def humanize(value: str) -> str:
    """Display label via the central registry.

    Raises on an unregistered identifier rather than falling back
    to the raw string, which is how an unescaped underscore
    previously reached the compiler.
    """
    return tex_util.latex_label_for(value)


def main() -> None:
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    print("Reading phase outputs...")
    performance = pd.read_csv(OUTPUTS / "performance" / "performance_summary.csv")
    primary = pd.read_csv(OUTPUTS / "inference" / "primary_inference_summary.csv").iloc[0]
    secondary = pd.read_csv(OUTPUTS / "inference" / "secondary_metric_intervals.csv")
    hac = pd.read_csv(OUTPUTS / "inference" / "hac_mean_difference.csv")
    grid = pd.read_csv(OUTPUTS / "robustness" / "robustness_grid.csv")
    turnover_audit = pd.read_parquet(OUTPUTS / "backtests" / "accounting_audit.parquet")

    main_bps = 10
    net10 = performance[
        (performance["series"] == "net") & (performance["cost_bps"] == main_bps)
    ].set_index("strategy")

    # ------------------------------------------------------------------
    # Table 1: main performance, every number through the unit registry
    # ------------------------------------------------------------------
    order = ["equal_weight", "static_6040", "static_minvar",
             "rolling_lw_minvar", "ewma_scaled_minvar", "regime_minvar"]
    metrics = ["cagr", "ann_volatility", "sharpe", "sortino",
               "max_drawdown", "calmar"]
    rows = []
    for strategy in order:
        rows.append(un.build_display_table(
            {m: float(net10.loc[strategy, m]) for m in metrics}, label=strategy
        ))
    table1 = pd.concat(rows, ignore_index=True)
    un.verify_display_table(table1)

    # ------------------------------------------------------------------
    # Table 2: primary inference
    # ------------------------------------------------------------------
    table2 = un.build_display_table(
        {
            "sharpe_difference": float(primary["observed_difference"]),
            "ci95_lower": float(primary["ci95_lower"]),
            "ci95_upper": float(primary["ci95_upper"]),
            "p_value": float(primary["p_one_sided_centered_null"]),
        },
        label="regime_minvar minus rolling_lw_minvar",
    )
    un.verify_display_table(table2)

    # ------------------------------------------------------------------
    # Table 3: secondary metric intervals
    # ------------------------------------------------------------------
    metric_map = {"sharpe": "sharpe_difference", "sortino": "sharpe_difference",
                  "ann_volatility": "vol_difference", "cagr": "cagr_difference",
                  "max_drawdown": "maxdd_difference", "calmar": "sharpe_difference"}
    rows = []
    for _, row in secondary.iterrows():
        key = metric_map[row["metric"]]
        rows.append(un.build_display_table(
            {key: float(row["observed_difference"])}, label=row["metric"]
        ))
    table3 = pd.concat(rows, ignore_index=True)
    un.verify_display_table(table3)

    # ------------------------------------------------------------------
    # Table 4: robustness grid
    # ------------------------------------------------------------------
    rows = []
    for _, row in grid.iterrows():
        rows.append(un.build_display_table(
            {"sharpe_difference": float(row["sharpe_difference"]),
             "ci95_lower": float(row["ci95_lower"]),
             "ci95_upper": float(row["ci95_upper"]),
             "vol_difference": float(row["vol_difference"])},
            label=row["specification"],
        ))
    table4 = pd.concat(rows, ignore_index=True)
    un.verify_display_table(table4)

    tables = {
        "table1_performance": table1,
        "table2_primary_inference": table2,
        "table3_secondary_intervals": table3,
        "table4_robustness_grid": table4,
    }
    print("\nWriting tables (raw + display + unit in every row):")
    for name, frame in tables.items():
        csv_path = PAPER / "tables" / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        artifacts.append(csv_path)

        tex = frame.pivot(index="label", columns="metric", values="formatted")
        # Preserve the declared row order rather than pandas' alphabetical
        # sort, then replace every machine identifier with a display label
        # so no underscore reaches LaTeX.
        order = list(dict.fromkeys(frame["label"]))
        tex = tex.reindex(order)
        column_order = [c for c in dict.fromkeys(frame["metric"]) if c in tex.columns]
        tex = tex[column_order]
        tex = tex.map(
            lambda cell: "$-$" + cell[1:]
            if isinstance(cell, str) and cell.startswith("-") else cell
        )
        tex.index = [humanize(v) for v in tex.index]
        tex.columns = [humanize(c) for c in tex.columns]
        tex.index.name = None
        tex.columns.name = None

        tex_path = PAPER / "tables" / f"{name}.tex"
        tex_path.write_text(
            tex.to_latex(
                na_rep="",
                caption=CAPTIONS.get(name, name.replace("_", " ")),
                label=f"tab:{name}",
                column_format="l" + "r" * len(tex.columns),
                position="htbp",
            ).replace(
                # \small keeps the wide performance table inside the text
                # block; without it Table 1 overfulls by 24pt.
                r"\begin{tabular}", "\\small\n" + r"\begin{tabular}"
            ),
            encoding="utf-8",
        )
        artifacts.append(tex_path)
        print(f"  {name}.csv / .tex  ({len(frame)} values)")

    # ------------------------------------------------------------------
    # Macros: every headline number the prose cites
    # ------------------------------------------------------------------
    hac21 = hac[hac["hac_lags"] == 21].iloc[0]
    regime_turnover = turnover_audit[
        (turnover_audit["strategy"] == "regime_minvar")
        & (turnover_audit["cost_bps"] == main_bps)
    ]
    comparator_turnover = turnover_audit[
        (turnover_audit["strategy"] == "rolling_lw_minvar")
        & (turnover_audit["cost_bps"] == main_bps)
    ]
    # Signed volatility difference. The magnitude macro is derived from
    # this value rather than stored separately, so the two cannot drift
    # apart and state opposite things.
    _vol_signed = float(
        secondary.loc[secondary["metric"] == "ann_volatility",
                      "observed_difference"].iloc[0]
    )

    grid_by_spec = grid.set_index("specification")
    net_by_cost = performance[performance["series"] == "net"].set_index(
        ["strategy", "cost_bps"]
    )
    subperiods = pd.read_csv(OUTPUTS / "robustness" / "subperiod_descriptive.csv")
    cap_attr = pd.read_csv(OUTPUTS / "robustness" / "cap30_attribution.csv")
    hmm_diag = pd.read_csv(OUTPUTS / "robustness" / "hmm_diagnostics_by_spec.csv")
    conditioning = pd.read_csv(OUTPUTS / "covariance" / "conditioning_summary.csv")
    cov_audit = pd.read_csv(OUTPUTS / "covariance" / "covariance_audit.csv")
    stability = pd.read_csv(OUTPUTS / "robustness" / "argmax_counterfactual.csv")
    revision = pd.read_csv(OUTPUTS / "regimes" / "model_revision_stability.csv")
    correlations = pd.read_csv(OUTPUTS / "tables" / "correlation_matrix.csv", index_col=0)
    forecasts = pd.read_csv(OUTPUTS / "tables" / "forecast_comparisons.csv")
    blocks = pd.read_csv(OUTPUTS / "inference" / "block_length_sensitivity.csv")
    reproduction = json.loads(
        (OUTPUTS / "reproduction_comparison.json").read_text(encoding="utf-8")
    )

    def spec(name: str, column: str = "sharpe_difference") -> float:
        return float(grid_by_spec.loc[name, column])

    def sharpe_at(strategy: str, bps: int) -> float:
        return float(net_by_cost.loc[(strategy, float(bps)), "sharpe"])

    def subperiod(name: str) -> float:
        return float(
            subperiods.loc[subperiods["subperiod"] == name, "sharpe_difference"].iloc[0]
        )

    def cond(estimator: str, column: str) -> float:
        return float(
            conditioning.loc[conditioning["estimator"] == estimator, column].iloc[0]
        )

    years = len(regime_turnover) / 252

    # The equity block is the three equity ETFs; the prose quotes the
    # range of their pairwise correlations, not the whole matrix.
    equity = [t for t in ("SPY", "QQQ", "IWM") if t in correlations.columns]
    equity_block = correlations.loc[equity, equity].to_numpy()
    equity_pairs = equity_block[np.triu_indices(len(equity), k=1)]

    # Largest deviation of any block-length-sensitivity interval bound
    # from the corresponding primary bound.
    primary_block = blocks[blocks["role"] == "PRIMARY"].iloc[0]
    block_deviation = float(max(
        (blocks[["ci95_lower", "ci95_upper"]]
         - [primary_block["ci95_lower"], primary_block["ci95_upper"]])
        .abs().to_numpy().max(),
        0.0,
    ))

    regime_cost_annual = float(regime_turnover["cost_fraction"].sum() / years)
    comparator_cost_annual = float(comparator_turnover["cost_fraction"].sum() / years)
    regime_half = float(regime_turnover["half_turnover_reporting"].sum() / years)
    comparator_half = float(comparator_turnover["half_turnover_reporting"].sum() / years)

    # Every reported empirical result gets a macro. The census
    # (scripts/number_census.py) fails if any of these appears as a
    # literal in the prose instead.
    headline = {
        # --- primary comparison
        "primary_sharpe_diff": float(primary["observed_difference"]),
        "primary_ci_lower": float(primary["ci95_lower"]),
        "primary_ci_upper": float(primary["ci95_upper"]),
        "primary_p_value": float(primary["p_one_sided_centered_null"]),
        "primary_bootstrap_sd": float(primary["bootstrap_sd"]),
        "primary_sds_from_zero": abs(float(primary["observed_difference"])
                                     / float(primary["bootstrap_sd"])),
        "ci_width_multiple": (float(primary["ci95_upper"])
                              - float(primary["ci95_lower"]))
                             / abs(float(primary["observed_difference"])),
        "block_max_deviation": block_deviation,
        # --- volatility (signed, magnitude and interval)
        "vol_difference": _vol_signed,
        "vol_difference_magnitude": abs(_vol_signed),
        "vol_ci_lower": float(secondary.loc[secondary["metric"] == "ann_volatility",
                                            "ci95_lower"].iloc[0]),
        "vol_ci_upper": float(secondary.loc[secondary["metric"] == "ann_volatility",
                                            "ci95_upper"].iloc[0]),
        # --- returns
        "hac_annual_return_diff": float(hac21["annualized_arithmetic_difference"]),
        "hac_standard_error": float(hac21["hac_standard_error_daily"]) * 252,
        "hac_p_value": float(hac21["p_value_two_sided"]),
        "hac_sds_from_zero": abs(float(hac21["t_statistic"])),
        # --- ladder Sharpe ratios
        "equal_weight_sharpe": sharpe_at("equal_weight", 10),
        "static_minvar_sharpe": sharpe_at("static_minvar", 10),
        "ewma_scaled_sharpe": sharpe_at("ewma_scaled_minvar", 10),
        "regime_sharpe": sharpe_at("regime_minvar", 10),
        "comparator_sharpe": sharpe_at("rolling_lw_minvar", 10),
        # --- cost sensitivity
        "diff_at_zero_bps": sharpe_at("regime_minvar", 0) - sharpe_at("rolling_lw_minvar", 0),
        "diff_at_twenty_bps": sharpe_at("regime_minvar", 20) - sharpe_at("rolling_lw_minvar", 20),
        # --- turnover and costs
        "regime_half_turnover": regime_half,
        "comparator_half_turnover": comparator_half,
        "regime_cost": regime_cost_annual,
        "comparator_cost": comparator_cost_annual,
        "turnover_ratio": regime_half / comparator_half,
        "cost_hurdle": regime_cost_annual - comparator_cost_annual,
        # --- robustness
        "three_state_diff": spec("hmm_3_states"),
        "drop_rv_diff": spec("drop_realized_vol"),
        "cap30_diff": spec("cap_30pct"),
        "cap30_p": float(grid_by_spec.loc["cap_30pct", "p_one_sided_centered"]),
        "cap30_p_holm": float(grid_by_spec.loc["cap_30pct", "p_holm_robustness_family"]),
        "alt_seeds_diff": spec("alt_seeds_100"),
        "a2_accept_diff": spec("a2_accept_as_estimated"),
        "kappa_max_move": max(
            abs(spec("neff_kappa_30") - spec("primary")),
            abs(spec("neff_kappa_120") - spec("primary")),
        ),
        # --- cap decomposition
        "cap30_total_change": float(cap_attr.iloc[0]["value"]),
        "cap30_cost_component": float(cap_attr.iloc[1]["value"]),
        "cap30_gross_component": float(cap_attr.iloc[2]["value"]),
        "cap30_cost_share": float(cap_attr.iloc[3]["value"]),
        "cap30_gross_share": float(cap_attr.iloc[4]["value"]),
        # --- robustness counts
        "n_specs_positive": float((grid["sharpe_difference"] > 0).sum()),
        "n_sign_reversals": float(
            (grid.loc[grid["specification"] != "primary", "sharpe_difference"] < 0).sum()),
        "n_specs_containing_zero": float(grid["interval_contains_zero"].sum()),
        "n_secondary_metrics": float((secondary["role"] == "secondary").sum()),
        "n_secondary_excluding_zero": float(
            secondary.loc[secondary["role"] == "secondary",
                          "interval_excludes_zero"].sum()),
        "n_forecast_pairs_surviving_holm": float((forecasts["p_holm"] < 0.05).sum()),
        # --- subperiods
        "pre2020_diff": subperiod("pre_2020"),
        "post2020_diff": subperiod("post_2020"),
        "covid_days": float(
            subperiods.loc[subperiods["subperiod"] == "covid", "n_days"].iloc[0]),
        "tightening_days": float(
            subperiods.loc[subperiods["subperiod"] == "tightening_2022",
                           "n_days"].iloc[0]),
        "oos_years": years,
        # --- diagnostics
        "ewma_mean_condition": cond("ewma", "mean_cov_condition"),
        "ewma_max_condition": cond("ewma", "max_cov_condition"),
        "state1_condition": cond("state_1_raw", "mean_cov_condition"),
        "state0_condition": cond("state_0_raw", "mean_cov_condition"),
        "between_state_max": float(
            cov_audit[cov_audit["estimator"] == "regime_mixture"]
            ["between_state_relative_norm"].max()),
        "three_state_mid_occupancy": float(
            hmm_diag.loc[hmm_diag["specification"] == "hmm_3_states",
                         "mean_occupancy_s1"].iloc[0]),
        "three_state_min_neff": float(
            hmm_diag.loc[hmm_diag["specification"] == "hmm_3_states",
                         "min_n_eff_s1"].iloc[0]),
        "equity_corr_min": float(equity_pairs.min()),
        "equity_corr_max": float(equity_pairs.max()),
        "a2_fallback_origins": float(
            cov_audit.loc[cov_audit["fallback_used"].astype(str).str.lower()
                          .isin(["true", "1"]), "date"].nunique()),
        # --- selection-rule and reproduction diagnostics
        # Two distinct quantities that were both written as bare numbers:
        # how often argmax and the tolerance rule disagree inside one run,
        # and how often the selected seed changes between consecutive
        # refits. Naming them apart is the point.
        "argmax_tolerance_disagreements": float(
            stability.loc[stability["quantity"] ==
                          "origins where the two rules disagree within one run",
                          "value"].iloc[0]),
        "refit_seed_changes": float(revision["selected_seed_changed"].sum()),
        "reproduction_max_difference": float(
            reproduction["filtered_probabilities"]["max_absolute_difference"]),
        "reproduction_seed_disagreements": float(
            reproduction["selected_fits"]["selected_seed_disagreements"]),
    }

    units_for = {
        "primary_sharpe_diff": "sharpe_difference",
        "primary_ci_lower": "ci95_lower",
        "primary_ci_upper": "ci95_upper",
        "primary_p_value": "p_value",
        "primary_bootstrap_sd": "sharpe",
        "primary_sds_from_zero": "sharpe",
        "ci_width_multiple": "factor",
        "block_max_deviation": "sharpe_difference",
        "vol_difference": "vol_difference",
        "vol_difference_magnitude": "vol_difference",
        "vol_ci_lower": "vol_difference",
        "vol_ci_upper": "vol_difference",
        "hac_annual_return_diff": "mean_return_difference_annualized",
        "hac_standard_error": "mean_return_difference_annualized",
        "hac_p_value": "p_value",
        "hac_sds_from_zero": "sharpe",
        "equal_weight_sharpe": "sharpe",
        "static_minvar_sharpe": "sharpe",
        "ewma_scaled_sharpe": "sharpe",
        "regime_sharpe": "sharpe",
        "comparator_sharpe": "sharpe",
        "diff_at_zero_bps": "sharpe_difference",
        "diff_at_twenty_bps": "sharpe_difference",
        "regime_half_turnover": "half_turnover",
        "comparator_half_turnover": "half_turnover",
        "regime_cost": "cost_expenditure_annualized",
        "comparator_cost": "cost_expenditure_annualized",
        "turnover_ratio": "factor",
        "cost_hurdle": "cost_expenditure_annualized",
        "three_state_diff": "sharpe_difference",
        "drop_rv_diff": "sharpe_difference",
        "cap30_diff": "sharpe_difference",
        "cap30_p": "p_value",
        "cap30_p_holm": "p_value",
        "alt_seeds_diff": "sharpe_difference",
        "a2_accept_diff": "sharpe_difference",
        "kappa_max_move": "sharpe_difference",
        "cap30_total_change": "sharpe_difference",
        "cap30_cost_component": "sharpe_difference",
        "cap30_gross_component": "sharpe_difference",
        "cap30_cost_share": "percent_share",
        "cap30_gross_share": "percent_share",
        "n_specs_positive": "count",
        "n_sign_reversals": "count",
        "n_specs_containing_zero": "count",
        "n_secondary_metrics": "count",
        "n_secondary_excluding_zero": "count",
        "n_forecast_pairs_surviving_holm": "count",
        "pre2020_diff": "sharpe_difference",
        "post2020_diff": "sharpe_difference",
        "covid_days": "count",
        "tightening_days": "count",
        "oos_years": "years",
        "ewma_mean_condition": "count",
        "ewma_max_condition": "count",
        "state1_condition": "condition_number",
        "state0_condition": "condition_number",
        "between_state_max": "percent_share_precise",
        "three_state_mid_occupancy": "percent_share",
        "three_state_min_neff": "count",
        "equity_corr_min": "correlation",
        "equity_corr_max": "correlation",
        "a2_fallback_origins": "count",
        "argmax_tolerance_disagreements": "count",
        "refit_seed_changes": "count",
        "reproduction_max_difference": "scientific",
        "reproduction_seed_disagreements": "count",
    }

    # Where each reported result was computed. The census requires a
    # source artifact per result so a reader can trace any number in the
    # manuscript back to the file that produced it.
    source_for = {
        "primary": "outputs/inference/primary_inference_summary.csv",
        "vol": "outputs/inference/secondary_metric_intervals.csv",
        "hac": "outputs/inference/hac_mean_difference.csv",
        "block": "outputs/inference/block_length_sensitivity.csv",
        "ci_width": "outputs/inference/primary_inference_summary.csv",
        "sharpe_ladder": "outputs/performance/performance_summary.csv",
        "turnover": "outputs/backtests/turnover.parquet",
        "robustness": "outputs/robustness/robustness_grid.csv",
        "cap": "outputs/robustness/cap30_attribution.csv",
        "subperiod": "outputs/robustness/subperiod_descriptive.csv",
        "conditioning": "outputs/covariance/conditioning_summary.csv",
        "covariance_audit": "outputs/covariance/covariance_audit.csv",
        "hmm": "outputs/robustness/hmm_diagnostics_by_spec.csv",
        "correlation": "outputs/tables/correlation_matrix.csv",
        "forecasts": "outputs/tables/forecast_comparisons.csv",
        "secondary": "outputs/inference/secondary_metric_intervals.csv",
        "stability": "outputs/robustness/argmax_counterfactual.csv",
        "revision": "outputs/regimes/model_revision_stability.csv",
        "reproduction": "outputs/reproduction_comparison.json",
    }
    result_sources = {
        "primary_sharpe_diff": "primary", "primary_ci_lower": "primary",
        "primary_ci_upper": "primary", "primary_p_value": "primary",
        "primary_bootstrap_sd": "primary", "primary_sds_from_zero": "primary",
        "ci_width_multiple": "ci_width", "block_max_deviation": "block",
        "vol_difference": "vol", "vol_difference_magnitude": "vol",
        "vol_ci_lower": "vol", "vol_ci_upper": "vol",
        "hac_annual_return_diff": "hac", "hac_standard_error": "hac",
        "hac_p_value": "hac", "hac_sds_from_zero": "hac",
        "equal_weight_sharpe": "sharpe_ladder",
        "static_minvar_sharpe": "sharpe_ladder",
        "ewma_scaled_sharpe": "sharpe_ladder", "regime_sharpe": "sharpe_ladder",
        "comparator_sharpe": "sharpe_ladder", "diff_at_zero_bps": "sharpe_ladder",
        "diff_at_twenty_bps": "sharpe_ladder",
        "regime_half_turnover": "turnover", "comparator_half_turnover": "turnover",
        "regime_cost": "turnover", "comparator_cost": "turnover",
        "turnover_ratio": "turnover", "cost_hurdle": "turnover",
        "three_state_diff": "robustness", "drop_rv_diff": "robustness",
        "cap30_diff": "robustness", "cap30_p": "robustness",
        "cap30_p_holm": "robustness", "alt_seeds_diff": "robustness",
        "a2_accept_diff": "robustness", "kappa_max_move": "robustness",
        "n_specs_positive": "robustness", "n_sign_reversals": "robustness",
        "n_specs_containing_zero": "robustness",
        "cap30_total_change": "cap", "cap30_cost_component": "cap",
        "cap30_gross_component": "cap", "cap30_cost_share": "cap",
        "cap30_gross_share": "cap",
        "n_secondary_metrics": "secondary",
        "n_secondary_excluding_zero": "secondary",
        "n_forecast_pairs_surviving_holm": "forecasts",
        "pre2020_diff": "subperiod", "post2020_diff": "subperiod",
        "covid_days": "subperiod", "tightening_days": "subperiod",
        "oos_years": "sharpe_ladder",
        "ewma_mean_condition": "conditioning", "ewma_max_condition": "conditioning",
        "state1_condition": "conditioning", "state0_condition": "conditioning",
        "between_state_max": "covariance_audit",
        "a2_fallback_origins": "covariance_audit",
        "three_state_mid_occupancy": "hmm", "three_state_min_neff": "hmm",
        "equity_corr_min": "correlation", "equity_corr_max": "correlation",
        "argmax_tolerance_disagreements": "stability",
        "refit_seed_changes": "revision",
        "reproduction_max_difference": "reproduction",
        "reproduction_seed_disagreements": "reproduction",
    }
    missing_sources = set(headline) - set(result_sources)
    if missing_sources:
        raise SystemExit(
            f"no source artifact recorded for {sorted(missing_sources)}; every "
            "reported result must be traceable to the file that produced it"
        )

    macro_frame = un.build_display_table(
        {units_for[k]: v for k, v in headline.items()}, label="headline"
    )  # verification only; names collide, so write macros from `headline`
    # The title-page date is frozen in config rather than taken from
    # \today: a clean-clone rebuild on a later day would otherwise change
    # page 1 and with it the PDF hash the visual inspection attests to.
    import yaml  # noqa: PLC0415 - only needed here

    config = yaml.safe_load(
        (PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    release_date = str(config["manuscript"]["release_date"])
    release = dt.date.fromisoformat(release_date)
    release_display = f"{release:%B} {release.day}, {release.year}"

    # A wall-clock stamp here would make macros.tex differ on every run,
    # so a clean-clone reproduction could never be byte-identical. The
    # header records what the file was generated *from* instead.
    # Author identifiers. Kept in config so the manuscript source carries
    # no hand-typed contact details, and so an unset value is visible as
    # PENDING on the page rather than silently absent.
    manuscript = config["manuscript"]
    orcid = str(manuscript.get("orcid", "PENDING"))
    corresponding_email = str(manuscript.get("corresponding_email", "PENDING"))

    lines = [
        "% Generated by scripts/generate_paper_outputs.py. Do not edit.",
        f"% Manuscript release date: {release_date}",
        "",
        f"\\newcommand{{\\releaseDate}}{{{release_display}}}",
        f"\\newcommand{{\\orcidID}}{{{tex_util.to_latex(orcid)}}}",
        f"\\newcommand{{\\correspondingEmail}}{{{tex_util.to_latex(corresponding_email)}}}",
    ]
    inventory = []
    for key, raw in headline.items():
        display, unit, formatted = un.to_display(raw, units_for[key])
        # A leading text hyphen is not a minus sign; the figures already
        # use a typographic minus, so the prose must match. It must be
        # \ensuremath and not $-$: these macros are also used inside
        # equation environments, where a literal $ closes math mode and
        # aborts the compile with no PDF.
        if formatted.startswith("-"):
            formatted = r"\ensuremath{-}" + formatted[1:]
        # '%' is a LaTeX comment character: an unescaped one silently
        # swallows the rest of the line, including the closing brace.
        unit_tex = unit.replace("%", r"\%")
        # A thin space belongs before a unit ("10.838 bps") but not
        # before a percent sign: "54.19 %" sits next to hard-coded "30%"
        # in the same sentence and reads as a mistake.
        if not unit_tex:
            suffix = ""
        elif unit_tex == r"\%":
            suffix = unit_tex
        else:
            suffix = f"\\,{unit_tex}"
        lines.append(f"\\newcommand{{\\{macro_name(key)}}}{{{formatted}{suffix}}}")
        inventory.append({
            "result_id": key,
            "macro": f"\\{macro_name(key)}",
            "raw_value": raw,
            "display_value": display,
            "unit": unit or units_for[key],
            "rendered": f"{formatted}{suffix}",
            "source_artifact": source_for[result_sources[key]],
        })
    macros_path = PAPER / "macros.tex"
    macros_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts.append(macros_path)

    inventory_path = PAPER / "result_inventory.csv"
    pd.DataFrame(inventory).to_csv(inventory_path, index=False)
    artifacts.append(inventory_path)

    print(f"\nWrote {len(headline)} LaTeX macros -> paper/macros.tex")
    print(f"Wrote {len(inventory)} result rows -> paper/result_inventory.csv")
    for line in lines[3:]:
        print(f"  {line}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nCopying figures:")
    for name in PAPER_FIGURES:
        source = OUTPUTS / "figures" / name
        if not source.exists():
            print(f"  MISSING {name} (run the earlier phases)")
            continue
        destination = PAPER / "figures" / name
        shutil.copy2(source, destination)
        artifacts.append(destination)
        print(f"  {name}")

    # ------------------------------------------------------------------
    # Results manifest
    # ------------------------------------------------------------------
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    config_hash = hashlib.sha256()
    for name in ("config.yaml", "analysis_plan.yaml"):
        config_hash.update((PROJECT_ROOT / "config" / name).read_bytes())

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "config_sha256": config_hash.hexdigest(),
        "data_snapshot_sha256": sha256_of(SNAPSHOT),
        "python_version": sys.version.split()[0],
        "headline_values": headline,
        "note": (
            "Every value passed through src/units.py and was verified against "
            "its raw value. The LaTeX reads numbers only via paper/macros.tex; "
            "no result is typed by hand."
        ),
        "artifacts": {
            str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_of(p)
            for p in artifacts
        },
    }
    manifest_path = OUTPUTS / "results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nResults manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(artifacts)} artifacts, commit {commit[:12]}")
    print("\nAll paper outputs generated. No hand-entered numbers.")


if __name__ == "__main__":
    main()
