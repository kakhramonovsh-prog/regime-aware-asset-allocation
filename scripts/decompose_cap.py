"""Exact gross/net decomposition of the weight-cap sensitivity.

Reporting step only: it re-derives the cap specifications' return series
and splits the Sharpe difference into the part removed by transaction
costs and the part that survives them. It changes no specification and
runs no new inference.

    cost effect = dSharpe(0 bps) - dSharpe(10 bps)

This is exact, unlike approximating cost expenditure divided by
volatility, which conflates the cost drag on the mean with its effect on
the ratio.

Usage::

    python scripts/decompose_cap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.run_robustness import Spec, covariances_for_spec  # noqa: E402
from src import backtest as bt  # noqa: E402
from src import data_loader, eda  # noqa: E402
from src import metrics as mt  # noqa: E402
from src import optimization as opt  # noqa: E402
from src import preprocessing as prep  # noqa: E402
from src import regimes as rg  # noqa: E402

SNAPSHOT = PROJECT_ROOT / "data" / "snapshots" / "manifest_2026-08-06.json"
TRADING_DAYS = 252


def main() -> None:
    config = data_loader.load_config()
    with open(PROJECT_ROOT / "config" / "analysis_plan.yaml", encoding="utf-8") as fh:
        plan = yaml.safe_load(fh)
    eda.verify_snapshot(SNAPSHOT, PROJECT_ROOT)

    processed = PROJECT_ROOT / config["data"]["processed_dir"]
    prices = pd.read_csv(processed / "prices.csv", index_col="Date", parse_dates=True)
    macro = pd.read_csv(processed / "macro.csv", index_col="Date", parse_dates=True)
    origins = rg.rebalance_origins(
        prep.log_returns(prices).index,
        first_signal_after=plan["sample"]["training_end"],
        min_observations=plan["volatility"]["min_observations"],
    )

    specs = [
        Spec("cap_30pct", "weight cap", max_weight=0.30),
        Spec("primary", "-", max_weight=0.40),
        Spec("cap_50pct", "weight cap", max_weight=0.50),
    ]
    hmm_cache: dict = {}
    rows: list[dict] = []

    for spec in specs:
        print(f"  {spec.name} ...", flush=True)
        consumed, rolling_lw, _, _ = covariances_for_spec(
            spec, prices, macro, origins, hmm_cache, config, plan
        )
        assets = list(spec.universe)
        daily = prices[assets].pct_change().iloc[1:]
        execution_map = bt.execution_dates_for(origins, daily.index)
        opt_cfg = opt.OptimizerConfig(max_weight=spec.max_weight)

        targets = {"regime": {}, "rolling": {}}
        for signal, execution in execution_map.items():
            for key, matrix in (("regime", consumed[signal]),
                                ("rolling", rolling_lw[signal])):
                targets[key][execution] = opt.min_variance_weights(matrix, opt_cfg)[0]

        paths = {k: bt.simulate_path(daily, v) for k, v in targets.items()}
        dates = paths["regime"]["events"].index
        risk_free = mt.risk_free_daily(dates, macro["DFF"], entry_date=dates[0])

        record = {"cap": f"{spec.max_weight:.0%}", "specification": spec.name}
        for bps in (0.0, 10.0):
            costed = {k: bt.apply_costs(v["events"], bps) for k, v in paths.items()}
            a, b = costed["regime"]["net_return"], costed["rolling"]["net_return"]
            label = "gross" if bps == 0 else "net"
            record[f"sharpe_diff_{label}"] = (
                mt.sharpe_ratio(a, risk_free) - mt.sharpe_ratio(b, risk_free)
            )
            excess_a = a - risk_free.reindex(a.index).fillna(0.0)
            excess_b = b - risk_free.reindex(b.index).fillna(0.0)
            record[f"excess_return_diff_ann_{label}"] = (
                excess_a.mean() - excess_b.mean()
            ) * TRADING_DAYS
            record[f"vol_diff_{label}"] = (
                mt.annualized_volatility(a) - mt.annualized_volatility(b)
            )
            record[f"vol_regime_{label}"] = mt.annualized_volatility(a)
            record[f"vol_rolling_{label}"] = mt.annualized_volatility(b)

        record["cost_effect_on_sharpe_diff"] = (
            record["sharpe_diff_gross"] - record["sharpe_diff_net"]
        )
        rows.append(record)

    frame = pd.DataFrame(rows)
    # How much of the cap-30 vs cap-40 change is attributable to costs?
    primary = frame[frame["specification"] == "primary"].iloc[0]
    cap30 = frame[frame["specification"] == "cap_30pct"].iloc[0]
    total_change = cap30["sharpe_diff_net"] - primary["sharpe_diff_net"]
    cost_change = primary["cost_effect_on_sharpe_diff"] - cap30["cost_effect_on_sharpe_diff"]
    gross_change = cap30["sharpe_diff_gross"] - primary["sharpe_diff_gross"]

    attribution = pd.DataFrame([
        {"component": "total change in net dSharpe (cap30 - cap40)", "value": total_change},
        {"component": "attributable to reduced cost drag", "value": cost_change},
        {"component": "attributable to gross (pre-cost) differences", "value": gross_change},
        {"component": "cost share of total", "value": cost_change / total_change},
        {"component": "gross share of total", "value": gross_change / total_change},
    ])

    out_dir = PROJECT_ROOT / "outputs" / "robustness"
    frame.to_csv(out_dir / "cap_gross_net_decomposition.csv", index=False)
    attribution.to_csv(out_dir / "cap30_attribution.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== GROSS / NET DECOMPOSITION ===")
    print(frame[[
        "cap", "sharpe_diff_gross", "sharpe_diff_net",
        "cost_effect_on_sharpe_diff", "excess_return_diff_ann_net", "vol_diff_net",
    ]].round(5).to_string(index=False))
    print("\n=== CAP-30 ATTRIBUTION ===")
    print(attribution.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
