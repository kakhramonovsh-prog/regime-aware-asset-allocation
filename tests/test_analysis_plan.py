"""The runtime config must match the preregistered analysis plan.

config/analysis_plan.yaml is frozen at tag v0.2.0-preregistered and is
never edited; config/config.yaml is the live runtime configuration.
This test fails if the two drift apart on any frozen parameter, so a
quiet change to the runtime config cannot silently deviate from the
preregistered design.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    with open(PROJECT_ROOT / "config" / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_config_matches_preregistered_plan():
    config = _load("config.yaml")
    plan = _load("analysis_plan.yaml")

    # Sample and timeline
    assert config["data"]["start_date"] == plan["sample"]["full_start"]
    assert config["backtest"]["training_end"] == plan["sample"]["training_end"]
    assert config["backtest"]["oos_start"] == plan["sample"]["oos_start"]
    assert config["backtest"]["rebalance_frequency"] == plan["sample"]["rebalance_frequency"]
    assert config["backtest"]["execution_lag_days"] == 1
    assert config["data"]["macro_signal_lag_days"] == plan["timeline"]["macro_signal_lag_days"]

    # Costs
    assert config["backtest"]["main_cost_bps"] == plan["hypothesis"]["primary_cost_bps"]
    assert set(plan["costs"]["sensitivity_bps"]).issubset(
        set(config["backtest"]["transaction_cost_bps"])
    )

    # Portfolio constraints and strategy ladder
    assert config["portfolio"]["max_weight"] == plan["portfolio"]["max_weight"]
    assert config["portfolio"]["allow_short"] is False
    assert plan["portfolio"]["long_only"] is True
    assert config["portfolio"]["fully_invested"] == plan["portfolio"]["fully_invested"]
    assert config["portfolio"]["strategies"] == plan["portfolio"]["strategies_main"]
    assert plan["portfolio"]["expected_return_forecasts"] == "none"

    # Volatility models
    assert config["volatility_models"]["ewma_lambda"] == plan["volatility"]["ewma_lambda"]
    assert config["volatility_models"]["hist_window_days"] == plan["volatility"]["hist_window_days"]
    assert config["volatility_models"]["portfolio_model"] == plan["volatility"]["portfolio_model"]

    # Regime model
    assert config["regimes"]["n_states_main"] == plan["hmm"]["n_states_main"]
    assert config["regimes"]["n_states_alt"] == plan["hmm"]["n_states_robustness"]
    assert config["regimes"]["features"] == plan["hmm"]["features_main"]
    assert config["regimes"]["n_initializations"] == plan["hmm"]["n_initializations"]
    assert config["regimes"]["init_seeds_start"] == plan["hmm"]["init_seeds_start"]
    assert config["regimes"]["min_state_occupancy"] == plan["hmm"]["min_state_occupancy"]

    # Seeds
    assert config["project"]["random_seed"] == plan["seeds"]["global"]


def test_plan_declares_no_results_at_freeze():
    plan = _load("analysis_plan.yaml")
    assert plan["meta"]["results_existing_at_freeze"] is False
    assert plan["hypothesis"]["primary_comparator"] == "rolling_lw_minvar"
    assert plan["portfolio"]["volatility_targeting"] == "excluded_from_main"
    assert plan["hmm"]["probabilities_for_trading"] == "filtered_at_t_only"


def test_amendment_a1_recorded_before_results():
    plan = _load("analysis_plan.yaml")
    a1 = plan["amendments"]["A1"]
    assert a1["results_seen"] is False
    assert (
        plan["regime_covariance"]["responsibilities"]
        == "smoothed_given_information_through_t"
    )
    assert plan["robustness_grid"]["neff_thresholds"] == [30, 60, 120]
    assert plan["regime_covariance"]["neff_threshold"] == 60


def test_amendment_a2_recorded_before_covariance_results():
    plan = _load("analysis_plan.yaml")
    a2 = plan["amendments"]["A2"]
    assert a2["results_seen"] is False
    assert a2["affected_dates"] == [
        "2009-12-31", "2012-03-30", "2012-04-30", "2012-05-31"
    ]
    # The excluded actions are recorded so the rule cannot drift later.
    assert set(a2["excluded"]) == {
        "replace_the_probability",
        "interpolate_from_neighboring_months",
        "delete_the_rebalance",
        "refit_until_preferred_transition_matrix",
        "change_absorbing_threshold_after_seeing_results",
    }
    rc = plan["regime_covariance"]
    assert rc["degeneracy_fallback"] == "unconditional_ledoit_wolf"
    assert rc["fallback_preserves_probability"] is True
    assert "absorbing_transition" in rc["degeneracy_reasons"]
    assert "accept_as_estimated" in plan["robustness_grid"]["degeneracy_handling"]
