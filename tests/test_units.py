"""Tests for centralized unit conversion.

Includes regression tests for the two conversion errors that actually
reached draft reports, so neither can recur silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import units as u


# ---------------------------------------------------------------------------
# The specified cases
# ---------------------------------------------------------------------------

def test_specified_conversions():
    assert u.decimal_to_bps(0.0000314) == pytest.approx(0.314)
    assert u.decimal_to_percentage_points(0.00489) == pytest.approx(0.489)
    assert u.decimal_to_percentage_points(0.00175) == pytest.approx(0.175)


def test_percent_and_bps_scales():
    assert u.decimal_to_percent(0.0873) == pytest.approx(8.73)
    assert u.decimal_to_bps(0.001) == pytest.approx(10.0)
    assert u.decimal_to_bps(0.0010) == pytest.approx(10.0)      # 10 bps cost
    assert u.bps_to_decimal(10.0) == pytest.approx(0.001)


def test_bps_and_percent_round_trip():
    for value in (0.0000314, 0.00175, 0.0873, 0.5):
        assert u.bps_to_decimal(u.decimal_to_bps(value)) == pytest.approx(value)
        assert u.decimal_to_percent(value) / 100 == pytest.approx(value)


def test_bps_is_one_hundred_times_percent():
    """A basis point is 1/100 of a percent; keeps the two scales pinned."""
    for value in (0.0001, 0.0034, 0.12):
        assert u.decimal_to_bps(value) == pytest.approx(
            100 * u.decimal_to_percent(value)
        )


# ---------------------------------------------------------------------------
# Regression tests for errors that actually happened
# ---------------------------------------------------------------------------

def test_regression_annualized_return_difference_is_point_three_bps():
    """Phase 11 error: 3.140e-05 written as '3.1 basis points'.

    The correct value is 0.314 bps. An order-of-magnitude slip in prose.
    """
    raw = 3.140394e-05
    assert u.decimal_to_bps(raw) == pytest.approx(0.314, abs=0.001)
    assert u.decimal_to_bps(raw) != pytest.approx(3.14, abs=0.1)


def test_regression_cost_effect_on_sharpe_is_point_zero_zero_six():
    """Phase 12 error: 5.73 bps on 9% volatility written as '0.06 Sharpe'.

    0.000573 / 0.09 = 0.0064, an order of magnitude smaller.
    """
    cost_change = u.bps_to_decimal(5.73)
    impact = u.sharpe_impact_of_return_change(cost_change, 0.09)
    assert impact == pytest.approx(0.0064, abs=0.0002)
    assert impact != pytest.approx(0.064, abs=0.005)


def test_sharpe_impact_rejects_nonpositive_volatility():
    with pytest.raises(ValueError, match="positive"):
        u.sharpe_impact_of_return_change(0.001, 0.0)


# ---------------------------------------------------------------------------
# Display registry
# ---------------------------------------------------------------------------

def test_registered_metrics_convert_correctly():
    display, unit, formatted = u.to_display(0.00175, "vol_difference")
    assert display == pytest.approx(0.175)
    assert unit == "pp"
    assert formatted == "0.175"

    display, unit, formatted = u.to_display(3.140394e-05,
                                            "mean_return_difference_annualized")
    assert display == pytest.approx(0.314, abs=0.001)
    assert unit == "bps"

    display, unit, _ = u.to_display(0.020966, "sharpe_difference")
    assert display == pytest.approx(0.020966)   # ratios are not scaled
    assert unit == ""


def test_unregistered_metric_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no registered display unit"):
        u.to_display(0.5, "some_new_quantity")


def test_significant_digits_follow_cochrane():
    """Two to three significant digits, not whatever the program emits."""
    _, _, formatted = u.to_display(0.0209661234, "sharpe_difference")
    assert formatted == "0.021"
    _, _, formatted = u.to_display(0.0872538, "cagr")
    assert formatted == "8.73"


# ---------------------------------------------------------------------------
# Table construction and verification
# ---------------------------------------------------------------------------

def test_display_table_carries_raw_and_display():
    frame = u.build_display_table(
        {"sharpe_difference": 0.020966, "vol_difference": -0.00175,
         "cagr": 0.087254},
        label="regime_minvar",
    )
    assert set(frame.columns) == {
        "label", "metric", "raw_value", "display_value", "display_unit", "formatted",
    }
    assert len(frame) == 3
    vol = frame[frame["metric"] == "vol_difference"].iloc[0]
    assert vol["raw_value"] == pytest.approx(-0.00175)
    assert vol["display_value"] == pytest.approx(-0.175)
    assert vol["display_unit"] == "pp"


def test_verification_passes_on_clean_table():
    frame = u.build_display_table({"sharpe": 0.9292, "cagr": 0.0873})
    u.verify_display_table(frame)          # must not raise


def test_verification_catches_a_hand_edited_display_value():
    """The exact failure mode this module exists to prevent."""
    frame = u.build_display_table({"mean_return_difference_annualized": 3.14e-05})
    frame.loc[0, "display_value"] = 3.14   # the error I made, by hand
    with pytest.raises(ValueError, match="does not match its raw value"):
        u.verify_display_table(frame)


def test_verification_handles_non_finite():
    frame = u.build_display_table({"sharpe": np.nan})
    u.verify_display_table(frame)
    assert frame.loc[0, "formatted"] == ""


def test_every_registered_metric_has_a_valid_unit():
    for metric, unit_key in u.METRIC_UNITS.items():
        assert unit_key in u.UNITS, f"{metric} points at unknown unit {unit_key}"
        display, name, formatted = u.to_display(0.01234, metric)
        assert np.isfinite(display)
        assert isinstance(formatted, str)


def test_percent_and_percentage_points_are_numerically_equal_but_named_apart():
    """Same arithmetic, different meanings; both must exist so prose can
    distinguish a level from a difference."""
    assert u.decimal_to_percent(0.05) == u.decimal_to_percentage_points(0.05)
    assert u.UNITS["percent"].name == "%"
    assert u.UNITS["percentage_points"].name == "pp"
