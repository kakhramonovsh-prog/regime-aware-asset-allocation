"""Tests for Phase 11 inference.

Covers the estimand alignment (daily excess returns, sqrt-252), paired
resampling with identical indices, seed reproducibility, degenerate
cases, block-length handling, HAC on paired differences, the centered
null distribution, and the prohibition on mislabeling an ordinary
bootstrap probability as a p-value.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src import statistical_tests as st

CFG = st.BootstrapConfig(n_replications=400, batch_size=200)


@pytest.fixture
def paired_series():
    idx = pd.bdate_range("2015-01-01", periods=800, name="date")
    rng = np.random.default_rng(21)
    common = rng.normal(0.0003, 0.008, 800)          # shared market factor
    a = common + rng.normal(0.0001, 0.002, 800)
    b = common + rng.normal(0.0000, 0.003, 800)
    rf = pd.Series(0.00002, index=idx)
    net_a, net_b = pd.Series(a, index=idx), pd.Series(b, index=idx)
    return net_a, net_b, net_a - rf, net_b - rf


# ---------------------------------------------------------------------------
# Bootstrap index construction
# ---------------------------------------------------------------------------

def test_indices_have_expected_shape_and_range():
    rng = np.random.default_rng(0)
    idx = st.stationary_bootstrap_indices(100, 21, 50, rng)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_blocks_are_contiguous_and_wrap():
    """Consecutive positions usually continue the previous index + 1."""
    rng = np.random.default_rng(1)
    idx = st.stationary_bootstrap_indices(200, 50, 20, rng)
    steps = (idx[:, 1:] - idx[:, :-1]) % 200
    continuation_rate = (steps == 1).mean()
    # With mean block 50 roughly 98% of steps continue a block.
    assert continuation_rate > 0.9


def test_shorter_blocks_break_more_often():
    rng = np.random.default_rng(2)
    short = st.stationary_bootstrap_indices(200, 5, 40, rng)
    long = st.stationary_bootstrap_indices(200, 50, 40, rng)
    short_rate = (((short[:, 1:] - short[:, :-1]) % 200) == 1).mean()
    long_rate = (((long[:, 1:] - long[:, :-1]) % 200) == 1).mean()
    assert short_rate < long_rate


def test_mean_block_must_be_positive():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="mean_block"):
        st.stationary_bootstrap_indices(50, 0, 5, rng)


# ---------------------------------------------------------------------------
# Pairing and reproducibility
# ---------------------------------------------------------------------------

def test_identical_series_give_exactly_zero_difference(paired_series):
    """Paired resampling must cancel exactly when both strategies are
    the same series — proof that indices are shared, not independent."""
    net_a, _, excess_a, _ = paired_series
    draws = st.paired_bootstrap_differences(
        excess_a, excess_a.copy(), net_a, net_a.copy(),
        metrics=("sharpe", "cagr", "ann_volatility"), cfg=CFG,
    )
    for metric, values in draws.items():
        np.testing.assert_allclose(values, 0.0, atol=1e-12)
        assert values.std() == pytest.approx(0.0, abs=1e-12)


def test_independent_indices_would_not_cancel(paired_series):
    """Contrast case: if the two columns were resampled independently
    the differences would be non-degenerate. This pins that the
    implementation is paired."""
    net_a, _, excess_a, _ = paired_series
    rng = np.random.default_rng(7)
    idx_a = st.stationary_bootstrap_indices(len(excess_a), 21, 200, rng)
    idx_b = st.stationary_bootstrap_indices(len(excess_a), 21, 200, rng)
    values = excess_a.to_numpy()
    unpaired = st._sharpe(values[idx_a]) - st._sharpe(values[idx_b])
    assert unpaired.std() > 1e-6


def test_seed_reproduces_identical_draws(paired_series):
    net_a, net_b, excess_a, excess_b = paired_series
    first = st.paired_bootstrap_differences(
        excess_a, excess_b, net_a, net_b, ("sharpe",), CFG
    )["sharpe"]
    second = st.paired_bootstrap_differences(
        excess_a, excess_b, net_a, net_b, ("sharpe",), CFG
    )["sharpe"]
    np.testing.assert_array_equal(first, second)


def test_same_length_samples_share_identical_index_matrices():
    """Two specifications over the same dates must receive the SAME
    bootstrap index matrix.

    This is what makes differences across robustness specifications
    reflect the return series rather than Monte Carlo noise.
    """
    n_obs, block, reps = 500, 21, 40
    first = st.stationary_bootstrap_indices(
        n_obs, block, reps, np.random.default_rng(12345)
    )
    second = st.stationary_bootstrap_indices(
        n_obs, block, reps, np.random.default_rng(12345)
    )
    np.testing.assert_array_equal(first, second)


def test_common_draws_across_two_different_return_series():
    """Different return data, same seed and length -> identical indices,
    so the resulting metric differences are attributable to the data."""
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.0004, 0.01, 400), index=idx)
    b = pd.Series(rng.normal(0.0002, 0.012, 400), index=idx)
    c = pd.Series(rng.normal(0.0003, 0.009, 400), index=idx)
    cfg = st.BootstrapConfig(n_replications=200, batch_size=200)

    spec_one = st.paired_bootstrap_differences(a, b, a, b, ("sharpe",), cfg)["sharpe"]
    spec_two = st.paired_bootstrap_differences(a, c, a, c, ("sharpe",), cfg)["sharpe"]
    # Same indices were used, so a shared column resampled identically:
    # verify by reproducing the index matrix and checking one leg.
    indices = st.stationary_bootstrap_indices(
        len(a), cfg.mean_block, cfg.n_replications, np.random.default_rng(cfg.seed)
    )
    sharpe_a = st._sharpe(a.to_numpy()[indices])
    reconstructed_one = sharpe_a - st._sharpe(b.to_numpy()[indices])
    reconstructed_two = sharpe_a - st._sharpe(c.to_numpy()[indices])
    np.testing.assert_allclose(spec_one, reconstructed_one, rtol=1e-12)
    np.testing.assert_allclose(spec_two, reconstructed_two, rtol=1e-12)


def test_different_seeds_give_different_draws(paired_series):
    net_a, net_b, excess_a, excess_b = paired_series
    other = st.BootstrapConfig(n_replications=400, batch_size=200, seed=999)
    a = st.paired_bootstrap_differences(excess_a, excess_b, net_a, net_b, ("sharpe",), CFG)["sharpe"]
    b = st.paired_bootstrap_differences(excess_a, excess_b, net_a, net_b, ("sharpe",), other)["sharpe"]
    assert not np.array_equal(a, b)


def test_batch_size_does_not_change_the_draw_count(paired_series):
    net_a, net_b, excess_a, excess_b = paired_series
    cfg = st.BootstrapConfig(n_replications=300, batch_size=97)
    draws = st.paired_bootstrap_differences(
        excess_a, excess_b, net_a, net_b, ("sharpe",), cfg
    )["sharpe"]
    assert len(draws) == 300


def test_mismatched_lengths_rejected(paired_series):
    net_a, net_b, excess_a, excess_b = paired_series
    with pytest.raises(ValueError, match="same length"):
        st.paired_bootstrap_differences(
            excess_a, excess_b.iloc[:-5], net_a, net_b, ("sharpe",), CFG
        )


# ---------------------------------------------------------------------------
# Estimand alignment
# ---------------------------------------------------------------------------

def test_sharpe_uses_daily_annualization():
    """The bootstrap statistic must annualize daily data by sqrt(252),
    matching the Phase 10 estimand."""
    r = np.full((1, 252), 0.001)
    r = r + np.random.default_rng(0).normal(0, 0.01, (1, 252))
    manual = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert st._sharpe(r)[0] == pytest.approx(manual, rel=1e-12)
    assert "252" in inspect.getsource(st._sharpe) or st.TRADING_DAYS == 252


def test_metric_registry_uses_correct_input_series():
    """Sharpe and Sortino consume EXCESS returns; wealth-based metrics
    consume NET returns."""
    assert st.METRIC_FUNCTIONS["sharpe"][0] == "excess"
    assert st.METRIC_FUNCTIONS["sortino"][0] == "excess"
    for metric in ("cagr", "ann_volatility", "max_drawdown", "calmar"):
        assert st.METRIC_FUNCTIONS[metric][0] == "net"


def test_path_dependent_metrics_flagged():
    assert st.PATH_DEPENDENT == {"max_drawdown", "calmar"}


# ---------------------------------------------------------------------------
# Intervals and the centered null
# ---------------------------------------------------------------------------

def test_percentile_interval_brackets_the_draws():
    draws = np.random.default_rng(0).normal(0.02, 0.05, 10_000)
    low, high = st.percentile_interval(draws, 0.95)
    assert low < 0.02 < high
    assert low == pytest.approx(np.quantile(draws, 0.025), rel=1e-12)
    assert high == pytest.approx(np.quantile(draws, 0.975), rel=1e-12)


def test_centered_null_has_approximately_zero_center():
    draws = np.random.default_rng(1).normal(0.5, 0.1, 20_000)
    null = draws - draws.mean()
    assert abs(null.mean()) < 1e-12
    # A large observed value against the centered null gives a small p.
    assert st.centered_bootstrap_pvalue(draws, 0.5, "greater") < 0.6
    # An observed value at the null center gives about one half.
    assert st.centered_bootstrap_pvalue(draws, 0.0, "greater") == pytest.approx(0.5, abs=0.02)


def test_centered_pvalue_differs_from_naive_fraction_below_zero():
    """The fraction of ordinary draws below zero is NOT the p-value."""
    draws = np.random.default_rng(2).normal(0.03, 0.02, 20_000)
    naive = float((draws < 0).mean())
    proper = st.centered_bootstrap_pvalue(draws, float(draws.mean()), "greater")
    assert proper != pytest.approx(naive, abs=1e-6)


def test_centered_pvalue_alternatives():
    draws = np.random.default_rng(3).normal(0.1, 0.05, 5_000)
    greater = st.centered_bootstrap_pvalue(draws, 0.1, "greater")
    two_sided = st.centered_bootstrap_pvalue(draws, 0.1, "two-sided")
    assert 0.0 <= greater <= 1.0
    assert two_sided == pytest.approx(2 * greater, abs=0.05)


# ---------------------------------------------------------------------------
# HAC
# ---------------------------------------------------------------------------

def test_hac_operates_on_paired_differences(paired_series):
    net_a, net_b, _, _ = paired_series
    result = st.hac_mean_difference(net_a, net_b, lags=21)
    assert result["n"] == len(net_a)
    assert result["mean_daily_difference"] == pytest.approx(
        float((net_a - net_b).mean()), rel=1e-12
    )
    assert result["annualized_arithmetic_difference"] == pytest.approx(
        result["mean_daily_difference"] * 252, rel=1e-12
    )
    assert result["ci95_lower_daily"] < result["mean_daily_difference"] < result["ci95_upper_daily"]


def test_hac_t_statistic_invariant_to_annualization(paired_series):
    """Annualizing both the estimate and the standard error by the same
    linear factor must leave the t-statistic and p-value unchanged.

    Guards against reporting an annualized mean against a daily (or
    sqrt-scaled) standard error, which would silently inflate or deflate
    significance.
    """
    from scipy import stats as sps

    net_a, net_b, _, _ = paired_series
    result = st.hac_mean_difference(net_a, net_b, lags=21)

    t_daily = result["mean_daily_difference"] / result["hac_standard_error_daily"]
    annualized_se = result["hac_standard_error_daily"] * st.TRADING_DAYS
    t_annual = result["annualized_arithmetic_difference"] / annualized_se

    assert t_daily == pytest.approx(t_annual, rel=1e-12)
    assert t_daily == pytest.approx(result["t_statistic"], rel=1e-9)

    manual_p = 2 * (1 - sps.norm.cdf(abs(t_daily)))
    assert manual_p == pytest.approx(result["p_value_two_sided"], abs=1e-6)

    # The annualized interval is the daily interval scaled by the same factor.
    assert result["ci95_lower_annualized"] == pytest.approx(
        result["ci95_lower_daily"] * st.TRADING_DAYS, rel=1e-12
    )
    assert result["ci95_upper_annualized"] == pytest.approx(
        result["ci95_upper_daily"] * st.TRADING_DAYS, rel=1e-12
    )


def test_hac_lag_choice_changes_standard_error(paired_series):
    net_a, net_b, _, _ = paired_series
    ses = {
        lags: st.hac_mean_difference(net_a, net_b, lags=lags)["hac_standard_error_daily"]
        for lags in (5, 21, 42)
    }
    assert len(set(ses.values())) > 1


def test_identical_series_give_zero_hac_difference(paired_series):
    net_a, _, _, _ = paired_series
    result = st.hac_mean_difference(net_a, net_a.copy(), lags=21)
    assert result["mean_daily_difference"] == pytest.approx(0.0, abs=1e-18)


# ---------------------------------------------------------------------------
# Scope discipline
# ---------------------------------------------------------------------------

def test_no_diebold_mariano_on_portfolio_returns():
    """DM belongs to the Phase 5 forecast comparison only; it must not
    appear in the strategy-inference module."""
    source = inspect.getsource(st)
    assert "def diebold_mariano" not in source
    functions = [name for name, _ in inspect.getmembers(st, inspect.isfunction)]
    assert not any("diebold" in name.lower() for name in functions)


def test_module_documents_pvalue_discipline():
    source = inspect.getsource(st)
    assert "centered" in source.lower()
    assert "not a hypothesis-test p-value" in source or "is *not* a hypothesis-test" in source


def test_holm_available_for_secondary_comparisons():
    adjusted = st.holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert (adjusted >= np.array([0.01, 0.04, 0.03])).all()
    assert (adjusted <= 1).all()
