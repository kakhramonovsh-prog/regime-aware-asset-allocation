"""Tests for Phase 7 covariance estimation.

Covers every required property: causality, symmetry/finiteness, horizon
probability normalization, responsibility provenance, shrinkage
direction, the mixture formula against a manual calculation, the
Amendment A2 fallback (flag-driven, exactly four on the frozen
snapshot, no rebalance deleted), and the rule that eigenvalue clipping
cannot silently repair a materially invalid matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import covariance as cv
from src import preprocessing as prep
from src import regimes as rg

CFG = cv.CovarianceConfig()


@pytest.fixture(scope="module")
def window(prices) -> pd.DataFrame:
    r = prep.log_returns(prices)
    r.columns = ["SPY", "IEF", "GLD"]
    return r


@pytest.fixture(scope="module")
def responsibilities(window) -> np.ndarray:
    """Two-state responsibilities with a clear high-vol stretch."""
    n = len(window)
    gamma_high = np.zeros(n)
    gamma_high[150:250] = 0.9
    gamma_high[np.r_[0:150, 250:n]] = 0.05
    return np.column_stack([1 - gamma_high, gamma_high])


# ---------------------------------------------------------------------------
# Unconditional estimators
# ---------------------------------------------------------------------------

def test_sample_covariance_matches_numpy(window):
    cov = cv.sample_covariance(window)
    np.testing.assert_allclose(
        cov, np.cov(window.to_numpy(), rowvar=False, ddof=1), rtol=1e-12
    )


def test_ewma_covariance_is_psd_and_weighted(window):
    cov = cv.ewma_covariance(window, lam=0.94)
    assert np.linalg.eigvalsh(cov).min() >= -1e-15
    np.testing.assert_allclose(cov, cov.T, atol=1e-18)
    # Higher lambda (slower decay) differs from a short memory.
    fast = cv.ewma_covariance(window, lam=0.80)
    assert not np.allclose(cov, fast)


def test_ewma_effective_sample_size_formula():
    """The weight-based n_eff for geometric weights is (1+l)/(1-l).

    Pins the corrected figure used in the Phase 7 report: at lambda 0.94
    n_eff is ~32.3 with an ~11.2-day half-life, NOT the 1/(1-lambda)
    ~16.7 decay heuristic.
    """
    lam = 0.94
    weights = lam ** np.arange(100_000)
    weights /= weights.sum()
    n_eff = weights.sum() ** 2 / (weights**2).sum()
    assert n_eff == pytest.approx((1 + lam) / (1 - lam), rel=1e-6)
    assert n_eff == pytest.approx(32.33, abs=0.05)
    half_life = np.log(0.5) / np.log(lam)
    assert half_life == pytest.approx(11.2, abs=0.1)


def test_a1_shrinkage_thresholds_are_not_symmetric():
    """Larger kappa means MORE shrinkage toward Ledoit-Wolf.

    Pins the corrected Phase 7 statement: at the minimum observed
    n_eff the kappa=120 case is materially different from kappa=60.
    """
    n_eff = 436.0
    alphas = {k: n_eff / (n_eff + k) for k in (30, 60, 120)}
    assert alphas[30] > alphas[60] > alphas[120]
    assert alphas[30] == pytest.approx(0.9356, abs=1e-4)
    assert alphas[60] == pytest.approx(0.8790, abs=1e-4)
    assert alphas[120] == pytest.approx(0.7842, abs=1e-4)
    # The 60 -> 120 move is a material change, not a negligible one.
    assert alphas[60] - alphas[120] > 0.09


def test_ledoit_wolf_is_well_conditioned(window):
    lw, intensity = cv.ledoit_wolf_covariance(window)
    sample = cv.sample_covariance(window)
    assert 0.0 <= intensity <= 1.0
    lw_cond, _ = cv.condition_numbers(lw)
    sample_cond, _ = cv.condition_numbers(sample)
    assert lw_cond <= sample_cond  # shrinkage improves conditioning


# ---------------------------------------------------------------------------
# State-conditioned estimation
# ---------------------------------------------------------------------------

def test_effective_sample_size_bounds(responsibilities):
    n_eff = cv.effective_sample_size(responsibilities)
    assert (n_eff > 0).all()
    assert (n_eff <= len(responsibilities)).all()
    # Hard 0/1 responsibilities give n_eff equal to the count.
    hard = np.zeros((100, 2))
    hard[:30, 1] = 1.0
    hard[30:, 0] = 1.0
    np.testing.assert_allclose(cv.effective_sample_size(hard), [70, 30])


def test_state_covariances_are_centered_and_psd(window, responsibilities):
    covs, means, n_eff = cv.state_conditional_moments(window, responsibilities)
    assert len(covs) == 2
    assert means.shape == (2, window.shape[1])
    for cov in covs:
        np.testing.assert_allclose(cov, cov.T, atol=1e-18)
        assert np.linalg.eigvalsh(cov).min() >= -1e-15
    # Manual check of state 1 against an explicit weighted computation.
    gamma = responsibilities[:, 1]
    X = window.to_numpy()
    mu = (gamma[:, None] * X).sum(axis=0) / gamma.sum()
    centered = X - mu
    manual = (centered * gamma[:, None]).T @ centered / gamma.sum()
    np.testing.assert_allclose(covs[1], manual, rtol=1e-12)
    np.testing.assert_allclose(means[1], mu, rtol=1e-12)


def test_state_moments_reject_misaligned_inputs(window):
    with pytest.raises(ValueError, match="align row-wise"):
        cv.state_conditional_moments(window, np.ones((len(window) - 5, 2)))


def test_shrinkage_direction_and_limits(window, responsibilities):
    covs, _, _ = cv.state_conditional_moments(window, responsibilities)
    lw, _ = cv.ledoit_wolf_covariance(window)
    state = covs[1]

    # Large effective sample -> weight on the state estimate.
    big, alpha_big = cv.shrink_state_covariance(state, lw, n_eff=6000, threshold=60)
    assert alpha_big > 0.98
    assert np.linalg.norm(big - state) < np.linalg.norm(big - lw)

    # Small effective sample -> move toward unconditional Ledoit-Wolf.
    small, alpha_small = cv.shrink_state_covariance(state, lw, n_eff=6, threshold=60)
    assert alpha_small < 0.10
    assert np.linalg.norm(small - lw) < np.linalg.norm(small - state)

    # Monotone in n_eff, and exactly the documented formula.
    assert alpha_small < alpha_big
    assert alpha_big == pytest.approx(6000 / (6000 + 60))
    # n_eff == threshold gives an even split.
    _, alpha_even = cv.shrink_state_covariance(state, lw, n_eff=60, threshold=60)
    assert alpha_even == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Horizon averaging and mixture
# ---------------------------------------------------------------------------

def test_horizon_probabilities_sum_to_one():
    transmat = np.array([[0.97, 0.03], [0.10, 0.90]])
    for p_t in ([1.0, 0.0], [0.0, 1.0], [0.4, 0.6]):
        for horizon in (1, 5, 21, 63):
            p_bar = cv.horizon_average_probabilities(np.array(p_t), transmat, horizon)
            assert p_bar.sum() == pytest.approx(1.0, abs=1e-12)
            assert (p_bar >= 0).all()


def test_horizon_one_step_matches_single_multiplication():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    p_t = np.array([0.7, 0.3])
    np.testing.assert_allclose(
        cv.horizon_average_probabilities(p_t, transmat, 1), p_t @ transmat
    )


def test_horizon_rejects_zero():
    with pytest.raises(ValueError, match="horizon"):
        cv.horizon_average_probabilities(np.array([0.5, 0.5]), np.eye(2), 0)


def test_mixture_matches_manual_calculation():
    p_bar = np.array([0.7, 0.3])
    c0 = np.array([[4.0, 1.0], [1.0, 3.0]])
    c1 = np.array([[9.0, 2.0], [2.0, 8.0]])
    means = np.array([[0.01, 0.02], [-0.03, 0.05]])

    within, relative = cv.regime_mixture(p_bar, [c0, c1], means, False)
    np.testing.assert_allclose(within, 0.7 * c0 + 0.3 * c1, rtol=1e-14)

    total, relative_total = cv.regime_mixture(p_bar, [c0, c1], means, True)
    mu_bar = 0.7 * means[0] + 0.3 * means[1]
    between = (
        0.7 * np.outer(means[0] - mu_bar, means[0] - mu_bar)
        + 0.3 * np.outer(means[1] - mu_bar, means[1] - mu_bar)
    )
    np.testing.assert_allclose(total, within + between, rtol=1e-14)
    assert relative == pytest.approx(relative_total)
    assert relative == pytest.approx(
        np.linalg.norm(between, "fro") / np.linalg.norm(within, "fro")
    )


def test_mixture_between_term_vanishes_with_equal_means():
    p_bar = np.array([0.5, 0.5])
    c = np.eye(2)
    equal_means = np.array([[0.01, 0.02], [0.01, 0.02]])
    total, relative = cv.regime_mixture(p_bar, [c, c], equal_means, True)
    np.testing.assert_allclose(total, c, atol=1e-18)
    assert relative == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# PSD policy
# ---------------------------------------------------------------------------

def test_psd_passthrough_for_valid_matrix(window):
    cov = cv.sample_covariance(window)
    corrected, diag = cv.enforce_psd(cov, "test")
    assert diag["psd_correction_used"] is False
    assert diag["psd_correction_magnitude"] == 0.0
    assert diag["min_eigenvalue_before"] > 0
    np.testing.assert_allclose(corrected, cv.symmetrize(cov), atol=1e-18)


def test_psd_clips_floating_point_noise():
    eigenvectors = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    eigenvalues = np.array([1.0, 0.5, -1e-17])   # noise-level negative
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    corrected, diag = cv.enforce_psd(cov, "noise")
    assert diag["min_eigenvalue_before"] < 0
    assert diag["min_eigenvalue_after"] >= 0
    assert diag["psd_correction_used"] is True
    # The repair lifts the offending eigenvalue to the floor, so the
    # correction is of that order and negligible against the matrix.
    assert diag["psd_correction_magnitude"] <= 10 * cv.EIGENVALUE_FLOOR
    assert diag["psd_correction_magnitude"] / np.linalg.norm(cov, "fro") < 1e-9


def test_psd_hard_fails_on_materially_negative_eigenvalue():
    """Clipping must not be able to repair a genuinely invalid matrix."""
    eigenvectors = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    eigenvalues = np.array([1.0, 0.5, -0.3])     # materially negative
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    with pytest.raises(cv.MateriallyNonPSDError, match="materially"):
        cv.enforce_psd(cov, "broken")


def test_psd_rejects_non_finite():
    cov = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(cv.MateriallyNonPSDError, match="non-finite"):
        cv.enforce_psd(cov, "nan")


def test_condition_numbers_reported_for_cov_and_corr():
    # Same correlation structure, wildly different scales.
    corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    scale = np.diag([1.0, 100.0])
    cov = scale @ corr @ scale
    cov_cond, corr_cond = cv.condition_numbers(cov)
    assert cov_cond > corr_cond          # scale inflates covariance conditioning
    assert corr_cond == pytest.approx((1 + 0.9) / (1 - 0.9), rel=1e-9)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def test_estimators_use_only_returns_through_t(window):
    cutoff = window.index[300]
    truncated = window.loc[:cutoff]
    perturbed = window.copy()
    future = perturbed.index > cutoff
    rng = np.random.default_rng(2)
    perturbed.loc[future] += rng.uniform(0.5, 1.0, (int(future.sum()), perturbed.shape[1]))

    for estimator in (cv.sample_covariance, cv.ewma_covariance):
        np.testing.assert_allclose(
            estimator(truncated), estimator(perturbed.loc[:cutoff]), rtol=1e-12
        )
    lw_a, _ = cv.ledoit_wolf_covariance(truncated)
    lw_b, _ = cv.ledoit_wolf_covariance(perturbed.loc[:cutoff])
    np.testing.assert_allclose(lw_a, lw_b, rtol=1e-12)


def test_state_moments_causal_under_future_perturbation(window, responsibilities):
    cutoff_position = 300
    truncated_returns = window.iloc[:cutoff_position]
    truncated_gamma = responsibilities[:cutoff_position]

    perturbed = window.copy()
    rng = np.random.default_rng(3)
    perturbed.iloc[cutoff_position:] += rng.uniform(
        0.5, 1.0, (len(window) - cutoff_position, window.shape[1])
    )

    a = cv.state_conditional_moments(truncated_returns, truncated_gamma)
    b = cv.state_conditional_moments(perturbed.iloc[:cutoff_position], truncated_gamma)
    for cov_a, cov_b in zip(a[0], b[0]):
        np.testing.assert_allclose(cov_a, cov_b, rtol=1e-12)


# ---------------------------------------------------------------------------
# Matrix hygiene across the panel
# ---------------------------------------------------------------------------

def test_all_matrices_symmetric_and_finite(window, responsibilities):
    covs, means, n_eff = cv.state_conditional_moments(window, responsibilities)
    lw, _ = cv.ledoit_wolf_covariance(window)
    shrunk = [cv.shrink_state_covariance(c, lw, n, 60)[0] for c, n in zip(covs, n_eff)]
    p_bar = np.array([0.6, 0.4])
    mixture, _ = cv.regime_mixture(p_bar, shrunk, means, False)
    for matrix in [*covs, *shrunk, lw, mixture]:
        assert np.all(np.isfinite(matrix))
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-15)
        assert np.linalg.eigvalsh(cv.symmetrize(matrix)).min() > -1e-12
