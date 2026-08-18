"""Covariance estimation and conditioning diagnostics (Phase 7).

Estimators produced at every rebalance origin, all from returns through
that origin only:

* sample covariance,
* exponentially weighted covariance (RiskMetrics lambda),
* unconditional Ledoit-Wolf shrinkage,
* state-conditioned covariances (low/high) from the smoothed
  responsibilities of the HMM fit **through that origin**,
* the same state covariances shrunk toward unconditional Ledoit-Wolf
  with weight ``alpha_k = n_eff_k / (n_eff_k + threshold)`` on the state
  estimate,
* the horizon-averaged regime mixture,
* the covariance actually consumed downstream, after the Amendment A2
  degeneracy rule.

Amendment A3 fixes the mixture definition: state estimators are
**centered** covariances and the main mixture is within-state only,
``Sigma_RA = sum_k pbar_k C_k``. The complete law-of-total-covariance
version, adding ``(mu_k - mubar)(mu_k - mubar)'``, is a robustness case;
the relative size of that omitted term is recorded at every rebalance.

Amendment A2 routing is driven exclusively by the HMM guard flags. The
four dates observed to trigger it are audit information and never
appear in any routing condition.

No optimizer weights or portfolio returns are computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# PSD policy: a negative eigenvalue smaller in magnitude than
# REL_PSD_TOL * max_eigenvalue is floating-point noise -> clip and log.
# Anything larger is a genuine estimator error -> hard fail. Clipping
# must never be able to repair a materially invalid matrix.
REL_PSD_TOL = 1e-8
EIGENVALUE_FLOOR = 1e-10


class MateriallyNonPSDError(RuntimeError):
    """Raised when an estimator returns a materially non-PSD matrix."""


@dataclass(frozen=True)
class CovarianceConfig:
    ewma_lambda: float = 0.94
    neff_threshold: float = 60.0          # A1: 30/120 are robustness
    horizon_days: int = 21
    include_between_state_term: bool = False  # A3: main = within-state only


# ---------------------------------------------------------------------------
# Unconditional estimators
# ---------------------------------------------------------------------------

def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Plain sample covariance (ddof=1) of the supplied window."""
    return np.cov(returns.to_numpy(), rowvar=False, ddof=1)


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance around the weighted mean.

    Weights decay geometrically into the past with factor ``lam`` and
    are normalized to sum to one, so the estimate is a proper weighted
    covariance and PSD by construction.
    """
    X = returns.to_numpy()
    n = len(X)
    weights = lam ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    mean = weights @ X
    centered = X - mean
    return (centered * weights[:, None]).T @ centered


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance and its shrinkage intensity."""
    from sklearn.covariance import LedoitWolf

    estimator = LedoitWolf(assume_centered=False).fit(returns.to_numpy())
    return estimator.covariance_, float(estimator.shrinkage_)


# ---------------------------------------------------------------------------
# State-conditioned estimators
# ---------------------------------------------------------------------------

def effective_sample_size(responsibilities: np.ndarray) -> np.ndarray:
    """n_eff_k = (sum_s gamma_sk)^2 / sum_s gamma_sk^2, per state."""
    total = responsibilities.sum(axis=0)
    return total**2 / (responsibilities**2).sum(axis=0)


def state_conditional_moments(
    returns: pd.DataFrame, responsibilities: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Centered state covariances, state means, and effective sample sizes.

    ``responsibilities`` are the smoothed posteriors from the HMM fit
    through the rebalance date (never the full-sample ex-post series),
    aligned row-wise with ``returns``. Each covariance is centered on
    its own state mean and is PSD by construction (a weighted sum of
    outer products).
    """
    X = returns.to_numpy()
    if len(X) != len(responsibilities):
        raise ValueError(
            f"returns ({len(X)}) and responsibilities "
            f"({len(responsibilities)}) must align row-wise"
        )
    covariances: list[np.ndarray] = []
    means = []
    for k in range(responsibilities.shape[1]):
        gamma = responsibilities[:, k]
        weight_sum = gamma.sum()
        mu = (gamma[:, None] * X).sum(axis=0) / weight_sum
        centered = X - mu
        cov = (centered * gamma[:, None]).T @ centered / weight_sum
        covariances.append(cov)
        means.append(mu)
    return covariances, np.array(means), effective_sample_size(responsibilities)


def shrink_state_covariance(
    state_cov: np.ndarray,
    unconditional: np.ndarray,
    n_eff: float,
    threshold: float = 60.0,
) -> tuple[np.ndarray, float]:
    """Shrink a state covariance toward the unconditional estimate.

    ``alpha = n_eff / (n_eff + threshold)`` is the weight on the **state**
    estimate: a large effective sample keeps the state-specific
    structure, a small one moves toward unconditional Ledoit-Wolf.
    Returns ``(shrunk, alpha)``.
    """
    alpha = float(n_eff / (n_eff + threshold))
    return alpha * state_cov + (1.0 - alpha) * unconditional, alpha


# ---------------------------------------------------------------------------
# Horizon averaging and mixture
# ---------------------------------------------------------------------------

def horizon_average_probabilities(
    p_t: np.ndarray, transmat: np.ndarray, horizon: int
) -> np.ndarray:
    """Average state probability over the next ``horizon`` steps.

    ``pbar = (1/H) sum_{h=1..H} p_t P^h``. Each term is a probability
    vector, so the average is one too.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    accumulated = np.zeros_like(p_t, dtype=float)
    current = np.asarray(p_t, dtype=float)
    for _ in range(horizon):
        current = current @ transmat
        accumulated += current
    return accumulated / horizon


def regime_mixture(
    p_bar: np.ndarray,
    state_covariances: list[np.ndarray],
    state_means: np.ndarray,
    include_between_state_term: bool = False,
) -> tuple[np.ndarray, float]:
    """Mixture covariance and the relative size of the between-state term.

    Main specification (Amendment A3) is within-state only. When
    ``include_between_state_term`` is True the complete law of total
    covariance is returned instead. The second return value is always
    ``||between|| / ||within||`` (Frobenius), so the magnitude of the
    omitted term is recorded whichever formula is used.
    """
    within = sum(w * cov for w, cov in zip(p_bar, state_covariances))
    mu_bar = p_bar @ state_means
    between = sum(
        w * np.outer(mu - mu_bar, mu - mu_bar)
        for w, mu in zip(p_bar, state_means)
    )
    within_norm = np.linalg.norm(within, ord="fro")
    relative = float(np.linalg.norm(between, ord="fro") / within_norm) if within_norm else np.nan
    total = within + between if include_between_state_term else within
    return total, relative


# ---------------------------------------------------------------------------
# Conditioning and PSD diagnostics
# ---------------------------------------------------------------------------

def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Force exact symmetry (removes asymmetry at machine precision)."""
    return 0.5 * (matrix + matrix.T)


def condition_numbers(cov: np.ndarray) -> tuple[float, float]:
    """Condition number of the covariance and of its correlation matrix.

    The correlation condition number strips scale differences between
    assets, isolating collinearity from heterogeneous volatility.
    """
    cov_eigenvalues = np.linalg.eigvalsh(symmetrize(cov))
    cov_condition = float(cov_eigenvalues.max() / cov_eigenvalues.min()) if cov_eigenvalues.min() > 0 else np.inf
    scale = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(scale, scale)
    corr = symmetrize(np.nan_to_num(corr, nan=0.0))
    corr_eigenvalues = np.linalg.eigvalsh(corr)
    corr_condition = float(corr_eigenvalues.max() / corr_eigenvalues.min()) if corr_eigenvalues.min() > 0 else np.inf
    return cov_condition, corr_condition


def enforce_psd(
    cov: np.ndarray, label: str, rel_tol: float = REL_PSD_TOL
) -> tuple[np.ndarray, dict]:
    """Validate and, if needed, minimally repair a covariance matrix.

    Floating-point noise (a negative eigenvalue smaller in magnitude
    than ``rel_tol * max_eigenvalue``) is clipped to a small positive
    floor and logged. A materially negative eigenvalue indicates a
    genuine estimator error and raises :class:`MateriallyNonPSDError`
    rather than being silently repaired.

    Returns the (possibly corrected) matrix and a diagnostics dict with
    pre- and post-correction eigenvalues and the correction magnitude.
    """
    cov = symmetrize(np.asarray(cov, dtype=float))
    if not np.all(np.isfinite(cov)):
        raise MateriallyNonPSDError(f"{label}: matrix contains non-finite entries")

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    min_before = float(eigenvalues.min())
    max_eigenvalue = float(eigenvalues.max())
    tolerance = rel_tol * max(max_eigenvalue, 0.0)

    if min_before < -tolerance:
        raise MateriallyNonPSDError(
            f"{label}: minimum eigenvalue {min_before:.3e} is materially "
            f"negative relative to the maximum {max_eigenvalue:.3e} "
            f"(tolerance {tolerance:.3e}); refusing to clip"
        )

    corrected = cov
    correction_magnitude = 0.0
    if min_before < EIGENVALUE_FLOOR:
        clipped = np.clip(eigenvalues, EIGENVALUE_FLOOR, None)
        corrected = symmetrize(eigenvectors @ np.diag(clipped) @ eigenvectors.T)
        correction_magnitude = float(np.linalg.norm(corrected - cov, ord="fro"))

    min_after = float(np.linalg.eigvalsh(corrected).min())
    cov_condition, corr_condition = condition_numbers(corrected)
    return corrected, {
        "min_eigenvalue_before": min_before,
        "min_eigenvalue_after": min_after,
        "max_eigenvalue": max_eigenvalue,
        "covariance_condition_number": cov_condition,
        "correlation_condition_number": corr_condition,
        "psd_correction_used": bool(correction_magnitude > 0),
        "psd_correction_magnitude": correction_magnitude,
    }


# ---------------------------------------------------------------------------
# Per-origin driver
# ---------------------------------------------------------------------------

def build_covariance_panel(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    hmm_cfg,
    cov_cfg: CovarianceConfig = CovarianceConfig(),
    verbose: bool = False,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, dict]:
    """Estimate every covariance object at every rebalance origin.

    The HMM is refit at each origin with the frozen protocol (identical
    seeds and selection rule as Phase 6, so the fits reproduce), and its
    **smoothed responsibilities through that origin** drive the
    state-conditioned estimates. Returns an audit frame (one row per
    estimator per date) and a dict of the consumed matrices keyed by
    date.

    Amendment A2: when the fit is degenerate, the consumed covariance is
    the unconditional Ledoit-Wolf estimate. Routing reads guard flags
    only — no date is ever consulted.

    The second return value maps each date to the matrices Phase 8 may
    consume (``consumed``, ``ledoit_wolf``, ``regime_mixture``), all
    estimated on this window so downstream strategies never re-derive
    them on a different one.
    """
    from src import regimes as rg

    audit_rows: list[dict] = []
    consumed: dict[pd.Timestamp, dict[str, np.ndarray]] = {}

    for i, t in enumerate(rebalance_dates):
        feature_window = features.loc[:t]
        return_window = returns.loc[feature_window.index[0] : t]
        X = rg.standardize_window(feature_window)

        model, records = rg.fit_multistart(X, hmm_cfg)
        perm = rg.canonical_permutation(model, list(feature_window.columns))
        responsibilities = model.predict_proba(X)[:, perm]
        transmat = model.transmat_[np.ix_(perm, perm)]
        p_t = responsibilities[-1]
        diagnostics = rg.state_diagnostics(model, X, feature_window, perm, hmm_cfg)
        routing = rg.covariance_consumption(diagnostics)

        # Responsibilities are indexed by the feature window; align the
        # return window to exactly those dates.
        aligned_returns = return_window.reindex(feature_window.index).dropna()
        if len(aligned_returns) != len(responsibilities):
            responsibilities = responsibilities[-len(aligned_returns):]

        common = {
            "date": t,
            "window_start": feature_window.index[0],
            "observations": len(aligned_returns),
            "fallback_used": routing["fallback_used"],
            "fallback_reason": routing["fallback_reason"],
        }

        estimates: dict[str, np.ndarray] = {}

        sample = sample_covariance(aligned_returns)
        estimates["sample"] = sample
        ewma = ewma_covariance(aligned_returns, cov_cfg.ewma_lambda)
        estimates["ewma"] = ewma
        lw, lw_intensity = ledoit_wolf_covariance(aligned_returns)
        estimates["ledoit_wolf"] = lw

        state_covs, state_means, n_eff = state_conditional_moments(
            aligned_returns, responsibilities
        )
        shrunk: list[np.ndarray] = []
        alphas: list[float] = []
        for k, cov_k in enumerate(state_covs):
            estimates[f"state_{k}_raw"] = cov_k
            shrunk_k, alpha_k = shrink_state_covariance(
                cov_k, lw, n_eff[k], cov_cfg.neff_threshold
            )
            shrunk.append(shrunk_k)
            alphas.append(alpha_k)
            estimates[f"state_{k}_shrunk"] = shrunk_k

        p_bar = horizon_average_probabilities(p_t, transmat, cov_cfg.horizon_days)
        mixture, between_relative = regime_mixture(
            p_bar, shrunk, state_means, cov_cfg.include_between_state_term
        )
        estimates["regime_mixture"] = mixture

        # Amendment A2: flag-driven choice of what Phase 8 may consume.
        consumed_label = (
            "ledoit_wolf" if routing["fallback_used"] else "regime_mixture"
        )
        estimates["consumed"] = estimates[consumed_label]

        for name, matrix in estimates.items():
            corrected, psd = enforce_psd(matrix, f"{t.date()}:{name}")
            if name in ("consumed", "ledoit_wolf", "regime_mixture"):
                # Exported for Phase 8 so strategies consume exactly the
                # matrices estimated here, on this window. Recomputing
                # them downstream on a different window would make the
                # A2 fallback subtly different from the rolling-LW
                # comparator it is meant to equal.
                consumed.setdefault(t, {})[name] = corrected
            state = (
                int(name.split("_")[1])
                if name.startswith("state_")
                else -1
            )
            audit_rows.append(
                {
                    **common,
                    "estimator": name,
                    "state": state,
                    "effective_sample_size": float(n_eff[state]) if state >= 0 else np.nan,
                    "shrinkage_intensity": (
                        alphas[state] if state >= 0 and name.endswith("_shrunk")
                        else (lw_intensity if name == "ledoit_wolf" else np.nan)
                    ),
                    "p_bar_high_state": float(p_bar[-1]),
                    "between_state_relative_norm": (
                        between_relative if name == "regime_mixture" else np.nan
                    ),
                    "covariance_consumed": name == "consumed" or name == consumed_label,
                    **psd,
                }
            )

        if verbose and (i % progress_every == 0):
            print(f"    origin {i + 1}/{len(rebalance_dates)}  {t.date()}  "
                  f"cond(corr)={audit_rows[-1]['correlation_condition_number']:.1f}",
                  flush=True)

    return pd.DataFrame(audit_rows), consumed
