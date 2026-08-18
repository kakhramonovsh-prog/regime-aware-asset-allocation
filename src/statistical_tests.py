"""Statistical inference for strategy comparison (Phase 11).

Frozen primary specification:

* stationary bootstrap (Politis-Romano), 10,000 replications, seed
  12345, **mean block length 21 trading days**,
* **paired** resampling: both strategies receive the *same* bootstrap
  indices in every replication, preserving contemporaneous pairing,
* **daily excess returns** annualized by sqrt(252) — the same estimand
  Phase 10 reported. Resampling ~199 monthly returns and presenting the
  interval around a daily Sharpe would change the estimand and is not
  done,
* primary cost 10 bps; two-sided 95% percentile confidence interval.

Block lengths 10, 42, and 63 are sensitivity checks and can never
replace the 21-day primary result.

**p-value discipline.** The fraction of ordinary bootstrap draws below
zero is *not* a hypothesis-test p-value and is never labeled as one.
When a one-sided p-value is reported it comes from the **centered**
bootstrap null ``dSR_b^0 = dSR_b - dSR_hat``, against which the observed
difference is compared.

**Diebold-Mariano is deliberately absent.** DM compares forecast losses
(Phase 5) and is not a valid test of whether one investment strategy
outperformed another.

Only regime-aware vs rolling Ledoit-Wolf at 10 bps is confirmatory.
Everything else is secondary or robustness, and Holm adjustment applies
if formal p-values are reported across those comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class BootstrapConfig:
    n_replications: int = 10_000
    seed: int = 12345
    mean_block: int = 21               # primary
    sensitivity_blocks: tuple[int, ...] = (10, 42, 63)
    confidence: float = 0.95
    batch_size: int = 500              # memory control only; no effect on draws


# ---------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(
    n_obs: int, mean_block: int, n_replications: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, shape (reps, n_obs).

    Each step continues the current block with probability
    ``1 - 1/mean_block`` and otherwise jumps to a fresh uniform start;
    blocks wrap circularly. Geometric block lengths make the resampled
    series stationary, which matters for serially dependent returns.
    """
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    p = 1.0 / mean_block
    positions = np.arange(n_obs)

    new_block = rng.random((n_replications, n_obs)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n_obs, size=(n_replications, n_obs))

    last_start_position = np.maximum.accumulate(
        np.where(new_block, positions, -1), axis=1
    )
    start_value = np.take_along_axis(starts, last_start_position, axis=1)
    offset = positions[None, :] - last_start_position
    return (start_value + offset) % n_obs


# ---------------------------------------------------------------------------
# Statistics computed on resampled paths
# ---------------------------------------------------------------------------

def _sharpe(excess: np.ndarray) -> np.ndarray:
    """Annualized Sharpe along the last axis of a resample matrix."""
    mean = excess.mean(axis=-1)
    sigma = excess.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigma > 0, mean / sigma * np.sqrt(TRADING_DAYS), np.nan)


def _sortino(excess: np.ndarray) -> np.ndarray:
    downside = np.minimum(excess, 0.0)
    denominator = np.sqrt((downside**2).mean(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0,
            excess.mean(axis=-1) / denominator * np.sqrt(TRADING_DAYS),
            np.nan,
        )


def _ann_volatility(returns: np.ndarray) -> np.ndarray:
    return returns.std(axis=-1, ddof=1) * np.sqrt(TRADING_DAYS)


def _cagr(returns: np.ndarray) -> np.ndarray:
    """Annualized growth of a resampled path.

    Resampled series carry no calendar, so the exponent is ``252/n``
    rather than Phase 10's calendar-day convention. Both strategies use
    the identical convention, so the *difference* — the quantity of
    interest — is unaffected.
    """
    n = returns.shape[-1]
    growth = np.exp(np.log1p(returns).sum(axis=-1))
    return growth ** (TRADING_DAYS / n) - 1.0


def _max_drawdown(returns: np.ndarray) -> np.ndarray:
    """Maximum drawdown of each resampled path (negative number).

    Path-dependent: resampling reorders history, so bootstrap intervals
    for this statistic describe a different chronology than the realized
    one and must be read cautiously.
    """
    wealth = np.cumprod(1.0 + returns, axis=-1)
    running_max = np.maximum.accumulate(wealth, axis=-1)
    return (wealth / running_max - 1.0).min(axis=-1)


def _calmar(returns: np.ndarray) -> np.ndarray:
    drawdown = _max_drawdown(returns)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(drawdown < 0, _cagr(returns) / np.abs(drawdown), np.nan)


METRIC_FUNCTIONS = {
    "sharpe": ("excess", _sharpe),
    "sortino": ("excess", _sortino),
    "ann_volatility": ("net", _ann_volatility),
    "cagr": ("net", _cagr),
    "max_drawdown": ("net", _max_drawdown),
    "calmar": ("net", _calmar),
}
PATH_DEPENDENT = {"max_drawdown", "calmar"}


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_differences(
    excess_a: pd.Series,
    excess_b: pd.Series,
    net_a: pd.Series,
    net_b: pd.Series,
    metrics: tuple[str, ...] = ("sharpe",),
    cfg: BootstrapConfig = BootstrapConfig(),
    mean_block: int | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap paired metric differences (A minus B).

    Both strategies are resampled with **identical indices** in every
    replication, so contemporaneous pairing — and therefore the
    correlation between the two return series — is preserved.
    """
    block = mean_block or cfg.mean_block
    arrays = {
        "excess": (excess_a.to_numpy(), excess_b.to_numpy()),
        "net": (net_a.to_numpy(), net_b.to_numpy()),
    }
    n_obs = len(excess_a)
    for name, (a, b) in arrays.items():
        if len(a) != n_obs or len(b) != n_obs:
            raise ValueError(f"{name} series must all share the same length")

    rng = np.random.default_rng(cfg.seed)
    draws: dict[str, list[np.ndarray]] = {m: [] for m in metrics}

    remaining = cfg.n_replications
    while remaining > 0:
        batch = min(cfg.batch_size, remaining)
        indices = stationary_bootstrap_indices(n_obs, block, batch, rng)
        for metric in metrics:
            kind, function = METRIC_FUNCTIONS[metric]
            a, b = arrays[kind]
            # Identical indices for both columns: pairing preserved.
            draws[metric].append(function(a[indices]) - function(b[indices]))
        remaining -= batch

    return {metric: np.concatenate(values) for metric, values in draws.items()}


def percentile_interval(
    draws: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided percentile confidence interval."""
    alpha = (1.0 - confidence) / 2.0
    finite = draws[np.isfinite(draws)]
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )


def centered_bootstrap_pvalue(
    draws: np.ndarray, observed: float, alternative: str = "greater"
) -> float:
    """One-sided p-value from the **centered** bootstrap null.

    The null distribution is ``dSR_b - dSR_hat``, which imposes a zero
    difference while retaining the bootstrap's shape and dependence. The
    p-value is the probability that a draw from that null is at least as
    extreme as the observed statistic. The fraction of *ordinary* draws
    below zero is a different quantity and is never used here.
    """
    finite = draws[np.isfinite(draws)]
    null = finite - np.mean(finite)
    if alternative == "greater":
        return float((null >= observed).mean())
    if alternative == "less":
        return float((null <= observed).mean())
    return float((np.abs(null) >= abs(observed)).mean())


# ---------------------------------------------------------------------------
# HAC inference on mean return differences
# ---------------------------------------------------------------------------

def hac_mean_difference(
    net_a: pd.Series, net_b: pd.Series, lags: int = 21, confidence: float = 0.95
) -> dict:
    """Newey-West inference on the paired daily net return difference.

    Tests whether mean returns differ, which is a different question
    from whether Sharpe ratios differ; both are reported separately.
    """
    import statsmodels.api as sm

    difference = (net_a - net_b).dropna()
    model = sm.OLS(difference.to_numpy(), np.ones(len(difference))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    mean_daily = float(difference.mean())
    se = float(model.bse[0])
    z = 1.959963984540054 if confidence == 0.95 else float(
        __import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + confidence / 2)
    )
    return {
        "hac_lags": lags,
        "n": int(len(difference)),
        "mean_daily_difference": mean_daily,
        "annualized_arithmetic_difference": mean_daily * TRADING_DAYS,
        "hac_standard_error_daily": se,
        "ci95_lower_daily": mean_daily - z * se,
        "ci95_upper_daily": mean_daily + z * se,
        "ci95_lower_annualized": (mean_daily - z * se) * TRADING_DAYS,
        "ci95_upper_annualized": (mean_daily + z * se) * TRADING_DAYS,
        "t_statistic": float(model.tvalues[0]),
        "p_value_two_sided": float(model.pvalues[0]),
    }


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (shared with Phase 5)."""
    from src.volatility import holm_adjust as _holm

    return _holm(p_values)
