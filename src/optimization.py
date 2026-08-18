"""Portfolio target-weight construction (Phase 8).

Six strategies forming an ablation ladder, each adding one ingredient:

1. ``equal_weight``       - 20% in each of the five ETFs
2. ``static_6040``        - 60% SPY / 40% IEF, zero elsewhere
3. ``static_minvar``      - minimum variance on the covariance estimated
                            through the training end; one fixed target
                            reused at every origin
4. ``rolling_lw_minvar``  - minimum variance on the current unconditional
                            Ledoit-Wolf matrix (the primary comparator)
5. ``ewma_scaled_minvar`` - minimum variance on EWMA volatility forecasts
                            combined with the Ledoit-Wolf *correlation*
                            matrix (never the raw, poorly conditioned
                            EWMA covariance)
6. ``regime_minvar``      - minimum variance on the Amendment
                            A2-consumed covariance

Constraints for optimized strategies: long-only, fully invested, 40%
maximum per asset. **The 60/40 benchmark is exempt from the cap** by
construction, since it deliberately holds 60% SPY; the cap governs the
optimized five-asset strategies only.

This module emits **target weights only**. It computes no portfolio
return, cost, turnover, or wealth path. When an optimization fails it
emits missing weights and ``fallback_requested = True`` rather than
inventing a substitute: reconstructing the pre-trade drifted holdings
requires subsequent asset returns, which belong to Phase 9. Phase 9
interprets a fallback request as "no trade".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Tolerances for independently validating a returned solution. SLSQP
# routinely returns weights like -3e-17; anything within these bounds is
# numerical noise and is cleaned, anything larger is treated as failure.
SUM_TOL = 1e-8
BOUND_TOL = 1e-9
CAP_BINDING_TOL = 1e-6


@dataclass(frozen=True)
class OptimizerConfig:
    max_weight: float = 0.40
    scale_by_trace: bool = True   # positive scaling leaves argmin unchanged
    max_iterations: int = 200
    ftol: float = 1e-14


def covariance_hash(cov: np.ndarray) -> str:
    """Stable short hash of a covariance matrix.

    Ties every emitted target weight to the exact matrix it came from,
    so a target can be traced back to its input without storing the
    matrix twice.
    """
    rounded = np.ascontiguousarray(np.round(np.asarray(cov, dtype=float), 18))
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Rule-based targets
# ---------------------------------------------------------------------------

def equal_weight(assets: list[str]) -> np.ndarray:
    """Equal allocation across all assets."""
    return np.full(len(assets), 1.0 / len(assets))


def fixed_weight(assets: list[str], mapping: dict[str, float]) -> np.ndarray:
    """Fixed allocation from an explicit mapping; unnamed assets get zero.

    Used for the 60/40 benchmark, which is exempt from the 40% cap.
    """
    total = sum(mapping.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"fixed weights must sum to 1, got {total}")
    unknown = set(mapping) - set(assets)
    if unknown:
        raise ValueError(f"unknown assets in mapping: {sorted(unknown)}")
    return np.array([mapping.get(a, 0.0) for a in assets])


# ---------------------------------------------------------------------------
# Covariance shaping
# ---------------------------------------------------------------------------

def ewma_scaled_covariance(
    ewma_variances: np.ndarray, lw_cov: np.ndarray
) -> np.ndarray:
    """EWMA volatility forecasts combined with Ledoit-Wolf correlations.

    ``Sigma = D R D`` with ``D = diag(sqrt(ewma_variances))`` and ``R``
    the correlation matrix implied by the Ledoit-Wolf covariance. This
    keeps the responsive EWMA volatility scale while borrowing the
    well-conditioned correlation structure, instead of using the raw
    EWMA covariance whose effective sample size is only about 32
    observations at lambda = 0.94.
    """
    scale = np.sqrt(np.diag(lw_cov))
    correlation = lw_cov / np.outer(scale, scale)
    d = np.sqrt(np.asarray(ewma_variances, dtype=float))
    return np.outer(d, d) * correlation


# ---------------------------------------------------------------------------
# Minimum-variance optimization
# ---------------------------------------------------------------------------

def validate_weights(
    weights: np.ndarray, max_weight: float
) -> tuple[np.ndarray, dict]:
    """Independently check a candidate solution against the constraints.

    Does not trust the solver's own success flag: recomputes the sum,
    the bound violations, and the cap-binding count directly. Noise-level
    violations are cleaned (clipped and renormalized) and reported;
    larger ones are surfaced through ``constraint_violation`` for the
    caller to reject.
    """
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)):
        return weights, {"constraint_violation": np.inf, "sum_weights": np.nan,
                         "minimum_weight": np.nan, "maximum_weight": np.nan,
                         "cap_binding_count": 0}

    sum_violation = abs(weights.sum() - 1.0)
    lower_violation = max(0.0, -weights.min())
    upper_violation = max(0.0, weights.max() - max_weight)
    violation = max(sum_violation, lower_violation, upper_violation)

    cleaned = weights
    if violation <= max(SUM_TOL, BOUND_TOL):
        cleaned = np.clip(weights, 0.0, max_weight)
        total = cleaned.sum()
        if total > 0:
            cleaned = cleaned / total

    return cleaned, {
        "constraint_violation": float(violation),
        "sum_weights": float(cleaned.sum()),
        "minimum_weight": float(cleaned.min()),
        "maximum_weight": float(cleaned.max()),
        "cap_binding_count": int((cleaned >= max_weight - CAP_BINDING_TOL).sum()),
    }


def min_variance_weights(
    cov: np.ndarray, cfg: OptimizerConfig = OptimizerConfig()
) -> tuple[np.ndarray | None, dict]:
    """Long-only, fully invested minimum-variance weights via SLSQP.

    The covariance is scaled by its trace before optimization because a
    raw daily covariance makes the objective ~1e-4 and starves the
    convergence test; positive scaling leaves the argmin unchanged. The
    reported ``objective_value`` is on the **unscaled** matrix so it is
    comparable across dates.

    Returns ``(weights, diagnostics)``; weights are ``None`` when the
    solution fails independent validation, in which case
    ``fallback_requested`` is True and Phase 9 treats it as no trade.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    scale = float(np.trace(cov)) / n if cfg.scale_by_trace else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    scaled = cov / scale

    def objective(w: np.ndarray) -> float:
        return float(w @ scaled @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * scaled @ w

    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, cfg.max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": cfg.max_iterations, "ftol": cfg.ftol},
    )

    cleaned, checks = validate_weights(result.x, cfg.max_weight)
    acceptable = (
        result.success
        and checks["constraint_violation"] <= max(SUM_TOL, BOUND_TOL)
        and np.all(np.isfinite(cleaned))
    )
    diagnostics = {
        "solver_success": bool(result.success),
        "solver_status": str(result.message)[:120],
        "iterations": int(result.nit),
        "objective_value": float(cleaned @ cov @ cleaned) if acceptable else np.nan,
        "covariance_hash": covariance_hash(cov),
        "fallback_requested": not acceptable,
        **checks,
    }
    return (cleaned if acceptable else None), diagnostics


# ---------------------------------------------------------------------------
# Strategy panel
# ---------------------------------------------------------------------------

RULE_BASED = {"equal_weight", "static_6040"}
OPTIMIZED = {"static_minvar", "rolling_lw_minvar", "ewma_scaled_minvar", "regime_minvar"}
STRATEGIES = ["equal_weight", "static_6040", "static_minvar",
              "rolling_lw_minvar", "ewma_scaled_minvar", "regime_minvar"]


def build_strategy_targets(
    assets: list[str],
    rebalance_dates: pd.DatetimeIndex,
    static_cov: np.ndarray,
    rolling_lw: dict[pd.Timestamp, np.ndarray],
    ewma_variances: dict[pd.Timestamp, np.ndarray],
    consumed_cov: dict[pd.Timestamp, np.ndarray],
    benchmark_mapping: dict[str, float],
    cfg: OptimizerConfig = OptimizerConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target weights and optimizer audit for all six strategies.

    Returns ``(weights_long, audit)``. Weights are emitted in long
    format (date, strategy, asset, weight); a failed optimization emits
    **no weight rows** for that date and strategy, with the failure
    recorded in the audit as ``fallback_requested = True``.
    """
    static_weights, static_diag = min_variance_weights(static_cov, cfg)
    if static_weights is None:
        raise RuntimeError(
            "static minimum-variance optimization failed on the training "
            f"window: {static_diag['solver_status']}"
        )

    weight_rows: list[dict] = []
    audit_rows: list[dict] = []

    for t in rebalance_dates:
        lw = rolling_lw[t]
        targets: dict[str, tuple[np.ndarray | None, dict, str]] = {
            "equal_weight": (
                equal_weight(assets),
                {"covariance_source": "none"},
                "none",
            ),
            "static_6040": (
                fixed_weight(assets, benchmark_mapping),
                {"covariance_source": "none"},
                "none",
            ),
        }

        weights, diag = static_weights, dict(static_diag)
        targets["static_minvar"] = (weights, diag, "training_window_ledoit_wolf")

        weights, diag = min_variance_weights(lw, cfg)
        targets["rolling_lw_minvar"] = (weights, diag, "rolling_ledoit_wolf")

        scaled_cov = ewma_scaled_covariance(ewma_variances[t], lw)
        weights, diag = min_variance_weights(scaled_cov, cfg)
        targets["ewma_scaled_minvar"] = (weights, diag, "ewma_vol_x_lw_correlation")

        weights, diag = min_variance_weights(consumed_cov[t], cfg)
        targets["regime_minvar"] = (weights, diag, "a2_consumed")

        for strategy in STRATEGIES:
            weights, diag, source = targets[strategy]
            row = {"date": t, "strategy": strategy, "covariance_source": source}
            if strategy in RULE_BASED:
                checks = {
                    "sum_weights": float(weights.sum()),
                    "minimum_weight": float(weights.min()),
                    "maximum_weight": float(weights.max()),
                    "constraint_violation": float(abs(weights.sum() - 1.0)),
                    "cap_binding_count": 0,
                }
                row.update(
                    solver_success=True, solver_status="rule_based",
                    iterations=0, objective_value=np.nan,
                    covariance_hash="", fallback_requested=False, **checks,
                )
            else:
                row.update({k: v for k, v in diag.items() if k != "covariance_source"})

            audit_rows.append(row)
            if weights is not None:
                for asset, w in zip(assets, weights):
                    weight_rows.append(
                        {"date": t, "strategy": strategy, "asset": asset,
                         "weight": float(w)}
                    )

    return pd.DataFrame(weight_rows), pd.DataFrame(audit_rows)
