"""Latent volatility-state identification via Hidden Markov Models (Phase 6).

Frozen specification (config/analysis_plan.yaml, docs/research_design.md §5):

* Features: SPY daily log return, backward-looking 21-day realized
  volatility, one-day-lagged log VIX, one-day-lagged yield slope.
* Standardization uses the estimation window only (expanding z-score
  through the refit date); full-sample standardization is look-ahead.
* Two-state Gaussian HMM, expanding window, refit at every month-end,
  16 predetermined initializations (seeds 42-57), the fit with the
  highest training log-likelihood is selected.
* States are canonically relabeled by training-only state-conditional
  mean realized volatility: state 0 = lower volatility.
* The trading signal at ``t`` is the **filtered** probability
  ``P(S_t = k | F_t)``: the posterior at the final observation of the
  sample truncated at ``t``, where forward-backward smoothing and
  forward filtering coincide because no observation follows the
  endpoint. Full-sample smoothed paths are computed separately, stored
  in their own file, and are ex-post descriptive only.

Responsibility policy (three distinct uses, never conflated):

1. **Trading signal at t** — the endpoint filtered probability, and
   only that. This is what a strategy may condition on.
2. **State-conditioned historical estimates** (Phase 7 covariances) —
   smoothed responsibilities ``P(S_s = k | F_t)`` for ``s <= t`` from
   the model fit through ``t``. Permissible because every observation
   involved is available at ``t``; they may never use data after ``t``.
3. **Ex-post description** — full-sample smoothed paths conditioning on
   the entire history including the future. Stored separately, never an
   input to any estimate or signal.

Sample boundary: the primary real-time file begins at the preregistered
first signal origin (the December 2009 month-end, executing on the
first trading day of January 2010). Earlier origins would use a shorter
training window than the frozen design specifies; if generated at all
they are exploratory and kept in a separate, clearly labeled file.
* Guards: a state whose training occupancy falls below
  ``min_occupancy`` is flagged degenerate; non-convergence, singular
  state covariances, and absorbing/degenerate transition rows are
  recorded per refit and cannot pass silently.

Nothing here estimates a covariance for allocation, optimizes weights,
or computes a portfolio return: those are Phases 7-9.

Implementation decisions (pre-estimation, NOT preregistered; same
disclosure convention as Phase 5) are collected by
:func:`implementation_parameters` for verbatim inclusion in the report.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import preprocessing as prep

FEATURE_NAMES = ["mkt_log_return", "realized_vol_21d", "log_vix_lag1", "yield_slope_lag1"]
REALIZED_VOL_FEATURE = "realized_vol_21d"


@dataclass(frozen=True)
class HMMConfig:
    """Estimation settings.

    ``n_states``, ``n_init``, ``seed_start``, and ``min_occupancy`` come
    from the frozen plan. The remainder (``covariance_type``,
    ``n_iter``, ``tol``, ``min_covar``, ``min_eigenvalue``,
    ``selection_tol_rel``) are pre-estimation implementation decisions
    and are reported as such.
    """

    n_states: int = 2
    n_init: int = 16
    seed_start: int = 42
    min_occupancy: float = 0.05
    min_observations: int = 252
    covariance_type: str = "full"
    n_iter: int = 200
    tol: float = 1e-4
    min_covar: float = 1e-3           # hmmlearn default variance floor
    min_eigenvalue: float = 1e-8      # singular-covariance guard
    selection_tol_rel: float = 1e-10  # near-tie band for fit selection


def implementation_parameters(
    cfg: HMMConfig, features: pd.DataFrame, first_origin: str
) -> pd.DataFrame:
    """Every estimation setting, with preregistration provenance.

    Written to the outputs so the report cannot silently omit a knob.
    ``source`` distinguishes values fixed in the frozen plan from
    pre-estimation implementation decisions.
    """
    rv = features[REALIZED_VOL_FEATURE]
    rows = [
        ("n_states", cfg.n_states, "preregistered"),
        ("n_initializations", cfg.n_init, "preregistered"),
        ("init_seeds", f"{cfg.seed_start}..{cfg.seed_start + cfg.n_init - 1}", "preregistered"),
        ("selection_rule", "max training log-likelihood", "preregistered"),
        ("min_state_occupancy", cfg.min_occupancy, "preregistered"),
        ("estimation_scheme", "expanding window, monthly refit", "preregistered"),
        ("feature_order", ", ".join(features.columns), "preregistered"),
        ("standardization", "expanding z-score, estimation window only", "preregistered"),
        ("macro_signal_lag_days", 1, "preregistered"),
        ("trading_signal", "endpoint filtered probability only", "preregistered"),
        ("first_signal_origin", first_origin, "preregistered"),
        ("covariance_type", cfg.covariance_type, "implementation decision"),
        ("n_iter", cfg.n_iter, "implementation decision"),
        ("convergence_tol", cfg.tol, "implementation decision"),
        ("min_covar", cfg.min_covar, "implementation decision"),
        ("initialization", "hmmlearn default (kmeans means, uniform start/trans)",
         "implementation decision"),
        ("selection_tol_rel", cfg.selection_tol_rel, "implementation decision"),
        ("selection_tie_break", "lowest seed within tolerance", "implementation decision"),
        ("min_eigenvalue_guard", cfg.min_eigenvalue, "implementation decision"),
        ("absorbing_state_threshold", "diag(P) >= 1 - 1e-12", "implementation decision"),
        ("n_eff_formula", "(sum gamma)^2 / sum(gamma^2)", "implementation decision"),
        ("failure_rule",
         "non-usable starts excluded; RuntimeError if all fail (no silent fallback)",
         "implementation decision"),
        ("guard_rule",
         "occupancy/singular-covariance/absorbing flags recorded per refit",
         "implementation decision"),
        ("min_realized_vol_observed", f"{rv.min():.6g}", "data property"),
        ("zero_realized_vol_days", int((rv <= 0).sum()), "data property"),
        ("near_zero_rv_handling",
         "none required: realized vol is strictly positive in-sample "
         "(no log transform is applied to it)", "implementation decision"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "source"])


def rebalance_origins(
    index: pd.DatetimeIndex,
    first_signal_after: str,
    min_observations: int = 252,
) -> pd.DatetimeIndex:
    """Month-end signal origins from the preregistered training end.

    The first origin is the last trading day of the month containing
    (or following) ``first_signal_after`` — for the frozen design,
    2009-12-31, whose execution falls on the first trading day of
    January 2010. Starting earlier would use a shorter training window
    than the preregistration specifies, so this function, not the
    minimum-observation rule, defines the primary sample.
    """
    from src.volatility import month_end_rebalance_dates

    origins = month_end_rebalance_dates(index, min_observations)
    cutoff = pd.Timestamp(first_signal_after)
    selected = origins[origins >= cutoff.replace(day=1)]
    if len(selected) == 0:
        raise ValueError(f"no rebalance origins on or after {first_signal_after}")
    return selected


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def build_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    market_ticker: str = "SPY",
    vol_window: int = 21,
    macro_lag_days: int = 1,
) -> pd.DataFrame:
    """Assemble the HMM feature panel, all backward-looking.

    Price-derived features use information through the close of ``t``;
    macro features are lagged ``macro_lag_days`` trading days per the
    frozen availability policy. **Forward-looking realized volatility
    never appears here** — it exists only as an evaluation target
    elsewhere in the project.
    """
    returns = prep.log_returns(prices)
    features = pd.DataFrame(
        {
            "mkt_log_return": returns[market_ticker],
            "realized_vol_21d": prep.rolling_volatility(
                returns[market_ticker], window=vol_window
            ),
            "log_vix_lag1": prep.apply_signal_lag(
                np.log(macro["VIXCLS"]), lag_days=macro_lag_days
            ),
            "yield_slope_lag1": prep.apply_signal_lag(
                prep.yield_curve_slope(macro), lag_days=macro_lag_days
            ),
        }
    ).dropna()
    features.index.name = "Date"
    return features[FEATURE_NAMES]


def standardize_window(window: pd.DataFrame) -> np.ndarray:
    """Z-score a feature window using only that window's moments.

    Called with data through the refit date, so the scaling itself
    contains no future information.
    """
    values = window.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    return (values - mean) / std


# ---------------------------------------------------------------------------
# Fitting with multiple starts
# ---------------------------------------------------------------------------

def fit_multistart(
    X: np.ndarray, cfg: HMMConfig
) -> tuple[object, list[dict]]:
    """Fit ``cfg.n_init`` HMMs from predetermined seeds; keep the best.

    Returns ``(selected_model, records)`` where ``records`` has one
    dict per initialization including failures. Selection maximizes
    training log-likelihood among fits that both converged and produced
    finite parameters. Raises ``RuntimeError`` if every start failed —
    a silent fallback would hide a broken estimation.
    """
    from hmmlearn.hmm import GaussianHMM

    records: list[dict] = []
    candidates: list[tuple[float, object]] = []

    for seed in range(cfg.seed_start, cfg.seed_start + cfg.n_init):
        record = {"seed": seed, "converged": False, "loglik": np.nan,
                  "n_iter_run": np.nan, "error": "", "usable": False}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=cfg.n_states,
                    covariance_type=cfg.covariance_type,
                    n_iter=cfg.n_iter,
                    tol=cfg.tol,
                    min_covar=cfg.min_covar,
                    random_state=seed,
                )
                model.fit(X)
                loglik = float(model.score(X))
            converged = bool(model.monitor_.converged)
            finite = (
                np.isfinite(loglik)
                and np.all(np.isfinite(model.means_))
                and np.all(np.isfinite(model.transmat_))
            )
            record.update(
                converged=converged,
                loglik=loglik,
                n_iter_run=len(model.monitor_.history),
                usable=bool(converged and finite),
            )
            if record["usable"]:
                candidates.append((loglik, model))
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        records.append(record)

    if not candidates:
        raise RuntimeError(
            "all HMM initializations failed or did not converge; "
            "refusing to return a fit (see initialization records)"
        )

    # Tolerance-based selection. Log-likelihoods from mathematically
    # identical fits can differ at ~1e-13 because BLAS summation order
    # is not bit-reproducible, and that difference can vary across
    # machines and thread counts. A plain argmax would therefore let an
    # irrelevant numerical wobble decide which fit is used. Instead:
    # collect every fit within a relative tolerance of the best
    # log-likelihood and take the LOWEST SEED among them, which is
    # stable across machines by construction.
    usable_indices = [i for i, r in enumerate(records) if r["usable"]]
    logliks = np.array([records[i]["loglik"] for i in usable_indices])
    best_loglik = float(logliks.max())
    tol = cfg.selection_tol_rel * max(1.0, abs(best_loglik))
    near_best = [
        idx for idx, loglik in zip(usable_indices, logliks)
        if best_loglik - loglik <= tol
    ]
    best_index = min(near_best, key=lambda idx: records[idx]["seed"])
    selected = dict(zip(usable_indices, (m for _, m in candidates)))[best_index]

    for i, record in enumerate(records):
        record["selected"] = i == best_index
        record["within_selection_tol"] = i in near_best
    return selected, records


# ---------------------------------------------------------------------------
# Canonical labeling
# ---------------------------------------------------------------------------

def canonical_permutation(model, feature_names: list[str] = FEATURE_NAMES) -> np.ndarray:
    """Order states by training-only mean realized volatility, ascending.

    Uses the state-conditional means of the standardized realized-vol
    feature, which are estimated from the training window alone. State
    0 becomes the lower-volatility state. No code path may assume raw
    hmmlearn labels carry this meaning.
    """
    idx = feature_names.index(REALIZED_VOL_FEATURE)
    return np.argsort(model.means_[:, idx])


def canonical_permutation_by_series(
    model, X: np.ndarray, series_values: np.ndarray
) -> np.ndarray:
    """Order states by responsibility-weighted mean of an external series.

    Used when realized volatility is not itself a model feature (the
    drop-realized-vol ablation). The weights are posteriors from the
    training window only, and ``series_values`` is the backward-looking
    realized-vol series over that same window, so the labeling rule
    stays "ascending training-only expected realized volatility".
    """
    posteriors = model.predict_proba(X)
    weighted = (posteriors * series_values[:, None]).sum(axis=0) / posteriors.sum(axis=0)
    return np.argsort(weighted)


def apply_permutation(
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Relabel HMM parameters by ``perm`` (perm[new] = old)."""
    return (
        startprob[perm],
        transmat[np.ix_(perm, perm)],
        means[perm],
        covars[perm],
    )


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------

def filtered_probability_at_end(model, X: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Posterior state probabilities at the final observation.

    At the last observation of a sample there is no future data, so the
    forward-backward posterior equals the forward (filtered)
    probability. ``tests/test_regimes.py`` verifies this against a
    hand-rolled forward pass rather than taking it on faith.
    """
    posteriors = model.predict_proba(X)
    return posteriors[-1][perm]


def forward_filtered_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Explicit forward-algorithm filtered probabilities (all dates).

    Reference implementation used to validate the endpoint identity in
    tests; scaled per step for numerical stability.
    """
    log_b = model._compute_log_likelihood(X)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    n_obs, n_states = b.shape
    alpha = np.zeros((n_obs, n_states))
    a = model.startprob_ * b[0]
    alpha[0] = a / a.sum()
    for t in range(1, n_obs):
        a = (alpha[t - 1] @ model.transmat_) * b[t]
        alpha[t] = a / a.sum()
    return alpha


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def state_diagnostics(
    model,
    X: np.ndarray,
    window: pd.DataFrame,
    perm: np.ndarray,
    cfg: HMMConfig,
) -> dict:
    """Occupancy, persistence, effective sample size, and guard flags."""
    posteriors = model.predict_proba(X)[:, perm]
    transmat = model.transmat_[np.ix_(perm, perm)]
    covars = model.covars_[perm]

    occupancy = posteriors.mean(axis=0)
    resp_sum = posteriors.sum(axis=0)
    n_eff = resp_sum**2 / (posteriors**2).sum(axis=0)
    diag = np.diag(transmat)
    with np.errstate(divide="ignore"):
        expected_duration = np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)
    min_eig = np.array([np.linalg.eigvalsh(c).min() for c in covars])

    raw_vol = window[REALIZED_VOL_FEATURE].to_numpy()
    state_mean_vol = (posteriors * raw_vol[:, None]).sum(axis=0) / resp_sum

    out = {
        "n_obs": len(X),
        "loglik": float(model.score(X)),
        "min_occupancy": float(occupancy.min()),
        "degenerate_occupancy": bool(occupancy.min() < cfg.min_occupancy),
        "singular_covariance": bool((min_eig < cfg.min_eigenvalue).any()),
        "absorbing_state": bool((diag >= 1.0 - 1e-12).any()),
        "transmat_rows_ok": bool(np.allclose(transmat.sum(axis=1), 1.0)),
    }
    for k in range(cfg.n_states):
        out[f"occupancy_s{k}"] = float(occupancy[k])
        out[f"n_eff_s{k}"] = float(n_eff[k])
        out[f"persistence_s{k}"] = float(diag[k])
        out[f"expected_duration_s{k}"] = float(expected_duration[k])
        out[f"mean_realized_vol_s{k}"] = float(state_mean_vol[k])
        out[f"min_eigenvalue_s{k}"] = float(min_eig[k])
    out["guard_triggered"] = bool(
        out["degenerate_occupancy"] or out["singular_covariance"]
        or out["absorbing_state"] or not out["transmat_rows_ok"]
    )
    return out


# Amendment A2 (2026-08-15): a degenerate fit stays in the diagnostics
# and its probability is preserved, but it does not supply a
# regime-conditioned covariance for that rebalance. Reasons are ordered
# so the audit log names the most specific condition first.
DEGENERACY_REASONS = (
    ("absorbing_state", "absorbing_transition"),
    ("degenerate_occupancy", "degenerate_occupancy"),
    ("singular_covariance", "singular_covariance"),
)
COV_REGIME = "regime_conditioned"
COV_FALLBACK = "unconditional_ledoit_wolf"


def covariance_consumption(diagnostics: dict) -> dict:
    """Decide which covariance Phase 7 may consume for this rebalance.

    Implements Amendment A2. The probability is never altered here; only
    the covariance source changes, and the reason is recorded. Applies
    symmetrically to either state because the underlying flags do.
    """
    reasons = [label for flag, label in DEGENERACY_REASONS if diagnostics.get(flag)]
    fallback = bool(reasons)
    return {
        "covariance_used_for_allocation": COV_FALLBACK if fallback else COV_REGIME,
        "fallback_used": fallback,
        "fallback_reason": ";".join(reasons),
    }


# ---------------------------------------------------------------------------
# Expanding-window driver
# ---------------------------------------------------------------------------

def run_expanding_hmm(
    features: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    cfg: HMMConfig = HMMConfig(),
    progress_every: int = 25,
    verbose: bool = False,
    label_series: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Refit the HMM at every rebalance date on data through that date.

    ``label_series`` supplies the realized-volatility series used for
    canonical state labeling when it is not among the model features
    (the drop-realized-vol ablation); otherwise labeling uses the
    feature's state-conditional means. Either way the rule is the same:
    ascending training-only expected realized volatility.

    Returns a dict of frames: ``realtime`` (filtered probabilities and
    guard flags per date), ``initializations`` (every start, including
    failures), ``diagnostics`` (selected-fit diagnostics), and
    ``transitions`` (long-format transition matrices).
    """
    realtime_rows: list[dict] = []
    init_rows: list[dict] = []
    diag_rows: list[dict] = []
    trans_rows: list[dict] = []
    revision_rows: list[dict] = []
    previous: dict | None = None

    for i, t in enumerate(rebalance_dates):
        window = features.loc[:t]
        if len(window) < cfg.min_observations:
            continue
        X = standardize_window(window)

        model, records = fit_multistart(X, cfg)
        for record in records:
            init_rows.append({"date": t, **record})

        if REALIZED_VOL_FEATURE in features.columns:
            perm = canonical_permutation(model, list(features.columns))
            vol_window = window
        else:
            if label_series is None:
                raise ValueError(
                    "label_series is required when realized volatility is "
                    "not a model feature (canonical labeling needs it)"
                )
            aligned = label_series.reindex(window.index)
            perm = canonical_permutation_by_series(model, X, aligned.to_numpy())
            vol_window = window.assign(**{REALIZED_VOL_FEATURE: aligned})
        probs = filtered_probability_at_end(model, X, perm)
        diagnostics = state_diagnostics(model, X, vol_window, perm, cfg)
        transmat = model.transmat_[np.ix_(perm, perm)]

        selected_seed = next(r["seed"] for r in records if r["selected"])
        row = {"date": t, "n_obs": len(window)}
        for k in range(cfg.n_states):
            row[f"prob_s{k}"] = float(probs[k])
        row["high_vol_state_prob"] = float(probs[-1])
        row["classified_high_vol"] = bool(probs[-1] > 0.5)
        row["guard_triggered"] = diagnostics["guard_triggered"]
        row["n_failed_inits"] = int(sum(1 for r in records if not r["usable"]))
        row["selected_seed"] = int(selected_seed)
        # Amendment A2 audit columns. The reported probability is the
        # estimated one, unchanged; only covariance consumption differs.
        row["hmm_probability_used_for_reporting"] = float(probs[-1])
        row.update(covariance_consumption(diagnostics))
        realtime_rows.append(row)

        # Model-revision stability against the previous refit, measured
        # on the historical window both models have seen.
        current = {
            "date": t,
            "posteriors": model.predict_proba(X)[:, perm],
            "means": model.means_[perm],
            "transmat": model.transmat_[np.ix_(perm, perm)],
            "seed": selected_seed,
            "n_obs": len(window),
        }
        if previous is not None:
            revision_rows.append(
                revision_stability(previous, current, n_overlap=previous["n_obs"])
            )
        previous = current

        diag_rows.append({"date": t, **diagnostics})
        for a in range(cfg.n_states):
            for b in range(cfg.n_states):
                trans_rows.append(
                    {"date": t, "from_state": a, "to_state": b,
                     "probability": float(transmat[a, b])}
                )
        if verbose and (i % progress_every == 0):
            print(f"    refit {i + 1}/{len(rebalance_dates)}  {t.date()}  "
                  f"P(high)={probs[-1]:.3f}", flush=True)

    realtime = pd.DataFrame(realtime_rows)
    # State evolution: month-to-month movement of the endpoint
    # probability. This tracks changing market conditions and is NOT a
    # measure of model instability (see revision_stability).
    if not realtime.empty:
        realtime["prob_change_vs_prev_month"] = realtime["high_vol_state_prob"].diff()
        realtime["classification_changed"] = (
            realtime["classified_high_vol"] != realtime["classified_high_vol"].shift()
        )
        realtime.loc[realtime.index[0], "classification_changed"] = False

    return {
        "realtime": realtime,
        "initializations": pd.DataFrame(init_rows),
        "diagnostics": pd.DataFrame(diag_rows),
        "transitions": pd.DataFrame(trans_rows),
        "revisions": pd.DataFrame(revision_rows),
    }


def revision_stability(
    prev: dict, curr: dict, n_overlap: int
) -> dict:
    """Model-revision stability between consecutive refits.

    Distinct from *state evolution* (how the endpoint probability moves
    month to month, which reflects changing markets rather than model
    instability). Here the two models are compared **on the same
    overlapping historical window** after canonical labeling: does
    adding one more month of data change what the model says about the
    past it already saw?

    Each model's posteriors are taken under its own standardization,
    which is the scaling that model was estimated with, so the
    comparison is between two self-consistent views of identical dates.
    """
    prev_probs = prev["posteriors"][:n_overlap, -1]
    curr_probs = curr["posteriors"][:n_overlap, -1]
    agreement = float(((prev_probs > 0.5) == (curr_probs > 0.5)).mean())
    return {
        "date": curr["date"],
        "n_overlap": int(n_overlap),
        "mean_abs_prob_revision": float(np.abs(curr_probs - prev_probs).mean()),
        "max_abs_prob_revision": float(np.abs(curr_probs - prev_probs).max()),
        "prob_revision_corr": float(np.corrcoef(prev_probs, curr_probs)[0, 1]),
        "classification_agreement": agreement,
        "mean_drift_l2": float(np.linalg.norm(curr["means"] - prev["means"])),
        "transmat_drift_l1": float(np.abs(curr["transmat"] - prev["transmat"]).sum()),
        "selected_seed_prev": int(prev["seed"]),
        "selected_seed_curr": int(curr["seed"]),
        "selected_seed_changed": bool(curr["seed"] != prev["seed"]),
    }


def run_ablation(
    features: pd.DataFrame,
    dropped_feature: str,
    rebalance_dates: pd.DatetimeIndex,
    cfg: HMMConfig = HMMConfig(),
) -> pd.DataFrame:
    """Re-run the expanding HMM with one feature removed.

    Preregistered robustness (drop VIX / drop realized volatility);
    returns the ``realtime`` frame for comparison against the main
    specification.
    """
    subset = features.drop(columns=[dropped_feature])
    label_series = (
        features[REALIZED_VOL_FEATURE]
        if dropped_feature == REALIZED_VOL_FEATURE
        else None
    )
    return run_expanding_hmm(
        subset, rebalance_dates, cfg, label_series=label_series
    )["realtime"]


def fit_expost_smoothed(
    features: pd.DataFrame, cfg: HMMConfig = HMMConfig()
) -> pd.DataFrame:
    """Full-sample smoothed probabilities — EX-POST DESCRIPTIVE ONLY.

    Conditions on the entire sample including the future, so it must
    never inform a trading decision. Stored in its own file with an
    explicit column marking it ex-post.
    """
    X = standardize_window(features)
    model, _ = fit_multistart(X, cfg)
    perm = canonical_permutation(model)
    posteriors = model.predict_proba(X)[:, perm]
    out = pd.DataFrame(
        {f"prob_s{k}": posteriors[:, k] for k in range(cfg.n_states)},
        index=features.index,
    )
    out["high_vol_state_prob"] = posteriors[:, -1]
    out["is_expost_smoothed"] = True
    return out.reset_index()


# ---------------------------------------------------------------------------
# Comparison against simple volatility thresholds
# ---------------------------------------------------------------------------

def distinctiveness_diagnostic(
    realtime: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """**Distinctiveness only** — does the HMM label observations
    differently from a simple threshold rule?

    This answers "are the classifications different?", NOT "does the
    HMM add predictive or investment value?". Disagreement with a
    VIX-median rule is not evidence of value: the HMM could differ and
    be worse. Incremental value can only be assessed later through
    covariance forecasts, realized risk, and portfolio performance
    (Phases 7-11), and this table must never be cited as if it had.

    Builds two real-time benchmark rules (feature above its expanding
    median through the same date, so both are causal), then reports
    agreement rate and Cohen's kappa against the HMM classification.
    """
    rows = []
    for feature in ("log_vix_lag1", REALIZED_VOL_FEATURE):
        expanding_median = features[feature].expanding(min_periods=252).median()
        rule = (features[feature] > expanding_median).reindex(realtime["date"])
        hmm = realtime.set_index("date")["classified_high_vol"]
        both = pd.DataFrame({"hmm": hmm, "rule": rule}).dropna()
        agree = float((both["hmm"] == both["rule"]).mean())
        p_hmm, p_rule = both["hmm"].mean(), both["rule"].mean()
        p_chance = p_hmm * p_rule + (1 - p_hmm) * (1 - p_rule)
        kappa = (agree - p_chance) / (1 - p_chance) if p_chance < 1 else np.nan
        rows.append(
            {
                "benchmark_rule": f"{feature} > expanding median",
                "n": len(both),
                "agreement_rate": agree,
                "cohens_kappa": float(kappa),
                "hmm_high_vol_share": float(p_hmm),
                "rule_high_vol_share": float(p_rule),
            }
        )
    return pd.DataFrame(rows)


def classification_agreement(
    base: pd.DataFrame, other: pd.DataFrame, label: str
) -> dict:
    """Agreement between two real-time classification paths (ablations)."""
    merged = base.merge(other, on="date", suffixes=("_base", "_alt"))
    agree = float(
        (merged["classified_high_vol_base"] == merged["classified_high_vol_alt"]).mean()
    )
    corr = float(
        merged["high_vol_state_prob_base"].corr(merged["high_vol_state_prob_alt"])
    )
    return {
        "specification": label,
        "n": len(merged),
        "classification_agreement": agree,
        "probability_correlation": corr,
        "alt_high_vol_share": float(merged["classified_high_vol_alt"].mean()),
        "base_high_vol_share": float(merged["classified_high_vol_base"].mean()),
    }
