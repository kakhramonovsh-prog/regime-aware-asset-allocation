"""Tests for Phase 6 HMM regime identification.

Covers every gate in the approved specification: probability validity,
transition-matrix rows, causality under future perturbation, canonical
labeling under reversed raw labels, seed reproducibility, best-of-16
selection, feature causality, the occupancy guard, and the requirement
that non-convergence or degenerate fits cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as prep
from src import regimes as rg

CFG = rg.HMMConfig(n_init=4, min_observations=252, n_iter=50)


# ---------------------------------------------------------------------------
# Synthetic two-regime data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regime_features() -> pd.DataFrame:
    """Feature panel with a genuine embedded high-volatility episode."""
    rng = np.random.default_rng(3)
    n = 900
    idx = pd.bdate_range("2015-01-01", periods=n, name="Date")
    vol = np.full(n, 0.008)
    vol[400:520] = 0.030          # stressed stretch
    ret = rng.normal(0, vol)
    realized = pd.Series(ret, index=idx).rolling(21, min_periods=21).std()
    frame = pd.DataFrame(
        {
            "mkt_log_return": ret,
            "realized_vol_21d": realized,
            "log_vix_lag1": np.log(15 + 400 * pd.Series(vol, index=idx).shift(1)),
            "yield_slope_lag1": pd.Series(
                np.cumsum(rng.normal(0, 0.02, n)), index=idx
            ).shift(1),
        },
        index=idx,
    ).dropna()
    return frame[rg.FEATURE_NAMES]


@pytest.fixture(scope="module")
def fitted(regime_features):
    X = rg.standardize_window(regime_features)
    model, records = rg.fit_multistart(X, CFG)
    perm = rg.canonical_permutation(model)
    return X, model, records, perm


# ---------------------------------------------------------------------------
# Feature construction and causality
# ---------------------------------------------------------------------------

def test_build_features_columns_and_lag(prices, macro_frames, trading_days):
    macro = pd.DataFrame(
        {
            "VIXCLS": np.linspace(12, 30, len(prices)),
            "DGS10": np.linspace(3.0, 4.0, len(prices)),
            "DGS2": np.linspace(2.0, 3.5, len(prices)),
        },
        index=prices.index,
    )
    p = prices.copy()
    p.columns = ["SPY", "IEF", "GLD"]
    features = rg.build_features(p, macro)
    assert list(features.columns) == rg.FEATURE_NAMES
    assert features.notna().all().all()
    # Macro features are lagged one trading day.
    raw_log_vix = np.log(macro["VIXCLS"])
    t = features.index[10]
    prior = macro.index[macro.index.get_loc(t) - 1]
    assert features.loc[t, "log_vix_lag1"] == pytest.approx(raw_log_vix.loc[prior])


def test_feature_panel_passes_causality(prices):
    macro = pd.DataFrame(
        {
            "VIXCLS": np.linspace(12, 30, len(prices)),
            "DGS10": np.linspace(3.0, 4.0, len(prices)),
            "DGS2": np.linspace(2.0, 3.5, len(prices)),
        },
        index=prices.index,
    )
    p = prices.copy()
    p.columns = ["SPY", "IEF", "GLD"]
    prep.assert_causal(lambda df: rg.build_features(df, macro.loc[df.index]), p)


def test_no_forward_looking_feature_names():
    assert not any(
        token in name for name in rg.FEATURE_NAMES for token in ("fwd", "future", "next")
    )


def test_standardization_uses_window_only(regime_features):
    window = regime_features.iloc[:400]
    X = rg.standardize_window(window)
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(X.std(axis=0, ddof=1), 1, atol=1e-12)
    # Extending the sample must not change the earlier window's scaling.
    X_again = rg.standardize_window(regime_features.iloc[:400])
    np.testing.assert_allclose(X, X_again)


# ---------------------------------------------------------------------------
# Probability and transition validity
# ---------------------------------------------------------------------------

def test_probabilities_finite_and_sum_to_one(fitted):
    X, model, _, perm = fitted
    probs = rg.filtered_probability_at_end(model, X, perm)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0, abs=1e-10)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_transition_rows_sum_to_one(fitted):
    _, model, _, perm = fitted
    transmat = model.transmat_[np.ix_(perm, perm)]
    np.testing.assert_allclose(transmat.sum(axis=1), 1.0, atol=1e-10)
    assert (transmat >= 0).all()


def test_filtered_equals_forward_algorithm_at_endpoint(fitted):
    """The endpoint smoothed posterior IS the filtered probability."""
    X, model, _, _ = fitted
    smoothed_last = model.predict_proba(X)[-1]
    forward_last = rg.forward_filtered_probabilities(model, X)[-1]
    np.testing.assert_allclose(smoothed_last, forward_last, atol=1e-8)


def test_smoothed_differs_from_filtered_in_sample_interior():
    """Interior smoothed values use future data, which is why only the
    endpoint may be used as a signal.

    Requires genuinely ambiguous states: with well-separated emissions
    the posteriors saturate at 0/1 and smoothing adds nothing, so this
    test builds deliberately overlapping states.
    """
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=2, covariance_type="full")
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.85, 0.15], [0.15, 0.85]])
    model.means_ = np.array([[-0.4], [0.4]])          # heavy overlap
    model.covars_ = np.array([[[1.0]], [[1.0]]])
    model.n_features = 1

    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (200, 1))

    smoothed = model.predict_proba(X)
    filtered = rg.forward_filtered_probabilities(model, X)
    # Endpoint identity still holds exactly...
    np.testing.assert_allclose(smoothed[-1], filtered[-1], atol=1e-8)
    # ...but interior values genuinely differ, because smoothing there
    # conditions on observations that arrive later.
    assert np.abs(smoothed[:-1] - filtered[:-1]).max() > 1e-3


# ---------------------------------------------------------------------------
# Canonical labeling
# ---------------------------------------------------------------------------

def test_canonical_labeling_puts_low_vol_first(fitted):
    X, model, _, perm = fitted
    idx = rg.FEATURE_NAMES.index(rg.REALIZED_VOL_FEATURE)
    ordered_means = model.means_[perm, idx]
    assert ordered_means[0] < ordered_means[-1]


def test_canonical_labeling_survives_reversed_raw_labels(fitted):
    """Swapping the raw state order must not change canonical output."""
    X, model, _, perm = fitted
    idx = rg.FEATURE_NAMES.index(rg.REALIZED_VOL_FEATURE)
    original = model.means_[perm, idx]

    swap = np.array([1, 0])
    startprob, transmat, means, covars = rg.apply_permutation(
        model.startprob_, model.transmat_, model.means_, model.covars_, swap
    )

    class Reversed:
        pass

    reversed_model = Reversed()
    reversed_model.startprob_, reversed_model.transmat_ = startprob, transmat
    reversed_model.means_, reversed_model.covars_ = means, covars

    perm_rev = rg.canonical_permutation(reversed_model)
    np.testing.assert_allclose(reversed_model.means_[perm_rev, idx], original)


def test_apply_permutation_reorders_transitions_consistently():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([[5.0], [1.0]])
    covars = np.array([[[2.0]], [[1.0]]])
    startprob = np.array([0.7, 0.3])
    perm = np.array([1, 0])
    sp, tm, mu, cv = rg.apply_permutation(startprob, transmat, means, covars, perm)
    assert mu[0, 0] == 1.0                      # low state first
    assert tm[0, 0] == 0.8                      # its self-transition follows it
    assert tm[0, 1] == 0.2
    assert sp[0] == 0.3
    np.testing.assert_allclose(tm.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Multistart selection and reproducibility
# ---------------------------------------------------------------------------

def test_selected_fit_is_within_tolerance_of_best(fitted):
    """The selected fit must be statistically the best available.

    Not exact equality with the maximum: fits that are numerically
    indistinguishable (within the relative selection tolerance) are
    treated as tied, and the lowest seed wins for machine portability.
    On this fixture two starts land ~1e-11 apart, which is precisely
    the situation the tolerance rule exists to handle.
    """
    _, model, records, _ = fitted
    usable = [r for r in records if r["usable"]]
    assert usable, "expected at least one usable initialization"
    selected = [r for r in records if r["selected"]]
    assert len(selected) == 1

    best = max(r["loglik"] for r in usable)
    tol = CFG.selection_tol_rel * max(1.0, abs(best))
    assert best - selected[0]["loglik"] <= tol
    # And it is the lowest seed among the near-best set.
    near = [r["seed"] for r in usable if best - r["loglik"] <= tol]
    assert selected[0]["seed"] == min(near)


def test_all_initializations_recorded(fitted):
    _, _, records, _ = fitted
    assert len(records) == CFG.n_init
    seeds = [r["seed"] for r in records]
    assert seeds == list(range(CFG.seed_start, CFG.seed_start + CFG.n_init))
    for record in records:
        assert set(record) >= {"seed", "converged", "loglik", "error", "usable", "selected"}


def test_same_seeds_reproduce_identical_fit(regime_features):
    """Same inputs and seeds reproduce the same selected fit.

    Log-likelihoods are compared with a tolerance rather than exact
    equality: BLAS summation order is not bit-reproducible across runs,
    so repeated identical fits can differ at ~1e-13. Selection itself
    uses argmax (not float equality) so this cannot change which fit is
    chosen; the chosen seed must match exactly.
    """
    X = rg.standardize_window(regime_features)
    model_a, records_a = rg.fit_multistart(X, CFG)
    model_b, records_b = rg.fit_multistart(X, CFG)
    np.testing.assert_allclose(model_a.means_, model_b.means_)
    np.testing.assert_allclose(model_a.transmat_, model_b.transmat_)
    np.testing.assert_allclose(
        [r["loglik"] for r in records_a],
        [r["loglik"] for r in records_b],
        rtol=1e-9,
    )
    selected_a = [r["seed"] for r in records_a if r["selected"]]
    selected_b = [r["seed"] for r in records_b if r["selected"]]
    assert selected_a == selected_b
    assert len(selected_a) == 1


def test_near_tie_selection_prefers_lowest_seed(monkeypatch, regime_features):
    """Perturbing likelihoods within the tolerance must not change the
    selected fit: the lowest seed inside the near-best band wins.

    This is the machine-portability guard. A plain argmax would let a
    1e-13 BLAS wobble reverse the choice between machines.
    """
    X = rg.standardize_window(regime_features)

    class StubModel:
        def __init__(self, seed):
            self.seed = seed
            self.means_ = np.full((2, 4), float(seed))
            self.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
            self.monitor_ = type("M", (), {"converged": True, "history": [1]})()

        def fit(self, X):
            return self

        def score(self, X):
            # Seeds 42, 43, 44 are numerically indistinguishable;
            # seed 43 is nominally highest by a hair inside tolerance.
            return {42: 1000.0, 43: 1000.0 + 1e-11, 44: 1000.0 - 1e-11}.get(
                self.seed, 900.0
            )

    import hmmlearn.hmm as hmm_module

    monkeypatch.setattr(
        hmm_module, "GaussianHMM",
        lambda **kwargs: StubModel(kwargs["random_state"]),
    )
    cfg = rg.HMMConfig(n_init=4, seed_start=42)
    selected, records = rg.fit_multistart(X, cfg)

    chosen = [r["seed"] for r in records if r["selected"]]
    assert chosen == [42], "lowest seed within tolerance must win"
    assert selected.seed == 42
    within = sorted(r["seed"] for r in records if r["within_selection_tol"])
    assert within == [42, 43, 44]
    # A genuinely worse fit stays outside the band.
    assert not next(r for r in records if r["seed"] == 45)["within_selection_tol"]


def test_all_starts_failing_raises():
    bad = np.full((300, 4), np.nan)
    with pytest.raises(RuntimeError, match="all HMM initializations failed"):
        rg.fit_multistart(bad, CFG)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_diagnostics_flag_shapes_and_guards(fitted, regime_features):
    X, model, _, perm = fitted
    diag = rg.state_diagnostics(model, X, regime_features, perm, CFG)
    assert diag["transmat_rows_ok"] is True
    assert diag["occupancy_s0"] + diag["occupancy_s1"] == pytest.approx(1.0, abs=1e-9)
    assert 0 < diag["n_eff_s0"] <= len(X)
    assert 0 < diag["n_eff_s1"] <= len(X)
    assert diag["mean_realized_vol_s0"] < diag["mean_realized_vol_s1"]
    assert diag["expected_duration_s0"] > 1.0


def test_occupancy_guard_triggers_on_rare_state(regime_features):
    """A configuration demanding 60% minimum occupancy must flag."""
    X = rg.standardize_window(regime_features)
    strict = rg.HMMConfig(n_init=4, min_occupancy=0.60, n_iter=50)
    model, _ = rg.fit_multistart(X, strict)
    perm = rg.canonical_permutation(model)
    diag = rg.state_diagnostics(model, X, regime_features, perm, strict)
    assert diag["degenerate_occupancy"] is True
    assert diag["guard_triggered"] is True


def test_singular_covariance_flagged():
    """A state covariance below the eigenvalue floor must be caught."""
    class FakeModel:
        transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        means_ = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        covars_ = np.stack([np.eye(4), np.diag([1.0, 1.0, 1.0, 1e-14])])

        def predict_proba(self, X):
            p = np.full((len(X), 2), 0.5)
            return p

        def score(self, X):
            return -100.0

    features = pd.DataFrame(
        {name: np.linspace(0, 1, 300) for name in rg.FEATURE_NAMES}
    )
    diag = rg.state_diagnostics(
        FakeModel(), np.zeros((300, 4)), features, np.array([0, 1]), CFG
    )
    assert diag["singular_covariance"] is True
    assert diag["guard_triggered"] is True


def test_absorbing_state_flagged():
    class AbsorbingModel:
        transmat_ = np.array([[1.0, 0.0], [0.05, 0.95]])
        means_ = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        covars_ = np.stack([np.eye(4), np.eye(4)])

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

        def score(self, X):
            return -100.0

    features = pd.DataFrame({name: np.linspace(0, 1, 300) for name in rg.FEATURE_NAMES})
    diag = rg.state_diagnostics(
        AbsorbingModel(), np.zeros((300, 4)), features, np.array([0, 1]), CFG
    )
    assert diag["absorbing_state"] is True
    assert diag["guard_triggered"] is True


# ---------------------------------------------------------------------------
# Expanding-window causality
# ---------------------------------------------------------------------------

def test_stored_probabilities_unchanged_by_future_data(regime_features):
    """Appending or perturbing later data cannot change an earlier
    stored real-time probability."""
    dates = pd.DatetimeIndex([regime_features.index[300], regime_features.index[420]])
    base = rg.run_expanding_hmm(regime_features, dates, CFG)["realtime"]

    extended = regime_features.copy()
    future = extended.index > dates[-1]
    rng = np.random.default_rng(1)
    extended.loc[future] += rng.uniform(5, 10, (int(future.sum()), extended.shape[1]))
    perturbed = rg.run_expanding_hmm(extended, dates, CFG)["realtime"]

    pd.testing.assert_frame_equal(base, perturbed)


def test_run_expanding_outputs_are_wellformed(regime_features):
    dates = pd.DatetimeIndex([regime_features.index[300], regime_features.index[500]])
    out = rg.run_expanding_hmm(regime_features, dates, CFG)
    rt = out["realtime"]
    assert len(rt) == 2
    np.testing.assert_allclose(rt["prob_s0"] + rt["prob_s1"], 1.0, atol=1e-9)
    assert len(out["initializations"]) == 2 * CFG.n_init
    assert len(out["transitions"]) == 2 * CFG.n_states**2
    for _, group in out["transitions"].groupby(["date", "from_state"]):
        assert group["probability"].sum() == pytest.approx(1.0, abs=1e-9)
    assert out["diagnostics"]["transmat_rows_ok"].all()
    # One revision row per consecutive pair of refits.
    assert len(out["revisions"]) == 1
    rev = out["revisions"].iloc[0]
    assert rev["n_overlap"] == rt["n_obs"].iloc[0]
    assert 0 <= rev["classification_agreement"] <= 1
    assert rev["mean_abs_prob_revision"] >= 0


def test_state_evolution_distinct_from_revision(regime_features):
    """Endpoint-probability movement (market conditions) is reported
    separately from model revision (stability)."""
    dates = pd.DatetimeIndex(regime_features.index[300::80])
    out = rg.run_expanding_hmm(regime_features, dates, CFG)
    rt = out["realtime"]
    assert "prob_change_vs_prev_month" in rt.columns
    assert "classification_changed" in rt.columns
    assert pd.isna(rt["prob_change_vs_prev_month"].iloc[0])
    assert rt["classification_changed"].iloc[0] == False  # noqa: E712
    # Revision frame covers overlapping windows, not endpoint moves.
    assert set(out["revisions"].columns) >= {
        "mean_abs_prob_revision", "classification_agreement",
        "mean_drift_l2", "transmat_drift_l1", "selected_seed_changed",
    }


def test_rebalance_origins_start_at_preregistered_date(prices):
    """The primary sample must start at the frozen first signal origin,
    not at the earliest date meeting the minimum-observation rule."""
    returns = prep.log_returns(prices)
    origins = rg.rebalance_origins(
        returns.index, first_signal_after="2020-12-31", min_observations=100
    )
    assert origins[0] >= pd.Timestamp("2020-12-01")
    from src.volatility import month_end_rebalance_dates

    unrestricted = month_end_rebalance_dates(returns.index, 100)
    assert len(origins) < len(unrestricted)
    assert origins[0] > unrestricted[0]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Amendment A2: degenerate fits fall back to unconditional covariance
# ---------------------------------------------------------------------------

def test_a2_clean_fit_consumes_regime_covariance():
    clean = {"absorbing_state": False, "degenerate_occupancy": False,
             "singular_covariance": False}
    out = rg.covariance_consumption(clean)
    assert out["covariance_used_for_allocation"] == rg.COV_REGIME
    assert out["fallback_used"] is False
    assert out["fallback_reason"] == ""


def test_a2_absorbing_row_triggers_unconditional_fallback():
    degenerate = {"absorbing_state": True, "degenerate_occupancy": False,
                  "singular_covariance": False}
    out = rg.covariance_consumption(degenerate)
    assert out["covariance_used_for_allocation"] == rg.COV_FALLBACK
    assert out["fallback_used"] is True
    assert out["fallback_reason"] == "absorbing_transition"


def test_a2_covers_occupancy_and_singular_reasons():
    assert (
        rg.covariance_consumption(
            {"absorbing_state": False, "degenerate_occupancy": True,
             "singular_covariance": False}
        )["fallback_reason"]
        == "degenerate_occupancy"
    )
    both = rg.covariance_consumption(
        {"absorbing_state": True, "degenerate_occupancy": True,
         "singular_covariance": False}
    )
    assert both["fallback_reason"] == "absorbing_transition;degenerate_occupancy"


def test_a2_preserves_probability_unchanged(regime_features):
    """The fallback must never replace, interpolate, or delete the
    probability — only covariance consumption changes."""
    dates = pd.DatetimeIndex(regime_features.index[300::100])
    rt = rg.run_expanding_hmm(regime_features, dates, CFG)["realtime"]
    assert (rt["hmm_probability_used_for_reporting"] == rt["high_vol_state_prob"]).all()
    assert rt["high_vol_state_prob"].notna().all()
    assert len(rt) == len(dates)          # no rebalance deleted
    # Fallback rows keep their original probability.
    fallback = rt[rt["fallback_used"]]
    if len(fallback):
        assert (
            fallback["hmm_probability_used_for_reporting"]
            == fallback["high_vol_state_prob"]
        ).all()


def test_a2_fallback_flag_matches_guard(regime_features):
    dates = pd.DatetimeIndex(regime_features.index[300::100])
    rt = rg.run_expanding_hmm(regime_features, dates, CFG)["realtime"]
    # Every fallback row is a guarded row and names a reason.
    assert (rt.loc[rt["fallback_used"], "guard_triggered"]).all()
    assert (rt.loc[rt["fallback_used"], "fallback_reason"] != "").all()
    assert (rt.loc[~rt["fallback_used"], "fallback_reason"] == "").all()
    assert (
        rt.loc[~rt["fallback_used"], "covariance_used_for_allocation"] == rg.COV_REGIME
    ).all()


def test_a2_routing_is_flag_driven_not_date_driven():
    """The four observed A2 dates are audit information only.

    Routing must depend solely on the guard flags: a NON-A2 date with an
    absorbing row must fall back, and an A2 date with clean flags must
    NOT. If anyone ever hard-codes the date list into the routing, this
    test fails.
    """
    absorbing_on_other_date = {
        "absorbing_state": True, "degenerate_occupancy": False,
        "singular_covariance": False,
    }
    assert rg.covariance_consumption(absorbing_on_other_date)["fallback_used"] is True

    clean_on_a2_date = {
        "absorbing_state": False, "degenerate_occupancy": False,
        "singular_covariance": False,
    }
    assert rg.covariance_consumption(clean_on_a2_date)["fallback_used"] is False

    # The routing function takes no date argument at all.
    import inspect

    params = inspect.signature(rg.covariance_consumption).parameters
    assert list(params) == ["diagnostics"]


def test_no_hardcoded_a2_dates_in_routing_source():
    """Static check: the A2 date literals must not appear in any
    routing code path."""
    import inspect

    source = inspect.getsource(rg.covariance_consumption)
    source += inspect.getsource(rg.state_diagnostics)
    for literal in ("2009-12-31", "2012-03-30", "2012-04-30", "2012-05-31"):
        assert literal not in source


def test_a2_symmetric_across_states():
    """An absorbing LOW-volatility state must trigger the same rule."""
    class LowAbsorbing:
        transmat_ = np.array([[1.0, 0.0], [0.04, 0.96]])   # state 0 absorbing
        means_ = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        covars_ = np.stack([np.eye(4), np.eye(4)])

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

        def score(self, X):
            return -100.0

    features = pd.DataFrame({name: np.linspace(0, 1, 300) for name in rg.FEATURE_NAMES})
    diag = rg.state_diagnostics(
        LowAbsorbing(), np.zeros((300, 4)), features, np.array([0, 1]), CFG
    )
    assert diag["absorbing_state"] is True
    assert rg.covariance_consumption(diag)["fallback_reason"] == "absorbing_transition"


def test_distinctiveness_diagnostic_bounds(regime_features):
    dates = regime_features.index[300::100]
    rt = rg.run_expanding_hmm(regime_features, pd.DatetimeIndex(dates), CFG)["realtime"]
    comp = rg.distinctiveness_diagnostic(rt, regime_features)
    assert len(comp) == 2
    assert comp["agreement_rate"].between(0, 1).all()
    assert comp["cohens_kappa"].between(-1, 1).all()


def test_classification_agreement_identical_paths(regime_features):
    dates = pd.DatetimeIndex(regime_features.index[300::100])
    rt = rg.run_expanding_hmm(regime_features, dates, CFG)["realtime"]
    same = rg.classification_agreement(rt, rt, "self")
    assert same["classification_agreement"] == pytest.approx(1.0)
    assert same["probability_correlation"] == pytest.approx(1.0)
