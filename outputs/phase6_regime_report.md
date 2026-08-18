# Phase 6 — Latent Volatility State Identification

**Generated from:** config SHA-256 · data snapshot `manifest_2026-08-06.json` ·
git commit per [phase6_manifest.json](phase6_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py --phase 6`
**Scope:** HMM estimation and regime diagnostics only. **No covariance
estimation, portfolio optimization, or strategy performance was
computed or examined.** All numbers are read from the generated tables.

**Status: CLOSED**, tag `v0.5.0-regimes`. The absorbing-state finding
(Section 6) was resolved by **Amendment A2**, adopted before any
covariance or portfolio result existed.

Throughout, the two states are called **latent volatility states**.
They are estimated from volatility-related features, and the evidence
below supports only that reading. Nothing here justifies calling them
bull/bear, expansion/recession, or economic regimes.

---

## 1. Sample and specification

| Item | Value |
|---|---|
| Signal origins | **200** month-ends, **2009-12-31** to 2026-07-31 |
| First execution | **2010-01-04** (first trading day after the first signal) |
| Model | 2-state Gaussian HMM, expanding window, monthly refit |
| Starts per refit | 16 (seeds 42-57), selection by training log-likelihood |
| Features | SPY log return, 21-day realized vol, log VIX (lag 1), yield slope (lag 1) |
| Standardization | expanding z-score, estimation window only |

The origin count corrects an earlier run that began in 2005-11 by
applying the 252-observation minimum instead of the preregistered
training window. That run was stopped, its partial artifacts deleted,
and no pre-2010 signal exists in any file. Every estimation parameter,
tagged as preregistered or as a pre-estimation implementation decision,
is in [`implementation_parameters.csv`](regimes/implementation_parameters.csv).

**Selection rule.** Fits within a relative tolerance (1e-10) of the best
training log-likelihood are treated as tied, and the lowest seed among
them is selected. **108 of 200 refits had two or more starts inside the
tolerance band** (mean 2.3), and at 193 of 200 refits the gap between
the best and second-best log-likelihood fell below 1e-9 — the median gap
is exactly zero, meaning several starts converge on the same optimum.

The rule is a safeguard against a ranking flip driven by numerically
immaterial differences. It has not been shown to be necessary: replaying
the counterfactual across two independent full runs, plain `argmax`
would have selected the same seed at all 200 origins (see
`docs/REPRODUCTION.md`). Where the log-likelihood gap is exactly zero
the two rules agree by construction, since both resolve to the lowest
seed in array order.

## 2. Do the states separate, and are they persistent without being degenerate?

State-conditional realized volatility separates cleanly and the ordering
is stable across all 200 refits (`selected_fit_diagnostics.csv`):

| Quantity | State 0 (lower vol) | State 1 (higher vol) |
|---|---|---|
| Mean realized vol | 11.1% | **26.0%** (≈2.3×) |
| Occupancy (mean) | 64.5% | 35.5% |
| Occupancy (min across refits) | 38.7% | 29.4% |
| Effective sample size (mean / min) | 2,217 / 673 | 1,181 / **436** |
| Persistence P(stay) | 0.993 | 0.987 |
| Expected duration | ~211 days | ~168 days (4 infinite, see §6) |

**Occupancy never approaches the 5% degeneracy threshold** — the
smaller state holds at least 29% at every refit — and effective sample
size for the high-volatility state never falls below 436 observations,
comfortably above the 60-observation floor the Phase 7 shrinkage rule
uses. Both states are strongly persistent, which is what makes a regime
model meaningful rather than a noisy relabeling of each day.

## 3. Are state probabilities stable across expanding refits?

Two different questions, reported separately
(`stability_diagnostics.csv`, `model_revision_stability.csv`):

**Model-revision stability** — refit at *t* and *t+1*, compared on the
overlapping history both models saw, after canonical labeling:

| Metric | Mean | Worst case |
|---|---|---|
| Mean absolute probability revision | **0.0054** | 0.270 |
| Probability correlation | **0.992** | 0.429 |
| Classification agreement | **99.5%** | 73.0% |
| Mean drift (L2, state means) | 0.032 | 0.817 |
| Transition-matrix drift (L1) | 0.0009 | 0.026 |

Adding one month of data almost never changes what the model says about
the past: typical revision is half a percentage point of probability,
and 99.5% of historical classifications survive a refit. A small number
of refits do revise materially (worst case: 27% mean revision, 73%
agreement), and those are the same episodes discussed in §6.

**State evolution** — month-to-month movement of the endpoint
probability — averages 0.158 in absolute value, with 31 classification
changes across 200 months. This reflects **changing market conditions,
not model instability**, and is reported separately for that reason.

**Selected-seed stability is poor and is a genuine finding**: the
selected seed changed at 152 of 199 consecutive refit pairs, and all 16
seeds were selected at some point (seed 42 most often, 29 times). The
EM likelihood surface has multiple local optima and different starts
reach the global one at different dates. Because parameter and
probability drift remain tiny, this is evidence that the 16-start
protocol is doing necessary work rather than evidence of instability in
the fitted model — but a single-start implementation would have been
materially arbitrary.

Zero initializations failed across all 3,200 fits (200 refits × 16).

## 4. Does the model differ from simply thresholding VIX or realized volatility?

`distinctiveness_diagnostic.csv` compares the HMM classification with
two causal benchmark rules (feature above its expanding median):

| Benchmark rule | Agreement | Cohen's κ | Rule high-vol share |
|---|---|---|---|
| lagged log VIX > expanding median | 81% | 0.62 | 48% |
| 21-day realized vol > expanding median | 77% | 0.53 | 44% |

The HMM assigns a different label in roughly one month in five, and the
kappas indicate moderate rather than near-perfect concordance. The HMM
also classifies fewer months as high-volatility (41%) than either
threshold rule.

**This is a distinctiveness result and nothing more.** It shows the
classifications differ; it does **not** show the HMM is better, more
informative, or economically useful. The model could differ from a
threshold rule and be worse. Incremental value can only be assessed
through covariance forecasts, realized risk, and portfolio performance
in Phases 7-11, and this table must not be cited as if it had answered
that question.

## 5. How sensitive are classifications to dropping a correlated feature?

Preregistered ablations (`feature_ablations.csv`), each re-running all
200 refits with one feature removed:

| Specification | Classification agreement | Probability correlation | High-vol share |
|---|---|---|---|
| Main (4 features) | — | — | 41% |
| Drop VIX | 85% | 0.71 | 46% |
| Drop realized volatility | 82% | 0.70 | 57% |

Sensitivity is **material**: removing either correlated volatility
feature changes 15-18% of monthly classifications, and dropping realized
volatility pushes the high-volatility share from 41% to 57%. The Phase 4
finding (VIX-RV level correlation 0.87) predicted overlap, and these
runs quantify it: the two features are neither redundant nor
interchangeable — each shifts the state boundary in its own direction.
No feature-set change is made; the main specification remains the frozen
four-feature model, and these remain robustness runs.

## 6. Finding that requires a decision: absorbing high-volatility states

**4 of 200 refits produced an absorbing high-volatility state** —
P(stay in state 1) = 1.000, implying the model believes the state is
never exited — at **2009-12-31, 2012-03-30, 2012-04-30, and 2012-05-31**.
All four emitted a real-time signal of P(high) = 1.0 and are flagged
`guard_triggered = True` in `realtime_probabilities.parquet`.

Facts, stated before any interpretation:

- Occupancy at these refits was healthy (29-43%), so the preregistered
  **occupancy** guard did not and would not fire. The degeneracy is in
  the transition matrix, which the frozen plan did not anticipate.
- All 16 starts converged at each of these dates; the absorbing fit was
  the likelihood-selected one, with 1-3 fits inside the tolerance band.
- **2009-12-31 is the first preregistered signal origin**, so this
  affects the very first rebalance of the out-of-sample period.
- The earliest refits have the shortest training windows (~1,270
  observations, dominated by the 2008-09 crisis), which is the most
  plausible mechanical explanation for a state that never exits
  in-sample.

**What the preregistration says.** The frozen plan specifies a fallback
to unconditional Ledoit-Wolf covariance only for **occupancy** below 5%.
It is silent on absorbing transition rows. The absorbing-state flag
itself was a pre-estimation implementation decision (documented in
`implementation_parameters.csv`), which is why the degeneracy was
detected at all.

**Why this is not resolved unilaterally.** Extending the fallback rule
changes how the regime signal is consumed at 4 of 200 rebalances (2%),
including the first. That is a specification decision, and it is being
raised **before** any covariance or portfolio result exists — no
performance information of any kind has been computed, so the choice
carries no performance information to motivate it. What is known: the four dates, their
flags, and P(high) = 1.0 at each.

### Resolution: Amendment A2 (adopted 2026-08-15, before any covariance or portfolio result)

An absorbing transition row is treated as a **degenerate fit**: the fit
stays in the regime diagnostics and **its probability is preserved and
reported unchanged**, but it does not supply a regime-conditioned
covariance for that rebalance. Phase 7 instead uses the unconditional
Ledoit-Wolf covariance estimated through that date. The rule applies
symmetrically to either state, and the fallback is audit-logged.

Rationale recorded with the amendment: the transition estimate sits on
the parameter boundary, and horizon-averaging `p_t P_t^h` would
otherwise mechanically assign 100% probability to the high-volatility
state for the entire holding period. Unconditional shrinkage is
conservative and matches the degeneracy policy the frozen plan already
applies to low occupancy.

Verified in `realtime_probabilities.parquet`:

| Date | P(high) reported | Covariance consumed | Reason |
|---|---|---|---|
| 2009-12-31 | 1.000 | unconditional_ledoit_wolf | absorbing_transition |
| 2012-03-30 | 1.000 | unconditional_ledoit_wolf | absorbing_transition |
| 2012-04-30 | 1.000 | unconditional_ledoit_wolf | absorbing_transition |
| 2012-05-31 | 1.000 | unconditional_ledoit_wolf | absorbing_transition |

All 196 other rebalances consume the regime-conditioned covariance.
`hmm_probability_used_for_reporting` equals `high_vol_state_prob` on
every row; no rebalance was deleted, no probability replaced or
interpolated, no refit repeated to obtain a different transition
matrix, and the absorbing threshold is unchanged from its
pre-amendment value. An `accept_as_estimated` case is in the frozen
robustness grid to show whether this conservative fallback materially
affects any conclusion.

## 6b. Reproducibility evidence

Phase 6 was executed twice: once before Amendment A2 and once after, on
identical data and seeds. Results:

- **Selected seeds identical at all 200 refits**, and **all 200
  classifications identical**.
- Filtered probabilities agree to **1.2e-11** (97 of 200 rows differ at
  that magnitude; BLAS summation order is not bit-reproducible).
- The closest any probability comes to the 0.5 decision boundary is
  **0.164**, roughly ten orders of magnitude larger than the numerical
  wobble, so no classification could flip from it.

The discrete outputs a strategy consumes — the state label and the
selected fit — are therefore stable, which is exactly the property the
tolerance-based selection rule was introduced to guarantee. Bit-exact
reproducibility of the floating-point probabilities is not claimed.

## 7. What the state path looks like

`regime_probabilities.png` (top panel is the only series a strategy may
use). Annual mean real-time P(high-volatility): 2010 1.00, 2011 0.99,
2012 0.42, 2013-14 ≈0.00, 2015 0.17, 2016 0.24, 2017 0.00, 2018 0.42,
2019 0.17, **2020 0.92**, 2021 0.35, **2022 1.00**, 2023 0.50, 2024 0.16,
2025 0.30, 2026 0.15.

The probabilities are highly bimodal, sitting at 0 or 1 far more often
than in between, so the classification is rarely marginal. As the
approved specification anticipated, the high-volatility state covering
2020 and 2022 is expected and is not itself evidence of a useful model;
the diagnostics in §2-§5 are the substantive findings. The ex-post
smoothed panel is shown solely to make the filtered/smoothed distinction
visible and is never an input to anything.

## Artifacts

`outputs/regimes/`: `realtime_probabilities.parquet` (200 signal rows),
`expost_smoothed_probabilities.parquet` (descriptive only),
`selected_fit_diagnostics.csv`, `all_initializations.csv` (3,200 rows,
every start including selection flags), `transition_matrices.parquet`,
`state_characteristics.csv`, `stability_diagnostics.csv`,
`model_revision_stability.csv`, `distinctiveness_diagnostic.csv`,
`feature_ablations.csv`, `implementation_parameters.csv`;
`outputs/figures/regime_probabilities.png`;
[phase6_manifest.json](phase6_manifest.json). Reproduce with
`python scripts/run_analysis.py --phase 6` (refuses to run off-snapshot).
Tests: 27 Phase 6 unit tests covering probability validity, transition
rows, endpoint filtered-vs-forward-algorithm identity, causality under
future perturbation, canonical labeling under reversed raw labels,
near-tie selection portability, sample-boundary enforcement, and every
degeneracy guard.
