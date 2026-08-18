# Phase 7 — Covariance Estimation and Conditioning

**Generated from:** config SHA-256 · data snapshot `manifest_2026-08-06.json` ·
git commit per [phase7_manifest.json](phase7_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py --phase 7`
**Scope:** covariance estimation and conditioning diagnostics only.
**No optimizer weights or portfolio returns were computed or examined.**

200 rebalance origins (2009-12-31 to 2026-07-31), 9 estimators per
origin, **1,800 matrices** in `covariance_audit.csv`.

---

## 1. What was estimated

At every origin, from returns through that origin only: sample
covariance, EWMA (λ = 0.94), unconditional Ledoit-Wolf, state-
conditioned covariances for both states (raw and shrunk), the horizon-
averaged regime mixture, and the matrix actually consumed downstream
after the Amendment A2 degeneracy rule.

State-conditioned estimates use the **smoothed responsibilities of the
HMM fit through that origin** — the model is refit at each origin with
the frozen protocol (16 seeds, tolerance selection), never the
full-sample ex-post series.

## 2. Conditioning: the case for shrinkage, measured

Mean and worst-case condition numbers across the 200 origins
(`conditioning_summary.csv`):

| Estimator | Cov condition (mean / max) | Corr condition (mean / max) |
|---|---|---|
| Ledoit-Wolf | **44.6** / 51.3 | **40.6** / 44.8 |
| Sample | 48.2 / 54.3 | 43.9 / 48.3 |
| **EWMA (λ=0.94)** | **116.3** / **504.9** | 86.0 / 425.5 |
| State 0 (low vol), raw | 36.8 / 40.0 | 28.0 / 31.6 |
| State 1 (high vol), raw | 61.4 / 68.4 | 58.4 / 66.3 |
| Regime mixture | 47.7 / 65.4 | 42.3 / 62.6 |
| Consumed (A2 applied) | 47.5 / 65.4 | 42.1 / 62.6 |

Three findings:

**EWMA is by far the worst-conditioned estimator**, averaging 2.4× the
Ledoit-Wolf condition number and peaking above 500. At λ = 0.94 the
weights have a **half-life of approximately 11 trading days and a
weight-based effective sample size of (1+λ)/(1−λ) ≈ 32 observations**,
which remains relatively small for estimating a five-asset covariance
matrix (15 free parameters). This is a concrete argument against
feeding a raw EWMA covariance to an optimizer, and it is consistent
with the Phase 4 finding that the equity block is highly collinear
(full-sample correlation condition number 46.5).

(The quantity 1/(1−λ) ≈ 16.7 sometimes quoted here is a decay-memory
heuristic, not the formal effective sample size; the weight-based
formula above is the correct one and is what the 32 refers to.)

**Ledoit-Wolf improves on the sample matrix at every origin**, as the
estimator is designed to. The improvement is modest (48.2 → 44.6 mean)
because 5 assets with 1,000+ observations is a comparatively benign
setting; the intrinsic Ledoit-Wolf shrinkage intensity averages just
0.0066, i.e. the data barely need shrinking at this dimension.

**The high-volatility state is materially worse conditioned than the
low-volatility state** (61.4 vs 36.8 mean covariance condition; 58.4 vs
28.0 on correlations). Correlations compress toward one in stress, so
the stressed-state matrix is closer to singular exactly when a
minimum-variance optimizer would lean on it hardest. This is the
mechanism the shrinkage rule exists to control.

## 3. Shrinkage behaves as specified, and does little here

`alpha_k = n_eff_k / (n_eff_k + 60)` weights the state estimate:

| State | n_eff (mean / min) | alpha (mean / min) |
|---|---|---|
| 0 (lower vol) | 2,217 / 673 | 0.968 / 0.918 |
| 1 (higher vol) | 1,181 / 436 | 0.946 / 0.879 |

Effective sample sizes never approach the 60-observation threshold, so
alpha stays near 1 and the shrinkage moves the state covariances only
slightly toward unconditional at the primary threshold (state 1 mean
condition 61.4 → 60.6). At κ = 60 the rule is therefore close to
inactive on this data: it is insurance against a short or rare state
rather than a driver of the results.

**The A1 robustness thresholds are not symmetric, and κ = 120 is not
negligible.** Since `alpha = n_eff / (n_eff + κ)`, a larger κ means
*more* shrinkage toward Ledoit-Wolf:

| n_eff | κ = 30 | κ = 60 (primary) | κ = 120 |
|---|---|---|---|
| 436 (minimum observed, state 1) | 0.936 | 0.879 | **0.784** |
| 1,181 (mean, state 1) | 0.975 | 0.952 | 0.908 |

At the minimum effective sample size the state estimate's weight falls
from 0.879 to 0.784 when κ moves from 60 to 120 — a materially larger
pull toward the unconditional matrix. The κ = 120 case may therefore
produce a visible sensitivity and **must not be dismissed in advance**;
the robustness section will report its outcome from generated output.

## 4. The A3 assumption, now measured

Amendment A3 fixed the main mixture as within-state only,
`Σ_RA = Σ_k p̄_k C_k`, explicitly omitting the between-state mean term.
Every origin records the relative Frobenius norm of that omitted term:

| Statistic | Value |
|---|---|
| Median | **0.069%** |
| Mean | 0.066% |
| Maximum | **0.159%** |
| Origins above 1% | **0 of 200** |
| Origins above 5% | 0 of 200 |

The omitted between-state mean dispersion is **at most 0.16% of the
within-state mixture** and never approaches 1%. The assumption is
therefore empirically innocuous at daily frequency, exactly as the
amendment anticipated but did not assume — state mean differences are
order basis points while daily variances are order 1e-4. The
`total_covariance_mixture` robustness case remains in the frozen grid
and is expected to be indistinguishable from the main specification;
that expectation will be reported from output, not asserted.

## 5. PSD integrity: nothing needed repair

**Zero PSD corrections across all 1,800 matrices.** The smallest
eigenvalue observed anywhere in the panel is 9.03e-07, comfortably
positive; no matrix came near the clipping tolerance, and the
hard-failure path was never reached.

This is the expected outcome — weighted covariances, EWMA, and
Ledoit-Wolf are all PSD by construction — and it means the eigenvalue
policy functioned as an unused safety net rather than a silent repair
mechanism. The policy distinguishes noise-level negatives (clip to a
1e-10 floor, log the magnitude) from materially negative eigenvalues
(raise `MateriallyNonPSDError`, refuse to clip); a unit test constructs
a matrix with eigenvalue −0.3 and asserts it fails rather than being
quietly fixed.

## 6. Amendment A2 in practice

Exactly **4 of 200 origins** fall back to the unconditional Ledoit-Wolf
covariance: 2009-12-31, 2012-03-30, 2012-04-30, 2012-05-31 — the same
dates the absorbing transition rows were observed at in Phase 6.

Verified directly from the audit:

- On all 4 fallback origins the consumed matrix **equals the
  Ledoit-Wolf estimate** and differs from the regime mixture.
- On non-fallback origins the consumed matrix equals the regime
  mixture (checked across sampled origins).
- All 200 origins are present; none was deleted.

Routing is driven **exclusively by the HMM guard flags**. The four
dates appear nowhere in any routing condition — tests assert that
`covariance_consumption()` takes no date argument, that an absorbing
row on any other date would fall back, that a clean fit on one of these
dates would not, and that the date literals do not occur in the routing
source.

## 7. Artifacts

`outputs/covariance/covariance_audit.csv` (1,800 rows: date,
estimator, window_start, observations, state, effective_sample_size,
shrinkage_intensity, min/max eigenvalues before and after,
covariance and correlation condition numbers, psd_correction_used and
magnitude, fallback_used, fallback_reason, covariance_consumed,
between_state_relative_norm, p_bar_high_state),
`consumed_covariances.parquet` (200 matrices in long format for
Phase 8), `conditioning_summary.csv`,
[phase7_manifest.json](phase7_manifest.json).

Reproduce with `python scripts/run_analysis.py --phase 7` (refuses to
run off-snapshot). Tests: 22 Phase 7 unit tests covering causality
under future perturbation, symmetry and finiteness, horizon
probabilities summing to one, responsibility alignment, shrinkage
direction and limits, the mixture against a manual calculation, the
A2 flag-driven routing, and the refusal to clip a materially non-PSD
matrix.
