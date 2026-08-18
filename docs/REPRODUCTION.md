# Reproduction

Two distinct modes. They are not interchangeable, and the difference is
not cosmetic.

| | **Exact reproduction** | **Live rebuild** |
|---|---|---|
| Data | The frozen 2026-08-06 snapshot | Whatever the vendors serve today |
| Hashes | Match `manifest_2026-08-06.json` | Will not match |
| Results | Reproduce the paper within stated tolerances | Will differ, possibly materially |
| Who can run it | Anyone holding the snapshot (see below) | Anyone |
| What it demonstrates | The paper's results | That the pipeline runs end to end |

**A live rebuild is not a reproduction of this paper.** Yahoo restates
adjusted closes whenever a distribution occurs, so re-downloading the
same tickers over the same dates returns different numbers than the
snapshot contains. Any document describing a live rebuild as
reproducing the paper is wrong.

---

## Why the snapshot is not in this repository

Neither vendor permits redistribution of the raw data.

**Yahoo Finance** (SPY, QQQ, IWM, IEF, GLD adjusted closes). Yahoo's
Terms of Service grant a *personal, non-transferable, non-exclusive*
licence for the sole purpose of using their service. Redistribution of
the downloaded data is outside that grant. Personal and internal
research use is permitted; publishing the dataset is not.

**FRED / CBOE** (VIXCLS). FRED's API Terms of Use state that a user
must contact the data owner for permission before using third-party
series for anything beyond personal use, and that FRED's provision of
the data does not override the owner's copyright. **VIXCLS is flagged
"Copyrighted: Citation Required" and carries a CBOE copyright notice.**

**FRED / public domain** (DGS2, DGS10, T10Y2Y, DFF). Treasury constant
maturity yields and the effective federal funds rate are US government
data and carry no such restriction.

Because two of the ten series cannot be republished, the repository
ships everything *except* the raw and processed price and VIX files:

- SHA-256 hashes and full metadata for every raw file
  (`data/snapshots/manifest_2026-08-06.json`)
- the download scripts that reconstruct the snapshot
  (`scripts/download_data.py`, `scripts/freeze_snapshot.py`)
- every generated table, figure and audit trail (`outputs/`)
- per-phase provenance manifests tying artifacts to code and data hashes
- the complete analysis code and test suite

A reader can therefore verify every computation, inspect every
intermediate result, and confirm that no number was entered by hand —
without receiving a byte of vendor data.

## Mode 1: Exact reproduction

Requires the frozen snapshot. Two ways to obtain it:

1. **From the author.** The snapshot is 12 CSV files, roughly 3 MB.
   Whether it can be shared depends on the recipient's own licence
   position with the vendors; a researcher with their own Yahoo and
   CBOE access is in a different position than an anonymous downloader.
   Contact details are in the repository metadata.
2. **Rebuild and verify.** Run `scripts/download_data.py`, then
   `scripts/freeze_snapshot.py`, and compare the resulting hashes
   against the committed manifest. If they match, the vendor has not
   restated anything relevant and the reproduction is exact. If they do
   not match, the mismatch itself is informative and is reported by the
   comparison, file by file.

Every analysis script calls `verify_snapshot()` before doing anything
and **refuses to run** when a hash differs. Exact reproduction cannot be
performed accidentally on the wrong data.

## Mode 2: Live rebuild

```bash
python scripts/download_data.py     # today's vendor data
python scripts/run_analysis.py --phase all --allow-unfrozen
python scripts/run_robustness.py --allow-unfrozen
```

Produces a complete set of outputs from current data. Use it to confirm
the pipeline executes, to inspect the code paths, or to extend the
sample. **Do not compare its numbers to the paper's and call a
difference an error.**

## Clean-clone procedure

Verifies that the repository is self-contained. Run in a directory with
no access to an existing virtual environment, cached models, or outputs.

```bash
git clone <repository-url> regime-aware-clean-test
cd regime-aware-clean-test
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest tests\ -q
```

The test suite runs entirely on synthetic fixtures and requires no
market data, so it passes on a bare clone. Then restore the snapshot and
run the pipeline in order:

```bash
.venv\Scripts\python scripts\verify_snapshot.py      # 1. snapshot integrity
.venv\Scripts\python scripts\download_data.py        # 2. (skip if snapshot restored)
.venv\Scripts\python scripts\run_analysis.py --phase 4    # 3. EDA
.venv\Scripts\python scripts\run_analysis.py --phase 5    # 4. volatility
.venv\Scripts\python scripts\run_analysis.py --phase 6    # 5. regimes
.venv\Scripts\python scripts\run_analysis.py --phase 7    # 6. covariance
.venv\Scripts\python scripts\run_analysis.py --phase 8    # 7. weights
.venv\Scripts\python scripts\run_analysis.py --phase 9    # 8. accounting
.venv\Scripts\python scripts\run_analysis.py --phase 10   # 9. performance
.venv\Scripts\python scripts\run_analysis.py --phase 11   # 10. inference
.venv\Scripts\python scripts\run_robustness.py            # 11. robustness
.venv\Scripts\python scripts\verify_robustness.py
.venv\Scripts\python scripts\generate_paper_outputs.py    # 12. paper tables
```

Never copy `.venv`, `outputs/`, or `data/` from an existing working
directory into the clean clone. Doing so tests nothing.

## Pass criteria and tolerances

Reproduction is judged on **numerical agreement within tolerance**, not
on byte-identical artifacts. Floating-point results are not bit-
reproducible across operating systems, BLAS implementations, or thread
counts: the same HMM fit run twice on this machine produced
log-likelihoods differing at 1e-11, which is why fit selection uses a
tolerance band and a lowest-seed tie-break rather than `argmax`.

| Quantity | Expected | Tolerance |
|---|---|---|
| Regime origins | 200 | exact |
| First origin / first execution | 2009-12-31 / 2010-01-04 | exact |
| A2 fallbacks | 4 (2009-12-31, 2012-03-30, 2012-04-30, 2012-05-31) | exact |
| Regime classifications | identical | exact (see note) |
| Selected HMM seeds | identical | exact (see note) |
| Primary ΔSharpe | +0.0210 | ±0.0005 |
| 95% CI | [−0.0748, +0.1147] | ±0.002 per bound |
| One-sided p | 0.327 | ±0.01 |
| Volatility difference | −0.175 pp | ±0.005 pp |
| Robustness specifications | 13 | exact |
| Holm-significant robustness results | 0 | exact |
| Display-table unit verification | all pass | exact |
| Hand-entered results in paper tables | 0 | exact |

**Note on the "exact" classification and seed criteria.** These are
discrete outputs of a continuous computation. They were bit-stable
across two full runs on this machine and should be stable on any
platform, because the nearest probability sits 0.164 from the 0.5
decision boundary — roughly ten orders of magnitude above the observed
1e-11 numerical noise. A platform that changes a classification would
indicate a genuine numerical problem, not ordinary variation.

**Bootstrap quantities** are seeded (12345) and reproduce exactly on a
given platform. Across platforms, tolerances above apply.

## Runtime and memory

Measured on Windows 11, Python 3.12, 8-core CPU, from a warm start.

| Step | Runtime | Peak memory |
|---|---|---|
| Install dependencies | 2–4 min | — |
| Test suite (214 tests) | ~12 s | ~400 MB |
| Data download | 30–60 s | ~200 MB |
| Phase 4 EDA | ~25 s | ~500 MB |
| Phase 5 volatility (1,245 GARCH fits) | ~4 min | ~400 MB |
| Phase 6 regimes (200 refits × 16 starts) | ~7 min | ~350 MB |
| Phase 7 covariance (refits HMM) | ~7 min | ~400 MB |
| Phase 8 weights | ~15 s | ~300 MB |
| Phase 9 accounting | ~40 s | ~600 MB |
| Phase 10 performance | ~30 s | ~500 MB |
| Phase 11 inference (10,000 bootstrap × 6 metrics) | ~3 min | ~800 MB |
| Phase 12 robustness (13 specs, 6 HMM refits) | **~40 min** | ~350 MB |
| **Total** | **~65 min** | **peak ~800 MB** |

The robustness grid dominates. It checkpoints each specification to
`outputs/robustness/partial/` and resumes automatically, so an
interrupted run costs only the unfinished specifications. An earlier
attempt stalled overnight when the machine slept; checkpointing exists
because of that.

Memory stays under 1 GB throughout. The bootstrap processes
replications in batches of 500 specifically to keep the index matrices
from growing to hundreds of megabytes.

## Smoke test for reviewers without the data

```bash
python scripts/smoke_test.py
```

Runs the complete pipeline — features, HMM, covariance, optimization,
accounting, metrics, bootstrap — on synthetic data with a known
embedded volatility regime, in about 30 seconds and with no market data
and no network access. It verifies that the machinery executes and that
the invariants hold (weights sum to one, costs charge on the full trade
sum, the wealth identity closes), not that the paper's numbers are
right. Intended for a reviewer who wants to confirm the code runs
before deciding whether to obtain the snapshot.

## One command for every paper output

```bash
python scripts/generate_paper_outputs.py
```

Regenerates every table and figure the paper references, writes them to
`paper/tables/` and `paper/figures/`, routes every number through
`src/units.py`, verifies each display value against its raw value, and
writes `outputs/results_manifest.json` recording the git commit, config
hash, data hash, and a SHA-256 for each artifact. The LaTeX pulls
numbers only through generated macros, so no result is typed by hand.

---

## Verified clean-clone reproduction (2026-08-16)

Performed on a fresh `git clone` into a new directory, with a new
virtual environment built from `requirements.txt`. No `.venv`, no
`outputs/`, and no cached models were copied from the working
directory. The frozen snapshot was restored by the documented method
and verified before any analysis ran.

| Criterion | Paper | Clean clone | Verdict |
|---|---|---|---|
| Test suite on bare clone (no market data) | 214 pass | 214 pass | PASS |
| Smoke test (synthetic, no network) | 32/32 | 32/32 | PASS |
| Snapshot hashes | 13 match | 13 match | PASS |
| Regime origins | 200 | 200 | PASS |
| A2 fallbacks | 4 | 4 | PASS |
| Regime classifications | — | identical | PASS |
| Selected HMM seeds | — | identical | PASS |
| Primary ΔSharpe | +0.0210 | +0.0210 | PASS |
| 95% CI | [−0.0748, +0.1147] | [−0.0748, +0.1147] | PASS |
| One-sided p | 0.3273 | 0.3273 | PASS |
| Volatility difference | −0.1754 pp | −0.1754 pp | PASS |
| Robustness specifications | 13 | 13 | PASS |
| Holm-significant robustness results | 0 | 0 | PASS |
| Max ΔSharpe deviation across 13 specs | — | 6.1e-09 | PASS |
| Display-table unit verification | all pass | all pass | PASS |
| LaTeX macros | 10 | 10, byte-identical | PASS |

**Floating-point behaviour.** Despite floating-point differences up to
**1.45e-11**, all selected seeds, classifications, and reported
conclusions reproduced. The tolerance-based tie-breaking rule provided a
safeguard against numerically immaterial ranking differences.

**The safeguard was not shown to be necessary.** Replaying the
counterfactual over both runs' initialization records — 3,200 fits
each — plain `argmax` would have selected the **same seed at all 200
origins**, so on this evidence the tolerance rule changed nothing.

What the records do show is that the conditions making `argmax` fragile
are present: 108 of 200 origins had two or more fits inside the
tolerance band, and at 193 of 200 origins the gap between the best and
second-best log-likelihood was below 1e-9 (median exactly zero, i.e.
multiple starts converging on the same optimum). The rule is therefore
justified as insurance against a ranking flip that these two runs did
not produce, not as a fix for an observed failure.

The two rules **disagree with each other at 144 of 200 origins within a
single run**, because the tolerance (1e-10 relative, roughly 5e-7
absolute here) is far wider than exact equality: `argmax` takes the
strictly highest log-likelihood while the tolerance rule takes the
lowest seed among fits within the band.

**What that disagreement does and does not establish.** Each rule was
stable across the two runs; that is established. Whether the *results*
are unchanged under argmax selection is **not** established, because the
argmax-selected fits were never propagated through Phases 6-11. Doing so
would require rerunning the regime, covariance, weight, accounting,
performance and inference phases under the alternative rule and
comparing the outputs. Until that is done, no claim is made that fits
inside the band are materially equivalent.

| Claim | Status |
|---|---|
| Plain argmax reproduced across both runs | Established |
| Tolerance selection reproduced across both runs | Established |
| Tolerance was necessary for reproducibility | **Not demonstrated** |
| The two rules selected different seeds at 144/200 origins | Established |
| Full results unchanged under argmax selection | **Not tested** |

The counterfactual is reproducible: `python
scripts/test_argmax_counterfactual.py RUN_A RUN_B`, with results in
`outputs/robustness/argmax_counterfactual.csv`.

Numerical reproduction and hash-identical artifacts are different
standards; this project meets the former by design and does not claim
the latter.

**Not yet verified:** the LaTeX build. No `.tex` source exists beyond
the generated macros and tables, so the PDF build will be checked when
the paper is written.
