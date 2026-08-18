# Does Regime Conditioning Improve Volatility-Aware Asset Allocation?

**A prospectively specified walk-forward evaluation**, 2010–2026.

> The analysis plan was time-stamped in a version-controlled repository
> before model estimation rather than lodged with a public registry, so
> the study is described as *prospectively specified* rather than
> *preregistered*.

[![paper](https://img.shields.io/badge/paper-15%20pages-blue)](outputs/paper_build/inspected/main.pdf)
[![release](https://img.shields.io/badge/release-v0.13.1-informational)](../../releases/tag/v0.13.1)
[![tests](https://img.shields.io/badge/tests-268%20passing-brightgreen)](tests/)

---

## The question

You can build a minimum-variance portfolio two ways: estimate the
covariance matrix from a rolling window, or condition that estimate on
which volatility state the market appears to be in. The second should
work better in principle. Does it pay, out of sample, after costs?

Everything turns on the comparator. Regime-aware allocation beats equal
weight and 60/40 by a wide margin, and that comparison establishes
nothing, because those benchmarks differ in optimization, covariance
estimation and regime conditioning at the same time. This study compares
against **rolling Ledoit–Wolf minimum variance**, which shares every
ingredient except the regime signal, so the difference isolates the
regime signal.

## The answer: no reliable improvement

> **ΔSharpe = +0.021**, 95% CI **[−0.075, +0.115]**, *p* = 0.327.
> The null is not rejected.

The mechanism is visible even though the effect is not. The regime-aware
portfolio did realize **0.175 pp lower annualized volatility**, the
outcome the theory predicts and the one secondary interval excluding
zero. Getting there took **3.2× the turnover**, and the estimated
advantage falls monotonically as costs rise, from +0.031 at zero to
+0.011 at 20 bps.

Three findings temper the point estimate further:

- The sign **reverses** under a three-state model and when realized
  volatility is dropped from the feature set.
- It **reverses again** between the pre- and post-2020 halves.
- After Holm adjustment across the robustness family, **nothing survives**.

Read that as inconclusive about small effects rather than as evidence
that regime conditioning has no value. With roughly 16.6 years of daily
data the study still cannot exclude economically modest gains or losses.
The binding constraint is information rather than observation count: the
strategy makes only about 200 rebalancing decisions.

One unplanned finding is labelled hypothesis-generating rather than
confirmatory. Tightening the weight cap from 40% to 30% moved the
estimate more than any modelling choice examined, and roughly
three-quarters of that change traces to altered exposures rather than to
cost savings.

## Why the negative result is worth reading

Backtests that report wins are easy to produce. This one was built so
that a win would have been hard to manufacture:

| Control | What it prevents |
|---|---|
| **Prospective specification**, git-tagged before any model ran | Choosing the hypothesis after seeing results |
| **Matched comparator** differing only by the regime signal | Attributing optimization's gains to regimes |
| **Filtered** state probabilities, never smoothed | Trading on information the investor lacked |
| **Truncation tests** | Look-ahead leaking through feature construction |
| **Frozen SHA-256 data snapshot** | Silent vendor restatements moving results |
| **Costs on full traded notional**, multiplicative | Flattering the higher-turnover strategy |
| **Paired bootstrap**, identical indices per replication | Breaking contemporaneous pairing |
| **Holm adjustment** over a bounded, pre-declared family | Reporting the best of thirteen tries |
| **Four dated amendments** | Silent post-hoc changes to the frozen plan |

These constraints reduced researcher discretion without removing it.
Implementation choices were made throughout, and four amendments modified
the frozen plan after it was tagged, each dated and each recording what
was known at the time.

Two independent CI builds from fresh checkouts produce a byte-identical
PDF. The title date and the PDF metadata are pinned, and a build check
fails if `\today` reappears in the manuscript source.

## Methods

Five liquid US ETFs (SPY, QQQ, IWM, IEF, GLD), monthly rebalancing,
2010-01-04 to 2026-08-06, net of 10 bps.

- **Regimes.** Two-state Gaussian HMM refit at every rebalance on an
  expanding window; four standardized features; 16 predetermined
  initializations with canonical state relabeling and a tolerance-based
  fit-selection rule.
- **Volatility.** 63-day historical, EWMA (λ = 0.94) and GARCH(1,1)
  compared out of sample under QLIKE. EWMA was fixed *ex ante* as the
  model feeding the portfolio.
- **Covariance.** State-conditional matrices shrunk toward unconditional
  Ledoit–Wolf by effective sample size, mixed over horizon-averaged
  state probabilities.
- **Backtest.** Signal at close *t*, execution at close *t+1*, daily
  drift recurrence, costs charged on full traded notional.
- **Inference.** Stationary bootstrap (10,000 replications, mean block
  21 days) on paired daily excess returns; Newey–West for mean
  differences; Holm across the robustness family.

## Reproduce it

The raw data cannot be redistributed. Yahoo grants a personal,
non-transferable licence and VIXCLS carries a CBOE copyright. This
repository therefore ships SHA-256 hashes and full metadata for every
raw file, the download scripts, and every generated table and figure,
but no vendor data. **A live re-download does not reproduce this paper**,
because Yahoo restates adjusted closes when distributions occur.
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) sets out both modes and
their tolerances.

```bash
pip install -r requirements.txt
```

Verify the analysis without any vendor data:

```bash
pytest
```

Rebuild every number in the paper from stored results:

```bash
python scripts/generate_paper_outputs.py
```

Confirm that no empirical result is hand-typed and that every citation is
verified:

```bash
python scripts/number_census.py && python scripts/check_citations.py --release
```

Live rebuild from freshly downloaded data, whose results will differ:

```bash
python scripts/download_data.py
python scripts/run_analysis.py --phase all --allow-unfrozen
```

## Repository map

| Path | What is there |
|---|---|
| [`paper/`](paper/) | Manuscript source, generated macros and tables, bibliography, citation audit, numeric census |
| [`outputs/paper_build/inspected/`](outputs/paper_build/inspected/) | **The released PDF and its build log** |
| [`docs/research_design.md`](docs/research_design.md) | The preregistration, plus amendments A1–A4 with dates |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Reproduction modes, tolerances, data licensing |
| [`config/analysis_plan.yaml`](config/analysis_plan.yaml) | Machine-readable frozen plan; a test fails if runtime config drifts from it |
| [`src/`](src/) | Library: regimes, covariance, optimization, backtest, metrics, inference |
| [`scripts/`](scripts/) | Pipeline, robustness, paper build, and the audit gates |
| [`outputs/`](outputs/) | Every generated table, figure and per-phase manifest |
| [`tests/`](tests/) | 268 tests, including a guard for each failure this project hit |

## Release

`v0.13.1`. PDF SHA-256
`3dc678f40de5a7c2fc64f79191fd5488770dc8bed2e9fe7ea31d270106d4ae58`,
15 pages, built by GitHub Actions on TeX Live 2024. Six rounds of
page-by-page inspection are logged in
[`outputs/paper_build/visual_inspection.md`](outputs/paper_build/visual_inspection.md).

## Limitations

One market, one asset-class mix, one regime specification family.
Choosing five funds that survived to 2026 embeds selection. The cost
model excludes market impact, taxes and borrowing constraints. Whether
the finding extends to other universes, frequencies or cost levels
remains untested.

## License

Code under [MIT](LICENSE). Data terms in
[`DATA_LICENSE.md`](DATA_LICENSE.md); no vendor data is redistributed.

---

Shakhrukh Kakhramonov, Baruch College, Zicklin School of Business
