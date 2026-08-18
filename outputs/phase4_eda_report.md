# Phase 4 — EDA and Data-Quality Report

**Generated from:** config SHA-256 `54ccd33d…855d6c` ·
data snapshot `manifest_2026-08-06.json` (SHA-256 `6ce35432…715c2`) ·
git commit per [eda_manifest.json](eda_manifest.json)
**Run:** 2026-08-15, `python scripts/run_analysis.py`
**Revised 2026-08-15** (documentation only, before Phase 5; tables
unchanged): figure palette moved to darker validated steps (IEF now
violet; all five slots ≥3:1 contrast on white), long-only covariance
phrasing corrected in §4, overlapping-window caveat added to the
forward-volatility correlation in §3.
**Scope:** descriptive only. **No volatility model, HMM, portfolio
optimization, or backtest was estimated or examined in this phase**, and
nothing below alters the preregistered specification
([docs/research_design.md](../docs/research_design.md), tag
`v0.2.0-preregistered`). Every number in this report is read from the
generated tables in `outputs/tables/`; per-artifact hashes are in
[eda_manifest.json](eda_manifest.json). The analysis refuses to run if any
data file deviates from the frozen SHA-256 snapshot (`verify_snapshot`).

---

## 1. What the data shows

**Sample.** 5,462 trading days, 2004-11-18 to 2026-08-06, five ETFs and
five macro series, no missing values after alignment
(`missingness_audit.csv`).

**Returns are fat-tailed and non-normal everywhere**
(`summary_statistics.csv`). Daily excess kurtosis runs from 2.7 (IEF) to
14.8 (SPY); skewness is negative for every risk asset (SPY −0.31,
IWM −0.51, GLD −0.45) and mildly positive for IEF (+0.11). Jarque-Bera
rejects normality at any conventional level for all five assets. Worst
single days: SPY −11.6%, IWM −14.2%, GLD −10.8%. This is the standard
volatility-clustering picture that motivates Phases 5-6; it neither
confirms nor tests them.

**Long-run performance** (CAGR / annualized vol / max drawdown):

| Asset | CAGR | Ann. vol | Max drawdown (trough date) |
|---|---|---|---|
| SPY | 11.0% | 18.9% | −55.2% (2009-03-09) |
| QQQ | 15.2% | 21.6% | −53.4% (2008-11-20) |
| IWM | 9.0% | 24.1% | −58.6% (2009-03-09) |
| IEF | 3.1% | 6.7% | −23.9% (2023-10-19) |
| GLD | 10.5% | 18.2% | −45.6% (2015-12-17) |

**Volatility moves in persistent episodes.** The 21- and 63-day rolling
volatilities (`rolling_volatility.png`) cluster around the GFC (SPY 21d
realized peaked near 94% annualized), the 2011 downgrade, the 2015-16 and
2018 corrections, COVID (second-largest spike), the 2022 tightening, and
a sharp but brief 2025 episode (realized ~52%).

**Cross-asset structure varies by period** (`subperiod_summaries.csv`,
descriptive only, small samples):

| Period | SPY CAGR | IEF CAGR | SPY-IEF corr | Mean VIX |
|---|---|---|---|---|
| Training window (2004-11 to 2009-12) | 0.8% | 4.9% | −0.36 | 21.3 |
| Pre-2020 (2010-2019) | 13.3% | 4.4% | −0.47 | 16.9 |
| COVID (2020-02 to 2020-12) | 19.2% | 7.0% | −0.50 | 30.6 |
| 2022 tightening (2022-01 to 2023-10) | −5.6% | −10.4% | **+0.13** | 22.0 |
| Post-2020 (2020-01 to 2026-08) | 15.6% | −0.2% | −0.06 | 20.8 |

Two facts matter for interpreting later results. First, the equity-bond
correlation flipped sign in the 2022 tightening: Treasuries amplified
rather than hedged equity losses, and IEF's worst drawdown (−23.9%) sits
in 2022-23, not 2008. Second, the initial training window (which contains
the GFC) has near-zero equity returns and strongly negative equity-bond
correlation, so the earliest expanding-window estimates will inherit a
crisis-heavy history. Both are properties of the era, recorded here so
that later results are read against them; neither changes the design.

## 2. Suspicious observations, examined

Everything flagged by the audits has an economic explanation; **nothing
was corrected, deleted, or smoothed** (`staleness_audit.csv`,
`calendar_audit.csv`).

- **Two calendar gaps longer than 4 days:** the closes before
  2007-01-03 (New Year holiday plus the national day of mourning for
  President Ford, 2007-01-02) and before 2012-10-31 (Hurricane Sandy
  closed the NYSE on 2012-10-29/30). Both are genuine market closures,
  not data holes. No duplicate dates; no weekend rows.
- **DFF is unchanged on 64.8% of days, longest identical run 289
  trading days.** This is the zero-interest-rate policy era, an
  economically real feature of a policy rate, not staleness.
- **DGS2 unchanged on 19.3% of days (DGS10: 8.4%).** H.15 yields are
  quoted to two decimals (1 bp granularity), so unchanged days are
  quantization, concentrated in the near-zero-rate years.
- **Price staleness is negligible:** longest identical-close run is 2-3
  days on any ETF; IEF has the most zero-return days (47 of 5,461,
  0.9%), consistent with a low-volatility bond ETF at cent-level price
  granularity.
- **Raw FRED files contain 172-233 missing rows** (market-holiday NaNs
  printed by FRED). All were filled within the preregistered
  5-trading-day forward-fill limit; the pipeline errors if a gap ever
  exceeds it, and none did. Zero missing values remain after alignment;
  no series required backfill (leading-NaN count is zero for all five).

## 3. Is the VIX / realized-volatility redundancy material?

Yes — material, and quantified (`vix_vs_realized_vol_stats.csv`,
`macro_feature_correlations.csv`):

- Level correlation VIX vs SPY 21-day realized vol: **0.87**
  (log-VIX vs RV: 0.80).
- 21-day change correlation: **0.62**.
- VIX today vs realized vol over the *next* 21 days: **0.71**, and the
  scatter (`vix_vs_realized_vol.png`) shows the familiar implied-premium
  wedge: VIX sits above realized on most days.

A caveat on the 0.71: consecutive 21-day realized-volatility windows
overlap, so the observations behind that correlation are strongly
serially dependent. The number is **descriptive only**; no ordinary
correlation p-value applies, and none is attached.

Implication: in the 4-feature HMM, VIX and realized volatility will
share most of their level variance, so the model may behave largely as a
volatility classifier. This is exactly the scenario the preregistration
anticipated: the main specification keeps both features, and the
**drop-VIX and drop-realized-vol ablations already in the frozen
robustness grid are the designated test** of whether either feature
carries independent regime information. The forward-looking component of
VIX is the a-priori reason to keep it. No specification change is made,
and none of these numbers was used to select features.

Forward-looking realized volatility (the next-21-day series) exists
**only** inside this descriptive statistic and in forecast-evaluation
targets; it is not a column of any feature panel, and the Phase 6
feature builder must pass the `assert_causal` truncation test before any
regime is estimated.

## 4. Do ETF correlations create conditioning problems?

The equity block is highly collinear (`correlation_matrix.csv`):
SPY-QQQ **0.92**, SPY-IWM **0.89**, QQQ-IWM 0.82, while IEF is
negatively correlated with equities (−0.24 to −0.28) and GLD is nearly
orthogonal (0.06-0.21). The full-sample correlation matrix's eigenvalues
are 2.87 / 1.18 / 0.71 / 0.18 / 0.06 (`correlation_conditioning.csv`):
the first principal component carries 57.4% of variance and the
**condition number is 46.5**. Because the portfolios are long-only, the
failure mode is not literal short-against-long offsetting: the highly
correlated equity block can create unstable or concentrated
minimum-variance weights, with allocations lurching between
near-substitutes from one estimation window to the next.

The preregistered design already carries the mitigations: Ledoit-Wolf
shrinkage in the primary comparator and regime strategy, the 40% weight
cap, long-only constraints, and the SPY/IEF/GLD core-universe robustness
run. Phase 7 will report per-rebalance condition numbers for each
estimator. Recorded as a known property; no change required.

## 5. Are any data-policy changes necessary?

**No.** The frozen policies (inner-join alignment, 5-day forward-fill
limit with hard error, no backfill, 1-trading-day macro signal lag,
adjusted closes as total-return proxy) survived contact with the full
audits without exceptions. Any future correction would require a dated
amendment plus a new snapshot manifest per the preregistration rules; as
of this report none is needed.

## 6. Confirmation of scope

No strategy performance, portfolio weights, volatility-model fits,
regime estimates, or model-selection statistics were computed, examined,
or informally previewed in Phase 4. The EDA pipeline is descriptive by
construction (`src/eda.py` contains no estimator or optimizer), and its
outputs feed no trading rule. The preregistered specification is
unchanged; the next phase implemented will be Phase 5 (volatility
forecasting) exactly as frozen.

## Artifacts

Tables: `summary_statistics`, `macro_summary`, `missingness_audit`,
`staleness_audit`, `calendar_audit`, `correlation_matrix`,
`correlation_conditioning`, `macro_feature_correlations`,
`vix_vs_realized_vol_stats`, `subperiod_summaries` (CSV, in
`outputs/tables/`). Figures: `normalized_prices`,
`return_distributions`, `rolling_volatility`, `rolling_correlations`,
`drawdowns`, `macro_features`, `vix_vs_realized_vol` (PNG, in
`outputs/figures/`). SHA-256 of every artifact:
[eda_manifest.json](eda_manifest.json). Reproduce with
`python scripts/run_analysis.py` (fails unless data matches the frozen
snapshot).
