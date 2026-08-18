# Data Documentation

No data files are committed to the repository. Everything below is
reproduced by:

```bash
python scripts/download_data.py
```

The script writes raw vendor files to `data/raw/` and the aligned
analysis panels to `data/processed/`. `data/raw/download_metadata.json`
records the download timestamp, source, date coverage, and row count of
every file, so the exact sample used in any run is auditable.

## Sources

| Source | Access | Series |
|---|---|---|
| Yahoo Finance | `yfinance` Python package | ETF daily OHLCV + adjusted close |
| FRED (St. Louis Fed) | `pandas-datareader`, no API key | VIX, Treasury yields, fed funds |

Yahoo Finance is a free vendor and its history can be revised
(dividend adjustments restate past adjusted closes). The metadata file
plus pinned download dates make each snapshot reproducible; for a
published paper, a cross-check against CRSP or Tiingo is listed as
future work.

## Data dictionary — raw files (`data/raw/`)

### Asset prices: `prices_<TICKER>.csv`

One file per ticker. Daily frequency (exchange trading days). Columns
as delivered by Yahoo Finance: `Open, High, Low, Close, Adj Close,
Volume`. Prices in USD; volume in shares. `Adj Close` is adjusted for
splits and dividends and is the basis for all return calculations
(a proxy for total return with reinvested distributions).

| Ticker | Instrument | Role in universe | First trade date |
|---|---|---|---|
| SPY | SPDR S&P 500 ETF | US large-cap equities | pre-sample (1993) |
| QQQ | Invesco QQQ Trust | Nasdaq-100 / large growth | pre-sample (1999) |
| IWM | iShares Russell 2000 ETF | US small-cap equities | pre-sample (2000) |
| IEF | iShares 7-10Y Treasury ETF | intermediate duration | pre-sample (2002) |
| GLD | SPDR Gold Shares | gold bullion | 2004-11-18 (binds sample start) |

### Macro series: `fred_<SERIES>.csv`

| Series ID | Description | Units | Native frequency |
|---|---|---|---|
| VIXCLS | CBOE Volatility Index, close | index points | daily (trading days) |
| DGS2 | 2-Year Treasury constant maturity | percent p.a. | daily (bond trading days) |
| DGS10 | 10-Year Treasury constant maturity | percent p.a. | daily (bond trading days) |
| T10Y2Y | 10Y minus 2Y Treasury spread | percentage points | daily (bond trading days) |
| DFF | Effective Federal Funds Rate | percent p.a. | daily (all calendar days) |

`DFF` is used instead of `FEDFUNDS` because `FEDFUNDS` is a monthly
average. `T10Y2Y` is downloaded as a cross-check; the slope used in
analysis is computed explicitly as `DGS10 - DGS2`.

## Processed files (`data/processed/`)

### `prices.csv`

Adjusted-close panel, dates x tickers. **Alignment:** inner join —
only dates on which all five ETFs have an adjusted close are kept, so
the panel starts at GLD's first trading day (late November 2004) and
the effective analysis sample begins in early 2005.

### `macro.csv`

Macro panel reindexed to `prices.csv` trading days.

**Missing-value treatment (deliberate, tested):**

* Forward-fill with a **5-trading-day limit**. A yield or index close
  is a stock variable: after a bond-market holiday, the last observed
  value is exactly the information an investor holds. The limit turns
  any longer gap into a hard error instead of silently propagating
  stale data.
* Weekend/holiday observations (e.g. `DFF`) carry forward into the
  next trading day via a union-reindex before filling.
* **No backward fill anywhere.** Values before a series' first
  observation stay `NaN` and are reported by the download script.

## Transformations downstream

Returns, rolling volatility/correlation/covariance, drawdowns, yield
slope, and VIX changes are computed in `src/preprocessing.py`. Every
rolling feature is backward-looking, and the test suite verifies via
truncation tests (`assert_causal`) that no feature value changes when
future data is removed.

## Known limitations

* Yahoo adjusted closes ignore taxes and assume frictionless dividend
  reinvestment at the close.
* The five-ETF universe was selected with hindsight knowledge that
  these funds survived and grew; conclusions are conditional on this
  universe (survivorship-aware framing is discussed in the README).
* FRED daily Treasury yields are par yields from the H.15 release, not
  zero-coupon yields.
