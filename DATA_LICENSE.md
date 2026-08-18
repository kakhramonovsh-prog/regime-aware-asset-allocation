# Data Licensing

The MIT licence in [`LICENSE`](LICENSE) covers **the code in this
repository only**. It does not cover the market and macroeconomic data
the code downloads, which belong to third parties under their own terms.

## Sources and their terms

| Series | Source | Redistribution | In this repository |
|---|---|---|---|
| SPY, QQQ, IWM, IEF, GLD (adjusted closes) | Yahoo Finance via `yfinance` | **Not permitted** | No — hashes only |
| VIXCLS | CBOE, via FRED | **Not permitted without CBOE permission** | No — hashes only |
| DGS2, DGS10, T10Y2Y, DFF | US Treasury / Federal Reserve, via FRED | Permitted (public domain) | No — omitted for consistency |

**Yahoo Finance.** Yahoo's Terms of Service grant a personal,
royalty-free, non-transferable, non-assignable, revocable, non-exclusive
licence to use the service. Downloading data for personal or internal
research sits inside that grant; republishing the dataset does not.

**CBOE VIX (VIXCLS).** FRED distributes this series under a CBOE
copyright and flags it "Copyrighted: Citation Required." FRED's API
Terms of Use state that users must obtain permission from the data owner
before using third-party series for anything beyond personal use, and
that FRED's provision of the data does not override the owner's
copyright. Required citation: *Copyright, Chicago Board Options
Exchange, Inc. Reprinted with permission.*

**Treasury and federal funds series.** US government data, no
restriction. Attribution to FRED is customary and appreciated.

## What this repository does publish

Everything derived, and nothing raw:

- SHA-256 hashes, row counts, date coverage and download timestamps for
  every raw file
- the scripts that reconstruct the dataset from the vendors
- computed results: returns, portfolio weights, covariance diagnostics,
  performance metrics, bootstrap draws, robustness tables, figures
- provenance manifests linking every artifact to the code and data that
  produced it

These are research outputs computed from the data, not the data itself,
which is the standard position for empirical finance papers built on
licensed sources. A reader can audit every step of the analysis without
receiving vendor data.

## Obtaining the data

Run `python scripts/download_data.py` with your own access. Both
vendors are free at the volumes this project uses and neither requires
an API key. Then run `python scripts/freeze_snapshot.py` and compare
against `data/snapshots/manifest_2026-08-06.json` to see whether the
vendor has restated anything since the frozen snapshot.

## Citation

If you use this code, cite the working paper (see `README.md`). If you
use the VIX series obtained through it, cite CBOE as above. If you use
the Treasury series, cite the Federal Reserve Bank of St. Louis FRED
database.
