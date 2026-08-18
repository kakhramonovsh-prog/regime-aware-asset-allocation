"""Download raw data and build the processed panels.

Usage::

    python scripts/download_data.py [--config config/config.yaml] [--end YYYY-MM-DD]

Steps:

1. Download daily OHLCV for each ticker in the config from Yahoo
   Finance and each macro series from FRED; write them unmodified to
   ``data/raw`` together with ``download_metadata.json``.
2. Build the aligned adjusted-close panel and macro panel (the only
   transformations are the inner-join alignment and the limited
   forward-fill documented in ``data/README.md``) and write them to
   ``data/processed``.

Analysis and modelling never happen here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data_loader, preprocessing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="path to config.yaml",
    )
    parser.add_argument(
        "--end", default=None,
        help="optional end date YYYY-MM-DD (default: config end_date, else latest)",
    )
    args = parser.parse_args()

    config = data_loader.load_config(args.config)
    data_cfg = config["data"]
    start = data_cfg["start_date"]
    end = args.end or data_cfg.get("end_date")
    tickers = list(data_cfg["tickers"])
    fred_series = list(data_cfg["fred_series"])
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]

    print(f"Downloading {len(tickers)} tickers from Yahoo Finance "
          f"({start} to {end or 'latest'}): {', '.join(tickers)}")
    price_frames = data_loader.download_prices(tickers, start=start, end=end)

    print(f"Downloading {len(fred_series)} series from FRED: {', '.join(fred_series)}")
    macro_frames = data_loader.download_fred(fred_series, start=start, end=end)

    meta_path = data_loader.save_raw(price_frames, macro_frames, raw_dir)
    print(f"Raw files and metadata written to {raw_dir} ({meta_path.name})")

    prices = preprocessing.build_price_panel(price_frames)
    macro = preprocessing.build_macro_panel(
        macro_frames,
        trading_days=prices.index,
        ffill_limit=data_cfg.get("macro_ffill_limit", 5),
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    prices_path = processed_dir / "prices.csv"
    macro_path = processed_dir / "macro.csv"
    prices.to_csv(prices_path)
    macro.to_csv(macro_path)

    print(f"\nProcessed panels written to {processed_dir}:")
    print(f"  prices.csv  {prices.shape[0]} days x {prices.shape[1]} assets  "
          f"({prices.index.min().date()} to {prices.index.max().date()})")
    print(f"  macro.csv   {macro.shape[0]} days x {macro.shape[1]} series")
    remaining_na = macro.isna().sum()
    if remaining_na.any():
        print("  NaN counts in macro.csv (leading values before a series begins):")
        for series_id, n in remaining_na[remaining_na > 0].items():
            print(f"    {series_id}: {n}")


if __name__ == "__main__":
    main()
