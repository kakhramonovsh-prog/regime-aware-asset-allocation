"""Freeze the current data snapshot: SHA-256 manifest of every data file.

Usage::

    python scripts/freeze_snapshot.py

Writes ``data/snapshots/manifest_<last-data-date>.json`` recording, for
every file in ``data/raw`` and ``data/processed``: SHA-256 hash, size,
row count, and date coverage; plus the git commit, a hash of the
configuration files, package versions, and the download timestamp from
``download_metadata.json``.

The manifest is committed; the data files themselves are not. Anyone
re-running the downloader can verify byte-identical data against the
hashes (or see exactly where the vendor restated history).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PINNED_PACKAGES = [
    "numpy", "pandas", "scipy", "statsmodels", "arch", "hmmlearn",
    "scikit-learn", "yfinance", "pandas-datareader", "matplotlib",
    "PyYAML", "pytest",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_csv(path: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return {
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "rows": int(len(df)),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
    }


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    snapshot_dir = PROJECT_ROOT / "data" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    meta_file = raw_dir / "download_metadata.json"
    if not meta_file.exists():
        sys.exit("No download_metadata.json found. Run scripts/download_data.py first.")
    download_meta = json.loads(meta_file.read_text(encoding="utf-8"))

    files: dict[str, dict] = {}
    last_dates = []
    for directory in (raw_dir, processed_dir):
        for path in sorted(directory.glob("*.csv")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            files[rel] = describe_csv(path)
            last_dates.append(files[rel]["last_date"])
    files["data/raw/download_metadata.json"] = {
        "sha256": sha256_of(meta_file),
        "bytes": meta_file.stat().st_size,
    }

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    config_hash = hashlib.sha256()
    for name in ("config.yaml", "analysis_plan.yaml"):
        config_hash.update((PROJECT_ROOT / "config" / name).read_bytes())

    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "downloaded_at_utc": download_meta["downloaded_at_utc"],
        "git_commit": git_commit,
        "config_sha256": config_hash.hexdigest(),
        "python_version": sys.version.split()[0],
        "package_versions": {
            pkg: importlib_metadata.version(pkg) for pkg in PINNED_PACKAGES
        },
        "files": files,
    }

    last_data_date = max(last_dates)
    out_path = snapshot_dir / f"manifest_{last_data_date}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Snapshot manifest written: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  files hashed: {len(files)}  |  git commit: {git_commit[:12]}")
    print(f"  data through: {last_data_date}")


if __name__ == "__main__":
    main()
