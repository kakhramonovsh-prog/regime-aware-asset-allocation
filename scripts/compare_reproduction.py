"""Compare this run's artifacts against an independent run's.

The manuscript reports the largest floating-point disagreement observed
between two independent executions. That number was transcribed by hand
into ``docs/REPRODUCTION.md`` and then re-typed into the paper, which is
the provenance the number census exists to eliminate. This script
recomputes it from the two runs' artifacts and writes a machine-readable
record that the macro generator reads.

    python scripts/compare_reproduction.py --other ../regime-aware-clean-test

Both directories must contain a completed run. The comparison is
symmetric and does not modify either run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD = PROJECT_ROOT / "outputs" / "reproduction_comparison.json"


def _commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def compare_probabilities(this: Path, other: Path) -> dict:
    """Largest absolute disagreement in filtered state probabilities."""
    left = pd.read_parquet(this / "outputs" / "regimes" / "realtime_probabilities.parquet")
    right = pd.read_parquet(other / "outputs" / "regimes" / "realtime_probabilities.parquet")

    common_index = left.index.intersection(right.index)
    common_columns = [c for c in left.columns if c in right.columns]
    numeric = [c for c in common_columns
               if pd.api.types.is_numeric_dtype(left[c])
               and pd.api.types.is_numeric_dtype(right[c])]

    difference = (left.loc[common_index, numeric].to_numpy(dtype=float)
                  - right.loc[common_index, numeric].to_numpy(dtype=float))
    absolute = np.abs(difference)

    # Both runs carry NaN in the same cells (columns that only apply to
    # some origins). A cell that is NaN in one run and finite in the
    # other is a real disagreement and must not be silently skipped.
    left_missing = np.isnan(left.loc[common_index, numeric].to_numpy(dtype=float))
    right_missing = np.isnan(right.loc[common_index, numeric].to_numpy(dtype=float))
    mismatched_missing = int((left_missing != right_missing).sum())
    finite = np.isfinite(absolute)

    return {
        "rows_compared": int(len(common_index)),
        "columns_compared": numeric,
        "max_absolute_difference": float(absolute[finite].max()),
        "mean_absolute_difference": float(absolute[finite].mean()),
        "n_exactly_equal": int((absolute[finite] == 0).sum()),
        "n_compared": int(finite.sum()),
        "n_missing_in_both": int((left_missing & right_missing).sum()),
        "n_missing_in_one_run_only": mismatched_missing,
    }


def compare_selected_seeds(this: Path, other: Path) -> dict:
    """Whether the two runs selected the same fit at every origin."""
    left = pd.read_csv(this / "outputs" / "regimes" / "selected_fit_diagnostics.csv")
    right = pd.read_csv(other / "outputs" / "regimes" / "selected_fit_diagnostics.csv")
    merged = left.merge(right, on="date", suffixes=("_this", "_other"))

    loglik = np.abs(merged["loglik_this"] - merged["loglik_other"])
    record = {
        "origins_compared": int(len(merged)),
        "max_absolute_loglik_difference": float(loglik.max()),
    }

    # The selected seed itself lives in the revision-stability artifact.
    left_seeds = pd.read_csv(this / "outputs" / "regimes" / "model_revision_stability.csv")
    right_seeds = pd.read_csv(other / "outputs" / "regimes" / "model_revision_stability.csv")
    seeds = left_seeds.merge(right_seeds, on="date", suffixes=("_this", "_other"))
    record["refits_compared"] = int(len(seeds))
    record["selected_seed_disagreements"] = int(
        (seeds["selected_seed_curr_this"] != seeds["selected_seed_curr_other"]).sum()
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--other", required=True,
                        help="path to the independent run's repository")
    args = parser.parse_args()

    other = Path(args.other).resolve()
    if not (other / "outputs" / "regimes").exists():
        raise SystemExit(f"{other} does not contain a completed run")

    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "this_run": {"path": str(PROJECT_ROOT), "commit": _commit(PROJECT_ROOT)},
        "other_run": {"path": str(other), "commit": _commit(other)},
        "filtered_probabilities": compare_probabilities(PROJECT_ROOT, other),
        "selected_fits": compare_selected_seeds(PROJECT_ROOT, other),
    }

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    probabilities = record["filtered_probabilities"]
    print(f"Compared {probabilities['rows_compared']} rows x "
          f"{len(probabilities['columns_compared'])} columns")
    print(f"  max |difference| = {probabilities['max_absolute_difference']:.6e}")
    print(f"  exactly equal    = {probabilities['n_exactly_equal']} / "
          f"{probabilities['n_compared']}")
    print(f"  selected fits    = {record['selected_fits']}")
    print(f"\nWrote {RECORD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
