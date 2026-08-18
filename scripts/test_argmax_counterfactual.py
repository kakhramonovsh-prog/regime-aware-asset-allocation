"""Test whether plain argmax selection would have diverged across runs.

The Phase 6 fit-selection rule treats log-likelihoods within a relative
tolerance as tied and takes the lowest seed, instead of using `argmax`.
The rule was introduced as a portability safeguard. Whether it was
*necessary* is a separate, testable question, and this script answers it
from the stored initialization records of two independent runs rather
than from argument.

    python scripts/test_argmax_counterfactual.py RUN_A_DIR RUN_B_DIR

Each directory must contain `outputs/regimes/all_initializations.csv`.
Writes `outputs/robustness/argmax_counterfactual.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load(run_dir: Path) -> pd.DataFrame:
    path = Path(run_dir) / "outputs" / "regimes" / "all_initializations.csv"
    if not path.exists():
        sys.exit(f"Not found: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def argmax_selection(records: pd.DataFrame) -> pd.Series:
    """Which seed a plain argmax rule would pick at each origin."""
    usable = records[records["usable"]]
    return usable.loc[usable.groupby("date")["loglik"].idxmax()].set_index("date")["seed"]


def tolerance_selection(records: pd.DataFrame) -> pd.Series:
    """Which seed the implemented rule actually picked."""
    return records[records["selected"]].set_index("date")["seed"]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(Path(sys.argv[1])), load(Path(sys.argv[2]))

    argmax_a, argmax_b = argmax_selection(a), argmax_selection(b)
    tol_a, tol_b = tolerance_selection(a), tolerance_selection(b)
    origins = argmax_a.index.intersection(argmax_b.index)

    argmax_divergences = int((argmax_a.loc[origins] != argmax_b.loc[origins]).sum())
    tolerance_divergences = int((tol_a.loc[origins] != tol_b.loc[origins]).sum())
    rules_disagree = int((argmax_a.loc[origins] != tol_a.loc[origins]).sum())

    # How close are the top two fits? A small gap is what would make an
    # argmax ranking flip possible in the first place.
    usable = a[a["usable"]].sort_values(["date", "loglik"], ascending=[True, False])
    grouped = usable.groupby("date")["loglik"]
    gaps = grouped.nth(0).to_numpy() - grouped.nth(1).to_numpy()
    within_band = a.groupby("date")["within_selection_tol"].sum()

    findings = pd.DataFrame([
        {"quantity": "origins compared", "value": len(origins)},
        {"quantity": "argmax selects a different seed across runs",
         "value": argmax_divergences},
        {"quantity": "tolerance rule selects a different seed across runs",
         "value": tolerance_divergences},
        {"quantity": "origins where the two rules disagree within one run",
         "value": rules_disagree},
        {"quantity": "origins with >1 fit inside the tolerance band",
         "value": int((within_band > 1).sum())},
        {"quantity": "origins with best-vs-second gap < 1e-9",
         "value": int((gaps < 1e-9).sum())},
        {"quantity": "median best-vs-second gap", "value": float(np.median(gaps))},
    ])

    print(findings.to_string(index=False))
    print()
    if argmax_divergences == 0:
        print("FINDING: plain argmax would have selected identically at every")
        print("origin. The tolerance rule was NOT necessary on this evidence.")
        print("It remains defensible as insurance: the conditions that make a")
        print("ranking flip possible are present, but no flip occurred.")
    else:
        print(f"FINDING: plain argmax diverges at {argmax_divergences} origins")
        print("across runs. The tolerance rule prevented a real instability.")

    out = PROJECT_ROOT / "outputs" / "robustness" / "argmax_counterfactual.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    findings.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
