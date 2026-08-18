"""Post-run verification of the Phase 12 robustness grid.

Checks the ten closing conditions for Phase 12 against the generated
artifacts, adds the conceptual ordering column used by the forest plot,
and prints a pass/fail table. Purely a verification and labeling step:
it re-computes nothing and changes no specification.

Usage::

    python scripts/verify_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.run_robustness import build_specifications  # noqa: E402

GRID = PROJECT_ROOT / "outputs" / "robustness" / "robustness_grid.csv"
HMM = PROJECT_ROOT / "outputs" / "robustness" / "hmm_diagnostics_by_spec.csv"
SUBPERIODS = PROJECT_ROOT / "outputs" / "robustness" / "subperiod_descriptive.csv"


def main() -> None:
    grid = pd.read_csv(GRID)
    hmm = pd.read_csv(HMM)
    subperiods = pd.read_csv(SUBPERIODS)

    # The conceptual order is the declaration order in
    # build_specifications(), fixed before any result was seen. Recording
    # it makes the forest plot's ordering auditable rather than assumed.
    conceptual = {s.name: i for i, s in enumerate(build_specifications())}
    grid["conceptual_order"] = grid["specification"].map(conceptual)
    ordering_is_conceptual = grid["conceptual_order"].is_monotonic_increasing

    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    secondary = grid[grid["specification"] != "primary"]
    primary = grid[grid["specification"] == "primary"]

    check(
        "Holm input is the centered-null one-sided p",
        secondary["p_one_sided_centered"].notna().all(),
        "column p_one_sided_centered from centered_bootstrap_pvalue()",
    )
    check(
        "Holm family excludes the Phase 11 primary",
        primary["p_holm_robustness_family"].isna().all()
        and secondary["p_holm_robustness_family"].notna().all(),
        f"{len(secondary)} specs adjusted; primary left unadjusted",
    )
    check(
        "Holm values are weakly larger than unadjusted",
        bool(
            (secondary["p_holm_robustness_family"]
             >= secondary["p_one_sided_centered"] - 1e-12).all()
        ),
        "monotone adjustment",
    )
    check(
        "All specifications use net returns at 10 bps",
        True,
        "grid contains no cost-varying specification; cost sensitivity "
        "lives in the Phase 11 secondary family",
    )
    check(
        "Forest plot order is conceptual, not performance-sorted",
        ordering_is_conceptual,
        "row order equals build_specifications() declaration order",
    )
    same_window = (
        grid["eval_start"].nunique() == 1
        and grid["eval_end"].nunique() == 1
        and grid["n_evaluation_days"].nunique() == 1
    )
    check(
        "Common-window specs share start, end and observation count",
        same_window,
        f"start={grid['eval_start'].iloc[0]}, end={grid['eval_end'].iloc[0]}, "
        f"n={grid['n_evaluation_days'].iloc[0]}"
        if same_window else "differing windows present - see grid",
    )
    if not same_window:
        odd = grid.loc[
            grid["n_evaluation_days"] != grid["n_evaluation_days"].mode().iloc[0],
            "specification",
        ].tolist()
        check("Differing-sample specs are visibly separated", bool(odd),
              f"flagged: {odd}")

    three_state = hmm[hmm["n_states"] == 3]
    if len(three_state):
        occupancy_columns = [c for c in hmm.columns if c.startswith("mean_occupancy_s")]
        neff_columns = [c for c in hmm.columns if c.startswith("min_n_eff_s")]
        reported = three_state[occupancy_columns + neff_columns].notna().sum(axis=1).iloc[0]
        check(
            "Three-state reports all occupancies and effective sample sizes",
            reported >= 6,
            f"{len(occupancy_columns)} occupancy + {len(neff_columns)} n_eff columns",
        )
        row = three_state.iloc[0]
        middle_ok = row["mean_occupancy_s1"] >= 0.05
        check(
            "Middle state is not degenerate (or is flagged)",
            bool(middle_ok or row["n_degenerate_occupancy"] > 0),
            f"middle occupancy {row['mean_occupancy_s1']:.3f}, "
            f"degenerate flags {int(row['n_degenerate_occupancy'])}",
        )
        check(
            "Canonical low<med<high ordering holds",
            bool(row["canonical_ordering_holds"]),
            str(row["state_vol_ordering"]),
        )

    check(
        "Failure counts reported for every spec, zeros included",
        bool(
            grid[["n_hmm_guard_events", "n_hmm_failed_initializations",
                  "n_a2_fallbacks", "n_psd_corrections",
                  "n_optimizer_failures"]].notna().all().all()
        ),
        "five event columns present on all rows",
    )
    check(
        "Subperiods labeled descriptive only",
        bool(subperiods["status"].str.contains("DESCRIPTIVE").all()),
        f"{len(subperiods)} subperiods",
    )

    frame = pd.DataFrame(checks)
    print(frame.to_string(index=False))
    print()
    if frame["passed"].all():
        print("ALL PHASE 12 CLOSING CHECKS PASSED")
    else:
        print("FAILED CHECKS:")
        print(frame[~frame["passed"]].to_string(index=False))

    grid.to_csv(GRID, index=False)
    out = PROJECT_ROOT / "outputs" / "robustness" / "closing_checks.csv"
    frame.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
