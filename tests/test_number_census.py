"""Tests for the numeric-literal census and the result inventory.

Substring matching for hand-typed numbers was unsound in both
directions: the volatility interval ``[-0.327, -0.033]`` was caught only
because ``0.327`` happened to collide with the p-value, and the
twelve-value inventory it belonged to had missed several results
entirely. These tests guard the exhaustive replacement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import number_census as census  # noqa: E402

INVENTORY = PROJECT_ROOT / "paper" / "result_inventory.csv"
CENSUS = PROJECT_ROOT / "paper" / "number_census.csv"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("a difference of 0.021", ["0.021"]),
    ("10{,}000 replications", ["10{,}000"]),
    ("net of 10 basis points", ["10"]),
    ("$-$0.175 pp", ["0.175"]),
])
def test_digits_are_extracted(text, expected):
    assert census.NUMBER.findall(text) == expected


def test_macro_invocations_are_not_mistaken_for_literals():
    """A generated value must not be counted as a hand-typed one."""
    assert census.NUMBER.findall(r"earned \primarySharpeDiff{} higher") == []
    assert census.NUMBER.findall(r"\capThirtyDiff{} at the 30\% cap") == ["30"]


def test_spelled_out_numbers_are_extracted():
    """'Eleven of thirteen' is a count with no digit in it."""
    found = census.WORD_NUMBER.findall("Eleven of thirteen estimates are positive")
    assert [f.lower() for f in found] == ["eleven", "thirteen"]


def test_hyphenated_compounds_are_not_counted():
    """'three-state' names a model; it is not a reported quantity."""
    assert census.WORD_NUMBER.findall("the three-state specification") == []


def test_typesetting_lengths_are_stripped():
    """\\vspace{0.5em} is layout, not a quantity the reader sees."""
    text = census.prose_text()
    assert "vspace" not in text
    assert "0.5em" not in text


# ---------------------------------------------------------------------------
# Macro detection
# ---------------------------------------------------------------------------

def test_macros_are_detected_without_braces():
    """Equations invoke macros bare; brace-only matching exempts them."""
    found = census.MACRO_USE.findall(r"p = \primaryPValue .")
    assert "primaryPValue" in found


def test_generated_names_are_distinguished_from_latex_commands():
    assert census.GENERATED_NAME.match("primarySharpeDiff")
    assert census.GENERATED_NAME.match("capThirtyDiff")
    assert not census.GENERATED_NAME.match("textbf")
    assert not census.GENERATED_NAME.match("sqrt")


# ---------------------------------------------------------------------------
# The manuscript's actual state
# ---------------------------------------------------------------------------

def test_every_literal_in_the_manuscript_is_classified():
    if not CENSUS.exists():
        pytest.skip("census not yet written")
    classified = set(pd.read_csv(CENSUS)["literal"].astype(str))
    unclassified = sorted(set(census.literals()) - classified)
    assert not unclassified, f"unclassified numeric literals: {unclassified}"


def test_no_empirical_result_is_typed_into_the_prose():
    if not CENSUS.exists():
        pytest.skip("census not yet written")
    frame = pd.read_csv(CENSUS)
    typed = frame.loc[frame["classification"] == "empirical_result", "literal"]
    assert not len(typed), (
        f"these results are hand-typed instead of coming from macros: "
        f"{list(typed)}"
    )


def test_every_classification_is_one_of_the_five_categories():
    if not CENSUS.exists():
        pytest.skip("census not yet written")
    valid = {"empirical_result", "methodological_assumption", "sample_or_date",
             "equation_constant", "literature_fact"}
    found = set(pd.read_csv(CENSUS)["classification"])
    assert found <= valid, f"unexpected classifications: {sorted(found - valid)}"


# ---------------------------------------------------------------------------
# Result inventory
# ---------------------------------------------------------------------------

def test_every_result_carries_all_six_fields():
    if not INVENTORY.exists():
        pytest.skip("inventory not yet generated")
    inventory = pd.read_csv(INVENTORY)
    for field in census.INVENTORY_FIELDS:
        assert field in inventory.columns, f"inventory lacks {field}"
        blank = inventory[inventory[field].isna()
                          | (inventory[field].astype(str).str.strip() == "")]
        assert blank.empty, f"{field} is blank for {list(blank['result_id'])}"


def test_result_ids_are_unique():
    if not INVENTORY.exists():
        pytest.skip("inventory not yet generated")
    ids = pd.read_csv(INVENTORY)["result_id"]
    assert not ids.duplicated().any(), f"duplicate ids: {list(ids[ids.duplicated()])}"


def test_every_source_artifact_exists():
    """A provenance record that points nowhere is not provenance."""
    if not INVENTORY.exists():
        pytest.skip("inventory not yet generated")
    for path in pd.read_csv(INVENTORY)["source_artifact"].unique():
        assert (PROJECT_ROOT / path).exists(), f"missing source artifact: {path}"


def test_every_result_macro_cited_in_the_prose_is_in_the_inventory():
    if not INVENTORY.exists():
        pytest.skip("inventory not yet generated")
    assert census.check_inventory() == []


def test_display_values_re_derive_from_raw_values():
    """The recorded display value must be its raw value converted."""
    if not INVENTORY.exists():
        pytest.skip("inventory not yet generated")
    from src import units as un

    inventory = pd.read_csv(INVENTORY)
    for _, row in inventory.iterrows():
        # The stored unit is the display unit name (e.g. "bps", "%"),
        # which may be shared by several metrics; re-derivation uses the
        # ratio, which is unit-independent and catches a scaling slip.
        raw, display = float(row["raw_value"]), float(row["display_value"])
        if raw == 0:
            assert display == 0
            continue
        ratio = display / raw
        assert any(abs(ratio - candidate) <= 1e-9 * candidate
                   for candidate in (1, un.PERCENT_PER_UNIT, un.BPS_PER_UNIT, 252)), (
            f"{row['result_id']}: display {display} is not a standard "
            f"conversion of raw {raw} (ratio {ratio})"
        )
