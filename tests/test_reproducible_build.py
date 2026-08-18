"""The manuscript PDF must be reproducible from fixed source.

A compile-time date makes the artifact irreproducible by construction:
the same source builds a different page 1, and a different hash, the
next day. The visual inspection attests to a specific hash, so this is
a correctness property of the release rather than a nicety.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PAPER = PROJECT_ROOT / "paper"
SOURCES = ("main.tex", "_sections_intro.tex", "_sections_results.tex")
FORBIDDEN = (r"\\today", r"\\pdfcreationdate", r"\\pdffilemoddate")


def live_source(text: str) -> str:
    """Source with comment lines removed.

    The prohibition is explained in a comment beside the date line, so a
    naive scan would flag its own documentation.
    """
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("%"))


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", SOURCES)
def test_no_compile_time_date_in_manuscript_source(name):
    body = live_source((PAPER / name).read_text(encoding="utf-8"))
    for command in FORBIDDEN:
        assert not re.search(command, body), (
            f"{name} uses {command}; the title date must come from the "
            "frozen releaseDate macro"
        )


def test_release_date_is_configured_and_valid():
    raw = str(load_config()["manuscript"]["release_date"])
    parsed = dt.date.fromisoformat(raw)          # raises if malformed
    assert parsed.year >= 2026


def test_release_date_macro_matches_config():
    macros = PAPER / "macros.tex"
    if not macros.exists():
        pytest.skip("macros not generated")
    match = re.search(r"newcommand\{\\releaseDate\}\{([^}]*)\}",
                      macros.read_text(encoding="utf-8"))
    assert match, "releaseDate is not generated"

    release = dt.date.fromisoformat(
        str(load_config()["manuscript"]["release_date"]))
    assert match.group(1) == f"{release:%B} {release.day}, {release.year}"


def test_generated_macros_carry_no_wall_clock_timestamp():
    """A build timestamp would make macros.tex differ on every run, so a
    clean-clone reproduction could never be byte-identical."""
    macros = PAPER / "macros.tex"
    if not macros.exists():
        pytest.skip("macros not generated")
    for line in macros.read_text(encoding="utf-8").splitlines()[:3]:
        assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", line), (
            f"wall-clock timestamp in the macros.tex header: {line}"
        )


def test_the_frozen_date_is_used_on_the_title_page():
    body = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert re.search(r"\\date\{\\releaseDate\}", body), (
        "main.tex does not set the title date from the frozen macro"
    )
