"""Tests for LaTeX escaping and label resolution.

Both failures these guard against actually reached a build: an
unescaped percent silently commented out a macro line, and raw
snake_case identifiers produced "Missing $ inserted" and no PDF.
"""

from __future__ import annotations

import pytest

from src import latex


# ---------------------------------------------------------------------------
# Every special character
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("\\", r"\textbackslash{}"),
    ],
)
def test_each_special_character_is_escaped(raw, expected):
    assert latex.escape(raw) == expected


def test_all_special_characters_together():
    raw = r"_%&#${}~^\ "
    escaped = latex.escape(raw)
    # No bare special character may survive, ignoring the ones that are
    # part of an escape sequence.
    assert "ann_volatility" not in escaped
    for character in "_%&#$":
        assert f"\\{character}" in escaped
    assert r"\textasciitilde{}" in escaped
    assert r"\textasciicircum{}" in escaped
    assert r"\textbackslash{}" in escaped


def test_backslash_escaped_first_so_replacements_are_not_mangled():
    # If the backslash were escaped last, the backslashes introduced by
    # earlier replacements would be double-escaped.
    assert latex.escape("a_b") == r"a\_b"
    assert latex.escape("100%") == r"100\%"
    assert latex.escape(r"\_") == r"\textbackslash{}\_"


def test_regression_percent_in_a_unit_string():
    """The bug that would have commented out a macro line."""
    assert latex.escape("54.19 %") == r"54.19 \%"
    assert "\\%" in latex.escape("%")


def test_regression_underscore_in_a_column_name():
    """The bug that produced 'Missing $ inserted' and no PDF."""
    assert latex.escape("ann_volatility") == r"ann\_volatility"
    assert latex.escape("max_drawdown") == r"max\_drawdown"


def test_escape_accepts_non_strings():
    assert latex.escape(0.021) == "0.021"
    assert latex.escape(None) == "None"


def test_plain_text_is_unchanged():
    assert latex.escape("Regime-aware min-var") == "Regime-aware min-var"


# ---------------------------------------------------------------------------
# Label registry
# ---------------------------------------------------------------------------

def test_registered_identifiers_resolve_to_raw_text():
    """label_for returns backend-neutral text, unescaped."""
    assert latex.label_for("regime_minvar") == "Regime-aware min-var"
    assert latex.label_for("sharpe") == "Sharpe"
    assert latex.label_for("cap_30pct") == "30% cap"


def test_unknown_identifier_raises_rather_than_falling_back():
    """A silent fallback to the raw identifier is how an unescaped
    underscore reaches the compiler."""
    with pytest.raises(latex.UnknownLabelError, match="no display label"):
        latex.label_for("some_new_metric")


def test_raw_labels_carry_no_latex_markup():
    """The registry is backend-neutral. A pre-escaped label would be
    escaped a second time at serialization."""
    for registry in latex._REGISTRIES:
        for key, label in registry.items():
            assert "\\" not in label, f"{key}: LaTeX markup in raw label {label!r}"
            assert "$" not in label, f"{key}: math markup in raw label {label!r}"


def test_serialized_labels_are_latex_safe():
    """Inspect the escaped OUTPUT rather than restricting the raw text."""
    for registry in latex._REGISTRIES:
        for key in registry:
            serialized = latex.latex_label_for(key)
            for i, character in enumerate(serialized):
                if character in "%&#_":
                    assert i > 0 and serialized[i - 1] == "\\", \
                        f"{key}: bare {character!r} in {serialized!r}"
            assert serialized.count("$") % 2 == 0, \
                f"{key}: unbalanced math mode in {serialized!r}"


def test_percent_label_serializes_once():
    assert latex.label_for("cap_30pct") == "30% cap"
    assert latex.latex_label_for("cap_30pct") == r"30\% cap"


def test_unicode_math_becomes_latex_math():
    assert latex.label_for("sharpe_difference") == "Δ Sharpe"
    assert latex.latex_label_for("sharpe_difference") == r"$\Delta$ Sharpe"
    assert latex.latex_label_for("neff_kappa_30") == r"$\kappa$ = 30"


def test_serialization_must_run_exactly_once():
    """Documents why to_latex lives only at the output boundary."""
    once = latex.to_latex("Max DD (%)")
    assert once == r"Max DD (\%)"
    assert latex.to_latex(once) != once, "double serialization must corrupt"


def test_every_strategy_in_the_ladder_is_registered():
    from src import optimization as opt

    for strategy in opt.STRATEGIES:
        assert latex.is_registered(strategy), f"{strategy} has no display label"


def test_every_robustness_specification_is_registered():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_robustness import build_specifications

    for spec in build_specifications():
        assert latex.is_registered(spec.name), f"{spec.name} has no display label"


# ---------------------------------------------------------------------------
# Macro naming (LaTeX command names are letters-only)
# ---------------------------------------------------------------------------

def test_macro_names_contain_no_digits():
    r"""\cap30Diff is invalid LaTeX and would also collide with \cap."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.generate_paper_outputs import macro_name

    for key in ("cap30_diff", "post2020_diff", "state1_condition",
                "neff_kappa_120", "var_95"):
        name = macro_name(key)
        assert name.isalpha(), f"{key} -> {name} is not letters-only"
        assert not any(ch.isdigit() for ch in name)


def test_macro_name_rejects_unmappable_digits():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.generate_paper_outputs import macro_name

    assert macro_name("cap30_diff") == "capThirtyDiff"
    assert macro_name("primary_sharpe_diff") == "primarySharpeDiff"


def test_generated_macros_file_is_latex_legal():
    r"""Every \newcommand in the generated file must be letters-only."""
    import re
    from pathlib import Path

    macros = Path(__file__).resolve().parents[1] / "paper" / "macros.tex"
    if not macros.exists():
        return
    names = re.findall(r"newcommand\{\\(\w+)\}", macros.read_text(encoding="utf-8"))
    assert names, "no macros found"
    for name in names:
        assert name.isalpha(), f"macro {name!r} contains a non-letter"


def test_generated_macros_are_math_mode_safe():
    """A macro containing a literal $ closes math mode inside an
    equation and aborts the compile with no PDF. This reached CI once:
    \primaryCiLower expanded to "$-$0.075" inside the primary-result
    equation."""
    from pathlib import Path

    macros = Path(__file__).resolve().parents[1] / "paper" / "macros.tex"
    if not macros.exists():
        return
    for line in macros.read_text(encoding="utf-8").splitlines():
        if line.startswith("\newcommand"):
            assert "$" not in line, (
                f"macro body contains a literal $, unsafe in math mode: {line}"
            )
