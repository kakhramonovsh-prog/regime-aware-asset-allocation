"""Backend-neutral display labels, escaped once at serialization.

Two build failures came from generated text reaching LaTeX unescaped: an
unescaped ``%`` commented out a macro line, and raw ``snake_case``
identifiers produced "Missing $ inserted" and no PDF.

The architecture that prevents both, and avoids trading them for a
double-escaping bug:

* the registry stores **raw human-readable text** --- ``Max DD (%)``,
  not ``Max DD (\\%)`` --- so the same labels serve CSV, Markdown and
  any other backend;
* mathematical symbols are stored as Unicode (``Δ``, ``κ``), not as
  LaTeX markup;
* :func:`to_latex` is called **exactly once**, at the LaTeX
  serialization boundary, and is the only place escaping happens.

An unregistered identifier raises rather than falling back to the raw
string, because that fallback is how the underscore reached the
compiler.
"""

from __future__ import annotations

# Every LaTeX special character, mapped in a single pass. A sequential
# replace is wrong in both directions: escaping the backslash first
# inserts braces the brace rules then re-escape, escaping it last mangles
# backslashes the earlier rules introduced.
_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Unicode symbols that carry mathematical meaning are rendered as math
# rather than escaped. Applied after escaping, so their backslashes are
# never themselves escaped.
_MATH_SYMBOLS: dict[str, str] = {
    "Δ": r"$\Delta$",
    "κ": r"$\kappa$",
    "σ": r"$\sigma$",
    "μ": r"$\mu$",
    "α": r"$\alpha$",
    "λ": r"$\lambda$",
    "Σ": r"$\Sigma$",
    "−": "$-$",     # U+2212 minus sign, not a hyphen
    "×": r"$\times$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
}


def escape(text: object) -> str:
    """Escape every LaTeX special character. Each character maps once."""
    return "".join(_ESCAPES.get(character, character) for character in str(text))


def to_latex(text: object) -> str:
    """Serialize raw display text to LaTeX.

    The single escaping boundary: escapes specials, then renders known
    Unicode mathematical symbols as math. Call once, at output time,
    never on already-serialized text.
    """
    result = escape(text)
    for symbol, replacement in _MATH_SYMBOLS.items():
        result = result.replace(symbol, replacement)
    return result


class UnknownLabelError(KeyError):
    """Raised when an identifier has no registered display label."""


# ---------------------------------------------------------------------------
# Registries: RAW text only. No LaTeX markup, no pre-escaping.
# ---------------------------------------------------------------------------

STRATEGY_LABELS: dict[str, str] = {
    "equal_weight": "Equal weight",
    "static_6040": "60/40",
    "static_minvar": "Static min-var",
    "rolling_lw_minvar": "Rolling LW min-var",
    "ewma_scaled_minvar": "EWMA-scaled min-var",
    "regime_minvar": "Regime-aware min-var",
    "regime_minvar minus rolling_lw_minvar": "Regime-aware − rolling LW",
    "headline": "Headline",
}

METRIC_LABELS: dict[str, str] = {
    "cagr": "CAGR (%)",
    "ann_volatility": "Volatility (%)",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max DD (%)",
    "calmar": "Calmar",
    "sharpe_difference": "Δ Sharpe",
    "vol_difference": "Δ Vol (pp)",
    "cagr_difference": "Δ CAGR (pp)",
    "maxdd_difference": "Δ Max DD (pp)",
    "ci95_lower": "CI lower",
    "ci95_upper": "CI upper",
    "p_value": "p-value",
    "mean_return_difference_annualized": "Mean return diff (bps)",
    "cost_expenditure_annualized": "Cost (bps p.a.)",
    "half_turnover": "Half-turnover (%)",
    "full_traded_notional": "Traded notional (%)",
    "terminal_wealth": "Terminal wealth",
    "n_observations": "N",
    "difference": "Difference",
}

SPEC_LABELS: dict[str, str] = {
    "primary": "Primary",
    "hmm_3_states": "Three states",
    "drop_vix": "Drop VIX",
    "drop_realized_vol": "Drop realized vol",
    "rolling_5y_window": "Rolling 5y window",
    "cap_30pct": "30% cap",
    "cap_50pct": "50% cap",
    "neff_kappa_30": "κ = 30",
    "neff_kappa_120": "κ = 120",
    "core_universe": "Core universe",
    "alt_seeds_100": "Alternative seeds",
    "a2_accept_as_estimated": "Accept degenerate fits",
    "a3_total_covariance": "Total covariance mixture",
}

_REGISTRIES = (STRATEGY_LABELS, METRIC_LABELS, SPEC_LABELS)


def label_for(identifier: str) -> str:
    """Raw display label for a machine identifier, unescaped.

    Raises :class:`UnknownLabelError` when unregistered; a silent
    fallback to the raw identifier is how an unescaped underscore
    reaches the compiler.
    """
    for registry in _REGISTRIES:
        if identifier in registry:
            return registry[identifier]
    raise UnknownLabelError(
        f"no display label registered for {identifier!r}. Add it to "
        "src/latex.py rather than letting a raw identifier reach output."
    )


def latex_label_for(identifier: str) -> str:
    """Display label serialized for LaTeX. Escaping happens here, once."""
    return to_latex(label_for(identifier))


def is_registered(identifier: str) -> bool:
    return any(identifier in registry for registry in _REGISTRIES)
