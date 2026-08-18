"""Centralized unit conversion for every reported number.

Two unit errors reached draft reports before review caught them: an
annualized return difference of 3.14e-05 written as "3.1 basis points"
when it is 0.314, and a cost effect of 5.73 bps on a 9% volatility
portfolio written as "0.06 Sharpe" when it is 0.0064. Both were
hand-arithmetic in prose. Both are the same failure mode: a raw decimal
converted to a display unit by eye.

Every number that reaches a table, figure, or the LaTeX passes through
this module. Tables store three columns — ``raw_value``,
``display_value``, ``display_unit`` — so the conversion is recorded
next to the result and can be checked mechanically instead of trusted.

No manual conversion may appear in the paper source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS_PER_UNIT = 10_000
PERCENT_PER_UNIT = 100


# ---------------------------------------------------------------------------
# Scalar conversions
# ---------------------------------------------------------------------------

def decimal_to_percent(x: float) -> float:
    """Decimal fraction to percent. 0.0873 -> 8.73 (%)."""
    return PERCENT_PER_UNIT * x


def decimal_to_percentage_points(x: float) -> float:
    """Decimal *difference* to percentage points. 0.00175 -> 0.175 (pp).

    Numerically identical to :func:`decimal_to_percent`; kept separate
    because the two describe different quantities. A level is reported
    in percent, a difference between two levels in percentage points,
    and conflating them in prose is how "0.175 pp" becomes "0.175%".
    """
    return PERCENT_PER_UNIT * x


def decimal_to_bps(x: float) -> float:
    """Decimal fraction to basis points. 0.0000314 -> 0.314 (bps)."""
    return BPS_PER_UNIT * x


def bps_to_decimal(x: float) -> float:
    """Basis points back to a decimal fraction. 10 -> 0.001."""
    return x / BPS_PER_UNIT


def sharpe_impact_of_return_change(
    return_change_decimal: float, annualized_volatility_decimal: float
) -> float:
    """Sharpe-ratio impact of a change in annualized mean return.

    ``delta_SR = delta_mu / sigma``, both in decimal units. Wrapped
    because doing it inline in prose produced the 0.06-versus-0.0064
    error: 5.73 bps on a 9% volatility portfolio is 0.000573 / 0.09 =
    0.0064, not 0.06.

    This is an approximation that holds only when volatility is
    unchanged. When both mean and volatility move, compute the Sharpe
    difference directly from the return series instead — see
    ``scripts/decompose_cap.py``, which uses the exact
    ``dSR(0bps) - dSR(10bps)`` identity.
    """
    if annualized_volatility_decimal <= 0:
        raise ValueError("volatility must be positive")
    return return_change_decimal / annualized_volatility_decimal


# ---------------------------------------------------------------------------
# Display registry
# ---------------------------------------------------------------------------

def _scientific(x: float, decimals: int = 2) -> str:
    """LaTeX scientific notation: 1.4503e-11 -> ``1.45 \\times 10^{-11}``.

    Wrapped in ``\\ensuremath`` rather than ``$...$`` so the value can be
    used inside an equation as well as in running text; a literal ``$``
    inside math mode closes it and aborts the compile.
    """
    mantissa, exponent = f"{x:.{decimals}e}".split("e")
    return rf"\ensuremath{{{mantissa} \times 10^{{{int(exponent)}}}}}"


@dataclass(frozen=True)
class Unit:
    """A display unit and the function that converts into it."""

    name: str
    convert: callable
    decimals: int
    render: callable | None = None

    def format(self, raw: float) -> str:
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
            return ""
        if self.render is not None:
            return self.render(self.convert(raw))
        return f"{self.convert(raw):.{self.decimals}f}"


UNITS: dict[str, Unit] = {
    "decimal": Unit("decimal", lambda x: x, 4),
    "percent": Unit("%", decimal_to_percent, 2),
    "percentage_points": Unit("pp", decimal_to_percentage_points, 3),
    "bps": Unit("bps", decimal_to_bps, 3),
    "ratio": Unit("", lambda x: x, 3),        # Sharpe, Sortino, Calmar
    "count": Unit("", lambda x: x, 0),
    "percent_share": Unit("%", decimal_to_percent, 1),
    "multiple": Unit("x", lambda x: x, 2),
    # A share reported to one decimal rounds 0.1586% to 0.2%, which is
    # not the precision the text depends on.
    "percent_share_precise": Unit("%", decimal_to_percent, 2),
    "factor": Unit("", lambda x: x, 1),           # "3.2 times the turnover"
    "years": Unit("", lambda x: x, 1),
    "correlation": Unit("", lambda x: x, 2),
    "condition_number": Unit("", lambda x: x, 1),
    "scientific": Unit("", lambda x: x, 2, render=_scientific),
}

# Which unit each reported quantity is displayed in. Adding a metric to
# a paper table without registering it here raises, which is the point:
# an unregistered quantity cannot reach the LaTeX with an unchecked
# hand conversion.
METRIC_UNITS: dict[str, str] = {
    "sharpe": "ratio",
    "sharpe_difference": "ratio",
    "sortino": "ratio",
    "calmar": "ratio",
    "cagr": "percent",
    "cagr_difference": "percentage_points",
    "ann_volatility": "percent",
    "vol_difference": "percentage_points",
    "max_drawdown": "percent",
    "maxdd_difference": "percentage_points",
    "var_95": "percent",
    "var_99": "percent",
    "es_95": "percent",
    "es_99": "percent",
    "mean_return_difference_annualized": "bps",
    "cost_expenditure_annualized": "bps",
    "half_turnover": "percent",
    "full_traded_notional": "percent",
    "terminal_wealth": "multiple",
    "ci95_lower": "ratio",
    "ci95_upper": "ratio",
    "p_value": "ratio",
    "n_observations": "count",
    "percent_share": "percent_share",
    "percent_share_precise": "percent_share_precise",
    "count": "count",
    "ratio": "ratio",
    "factor": "factor",
    "years": "years",
    "correlation": "correlation",
    "condition_number": "condition_number",
    "scientific": "scientific",
}


def to_display(raw: float, metric: str) -> tuple[float, str, str]:
    """Convert one value; return ``(display_value, unit_name, formatted)``.

    Raises for an unregistered metric rather than guessing a unit.
    """
    if metric not in METRIC_UNITS:
        raise KeyError(
            f"metric '{metric}' has no registered display unit; add it to "
            "METRIC_UNITS rather than converting by hand"
        )
    unit = UNITS[METRIC_UNITS[metric]]
    return unit.convert(raw), unit.name, unit.format(raw)


def build_display_table(
    values: dict[str, float] | pd.Series, label: str = ""
) -> pd.DataFrame:
    """Long-format table carrying raw and display values side by side.

    Columns: ``label``, ``metric``, ``raw_value``, ``display_value``,
    ``display_unit``, ``formatted``. The raw value travels with its
    display value so any conversion can be re-derived and checked.
    """
    rows = []
    for metric, raw in dict(values).items():
        display, unit_name, formatted = to_display(raw, metric)
        rows.append({
            "label": label,
            "metric": metric,
            "raw_value": raw,
            "display_value": display,
            "display_unit": unit_name,
            "formatted": formatted,
        })
    return pd.DataFrame(rows)


def verify_display_table(frame: pd.DataFrame, rtol: float = 1e-12) -> None:
    """Re-derive every display value from its raw value.

    Called before any table is written to the paper directory, so a
    corrupted or hand-edited display column fails loudly.
    """
    for _, row in frame.iterrows():
        expected, _, _ = to_display(row["raw_value"], row["metric"])
        if not np.isclose(row["display_value"], expected, rtol=rtol, equal_nan=True):
            raise ValueError(
                f"display value for {row['metric']} ({row['display_value']}) does "
                f"not match its raw value {row['raw_value']} converted to "
                f"{row['display_unit']} (expected {expected})"
            )
