"""Validate the LaTeX source without compiling it.

LaTeX is not installed on every machine, and a missing macro, input or
figure is worth catching before a build attempt. Checks that every macro
used is defined, every \\input and \\includegraphics target exists, and
every citation resolves to the bibliography.

    python scripts/check_paper.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"

SOURCES = ["main.tex", "_sections_intro.tex", "_sections_results.tex"]


def main() -> None:
    text = "".join((PAPER / name).read_text(encoding="utf-8") for name in SOURCES)
    problems: list[str] = []

    macros_file = PAPER / "macros.tex"
    macros_source = macros_file.read_text(encoding="utf-8")
    macro_bodies = dict(re.findall(r"newcommand\{\\(\w+)\}\{(.*)\}\s*$",
                                   macros_source, re.MULTILINE))
    defined = set(macro_bodies)

    # A generated macro is camelCase (lowercase run, then a capital);
    # every LaTeX kernel command is all-lowercase. Matching on that
    # convention checks the whole macro set instead of a prefix list,
    # which previously let a new family through unchecked.
    generated = re.compile(r"^[a-z]+[A-Z]")
    used = {name for name in re.findall(r"\\(\w+)", text) if generated.match(name)}
    undefined = used - defined
    print(f"Macros defined: {len(defined)}")
    print(f"  {', '.join(sorted(defined))}")
    if undefined:
        problems.append(f"macros used but not defined: {sorted(undefined)}")
    else:
        print("  all macros used in the text are defined")

    # A macro whose body contains a literal '$' closes math mode when it
    # is used inside an equation, which is a fatal error and produces no
    # PDF. This exact failure reached CI: \primaryCiLower expanded to
    # "$-$0.075" inside the primary-result equation.
    print("\nMath-mode safety:")
    math_regions = (
        re.findall(r"\\begin\{(?:equation|align|gather)\*?\}(.*?)\\end\{"
                   r"(?:equation|align|gather)\*?\}", text, re.DOTALL)
        + re.findall(r"\\\[(.*?)\\\]", text, re.DOTALL)
        + re.findall(r"(?<!\\)\$([^$]+)\$", text)
    )
    unsafe = set()
    for region in math_regions:
        for name in re.findall(r"\\(\w+)", region):
            if name in macro_bodies and "$" in macro_bodies[name]:
                unsafe.add(name)
    if unsafe:
        for name in sorted(unsafe):
            print(f"  [FATAL] \\{name} -> {macro_bodies[name]!r} used in math mode")
        problems.append(
            f"macro(s) containing a literal '$' used inside math mode: "
            f"{sorted(unsafe)}; use \\ensuremath instead"
        )
    else:
        print(f"  {len(math_regions)} math regions, no macro closes math mode")

    # A date that resolves at compile time makes the artifact
    # irreproducible: the same source builds a different page 1, and a
    # different PDF hash, tomorrow. The visual inspection attests to a
    # specific hash, so this must fail the build rather than be noticed
    # later by comparing hashes.
    print("\nReproducible date:")
    nondeterministic = []
    for command in (r"\\today", r"\\pdfcreationdate", r"\\pdffilemoddate"):
        for name in SOURCES:
            body = (PAPER / name).read_text(encoding="utf-8")
            # Ignore commented-out lines: the prohibition is explained in
            # a comment next to the \date line.
            live = "\n".join(line for line in body.splitlines()
                             if not line.lstrip().startswith("%"))
            if re.search(command, live):
                nondeterministic.append(f"{name}: {command.replace(chr(92) * 2, chr(92))}")
    if nondeterministic:
        for item in nondeterministic:
            print(f"  [FAIL] {item}")
        problems.append(
            f"compile-time date command(s) in manuscript source: "
            f"{nondeterministic}; use the frozen \\releaseDate macro "
            f"(config/config.yaml manuscript.release_date)"
        )
    else:
        print("  no compile-time date commands; the title date is frozen")

    print("\nInputs:")
    for target in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = PAPER / target
        if not candidate.suffix:
            candidate = candidate.with_suffix(".tex")
        status = "OK" if candidate.exists() else "MISSING"
        print(f"  [{status}] {target}")
        if status == "MISSING":
            problems.append(f"missing input: {target}")

    print("\nFigures:")
    for target in re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        candidate = PAPER / target
        status = "OK" if candidate.exists() else "MISSING"
        print(f"  [{status}] {target}")
        if status == "MISSING":
            problems.append(f"missing figure: {target}")

    print("\nCitations:")
    cited = set(re.findall(r"\\cite[tp]?\{([^}]+)\}", text))
    cited = {key.strip() for group in cited for key in group.split(",")}
    entries = set(re.findall(r"@\w+\{([^,]+),",
                             (PAPER / "references.bib").read_text(encoding="utf-8")))
    for key in sorted(cited):
        status = "OK" if key in entries else "MISSING"
        print(f"  [{status}] {key}")
        if status == "MISSING":
            problems.append(f"citation without bib entry: {key}")
    unused = entries - cited
    if unused:
        print(f"  note: in bibliography but not cited: {sorted(unused)}")

    # ------------------------------------------------------------------
    # Hand-typed empirical results
    # ------------------------------------------------------------------
    # The rule is narrow and deliberate: *reported empirical results*
    # must come from generated macros. Dates, sample sizes, methodological
    # constants (lambda = 0.94, 21-day blocks, 10 bps), equation
    # parameters and facts about the literature are legitimately written
    # by hand and are not flagged.
    print("\nHand-typed empirical results:")
    prose = re.sub(r"\\input\{[^}]+\}", "", text)
    prose = re.sub(r"%.*", "", prose)

    # Values this project reports as findings. A literal occurrence of
    # one of these in the prose means a result was typed rather than
    # pulled from macros.tex.
    reported_results = {
        "0.021": "primary Sharpe difference",
        "0.0210": "primary Sharpe difference",
        "-0.075": "CI lower bound",
        "-0.0748": "CI lower bound",
        "0.115": "CI upper bound",
        "0.1147": "CI upper bound",
        "0.327": "one-sided p-value",
        "0.3273": "one-sided p-value",
        "0.175": "volatility difference",
        "0.314": "annualized return difference",
        "0.929": "regime Sharpe",
        "0.908": "comparator Sharpe",
    }
    typed: list[str] = []
    for value, description in reported_results.items():
        # Ignore occurrences inside a macro definition or a comment.
        for match in re.finditer(re.escape(value), prose):
            start = max(0, match.start() - 60)
            context = prose[start:match.start()]
            if "newcommand" in context:
                continue
            typed.append(f"{value} ({description})")
            break

    if typed:
        for item in sorted(set(typed)):
            print(f"  TYPED: {item}")
        problems.append(
            f"{len(set(typed))} reported empirical result(s) typed into the "
            "prose instead of drawn from macros.tex"
        )
    else:
        print("  none - every reported empirical result comes from a macro")
        print("  (dates, sample sizes, method constants and literature")
        print("   facts are legitimately hand-written and not checked)")

    words = len(re.findall(r"\b\w+\b", re.sub(r"\\[a-zA-Z]+|%.*", "", text)))
    print(f"\nApproximate word count: {words:,}")

    print()
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("LaTeX source validates: all macros, inputs, figures and citations resolve.")


if __name__ == "__main__":
    main()
