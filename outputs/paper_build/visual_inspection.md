# Visual Inspection Audit

Inspection is a property of a **specific artifact**, not of the
manuscript in general. Any source change invalidates the inspected PDF:
a one-word edit can move a float and repaginate everything after it, so
the whole pass repeats against a newly built and newly hashed artifact.

## Round 1 — FAILED, corrections required

| Field | Value |
|---|---|
| PDF SHA-256 | `3b653741427bd63f84d345e09957bb7e07b840c230ee1ffcd60e94b1c85ef516` |
| Commit | `ca0a7f140b22ca1cc56847cbc5bfb85af9336cb6` |
| CI run | 31975553015 |
| Page count | 15 |
| Inspection date | 2026-08-16 |
| Automated gate | PASSED (bbl 6,637 bytes, 0 BibTeX errors, 0 unresolved markers, 0 overfull > 5pt) |
| Visual result | **7 defects found** |

### Page-by-page

| Page | Elements checked | Issue | Correction | Reverified |
|---|---|---|---|---|
| 1 | Title, author, abstract, keywords, JEL | None | — | pending round 2 |
| 2 | Introduction, macros, citations | **D1** hyphen used for negative numbers where a minus sign belongs | Serialize negatives as math minus | pending |
| 3 | Literature review, 14 citations | None | — | pending |
| 4 | Literature (methods), citations | None | — | pending |
| 5 | Data section, licensing paragraph | None | — | pending |
| 6 | Design, hypothesis equation (1), timeline | None | — | pending |
| 7 | Estimation, equation (3), Greek symbols | None | — | pending |
| 8 | Covariance, degeneracy policy | None | — | pending |
| 9 | Results, Table 2, inference prose | **D2** "which is -0.175 pp lower" is a double negative and states the opposite of what is meant; **D3** hyphens for negatives in Table 2 | Reword to "0.175 pp lower"; math minus in tables | pending |
| 10 | Figure 1, caption, axis labels | **D4** y-axis shows machine identifiers (`hmm_3_states`, `a2_accept_as_estimated`) instead of display labels; **D5** "primary (primary)" duplicated; **D6** figure alone on the page with large white space; **D7** axis says "Ledoit-Wolf" (hyphen) while the caption uses an en-dash | Use the label registry in the figure; drop the duplicate; tighten float placement; en-dash | pending |
| 11 | Robustness prose, sign instability | None | — | pending |
| 12 | Table 3 (13 rows), limitations | **D3** hyphens for negatives | Math minus | pending |
| 13 | Limitations, numerical reproduction | None | — | pending |
| 14 | Discussion, conclusion, cap-30 framing | None | — | pending |
| 15 | Bibliography | None observed; formatting consistent | — | pending |

### Defects in detail

**D1/D3 — hyphen where a minus sign belongs.** Negative numbers render
with a text hyphen (`-0.075`) rather than a typographic minus
(`−0.075`). The figure axis already uses a proper minus, so the document
is internally inconsistent. Affects every negative value in Tables 2, 3
and the macros.

**D2 — semantic error.** "annualized volatility, which is -0.175 pp
lower" says the volatility is *higher* by 0.175 pp, the opposite of the
finding. Either "0.175 pp lower" or "the difference is −0.175 pp" is
correct; the two must not be combined.

**D4 — machine identifiers in a figure.** The forest plot predates the
label registry and renders `drop_realized_vol`, `neff_kappa_30`,
`a3_total_covariance`. These are the same identifiers that broke the
LaTeX build; here they merely look unfinished, but the fix is the same
registry.

**D5 — duplicated text.** The primary row reads "primary  (primary)".

**D6 — float placement.** Figure 1 sits alone on page 10 with roughly
40% of the page blank above it.

**D7 — inconsistent dash.** Axis label "Ledoit-Wolf" versus caption
"Ledoit–Wolf".

### Verdict

Round 1 fails. The automated gate passed all seven of its criteria, and
the visual pass still found a semantic error that reverses the meaning
of a reported result. That is the argument for keeping the manual gate.

## Round 2 — pending

To be recorded after correction, rebuild in CI, and a fresh
page-by-page pass against the new hash.

---

## Round 2 — FAILED, corrections required

Round 1's seven defects are all confirmed fixed (see the "Round 1
resolution" column below). Round 2 inspects a newly built artifact and
records an explicit disposition for **every one of the 15 pages**, not
only the pages that were previously defective.

| Field | Value |
|---|---|
| PDF SHA-256 | `7019d029da61acebab724c290ca7a1297500fe174afab7746f80cab5e3c5be0a` |
| Commit | `6a77bef027eab081fc1308c997abf7fcc32af8af` |
| CI run | 31978675959 |
| Page count | 15 |
| Inspection date | 2026-08-16 |
| Automated gate | PASSED (latexmk clean, non-empty bbl, 0 unresolved markers, 0 overfull > 5pt) |
| Number census | PASSED (48 literals classified; 0 empirical results typed) |
| Visual result | **7 defects found, all typographic; no numerical error** |

### Page-by-page disposition (all 15)

| Page | Elements checked | Disposition | Round 1 resolution |
|---|---|---|---|
| 1 | Title, author, date, abstract, keywords, JEL, footnote, intro opening | **PASS** — minus signs typographic throughout (−0.075); abstract states magnitude correctly ("lowered ... by 0.175 pp") | — |
| 2 | Introduction, 6 macro citations, preregistration paragraph | **PASS** with note — D1 (hyphen-as-minus) is fixed; wording "an interval 9.0 times its width" is loose but not an error | D1 fixed |
| 3 | Literature, 14 author-year citations, Section 8 cross-reference | **PASS** — every citation resolves, no `[?]`, cross-reference resolves to "Section 8" | — |
| 4 | Literature (methods), 10 citations, data section, alignment paragraph | **PASS** — equity correlations render from macros (0.82, 0.92) | — |
| 5 | Licensing, preregistered design, equation (1), sample split, timeline | **PASS** — equation (1) typesets correctly | — |
| 6 | Strategy ladder, costs, equation (2), volatility, regimes | **FAIL** — **D8** "volatility- scaled" stray space after hyphen; **D9** "the fifteen pairwise comparisons 1 survives" mixes a word and a digit in one clause | — |
| 7 | Covariance construction, equation (3), conditioning diagnostics | **FAIL** — **D10** "0.16 %" renders a thin space before the percent sign while "1%" on the same line does not | — |
| 8 | Results, Table 1 (6 rows), equation (4), bootstrap paragraph | **PASS** — equation (4) is the exact line that aborted the previous build; it now typesets. Table 1 minus signs correct | — |
| 9 | Block length, returns/volatility, trading costs, Table 2 | **FAIL** — **D10** "54.19 %", "16.94 %"; **D11** "lower by 0.175 pp, a difference of −0.175 pp" states magnitude and signed value back to back; **D9** "11 of thirteen ... 12 of thirteen"; "Of the 5 secondary metric intervals" | — |
| 10 | Figure 1, 13 rows, axis labels, caption, footnote | **PASS** — D4/D5/D7 all fixed: display labels ("Three states", "κ = 30"), no duplicate, en-dash in the axis label. D6 (float alone on page) persists but is acceptable for a full-width figure | D4, D5, D7 fixed; D6 accepted |
| 11 | Sign instability, weight cap, minor factors, subperiods, limitations | **FAIL** — **D12** paragraph opens "2 preregistered variations", a sentence beginning with a digit; **D10** "45.4 %", "26.6 %", "73.4 %" beside "30%" and "40%" | — |
| 12 | Table 3 (13 rows), limitations, numerical reproduction | **PASS** — drop-realized-vol reads −0.023, matching the corrected prose; 1.45 × 10⁻¹¹ typesets via `\ensuremath` | D3 fixed |
| 13 | Researcher discretion, discussion, conclusion, cap-30 framing | **FAIL** — **D10** "73.4 %" directly beneath "40% to 30%" | — |
| 14 | Bibliography, entries Ang–Engle (9 entries) | **FAIL** — **D13** BibTeX lowercased protected words: "the em algorithm" (Dempster), "united kingdom inflation" (Engle), "1/n portfolio strategy" (DeMiguel) | — |
| 15 | Bibliography, entries Guidolin–Politis (13 entries), page fill | **FAIL** — **D13** "with the sharpe ratio" (Ledoit–Wolf 2008), "The markowitz optimization enigma" (Michaud). Page is ~90% full, satisfying the half-full rule | — |

### Defects in detail

**D8 — "volatility- scaled" (page 6).** A hyphen at the end of a source
line followed by a newline renders as "volatility- scaled". Cosmetic but
conspicuous in a compound the paper uses repeatedly.

**D9 — mixed word and digit counts in one clause (pages 6, 9).** Moving
counts into macros forced digits into sentences whose other numbers are
spelled out: "across the fifteen pairwise comparisons 1 survives", "11
of thirteen point estimates are positive and 12 of thirteen intervals
contain zero", "Of the 5 secondary metric intervals". The counts are
correct; the register is inconsistent. The fix is to phrase the sentence
so a digit reads naturally, not to revert to hand-typed words.

**D10 — thin space before the percent sign (pages 6, 7, 9, 11, 13).**
Generated percentages render as "54.19 %" because the macro writer
appends `\,` before every unit. That is right for "bps" and wrong for
"%", and the inconsistency is visible on the same line as hard-coded
values: page 13 reads "40% to 30% ... 73.4 % of that change".

**D11 — magnitude and signed value stated consecutively (page 9).**
"the regime-aware strategy's is lower by 0.175 pp, a difference of
−0.175 pp with a 95% interval ..." Both macros are correct; using both
in one sentence is redundant. Splitting signed from magnitude fixed a
semantic reversal in Round 1, and this is the cost of that split showing
up at the point of use.

**D12 — sentence begins with a digit (page 11).** "**Sign instability.**
2 preregistered variations reverse the sign ..." A sentence may not open
with a numeral. Same cause as D9.

**D13 — BibTeX case-folding (pages 14, 15).** `plainnat` lowercases
title words that are not brace-protected, so acronyms and proper nouns
were silently downcased: "em algorithm", "united kingdom", "1/n",
"sharpe ratio", "markowitz optimization". These are wrong as
bibliographic records, not merely ugly, and they are exactly the class
of error a build gate cannot see: BibTeX succeeded.

### Observation (not a defect)

`paper/tables/table3_secondary_intervals.tex` is generated but never
`\input` into the manuscript; the five secondary intervals appear in
prose only. Worth a decision before release: either include the table or
stop generating it.

### What Round 2 establishes

No numerical defect was found. Every value on every page traces to a
macro in `paper/result_inventory.csv`, the two corrections the census
surfaced (drop-realized-vol −0.023, between-state 0.16%) render as
corrected, and equation (4) — the line that aborted the previous build —
typesets. All seven defects are typographic or bibliographic.

Five of the seven (D9, D10, D11, D12 and part of D8) were introduced by
this round's own change: moving hand-typed results into macros. That is
the expected cost of the change and the reason the pass repeats rather
than being assumed.

---

## Round 3 — PASSED (one wording residual corrected)

| Field | Value |
|---|---|
| PDF SHA-256 | `5c7795e458c578ecd993a4507f56421b97ac6d2e4814f8e6cbb086e744853156` |
| Commit | `b102c43634394a3cd90acf5579cee4175751d7ff` |
| CI run | 31979149256 |
| Page count | 15 |
| Inspection date | 2026-08-16 |
| Automated gate | PASSED |
| Number census | PASSED |
| Visual result | **All 6 Round 2 defects fixed; 1 wording residual** |

### Verification of every Round 2 defect

| Defect | Page | Round 3 state |
|---|---|---|
| D8 stray space in "volatility- scaled" | 6 | **FIXED** — renders "volatility-scaled minimum variance" |
| D9 mixed word/digit counts | 6, 9 | **FIXED** — "only 1 of the fifteen pairwise comparisons survives"; "Point estimates are positive in 11 of the thirteen specifications, and 12 of the thirteen intervals contain zero" |
| D10 thin space before percent | 6, 7, 9, 11, 13 | **FIXED** — "54.19%", "16.94%", "0.16%", "45.4%", "26.6%", "73.4%"; page 13 now reads "40% to 30% ... 73.4% of that change" consistently |
| D11 magnitude and signed value consecutive | 9 | **FIXED** — "lower by 0.175 pp, with a 95% interval on that difference running from −0.327 pp to −0.033 pp" |
| D12 sentence opening with a digit | 11 | **FIXED** — "The estimate reverses sign in 2 of the preregistered variations" |
| D13 BibTeX case-folding | 14, 15 | **FIXED** — "1/N portfolio strategy", "the EM algorithm", "United Kingdom inflation", "with the Sharpe ratio", "The Markowitz optimization enigma" |

### Pages re-verified as unchanged

Pages 1–5, 8, 10 and 12 were re-read and remain as dispositioned in
Round 2. Page count is unchanged at 15, so no float moved and nothing
repaginated. Table 3 still shows drop-realized-vol at −0.023 and the
scientific-notation macro still typesets.

### Residual corrected after this inspection

**D14 (page 9)** — "Only one of the 5 secondary metric intervals
excludes zero" places a spelled-out number beside a digit in the same
clause, the last instance of D9's pattern. Reworded to "Among the 5
secondary metric intervals, only annualized volatility excludes zero."
This is a source change, so it invalidates the Round 3 artifact and
requires one confirming pass on the next build.

---

## Round 4 — PASSED, no defects

| Field | Value |
|---|---|
| PDF SHA-256 | `3e7e5c4e54956dbd7dcd2f8186a24b04ba7696497f594e5f9240c8d79b2d73fa` |
| Commit | `deea4bb0aa6c2a8d70682229f88cd295055d1e36` |
| CI run | 31979428122 |
| Page count | 15 |
| Inspection date | 2026-08-16 |
| Automated gate | PASSED |
| Number census | PASSED |
| Visual result | **No defects** |

Confirming pass for D14. Page 9 now reads "Among the 5 secondary metric
intervals, only annualized volatility excludes zero". The page begins
and ends on the same content as Round 3 and the document is still 15
pages, so the edit reflowed within its own paragraph and nothing
repaginated.

The inspected PDF and its build log are preserved at
`outputs/paper_build/inspected/`, and `inspected_artifact.json` records
the hash, run, commit and the full four-round history.

### Standing rule

This pass attests to PDF `3e7e5c4e...73fa` and to nothing else. Any
change to `paper/`, `src/` or the build scripts invalidates it: a
one-word edit can move a float and repaginate everything after it. The
next release candidate needs a new build, a new hash and a fresh pass.

### Release gate still open

The manuscript cannot be tagged `v0.13.0-manuscript` on visual grounds
alone. `scripts/check_citations.py --release` still blocks on 14
references whose bibliographic details and supporting claims are not
yet verified.

---

## Round 5 — PASSED, release artifact for `v0.13.0-manuscript`

Round 4 was a pre-release inspected artifact. The citation pass changed
`references.bib`, which can repaginate the bibliography and everything
after it, so the approval was invalidated and the pass repeated.

| Field | Value |
|---|---|
| PDF SHA-256 | `73dac58e8537d34ea88ebe995ce49b44b9601962daded6d020574f8f62cf9639` |
| Commit | `7e43350` |
| CI run | 31992993874 |
| Page count | 15 |
| Bytes | 346,347 |
| Inspection date | 2026-08-17 |
| Build environment | pdfTeX 3.141592653-2.6-1.40.26, TeX Live 2024, format pdflatex 2025.3.10, Python 3.12, GitHub Actions ubuntu-latest, compiled 17 AUG 2026 04:02 |
| Automated gate | PASSED |
| Number census | PASSED |
| Citation release gate | PASSED (22 cited, 22 verified, 0 pending) |
| Visual result | **No defects** |

### Method

Eyeballing 15 pages for pagination shifts is weaker than comparing them,
so this round extracted the text of every page from both the Round-4 PDF
and this one and diffed them page by page. That localizes any shift
exactly rather than relying on a reader noticing one.

| Pages | Result |
|---|---|
| 2–14 | **Byte-identical text** to the Round-4 artifact, which was inspected page by page in Rounds 3 and 4. No float moved and nothing repaginated. |
| 1 | Differs only in the title date (`\today`: 16 → 17 August 2026). Visually confirmed: layout, abstract, keywords, JEL and footnote unchanged. |
| 15 | Differs only in `58(4):1651–1684` → `58(4):1651–1683`. Visually confirmed. |

Visual confirmation was additionally performed on pages 1, 10, 14 and 15
— the two pages that changed, plus the figure float on page 10 and the
first bibliography page, being the highest layout-risk pages. The figure
is unmoved and all thirteen specification rows render.

### Bibliography verified in the rendered PDF

- Jagannathan and Ma reads **1651–1683**, the corrected range.
- Holm carries **no DOI**, while every other entry does — correct, since
  `10.2307/4615733` does not resolve.
- Case protections held: "1/N", "EM algorithm", "United Kingdom",
  "Sharpe ratio", "Markowitz optimization enigma".

### On the 22-versus-23 discrepancy

Removing `hansen2005` changed the PDF not at all, which is the direct
confirmation that it was never rendered: `plainnat` emits only cited
entries, so an uncited entry inflates the `.bib` count while leaving the
bibliography at 22. The discrepancy is now closed at the source.

### Standing rule

This pass attests to PDF `73dac58e…9639` and nothing else. Any change to
`paper/`, `src/` or the build scripts invalidates it and requires a new
build, a new hash and a fresh pass.

---

## Round 6 — PASSED, reproducible release artifact (`v0.13.1`)

The `v0.13.0-manuscript` artifact was built with `\date{\today}`, so the
same source would have produced a different page 1, and a different
hash, on any later day. The inspection attests to a hash, so that
artifact was reproducible only on the day it was cut. The date is now
frozen and the pass repeated.

| Field | Value |
|---|---|
| PDF SHA-256 | `3dc678f40de5a7c2fc64f79191fd5488770dc8bed2e9fe7ea31d270106d4ae58` |
| Commit | `91c6807` |
| CI run | 31995859922 |
| Page count | 15 |
| Bytes | 346,347 |
| TeX | pdfTeX 3.141592653-2.6-1.40.26, TeX Live 2024 |
| Python | 3.12 (GitHub Actions ubuntu-latest) |
| Tests | 268 passing |
| Citations | 22 cited, 22 verified, 0 pending |
| Visual result | **No defects** |

### Reproducibility, demonstrated rather than asserted

| Control | State |
|---|---|
| Title date | `config.yaml` `manuscript.release_date` → generated `\releaseDate`. `\today` removed. |
| Guard | `check_paper.py` fails on `\today`, `\pdfcreationdate`, `\pdffilemoddate` in manuscript source. Mutation-tested: reinstating `\today` exits 1. |
| PDF metadata | `SOURCE_DATE_EPOCH=1786924800` (2026-08-17T00:00:00Z) and `FORCE_SOURCE_DATE=1`, both derived from the same release date. |
| macros.tex | Wall-clock header stamp removed; it recorded a build time that made the file differ on every run. |
| Tests | `tests/test_reproducible_build.py`, five checks. |

**Independent rebuild:** the workflow was re-run on a fresh runner with
a fresh checkout of the same commit. The two PDFs are **byte-identical**
(`3dc678f4…ae58` both times). Reproducibility is verified, not assumed.

### Visual result

Page-by-page text extraction against the `v0.13.0-manuscript` artifact
shows **all 15 pages identical**. The visible document did not change;
only the PDF metadata that was previously taken from the wall clock,
which is why the hash differs. Page 1 was additionally confirmed by eye:
the title date reads "August 17, 2026", now sourced from the frozen
macro rather than from the compile date.

### Tag history — disclosure

`v0.13.0-manuscript` was pushed, then **moved**. The original tag object
`ea47d80` asserted "Tests 260 passing"; the true count was 261, and the
tag was force-updated to `09cdccc` with the correct figure roughly a
minute later. It had been published, so moving it was the wrong remedy
even though it was unconsumed.

**Going forward, published tags are preserved.** A correction is issued
as a new tag — which is why the date freeze ships as `v0.13.1` rather
than by moving `v0.13.0-manuscript` again. `v0.13.0-manuscript` remains
pointing at `842f4ff` and is left alone.
