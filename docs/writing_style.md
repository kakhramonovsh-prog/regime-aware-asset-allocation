# Writing Style Guide for the Paper

Rules for drafting *Regime-Aware Volatility Forecasting and Dynamic Asset
Allocation*. Written before the paper, so the draft is built to these
standards rather than edited toward them afterward.

Sources: John Cochrane's *Writing Tips for Ph.D. Students* (2005, Chicago
Booth) — the canonical writing guide in financial economics, read in full
for this guide; DeMiguel, Garlappi and Uppal (2009, *Review of Financial
Studies*) as the model for framing a null result; Harvey and Liu on
backtest haircuts and multiple testing; the `stop-slop` prose skill; and
the human-voice research in the project memory. Citation details and
their verification status appear in §10.

---

## 1. The one-sentence contribution

Cochrane's first instruction: figure out the single central contribution
and write it in one paragraph, concretely. Everything else follows from
getting this right.

For this paper:

> Conditioning a minimum-variance portfolio on latent volatility states
> estimated in real time from a two-state Gaussian HMM raises the
> out-of-sample net Sharpe ratio by 0.021 over an otherwise identical
> non-regime strategy, a difference whose 95% bootstrap interval runs
> from −0.075 to +0.115. The strategy reduces annualized volatility by
> 0.175 percentage points but trades 3.2 times as much, and the design
> was frozen and published before any result existed.

That paragraph is concrete: it names the number, the interval, the
mechanism, and the design safeguard. Compare the version to avoid: "We
investigate whether regime-switching models improve portfolio
performance and find mixed evidence." That says nothing.

**Test:** if a reader can restate the contribution after reading only
the abstract, the paper is organized correctly.

## 2. Shape: newspaper, not mystery novel

Cochrane's rule is that papers should be triangular — most important
first, details later — because readers skim and no one reads
start-to-finish. A joke or a novel builds to a punchline; a paper puts
the punchline first and then explains it.

The binding rule for the body: **nothing may appear before the main
result that a reader does not need in order to understand the main
result.**

For this paper that means the order is: contribution → design → main
result → mechanism → robustness → limitations. It does *not* mean:
motivation → literature → data description → volatility models → regime
models → covariance → strategies → and finally, on page 22, the Sharpe
difference. The temptation here is severe, because the project was
*built* in that order across nine phases. The paper must not retrace
the construction sequence. Cochrane: a good paper is not a travelogue of
the search process.

Practical consequence: Phases 4 through 8 (EDA, volatility forecasting,
regime diagnostics, covariance conditioning, weight construction) are
**method and appendix material**, not narrative chapters. The volatility
model comparison, where GARCH won on QLIKE, is one paragraph plus an
appendix table, not a section — it feeds nothing in the main
specification, since EWMA was fixed ex ante.

## 3. Abstract

100–150 words. State what the paper finds, not what it looks for. No
literature. No "we investigate whether."

Draft skeleton, to be filled from generated output only:

1. What was tested, concretely (one sentence naming the strategies and
   the sample).
2. The headline number with its interval.
3. The mechanism: volatility fell, turnover rose, costs consumed the
   difference.
4. The preregistration, stated as a fact about the design rather than a
   virtue claim.

## 4. Introduction

Start with the contribution. First sentence is the hardest and must not
be philosophy ("Financial economists have long wondered whether markets
are efficient"), a literature observation ("A growing literature
examines regime-switching models"), or a policy motivation. Cochrane
calls all of it throat-clearing.

Three pages maximum. A roadmap paragraph is optional and Cochrane skips
it; skip it here.

**Opening candidates for this paper** (pick one, do not stack them):

- *Result-first:* "A minimum-variance portfolio that conditions on
  real-time volatility regimes earned a Sharpe ratio 0.021 higher than
  the same portfolio without regime conditioning, net of 10 basis points
  in costs. The 95% confidence interval on that difference runs from
  −0.075 to +0.115."
- *Design-first:* "I registered the hypothesis, the comparison, the
  sample split and the entire robustness grid before estimating a single
  model, then reported whatever the data produced."

The result-first opening is stronger and matches Cochrane's rule. The
design-first framing belongs in the second paragraph, where it does real
work: it converts a null result from a weak finding into a credible one.

## 5. Literature review

Placed **after** the contribution, in its own section a reader can skip.
Cochrane: readers cannot judge how the paper differs from others before
they understand the paper, and most have not read the other papers.

Set the paper against two or three closest works, not a catalogue. For
this paper the three that matter:

1. **DeMiguel, Garlappi and Uppal (2009)** — the direct precedent. They
   evaluate 14 portfolio models across seven datasets and find none
   consistently beats naive 1/N on Sharpe ratio, certainty-equivalent
   return or turnover, because estimation error offsets the theoretical
   gain from optimization. This paper extends the same question one rung
   further: does *regime conditioning* survive the same out-of-sample
   test? The answer has the same shape, and saying so explicitly places
   a null result inside an established and highly cited tradition rather
   than presenting it as a personal failure.
2. **Hamilton (1989)** — the regime-switching framework itself, and the
   source of the discipline that states are latent and inferred, not
   observed.
3. **Harvey and Liu** on backtest haircuts and multiple testing — the
   reason this paper preregisters one primary comparison and applies
   Holm adjustment to everything else. Their argument is that with
   hundreds of tested factors, a t-statistic above 2.0 no longer
   establishes anything. A paper that reports a *single* preregistered
   comparison plus an adjusted robustness family is answering that
   critique directly.

Be generous with citations. Nobody has to be wrong for this work to be
worth reading.

## 6. Prose rules

From Cochrane, with the diagnostic that finds each violation.

| Rule | Diagnostic |
|---|---|
| Active voice. "I estimate," not "the model was estimated." | Search `is`, `are`, `was`, `were` |
| "I" for a sole author. Never the royal "we." "We" means you-and-the-reader. | Search `we` |
| Present tense, kept consistent within a paragraph. | Read each paragraph for tense drift |
| Delete everything before "that" in most sentences. "It should be noted that" is the worst offender. | Search `that` |
| Clothe the naked "This." Write "this regression shows," never "this shows." | Search `This ` at sentence start |
| No adjectives describing your own work. Not "striking results," not "significant improvement." If the work merits adjectives, readers supply them. | Search the adjective list |
| Short words. "Use" not "utilize," "several" not "diverse." | Read aloud |
| Subject-verb-object. Keep clauses few. | Any sentence over ~30 words |
| No previews or recalls. "As we will see in Table 6" signals bad ordering. | Search `we will see`, `recall` |
| "In which" for models; "where" for places. | Search `where` |
| No cute opening quotation. | — |

**Self-audit of my own project reports**, run mechanically over 345
sentences across nine phase reports:

| Check | Result |
|---|---|
| Passive-voice candidates | **88 (26%)** |
| Naked "This" | **15** |
| Self-describing adjectives | 5 |
| Throat-clearing openers | 0 |
| Mean sentence length | 22.1 words |
| Runs of 3 same-length sentences | 12 |

The passive rate is the real problem and it will not survive into the
paper. Two examples of my own prose and their repairs:

> **Before:** "The state-conditioned estimates are computed from the
> smoothed responsibilities of the HMM fit through that origin."
> **After:** "I compute the state-conditioned estimates from the
> smoothed responsibilities of the HMM fit through that origin."

> **Before:** "Every solve converged and every solution passed
> independent validation."
> **After:** "All 800 optimizations converged, and I verified each
> solution's constraints directly rather than trusting the solver flag."

The second repair does more than fix voice: it replaces "every" with the
count and names what the verification actually was.

## 7. Tables and figures

- Self-contained captions. A skimming reader must understand the table
  without hunting through the text for definitions.
- **Two to three significant digits.** Not 0.020966 — write 0.021. A
  Sharpe difference reported to six decimals claims precision the data
  cannot support.
- Sensible units. Percentages and basis points, not 0.0000023.
- **No number in a table that goes undiscussed in the text.** "Table 5
  shows summary statistics" alone is not acceptable. If a number is not
  worth a sentence, cut it from the table.
- Figures beat tables for showing patterns. The forest plot across 13
  robustness specifications communicates the sign instability better
  than any table could, and it belongs in the main paper.
- Label axes. Verbal definition of every symbol in the caption.

## 8. Reporting the result

**Economic magnitude, not just statistical significance.** Cochrane's
point is that significance is cheap in large samples and says nothing
about whether an effect matters. This paper has the opposite problem —
nothing is significant — so the discipline runs the other way: report
the magnitude honestly and resist implying importance the interval does
not support.

Rules for this specific paper:

- Never write "outperformed," "improved" or "beat" for a difference
  whose interval contains zero. Write what happened: "earned a Sharpe
  ratio 0.021 higher in this sample."
- Always attach the interval to the point estimate in the same sentence
  or the one immediately following.
- State the economic size in an interpretable unit. A 0.021 Sharpe
  difference and a 0.314 basis point annualized return difference are
  both small; say so plainly rather than leaving the reader to work it
  out.
- Distinguish the confirmatory comparison from everything else in the
  sentence itself, not only in a table caption.
- Report the volatility reduction as suggestive and unadjusted, never as
  established.

**Claims about research integrity — use this wording, not stronger.**
The safeguards constrained discretion; they did not eliminate it.
Implementation decisions were made throughout (zero-mean GARCH, ×100
scaling, HAC lag 3, `covariance_type`, `n_iter`, the selection
tolerance), and four amendments changed the frozen plan after it was
tagged. Claiming the pipeline "could not be steered" overstates what
preregistration achieves and invites a reviewer to find the one
counterexample.

The sanctioned sentence:

> Preregistration, frozen data, causal tests, automated validation, and
> dated amendments substantially constrained ex-post researcher
> discretion. All amendments were documented before viewing the results
> they could influence.

Banned formulations: "could not have been steered," "impossible to
influence," "the result was inevitable," "guarantees objectivity," any
phrasing implying zero remaining discretion.

**Framing the null (the DeMiguel model).** Their abstract states the
scope tested, then the plain negative finding, then the mechanism:
estimation error more than offsets the theoretical gain. The mechanism
sentence is what makes a null result publishable — a finding that
something failed is uninteresting; a finding that explains *why* the
theoretically superior method fails is a contribution.

The analogous structure here: the theoretical gain from conditioning on
volatility states is real and shows up as a measurable volatility
reduction, but it is small relative to sampling variation, and the
turnover required to harvest it consumes a meaningful share of it. That
sentence is the paper.

## 9. What to cut

Cochrane, applied to material this project has in abundance:

- **The construction narrative.** Nine phases of scaffolding is process,
  not contribution.
- **Warm-up exercises and preliminary estimates.** The volatility model
  comparison is one paragraph plus an appendix table.
- **Extensive description of well-known data.** SPY and IEF need no
  introduction.
- **The 57 robustness checks in the body.** Summarize the grid in the
  text, put the full table in an appendix, keep the forest plot in the
  main paper.
- **"I leave X for future research."** Cochrane: readers are less
  interested in plans and excuses than in results.
- **Anything that reads as defending the effort rather than reporting
  the finding.** The preregistration is a fact about the design, stated
  once with a tag reference, not a recurring argument for credibility.

Target length: 25–35 pages including tables. Cochrane caps final papers
at 40 and says shorter is better.

## 10. Paper structure

**Working title:** *Do Latent Volatility Regimes Improve Risk-Based
Asset Allocation? Evidence from a Preregistered Walk-Forward ETF Study*

Main text, ten sections:

1. Introduction and research question
2. Literature review
3. Data and information-availability policy
4. Preregistered empirical design
5. Volatility and regime estimation
6. Covariance and portfolio construction
7. Primary performance and inference
8. Robustness
9. Limitations
10. Conclusion

**Core results, kept compact.** These seven belong in the paper's spine;
everything else is appendix:

- Primary ΔSharpe = +0.021
- 95% CI [−0.075, +0.115]
- One-sided p = 0.327
- Suggestive −0.175 percentage-point volatility difference
- Sign reversals under three states and dropping realized volatility
- No robustness comparison surviving Holm
- Strong sensitivity to weight constraints and realized exposures

**The 30% cap stays out of the abstract**, with at most one sentence
noting that portfolio constraints materially affected apparent
performance. It is a robustness specification that fails multiplicity
adjustment; giving it abstract space would misrepresent its standing.

**Closing sentence, fixed:**

> Regime conditioning modestly reduced realized volatility in the
> primary specification but did not produce statistically credible
> improvement in net risk-adjusted performance. Results were sensitive
> to state specification, feature selection, sample period, and
> portfolio constraints.

**Unit discipline.** Every number in every table passes through
`src/units.py`, which stores `raw_value`, `display_value` and
`display_unit` together and re-derives the conversion before writing.
No manual conversion enters the LaTeX. Two order-of-magnitude errors
reached draft reports before this existed; both now have regression
tests.

## 11. Sources and verification status

Full bibliographic details go into `references.bib` only after each is
checked against the publisher record, per the project's standing rule.

| Source | Status |
|---|---|
| Cochrane, J. H. (2005). *Writing Tips for Ph.D. Students.* Chicago GSB, June 8 2005. | **Read in full** from the Rice University hosted copy; every rule above traces to the document |
| DeMiguel, V., Garlappi, L., Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies* 22(5), 1915–1953. | Volume, issue, pages and finding confirmed across RFS, SSRN and RePEc listings; **DOI still to verify** |
| Harvey, C. R., Liu, Y. Backtesting. SSRN 2345489; Harvey, Liu, Zhu, "…and the Cross-Section of Expected Returns," NBER w20592. | Findings confirmed from NBER and SSRN listings; **full citation to verify** |
| Hamilton (1989), Ledoit-Wolf (2004), Patton (2011), Diebold-Mariano (1995), Politis-Romano (1994), Hansen (2005) | Verified with DOIs in `docs/research_design.md` §13 |

## 12. Pre-submission checklist

1. Can a reader state the contribution after the abstract alone?
2. Does the first sentence contain a result rather than a motivation?
3. Is the literature review after the contribution and skippable?
4. Does anything precede the main result that is not needed to
   understand it?
5. Run the passive-voice search. Fix every instance.
6. Run the naked-"This" search.
7. Does every table number get discussed in the text?
8. Is every figure caption self-contained with labeled axes?
9. Two to three significant digits everywhere?
10. Does every point estimate carry its interval?
11. Does any sentence claim more than the interval supports?
12. Read aloud: any sentence you would never say?
13. Any three consecutive sentences of the same length?
14. Is the conclusion short, free of new claims, and free of a plans list?
