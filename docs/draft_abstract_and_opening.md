# Draft: Abstract and Introduction Opening

Written to `docs/writing_style.md`. Every number traces to a generated
artifact. Not yet the paper — a demonstration that the rules produce
usable prose, and a target for the full draft.

---

## Abstract (147 words)

Between 2010 and 2026 I ran a minimum-variance portfolio of five US
exchange-traded funds that conditioned its covariance forecast on
volatility states inferred in real time from a two-state Gaussian hidden
Markov model, and compared it against the identical portfolio without
regime conditioning. Net of 10 basis points in transaction costs, the
regime-aware strategy earned an annualized Sharpe ratio 0.021 higher.
The 95% stationary-bootstrap interval on that difference runs from
−0.075 to +0.115, and the annualized return difference is 0.3 basis
points. Regime conditioning did lower annualized volatility by 0.175
percentage points, but it required 3.2 times the turnover of its
comparator, and the point estimate falls from 0.031 to 0.011 as costs
rise from zero to 20 basis points. Across thirteen preregistered
robustness specifications the sign reverses twice and no comparison
survives multiplicity adjustment. I registered the hypothesis, sample
split, and robustness grid before estimating any model.

**Notes on construction.** The first sentence says what I did and over
what sample, concretely. The second and third give the result and its
interval together. The fourth gives the mechanism — the sentence that
makes a null result worth reading. The fifth reports the fragility. The
last states the design as a fact, not a virtue. No literature, no "we
investigate whether," 147 words.

---

## Introduction, first three paragraphs

A minimum-variance portfolio that conditions its covariance forecast on
real-time volatility regimes earned an annualized Sharpe ratio 0.021
higher than the same portfolio without regime conditioning over
2010–2026, net of 10 basis points in costs. The 95% confidence interval
on that difference runs from −0.075 to +0.115. The strategy did what the
theory predicts — annualized volatility fell 0.175 percentage points,
with an interval excluding zero — but it traded 3.2 times as much as its
comparator, and the advantage shrinks monotonically as costs rise. On
this evidence, regime conditioning delivers a measurable risk reduction
that is too small relative to sampling variation to establish a
risk-adjusted improvement once trading costs are paid.

That conclusion is worth more than the usual null because I could not
have engineered it. Before estimating a single model I published the
hypothesis, the comparator, the sample split, the cost convention, and
all thirteen robustness specifications, and tagged them in a public
repository. Four later amendments each carry a date and a statement of
what was known when I made them; three were adopted before any portfolio
return existed. The comparison that reaches the abstract is the one I
committed to in advance, not the one that survived a search.

The choice of comparator does most of the work. Regime-aware minimum
variance beats equal weight and 60/40 comfortably on realized Sharpe,
and a paper could stop there. It should not: those benchmarks differ
from the regime strategy in optimization, in covariance estimation, and
in regime conditioning simultaneously, so a gap between them attributes
nothing. I therefore compare against rolling Ledoit-Wolf minimum
variance, which shares every ingredient except the regime signal. The
0.021 difference is what regime conditioning adds on top of dynamic
covariance estimation, and it is indistinguishable from zero.

**Notes on construction.** The first sentence is a result. Nobody
wonders about efficient markets, nobody reviews a literature, nobody
explains why volatility regimes matter to policymakers. Paragraph two
puts the preregistration to work — it converts a weak finding into a
credible one — rather than parading it. Paragraph three anticipates the
obvious objection (why not benchmark against 60/40, where the strategy
looks good?) and answers it with the ablation logic. Active voice
throughout. No adjective describes my own work.

---

## Rejected openings, and why

> "Financial economists have long debated whether asset returns exhibit
> regime-dependent behavior."

Cochrane's exact prohibition. Philosophy, throat-clearing, and it makes
the paper interesting only because others wrote about the subject.

> "This paper investigates whether regime-switching models can improve
> portfolio performance."

Says what I look for, not what I find. Cochrane's abstract rule applies
to the introduction too.

> "Volatility clustering is one of the most robust empirical regularities
> in financial markets, documented since Mandelbrot (1963)."

A literature review disguised as an opening. The reader still does not
know what this paper does by the end of the sentence.

> "Can regime identification improve out-of-sample risk-adjusted
> performance?"

The research question as an opener. Tempting and common, but it delays
the answer by a paragraph for no gain. State the finding; the question
is implied by it.
