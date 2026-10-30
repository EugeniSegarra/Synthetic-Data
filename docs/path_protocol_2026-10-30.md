# SPY path study: protocol and publication gate

Recorded 30 October 2026 before inspecting numerical results of this run. This is a retrospective protocol, NOT preregistration of unseen financial history. The historical episodes are known. Bug repairs and numerical corrections must be documented; outcome-driven parameter retuning is not permitted in the reported comparison.

## Data and information set

SPY adjusted daily closes, Yahoo snapshot requested from 1993-01-01 through the exclusive end 2026-10-30. Preserve its file hash and metadata. Check session completeness against an independently maintained NYSE calendar. All forecasts use only a prefix ending at the issue date. Revisions to Yahoo history mean pseudo-out-of-sample rather than immutable point-in-time evidence.

## Fixed experiment

- Rolling training window: 1,260 returns, refit at every origin.
- First origin on or after 2000-01-01; subsequent origins every 21 observed sessions.
- Common complete-origin grid for 21, 63 and 126 sessions; exclude uncompleted 126-session targets at the sample end.
- 2,000 full daily paths per method and origin, seed 20261030; no downsampling for metrics.
- Historical diagnostic period: 2000-2014. Main evaluation slice: origin >= 2015-01-01. Neither slice is claimed to be prospectively untouched.
- Zero conditional expected log return for all methods. This is a risk-model comparison, not an alpha forecast. Zero log mean is not zero arithmetic expected return.

## Methods, fixed before results

1. Published FHS with admissible GARCH/GJR training-only BIC selection.
2. Mean-centered iid historical returns.
3. Circular stationary bootstrap of mean-centered returns, mean block length 10 sessions.
4. Experimental conditional analogue continuations: 100 nearest matured historical starting states; donor outcomes fully contained in training; whole contiguous continuations; past-21-day RMS scaling; exact lead-wise weighted-mean removal. State features: short/long and medium/long volatility ratios, normalized 63-day momentum and known 252-day drawdown.
5. The same analogue with an additional scale-free time/quadratic-variation area feature. This ablation tests a rough-path-inspired timing feature, not an assumed advantage.
6. A mixture of FHS, block and area-analogue experts. Discounted exponential weights, learning rate 1 and forgetting 0.97. Update only when a prior 126-session forecast has completely matured, using its full-path kernel score. Mix complete paths, not daily fragments.

## Outcomes and scores

Forward maximum drawdown includes the origin in the peak search. It resets the prospective peak to the origin, and is kept separate from recovery of a known pre-origin peak. Evaluate terminal simple return, close-to-close RMS volatility (252 convention), maximum drawdown, time-underwater fraction, longest observed underwater run and average drawdown. Observed run length is not an uncensored completed episode duration.

Report CRPS for path functionals, upper-tail drawdown CRPS above 10%, 90% interval coverage with separate tails, and full daily path kernel scores. The path score is a fixed mixture of Gaussian kernels on an injective cumulative-return embedding augmented with exact second-level antisymmetric areas. It is NOT the full signature kernel from the recent weather literature. Fixed bandwidths: 0.5, 1, 2; 4,096 sampled distinct ensemble-member pairs.

Evaluate cumulative first-passage probabilities for 5%, 10% and 20% drawdowns with endpoint and integrated Brier scores. Evaluate origin-relative 10% losses separately. For origins already below their historical peak, score recovery by each future day; unrecovered paths are right-censored and remain non-events within the horizon. Do not report infinite-horizon mean recovery times.

## Uncertainty and failed origins

The observation unit is the forecast origin, NOT one Monte Carlo path. Report all fit failures and explicit common-method denominators. Main uncertainty is a paired stationary-bootstrap interval over ordered forecast origins. Block lengths are ceil(H/21), twice that value (main), and four times that value. Use 2,000 resamples. These are pointwise approximate intervals under dependence assumptions, not multiplicity-adjusted discoveries or universal finite-sample guarantees. Show all non-overlapping phase views; phases are not independent replications.

More simulated paths reduce simulator noise, not uncertainty about the real financial data-generating mechanism. Do not infer regulatory or production risk fitness from a passing software suite.

## Adversarial gate

Numerical checks use independent scalar references and exact small-pair calculations. Attack equal endpoints with reordered losses, first-day crashes, barrier ties, unrecovered episodes, degenerate input, future-data mutation, premature delayed-feedback updates, missing origins, output overwrites and reproducibility. Approval refers to software and the validity of reported diagnostics; an empirical model failure must remain published rather than being removed by the gate. No external-human or independent-language-model review is claimed by this protocol.

## Promotion rule

No challenger silently replaces the default engine. Publish all comparator results and adverse evidence. A useful improvement in one metric does not establish calibration of all path properties. Any future model-selection policy needs a new locked prospective test.
