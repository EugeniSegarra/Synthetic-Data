# Full-path risk validation: research and engineering decisions

Research cutoff: 30 October 2026. This is a targeted extension of [the original review](research.md), not a claim to have read every paper on synthetic finance. The implementation is in `synthetic_market.path_validation`; the fixed experiment is in [the protocol](path_protocol_2026-10-30.md). Methods inspired by other fields are distinguished from exact replications, unimplemented ideas, and proved properties.

## 1. The object being forecast is a path

A one-day return model can pass marginal checks and still misrepresent accumulated losses. A terminal-price distribution also does not determine maximum drawdown, the placement of volatility, or the duration of losses. Our adversarial example rearranges the same four returns: terminal return and total quadratic variation stay unchanged while maximum drawdown changes. All path outcomes use the complete daily grid, never the downsampled display used in an earlier visualization.

For each issue date, the implementation generates an open-loop distribution of the next 126 daily returns, then evaluates its 21-, 63- and 126-step prefixes. No observed future return is injected into a simulated trajectory. Forecast parameters are re-estimated at each new issue date using the preceding 1,260 returns. This is different from evaluating 126 sequential one-day forecasts that receive realized information along the way.

The empirical study concerns adjusted daily SPY prices and closing-price drawdowns. It does not observe intraday highs/lows, execution, order queues, liquidity, or margin liquidation. Future maximum drawdown includes the issue-date price in the peak search. Recovery of a previously known peak is a separate outcome.

## 2. Meteorology: coherent ensemble verification, not attractive fan charts

Lorenz's analogue forecasting asks what followed historical states similar to today's state [1]. Hydrometeorology's Schaake shuffle shows that plausible marginal forecasts can have the wrong spatial and temporal ordering [2]. The transfer is methodological: preserve and test joint temporal structure. It is not a claim that asset prices satisfy atmospheric physical equations.

We implement an experimental analogue continuation generator. Eligible donor starting states use only information available before the donor continuation. The entire continuation must already be observed by the current financial forecast origin. Four state coordinates describe relative short/medium volatility, normalized momentum and known trailing drawdown. Robust scales are estimated from eligible past states. The 100 nearest states receive smooth distance weights; whole contiguous continuations are drawn and scaled by the ratio of current to historical past-21-day RMS volatility. The exact weighted lead-wise mean is removed. This prevents the analogue from receiving an unintended return-prediction advantage over zero-log-mean FHS.

This is not an implementation of Schaake's rank-reordering algorithm. It is a conditional continuation bootstrap inspired by the broader coherence problem. All design constants were recorded before inspecting this run's results. The continuation law remains finite-support and is particularly vulnerable to sparse independent episodes and shifts between historical states and future outcomes.

The first empirical outcome is adverse: both analogue variants have worse six-month drawdown CRPS than FHS. They remain challengers, not defaults.

## 3. Rough paths and geometry: distinguish variance level from its timing

The time at which variation arrives can matter even when total variation is unchanged. On each discrete path we augment cumulative return with normalized time and cumulative squared return. We compute two exact second-level antisymmetric areas for the piecewise-linear interpolation: time/return and time/quadratic-variation. A midpoint-integral oracle verifies these formulas independently. The analogue ablation adds a scale-free coordinate describing whether past variation occurred early or late in the last 63 sessions.

Dodson and Dutta's weather work, revised April 2026, uses a signature-kernel scoring rule to evaluate entire spatio-temporal forecasts [3]. Lyu and Lee's October 2026 preprint distinguishes variance-scale information from mean-reversion-speed information and reports that improved regime identification does not automatically produce a resolved portfolio Sharpe gain [4]. These motivate a testable hypothesis, not an assertion that signature features must help SPY.

Our implemented score is deliberately simpler than [3]: a fixed mixture of Gaussian RBF kernels on an injective full daily cumulative-return vector augmented by the two areas. It is NOT the full signature kernel, a truncated-signature-only distance, or a replication of either paper. Since the complete cumulative-return vector is retained, the embedding is injective for a fixed daily horizon and positive training scale. The characteristic Gaussian kernel therefore identifies distributions on this finite-grid embedding; areas alone would not. Positive mixtures retain this property. This reasoning concerns the population kernel score, not a claim that a finite historical sample can distinguish every pair of models.

The loss is `0.5 E[k(X,X')] - E[k(X,y)] + 0.5`, with fixed bandwidths 0.5, 1 and 2. The normalizing scale is computed from training returns only. Distinct sampled ensemble-member pairs estimate the pair expectation; the approximation can be slightly negative and is not clipped. This is a bounded distribution score with an explicit Monte Carlo error, rather than an unbounded high-degree signature expansion. For the mixture, simulation is conditional on its issued empirical expert ensembles. CRPS in the report is the exact score of the finite empirical forecast distribution; it is not advertised as an unbiased finite-ensemble estimator of the underlying latent CRPS.

The area feature's ablation is explicit: compare `analog_area` with `analog`, not just with an unrelated baseline. Its observed six-month improvement over the plain analogue is small relative to uncertainty, and both remain worse than FHS. No stronger conclusion is supported.

## 4. Biostatistics and reliability: crossing a loss barrier is an event-time problem

A medical study that ends before recovery cannot label a subject recovered at the study end. The identical censoring mistake appears when simulation software converts every unfinished drawdown into a completed H-day episode. Kvamme and Borgan analyze administrative censoring in probability scoring [5]; Jonkers et al.'s 2026 preprint formalizes proper scoring on the law actually observable under censoring [6].

We implement cumulative first-passage probabilities for forward drawdowns of 5%, 10% and 20%. A loss of 10% relative to the issue-date price is evaluated separately: after first rising, a path can suffer a 10% drawdown without losing 10% of initial wealth. The event is an inequality including boundary equality within a documented numerical tolerance. An unhit barrier remains a non-event at every observed time, not a fabricated final-day hit.

For origins below their already known historical peak, we evaluate the probability of recovering that peak by every future session. Paths not recovered by H are right-censored. Integrated Brier scores compare the complete observed event-indicator curve with forecast event probabilities. Because every scored origin has all H outcomes observed, no estimated censoring weights are necessary within that horizon. This is a fixed-horizon evaluation, not unrestricted lifetime inference.

We also report the restricted mean `E[min(T,H)]`, and keep the censoring flag separate. At-or-above-peak origins are excluded from recovery scoring rather than counted as easy successes. The longest observed underwater run within a forecast remains a bounded path feature; it is not mislabeled a completed episode duration. Daily closing barriers may miss intraday crossings.

## 5. Ecology: pseudoreplication applies to financial simulations

Hurlbert's ecological field-experiment critique distinguishes many measurements from many independent experimental units [7]. The financial counterpart is exact: 2,000 simulated paths from one fitted generator do not provide 2,000 independent real market histories. Similarly, overlapping forecast windows can repeatedly contain the same crash.

The study's uncertainty unit is the real forecast origin. Ordered paired loss differences are resampled using the stationary bootstrap [8], with geometric blocks of three predeclared lengths. Six non-overlapping phase views are shown for 126-day outcomes issued every 21 sessions. These phases are sensitivity views, not six independent studies. Bootstrap intervals are approximate, pointwise and conditional on dependence/stationarity assumptions; regime changes can undermine them. No multiplicity-adjusted discovery or universal coverage guarantee is claimed.

The realized path record now attributes each maximum drawdown to its peak and trough dates. In the main sample, all six FHS six-month upper-95% drawdown breaches share the February 2020 peak. They are not six independent crisis successes or failures. The analogue audit also counts the maximum number of disjoint donor continuations. A set of 100 weighted neighbors typically contains only four disjoint six-month continuations here. Even disjointness is not independence. A Kish weight count near 100 must not be presented as 100 independent episodes.

## 6. Online learning: adaptation must wait for its evidence

A six-month path forecast issued today cannot be given its final six-month score tomorrow. Joulani, Gyorgy and Szepesvari study learning with delayed feedback [9]. We implement a transparent discounted exponential mixture of FHS, the stationary bootstrap and the area analogue. A forecast's kernel loss enters the weight update only once its full 126-session target has matured. All horizons use the same coherent full-path mixture; daily fragments are not mixed into a path after observing their success.

Learning rate 1 and forgetting 0.97 are fixed experimental design choices, not optimized results. The implementation does not inherit the paper's regret guarantees automatically. Future-data mutation tests verify that information after an issuance date cannot change its forecasts or available mixture weights. Empirically the mixture does not establish an improvement over FHS at six months, so it does not replace the default.

## 7. Physics and rare events: useful transfers that do not solve the wrong problem

Maximum drawdown can be viewed through first-passage mathematics; Brownian analyses provide tractable laboratory cases [10]. They do not justify importing Brownian scaling, a universal Hurst exponent, or a universal crisis law into empirical SPY forecasts. Our controls test barrier definitions and path ordering without claiming those physical assumptions describe the market.

Adaptive multilevel splitting, developed for rare-event simulation and related to interacting-particle methods, is a plausible way to estimate very small barrier probabilities efficiently within a known simulator [11]. It is NOT implemented in this release. It would reduce simulation error conditional on the FHS state transition, but would not repair an incorrect innovation law or manufacture independent crises. Zero hits among 2,000 paths now receive a Wilson upper bound instead of being communicated as certainty. Even that interval describes simulator sampling error only, not financial model uncertainty.

Adaptive conformal methods are another useful research direction [12]. They can correct prediction sets under distribution shift for specified coverage objectives. They do not by themselves identify a coherent joint path distribution, provide conditional crisis coverage, or validate drawdown/recovery dynamics. We therefore do not inflate intervals after observing this holdout and then declare the generator correct.

## 8. Scope of the delivered evidence

The fixed protocol, all six comparators, ablation, adverse calibration, dependent-origin intervals, source hashes and numerical referee are published together. No tested challenger is promoted. The output is useful for diagnosing which property fails, comparing alternative generators on the same path problem, and auditing changes without silently changing the benchmark. It is not a production-risk approval, evidence of alpha, or a claim that this is the universally best synthetic-data method.

The research data are a revisable Yahoo snapshot. This is rolling-origin pseudo-out-of-sample evaluation on known history, not a prospectively locked unseen period. No parameter or model uncertainty is created merely by increasing the path count. The original module's finite innovation support, fixed fitted parameters within each path, and drift assumptions continue to constrain interpretation.

## References and access scope

Primary sources were used. Full HTML was inspected for [3]; linked primary abstracts, publisher descriptions and methodological records were consulted for the other references below. The release does not claim a line-by-line replication of papers whose full text was not inspected. Preprints are labeled.

1. Lorenz, E. (1969), *Atmospheric Predictability as Revealed by Naturally Occurring Analogues*. Journal of the Atmospheric Sciences 26, 636-646. [Publisher](https://journals.ametsoc.org/view/journals/atsc/26/4/1520-0469_1969_26_636_aparbn_2_0_co_2.xml).
2. Clark, M., Gangopadhyay, S., Hay, L., Rajagopalan, B. and Wilby, R. (2004), *The Schaake shuffle: A method for reconstructing space-time variability in forecasted precipitation and temperature fields*. Journal of Hydrometeorology 5, 243-262. [USGS primary publication](https://pubs.usgs.gov/publication/70027101).
3. Dodson, A. and Dutta, R. (2026 revision), *Signature Kernel Scoring Rule: A Spatio-Temporal Diagnostic for Probabilistic Weather Forecasting*. arXiv:2510.19110v2, 30 April 2026, preprint. [Full HTML](https://arxiv.org/html/2510.19110v2).
4. Lyu, Y. and Lee, K. (2026), *Scale or speed? When do path signatures improve volatility-regime forecasts?* arXiv:2610.06690, 5 October 2026, preprint. [Primary abstract](https://arxiv.org/abs/2610.06690). Its target and feature family differ from our experiment.
5. Kvamme, H. and Borgan, O. (2023), *The Brier Score under Administrative Censoring: Problems and a Solution*. JMLR 24(2), 1-26. [JMLR](https://www.jmlr.org/papers/v24/19-1030.html).
6. Jonkers, J., Van Wallendael, G., Duchateau, L. and Van Hoecke, S. (2026), *Proper Scoring Rules for Right-Censored Survival Data*. arXiv:2606.06393, 4 June 2026, preprint. [Primary abstract](https://arxiv.org/abs/2606.06393).
7. Hurlbert, S. (1984), *Pseudoreplication and the Design of Ecological Field Experiments*. Ecological Monographs 54(2), 187-211. [Publisher DOI](https://doi.org/10.2307/1942661).
8. Politis, D. and Romano, J. (1994), *The Stationary Bootstrap*. JASA 89, 1303-1313. [DOI](https://doi.org/10.1080/01621459.1994.10476870); [earlier Stanford technical report](https://statistics.stanford.edu/technical-reports/stationary-bootstrap).
9. Joulani, P., Gyorgy, A. and Szepesvari, C. (2013), *Online Learning under Delayed Feedback*. ICML, PMLR 28(3), 1453-1461. [PMLR](https://proceedings.mlr.press/v28/joulani13.html).
10. Magdon-Ismail, M., Atiya, A., Pratap, A. and Abu-Mostafa, Y. (2004), *On the Maximum Drawdown of a Brownian Motion*. Journal of Applied Probability. [Caltech record](https://authors.library.caltech.edu/records/2aprk-1ee98).
11. Cerou, F., Guyader, A. and Rousset, M. (2019), *Adaptive multilevel splitting: Historical perspective and recent results*. Chaos 29, 043108. [Publisher](https://pubs.aip.org/aip/cha/article/29/4/043108/1027309/Adaptive-multilevel-splitting-Historical).
12. Gibbs, I. and Candes, E. (2024), *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts*. JMLR 25(162), 1-36. [JMLR](https://jmlr.org/beta/papers/v25/22-1218.html).
13. Gneiting, T. and Raftery, A. (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA 102, 359-378. [DOI](https://doi.org/10.1198/016214506000001437); [author-institution record](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation-revised).
14. Gurrola-Perez, P. and Murphy, D. (2015), *Filtered historical simulation Value-at-Risk models and their competitors*. Bank of England Working Paper 525. [Bank of England](https://www.bankofengland.co.uk/working-paper/2015/filtered-historical-simulation-value-at-risk-models-and-their-competitors).
