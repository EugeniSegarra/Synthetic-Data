# FHS2 research map and methodological decisions

Information cutoff: **30 October 2026**. This is a targeted review supporting an executable experiment, not a claim to have read all synthetic-finance literature. The original `research.md` and `path_research.md` remain unchanged. Publisher abstracts/institutional records were inspected unless the access note below says full HTML or technical documentation. No theorem is claimed merely by citing its title.

## From the observed discrepancy to a falsifiable experiment

The old six-month drawdown distribution is too rarely realized in its mild lower tail, but six upper-tail breaches all involve the same 2020 episode. Three mechanisms may produce this pattern: a location assumption (zero log drift), variance dynamics (a single short-run recursion), and innovation ordering/shape conditional on the recent state. They are not identified by a single aggregate coverage statistic. An eight-candidate factorial separates them and retains the original FHS as control. Adding drift uses the identical risk path; conditioning shocks preserves each state's exact weighted zero mean and unit variance. These choices isolate mechanisms rather than silently changing several at once.

The expanding mean is shrunk toward zero with 5,040 pseudo-observations. This regularization strength is a declared design choice, not an empirically optimal constant. Welch and Goyal [1] explain why fitted equity-premium predictability is a weak basis for forecasting. Campbell and Thompson [2] motivate economic restrictions, not permission to use later realized equity returns as a prior. This experiment does not equate log drift with an arithmetic premium, nor add a risk-free yield whose availability is not verified. An observed performance improvement would concern this mean assumption and this history, not proof of predictable alpha.

## Heterogeneous trading timescales, without mislabeled models

Corsi's HAR-RV [3] combines realized-volatility information over heterogeneous horizons. Engle and Lee's component model [4] separates long- and short-run volatility components. Neither is identical to our implementation. We implement **HARCH(1,5,22)** using daily squared-return averages, with the exact recurrence documented by `arch` [5]:

`h[t] = omega + alpha1*eps[t-1]^2 + alpha5*mean(eps[t-5:t]^2) + alpha22*mean(eps[t-22:t]^2)`.

The coefficients are estimated by Gaussian quasi-likelihood; simulation resamples centered, standardized empirical residuals. Positivity and coefficient sum below 0.9999 are checked. Training and simulation use the same equation. This is a restricted ARCH with heterogeneous lag windows, not a long-memory theorem, intraday realized variance, or an implementation of Component-GARCH. Comparing it with GARCH/GJR is a parsimonious test of time-scale sensitivity. Full Component-GARCH and measurement-equation Realized GARCH remain separate candidates requiring their own code, data and validation.

## A causal state law rather than copied future regimes

The state at time t depends on returns through t-1: whether preceding 21-session squared-return variance exceeds its 252-session counterpart, and the sign of the last return. Four pools use the same causal labels in estimation and simulation. The local empirical distribution is shrunk toward the full training pool with weight `n_state/(n_state+500)`. Weighted centering and normalization remove unintended conditional mean and scale changes. For GJR, every state's negative second moment enters its stability check; an inadmissible conditional model is rejected, not repaired by coefficient clipping.

This is inspired by state-dependent transitions, not a fitted hidden Markov model. Hamilton-style hidden regimes would require filtered state probabilities at issuance and simulated future transitions; smoothed labels computed using later prices are prohibited. GAS [6] provides another observation-driven parameter mechanism. Full HMM/GAS versions are reviewed but **not implemented or empirically rejected here**. The current state law can miss magnitude dependence, transition duration and conditional drift. Its weighted effective sample size measures weight concentration, not the number of independent crises.

## Transfers from other disciplines

**Experimental design:** the factorial holds draws, information sets and horizons fixed. Common uniforms reduce noise in paired comparisons; they do not increase the number of observed markets. An apparent two-factor interaction is descriptive evidence about model components, not a causal claim about how investors behave.

**Ecology and clustered experiments:** overlapping forecast targets and synchronized global markets create pseudoreplication. The evaluation unit is the issue date, with block resampling and non-overlapping phase diagnostics. EWJ, EEM and VGK are external transfer checks, not three independent copies of the COVID crash. Raw row counts and weighted donor counts cannot certify independent tail evidence.

**Meteorological forecast verification:** scoring a full distribution rewards both calibration and useful sharpness. Threshold-weighted CRPS [7] evaluates severe drawdowns without selecting only observations whose outcomes were extreme. Selecting the test sample by realized severity can reward a permanently pessimistic forecaster. The full-path kernel and first-passage diagnostics from the prior module remain guardrails; endpoint fit alone cannot identify ordering of losses.

**Survival analysis:** recovery of a known previous peak remains administratively censored at the forecast horizon. Unrecovered paths are not assigned an invented completed recovery date. All paths used for scoring are complete for the relevant horizon, while incomplete annual targets remain unscored. This distinction prevents the availability of more recent short-horizon observations from silently fabricating annual evidence.

**Sequential learning with delayed outcomes:** a 126-session forecast cannot supply a performance label until its target matures. Joulani et al. [8] study delayed-feedback learning. Our selector is a simple last-60-matured-origin empirical-risk minimizer, with a 24-origin minimum, not their algorithm or a new regret theorem. The same maturity discipline applies to the separate isotonic event-probability comparator. It does not regenerate paths and is never described as a repaired joint price law.

## Why the advanced models are not silently promoted

| Family | Mechanism worth testing | Data/assumptions | Falsification and current status |
|---|---|---|---|
| Rough volatility [9] | Irregular latent volatility and forecast scaling | High-frequency measurement quality, separating noise from roughness | Test forecast RV across scales and then drawdowns; not implemented, no superiority claimed. |
| Hawkes / self-exciting jumps [10] | Residual clustering of extreme events | Defined event threshold, observation frequency, stable excitation | Must improve residual-event clustering and path scores after volatility filtering; not implemented. |
| GARCH + extreme-value tails [11] | Extrapolate standardized residual tails beyond finite historical support | Threshold stability, enough independent exceedances, parameter uncertainty | Compare thresholds only in inner windows; cannot manufacture independent crises; not implemented. |
| Quant GAN / TimeGAN [12,13] | Flexible nonlinear temporal dependence | Larger training design, seed sensitivity, explicit conditioning information | Must beat locked conditional path forecasts, not just unconditional stylized facts; not implemented. |
| Conditional diffusion [14,15] | Controllable conditional path generation | Conditions available or forecast at issuance, adequate sample/computation | A generator conditioned on realized future trend/volatility is not a forecast of their distribution. No neural training was performed. |
| Full path signatures [16] | Path ordering and geometry | Correct time augmentation, normalization, kernel bandwidths | Prior area ablation is not a full signature kernel. Existing full-grid score retained; no false exact replication. |
| Adaptive conformal [17] | Coverage adaptation under distribution change | Feedback arrival and objective-specific assumptions | Interval/event correction is separate from coherent joint paths; reviewed, not claimed as a universal coverage guarantee. |
| Rare-event splitting [18] | Efficient estimation inside a specified simulator | Valid state cloning and conditional transition law | Can reduce Monte Carlo error, not model misspecification or sampling uncertainty; reviewed, not implemented. |

Recent results were inspected up to the cutoff, including CoFinDiff (2025), a GAN-diffusion framework (May 2026), and the August 2026 diffusion survey [14,15,19]. They broaden the candidate set; they do not substitute for replication on this project's task. CoFinDiff's controllable trend/volatility conditions are particularly useful for scenario design, but those conditions cannot be borrowed from the forecast's future outcome. The GAN-diffusion work focuses on multi-asset correlation realism, which differs from six-month univariate drawdown calibration. These are distinctions in objectives, not evidence those approaches fail.

## Uncertainty, recalibration and decision value

There are four different uncertainties: finite Monte Carlo resolution, parameter estimation, model structure and future market nonstationarity. The primary study quantifies paired historical-score uncertainty using a stationary bootstrap over forecast origins. Simultaneous max-standardized intervals address the declared comparison family approximately; stationarity/dependence assumptions remain material. Block lengths are sensitivity parameters fixed before evaluation, not chosen for significance.

Separate block-bootstrap sensitivities cover the regularized drift and volatility parameters. Across six issue states, 180 parameter refits were attempted and 171 accepted; all nine rejections are recorded. Each accepted parameter vector is re-filtered on the actual observed history before simulation, so a resampled endpoint is never substituted for the current state. Equal-weight mixtures of accepted refits are sensitivity distributions, not calibrated Bayesian posterior predictives. A fixed ensemble of candidate models is also not a probability-weighted posterior. Failed calibration rates and missing comparisons remain visible. A timestamped issuance file records its actual creation time; a historical data cutoff does not retrospectively turn that file into an archived live forecast.

The loss ratio 5:1 for a missed drawdown alarm versus a false alarm yields threshold 1/6. These are abstract, declared decision costs. We compare with always and never alert. There is no assumed fill price, transaction cost, leverage or trading return. Better average CRPS cannot automatically be reported as improved fund performance.

VIX must not be treated as physical volatility without considering the variance-risk premium [20]. VIX and IRX acquisition failed the current price-series data contract; neither is smuggled into the experiment or forward-filled. Daily rates/indices need an appropriate, availability-aware data contract. A separate successful VIX3M snapshot is retained for research but not used as a feature. No point-in-time macro feed, credit data, intraday RV, or pre-1993 SPY history is asserted.

## Sources and access scope

1. Welch & Goyal, *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction*. NBER working paper 10483 (2004), published RFS 2008. Institutional abstract: https://www.nber.org/papers/w10483
2. Campbell & Thompson, *Predicting the Equity Premium Out of Sample*. NBER 11468 (2005), published RFS 2008. Institutional abstract: https://www.nber.org/papers/w11468
3. Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*. Publisher abstract: https://doi.org/10.1093/jjfinec/nbp001
4. Engle & Lee (1999), *A Long-Run and Short-Run Component Model of Stock Return Volatility*. Publisher record: https://academic.oup.com/book/52280/chapter-abstract/421087965
5. `arch` HARCH technical documentation, full formula inspected and checked against installed implementation: https://arch.readthedocs.io/en/stable/univariate/generated/arch.univariate.HARCH.html
6. Creal, Koopman & Lucas (2013), *Generalized Autoregressive Score Models with Applications*. Publisher abstract/record: https://doi.org/10.1002/jae.1279
7. Gneiting & Ranjan (2011), *Comparing Density Forecasts Using Threshold- and Quantile-Weighted Scoring Rules*. Author-institution record: https://stat.uw.edu/research/tech-reports/comparing-density-forecasts-using-threshold-and-quantile-weighted-scoring-rules
8. Joulani, Gyorgy & Szepesvari (2013), *Online Learning under Delayed Feedback*. Primary abstract: https://proceedings.mlr.press/v28/joulani13.html
9. Gatheral, Jaisson & Rosenbaum, *Volatility is rough*, arXiv 1410.3394 (2014); primary abstract: https://arxiv.org/abs/1410.3394
10. Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance*. Primary abstract: https://arxiv.org/abs/1502.04592
11. McNeil & Frey (2000), *Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach*. Publisher abstract: https://www.sciencedirect.com/science/article/pii/S0927539800000128
12. Wiese et al., *Quant GANs*, arXiv 1907.06673 (2019), Quantitative Finance 2020. Primary abstract: https://arxiv.org/abs/1907.06673
13. Yoon, Jarrett & van der Schaar (2019), *Time-series Generative Adversarial Networks*. Primary abstract: https://proceedings.neurips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
14. Tanaka et al. (2025), *CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation*. Primary HTML methods inspected: https://arxiv.org/abs/2503.04164
15. Masi, Coletta & Bartolini (2026), *High-Quality Synthetic Financial Time-Series using a GAN-Diffusion Framework*. Preprint, full HTML sections inspected: https://arxiv.org/html/2605.27113v1
16. Dodson & Dutta, *Signature Kernel Scoring Rule*, arXiv 2510.19110; see the prior `path_research.md` access record and explicit distinction from this implementation.
17. Gibbs & Candes (2024), *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts*. JMLR primary abstract: https://jmlr.org/beta/papers/v25/22-1218.html
18. Cerou, Guyader & Rousset (2019), *Adaptive multilevel splitting*. See prior `path_research.md` for the primary citation and access limitation; no replication claimed.
19. Wang & Ventre (2026), *Diffusion Models in Finance: A Survey*. Preprint; primary HTML inspected as a research map, not original evidence of individual models: https://arxiv.org/abs/2608.12583
20. Bekaert & Hoerova (2014), *The VIX, the variance premium and stock market volatility*. Institutional record: https://www.nber.org/papers/w18995
21. Gurrola-Perez & Murphy (2015), *Filtered historical simulation Value-at-Risk models and their competitors*, Bank of England WP525. Institutional overview: https://www.bankofengland.co.uk/working-paper/2015/filtered-historical-simulation-value-at-risk-models-and-their-competitors
