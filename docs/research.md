# Research basis and model decisions

**Evidence review date: 30 October 2026.** This is a targeted review for a daily,
multi-asset scenario generator. It covers the econometric foundations, current
clearing-house practice, probabilistic evaluation, and selected modern generative
models. It is not a claim to have read every publication or to have established a
universally best generator. Recent methods are distinguished from demonstrated
improvements on this repository's task.

## 1. What this release is designed to do

The release produces conditional daily scenarios for an explicitly supplied,
aligned set of positive asset price or adjusted-price series. It is intended for
portfolio scenario analysis, reproducible experiments, and testing analytical
software. It does not infer expected excess returns, generate a historical
point-in-time investment universe, price derivatives under a risk-neutral
measure, or reproduce the sequence of transactions within a trading session.

The chosen model is **joint filtered historical simulation (FHS), with a
GARCH(1,1) or GJR-GARCH(1,1) volatility filter for each asset**. By default,
the lower-BIC admissible model is selected using the training data only; either
family can also be selected explicitly. The volatility filter is
estimated using a Gaussian quasi-likelihood. The simulation distribution is the
empirical distribution of centered, unit-variance standardized residual vectors;
it is not a Gaussian innovation distribution. A single historical row is drawn
for all assets at a simulated time step. The default log-return mean is zero;
an estimated constant mean is an explicit alternative. Zero log drift does not
mean zero expected arithmetic return, and neither choice is an alpha forecast.

This architecture follows the semiparametric logic of FHS [3]. GARCH represents
time-varying conditional scale [1]; the GJR term allows negative and positive
shocks of the same magnitude to have different effects on subsequent variance
[2]. The empirical joint shock distribution supplies the observed marginal
shapes and contemporaneous dependence without deriving a copula tail parameter
from unrelated marginal parameters.

## 2. Evidence translated into implementation choices

| Evidence | Decision in this release | What the decision does not establish |
| --- | --- | --- |
| GARCH and GJR provide parsimonious volatility dynamics [1, 2]. | Use the explicit one-lag recurrence (gamma=0 for GARCH), select only converged, empirically stable fits by training BIC, and retain the last observed state. | The model is not guaranteed to capture long memory, jumps, structural breaks, or every leverage effect. |
| FHS rescales historical shocks to current conditional volatility [3]. | Resample whole cross-asset residual rows and propagate conditional volatility along each path. | More simulations do not create more information about the true market distribution. |
| Copula dependence and marginal tails are distinct objects [4]. | Remove the heuristic that inferred a t-copula's degrees of freedom from marginal estimates. No parametric copula is included in v1. | An empirical joint distribution is not an extrapolation model for previously unseen joint shocks. |
| Dynamic conditional correlation is a separate model family [5]. | Keep the empirical residual dependence fixed within a fitted model and state that limitation. | Volatility variation alone is not a model of changing residual correlations. |
| Empirical return properties require multiple diagnostics [6]. | Assess tails, serial dependence, volatility behavior, and portfolio-relevant dependence separately. | Resembling a few stylized facts does not certify a generator. |
| Proper scores assess distributions against realized observations [7, 8]. | Compare chronological one-step forecasts with a joint, unfiltered historical-bootstrap baseline using CRPS, energy, and variogram scores. | A favorable score on one dataset does not establish universal superiority. |
| Conditional coverage matters as well as average coverage [9, 10]. | Report tail exceedances and joint exceedances in addition to scores; distinguish descriptive coverage from formal certification. | Few tail observations cannot validate rare-event risk. |
| Clearing-house frameworks combine historical risk with separate stress components [11, 12]. | Treat FHS as a useful scenario engine with explicit empirical support limits. | This package does not reproduce a CCP margin model or satisfy its regulatory validation requirements. |

### Standardization and stability are part of the model

The two-stage procedure first fits a volatility filter and then centers and
scales its standardized residuals to define a finite empirical innovation law.
This is an explicit semiparametric modeling choice. It must not be described as
sampling the exact Gaussian likelihood fitted in the first stage. The estimated
filter parameters, residual-center correction, and residual-scale correction
belong in the model diagnostics so that material discrepancies can be noticed.

For the recurrence

\[
h_{t+1}=\omega+\alpha\epsilon_t^2
       +\gamma\epsilon_t^2\mathbf{1}_{\{\epsilon_t<0\}}+\beta h_t,
\qquad \epsilon_t=\sqrt{h_t}\,z_t,
\]

the empirical innovations have mean zero and second moment one. The relevant
second-moment persistence is therefore

\[
\beta+\alpha+\gamma\,E_{\widehat F}
   [z^2\mathbf{1}_{\{z<0\}}].
\]

Replacing the last expectation by one half requires symmetry that an empirical
shock distribution need not have. The implementation checks the actual empirical
quantity and rejects a fit that fails its stability requirement; it must not
silently clip coefficients to make a failed model appear stable. This condition
concerns finite unconditional variance under the specified iid innovation law.
It is not a guarantee against large simulated excursions or future structural
change.

With a finite empirical shock support and finite initial state, every fixed,
finite simulation horizon has bounded model support in exact arithmetic. This
avoids the non-existent positive exponential moment obtained by directly
exponentiating an untruncated Student-t log return. Numerical overflow remains
possible for extreme inputs or horizons and must produce an explicit failure,
not silently clipped prices. Empirical support also has a cost: the model cannot
learn an unobserved standardized shock from historical resampling. Conditional
volatility can still produce a return larger than any raw return in the sample.

### What joint row resampling preserves

Using one row index across assets preserves the empirical contemporaneous joint
innovation distribution. Resampling each asset independently would destroy that
dependence. It also avoids imposing t-copula tail dependence on independent,
heavy-tailed input merely because the marginal distributions have finite degrees
of freedom. Sampling error in the input remains: the empirical distribution of
independent finite samples need not have exactly zero sample correlation or the
population co-exceedance rate.

The default resampling is independent across simulated times. The volatility
recurrence generates temporal dependence in return magnitudes and squared
returns; a constant-mean model does not create linear return predictability.
Additional residual serial dependence is not automatically preserved. A block
bootstrap [13] is a legitimate alternative when residual dynamics justify it,
but it is not included as an unexplained default: it can preserve misspecification
as well as useful temporal structure. Residual diagnostics are necessary before
making that extension.

## 3. OHLC: empirical bar construction and its limits

OHLC generation requires actual, consistently adjusted OHLC observations.
Closing prices alone do not identify overnight gaps or intraday extrema. For
each historical asset-day, the model records the overnight log gap and the
nonnegative upper and lower log excursions, divided by the fitted daily
conditional volatility. They are sampled using the same historical row as the
close-return innovation vector.

Reconstruction uses dimensionless log quantities: the open is obtained from the
previous close through an exponential gap; the high is the greater of open and
close multiplied by an exponential nonnegative upper excursion; the low is the
lesser of open and close multiplied by an exponential negative lower excursion.
Thus positivity, the OHLC inequalities, and invariance to a change of price units
follow from the construction, subject to explicit floating-point checks. This
is the repository's empirical modeling choice, not a claim that one of the cited
papers proves the resulting bars reproduce all properties of a trading session.

Centering/scaling the innovation and choosing the daily mean changes the
synthetic close relative to the raw historical gap template. This is deliberate:
the adjusted body closes the accounting identity, but historical intraday body
means and joint body distributions are not preserved exactly. OHLC integrity
tests do not establish out-of-sample calibration of gaps or extrema; the release's
predictive benchmark evaluates close returns, not intraday trading performance.

No synthetic 24-day cycle is labeled an intraday pattern. No uncalibrated
asymmetric-normal gap distribution is inserted. Because the output does not
specify whether the high or low occurred first, it cannot settle the execution
of competing stops and targets or provide a transaction-cost model. Adjusted
bars represent an adjusted-price process; future dividends, splits, liquidity,
spreads, and market impact are not separately simulated.

Observation steps are kept distinct from exchange sessions. A business-day
frequency that only removes weekends is not an exchange calendar. Alignment
across assets, common currency, missing observations, and synchronous session
closes remain explicit input responsibilities; they must not be hidden by
automatic price forward-filling.

## 4. Validation: correctness, calibration, and usefulness are different

**Engineering correctness** concerns deterministic contracts: asset ordering,
units, valid inputs, horizon length, preservation of the input, reproducible RNG
behavior, exact recurrence calculations, positive prices, OHLC inequalities,
serialization, and explicit optimizer failures. Tests for these properties can
pass even when the economic model is inadequate for a particular dataset.

**Controlled model experiments** use processes with known properties. Examples
include independent heavy-tailed innovations, a known correlated process, a
negative-shock volatility response, and scaling all prices by the same constant.
The independence experiment specifically checks that near-zero rank correlation
does not conceal artificially imposed extreme co-movement. Monte Carlo
tolerances must account for both the finite calibration sample and the number
of simulated draws; exact equality to population statistics is not expected.

**Chronological predictive evaluation** fits on the training interval only. The
release evaluates the next observation, updates the volatility state with that
observed return, and moves forward while leaving fitted parameters and the
empirical shock distribution frozen. This is a fixed-parameter, sequential
one-step evaluation. It is not a rolling-refit exercise, an unconditional
multi-year validation, or an assertion that the realized test path should lie
near the median of one long simulated path.

CRPS evaluates each marginal predictive distribution; the energy score evaluates
the joint vector. The variogram score adds sensitivity to dependence
misspecification because the energy score can be comparatively insensitive to
incorrect correlation [7, 8]. Coverage and joint-tail frequencies answer
different questions and are reported separately. All metrics use the same
chronological observations and are compared with a joint historical-bootstrap
baseline. Any aggregation or scaling convention must be shown in the evaluation
output rather than left implicit.

Rare-tail validation has a sample-size limit. At a nominal one-percent tail,
250 test observations imply only 2.5 expected marginal exceedances; joint
exceedances can be much rarer. A small count, a large p-value, or no observed
breach is not proof that tail risk is correctly modeled. Numerical regression
tests, distributional diagnostics, and a benchmark report are evidence of
different things and must not be compressed into an unsupported claim of
"financial realism passed."

Synthetic data do not constitute an independent test set for the generator that
created them. For machine-learning augmentation, fit the generator entirely
inside the training interval, select methods using a separate validation
interval, and retain untouched real observations for the final evaluation.
Simulation uncertainty conditional on a fitted model is different from
parameter uncertainty, sampling uncertainty, and model uncertainty.

## 5. Modern generative research and why it is not the default

TimeGAN combines adversarial and supervised objectives to preserve sequence
dynamics [14]. Quant GANs use temporal convolutional networks for financial
process generation [15]. Conditional signature-based GANs target richer
conditional path information [16]. These are relevant alternatives, not evidence
that a neural model automatically improves this repository's daily portfolio
scenarios. A conditional path model, unconditional synthetic-data generator,
probabilistic forecaster, and risk-neutral simulator solve different problems.

Takahashi and Mizuno's diffusion study [17] uses wavelet representations of
minute-level AAPL returns, spreads, and volumes. Its reported experiment uses
one stock, shuffles daily samples into training and validation groups, and
winsorizes extreme transformed observations. That supports the method's promise
for its stated intraday representation problem; it does not establish superior
chronological, multi-asset tail-risk forecasts. Adopting it here would require a
new benchmark, a substantially different data contract, and additional training
infrastructure.

The 2026 MarketGAN preprint [18] is closer to the portfolio problem: it combines
an asset-pricing factor structure with joint return generation and evaluates
cross-sectional and tail dependence. Its portfolio findings depend on the
factor-information setup, and the paper is a preprint in the retrieved record.
It is a candidate for a future comparison, not an implementation or verified
result supplied by this package. A recent survey published in October 2026 [19]
also organizes evaluation around distributional fidelity, financial properties,
and downstream utility. Only its abstract and publication metadata were
accessible in this review; no claim is made to have independently reviewed its
131 underlying studies.

The release therefore adopts an evidence-informed, transparent baseline. It
does not claim a state-of-the-art performance ranking. A more complex method
would merit inclusion after a comparable, leakage-controlled benchmark shows
that its benefits survive different market periods and exceed its estimation,
computational, and maintenance costs. Publishing an untested GAN beside a
working econometric model would not meet that standard.

## 6. Sources and their specific role

The links below point to primary papers, author/institutional records, official
proceedings, and practitioner documentation. Publication years are distinguished
from preprint or online-posting dates where relevant. Summaries are selective
paraphrases; no paper's results are represented as results of this repository.

1. **Bollerslev, T. (1986).** *Generalized autoregressive conditional
   heteroskedasticity.* Journal of Econometrics 31, 307–327.
   [DOI](https://doi.org/10.1016/0304-4076%2886%2990063-1).
   Basis for the conditional variance recurrence.

2. **Glosten, L. R., Jagannathan, R., and Runkle, D. E. (1993).** *On the
   Relation between the Expected Value and the Volatility of the Nominal Excess
   Return on Stocks.* Journal of Finance 48, 1779–1801.
   [DOI](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x);
   [institutional record](https://www.minneapolisfed.org/research/staff-reports/on-the-relation-between-the-expected-value-and-the-volatility-of-the-nominal-excess-return-on-stocks).
   Basis for asymmetric variance response.

3. **Barone-Adesi, G., Giannopoulos, K., and Vosper, L. (1999).** *VaR without
   correlations for portfolios of derivative securities.* Journal of Futures
   Markets 19, 583–602.
   [Institutional paper record](https://susi.usi.ch/usi/documents/318043);
   [paper](https://folia.unifr.ch/documents/318043/files/wp9904.pdf).
   Basis for filtered historical simulation and joint historical shocks.

4. **Demarta, S., and McNeil, A. J. (2005; working paper 2004).** *The t Copula
   and Related Copulas.* International Statistical Review 73, 111–129.
   [DOI](https://doi.org/10.1111/j.1751-5823.2005.tb00254.x);
   [authors' paper](https://www.risknet.de/fileadmin/eLibrary/t-Copula-Demarta-ETH.pdf).
   Distinction between marginal laws, joint tail dependence, and copula estimation.

5. **Engle, R. (2002).** *Dynamic Conditional Correlation: A Simple Class of
   Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models.*
   Journal of Business & Economic Statistics 20, 339–350.
   [DOI](https://doi.org/10.1198/073500102288618487);
   [author's paper](https://pages.stern.nyu.edu/~rengle/dccfinal.pdf).
   Documents a possible dynamic-dependence extension; not implemented here.

6. **Cont, R. (2001).** *Empirical properties of asset returns: stylized facts
   and statistical issues.* Quantitative Finance 1, 223–236.
   [DOI](https://doi.org/10.1080/713665670).
   Motivation for assessing several distributional and temporal properties.

7. **Gneiting, T., and Raftery, A. E. (2007).** *Strictly Proper Scoring Rules,
   Prediction, and Estimation.* Journal of the American Statistical Association
   102, 359–378.
   [DOI](https://doi.org/10.1198/016214506000001437);
   [earlier institutional report](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation).
   Basis for CRPS and energy-score evaluation.

8. **Scheuerer, M., and Hamill, T. M. (2015).** *Variogram-Based Proper Scoring
   Rules for Probabilistic Forecasts of Multivariate Quantities.* Monthly Weather
   Review 143, 1321–1334.
   [DOI](https://doi.org/10.1175/MWR-D-14-00269.1);
   [NOAA paper](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf).
   Motivation for supplementing energy scores with dependence-sensitive scores.

9. **Christoffersen, P. F. (1998).** *Evaluating Interval Forecasts.*
   International Economic Review 39, 841–862.
   [DOI](https://doi.org/10.2307/2527341);
   [paper](https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf).
   Distinguishes average coverage from conditional coverage and clustered breaches.

10. **Diebold, F. X., Gunther, T. A., and Tay, A. S. (1998).** *Evaluating
    Density Forecasts with Applications to Financial Risk Management.*
    International Economic Review 39, 863–883.
    [Institutional record](https://archive.nyu.edu/handle/2451/14779);
    [NBER working-paper record](https://www.nber.org/papers/t0215).
    Framework for evaluating conditional predictive distributions.

11. **Eurex Clearing.** *Eurex Clearing Prisma*, official methodology brochure,
    accessed 30 October 2026.
    [Brochure](https://www.eurex.com/resource/blob/32818/179aa114c0a294159c37e54f70eab745/data/brochure_eurex-clearing-prisma.pdf).
    Practitioner use of filtered historical scenarios together with stress,
    event-risk, and other model adjustments; not an endorsement of this software.

12. **CME Group.** *SPAN 2 Methodology*, official documentation, accessed
    30 October 2026.
    [Methodology](https://www.cmegroup.com/clearing/risk-management/span-overview/span-2-methodology.html);
    [framework document](https://www.cmegroup.com/clearing/files/cme-span-2-margin-framework.pdf).
    Practitioner use of historical scenarios with volatility/correlation scaling
    and separate stress/liquidity components.

13. **Politis, D. N., and Romano, J. P. (1994).** *The Stationary Bootstrap.*
    Journal of the American Statistical Association 89, 1303–1313.
    [DOI](https://doi.org/10.1080/01621459.1994.10476870).
    Reference for potential dependent-resampling extensions; not the iid default.

14. **Yoon, J., Jarrett, D., and van der Schaar, M. (2019).** *Time-series
    Generative Adversarial Networks.* NeurIPS 2019.
    [Official proceedings](https://papers.neurips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html).
    Representative learned sequence generator, not implemented in this release.

15. **Wiese, M., Knobloch, R., Korn, R., and Kretschmer, P. (2020).** *Quant
    GANs: deep generation of financial time series.* Quantitative Finance 20,
    1419–1440.
    [DOI](https://doi.org/10.1080/14697688.2020.1730426);
    [authors' preprint](https://arxiv.org/abs/1907.06673).
    Representative financial generator based on temporal convolutions.

16. **Liao, S., Ni, H., Sabate-Vidales, M., Szpruch, L., Wiese, M., and Xiao,
    B. (2024; published online 2023).** *Sig-Wasserstein GANs for conditional time
    series generation.* Mathematical Finance.
    [DOI](https://doi.org/10.1111/mafi.12423);
    [institutional record](https://discovery.ucl.ac.uk/id/eprint/10181357/).
    Reference for conditional path-distribution matching using signature features.

17. **Takahashi, T., and Mizuno, T. (2025).** *Generation of synthetic
    financial time series by diffusion models.* Quantitative Finance,
    1507–1516; published online 5 August 2025.
    [DOI](https://doi.org/10.1080/14697688.2025.2528697).
    Recent intraday diffusion evidence; its data and evaluation scope differ
    materially from this package's daily multi-asset use case.

18. **Huh, J., Jeong, S., Kim, H.-G., Koo, H. K., and Lim, B. H. (2026).**
    *MarketGANs: Multivariate financial time-series data augmentation using
    generative adversarial networks.* Preprint, submitted 25 January 2026.
    [arXiv record](https://arxiv.org/abs/2601.17773);
    [full text](https://arxiv.org/html/2601.17773v1).
    Recent factor-structured multivariate generation and portfolio experiments;
    no independent replication or superiority claim is made here.

19. **Zhang, Y., Duan, J., Zou, Y., Feng, K., and Xiao, J. (2026).**
    *Financial time-series generation: a systematic survey of taxonomy,
    evaluation, datasets, and open challenges.* npj Artificial Intelligence;
    accepted article published 19 October 2026.
    [DOI](https://doi.org/10.1038/s44387-026-00166-3).
    The accessible abstract reports a review of 131 studies from 2021 to early
    2026. Used as a recent map of evaluation dimensions, not as a substitute for
    reading and testing each underlying method.

