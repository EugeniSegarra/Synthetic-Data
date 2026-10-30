# Validation report: implementation evidence and market benchmarks

**Review date: 30 October 2026. Package version: 1.0.0.**

The joint filtered historical simulation (FHS) model has lower average CRPS,
energy, and variogram scores than the joint unfiltered bootstrap in each of the
three retrospective windows below. This is evidence for this comparison on this
snapshot, not a universal performance ranking or certification of risk
probabilities. The machine-readable results, configuration, fitted parameters,
diagnostics, and provenance are in [market_benchmark.json](../reports/market_benchmark.json).

> **Material limitation: the 2022 portfolio 5% tail is undercovered.** FHS has
> **31 breaches in 251 observations (12.35%)**, versus **12.55 nominally expected**
> at a 5% tail probability. Its observed-to-nominal ratio is **2.47**. The
> unfiltered baseline has 51 breaches (20.32%). Improving on the baseline does
> not establish calibration. The model should not be described as a calibrated
> standalone 95% VaR estimator across these regimes.

![Retrospective benchmark comparison, including portfolio tail coverage](assets/benchmark.svg)

## 1. Experiment and scope

The data are daily adjusted SPY, TLT, and GLD OHLC observations obtained through
Yahoo Finance / yfinance, with 4,232 aligned price observations from 4 January
2010 through 29 October 2026. The request used an exclusive end date of
30 October 2026 and `auto_adjust=True`. Strict alignment retained all input
dates; no dates were dropped. Adjusted OHLC are research price series, not
executable historical quotes. The benchmark evaluates close returns only.

For each window, the model is fitted once on the training observations. Each
asset uses Gaussian quasi-likelihood estimation, selecting the lower-BIC
admissible GARCH(1,1) or GJR-GARCH(1,1) candidate from training data only. The
log-return mean is zero. Standardized training residuals are centered and scaled
to population standard deviation one, and complete residual rows define the
joint empirical forecast distribution. The model does not draw independent
shocks for different assets.

Each test-day forecast is produced before that day's return is used. The first
test return includes the last training close. After scoring, the observed return
updates the variance state for the next forecast. Parameters, training residuals,
and their joint empirical dependence remain fixed throughout the test window.
This is **fixed-parameter, sequential one-step evaluation**, not rolling refitting
or a test of unconditional multi-year simulated paths.

The baseline uses complete raw training return rows, recentered to the same
log-return mean as the model. Both methods therefore share the mean assumption.
The portfolio has one-third weight in each asset for every one-period forecast,
using `expm1(asset_log_returns) @ weights`. This is a fixed-weight daily portfolio
return calculation. It is not a buy-and-hold wealth backtest and includes no
transaction costs, execution assumptions, or currency conversion.

| Window | Training price interval | Training returns | Test return interval | Test observations |
| --- | --- | ---: | --- | ---: |
| COVID 2020 | 2010-01-04 to 2019-12-31 | 2,515 | 2020-01-02 to 2020-12-31 | 253 |
| Inflation 2022 | 2010-01-04 to 2021-12-31 | 3,020 | 2022-01-03 to 2022-12-30 | 251 |
| Recent 2025–2026 | 2010-01-04 to 2024-12-31 | 3,773 | 2025-01-02 to 2026-10-29 | 458 |

These are **retrospective chronological out-of-sample experiments**. They were
not prospectively preregistered, and the development process inspected their
results. Chronological separation prevents future observations from entering
an individual fitted forecast; it does not make this release's research choices
independent of all observed outcomes. There was no recalibration against 2022
to remove its reported failure. A descriptive reporting flag was added after
reviewing the diagnostics; it changes neither forecasts nor scores. The windows
are not independent replications: their training histories overlap and all use
the same three assets.

## 2. Distributional scores

Lower is better for all three scores. CRPS is calculated exactly against every
training innovation row, then averaged equally over assets and test dates. The
energy score uses an exact first term and a Monte Carlo estimate of its pairwise
second term: 4,096 iid index pairs sampled with replacement, seed `20261030`.
The same pairs are used for model and baseline, and across dates within a window.
The energy-score differences below are observed estimates; no claim of
statistical significance or Monte Carlo confidence interval is made.

The variogram score uses order 0.5 and equal weights over the three unordered
asset pairs, with weights summing to one. All scores use decimal log-return
vectors. Scores have different definitions and scales, so comparisons are
within a score, not between score columns. The scoring rationale follows
[Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437) and
[Scheuerer and Hamill (2015)](https://doi.org/10.1175/MWR-D-14-00269.1);
their specific roles are discussed in [the research review](research.md).

| Window | Score | FHS | Joint unfiltered baseline | Reduction versus baseline |
| --- | --- | ---: | ---: | ---: |
| COVID 2020 | Mean marginal CRPS | 0.00740611 | 0.00791210 | 6.40% |
| COVID 2020 | Energy | 0.01492674 | 0.01597456 | 6.56% |
| COVID 2020 | Variogram | 0.00258851 | 0.00386067 | 32.95% |
| Inflation 2022 | Mean marginal CRPS | 0.00716936 | 0.00736546 | 2.66% |
| Inflation 2022 | Energy | 0.01453821 | 0.01477569 | 1.61% |
| Inflation 2022 | Variogram | 0.00197567 | 0.00215988 | 8.53% |
| Recent 2025–2026 | Mean marginal CRPS | 0.00570464 | 0.00591057 | 3.48% |
| Recent 2025–2026 | Energy | 0.01183373 | 0.01209285 | 2.14% |
| Recent 2025–2026 | Variogram | 0.00202964 | 0.00234302 | 13.37% |

Reduction is `(baseline - FHS) / baseline`. Values are rounded here; the JSON
contains full precision. The aggregate result does not imply improvement for
every asset: for example, GLD's 2022 CRPS is 0.00536644 for FHS versus 0.00534921
for the baseline, a small deterioration.

## 3. Portfolio tail coverage

A breach means the realized simple portfolio return is **strictly below** the
forecast's empirical lower-tail quantile. Quantiles use the inverse empirical
CDF (`inverted_cdf`). With a finite discrete distribution, the probability
strictly below the quantile can be smaller than the nominal tail probability.
The JSON reports both nominal expected counts and the sum of predicted strict
breach probabilities. This convention explains small expected-count differences;
it does not explain the 2022 failure.

### Lower 5% tail: nominal 95% coverage

| Window | Observations | Nominal expected breaches | FHS breaches | FHS breach rate | Baseline breaches | Baseline breach rate | FHS observed / nominal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COVID 2020 | 253 | 12.65 | 13 | 5.14% | 22 | 8.70% | 1.03 |
| **Inflation 2022** | **251** | **12.55** | **31** | **12.35%** | **51** | **20.32%** | **2.47** |
| Recent 2025–2026 | 458 | 22.90 | 23 | 5.02% | 34 | 7.42% | 1.00 |

The predicted strict expected counts are 12.5746, 12.4669, and 22.8211,
respectively, for both methods. In 2022 the FHS observed-to-empirical-expected
ratio is 2.4866 and realized coverage is only 87.65%.

### Lower 1% tail: nominal 99% coverage

| Window | Observations | Nominal expected breaches | FHS breaches | FHS breach rate | Baseline breaches | Baseline breach rate | FHS observed / nominal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COVID 2020 | 253 | 2.53 | 4 | 1.58% | 15 | 5.93% | 1.58 |
| Inflation 2022 | 251 | 2.51 | 4 | 1.59% | 7 | 2.79% | 1.59 |
| Recent 2025–2026 | 458 | 4.58 | 8 | 1.75% | 14 | 3.06% | 1.75 |

The corresponding strict expected counts are 2.5149, 2.4934, and 4.4914. All
three FHS observed counts exceed nominal expectation at this tail. The small
expected counts limit inference; neither approximate 5% coverage in two windows
nor a small number of 1% events establishes conditional coverage. Formal
independent-binomial p-values are not reported. The distinction between average
coverage and conditional coverage is discussed by
[Christoffersen (1998)](https://doi.org/10.2307/2527341) and in
[the research review](research.md#4-validation-correctness-calibration-and-usefulness-are-different).

The reporting heuristic flags `descriptive_undercoverage` when nominal expected
breaches are at least 10 and observed breaches are at least twice nominal
expectation. It flags both methods at 5% in 2022. The threshold is an explicit
interpretation aid, not a statistically derived critical value, model-selection
criterion, or forecast adjustment. Its minimum-count rule means the 1% tails
receive no such flag here; **absence of a flag is not a calibration pass**.

Expected shortfall is calculated from the worst alpha mass of the empirical
distribution, including fractional boundary mass. Reported ES values are
forecasts, not evidence that ES itself has been validated. Joint-tail counts
and their forecast expected counts are also available in the JSON; rare joint
events provide limited information about dependence.

## 4. Why 2022 is a substantive limitation

The following additional diagnostics use the same saved input and the 2022
report. Training correlations refer to the fitted standardized residual pool.
Test correlations use realized shocks
`z_t = (observed_log_return_t - fitted_mean) / forecast_sigma_t`, where
`forecast_sigma_t` was available before observing that return. Test moments are
calculated ex post for diagnosis, not fed into forecasts.

| Pair | Training residual correlation, 2010–2021 | Realized filtered-shock correlation, 2022 |
| --- | ---: | ---: |
| SPY / TLT | -0.3884 | +0.0837 |
| SPY / GLD | +0.0247 | +0.1971 |
| TLT / GLD | +0.2303 | +0.3460 |

The frozen training pool carries an equity–bond diversification relationship
that weakens materially in 2022. Realized filtered-shock second moments are
1.3537 for SPY, 1.3211 for TLT, and 1.0541 for GLD, compared with one by
construction in the training innovation pool. Thus both dependence and marginal
scale are relevant to the failure.

A descriptive covariance decomposition is consistent with that interpretation.
Let `C_train` be the training innovation covariance and `D_t` the diagonal matrix
of forecast standard deviations. Compare the average `D_t @ C_train @ D_t`
with the population-denominator sample covariance of 2022 log returns. For the
equal-weight vector, the respective portfolio log-variance quantities are
0.0000429191 and 0.0000734976. Off-diagonal covariance terms account for 88.7%
of their difference. This is an ex-post accounting comparison between realized
sample variation and average predicted conditional variation, not a causal
decomposition or a formal estimate of model-error components.

The mean assumption is another possible contributor. The realized daily simple
portfolio mean is -0.0692%. As a diagnostic only, subtracting that full-year
realized mean from every observed portfolio return reduces the 5% breach count
from 31 to 28 against the original thresholds. This oracle calculation uses
future information and is not an alternative backtest. It indicates that
removing the realized mean alone would not eliminate most of the undercoverage.

The broader economic context is consistent with the dependence shift.
[Lombardi and Sushko, BIS Quarterly Review, December 2023, Box A](https://www.bis.org/publications/qr-202312/markets-adjust-to-higher-longer)
document the change toward positive equity–bond return correlations from 2021
and discuss how high inflation and monetary-policy responses weaken bonds'
diversification role. That source supplies context; it does not prove the
cause of this model's forecast errors.

There is no evidence in this diagnostic that the 2022 result is an arithmetic
or chronological-evaluation bug. It is consistent with limitations of a fixed
innovation law and a parsimonious volatility filter. A dynamic-dependence or
regime-conditioned extension would require its own specified training protocol
and fresh evaluation. The original failure remains reported, without tuning a
correlation window, residual half-life, or drift to make 2022 pass.

## 5. Implementation tests and appropriate use

Implementation tests check contracts and formulas: the variance recurrence,
optimizer rejection, empirical standardization, joint row sampling, seeded
reproducibility without changing global RNG state, input preservation, price
units, positive prices, OHLC inequalities, CRPS and ES against known examples,
and chronological isolation of the training set and first forecast. Validation
tests also check that changing future observations cannot change the initial
forecast or the training fingerprint. These are meaningful engineering checks;
passing them does not establish statistical adequacy for an asset universe or
investment decision. See [the tests](../tests/) and
[the research discussion](research.md#4-validation-correctness-calibration-and-usefulness-are-different).

The evidence supports using this release as a transparent, reproducible
conditional scenario engine and baseline for further research. It does not
validate investment alpha, derivative pricing, standalone risk limits,
out-of-sample OHLC extrema, stop/target execution, liquidity, transaction costs,
or machine-learning augmentation. More simulated paths do not remove parameter,
sampling, regime, or model uncertainty.

Forecast paths use an observation-step index named **`Step`**, not invented
future exchange-session dates. Historical evaluation uses the actual observed
dates. Mapping future steps to a trading calendar, or interpreting bars across
markets with different closing times, requires an explicit external convention.

## 6. Reproduction and data provenance

Run the benchmark script from the repository root after installing its optional
Yahoo dependency. Use a new output directory for each run:

```bash
python -m pip install -e '.[yahoo]'
python examples/run_benchmarks.py --out benchmark_run
```

The script fixes the windows, weights, model configuration, seed, and energy-pair
count described above. It saves `input.csv`, `input_metadata.json`, the three
complete reports including per-observation forecasts, and `benchmark_summary.json`.
To rerun against the saved snapshot rather than download again:

```bash
python examples/run_benchmarks.py \
  --csv benchmark_run/input.csv \
  --metadata benchmark_run/input_metadata.json \
  --out benchmark_replay
```

This repository publishes derived benchmark results, not the raw vendor price
snapshot. Obtain data through the provider under its applicable terms and retain
the local CSV and metadata if you need to reproduce your own run. Yahoo can
revise historical observations and corporate-action adjustments; a later download
is not guaranteed to reproduce this report's fingerprint or numbers. These
downloaded adjusted data are not a reconstructed point-in-time data vintage.
Snapshot replay controls input revisions. Matching package and numerical-library
versions further reduces numerical differences, but floating-point optimizer
results need not be bitwise identical across platforms.

The full input's normalized `MarketData` SHA-256 is:

```text
4869dba4e670bd94931ab3a448b11b20896530de3dfcd035ec92e9f7b486c647
```

The fingerprint covers numerical prices, OHLC, dates, asset order, and economic
conventions. It is not a SHA-256 of CSV file bytes and excludes request paths and
other non-economic provenance. Training fingerprints provide an additional
check on the actual estimation inputs:

| Window | Training `data_sha256` |
| --- | --- |
| COVID 2020 | `1fcfbfcdf2f41da2655884e216b54249275b02d73d38a2687adfa89491dc0f76` |
| Inflation 2022 | `d7c7b2332f93d7084c1c7735e5c39a43f5cfd5f49139c1d345dea1eb199f1f5e` |
| Recent 2025–2026 | `8534bb6e03385dabbb3ba31e18c28dff4e4cbf1d3c77af496ea6a68700b7c86d` |

### Recorded execution environment

| Component | Version |
| --- | --- |
| Package | 1.0.0 |
| Python | 3.12.14 |
| Platform | Linux-6.18.57-x86_64-with-glibc2.39 |
| arch | 8.0.0 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| statsmodels | 0.15.0 |
| yfinance | 1.7.0 |

The FHS, GARCH/GJR, scoring, conditional-coverage, and possible dynamic-dependence
foundations are linked and annotated in
[Research basis and model decisions](research.md#6-sources-and-their-specific-role).
The references explain the methods; the numerical evidence for this release is
the explicitly dated experiment and its published JSON.
