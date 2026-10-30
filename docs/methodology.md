# Methodology, assumptions and decision record

This document describes the implemented model, not an idealized future version.
The supporting literature and the treatment of recent generative methods are in
[research.md](research.md).

## Intended statistical object

The output is a conditional probability model for future **daily observations**
of an explicitly supplied set of assets. Every scenario begins at the final
observed prices, residuals and conditional variances. This initial condition is
appropriate for forward scenarios; a stationary burn-in would describe a
different question. Portfolio arithmetic is long-only and fully invested,
with returns expressed in a common currency. The engine does not infer a fund's
actual holdings or expected return.

## Estimation and state

For each asset the training log-return series is multiplied by 100 for numerical
optimization. `arch` fits a zero or constant conditional mean and a Gaussian
quasi-likelihood volatility model. All persisted returns and variances are
converted back to decimal log-return units. In `auto` mode, GARCH(1,1) and
GJR-GARCH(1,1) are compared by training-sample BIC after convergence, positivity,
and empirical stability checks. If neither passes, the whole fit fails with the
asset name and candidate reasons; the universe is never reduced silently.

The common recurrence is:

\[
h_{t+1}=\omega+\beta h_t+
        (\alpha+\gamma\mathbf1_{\{\epsilon_t<0\}})\epsilon_t^2.
\]

GARCH is the gamma=0 case. Valid coefficients require omega>0, beta>=0,
alpha>=0 and alpha+gamma>=0; gamma itself need not be positive. Only one-lag
models are supported. No code pretends to simulate arbitrary p,q orders with
one stored lag. The first scenario update uses the final **observed** residual
and variance, then uses the simulated states at later steps.

## The empirical innovation distribution

Let u be the filtered residual divided by its fitted conditional standard
deviation. The bootstrap pool is z=(u-mean(u))/std(u), with the population
standard deviation (ddof=0). Its empirical mean is zero and empirical second
moment is one. Both corrections are stored in `model.json`. This is a two-stage
FHS estimator, not an assertion that a Gaussian or Student-t parametric fit is
being reproduced exactly.

The actual second-moment persistence is beta+alpha+gamma*m_minus, where m_minus
is the empirical mean of z² for negative z, with zero contribution for other
observations. It is **not** gamma/2 unless the relevant distribution is symmetric.
The default admissibility threshold is 0.9999, configurable below one through
`ModelConfig`. Rejected coefficients are not clipped. Innovations are drawn iid
over simulated times as complete rows across assets. This preserves the
empirical contemporaneous dependence and does not impose a separate parametric
tail dependence on independent input.

This model has finite innovation support and finite price moments at any fixed
finite horizon in exact arithmetic. It does not extrapolate new standardized
crashes. Numerical paths outside the representable price range fail explicitly;
they are not winsorized into plausible-looking prices. Conditional on the
chosen mean, variability also affects arithmetic expected growth: zero **log**
drift is not a claim of zero expected arithmetic return.

## OHLC semantics

OHLC is available only when observed OHLC is supplied with the same adjustment
as close. Each training row contains, for every asset, its standardized close
innovation, overnight log gap divided by fitted sigma, upper log excursion
above max(open,close) divided by sigma, and lower log excursion below
min(open,close) divided by sigma. All components use the same resampled date.

Open is previous close times exp(sigma*gap_template). High and low are built
above/below the new open and close by exponential excursions. Bounds and
positivity follow from the construction; very small negative excursions due
solely to the input validation's floating-point tolerance are treated as zero.
Changing price units multiplies every output price consistently, including
the wicks. The log gap plus the open-to-close log body equals the
close-to-close log return exactly up to floating-point rounding.

Centering the close innovation or changing its mean changes the body relative
to the original historical day. Consequently the model does not promise exact
preservation of raw intraday body distributions or overnight/intraday means.
No chronology between high and low is provided. No execution model, future
corporate-action process, or simulated volume is supplied.

## Input decisions

Inputs must be positive finite daily prices with unique increasing dates,
unique named assets, compatible sessions and a consistent price basis. CSV
dates must use `YYYY-MM-DD`; API indexes must be timezone-naive midnight
session labels. Time-zone-aware or intraday timestamps are rejected. The caller
must map such observations to their correct local trading-session dates; the
reader does not silently drop time zones or truncate times. The Yahoo daily
adapter deliberately takes the provider's local daily labels. No
forward-fill, outlier clipping, universe substitution, automatic index-member
selection, or synthetic replacement of download errors is performed. An
explicit intersection may remove incomplete edges; an internal missing date
is rejected because bridging it would change the observation horizon.

Dates absent in every series cannot be detected without an external calendar.
The user must verify consecutive common sessions and synchronous economic
timing. Close-only input produces close-only output. Yahoo is explicitly
auto-adjusted on all OHLC fields; these describe adjusted prices, not a tape of
executable historical quotes or a point-in-time fundamentals database.

## Evaluation decisions

The inclusive split date determines training. The test period begins with the
next observation, including its return from the last training close. At each
test date the model predicts before observing the realization, then updates the
state from that realization. It never refits on future observations. Parameters
and the empirical pool remain fixed. Separate training, validation and final
test intervals are required if the user tunes model choices against results.

The baseline uses the same joint training return rows, removes their sample
means, and adds the model's mean vector. It has no conditional volatility
filter. Exact empirical CRPS is evaluated by sorted-sample identities. Energy
score uses a local, reproducible iid pair estimator of the second expectation,
with the same pairs for model and baseline. Variogram score uses power 0.5 and
equal average weights over all distinct asset pairs. Scores use decimal log
returns, not separately normalized test data. There is no variogram statistic
for one asset, reported as null rather than a fabricated zero.

Portfolio forecasts transform each component using expm1 before applying
long-only weights that sum to one. Quantiles use the inverse empirical CDF.
Expected shortfall integrates exactly the required fraction of empirical lower
tail probability, including fractional mass at the boundary. Exceedances use
strictly less than the quantile, so discrete ties are reported separately from
nominal coverage. Joint exceedances are measured on the joint rows; multiplying
marginal probabilities would incorrectly assume independence.

Coverage and serial diagnostics are descriptive. The report includes ratios
of observed exceedances to both nominal and empirical-strict expectations.
A descriptive undercoverage flag is raised when the nominal expected count
is at least ten and the observed-to-nominal ratio is at least two. This is a
transparent screening heuristic, not a significance test, a calibration
certificate, or a trigger that silently changes the forecasts. An absent flag
does not establish calibration. Small rare-tail counts cannot certify
rare-event risk. The benchmark validates sequential one-step **close
return distributions**, not multi-year forecast accuracy, OHLC calibration or
intraday strategy performance. Residual and squared-residual autocorrelations
are included in model diagnostics to expose dynamics the iid bootstrap omits.

## Reproducibility and publication scope

RNGs are local and explicit. Running diagnostics or requesting OHLC cannot alter
close scenarios from an otherwise identical call. Input arrays are copied at
fit time. Runtime versions, seed streams, input hashes and output checksums are
recorded. Numerical reproducibility is scoped to the recorded dependency stack;
different optimizers/BLAS/library versions may change the last digits of fits.

The package publishes tested software and bounded empirical evidence. Passing
tests is not proof of universal model correctness, superior investment returns,
or regulatory fitness. The research review includes modern neural alternatives,
but no unbenchmarked neural implementation is presented as a validated advance.
