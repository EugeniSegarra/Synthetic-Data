# Adversarial release review

**Review date: 30 October 2026.** The release was challenged by a separate
automated reviewer agent, with implementation work and review performed in
successive rounds. This is an engineering review record, not independent human
peer review, a formal proof, or certification of a financial risk model.

## Acceptance scope

The target is an installable, reproducible **daily FHS scenario and diagnostic
engine** with explicit input contracts. Acceptance requires the numerical and
integration tests to pass, no unresolved blocking findings in that scope, and
the release's CI checks to pass before merging. The exact commit and check
results are recorded in the release pull request and
[GitHub Actions](https://github.com/EugeniSegarra/Synthetic-Data/actions).

Acceptance does not mean that nominal VaR probabilities are calibrated in all
regimes. The real-market benchmark has a material 2022 failure: **31 portfolio
breaches in 251 sessions at the nominal 5% tail**. The
[validation report](validation_report.md) and machine-readable results retain
that failure. It was not removed through retrospective parameter tuning.

## Challenges and resulting changes

The reviewer did not approve the first implementation. The initial challenge
round exposed internal missing-session handling and weaknesses in the
independence evidence. A second round tightened coefficient positivity and
required unfavorable tail coverage to be visible alongside favorable average
scores. A third round reproduced the time-zone-abbreviation parsing bypass
after a first daylight-saving-time fix; a strict ISO date whitelist closed it.
Each correction was followed by targeted regression checks and a final suite
run before creating the release pull request.

| Challenge | Result and release decision |
| --- | --- |
| Could a high-order volatility fit be simulated with an incorrect one-lag recurrence? | Support only explicit GARCH(1,1) and GJR-GARCH(1,1), with hand-computed recurrence and first-step tests. |
| Could heavy marginal tails manufacture extreme cross-asset dependence? | Replace heuristic t-copula parameter inference with joint empirical row resampling. Test independent heavy tails and exact Cartesian-product independence through the actual simulator, plus perfect dependence. |
| Does negative GJR asymmetry always mean an invalid model? | No. Admit negative gamma when alpha+gamma is nonnegative. Test the positive and negative shock recurrences independently. |
| Could an approximately valid coefficient break positivity for an extreme shock? | Reject alpha+gamma below zero, including a negative value of order 1e-13. Do not clip fitted coefficients. |
| Can the usual gamma/2 persistence check fail for empirical shocks? | Check the actual empirical negative-shock second moment. An asymmetric counterexample distinguishes this from the symmetric shortcut. |
| Are generated OHLC bars dimensionally coherent? | Use observed, same-row, volatility-normalized log gaps and excursions. Test price-unit changes, inequalities, joint row provenance, and the gap/body accounting identity. |
| Can data preparation silently turn two sessions into one modeled day? | Reject missing internal sessions. An explicitly requested intersection may trim incomplete edges only. Dates omitted by every asset require an external calendar and remain a disclosed input responsibility. |
| Can dates silently lose time zones or fail a CSV round trip across daylight-saving time? | Require ISO `YYYY-MM-DD` CSV dates and timezone-naive midnight API session labels; reject time-zone abbreviations, offsets and intraday timestamps instead of silently normalizing them. |
| Can a validation statistic use future observations? | Fit parameters and the innovation pool using training data only. Predict first, score the new observation, then update the volatility state. Test the training boundary and invariance of the fitted training model to changes in the holdout. |
| Can discrete empirical quantiles and ES conceal ties or incorrect tail mass? | Test inverse-CDF quantiles, fractional tail mass in expected shortfall, strict exceedance probabilities, and joint rather than independently multiplied tail probabilities. |
| Can a favorable average score conceal bad portfolio tail coverage? | Display observed/expected counts, ratios and 5%/1% frequencies for both methods. Add explicitly descriptive undercoverage flags. Preserve the 2022 failure in the README, report, chart and CLI summary. |
| Can diagnostics or optional OHLC change the close paths? | Use explicit local RNGs and separate CLI streams; test global RNG isolation, repeatability and identical close paths with or without OHLC. |
| Can a failed run overwrite an existing result or publish a partial bundle? | Stage artifacts and use an exclusive directory commit; test failure cleanup and a destination-creation race. Existing output directories are rejected. |
| Can a source checkout mask broken packaging? | Build a wheel, install it into a separate environment, and execute the offline example outside the checkout. CI repeats this gate on each configured platform. |

## Verification layers

The executable evidence is in
[test_model.py](../tests/test_model.py),
[test_adversarial.py](../tests/test_adversarial.py),
[test_data.py](../tests/test_data.py),
[test_validation.py](../tests/test_validation.py), and
[test_cli.py](../tests/test_cli.py). Tests use independent manual calculations,
known synthetic processes, invalid-input counterexamples, and complete command
invocations. They do not rely solely on generating data and checking that the
same implementation can read its own output.

The CI matrix runs Python 3.11, 3.12 and 3.13 on Ubuntu and Python 3.12 on macOS.
It checks Ruff lint/format, the numerical and adversarial tests, an installed
offline CLI example, distribution builds, and an isolated wheel installation.
CI uses dependency ranges from `pyproject.toml`; the recorded market benchmark
also supplies the exact tested constraints and runtime versions. Read the
actual check result for the commit being used, rather than treating this
description of the workflow as a test result.

A separate real-data integration run exercised adjusted Yahoo OHLC for
SPY/TLT/GLD, path generation, chronological evaluation, NPZ loading with
`allow_pickle=False`, and artifact checksums. That verifies the data and
execution path. Statistical adequacy is assessed separately in the published
benchmark.

## Remaining model risks are explicit

The empirical innovation law has finite historical support and fixed residual
dependence. It cannot infer an unseen joint crisis or automatically adapt a
correlation regime. A successful 5% coverage count in one window does not
validate the whole multivariate distribution, and small 1% event counts cannot
establish rare-tail accuracy. Parameter uncertainty is not represented.

The predictive experiment is sequential and one step ahead. It does not
validate a 63-day or multi-year path distribution, intraday extrema, stop/target
execution, transaction costs, future corporate actions or trading alpha. Yahoo
data are a revisable research source, not an institutional point-in-time feed.
These boundaries are part of the accepted software contract, not issues hidden
by a passing test suite.
