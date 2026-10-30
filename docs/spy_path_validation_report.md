# SPY path validation: evidence, not visual plausibility

Study cutoff: 30 October 2026. The result is a working full-path evaluation layer, not a certification that any generator is the true SPY process. The existing FHS default is unchanged. Experimental analogue and mixture models are retained as explicit comparators because none establishes a six-month improvement over FHS in this experiment.

## Design and provenance

The [protocol](path_protocol_2026-10-30.md) was recorded in commit `be1b8270609d2d44f2476c9089538d436c736a75` before inspecting this run's benchmark outcomes. This is retrospective protocol control, not prospective preregistration: the market history was already known. No outcome-driven parameter retuning was performed.

The source snapshot contains 8,496 adjusted daily SPY observations from 29 January 1993 through 29 October 2026. An independent XNYS session-calendar comparison using exchange_calendars 4.13.1 found 8,496 expected dates, zero missing and zero extra dates. This calendar check does not certify the accuracy of every provider price. Yahoo histories are revisable, so the experiment is pseudo-out-of-sample, not immutable point-in-time evidence.

Input CSV SHA-256: `264ff7216c975000c8ffb8be3fd45a79d37d9642d22c05a1b63f68742e0428c6`.

Input capture: [workflow run 41033349796](https://github.com/EugeniSegarra/Synthetic-Data/actions/runs/41033349796), artifact `path-research-input`. Preserve the downloaded snapshot: workflow artifacts expire. Raw provider prices are not committed to the source repository. A fresh provider download is a new snapshot, not automatically a bit-identical reproduction.

The evaluation refits on the preceding 1,260 returns at each origin. It issues 2,000 full daily paths for 126 future sessions and evaluates prefixes of 21, 63 and 126 sessions. Origins are spaced 21 sessions apart. All models use zero expected log drift, which is not zero arithmetic expected return and is not an estimated equity premium.

There are 316 complete origins from 3 January 2000 through 23 April 2026, with no rejected FHS origin in this run. The main evaluation contains 136 origins from 13 January 2015 through 23 April 2026; its last 126-session target ends on 22 October 2026. The common origin grid deliberately excludes immature targets at the end of the snapshot. The earlier 180 origins are a separate diagnostic period and also provide matured feedback to the adaptive mixture.

Each trajectory is open-loop. Daily observations arriving after issuance do not enter that trajectory. The mixture updates only from forecasts whose full 126-session targets have matured. There are 5,688 origin/horizon/method records, not 5,688 independent market episodes.

## What is measured

Forward maximum drawdown includes the origin price as the initial peak. It is distinct from a loss relative to the fixed origin, and from recovery of a peak already known before issuance. Outcomes also include terminal simple return, close-to-close RMS volatility, the fraction of sessions below a forward peak, the longest observed underwater run and mean drawdown depth. Intraday extrema are not observed by these metrics.

Distribution scores include empirical CRPS, drawdown CRPS censored below 10%, a full-daily-path kernel score, and first-passage Brier scores integrated over all future sessions. Recovery is scored only for origins already below their known historical peak; unrecovered paths remain censored. A restricted mean recovery time is bounded at the forecast horizon and is never described as an unrestricted expected recovery time.

The full implementation, interdisciplinary sources and distinctions from the original papers are described in [path_research.md](path_research.md).

## FHS calibration: one property does not validate the others

Coverage of the nominal central 90% interval, main sample, denominator 136 for every cell:

| Property | 21 sessions | 63 sessions | 126 sessions |
|---|---:|---:|---:|
| Terminal return | 92.65% | 92.65% | 92.65% |
| Close-to-close RMS volatility | 89.71% | 83.82% | 83.82% |
| Forward maximum drawdown | 86.03% | 85.29% | 83.09% |

The six-month drawdown coverage has an approximate 95% stationary-bootstrap interval of **70.59% to 93.38%**, using the predeclared main block length of 12 forecast origins. The nominal 90% is inside that wide interval. This is neither decisive rejection at that level nor evidence that calibration has been established. The interval is pointwise, assumption-dependent and not a universal nonstationary-market guarantee.

The six-month undercoverage is asymmetric. Seventeen of 136 realized drawdowns are BELOW the forecast fifth percentile (12.50% rather than nominal 5%); six are ABOVE the 95th percentile (4.41% rather than nominal 5%). The central coverage miss therefore cannot honestly be summarized as simply underestimating crash risk. This version often allocates too little probability to very mild drawdowns, while its crisis evidence remains sparse.

For a forward drawdown of at least 10% within 126 sessions, the mean predicted probability is 47.28%, versus an observed event frequency of 35.29%. These unconditional averages are descriptive calibration diagnostics, not a test of every state-conditional probability. Among 115 already-underwater main-sample origins, the mean predicted probability of recovering the known peak within 126 sessions is 68.27%; 82.61% actually recover. Zero log drift is a material horizon assumption and may contribute, but this experiment does not isolate it as the sole cause.

For reference, FHS maximum-drawdown central coverage in the earlier 2000-2014 diagnostic period is 87.78%, 91.11% and 83.89% for 21, 63 and 126 sessions. Its six-month weakness is not hidden by omitting that earlier period.

## Six different issue dates can contain one crisis

All six FHS six-month upper-95% drawdown breaches in the main sample share a realized peak on **19 February 2020**. Their issue dates are 16 September, 15 October, 13 November and 13 December 2019, and 15 January and 14 February 2020. Five windows include the full 33.72% adjusted closing drawdown; the earliest ends before the final trough and contains a 29.11% drawdown.

These are six forecast failures, but not six independent crises. Treating them as independent observations would exaggerate the amount of tail evidence. The per-origin CSV records the realized peak and trough dates so this fact is inspectable.

At 126 sessions and a 21-session origin spacing, the six non-overlapping phase views contain only 23, 23, 23, 23, 22 and 22 windows. Their drawdown coverages range from 81.82% to 86.36%. The phases share history and are not independent replications. Simulating more paths cannot expand this real-history sample.

## Challenger results: more complicated is not better here

Six-month main-sample maximum-drawdown CRPS; lower is better. CRPS has drawdown-fraction units; a relative change is not a change in investment return.

| Generator | Mean drawdown CRPS | Change vs FHS | Central 90% drawdown coverage |
|---|---:|---:|---:|
| FHS | 0.038753 | reference | 83.09% |
| iid centered historical returns | 0.042178 | +8.84% worse | 65.44% |
| Stationary blocks, mean length 10 | 0.039495 | +1.92% worse | 74.26% |
| Conditional analogues | 0.054215 | +39.90% worse | 60.29% |
| Analogues with time/variation area | 0.052124 | +34.50% worse | 58.82% |
| Delayed adaptive whole-path mixture | 0.039335 | +1.50% worse | 80.88% |

Paired differences in drawdown CRPS relative to FHS, with approximate pointwise 95% stationary-bootstrap intervals (main block 12): iid +0.003425 [-0.000332, 0.007490]; blocks +0.000743 [-0.001498, 0.002910]; analogues +0.015462 [0.005282, 0.029264]; area analogues +0.013371 [0.004811, 0.024186]; mixture +0.000582 [-0.000969, 0.002285]. The comparison does not establish a material six-month gain for blocks or the mixture. The adverse analogue result is stronger, but these are not multiplicity-adjusted discovery claims.

The planned area ablation compares area analogues directly with plain analogues: difference -0.002091, interval [-0.005554, 0.000294]. The feature modestly helps this particular challenger on average, but the interval includes zero and the resulting model remains worse than FHS. This is not a rejection of every signature method or every meteorological analogue approach.

The analogue's 100 nearest states typically contain only **four disjoint six-month donor continuations** (median; minimum one). Disjoint is not the same as independent. This exposes sparse effective episode diversity that a nearly uniform 100-neighbor weight count can conceal. It is a plausible limitation, not an experimentally isolated explanation for all of the performance gap.

Full-path kernel scores at 126 sessions tell a consistent descriptive story: FHS 0.177681, iid 0.193839, blocks 0.188450, plain analogues 0.206806, area analogues 0.206078 and mixture 0.182598. At 21 sessions the mixture's kernel score is marginally lower than FHS; that isolated small advantage is not used to promote a new default.

## Numerical and adversarial review

The complete local test suite passes **198 cases**, including 59 new path-validation cases. The separate executable referee independently recomputes 948 realized path targets directly from prices using scalar reference algorithms, checks the 5,688 forecast records, maturity dates, rolling windows, donor eligibility, mixture-weight constraints, quantile/coverage identities and artifact checksums.

The referee is an executable numerical gate authored in the same workflow, NOT an independent human or separately commissioned language-model reviewer. Its approval is restricted to implementation/evidence consistency within checked scope, not market calibration, all possible bugs, production VaR or regulatory model approval.

Adversarial corrections included exact barrier equality tolerances, zero-hit Monte Carlo intervals, incomplete recovery censoring, explicit data-price convention, rejecting duplicate/missing summary records, refusing publication when input or source changes during a run, and distinguishing expected calibration failures from unexpected programming exceptions. An injected future-price mutation leaves earlier issued forecasts and immature mixture weights unchanged. An injected fit failure remains visible and is never silently replaced by another model.

The final replay agrees exactly with the first benchmark on every previously recorded numerical forecast/score column (maximum absolute difference zero). Later changes added diagnostics, provenance and edge-case guards; they did not retune methods to improve this history.

Local scientific implementation SHA-256: `0990df6a03d1c438239f18eeda46c2c7df3e3313c46ac2835b084d4814911f42`.
Local results manifest SHA-256: `745ec6994aac2a40a9ed5bcad83ff3fe62eee49e6fd971f0f82bc22a13eab625`.
Local path forecasts CSV SHA-256: `c804c64a5792411d6d843579b9043c0ae7daa37d98009e1816b3acd74505b037`.
Local path summary JSON SHA-256: `9f0c75b0d3c5159bffbec2587a02b87cbf8847aac461ab4d3b8cc832cba4f8c1`.
Recorded core versions: numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, arch 8.0.0. Exact floating-point calibrations may vary across numerical environments. The workflow retains its own environment and manifest rather than pretending cross-platform bit identity.

## Reproduction

From a fresh checkout and Python 3.11-3.13:

```bash
python -m pip install -e '.[yahoo,dev]'
synthetic-market download --tickers SPY --start 1993-01-01 --end 2026-10-30 --out outputs/spy-input
python -m synthetic_market.path_validation \
  --csv outputs/spy-input/input.csv \
  --metadata outputs/spy-input/input_metadata.json \
  --price-basis adjusted --out outputs/spy-paths
python examples/audit_path_study.py \
  --results outputs/spy-paths --input outputs/spy-input/input.csv \
  --out outputs/path-referee.json
python -m pytest -q
```

Use the preserved input artifact instead of a new download to reproduce the recorded snapshot. Existing output directories/files are rejected. The normal test suite is offline. The separate `Reproducible SPY path research` workflow uses the preserved artifact and verifies its hash before running the complete experiment and referee. If that input artifact expires, it fails rather than silently switching to revised data. Retrieve its `spy-path-evidence` artifact for the full CSV, audit, summaries and environment; artifacts have explicit retention limits.

The default evaluator is SPY-focused and univariate. Other assets require their own input quality checks and validation, not automatic transfer of these empirical results. Daily adjusted-path diagnostics are not an execution backtest and should not be represented as one.
