# FHS2: factorial results, external evidence and economic limits

Information cutoff: 30 October 2026. This release adds a working research layer, not a new default production model. **No candidate passes the predeclared six-month promotion screen in any of the four markets.** The original FHS engine and its reports remain unchanged.

## What was tested

The [protocol](fhs2_protocol_2026-10-30.md) was committed as `72fdf8615751f59a10428c8d891bbbec16c15bf8` before inspecting the new experiment. The earlier SPY outcomes were already known. This is retrospective, protocol-controlled research using revisable provider snapshots, not an untouched or genuinely point-in-time test.

Eight fixed candidates cross three binary changes: the original GARCH/GJR FHS filter versus daily HARCH(1,5,22); global versus causal state-conditioned empirical innovations; and zero log drift versus an expanding historical log mean shrunk toward zero with 5,040 pseudo-observations. The risk path is exactly the same within each drift pair. State pools retain exact weighted mean zero and variance one, so the state factor changes shape/order rather than an unreported location or scale. All models use a 1,260-return training window. There are 2,000 simulated daily paths per origin and a coherent 252-session simulation, evaluated at 21, 63, 126 and 252 sessions only when the respective real outcome has matured.

A ninth method selects among the fixed candidates using the last 60 fully matured six-month CRPS observations, with a minimum of 24. This is delayed rolling inner selection, not full nested hyperparameter optimization. The separate isotonic comparator recalibrates one event probability; it does not change price paths.

The SPY original snapshot has 8,496 observations from 29 January 1993 through 29 October 2026 and retains SHA256 `264ff7216c975000c8ffb8be3fd45a79d37d9642d22c05a1b63f68742e0428c6`. External checks use EWJ (7,705 observations), EEM (5,925) and VGK (5,445), each ending 29 October 2026. A separate XNYS calendar check found zero missing or extra session dates in each input. It does not certify every price, adjustment or historical availability timestamp. The three external ETFs are USD-listed foreign-equity exposures; global crises shared across them are not independent replications.

The experiment contains 1,048 market-origin records, 4,120 realized origin/horizon targets and 36,828 candidate forecast records. These are not independent market episodes. Main comparisons use origins from 2015 onward. Common-case six-month sample sizes are 134 for SPY, 135 for EWJ and 136 for each of EEM and VGK. Aggregates may contain more observations for models that remain available when another candidate fails. Tables and CSVs make the denominator explicit.

## Primary six-month result

CRPS is an error score for a predicted distribution; smaller is better. Changes below are score changes, not investment returns. All SPY rows in this table use the same 134 origins and a drawdown expressed as a fraction.

| Candidate | Maximum-drawdown CRPS | Change vs FHS | Terminal-return CRPS change | Central 90% MDD coverage |
|---|---:|---:|---:|---:|
| FHS / global / zero | 0.039321 | reference | reference | 84.33% |
| FHS / global / shrunk drift | 0.038465 | -2.18% | -11.70% | 86.57% |
| FHS / state / zero | 0.039145 | -0.45% | +0.01% | 84.33% |
| FHS / state / shrunk drift | 0.038320 | -2.55% | -11.71% | 85.82% |
| HARCH / global / zero | 0.039148 | -0.44% | +4.89% | 78.36% |
| HARCH / global / shrunk drift | 0.038349 | -2.47% | -8.92% | 81.34% |
| HARCH / state / zero | 0.038900 | -1.07% | +4.67% | 79.10% |
| HARCH / state / shrunk drift | 0.038125 | -3.04% | -9.17% | 81.34% |
| Delayed selector | 0.039195 | -0.32% | -6.44% | 82.84% |

The principal screen requires at least a 5% mean drawdown-CRPS reduction, an approximate simultaneous upper confidence bound below zero, and at most 2% deterioration in each terminal, volatility and full-path score guardrail. No candidate qualifies. The original default also would not be replaced automatically by a retrospective screen alone.

For FHS with shrunk drift, the paired MDD-CRPS difference has a simultaneous approximate 95% interval of **[-0.002434, +0.000722]**. For HARCH/state/shrunk drift it is **[-0.004565, +0.002171]**. The intervals use a stationary bootstrap with main mean block length 12 forecast origins, common draws and a maximum-standardized comparison statistic; lengths 6 and 24 are also reported. These intervals are dependent-sample approximations, not guarantees under arbitrary nonstationarity. Common-case inference conditions on admissible calibrations; skipped dates create gaps and can make this support selective. Failures are not removed from the evidence ledger.

The original study's 83.09% six-month coverage must not be compared to the new 84.56% full-136-origin baseline as a model improvement. The law of the reference filter is retained, but the new run uses a different common-uniform coupling and a 252-step ensemble. The old output remains frozen. Within-run candidate comparisons share the same controls and, for drift, exactly the same shocks.

## What the drift experiment establishes, and what it does not

On the 136 available FHS/global six-month origins, adding the declared shrunk mean reduces mean predicted probability of a 10% drawdown from **47.33% to 42.77%**, versus an observed **35.29%**. Among the 115 origins already below a known peak, predicted recovery rises from **68.07% to 73.55%**, versus **82.61%** observed. The fraction finishing above the forecast median falls from **73.53% to 69.85%**, still far from the 50% benchmark.

Thus the mean assumption explains part of the directional discrepancy but does not repair the full path distribution. At 252 sessions, on 128 common origins, FHS/shrunk drift improves terminal CRPS by 19.31%, but drawdown CRPS by only 2.56%. This is not an estimated equity premium, predictive alpha or permission to use the realized future trend as a conditioning variable.

The effect is not stable across epochs. At six months, on common support, FHS/shrunk drift changes MDD CRPS by -1.21% in 2015-2019, **+0.76% worse in 2020-2022**, and -7.35% in 2023-cutoff. HARCH/state/shrunk drift gives -0.21%, **+2.33% worse**, and -14.51%, respectively. A good full-period average would conceal this concentration in recent history.

## Conditional calibration remains a material limitation

These are fixed, issue-time bins of annualized RMS over the previous 252 returns, not ex-post equal-sized terciles. The counts are overlapping forecast origins, not independent regimes.

| Initial annual RMS | Origins | FHS predicted P(MDD >=10%) | FHS + drift | Realized frequency |
|---|---:|---:|---:|---:|
| Below 15% | 69 | 44.64% | 40.04% | 53.62% |
| 15%-25% | 55 | 50.75% | 46.31% | 18.18% |
| At least 25% | 12 | 47.13% | 42.28% | 8.33% |

Lowering predicted downside risk improves some unconditional discrepancies but may worsen forecasts issued from apparently calm markets. The high-volatility group has only 12 overlapping observations. This is a descriptive diagnostic and a research hypothesis, not evidence that high volatility causally prevents subsequent losses. The HARCH alternative does not consistently resolve this conditional mismatch.

## External transfer does not confirm a robust gain

Relative six-month MDD-CRPS changes on common support:

| Candidate | SPY | EWJ | EEM | VGK |
|---|---:|---:|---:|---:|
| FHS + shrunk drift | -2.18% | -0.17% | -0.21% | -0.73% |
| HARCH + state + drift | -3.04% | +2.33% | +1.00% | -0.35% |
| Delayed selector | -0.32% | +0.32% | +3.04% | +0.31% |

These checks neither reject every multiscale/state model nor justify promoting this one. They show that the particular changes tested do not establish a material, robust improvement over FHS.

## Event recalibration and economic decisions

The matured-only, shrunk isotonic event comparator lowers SPY's mean predicted 10% drawdown probability from 47.33% to 43.85%. Yet its Brier error **worsens from 0.265867 to 0.268952**. Bringing an unconditional mean closer to the realized frequency does not establish more informative conditional forecasts. EWJ and VGK improve slightly on this score while EEM worsens; no general recalibration advantage is claimed.

The economic experiment deliberately uses declared abstract costs: false alarm 1, missed drawdown 5, giving alert threshold 1/6. At six months, every available candidate issues an alarm at every origin. On the 136 SPY baseline origins this means **136 alarms, 88 false alarms and zero missed events**, exactly the always-alert policy, with average loss 88/136 = 0.6471. Never alert has loss 240/136 = 1.7647. The loss function does not demonstrate model-specific economic value. The threshold was not retuned after seeing this result. No trades, transaction costs, executable fills, financing returns or fund P&L are fabricated.

## Calibration failures and uncertainty

A state-conditioned FHS candidate is unavailable when its empirical conditional innovation law violates the declared conservative GJR stability guard; HARCH also rejects inadmissible fits. In the main SPY six-month comparison, 16 April 2020 lacks the state-conditioned FHS variant and 15 September 2022 lacks HARCH, leaving 134 common origins. The full failed-origin CSV includes earlier failures and those in EEM/VGK. Such cases are a deployment constraint, not silently replaced forecasts. Rejection by this sufficient guard does not prove that every rejected nonlinear process would explode.

A separate diagnostic performs 180 block-bootstrap parameter refits across six issue states; **171 are accepted and nine rejected**. Accepted parameters are re-filtered on the actual observed history before simulation. For 29 October 2026, the 5th and 95th percentiles across accepted estimates of the model's own 95th-percentile drawdown are approximately **24.73% and 33.53%**. This measures sensitivity of a tail estimate to calibration, not a 90% market outcome band or a calibrated Bayesian credible interval. Drift-prior and Monte Carlo seed sensitivities are recorded separately. More simulation paths cannot resolve this uncertainty or create new historical crisis types.

## Reproducibility and numerical acceptance

All 198 original tests remain; the complete suite passes **238 tests**. The scalar referee reconciles all 36,828 records and 4,120 realized targets across four experiments. A separate five-origin replay checks 120 CRPS values with explicit pairwise-distance calculations and an independent price-level drawdown implementation. These are independent numerical algorithms authored by the same assistant, not independent human or separately commissioned AI peer review.

The full SPY run was repeated in [GitHub Actions 41141742760](https://github.com/EugeniSegarra/Synthetic-Data/actions/runs/41141742760) at source commit `9794d265aec7c7fd35ffc8746cd60c32e1c00b76`. All 11,214 forecast rows align, and every compared numerical forecast/score column agrees exactly with the local run: maximum absolute difference zero. Package source hashes also agree. This is observed reproducibility for the pinned environment, not a promise of bit identity on every future dependency or platform. Exact final release CI is recorded in [PR #4](https://github.com/EugeniSegarra/Synthetic-Data/pull/4).

Derived tables are published under `reports/fhs2_*.csv`; complete per-origin bundles are retained in the accompanying evidence archive. Workflow artifacts expire. Preserve authorized original snapshots and verify their hashes before replay; a fresh provider download is a different snapshot unless it matches. Raw provider histories are not committed to the repository.

## Run the extension

Install the project normally, then use the original SPY snapshot or an explicitly identified new one:

```bash
python -m pip install -e '.[yahoo,dev]'
python -m synthetic_market.research2 evaluate --csv input.csv --metadata input_metadata.json --price-basis adjusted --out outputs/SPY
python examples/audit_fhs2.py --results outputs/SPY --input input.csv --out outputs/SPY_referee.json
python examples/replay_fhs2.py --csv input.csv --results outputs/SPY --out outputs/score_replay.json
python examples/fhs2_sensitivity.py --csv input.csv --out outputs/sensitivity.json
python -m synthetic_market.research2 issue --csv input.csv --price-basis adjusted --method fhs_global_zero --out outputs/issuance
python -m pytest -q
```

Repeat evaluation separately for each external input; do not combine currencies or different markets silently. With the four result directories under `outputs`, `python examples/collect_fhs2.py --root outputs --out derived` regenerates tables. After installing optional matplotlib, `python examples/render_fhs2.py --results-root outputs --csv input.csv --out figures` creates seven separate evidence figures, including scenarios, terminal distributions, drawdowns, conditional barriers, recovery, volatility and cross-market scores.

The issuance command writes a timestamp and an unscored snapshot. A forecast generated after its data cutoff is not automatically genuinely prospective: confirm which sessions occur after actual issuance before scoring it. No future target results are claimed.

## Boundaries of this release

The [23-task ledger](fhs2_task_ledger.md) distinguishes implementation, investigated alternatives and remaining work. Not delivered as implementations: full Component-GARCH, intraday HAR/Realized GARCH, estimated HMM/GAS, neural/rough/Hawkes/EVT generators, a vintage-aware rate-plus-premium specification, a completed prospective study, full nested hyperparameter selection, a full leave-crisis-out training study, or trading execution/portfolio P&L. Literature coverage does not turn these into implemented features. The data contract rejected VIX and IRX acquisition; those inputs were not filled or used. VIX3M was acquired separately but not used.

The implemented extension is useful for reproducible model comparison and falsification. It has not demonstrated a robust six-month risk-model replacement, a profitable risk policy or universally calibrated tail probabilities. Keeping the original default is the outcome required by the evidence, not an unfinished attempt to force a statistical winner.
