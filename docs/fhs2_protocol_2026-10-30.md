# FHS2: locked exploratory factorial protocol

Information cutoff: 2026-10-30. Base commit: 4230097d819d403e9a8a9451077da3002f1a74a8. The prior 2015-2026 SPY results are already known. This is protocol-controlled retrospective research, NOT an untouched prospective test. No automated default promotion is authorized by this study alone.

## Frozen reference and provenance

Preserve the original SPY CSV SHA256 264ff7216c975000c8ffb8be3fd45a79d37d9642d22c05a1b63f68742e0428c6, old outputs and all 198 baseline tests. Do not alter the original model, path evaluator or historical reports. New files implement research extensions. Check installed baseline source against the archive. Record exact data, scientific-code and configuration hashes. Separate a fresh provider snapshot from the original snapshot. Raw prices are not committed to GitHub.

## Primary experiment

Refit from 1,260 prior returns, origins every 21 sessions from the first eligible date on/after 2000-01-01. Simulate 2,000 daily paths, 252 steps, seed 20261030. Evaluate 21, 63, 126 and 252 sessions ONLY where the respective outcome has matured. This preserves the old origin grid for six-month comparisons while not inventing incomplete annual outcomes. Historical results before 2015 are diagnostic; the principal table uses origins from 2015 onward. All outcomes are adjusted daily close returns, not executable quotes or intraday extrema.

A 2 x 2 x 2 factorial crosses:

1. Volatility: unchanged zero-mean FHS GARCH/GJR BIC selection versus HARCH(1,5,22), fitted by Gaussian quasi-likelihood using arch. HARCH uses daily squared-return averages, NOT intraday HAR-RV or Component-GARCH. Positivity, optimizer convergence and persistence below 0.9999 are required. No parameter clipping or hidden replacement of failed models.
2. Drift: zero versus the expanding observed log-return mean shrunk to zero with 5,040 pseudo-observations (20 trading years). Formula mu=sum(past log returns)/(n+5040). The risk filter is identical for both drift variants; drift is added to the risk paths and does not contaminate the innovation state. This is an explicit mean sensitivity, not an estimate of the excess premium or a promise of alpha. Negative estimated drift is retained.
3. Innovation law: global centered/unit-variance empirical residuals versus a causal four-state pool (prior 21/252-day squared-return ratio above one and sign of the prior return). Each state's local sample is shrunk toward the global empirical distribution with local weight n_state/(n_state+500). Weighted mixture residuals are centered and standardized per state. Conditional state transitions are recomputed from simulated shocks, never realized future data. Thus this factor tests innovation shape/temporal dependence, not an unreported extra conditional mean/variance. Record local sample sizes, effective sample sizes and weighted negative second moments; GJR stability must hold under every state's innovation law.

Use common uniforms across factorial variants, identical risk paths within each drift pair, and a fixed full simulation horizon so metric prefixes are coherent. This run need not reproduce the earlier Monte Carlo draws bitwise; the earlier study remains frozen and new comparators share this run's baseline. Exact controlled one-step/recursion equivalence is tested separately.

## Prequential model selection, not hindsight selection

A ninth research method selects one of the eight fixed candidates using the last 60 *fully matured* 126-session CRPS observations, with at least 24 common origins. Before enough feedback exists, select FHS/zero/global. Ties favor the simpler order. No future outcomes enter available scores. Evaluate the selector as issued; do not retroactively choose the globally best model. This is delayed rolling inner selection over fixed models, not a claim of full hyperparameter nested optimization.

## Evaluation and uncertainty

Primary: maximum-drawdown CRPS at 126 sessions. Guardrails: terminal-return CRPS, realized RMS-volatility CRPS, full daily-path kernel score, drawdown threshold CRPS above 10%, recovery/first-passage integrated Brier scores and separate lower/upper interval violations. Report 5/10/20% drawdown barriers and fixed-origin loss separately. Retain previous definitions including origin peak and censored recovery.

Conditional diagnostics use only issue-time information: prior 252-day annual RMS bins <15%, 15-25%, >=25%; prior21/prior252 ratio; sign of past63-day return; known prior drawdown >=10%. Fixed cutoffs are not refitted to the evaluation sample. These groups are descriptive, dependent and potentially sparse.

Paired stationary bootstrap over ordered forecast origins, 2,000 resamples; main mean block length 2*ceil(H/21), sensitivity ceil(H/21) and 4*ceil(H/21). Use common resamples and approximate simultaneous studentized-max intervals across candidate comparisons for primary drawdown CRPS. These are assumption-dependent approximate intervals, not universal finite-sample guarantees. Also report non-overlapping phases and episode clusters. Never count simulated paths as real-world observations. Independent simulation-seed sensitivity is separate from sampling/model uncertainty.

Numeric screening for a candidate: >=5% improvement in primary mean CRPS, simultaneous upper confidence bound below zero, and no >2% deterioration in each terminal/volatility/path-score guardrail. Report 2015-2019, 2020-2022 and 2023-cutoff separately. The original production/default FHS remains unchanged even if this retrospective screen passes; external and prospective evidence is still required.

## External information and economic diagnostic

Attempt EWJ, EEM and VGK adjusted daily snapshots through 2026-10-29. These US-listed foreign-equity ETFs are transfer checks, not independent replications of global crises. Do not concatenate their histories or count one worldwide episode multiple times. Record failed acquisition explicitly. Attempt VIX, VIX3M and IRX histories only as future research inputs; no option-implied measure is silently used as a physical probability, and no revised series is called point-in-time.

Evaluate a declared alarm for drawdown>=10% with false-positive cost 1 and false-negative cost 5, so alert threshold is 1/6. Costs are abstract decision-loss units, NOT trading P&L or measured execution costs. Compare always-alert, never-alert and candidate decisions on common origins; report component counts. A trading strategy/alpha claim is outside this experiment.

## Additional scoped components

Implement explicit stress paths without probability labels; non-overwriting forecast ledger with issuance-time and data-hash provenance; prior/seed and block-bootstrap drift-uncertainty sensitivity. Recalibration of event probabilities or quantiles must be a separately labeled delayed-feedback comparator, not evidence that the joint generator is corrected. No HMM/GAS/rough/Hawkes/EVT/neural method is claimed implemented without executable code and tests. Review their mechanisms, assumptions and falsification experiments; document remaining data/computation requirements.

## Adversarial acceptance

Independent scalar references for HARCH recursion, empirical mixture moments, drawdowns, CRPS, barrier events and decision losses. Attack first-day losses, order permutations, exact barrier equality, scale invariance, no-future mutation, immature labels, unsupported states, failed optimizers, sparse tails, incomplete targets, duplicate records and source/input mutation during a run. Run all existing tests. Verify a clean wheel install, offline demo, exact remote source, PR CI and final main head. The referee is numerical code authored in this workflow, not an independent human or second AI unless such a reviewer is actually available. Publish limitations and a status/evidence ledger for all 23 requested tasks. No false declaration that every possible defect or model risk has been eliminated.
