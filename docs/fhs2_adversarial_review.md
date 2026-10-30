# FHS2 adversarial numerical review

Scope: added research layer, its reproducibility and the correctness of reported evidence. The original engine is not replaced. The referee uses separate scalar/explicit-pair algorithms authored in the same workflow. It is not independent external peer review, a proof of all possible code paths or certification of market probabilities.

## Review rounds and concrete checks

The initial 29 new cases covered causality of state assignment, weighted innovation moments, exact HARCH/GJR recursions, library one-step comparisons, drift pairing, horizon-prefix invariance, unit-scale sensitivity, RNG isolation, mature-label selection and future-data mutation. Later rounds added an exhaustive small-case reference for isotonic regression, delayed recalibration, parameter-refit sensitivity, injected failures, output atomicity, input mutation, safe NPZ, invalid metadata and the information cutoff. The final round adds actual-history refiltering and an end-to-end tampering test. The complete count is 40 new cases and 238 tests including the 198 preserved baseline tests.

An independent price-level running-peak algorithm recomputes realized drawdowns, terminal returns, underwater time, RMS variation, fixed-origin losses and recovery. It reconciles 4,120 origin/horizon targets and 36,828 forecast rows. It also reconstructs the selector's matured score panel, verifies origin/training/target dates, checks quantile ordering and coverage identities, recalculates economic alarm losses, verifies summary denominators and checks artifact hashes. The tampering test deliberately changes a realized outcome and updates the file hash, so rejection must come from the independent numerical oracle rather than the checksum alone.

Five deterministic SPY origins are re-simulated with the recorded seeds and parameters. Three score families for all eight candidates produce 120 CRPS comparisons against an explicit pairwise-distance formula. Price-level drawdown calculations are separate from the production log-domain implementation. This is a sampled forecast-score audit, not re-evaluation of every distribution by a wholly independent simulator.

## Issues resolved and limits retained

- Conditional residual pools are centered and scaled using their exact mixture weights. Tests verify the actual state-wise second moment rather than assuming a Student-t or symmetric negative moment.
- Every GJR state law must pass the declared stability guard. Rejected fits remain listed; a missing selected model does not silently fall back to another model. Inference reports common support and separate all-available aggregates.
- A bootstrap calibration is re-filtered on actual history, not simulated from its resampled endpoint. Tests demonstrate that the real last shock changes first-step risk as required.
- Isotonic ties are pooled before fitting. Equal-probability observations do not get arbitrary different calibration values. Feedback labels are unavailable until the entire horizon matures.
- Source/input mutation during a run prevents bundle publication. Existing output directories are rejected. Numeric NPZ loading requires no pickle.
- The initial local build lacked wheel; the dependency was installed from the captured offline package archive. A source-distribution manifest was added so the adversarial examples required by tests are included in the distributable tarball. This fixes packaging, not financial forecasts.
- The original VIX/IRX price-adapter failures remain acquisition failures. No risk-free rate or physical probability is inferred from a failed or unverified index history.
- A timestamped forecast bundle is explicitly unscored. It is not retroactively described as a live forecast issued at the input cutoff.

## Reproduction and release evidence

Local full suite: 238 passed. Lint and formatting pass. Wheel and source distribution build, including the sensitivity/referee examples. The installed wheel's new research issuance command runs outside the checkout; the local wheel check shares numerical dependency installations, which is explicitly weaker than a fully fresh dependency environment. GitHub CI separately installs a clean environment across Ubuntu Python 3.11/3.12/3.13 and macOS Python 3.12, builds distributions and exercises the installed wheel. The final workflow also checks the new research module's installed path and offline issuance bundle.

The full SPY science replay [41141742760](https://github.com/EugeniSegarra/Synthetic-Data/actions/runs/41141742760) passed at `9794d265aec7c7fd35ffc8746cd60c32e1c00b76`. Compared with local evidence, all 11,214 numerical forecast rows agree exactly and package source hashes are identical. All coverage decisions are equal. The remote scalar and sampled-score referees also pass. Exact-head final CI and the merge record are linked from [PR #4](https://github.com/EugeniSegarra/Synthetic-Data/pull/4).

Input SPY SHA256: `264ff7216c975000c8ffb8be3fd45a79d37d9642d22c05a1b63f68742e0428c6`.
Local SPY manifest SHA256: `3a817ba0bd8d479b66bd0bce9bd8f587edbcd36d3886fe11712691d64907146d`.
Remote SPY manifest SHA256: `2e73bfa0b49ec37d4c8a246ab7097d355f47c6991d9351f580fba762ca23fac6`.
Manifests differ because environment and input paths are recorded; forecast values and scientific source agree. No false cross-platform universal bit-reproducibility claim is made.

## Verdict

**Accepted within the executed numerical and reporting scope as an experimental research extension. No new risk model is promoted.** No critical defect is known in that checked scope. Remaining scientific limitations include reused historical information, revisable provider data, overlapping outcomes, sparse crises, conditional-on-admissibility comparisons, unvalidated parameter mixtures and no demonstrated economic advantage under the declared alarm policy. Full literature review, a functioning algorithm and a passing test suite do not remove these limitations. The 23-task ledger retains unimplemented and partially addressed work explicitly.
