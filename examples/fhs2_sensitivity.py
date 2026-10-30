"""Declared sensitivity exercises, not a new tuned model or Bayesian posterior.

Bootstrap refits are re-filtered on the ACTUAL observed history before simulation.
No bootstrap endpoint is used as today's state. Failed refits remain in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from synthetic_market.data import MarketData, read_csv
from synthetic_market.model import CalibrationError, ModelConfig, _fit_asset
from synthetic_market.path_validation import path_features, stationary_indices
from synthetic_market.research2.controls import bootstrap_drift
from synthetic_market.research2.generators import RiskFilter, fit_factorial, shrunk_drift

DATES = ("2008-10-01", "2010-05-06", "2019-12-13", "2020-04-15", "2022-06-15", "2026-10-29")


def refilter(history, detail):
    """Map a bootstrap parameter vector to actual-history conditional variances."""
    r = np.asarray(history, dtype=float)
    if r.ndim != 1 or len(r) < 253 or not np.isfinite(r).all():
        raise ValueError("need finite actual returns")
    if detail["selected"] not in {"garch", "gjr"}:
        raise ValueError("unsupported variance model")
    o = int(detail["selected"] == "gjr")
    a, g, b = (float(detail[k]) for k in ("alpha", "gamma", "beta"))
    w = float(detail["omega"])
    if not np.isfinite([w, a, g, b]).all() or w <= 0 or min(a, b, a + g) < 0:
        raise CalibrationError("invalid bootstrap parameter vector")
    parameters = [w * 10000, a] + ([g] if o else []) + [b]
    fixed = arch_model(r * 100, mean="Zero", vol="GARCH", p=1, o=o, q=1, rescale=False).fix(
        parameters
    )
    sigma = np.asarray(fixed.conditional_volatility) / 100
    z0 = r / sigma
    if not np.isfinite(z0).all() or z0.std() < 1e-12:
        raise CalibrationError("invalid actual-history refilter")
    z = (z0 - z0.mean()) / z0.std()
    persistence = a + b + g * np.mean(z**2 * (z < 0))
    if not 0 <= persistence < 0.9999:
        raise CalibrationError("refiltered empirical law violates stability")
    return RiskFilter(
        "fhs", w, np.array([a, g, b]), z, r.copy(), float(sigma[-1] ** 2), dict(detail)
    )


def parameter_filters(history, reps=30, seed=20261030):
    r = np.asarray(history, dtype=float)
    indices = stationary_indices(len(r), reps, len(r), 21, seed)
    accepted, rejected = [], []
    for number, rows in enumerate(indices):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detail, *_ = _fit_asset(pd.Series(r[rows]), "bootstrap", ModelConfig())
            accepted.append(refilter(r, detail))
        except CalibrationError as exc:
            rejected.append({"replicate": number, "error": str(exc)})
    return accepted, rejected


def quantiles(values):
    return dict(zip(("p05", "p50", "p95"), np.quantile(values, [0.05, 0.5, 0.95]).tolist()))


def study(data, paths=1000, reps=30):
    records = []
    for date in DATES:
        close = data.close.loc[:date]
        if len(close) < 1261:
            continue
        prefix = MarketData(close, metadata=dict(data.metadata))
        fitted = fit_factorial(prefix)
        history = np.diff(np.log(close.iloc[:, 0]))
        risk = fitted.filters["fhs"]
        uniforms = np.random.default_rng(20261030).random((126, paths))
        no_drift, sigma = risk.simulate(uniforms)
        drift_draws = bootstrap_drift(history, reps=200, seed=20261030)
        parameter_draws, failures = parameter_filters(history[-1260:], reps=reps)
        param_mdd = []
        param_terminal = []
        mixture_mdd, mixture_terminal = [], []
        for sample in parameter_draws:
            r, _ = sample.simulate(uniforms)
            features = path_features(r)
            mixture_mdd.append(features["max_drawdown"])
            mixture_terminal.append(features["terminal_return"])
            param_mdd.append(float(np.quantile(features["max_drawdown"], 0.95)))
            param_terminal.append(float(np.quantile(features["terminal_return"], 0.05)))
        seeds = []
        for seed in (17, 41, 97):
            r, _ = risk.simulate(np.random.default_rng(seed).random((126, paths)))
            f = path_features(r)
            seeds.append(
                {
                    "seed": seed,
                    "drawdown_q95": float(np.quantile(f["max_drawdown"], 0.95)),
                    "hit10_probability": float(np.mean(f["max_drawdown"] >= 0.10)),
                }
            )
        records.append(
            {
                "asof": str(close.index[-1].date()),
                "paths_per_calibration": paths,
                "bootstrap_mean_annual_log": quantiles(252 * drift_draws),
                "drift_prior_sensitivities": {
                    str(prior): {
                        "annual_log_mean": 252 * shrunk_drift(history, prior),
                        "drawdown_q95": float(
                            np.quantile(
                                path_features(no_drift + shrunk_drift(history, prior))[
                                    "max_drawdown"
                                ],
                                0.95,
                            )
                        ),
                    }
                    for prior in (2520, 5040, 10080)
                },
                "parameter_refits_attempted": reps,
                "parameter_refits_accepted": len(parameter_draws),
                "parameter_refits_rejected": failures,
                "accepted_parameters": [
                    {k: f.detail[k] for k in ("selected", "omega", "alpha", "gamma", "beta")}
                    for f in parameter_draws
                ],
                "equal_weight_accepted_parameter_mixture": {
                    "drawdown_quantiles": quantiles(np.concatenate(mixture_mdd)),
                    "terminal_quantiles": quantiles(np.concatenate(mixture_terminal)),
                    "hit10_probability": float(np.mean(np.concatenate(mixture_mdd) >= 0.10)),
                    "interpretation": "bootstrap parameter sensitivity, not validated posterior predictive law",
                }
                if parameter_draws
                else None,
                "parameter_refit_drawdown_q95_range": quantiles(param_mdd) if param_mdd else None,
                "parameter_refit_terminal_q05_range": quantiles(param_terminal)
                if param_terminal
                else None,
                "seed_sensitivity": seeds,
                "baseline_average_sigma_by_month": [
                    float(np.mean(sigma[:, k : k + 21]) * np.sqrt(252)) for k in range(0, 126, 21)
                ],
            }
        )
    return {
        "purpose": "post-benchmark diagnostic sensitivity, never a retuned forecast comparison",
        "assumptions": [
            "stationary block bootstrap, mean length 21",
            "bootstrap refit re-filtered on actual history",
            "conditional on accepted admissible refits; rejected replicates are explicitly reported",
            "no new crises, no calibrated posterior, no validated parameter-mixture predictive distribution",
        ],
        "origins": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    data = read_csv(args.csv, price_basis="adjusted")
    if data.close.shape[1] != 1 or str(data.close.index[-1].date()) > "2026-10-30":
        raise ValueError("need univariate input within cutoff")
    result = study(data, args.paths, args.reps)
    if hashlib.sha256(args.csv.read_bytes()).hexdigest() != digest:
        raise RuntimeError("input changed")
    result["input_sha256"] = digest
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
