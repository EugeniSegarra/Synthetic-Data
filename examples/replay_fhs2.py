"""Replay selected forecast origins and audit scores with explicit pairwise distances.

This is a sampled numerical audit, not a re-run of every market forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_market.data import MarketData, read_csv
from synthetic_market.path_validation import _seed
from synthetic_market.research2.generators import fit_factorial


def direct_crps(values, observation):
    x = np.asarray(values, dtype=float)
    pair_sum = sum(
        float(np.abs(x[start : start + 128, None] - x[None, :]).sum())
        for start in range(0, len(x), 128)
    )
    return float(np.abs(x - observation).mean() - pair_sum / (2 * len(x) ** 2))


def independent_features(log_returns):
    r = np.asarray(log_returns, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    price = np.exp(np.cumsum(r, axis=1))
    peak = np.ones(len(r))
    worst = np.zeros(len(r))
    for p in price.T:
        peak = np.maximum(peak, p)
        worst = np.maximum(worst, 1 - p / peak)
    return {
        "terminal_return": price[:, -1] - 1,
        "max_drawdown": worst,
        "realized_volatility": np.sqrt(252 * np.mean(r * r, axis=1)),
    }


def replay(csv, results):
    data = read_csv(csv, price_basis="adjusted")
    frame = pd.read_csv(Path(results) / "forecasts.csv", float_precision="round_trip")
    config = json.loads((Path(results) / "metadata.json").read_text())["config"]
    choices = frame.loc[(frame.horizon == 126) & (frame.method == "fhs_global_zero")].sort_values(
        "origin_index"
    )
    chosen = choices.iloc[np.unique(np.linspace(0, len(choices) - 1, 5).astype(int))]
    checked = []
    log = np.log(data.close.iloc[:, 0].to_numpy())
    for row in chosen.itertuples():
        i = int(row.origin_index)
        prefix = MarketData(data.close.iloc[: i + 1], metadata=data.metadata)
        model = fit_factorial(prefix, config["train_window"], config["prior_sessions"])
        paths, errors = model.simulate(
            config["paths"], max(config["horizons"]), _seed(config["seed"], i, 201)
        )
        observed = independent_features(np.diff(log[i : i + 127]))
        for method, draw in paths.items():
            forecast = frame.loc[
                (frame.origin_index == i) & (frame.method == method) & (frame.horizon == 126)
            ]
            if len(forecast) != 1:
                raise ValueError("missing replay forecast")
            y = forecast.iloc[0]
            features = independent_features(draw[:, :126])
            for feature, value in features.items():
                score = direct_crps(value, observed[feature][0])
                if not np.isclose(score, y[f"{feature}_crps"], atol=2e-11, rtol=0):
                    raise ValueError(f"independent CRPS mismatch {row.origin} {method} {feature}")
            checked.append({"origin": row.origin, "method": method, "scores_checked": 3})
    return {
        "verdict": "PASS sampled pairwise-score replay",
        "scope": "five deterministic origins; same generator, separate path-functional and score algorithms",
        "checks": checked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    output = replay(args.csv, args.results)
    with args.out.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
