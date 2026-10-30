"""Executable numerical referee. Same author, independent scalar reference algorithms.

A pass certifies the checks performed, never market calibration or absence of all bugs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(results, csv):
    results, csv = Path(results), Path(csv)
    manifest = json.loads((results / "manifest.json").read_text())
    metadata = json.loads((results / "metadata.json").read_text())
    summary = json.loads((results / "summary.json").read_text())
    frame = pd.read_csv(results / "forecasts.csv", float_precision="round_trip")
    require(sha(csv) in manifest["inputs"].values(), "unrecognized input digest")
    for name, value in manifest["files"].items():
        require(Path(name).name == name, "unsafe artifact name")
        require(sha(results / name) == value, f"hash mismatch: {name}")
    raw = pd.read_csv(csv, float_precision="round_trip")
    require(
        raw.Date.is_unique and raw.Date.is_monotonic_increasing, "input dates not unique/ordered"
    )
    if "Asset" in raw:
        require(raw.Asset.nunique() == 1, "expected one asset")
        prices = raw.Close.to_numpy()
    else:
        require(raw.shape[1] == 2, "expected date and one price column")
        prices = raw.iloc[:, 1].to_numpy()
    require(np.isfinite(prices).all() and np.all(prices > 0), "invalid input prices")
    require(
        not frame.duplicated(["origin_index", "horizon", "method"]).any(), "duplicate forecasts"
    )
    require(frame.target_end.max() <= "2026-10-30", "target beyond cutoff")
    checks = 0
    for key, part in frame.groupby(["origin_index", "horizon"]):
        i, h = map(int, key)
        p = prices[i : i + h + 1]
        require(len(p) == h + 1, "incomplete target")
        require((part.origin == raw.Date.iloc[i]).all(), "origin mismatch")
        require((part.target_end == raw.Date.iloc[i + h]).all(), "maturity mismatch")
        # Independent price-level high-water-mark loop, not log-domain cummax.
        maximum = p[0]
        dd = []
        run = 0
        longest = 0
        for value in p[1:]:
            maximum = max(maximum, value)
            draw = 1 - value / maximum
            dd.append(draw)
            run = run + 1 if draw > 1e-12 else 0
            longest = max(longest, run)
        log_returns = np.array([np.log(p[j + 1] / p[j]) for j in range(h)])
        expected = {
            "terminal_return_actual": p[-1] / p[0] - 1,
            "max_drawdown_actual": max(dd),
            "realized_volatility_actual": np.sqrt(252 * sum(v * v for v in log_returns) / h),
            "underwater_fraction_actual": sum(v > 1e-12 for v in dd) / h,
            "longest_underwater_actual": longest,
            "drawdown_area_actual": sum(dd) / h,
        }
        for name, value in expected.items():
            require(
                np.allclose(part[name], value, atol=3e-12, rtol=0),
                f"outcome oracle disagreement: {name}",
            )
        for barrier in (5, 10, 20):
            event = int(max(dd) >= barrier / 100 - 1e-12)
            require(
                (part[f"drawdown_hit{barrier}_actual"] == event).all(), "barrier event mismatch"
            )
        require(
            (part.origin_loss10_actual == int(min(p[1:]) / p[0] <= 0.9 + 1e-12)).all(),
            "origin-loss mismatch",
        )
        prior_max = max(prices[: i + 1])
        if prior_max / prices[i] > 1 + 1e-12:
            require(
                (part.recovery_actual == int(max(p[1:]) >= prior_max * (1 - 1e-12))).all(),
                "recovery mismatch",
            )
        checks += 1
    for name in (
        "terminal_return",
        "realized_volatility",
        "max_drawdown",
        "underwater_fraction",
        "longest_underwater",
        "drawdown_area",
    ):
        q = frame[[f"{name}_q05", f"{name}_q50", f"{name}_q90", f"{name}_q95"]].to_numpy()
        require(np.isfinite(q).all() and np.all(np.diff(q, axis=1) >= 0), "invalid quantile order")
        y = frame[f"{name}_actual"].to_numpy()
        require(
            np.array_equal(
                (q[:, 0] <= y) & (y <= q[:, -1]), frame[f"{name}_coverage90"].astype(bool)
            ),
            "coverage identity",
        )
    event = frame.drawdown_hit10_actual.to_numpy()
    alarm = (frame.drawdown_hit10_prob >= 1 / 6).to_numpy()
    loss = alarm * (1 - event) + 5 * (~alarm) * event
    require(np.array_equal(loss, frame.decision_loss), "economic-loss accounting")
    candidates = [
        f"{v}_{p}_{d}"
        for v in ("fhs", "harch")
        for p in ("global", "state")
        for d in ("zero", "shrink")
    ]
    for meta in metadata["origins"]:
        i = meta["origin_index"]
        require(meta["training_end"] == raw.Date.iloc[i], "training end mismatch")
        require(
            meta["training_start"] == raw.Date.iloc[i - metadata["config"]["train_window"]],
            "training start mismatch",
        )
        expected = sum(np.log(prices[j + 1] / prices[j]) for j in range(i)) / (
            i + metadata["config"]["prior_sessions"]
        )
        require(
            np.isclose(meta["fit"]["drift_daily_log"], expected, atol=1e-14, rtol=0),
            "drift has wrong prefix",
        )
        matured = frame.loc[
            (frame.horizon == 126) & (frame.origin_index + 126 <= i) & frame.method.isin(candidates)
        ]
        pivot = (
            matured.pivot(index="origin_index", columns="method", values="max_drawdown_crps")
            .reindex(columns=candidates)
            .dropna()
            .sort_index()
            .tail(60)
        )
        selected = (
            candidates[0]
            if len(pivot) < 24
            else candidates[int(np.argmin(pivot.mean(axis=0).to_numpy()))]
        )
        require(meta["selected"] == selected, "selector used incorrect/maturing information")
        require(meta["selector_matured_count"] == len(pivot), "selector count mismatch")
        require(
            meta["selector_last_maturity"] is None or meta["selector_last_maturity"] <= i,
            "premature selection feedback",
        )
        block = frame.loc[frame.origin_index == i]
        for row in block.loc[block.method == "delayed_selector"].itertuples():
            source = block.loc[(block.method == selected) & (block.horizon == row.horizon)]
            require(len(source) == 1, "missing selected forecast")
            require(
                row.max_drawdown_crps == float(source.max_drawdown_crps.iloc[0]),
                "selector score differs from issued candidate",
            )
    for row in summary["aggregate"]:
        part = frame.loc[
            (frame.origin >= summary["evaluation_start"])
            & (frame.method == row["method"])
            & (frame.horizon == row["horizon"])
        ]
        require(len(part) == row["origins"], "summary denominator mismatch")
        for name in (
            "max_drawdown_crps",
            "terminal_return_crps",
            "kernel_path_score",
            "max_drawdown_coverage90",
            "decision_loss",
        ):
            require(
                np.isclose(part[name].mean(), row[name], atol=1e-12, rtol=1e-10),
                "summary arithmetic mismatch",
            )
    recal = pd.read_csv(results / "event_recalibration.csv")
    require(
        (recal.last_matured_index.fillna(-1) <= recal.origin_index).all(),
        "recalibration leaks immature labels",
    )
    require(recal.recalibrated_probability.between(0, 1).all(), "invalid recalibrated probability")
    require(
        all(
            v["probability"] is None
            for v in json.loads((results / "stress.json").read_text()).values()
        ),
        "stress assigned probabilities",
    )
    require(summary["default_promoted"] is False, "unauthorized default promotion")
    return {
        "verdict": "PASS within executed numerical checks; no market-calibration certificate",
        "independence": "separate scalar algorithms, same authoring assistant",
        "forecast_rows": len(frame),
        "realized_targets_checked": checks,
        "origins": len(metadata["origins"]),
        "manifest_sha256": sha(results / "manifest.json"),
        "input_sha256": sha(csv),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.results, args.input)
    with args.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
