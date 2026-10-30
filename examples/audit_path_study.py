"""Executable numerical referee; not an independent human/LLM model certification.

Recomputes realized path outcomes directly from prices, checks provenance and
chronology, and preserves empirical failures as reported results. Does not tune
models or turn an imperfect empirical calibration into a release-blocking bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_market.data import read_csv


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(results: Path, input_csv: Path):
    manifest = json.loads((results / "manifest.json").read_text())
    meta = json.loads((results / "path_metadata.json").read_text())
    summary = json.loads((results / "path_summary.json").read_text())
    frame = pd.read_csv(results / "path_forecasts.csv")
    require(digest(input_csv) == manifest["input_file_sha256"], "input hash mismatch")
    for name, expected in manifest["files"].items():
        require(Path(name).name == name, "unsafe artifact path")
        require(digest(results / name) == expected, f"artifact hash mismatch: {name}")
    cfg = meta["config"]
    asset = manifest["argv"]["asset"]
    data = read_csv(input_csv, price_basis=manifest["argv"]["price_basis"])
    price = data.close[asset]
    require(not frame.duplicated(["origin_index", "horizon", "method"]).any(), "duplicate rows")
    actual_checks = 0
    for row in frame.drop_duplicates(["origin_index", "horizon"]).itertuples():
        i, h = row.origin_index, row.horizon
        raw = price.iloc[i : i + h + 1].to_numpy()
        require(len(raw) == h + 1, "unmatured target")
        require(row.origin == str(price.index[i].date()), "wrong origin date")
        require(row.target_end == str(price.index[i + h].date()), "wrong maturity date")
        # Independent O(H^2) price-based oracle, not the implementation's log recursion.
        drawdowns = [max(1 - raw[t] / raw[j] for j in range(t + 1)) for t in range(1, h + 1)]
        require(
            np.isclose(max(drawdowns), row.max_drawdown_actual, atol=2e-12, rtol=0),
            "drawdown oracle disagrees",
        )
        require(
            np.isclose(raw[-1] / raw[0] - 1, row.terminal_return_actual, atol=2e-12, rtol=0),
            "terminal-return oracle disagrees",
        )
        underwater = np.array(drawdowns) > 1e-12
        require(
            np.isclose(underwater.mean(), row.underwater_fraction_actual),
            "underwater frequency wrong",
        )
        current = longest = 0
        for flag in underwater:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        require(longest == row.longest_underwater_actual, "observed duration wrong")
        known_peak = price.iloc[: i + 1].max()
        if known_peak / raw[0] > 1 + 1e-12:
            recovered = bool(np.any(raw[1:] >= known_peak * (1 - 1e-12)))
            require(recovered == bool(row.recovery_actual), "censored recovery mishandled")
        actual_checks += 1
    for record in meta["audit"]:
        i = record["origin_index"]
        require(record["train_end"] == record["origin"], "training uses future")
        require(
            record["train_start"] == str(price.index[i - cfg["train_window"]].date()),
            "wrong rolling window",
        )
        require(
            record["last_matured_index"] is None or record["last_matured_index"] <= i,
            "premature feedback",
        )
        weights = np.array(list(record["weights"].values()))
        require(
            np.isfinite(weights).all() and (weights >= 0).all() and np.isclose(weights.sum(), 1),
            "invalid mixture",
        )
        for name in ("analog", "analog_area"):
            donor = record["status"][name]
            require(
                donor["maximum_donor_outcome_index"] <= donor["history_last_index"],
                "unmatured donor",
            )
    for feature in (
        "terminal_return",
        "realized_volatility",
        "max_drawdown",
        "underwater_fraction",
        "longest_underwater",
        "drawdown_area",
    ):
        q = frame[[f"{feature}_q{x}" for x in ("05", "50", "90", "95")]].to_numpy()
        require(np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all(), "invalid quantiles")
        y = frame[f"{feature}_actual"].to_numpy()
        coverage = (q[:, 0] <= y) & (y <= q[:, -1])
        require(
            np.array_equal(coverage, frame[f"{feature}_coverage90"].to_numpy().astype(bool)),
            "coverage identity failed",
        )
    complete = frame.loc[(frame.origin >= summary["evaluation_start"]) & frame.complete_expert_set]
    for row in summary["aggregate"]:
        part = complete.loc[
            (complete.method == row["method"]) & (complete.horizon == row["horizon"])
        ]
        require(len(part) == row["origins"], "summary denominator mismatch")
        for key in ("max_drawdown_crps", "kernel_path_score", "max_drawdown_coverage90"):
            require(
                np.isclose(part[key].mean(), row[key], rtol=1e-10, atol=1e-12),
                "summary arithmetic mismatch",
            )
    return {
        "verdict": "PASS: implementation/evidence consistency within checked scope",
        "not_certified": "market calibration, absence of all bugs, independent peer review or production VaR",
        "independence": "separate executable scalar reference algorithms; same authoring assistant",
        "realized_path_oracles_checked": actual_checks,
        "forecast_rows": len(frame),
        "requested_origins": meta["requested_origins"],
        "complete_origins": meta["complete_origins"],
        "evaluation_origins": summary["complete_evaluation_origins"],
        "manifest_sha256": digest(results / "manifest.json"),
        "input_sha256": digest(input_csv),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results, args.input)
    # Exclusive creation; never overwrite a prior audit record.
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
