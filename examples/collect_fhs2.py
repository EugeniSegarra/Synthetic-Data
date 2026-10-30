"""Collect derived cross-market evidence without publishing provider price histories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

METHODS = [
    f"{v}_{p}_{d}"
    for v in ("fhs", "harch")
    for p in ("global", "state")
    for d in ("zero", "shrink")
] + ["delayed_selector"]
BASE = "fhs_global_zero"


def collect(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tables, epochs, horizons, failed, recal, provenance = [], [], [], [], [], {}
    for asset in ("SPY", "EWJ", "EEM", "VGK"):
        folder = root / asset
        frame = pd.read_csv(folder / "forecasts.csv", float_precision="round_trip")
        summary = json.loads((folder / "summary.json").read_text())
        meta = json.loads((folder / "metadata.json").read_text())
        provenance[asset] = {
            "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
            "forecast_rows": len(frame),
            "origins": len(meta["origins"]),
        }
        principal = frame.loc[frame.origin >= "2015-01-01"].copy()
        for h, part in principal.groupby("horizon"):
            pivot = (
                part.pivot(index="origin_index", columns="method", values="max_drawdown_crps")
                .reindex(columns=METHODS)
                .dropna()
            )
            support = pivot.index
            paired = part.loc[part.origin_index.isin(support)]
            means = paired.groupby("method").mean(numeric_only=True)
            for method in METHODS:
                row = {
                    "asset": asset,
                    "horizon": int(h),
                    "method": method,
                    "common_origins": len(support),
                }
                for metric in (
                    "max_drawdown_crps",
                    "terminal_return_crps",
                    "realized_volatility_crps",
                    "kernel_path_score",
                ):
                    row[metric] = float(means.loc[method, metric])
                    row[metric + "_change"] = float(
                        means.loc[method, metric] / means.loc[BASE, metric] - 1
                    )
                horizons.append(row)
                if h == 126:
                    row = dict(row)
                    for metric in (
                        "max_drawdown_coverage90",
                        "drawdown_hit10_prob",
                        "drawdown_hit10_actual",
                        "decision_loss",
                    ):
                        row[metric] = float(means.loc[method, metric])
                    comparisons = [
                        x
                        for x in summary["simultaneous_comparisons"]
                        if x["horizon"] == 126 and x["block_length"] == 12 and x["method"] == method
                    ]
                    if comparisons:
                        row.update(
                            {
                                k: comparisons[0][k]
                                for k in (
                                    "simultaneous_lower95",
                                    "simultaneous_upper95",
                                    "numeric_screen_pass",
                                )
                            }
                        )
                    tables.append(row)
            if h == 126:
                for label, first, last in [
                    ("2015-2019", "2015-01-01", "2019-12-31"),
                    ("2020-2022", "2020-01-01", "2022-12-31"),
                    ("2023-cutoff", "2023-01-01", "2026-10-30"),
                ]:
                    group = paired.loc[paired.origin.between(first, last)]
                    for method, g in group.groupby("method"):
                        baseline = group.loc[group.method == BASE]
                        epochs.append(
                            {
                                "asset": asset,
                                "epoch": label,
                                "method": method,
                                "origins": len(g),
                                "drawdown_crps": float(g.max_drawdown_crps.mean()),
                                "change": float(
                                    g.max_drawdown_crps.mean() / baseline.max_drawdown_crps.mean()
                                    - 1
                                ),
                            }
                        )
        for origin in meta["origins"]:
            for method, error in origin["errors"].items():
                failed.append(
                    {
                        "asset": asset,
                        "origin": origin["origin"],
                        "component": method,
                        "error": error,
                    }
                )
        r = pd.read_csv(folder / "event_recalibration.csv")
        r = r.loc[(r.origin >= "2015-01-01") & (r.horizon == 126)]
        recal.append(
            {
                "asset": asset,
                "origins": len(r),
                **{
                    k: float(r[k].mean())
                    for k in (
                        "raw_brier",
                        "recalibrated_brier",
                        "raw_probability",
                        "recalibrated_probability",
                        "event",
                    )
                },
            }
        )
    for name, table in [
        ("six_month_comparison", tables),
        ("by_horizon", horizons),
        ("by_epoch", epochs),
        ("failed_origins", failed),
        ("event_recalibration", recal),
    ]:
        pd.DataFrame(table).to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return pd.DataFrame(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.out)


if __name__ == "__main__":
    main()
