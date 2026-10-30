"""Reproduce three explicitly dated real-data benchmarks; never select winners.

Run after installing .[yahoo]. Inputs are saved locally, not redistributed in the
repository. The periods below are evaluation windows, not a claim of prospective
pre-registration or untouched final evidence after this release's development.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_market.cli import _environment, _new_output_directory, _write_json
from synthetic_market.data import MarketData, download_yahoo, read_csv, write_csv
from synthetic_market.model import ModelConfig
from synthetic_market.validation import evaluate_holdout

WINDOWS = [
    ("covid_2020", "2019-12-31", "2020-12-31"),
    ("inflation_2022", "2021-12-31", "2022-12-31"),
    ("recent_2025_2026", "2024-12-31", "2026-10-29"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Reuse the adjusted SPY/TLT/GLD input snapshot")
    parser.add_argument("--metadata", type=Path, help="Original input metadata JSON, with --csv")
    args = parser.parse_args()
    if args.metadata is not None and args.csv is None:
        parser.error("--metadata requires --csv")
    if args.csv is None:
        data = download_yahoo(["SPY", "TLT", "GLD"], "2010-01-01", "2026-10-30")
    else:
        data = read_csv(args.csv, price_basis="adjusted")
        if args.metadata:
            data = MarketData(data.close, data.ohlc, json.loads(args.metadata.read_text()))
    if data.assets != ("SPY", "TLT", "GLD") or data.metadata["price_basis"] != "adjusted":
        raise ValueError("This specified benchmark requires adjusted SPY, TLT, GLD in that order")
    weights = {asset: 1 / 3 for asset in data.assets}
    summary = {
        "review_date": "2026-10-30",
        "data_sha256": data.fingerprint(),
        "assets": list(data.assets),
        "source": data.metadata,
        "environment": _environment(),
        "config": {"volatility": "auto", "mean": "zero"},
        "weights": weights,
        "seed": 20261030,
        "energy_pairs": 4096,
        "windows": [],
        "interpretation": "Observed retrospective benchmark, not universal calibration or prospective performance.",
    }
    with _new_output_directory(args.out) as out:
        write_csv(data, out / "input.csv")
        _write_json(out / "input_metadata.json", data.metadata)
        for name, split, end in WINDOWS:
            report = evaluate_holdout(
                data.subset(end=end),
                split_date=split,
                config=ModelConfig(),
                weights=weights,
                seed=20261030,
                energy_pairs=4096,
            )
            _write_json(out / f"{name}.json", report)
            compact = {key: value for key, value in report.items() if key != "per_observation"}
            compact["name"] = name
            summary["windows"].append(compact)
            print(
                name,
                "model CRPS",
                report["metrics_model"]["crps_mean"],
                "baseline",
                report["metrics_baseline"]["crps_mean"],
                flush=True,
            )
        _write_json(out / "benchmark_summary.json", summary)
    print("Saved complete benchmark:", args.out)


if __name__ == "__main__":
    main()
