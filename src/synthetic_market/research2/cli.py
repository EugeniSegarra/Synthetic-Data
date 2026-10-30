"""Auditable research bundles, with immutable input/source hashes and safe publication."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..cli import _environment, _new_output_directory, _sha256, _write_json
from ..data import read_csv
from ..path_validation import path_features
from .controls import delayed_recalibration, stress_report
from .evaluation import ResearchConfig, evaluate, summarize
from .generators import BASELINE, CANDIDATES, fit_factorial


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): _sha256(p) for p in sorted(root.rglob("*.py"))}


def load(csv, price_basis, metadata=None):
    data = read_csv(csv, price_basis=price_basis)
    if data.close.shape[1] != 1:
        raise ValueError("research bundles require one asset per input")
    if metadata is not None:
        meta = json.loads(Path(metadata).read_text(encoding="utf8"))
        if not isinstance(meta, dict) or meta.get("price_basis", price_basis) != price_basis:
            raise ValueError("metadata price basis mismatch")
        for key, position in [("observed_start", 0), ("observed_end", -1)]:
            if key in meta and str(meta[key])[:10] != str(data.close.index[position].date()):
                raise ValueError("metadata dates disagree with input")
        data.metadata["provider_provenance"] = meta
    if str(data.close.index[-1].date()) > "2026-10-30":
        raise ValueError("data exceed information cutoff")
    return data


def run(args):
    before = source_hashes()
    inputs = {str(args.csv): _sha256(args.csv)}
    if args.metadata:
        inputs[str(args.metadata)] = _sha256(args.metadata)
    data = load(args.csv, args.price_basis, args.metadata)
    config = ResearchConfig(
        paths=args.paths,
        seed=args.seed,
        pairs=args.pairs,
        bootstrap_reps=args.bootstrap_reps,
        train_window=args.train_window,
        stride=args.stride,
    )
    with _new_output_directory(args.out) as staging:
        if args.command == "evaluate":
            frame, meta = evaluate(data, config, args.start, progress=True)
            summary = summarize(frame, config, args.evaluation_start)
            frame.to_csv(staging / "forecasts.csv", index=False, float_format="%.17g")
            delayed_recalibration(frame).to_csv(
                staging / "event_recalibration.csv", index=False, float_format="%.17g"
            )
            _write_json(staging / "summary.json", summary)
            _write_json(
                staging / "metadata.json",
                {"config": asdict(config), "origins": meta, "source": data.metadata},
            )
            _write_json(staging / "stress.json", stress_report())
            (staging / "README.txt").write_text(
                "Experimental, retrospective path forecasts. Default FHS unchanged.\n"
                "See docs/fhs2_protocol_2026-10-30.md. Recalibrated probabilities do not define price paths.\n"
                "Stress scenarios have no estimated probabilities. Decision costs are abstract, not P&L.\n"
            )
        else:
            model = fit_factorial(data, config.train_window)
            paths, errors = model.simulate(config.paths, 252, config.seed)
            if args.method not in paths:
                raise ValueError(f"requested method unavailable: {errors}")
            values = paths[args.method]
            np.savez_compressed(
                staging / "forecast.npz", log_returns=values, initial_price=data.close.iloc[-1, 0]
            )
            now = datetime.now(timezone.utc)
            _write_json(
                staging / "issuance.json",
                {
                    "issued_at_utc": now.isoformat(),
                    "data_asof": model.origin,
                    "method": args.method,
                    "config": asdict(config),
                    "model": model.summary(),
                    "status": "unscored_snapshot_not_proof_of_prospective_issuance",
                    "warning": "Verify data availability and that each target session follows issuance before prospective scoring.",
                    "terminal_and_path_quantiles": {
                        k: np.quantile(v, [0.05, 0.5, 0.95]).tolist()
                        for k, v in path_features(values).items()
                    },
                },
            )
        if source_hashes() != before or any(_sha256(Path(p)) != h for p, h in inputs.items()):
            raise RuntimeError("input or source changed during run; no bundle published")
        files = {p.name: _sha256(p) for p in staging.iterdir() if p.is_file()}
        _write_json(
            staging / "manifest.json",
            {
                "base_commit": "4230097d819d403e9a8a9451077da3002f1a74a8",
                "protocol_commit": "72fdf8615751f59a10428c8d891bbbec16c15bf8",
                "inputs": inputs,
                "source_hashes": before,
                "environment": _environment(),
                "config": asdict(config),
                "command": args.command,
                "files": files,
            },
        )
    return args.out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "issue"])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--price-basis", choices=["adjusted", "raw"], required=True)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--evaluation-start", default="2015-01-01")
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--stride", type=int, default=21)
    parser.add_argument("--method", choices=CANDIDATES, default=BASELINE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"Published research bundle: {run(args)}")
    return 0
