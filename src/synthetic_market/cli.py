"""Reproducible command-line runs with auditable, atomically published artifacts."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .data import MarketData, download_yahoo, make_demo_data, read_csv, write_csv
from .model import ModelConfig, fit_market
from .validation import _weights, empirical_tail_statistics, evaluate_holdout


def _positive_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return value


def _nonnegative_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out", required=True, type=Path, help="New output directory; never overwritten"
    )


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mean", choices=["zero", "constant"], default="zero")
    parser.add_argument("--volatility", choices=["auto", "garch", "gjr"], default="auto")
    parser.add_argument("--seed", type=_nonnegative_integer, default=42)
    parser.add_argument(
        "--weights", help="Long-only common-currency weights, e.g. SPY=0.6,TLT=0.3,GLD=0.1"
    )
    parser.add_argument("--energy-pairs", type=_positive_integer, default=2048)


def _add_scenarios(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paths", type=_positive_integer, default=250)
    parser.add_argument(
        "--horizon", type=_positive_integer, default=63, help="Future observation steps"
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Omit large scenarios.csv; retain safe NPZ"
    )


def _add_input(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Wide Close or long OHLC CSV snapshot")
    source.add_argument(
        "--tickers", nargs="+", help="Explicit Yahoo symbols; requires --start and --end"
    )
    parser.add_argument("--start", help="Yahoo start date, inclusive")
    parser.add_argument(
        "--end", help="Yahoo end date, exclusive; required to make requests explicit"
    )
    parser.add_argument("--alignment", choices=["strict", "intersection"], default="strict")
    parser.add_argument(
        "--price-basis",
        choices=["adjusted", "raw", "unspecified"],
        default=None,
        help="CSV economic convention; Yahoo always uses adjusted OHLC",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic-market",
        description="Auditable joint filtered historical market scenarios and chronological diagnostics.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Offline synthetic fixture, holdout evaluation, and scenarios"
    )
    demo.add_argument("--observations", type=_positive_integer, default=1600)
    _add_model(demo)
    _add_scenarios(demo)
    _add_output(demo)

    simulate = commands.add_parser(
        "simulate", help="Fit supplied history and generate future scenarios"
    )
    _add_input(simulate)
    _add_model(simulate)
    _add_scenarios(simulate)
    simulate.add_argument(
        "--validate-split", help="Optional inclusive training cutoff for holdout diagnostics"
    )
    _add_output(simulate)

    validate = commands.add_parser(
        "validate", help="Chronological one-step evaluation against a baseline"
    )
    _add_input(validate)
    _add_model(validate)
    validate.add_argument(
        "--split-date", required=True, help="Inclusive training cutoff; test dates follow it"
    )
    _add_output(validate)

    download = commands.add_parser(
        "download", help="Save a reproducible adjusted Yahoo OHLC snapshot"
    )
    download.add_argument("--tickers", nargs="+", required=True)
    download.add_argument("--start", required=True, help="Inclusive")
    download.add_argument("--end", required=True, help="Exclusive")
    download.add_argument("--alignment", choices=["strict", "intersection"], default="strict")
    _add_output(download)
    return parser


def _input_data(args: argparse.Namespace) -> MarketData:
    if args.command == "demo":
        return make_demo_data(n_obs=args.observations, seed=1729)
    if getattr(args, "csv", None) is not None:
        if args.start is not None or args.end is not None:
            raise ValueError(
                "--start/--end belong to Yahoo requests; supply the intended CSV history directly"
            )
        return read_csv(
            args.csv, price_basis=args.price_basis or "unspecified", alignment=args.alignment
        )
    if not args.start or not args.end:
        raise ValueError("Yahoo requests require explicit --start and --end (end is exclusive)")
    if getattr(args, "price_basis", None) is not None:
        raise ValueError(
            "--price-basis applies only to CSV input; Yahoo prices are explicitly adjusted"
        )
    return download_yahoo(args.tickers, args.start, args.end, alignment=args.alignment)


def _parse_weights(raw: str | None, assets: tuple[str, ...]) -> dict[str, float]:
    parsed = None
    if raw is not None:
        parsed = {}
        for item in raw.split(","):
            if item.count("=") != 1:
                raise ValueError("--weights must use Asset=value pairs separated by commas")
            asset, value = (part.strip() for part in item.split("=", 1))
            if not asset or asset in parsed:
                raise ValueError("--weights must name each asset once, with no duplicates")
            try:
                parsed[asset] = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid weight for {asset}") from exc
    values = _weights(assets, parsed)
    return {asset: float(value) for asset, value in zip(assets, values, strict=True)}


def _seeds(master: int) -> dict[str, Any]:
    evaluation, simulation = np.random.SeedSequence(master).spawn(2)
    return {
        "master": master,
        "evaluation": int(evaluation.generate_state(1, dtype=np.uint32)[0]),
        "simulation": int(simulation.generate_state(1, dtype=np.uint32)[0]),
        "derivation": "SeedSequence(master).spawn(2), then one uint32 from each child",
        "demo_data": 1729,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "arch", "statsmodels", "yfinance"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
        "dependencies": packages,
    }


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomic same-filesystem publication without clobbering a racing destination.

    Linux and macOS expose exclusive rename flags; Windows os.rename already
    refuses an existing destination. No check-then-replace fallback is used.
    """
    if sys.platform == "win32":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 4)
    else:
        raise RuntimeError(
            "Atomic exclusive directory publication requires Linux, macOS, or Windows"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"Output directory already exists: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def _new_output_directory(destination: Path):
    destination = destination.expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"Output directory already exists: {destination}; choose a new --out")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        yield staging
        _rename_no_replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _scenario_arrays(result) -> dict[str, np.ndarray]:
    arrays = {
        "close": result.close,
        "log_returns": result.log_returns,
        "sigma": result.sigma,
        "sampled_rows": result.sampled_rows,
        "assets": np.asarray(result.assets, dtype=str),
        "source_dates": np.asarray(result.source_dates, dtype=str),
        "origin": np.asarray(result.origin, dtype=str),
        "seed": np.asarray(result.seed, dtype=np.uint64),
        "steps": np.arange(1, result.close.shape[1] + 1, dtype=np.int64),
    }
    for field in ("open", "high", "low"):
        value = getattr(result, field)
        if value is not None:
            arrays[field] = value
    for name, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError(f"Object dtype is not permitted in scenarios.npz: {name}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite scenario output: {name}")
    return arrays


def _scenario_summary(result, data, weights):
    origin = data.close.iloc[-1].to_numpy(dtype=float)
    terminal = result.close[:, -1, :] / origin - 1.0
    weight_vector = np.array([weights[asset] for asset in result.assets])
    portfolio = terminal @ weight_vector
    if not np.isfinite(terminal).all() or not np.isfinite(portfolio).all():
        raise ValueError("Terminal scenario return calculation overflowed")
    levels = [0.05, 0.50, 0.95]
    quantiles = np.quantile(terminal, levels, axis=0, method="inverted_cdf")
    portfolio_quantiles = np.quantile(portfolio, levels, method="inverted_cdf")
    return {
        "origin": result.origin,
        "paths": result.close.shape[0],
        "horizon_observations": result.close.shape[1],
        "price_basis": data.metadata["price_basis"],
        "terminal_asset_simple_return_quantiles": {
            asset: {str(q): float(quantiles[k, j]) for k, q in enumerate(levels)}
            for j, asset in enumerate(result.assets)
        },
        "portfolio": {
            "assumption": "Buy-and-hold initial weights, common currency, no rebalancing or costs",
            "weights": weights,
            "terminal_simple_return_quantiles": {
                str(q): float(portfolio_quantiles[k]) for k, q in enumerate(levels)
            },
            "lower_5pct": empirical_tail_statistics(portfolio, 0.05),
        },
        "interpretation": (
            "Conditional scenarios under the fitted empirical model, not expected investment alpha. "
            "Quantiles depend on model assumptions and path count; model uncertainty is not simulated."
        ),
    }


def _summary_markdown(args, data, seeds, model, validation, scenarios) -> str:
    lines = [
        "# Synthetic market run",
        "",
        f"Command: `{args.command}`. Assets: {', '.join(data.assets)}.",
        "",
        f"Input: {len(data.close)} observations from {data.close.index[0].date()} "
        f"through {data.close.index[-1].date()}. Price convention: `{data.metadata['price_basis']}`.",
        "",
        f"Source: {data.metadata.get('source', 'unspecified')}",
        "",
    ]
    if args.command == "demo":
        lines.extend(
            [
                "**This input is a synthetic software demonstration, not a real-market validation.**",
                "",
                "The last 20% of observations are held out chronologically for diagnostics. "
                "Future scenarios are then fitted to the complete supplied history at its final origin.",
                "",
            ]
        )
    if seeds:
        lines.extend(
            [
                f"Master seed: `{seeds['master']}`; evaluation stream: `{seeds['evaluation']}`; "
                f"simulation stream: `{seeds['simulation']}`. Streams are independent.",
                "",
            ]
        )
    if model is not None:
        lines.extend(
            [
                "## Fitted model",
                "",
                "Joint filtered historical simulation. Volatility candidates are fitted only to the "
                "supplied history; inadmissible models are rejected, never silently clipped.",
                "",
                "| Asset | Selected filter | Empirical persistence |",
                "| --- | --- | ---: |",
            ]
        )
        for asset, detail in model["fit"].items():
            lines.append(
                f"| {asset} | {detail['selected']} | {detail['empirical_persistence']:.6f} |"
            )
        lines.append("")
    if validation is not None:
        lines.extend(
            [
                "## Chronological holdout diagnostics",
                "",
                f"Training ends {validation['training']['end']}; the test contains "
                f"{validation['test']['observations']} one-step observations. "
                "Only past observed returns update each forecast state; parameters stay frozen.",
                "",
                "| Mean score (lower is better) | Filtered model | Unfiltered joint bootstrap |",
                "| --- | ---: | ---: |",
            ]
        )
        for key in ("crps_mean", "energy_score_mean", "variogram_score_mean"):
            a, b = validation["metrics_model"][key], validation["metrics_baseline"][key]
            a_text = "not applicable" if a is None else f"{a:.8g}"
            b_text = "not applicable" if b is None else f"{b:.8g}"
            lines.append(f"| {key} | {a_text} | {b_text} |")
        lines.extend(
            [
                "",
                "### Portfolio lower-tail coverage",
                "",
                "A breach means the realized portfolio return was strictly below that day's "
                "forecast quantile. Nominal expected breaches equal test observations × tail probability. "
                "These counts must be assessed alongside average scores; a better average score "
                "does not establish calibrated tail risk.",
                "",
                "| Method | Nominal tail | Observed breaches | Test observations | "
                "Nominal expected breaches | Observed / nominal expected | Observed breach rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        undercoverage = []
        for label, display_name in (
            ("model", "Filtered model"),
            ("baseline", "Unfiltered joint bootstrap"),
        ):
            for key in ("0.05", "0.01"):
                tail = validation["portfolio"][label][key]
                observed = tail["violations"]
                nominal = tail["expected_violations_nominal"]
                ratio = observed / nominal if nominal > 0 else None
                ratio_text = "not applicable" if ratio is None else f"{ratio:.2f}×"
                lines.append(
                    f"| {display_name} | {tail['alpha']:.0%} | {observed} | "
                    f"{tail['observations']} | {nominal:.2f} | {ratio_text} | "
                    f"{tail['violation_rate']:.2%} |"
                )
                if label == "model" and nominal >= 10 and ratio is not None and ratio >= 2:
                    undercoverage.append(
                        f"{tail['alpha']:.0%} tail: {observed} observed versus "
                        f"{nominal:.2f} nominal expected breaches ({ratio:.2f}×)"
                    )
        if undercoverage:
            lines.extend(
                [
                    "",
                    "**Material undercoverage in this sample; do not treat as calibrated VaR. "
                    "Descriptive flag, not a statistical test.**",
                    "",
                    "Filtered model — " + "; ".join(undercoverage) + ".",
                ]
            )
        lines.extend(
            [
                "",
                "These scores are diagnostics, not an automatic validation pass or evidence of alpha. "
                "See validation.json for every forecast, empirical boundary probabilities, and conventions.",
                "",
            ]
        )
        diagnostic_warnings = list(dict.fromkeys(validation.get("warnings", [])))
        if diagnostic_warnings:
            lines.extend(["### Diagnostic cautions", ""])
            lines.extend(f"- {warning}" for warning in diagnostic_warnings)
            lines.append("")
    if scenarios is not None:
        lines.extend(
            [
                "## Conditional future scenarios",
                "",
                f"{scenarios['paths']} paths × {scenarios['horizon_observations']} future observation steps. "
                "Step 1 includes a new return; no future exchange dates are invented.",
                "",
                "| Asset | Terminal return P05 | P50 | P95 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for asset, values in scenarios["terminal_asset_simple_return_quantiles"].items():
            lines.append(
                f"| {asset} | {values['0.05']:.2%} | {values['0.5']:.2%} | {values['0.95']:.2%} |"
            )
        lines.extend(
            [
                "",
                "Terminal portfolio statistics in scenario_summary.json use buy-and-hold initial "
                "weights. Holdout portfolio diagnostics instead refer to each one-period forecast.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The empirical innovation support is finite; it does not extrapolate unseen standardized "
            "shocks. More paths do not remove parameter or model uncertainty. Adjusted OHLC are not "
            "executable historical quotes. Daily candles contain no intraday ordering, liquidity, "
            "volume, transaction costs, or execution model.",
            "",
            "Manifest hashes cover every artifact except the manifest itself. NPZ files contain only "
            "numeric/string arrays and can be opened with numpy.load(..., allow_pickle=False).",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace, argv: list[str] | None = None) -> Path:
    """Prepare a complete run in a sibling staging directory, then publish once."""
    out = args.out.expanduser().absolute()
    if os.path.lexists(out):
        raise FileExistsError(f"Output directory already exists: {out}; choose a new --out")
    data = _input_data(args)
    weights = None if args.command == "download" else _parse_weights(args.weights, data.assets)
    seeds = None if args.command == "download" else _seeds(args.seed)
    config = (
        None
        if args.command == "download"
        else ModelConfig(volatility=args.volatility, mean=args.mean)
    )
    with _new_output_directory(out) as staging:
        write_csv(data, staging / "input.csv")
        _write_json(staging / "input_metadata.json", data.metadata)
        validation = model_summary = scenario_summary = None
        split = None
        if args.command == "demo":
            training_prices = int(np.floor(0.8 * len(data.close)))
            split = data.close.index[training_prices - 1]
        elif args.command == "validate":
            split = args.split_date
        elif args.command == "simulate":
            split = args.validate_split
        if split is not None:
            validation = evaluate_holdout(
                data,
                split,
                config=config,
                weights=weights,
                seed=seeds["evaluation"],
                energy_pairs=args.energy_pairs,
            )
            _write_json(staging / "validation.json", validation)
        if args.command in {"demo", "simulate"}:
            model = fit_market(data, config=config)
            model_summary = model.summary()
            _write_json(staging / "model.json", model_summary)
            scenarios = model.simulate(
                n_paths=args.paths, horizon=args.horizon, seed=seeds["simulation"]
            )
            np.savez_compressed(staging / "scenarios.npz", **_scenario_arrays(scenarios))
            if not args.no_csv:
                scenarios.to_frame().reset_index().to_csv(
                    staging / "scenarios.csv", index=False, float_format="%.17g"
                )
            scenario_summary = _scenario_summary(scenarios, data, weights)
            _write_json(staging / "scenario_summary.json", scenario_summary)
        (staging / "summary.md").write_text(
            _summary_markdown(args, data, seeds, model_summary, validation, scenario_summary),
            encoding="utf-8",
        )
        artifacts = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "arguments": vars(args),
            "argv": list(argv) if argv is not None else None,
            "environment": _environment(),
            "input_sha256": data.fingerprint(),
            "assets": list(data.assets),
            "price_basis": data.metadata["price_basis"],
            "seed_streams": seeds,
            "weights": weights,
            "artifacts": artifacts,
            "scope": "Conditional model scenarios and diagnostics; no claim of investment alpha or universal validity",
        }
        _write_json(staging / "manifest.json", manifest)
    return out


def main(argv: list[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(actual_argv)
    try:
        out = run(args, actual_argv)
    except KeyboardInterrupt:
        print("error: interrupted; no incomplete run was published", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.command} artifacts to {out}")
    return 0
