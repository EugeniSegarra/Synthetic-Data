"""Actual CLI runs, artifact contracts, and failure-safe publication."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from synthetic_market import cli
from synthetic_market.data import make_demo_data, read_csv, write_csv


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli-demo") / "run"
    code = cli.main(
        [
            "demo",
            "--out",
            str(out),
            "--observations",
            "500",
            "--paths",
            "4",
            "--horizon",
            "3",
            "--seed",
            "42",
        ]
    )
    assert code == 0
    return out


def test_demo_artifacts_hashes_and_safe_npz(demo_output):
    expected = {
        "input.csv",
        "input_metadata.json",
        "model.json",
        "validation.json",
        "scenarios.npz",
        "scenarios.csv",
        "scenario_summary.json",
        "manifest.json",
        "summary.md",
    }
    assert {p.name for p in demo_output.iterdir()} == expected
    manifest = json.loads((demo_output / "manifest.json").read_text())
    assert set(manifest["artifacts"]) == expected - {"manifest.json"}
    for name, facts in manifest["artifacts"].items():
        raw = (demo_output / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == facts["sha256"]
        assert len(raw) == facts["bytes"]
    data = read_csv(demo_output / "input.csv", price_basis="synthetic")
    assert data.fingerprint() == manifest["input_sha256"]
    with np.load(demo_output / "scenarios.npz", allow_pickle=False) as archive:
        for name in archive.files:
            assert not archive[name].dtype.hasobject
        assert archive["close"].shape == (4, 3, 3)
        assert archive["sampled_rows"].shape == (4, 3)
        np.testing.assert_array_equal(archive["steps"], [1, 2, 3])
        assert (archive["sigma"] > 0).all()
        np.testing.assert_allclose(
            archive["log_returns"][:, 0],
            np.log(archive["close"][:, 0] / data.close.iloc[-1].to_numpy()),
            rtol=1e-11,
            atol=1e-14,
        )
        assert (archive["high"] >= np.maximum(archive["open"], archive["close"])).all()
        assert (archive["low"] <= np.minimum(archive["open"], archive["close"])).all()
    report = json.loads((demo_output / "validation.json").read_text())
    assert report["training"]["price_observations"] == 400
    assert report["test"]["observations"] == 100
    assert len(report["per_observation"]) == 100
    csv = pd.read_csv(demo_output / "scenarios.csv")
    assert len(csv) == 4 * 3 * 3
    assert set(csv.Step) == {1, 2, 3}
    assert "ResampledDate" in csv
    assert manifest["seed_streams"]["evaluation"] != manifest["seed_streams"]["simulation"]
    assert "synthetic software demonstration" in (demo_output / "summary.md").read_text()


def test_existing_directory_is_never_overwritten(demo_output, capsys):
    manifest_before = (demo_output / "manifest.json").read_bytes()
    code = cli.main(["demo", "--out", str(demo_output)])
    assert code == 2
    assert "already exists" in capsys.readouterr().err
    assert (demo_output / "manifest.json").read_bytes() == manifest_before


def test_failure_after_staging_leaves_no_published_or_temporary_directory(tmp_path, capsys):
    out = tmp_path / "insufficient-history"
    code = cli.main(
        [
            "demo",
            "--out",
            str(out),
            "--observations",
            "260",
            "--paths",
            "2",
            "--horizon",
            "2",
        ]
    )
    assert code == 2
    assert "return observations" in capsys.readouterr().err
    assert not out.exists()
    assert list(tmp_path.glob(".insufficient-history.staging-*")) == []


def test_late_write_failure_cleans_up_all_partial_artifacts(tmp_path, monkeypatch):
    original = cli._write_json

    def fail_on_manifest(path, payload):
        if path.name == "manifest.json":
            raise OSError("Controlled final-write failure")
        original(path, payload)

    monkeypatch.setattr(cli, "_write_json", fail_on_manifest)
    out = tmp_path / "late-failure"
    code = cli.main(
        [
            "demo",
            "--out",
            str(out),
            "--observations",
            "500",
            "--paths",
            "2",
            "--horizon",
            "2",
        ]
    )
    assert code == 2
    assert not out.exists()
    assert list(tmp_path.glob(".late-failure.staging-*")) == []


def test_publication_does_not_overwrite_a_racing_empty_directory(tmp_path):
    out = tmp_path / "race"
    with pytest.raises(FileExistsError):
        with cli._new_output_directory(out) as staging:
            (staging / "new.txt").write_text("must not overwrite the racing directory")
            out.mkdir()
            before_inode = out.stat().st_ino
    assert out.stat().st_ino == before_inode
    assert list(out.iterdir()) == []
    assert list(tmp_path.glob(".race.staging-*")) == []


def test_python_module_entry_point_csv_and_no_csv_option(demo_output, tmp_path):
    out = tmp_path / "subprocess"
    source = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "synthetic_market",
            "simulate",
            "--csv",
            str(demo_output / "input.csv"),
            "--out",
            str(out),
            "--paths",
            "3",
            "--horizon",
            "2",
            "--no-csv",
            "--seed",
            "9",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Wrote simulate artifacts" in completed.stdout
    assert (out / "scenarios.npz").exists()
    assert (out / "input.csv").exists()
    assert not (out / "scenarios.csv").exists()
    assert not (out / "validation.json").exists()


def test_optional_validation_does_not_change_scenario_rng(demo_output, tmp_path):
    base = [
        "simulate",
        "--csv",
        str(demo_output / "input.csv"),
        "--paths",
        "3",
        "--horizon",
        "4",
        "--seed",
        "123",
        "--no-csv",
    ]
    before, after = tmp_path / "without-validation", tmp_path / "with-validation"
    assert cli.main(base + ["--out", str(before)]) == 0
    dates = read_csv(demo_output / "input.csv").close.index
    assert cli.main(base + ["--out", str(after), "--validate-split", str(dates[399].date())]) == 0
    with np.load(before / "scenarios.npz", allow_pickle=False) as first:
        with np.load(after / "scenarios.npz", allow_pickle=False) as second:
            for name in first.files:
                np.testing.assert_array_equal(first[name], second[name])


def test_validate_command_never_generates_future_scenarios(demo_output, tmp_path):
    data = read_csv(demo_output / "input.csv")
    out = tmp_path / "validation-only"
    assert (
        cli.main(
            [
                "validate",
                "--csv",
                str(demo_output / "input.csv"),
                "--out",
                str(out),
                "--split-date",
                str(data.close.index[399].date()),
                "--energy-pairs",
                "200",
            ]
        )
        == 0
    )
    assert (out / "validation.json").exists()
    assert not (out / "scenarios.npz").exists()
    assert not (out / "model.json").exists()


@pytest.mark.parametrize(
    "weights",
    [
        "DEMO_EQUITY=0.5,DEMO_BONDS=0.5",  # Missing asset.
        "DEMO_EQUITY=1,DEMO_BONDS=1,DEMO_GOLD=1",  # Budget.
        "DEMO_EQUITY=-1,DEMO_BONDS=1,DEMO_GOLD=1",  # Short.
        "DEMO_EQUITY=nan,DEMO_BONDS=0.5,DEMO_GOLD=0.5",
        "DEMO_EQUITY=0.2,DEMO_EQUITY=0.3,DEMO_BONDS=0.3,DEMO_GOLD=0.2",
    ],
)
def test_invalid_weights_are_rejected_before_publication(demo_output, tmp_path, weights):
    out = tmp_path / "bad-weights"
    code = cli.main(
        [
            "simulate",
            "--csv",
            str(demo_output / "input.csv"),
            "--out",
            str(out),
            "--weights",
            weights,
        ]
    )
    assert code == 2
    assert not out.exists()


def test_download_requires_dates_and_saves_provider_snapshot(tmp_path, monkeypatch):
    calls = []
    data = make_demo_data(20)
    data.metadata["price_basis"] = "adjusted"

    def provider(tickers, start, end, *, alignment):
        calls.append((tickers, start, end, alignment))
        return data

    monkeypatch.setattr(cli, "download_yahoo", provider)
    out = tmp_path / "download"
    assert (
        cli.main(
            [
                "download",
                "--tickers",
                "SPY",
                "TLT",
                "GLD",
                "--start",
                "2020-01-01",
                "--end",
                "2022-01-01",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert calls == [(["SPY", "TLT", "GLD"], "2020-01-01", "2022-01-01", "strict")]
    assert {p.name for p in out.iterdir()} == {
        "input.csv",
        "input_metadata.json",
        "manifest.json",
        "summary.md",
    }
    assert read_csv(out / "input.csv", price_basis="adjusted").fingerprint() == data.fingerprint()


def test_missing_dates_do_not_call_yahoo(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(cli, "download_yahoo", forbidden)
    out = tmp_path / "missing-dates"
    assert cli.main(["simulate", "--tickers", "SPY", "--out", str(out)]) == 2
    assert "explicit --start and --end" in capsys.readouterr().err
    assert not out.exists()


def test_one_asset_close_only_csv_produces_no_invented_ohlc(tmp_path):
    from synthetic_market.data import MarketData

    data = make_demo_data(500)
    close_only = MarketData(data.close.iloc[:, :1])
    source = tmp_path / "close.csv"
    write_csv(close_only, source)
    out = tmp_path / "close-only"
    assert (
        cli.main(
            [
                "simulate",
                "--csv",
                str(source),
                "--out",
                str(out),
                "--paths",
                "2",
                "--horizon",
                "2",
            ]
        )
        == 0
    )
    with np.load(out / "scenarios.npz", allow_pickle=False) as archive:
        assert archive["close"].shape == (2, 2, 1)
        assert {"open", "high", "low"}.isdisjoint(archive.files)


def _coverage_report(observations=251, model_five_percent_breaches=31):
    """Adversarial fixture: better average scores coexist with poor tail coverage."""
    scores = ("crps_mean", "energy_score_mean", "variogram_score_mean")
    report = {
        "training": {"end": "2021-12-31"},
        "test": {"observations": observations},
        "metrics_model": dict.fromkeys(scores, 0.9),
        "metrics_baseline": dict.fromkeys(scores, 1.0),
        "portfolio": {"model": {}, "baseline": {}},
        "warnings": [
            "Rare-event conclusions are sample-limited; tail counts are not a certification.",
            "No independent-binomial p-values or dependence-robust confidence intervals are claimed.",
        ],
    }
    for label, counts in (("model", (model_five_percent_breaches, 8)), ("baseline", (18, 3))):
        for alpha, observed in zip((0.05, 0.01), counts, strict=True):
            report["portfolio"][label][str(alpha)] = {
                "alpha": alpha,
                "observations": observations,
                "violations": observed,
                "expected_violations_nominal": observations * alpha,
                "violation_rate": observed / observations,
            }
    return report


def test_summary_exposes_2022_style_undercoverage_despite_better_average_scores():
    text = cli._summary_markdown(
        SimpleNamespace(command="validate"),
        make_demo_data(2),
        None,
        None,
        _coverage_report(),
        None,
    )
    assert "| Filtered model | 5% | 31 | 251 | 12.55 | 2.47× | 12.35% |" in text
    assert "| Filtered model | 1% | 8 | 251 | 2.51 | 3.19× | 3.19% |" in text
    assert "| Unfiltered joint bootstrap | 5% | 18 | 251 | 12.55 | 1.43× | 7.17% |" in text
    assert "| Unfiltered joint bootstrap | 1% | 3 | 251 | 2.51 | 1.20× | 1.20% |" in text
    assert (
        "Material undercoverage in this sample; do not treat as calibrated VaR. "
        "Descriptive flag, not a statistical test."
    ) in text
    assert "Rare-event conclusions are sample-limited" in text
    assert "No independent-binomial p-values" in text
    assert "a better average score does not establish calibrated tail risk" in text


def test_summary_material_undercoverage_flag_requires_ten_nominal_expected_breaches():
    text = cli._summary_markdown(
        SimpleNamespace(command="validate"),
        make_demo_data(2),
        None,
        None,
        _coverage_report(observations=100, model_five_percent_breaches=20),
        None,
    )
    assert "| Filtered model | 5% | 20 | 100 | 5.00 | 4.00× | 20.00% |" in text
    assert "Material undercoverage in this sample" not in text
    assert "Rare-event conclusions are sample-limited" in text
