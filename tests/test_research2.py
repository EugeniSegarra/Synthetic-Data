"""Independent counterexamples for the experimental layer; no empirical victory implied."""

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from synthetic_market.data import MarketData
from synthetic_market.model import ModelConfig, fit_market
from synthetic_market.research2.evaluation import (
    MaturedSelector,
    ResearchConfig,
    evaluate,
    simultaneous_intervals,
    summarize,
)
from synthetic_market.research2.generators import (
    BASELINE,
    CANDIDATES,
    InnovationPool,
    RiskFilter,
    causal_states,
    fit_factorial,
    shrunk_drift,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(33)
    r = rng.standard_normal(650) * 0.008
    dates = pd.bdate_range("2007-01-01", periods=651)
    return MarketData(
        pd.DataFrame({"SPY": 100 * np.exp(np.r_[0, r.cumsum()])}, index=dates),
        metadata={"price_basis": "adjusted"},
    )


@pytest.mark.parametrize("prior", [0, 20, 5040])
def test_drift_exact(prior):
    r = np.array([0.02, -0.01, 0.004])
    assert shrunk_drift(r, prior, 0.04) == pytest.approx(
        (sum(r) + prior * 0.04 / 252) / (3 + prior)
    )
    assert shrunk_drift(-np.abs(r), prior) < 0


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[1, 2]]])
def test_drift_invalid(bad):
    with pytest.raises(ValueError):
        shrunk_drift(bad)


def test_state_is_past_only():
    r = np.random.default_rng(4).normal(0, 0.01, 600)
    s = causal_states(r)
    changed = r.copy()
    changed[400:] = -4
    np.testing.assert_array_equal(s[:401], causal_states(changed)[:401])
    for t in [252, 400, 599]:
        expected = 2 * int(np.mean(r[t - 21 : t] ** 2) > np.mean(r[t - 252 : t] ** 2)) + int(
            r[t - 1] < 0
        )
        assert s[t] == expected


def test_weighted_state_moments():
    r = np.random.default_rng(5).standard_t(5, 900) * 0.01
    z = (r - r.mean()) / r.std()
    pool = InnovationPool.fit(z, r)
    for s in range(4):
        w = np.diff(np.r_[0, pool.cdf[s]])
        transformed = (z - pool.centers[s]) / pool.scales[s]
        assert w.sum() == pytest.approx(1)
        assert np.dot(w, transformed) == pytest.approx(0, abs=1e-12)
        assert np.dot(w, transformed**2) == pytest.approx(1, abs=1e-12)
        assert np.dot(w, transformed**2 * (transformed < 0)) == pytest.approx(
            pool.negative_moments[s]
        )
        assert np.all(w > 0)


@pytest.mark.parametrize("u,s", [([1.0], [0]), ([-0.1], [0]), ([0.1], [4]), ([0.1], [0.5])])
def test_pool_invalid(u, s):
    r = np.random.default_rng(5).normal(size=400)
    pool = InnovationPool.fit(r, r)
    with pytest.raises(ValueError):
        pool.draw(np.array(u), np.array(s))


def test_harch_scalar_reference():
    history = np.linspace(-0.02, 0.025, 300)
    z = np.array([-1.3, -0.7, 0.4, 1.6])
    z = (z - z.mean()) / z.std()
    f = RiskFilter("harch", 0.00001, np.array([0.2, 0.3, 0.1]), z, history, 0.0002, {})
    u = np.random.default_rng(8).random((270, 3))
    returns, sigma = f.simulate(u)
    for path in range(3):
        e = list(history)
        for t in range(270):
            variance = (
                0.00001
                + 0.2 * e[-1] ** 2
                + 0.3 * sum(v * v for v in e[-5:]) / 5
                + 0.1 * sum(v * v for v in e[-22:]) / 22
            )
            shock = np.sqrt(variance) * z[int(u[t, path] * len(z))]
            assert sigma[path, t] ** 2 == pytest.approx(variance, rel=1e-10, abs=1e-15)
            assert returns[path, t] == pytest.approx(shock, rel=1e-10, abs=1e-13)
            e.append(shock)


def test_fhs_scalar_reference():
    history = np.random.default_rng(3).normal(0, 0.01, 300)
    z = np.array([-1.0, 1.0])
    f = RiskFilter("fhs", 0.000005, np.array([0.08, 0.1, 0.8]), z, history, 0.0001, {})
    u = np.random.default_rng(11).random((50, 4))
    r, s = f.simulate(u)
    for path in range(4):
        e = history[-1]
        h = 0.0001
        for t in range(50):
            h = 0.000005 + (0.08 + 0.1 * (e < 0)) * e**2 + 0.8 * h
            e = np.sqrt(h) * z[int(2 * u[t, path])]
            assert r[path, t] == pytest.approx(e, abs=1e-14)


def test_baseline_one_step_and_frozen(data):
    model = fit_factorial(data, 500)
    baseline = fit_market(
        MarketData(data.close.iloc[-501:], metadata=data.metadata), ModelConfig(mean="zero")
    )
    expected = float(
        baseline.omega[0]
        + (baseline.alpha[0] + baseline.gamma[0] * (baseline.last_residual[0] < 0))
        * baseline.last_residual[0] ** 2
        + baseline.beta[0] * baseline.last_variance[0]
    )
    _, sig = model.filters["fhs"].simulate(np.full((1, 5), 0.7))
    np.testing.assert_allclose(sig**2, expected)
    original = data.close.copy()
    model.simulate(50, 20, 5)
    pd.testing.assert_frame_equal(original, data.close)


def test_harch_one_step_library(data):
    model = fit_factorial(data, 500)
    fit = arch_model(
        np.diff(np.log(data.close.iloc[-501:, 0])) * 100,
        mean="Zero",
        vol="HARCH",
        p=[1, 5, 22],
        rescale=False,
    ).fit(disp="off")
    expected = float(fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) / 10000
    _, sig = model.filters["harch"].simulate(np.full((1, 3), 0.2))
    np.testing.assert_allclose(sig**2, expected, rtol=1e-7)


def test_drift_only_and_prefix_invariance(data):
    model = fit_factorial(data, 500)
    a, _ = model.simulate(60, 21, 2)
    b, _ = model.simulate(60, 45, 2)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name][:, :21])
    for v in ("fhs", "harch"):
        for law in ("global", "state"):
            np.testing.assert_allclose(
                a[f"{v}_{law}_shrink"] - a[f"{v}_{law}_zero"], model.drift, rtol=1e-10, atol=1e-15
            )


def test_no_global_rng_mutation(data):
    model = fit_factorial(data, 500)
    np.random.seed(19)
    state = np.random.get_state()
    model.simulate(50, 21, 9)
    current = np.random.get_state()
    np.testing.assert_array_equal(state[1], current[1])
    assert state[2:] == current[2:]


def test_scale_invariance(data):
    model = fit_factorial(data, 500)
    scaled = MarketData(data.close * 100, metadata=data.metadata)
    other = fit_factorial(scaled, 500)
    a, _ = model.simulate(50, 21, 7)
    b, _ = other.simulate(50, 21, 7)
    for method in a:
        np.testing.assert_allclose(a[method], b[method], rtol=0.003, atol=1e-5)


def test_selector_no_premature_feedback():
    selector = MaturedSelector(1, 3)
    assert selector.choose(100) == BASELINE
    losses = np.arange(8, dtype=float)
    losses[-1] = -1
    selector.enqueue(226, losses)
    assert selector.choose(225) == BASELINE
    assert selector.choose(226) == CANDIDATES[-1]
    assert selector.last_maturity == 226
    with pytest.raises(ValueError):
        selector.choose(200)
    with pytest.raises(ValueError):
        selector.enqueue(226, losses)


def test_simultaneous_identical():
    x = np.c_[np.zeros(80), np.full(80, 0.02)]
    mean, lo, hi = simultaneous_intervals(x, 8, 100, 4)
    np.testing.assert_allclose(mean, [0, 0.02], atol=1e-15)
    np.testing.assert_allclose(lo, mean, atol=1e-15)
    np.testing.assert_allclose(hi, mean, atol=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"paths": True},
        {"paths": 49},
        {"train_window": 200},
        {"horizons": (21, 21, 126)},
        {"horizons": (21, 63)},
        {"seed": -2},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ResearchConfig(**kwargs)


def test_evaluator_smoke_and_future_mutation(data):
    c = ResearchConfig(train_window=300, paths=50, stride=126, pairs=20, bootstrap_reps=20)
    f, m = evaluate(data, c, start=str(data.close.index[300].date()))
    assert len(f) > 0
    for record in m:
        assert (
            record["selector_last_maturity"] is None
            or record["selector_last_maturity"] <= record["origin_index"]
        )
    altered = data.close.copy()
    altered.iloc[450:] *= 2
    g, _ = evaluate(
        MarketData(altered, metadata=data.metadata), c, start=str(data.close.index[300].date())
    )
    left = f[(f.origin_index == 300) & (f.horizon == 21)].sort_values("method")
    right = g[(g.origin_index == 300) & (g.horizon == 21)].sort_values("method")
    np.testing.assert_array_equal(left.max_drawdown_q95, right.max_drawdown_q95)
    s = summarize(f, c, evaluation_start="2007-01-01")
    assert s["default_promoted"] is False
    with pytest.raises(ValueError):
        summarize(pd.concat([f, f]), c, "2007-01-01")


def test_pav_against_all_contiguous_blocks():
    from synthetic_market.research2.controls import isotonic_fit

    p = np.arange(5) / 5
    y = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    x, result = isotonic_fit(p, y)
    # For squared error, an independent brute-force search over all 2^(n-1)
    # contiguous partitions finds the same constrained minimum.
    choices = []
    for mask in range(1 << (len(y) - 1)):
        edges = [0] + [i + 1 for i in range(len(y) - 1) if mask & (1 << i)] + [len(y)]
        values = np.concatenate(
            [np.full(b - a, np.mean(y[a:b])) for a, b in zip(edges[:-1], edges[1:])]
        )
        if np.all(np.diff(values) >= 0):
            choices.append((np.sum((values - y) ** 2), values))
    reference = min(choices, key=lambda t: t[0])[1]
    np.testing.assert_allclose(result, reference)
    np.testing.assert_array_equal(x, p)
    x, result = isotonic_fit([0.1, 0.1, 0.3], [1, 0, 1])
    np.testing.assert_allclose(result, [0.5, 1])


def test_recalibration_causality():
    from synthetic_market.research2.controls import delayed_recalibration

    rows = [
        dict(
            method=BASELINE,
            origin_index=i,
            origin=str(pd.Timestamp("2000-01-01") + pd.Timedelta(days=i)),
            target_end="unused",
            horizon=10,
            drawdown_hit10_prob=0.3 + i / 1000,
            drawdown_hit10_actual=int(i % 4 == 0),
        )
        for i in range(100)
    ]
    f = pd.DataFrame(rows)
    original = delayed_recalibration(f, memory=30, minimum=10)
    changed = f.copy()
    changed.loc[changed.origin_index >= 50, "drawdown_hit10_actual"] = 1
    mutated = delayed_recalibration(changed, memory=30, minimum=10)
    np.testing.assert_array_equal(
        original.loc[original.origin_index < 60, "recalibrated_probability"],
        mutated.loc[mutated.origin_index < 60, "recalibrated_probability"],
    )
    assert np.all(
        original.last_matured_index.dropna()
        <= original.origin_index[original.last_matured_index.notna()]
    )


def test_bootstrap_and_stress():
    from synthetic_market.research2.controls import bootstrap_drift, stress_report

    drift = bootstrap_drift(np.full(300, 0.001), reps=20, prior_sessions=200, seed=5)
    np.testing.assert_allclose(drift, 0.0006)
    report = stress_report()
    assert all(v["probability"] is None for v in report.values())
    assert report["instant_20pct_loss"]["max_drawdown"] == pytest.approx(0.2)
    assert report["five_4pct_losses_then_full_recovery"]["terminal_return"] == pytest.approx(
        0, abs=1e-12
    )
    assert report["five_consecutive_4pct_losses"]["max_drawdown"] == pytest.approx(1 - 0.96**5)


@pytest.mark.parametrize("name", ["fit", "simulation"])
def test_failures_not_replaced(data, monkeypatch, name):
    import synthetic_market.research2.generators as module

    if name == "fit":

        def failure(r):
            raise module.CalibrationError("injected failure")

        monkeypatch.setattr(module, "_fit_harch", failure)
        model = module.fit_factorial(data, 500)
        paths, errors = model.simulate(50, 20)
        assert "harch" in errors
        assert not any(k.startswith("harch") for k in paths)
    else:
        model = module.fit_factorial(data, 500)

        def failure(*a, **kw):
            raise module.SimulationError("injected overflow")

        monkeypatch.setattr(model.filters["harch"], "simulate", failure)
        paths, errors = model.simulate(50, 20)
        assert "harch_global" in errors
        assert not any(k.startswith("harch") for k in paths)


def test_bundle_atomicity_and_input_mutation(data, tmp_path, monkeypatch):
    from synthetic_market.data import write_csv
    from synthetic_market.research2 import cli

    path = tmp_path / "input.csv"
    write_csv(data, path)
    out = tmp_path / "issue"
    args = [
        "issue",
        "--csv",
        str(path),
        "--price-basis",
        "adjusted",
        "--train-window",
        "500",
        "--paths",
        "50",
        "--out",
        str(out),
    ]
    assert cli.main(args) == 0
    with pytest.raises(FileExistsError):
        cli.main(args)
    with np.load(out / "forecast.npz", allow_pickle=False) as z:
        assert z["log_returns"].shape == (50, 252)
    original = cli.fit_factorial

    def mutation(*a, **kw):
        model = original(*a, **kw)
        with path.open("a") as f:
            f.write("\n")
        return model

    monkeypatch.setattr(cli, "fit_factorial", mutation)
    args[-1] = str(tmp_path / "mutated")
    with pytest.raises(RuntimeError, match="changed during run"):
        cli.main(args)
    assert not (tmp_path / "mutated").exists()


def test_invalid_price_basis_metadata(data, tmp_path):
    import json

    from synthetic_market.data import write_csv
    from synthetic_market.research2.cli import load

    source = tmp_path / "data.csv"
    write_csv(data, source)
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps({"price_basis": "raw"}))
    with pytest.raises(ValueError, match="price basis"):
        load(source, "adjusted", metadata)


def test_cutoff_rejected(data):
    altered = data.close.copy()
    altered.index = pd.bdate_range("2026-01-01", periods=len(altered))
    with pytest.raises(ValueError, match="cutoff"):
        evaluate(
            MarketData(altered, metadata=data.metadata), ResearchConfig(train_window=300, paths=50)
        )


def _example(name):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "examples" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refilter_uses_actual_not_bootstrap_endpoint():
    module = _example("fhs2_sensitivity")
    history = np.random.default_rng(8).normal(0, 0.01, 600)
    detail = dict(selected="gjr", omega=0.000002, alpha=0.05, gamma=0.05, beta=0.85)
    fixed = module.refilter(history, detail)
    assert fixed.history[-1] == history[-1]
    direct = arch_model(history * 100, mean="Zero", vol="GARCH", p=1, o=1, q=1, rescale=False).fix(
        [0.02, 0.05, 0.05, 0.85]
    )
    assert fixed.last_variance == pytest.approx((direct.conditional_volatility[-1] / 100) ** 2)
    altered = history.copy()
    altered[-1] = -0.04
    other = module.refilter(altered, detail)
    uniforms = np.full((1, 1), 0.2)
    _, sigma = fixed.simulate(uniforms)
    _, changed = other.simulate(uniforms)
    assert sigma[0, 0] != changed[0, 0]
    np.testing.assert_array_equal(fixed.history, history)


def test_refilter_rejects_invalid_model():
    module = _example("fhs2_sensitivity")
    r = np.random.default_rng(3).normal(0, 0.01, 400)
    with pytest.raises(ValueError, match="unsupported"):
        module.refilter(r, dict(selected="harch"))
    with pytest.raises(ValueError):
        module.refilter(r, dict(selected="garch", omega=-1, alpha=0.1, gamma=0, beta=0.5))


def test_executable_referee_detects_tampered_statistics(data, tmp_path):
    import hashlib
    import json

    from synthetic_market.data import write_csv
    from synthetic_market.research2 import cli

    source = tmp_path / "source.csv"
    write_csv(data, source)
    result = tmp_path / "result"
    cli.main(
        [
            "evaluate",
            "--csv",
            str(source),
            "--price-basis",
            "adjusted",
            "--train-window",
            "500",
            "--stride",
            "21",
            "--paths",
            "50",
            "--pairs",
            "50",
            "--bootstrap-reps",
            "20",
            "--evaluation-start",
            "2000-01-01",
            "--out",
            str(result),
        ]
    )
    referee = _example("audit_fhs2")
    assert referee.audit(result, source)["realized_targets_checked"] > 0
    frame = pd.read_csv(result / "forecasts.csv", float_precision="round_trip")
    frame.loc[0, "max_drawdown_actual"] += 0.01
    frame.to_csv(result / "forecasts.csv", index=False, float_format="%.17g")
    manifest = json.loads((result / "manifest.json").read_text())
    manifest["files"]["forecasts.csv"] = hashlib.sha256(
        (result / "forecasts.csv").read_bytes()
    ).hexdigest()
    (result / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outcome oracle"):
        referee.audit(result, source)
