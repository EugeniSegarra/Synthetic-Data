"""Numerical/causal referee; scalar oracles deliberately do not reuse features."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from synthetic_market.data import MarketData
from synthetic_market.path_validation import (
    DelayedMixture,
    PathConfig,
    analogue_paths,
    block_mean_interval,
    evaluate_paths,
    first_passage_cdf,
    issue_forecasts,
    kernel_path_score,
    main,
    path_embedding,
    path_features,
    recovery_cdf,
    signed_area,
    stationary_indices,
    summarize_paths,
)


def scalar_features(r):
    price = [1.0]
    for x in r:
        price.append(price[-1] * np.exp(x))
    dd = []
    for t in range(1, len(price)):
        dd.append(max(1 - price[t] / price[j] for j in range(t + 1)))
    longest = run = 0
    for value in dd:
        run = run + 1 if value > 1e-12 else 0
        longest = max(longest, run)
    return dict(
        terminal_return=price[-1] - 1,
        realized_volatility=np.sqrt(252 * sum(float(x * x) for x in r) / len(r)),
        max_drawdown=max(dd),
        underwater_fraction=sum(x > 1e-12 for x in dd) / len(dd),
        longest_underwater=longest,
        drawdown_area=sum(dd) / len(dd),
    )


@pytest.mark.parametrize("seed", range(12))
def test_scalar_oracle(seed):
    r = np.random.default_rng(seed).normal(0, 0.03, (5, 29))
    result = path_features(r)
    for i, path in enumerate(r):
        for name, value in scalar_features(path).items():
            assert result[name][i] == pytest.approx(value, abs=1e-12)


def test_origin_is_part_of_high_water_mark():
    actual = path_features([np.log(0.7)])
    assert actual["max_drawdown"][0] == pytest.approx(0.3)
    assert actual["longest_underwater"][0] == 1


def test_same_endpoint_same_returns_different_drawdown():
    a = np.array([0.1, -0.1, 0.1, -0.1])
    b = np.array([0.1, 0.1, -0.1, -0.1])
    for key in ("terminal_return", "realized_volatility"):
        assert path_features(a)[key] == pytest.approx(path_features(b)[key])
    assert path_features(a)["max_drawdown"][0] < path_features(b)["max_drawdown"][0]
    assert not np.allclose(path_embedding(a, 0.1), path_embedding(b, 0.1))
    paths = np.repeat(a[None, :], 50, axis=0)
    assert kernel_path_score(paths, a, 0.1) == pytest.approx(0)
    assert kernel_path_score(paths, b, 0.1) > 0.01


@pytest.mark.parametrize("level", [0.05, 0.1, 0.2])
def test_barrier_equality_and_different_definitions(level):
    r = np.diff(np.log([1.0, 1.2, 1.2 * (1 - level)]))
    assert first_passage_cdf(r, level)[-1] == 1
    assert first_passage_cdf(r, level, "origin_loss")[-1] == 0
    assert first_passage_cdf([np.log1p(-level)], level)[0] == 1
    assert first_passage_cdf([np.log1p(-level + 1e-8)], level)[0] == 0


def test_recovery_censoring():
    r = np.array([[0.02, 0.02], [-0.02, 0.01]])
    assert np.array_equal(recovery_cdf(r, 0.1), [0, 0])
    assert np.array_equal(recovery_cdf(r, 0), [1, 1])
    assert recovery_cdf(r, 0.04)[-1] == 0.5
    assert path_features(np.array([-0.05, -0.02]))["longest_underwater"][0] == 2


@pytest.mark.parametrize("bad", [-1, 0, 1, 2, np.nan, np.inf, True])
def test_bad_barrier(bad):
    with pytest.raises(ValueError):
        first_passage_cdf([0.01], bad)


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[[1]]]])
def test_bad_returns(bad):
    with pytest.raises(ValueError):
        path_features(bad)


def test_signed_area_independent_midpoint_integral():
    r = np.array([0.01, -0.03, 0.02, 0.04])
    sigma = 0.02
    h = len(r)
    time = np.arange(h + 1) / h
    channels = [
        np.r_[0, r.cumsum()] / (sigma * np.sqrt(h)),
        np.r_[0, (r * r).cumsum()] / (sigma * sigma * h),
    ]
    oracle = []
    for v in channels:
        integral = 0
        for i in range(h):
            integral += 0.5 * (
                (time[i] + time[i + 1]) / 2 * (v[i + 1] - v[i])
                - (v[i] + v[i + 1]) / 2 * (time[i + 1] - time[i])
            )
        oracle.append(integral)
    assert signed_area(r, sigma)[0] == pytest.approx(oracle)


def test_time_area_sees_order_not_just_total_variation():
    a = np.array([0.04, 0.01, 0.01, 0.01])
    b = a[::-1]
    assert np.sum(a * a) == pytest.approx(np.sum(b * b))
    assert signed_area(a, 0.02)[0, 1] == pytest.approx(-signed_area(b, 0.02)[0, 1])


def test_kernel_against_exact_pair_oracle():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, (25, 8))
    y = rng.normal(0, 0.01, 8)
    ex = path_embedding(x, 0.01)
    ey = path_embedding(y, 0.01)
    dxx = cdist(ex, ex, "sqeuclidean")
    dxy = cdist(ex, ey, "sqeuclidean")
    vals = []
    for b in (0.5, 1, 2):
        k = np.exp(-dxx / (2 * b * b))
        vals.append(
            0.5 * (k.sum() - np.trace(k)) / (25 * 24) - np.exp(-dxy / (2 * b * b)).mean() + 0.5
        )
    assert kernel_path_score(x, y, 0.01, seed=4, pairs=150000) == pytest.approx(
        np.mean(vals), abs=0.0015
    )


def test_correct_oracle_beats_collapsed_and_overdispersed_on_average():
    rng = np.random.default_rng(12)
    y = rng.normal(0, 0.02, (80, 12))
    base = rng.normal(0, 0.02, (150, 12))
    scores = []
    for mult in (0, 1, 4):
        scores.append(
            np.mean([kernel_path_score(base * mult, a, 0.02, seed=8, pairs=1000) for a in y])
        )
    assert scores[1] < scores[0] and scores[1] < scores[2]


def test_stationary_blocks_and_iid_marginals():
    rows = stationary_indices(80, 2000, 30, 10, 20)
    continuation = (rows[:, 1:] == (rows[:, :-1] + 1) % 80).mean()
    assert continuation == pytest.approx(0.9 + 0.1 / 80, abs=0.01)
    assert rows.mean() == pytest.approx(39.5, abs=0.7)
    iid = stationary_indices(80, 2000, 30, 1, 20)
    assert (iid[:, 1:] == (iid[:, :-1] + 1) % 80).mean() == pytest.approx(1 / 80, abs=0.005)


def test_analogue_donors_complete_and_units_invariant():
    r = np.random.default_rng(7).normal(0, 0.01, 800)
    a, meta = analogue_paths(r, 63, 100, 13, neighbors=50)
    b, _ = analogue_paths(r, 63, 100, 13, neighbors=50)
    assert np.array_equal(a, b)
    assert meta["maximum_donor_outcome_index"] <= len(r) - 1
    assert 1 <= meta["effective_weighted_neighbors"] <= 50.00001
    assert a.shape == (100, 63)


def test_analogue_is_centered_in_expectation():
    r = np.random.default_rng(3).normal(0.002, 0.01, 600)
    draws, _ = analogue_paths(r, 21, 100000, 12, neighbors=50)
    assert abs(draws.sum(axis=1).mean()) < 0.001


def test_delay_does_not_leak_precomputed_future_losses():
    a, b = DelayedMixture(), DelayedMixture()
    assert np.allclose(a.weights(100), [1 / 3] * 3)
    b.weights(100)
    a.enqueue(226, [0, 1, 1])
    b.enqueue(226, [1, 0, 1])
    assert np.array_equal(a.weights(225), b.weights(225))
    assert not np.array_equal(a.weights(226), b.weights(226))
    assert a.last_matured == 226 and a.updates == 1
    with pytest.raises(ValueError):
        a.weights(225)
    with pytest.raises(ValueError):
        a.enqueue(220, [1, 1, 1])


def test_constant_bootstrap_and_positive_scale_affinity():
    assert block_mean_interval(np.ones(100), 5, 100, 3)["lower95"] == 1
    r = np.random.default_rng(4).normal(size=100)
    a = block_mean_interval(r, 5, 100, 3)
    b = block_mean_interval(r * 2 + 1, 5, 100, 3)
    for key in ("mean", "lower95", "upper95"):
        assert b[key] == pytest.approx(2 * a[key] + 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"paths": True},
        {"paths": 0},
        {"seed": -1},
        {"stride": 1.5},
        {"horizons": (0,)},
        {"horizons": (63, 21)},
        {"train_window": 20},
        {"mix_forgetting": 1.1},
    ],
)
def test_config_rejections(kwargs):
    with pytest.raises(ValueError):
        PathConfig(**kwargs)


def fixture_data(n=680):
    rng = np.random.default_rng(44)
    closes = 100 * np.exp(np.r_[0, rng.normal(0, 0.01, n - 1).cumsum()])
    return MarketData(
        pd.DataFrame({"SPY": closes}, index=pd.bdate_range("2010-01-01", periods=n)),
        metadata={"price_basis": "adjusted", "source": "explicit iid test fixture"},
    )


def test_future_mutation_cannot_change_issued_forecast():
    data = fixture_data()
    cfg = PathConfig(
        horizons=(5, 21), train_window=500, paths=50, neighbors=50, pairs=32, bootstrap_reps=20
    )
    origin = 550
    a, _ = issue_forecasts(MarketData(data.close.iloc[50:551], metadata=data.metadata), cfg, origin)
    modified = data.close.copy()
    modified.iloc[551:] *= 1000
    b, _ = issue_forecasts(MarketData(modified.iloc[50:551], metadata=data.metadata), cfg, origin)
    for key in a:
        assert np.array_equal(a[key], b[key])


def test_full_evaluation_horizon_dates_and_prefix_invariance():
    data = fixture_data(600)
    cfg = PathConfig(
        horizons=(5, 21),
        train_window=500,
        stride=21,
        paths=50,
        neighbors=50,
        pairs=32,
        bootstrap_reps=20,
    )
    f, m = evaluate_paths(data, cfg, "2010-01-01")
    assert m["requested_origins"] == 4
    assert f.groupby(["origin", "horizon"]).method.nunique().min() == 6
    one = f.iloc[0]
    i = int(one.origin_index)
    h = int(one.horizon)
    realized = np.log(
        data.close.iloc[i + 1 : i + h + 1, 0].to_numpy() / data.close.iloc[i : i + h, 0].to_numpy()
    )
    assert one.terminal_return_actual == pytest.approx(np.expm1(realized.sum()))
    assert one.target_end == str(data.close.index[i + h].date())
    for audit in m["audit"]:
        assert audit["train_end"] == audit["origin"]
        assert (
            audit["last_matured_index"] is None
            or audit["last_matured_index"] <= audit["origin_index"]
        )
    summary = summarize_paths(f, cfg, "2010-01-01")
    assert len(summary["aggregate"]) == 12
    json.dumps(summary, allow_nan=False)


def test_monotone_first_passage_and_mc_resolution():
    r = np.random.default_rng(6).normal(0, 0.02, (200, 30))
    a, b = first_passage_cdf(r, 0.1), first_passage_cdf(r, 0.2)
    assert np.all(np.diff(a) >= 0) and np.all(a >= b)
    assert np.all((a >= 0) & (a <= 1))


def test_output_transaction_and_csv_roundtrip(tmp_path):
    data = fixture_data(580)
    csv = tmp_path / "input.csv"
    data.close.to_csv(csv, index_label="Date")
    out = tmp_path / "results"
    args = [
        "--csv",
        str(csv),
        "--price-basis",
        "adjusted",
        "--out",
        str(out),
        "--start",
        "2010-01-01",
        "--evaluation-start",
        "2010-01-01",
        "--train-window",
        "500",
        "--horizons",
        "5",
        "21",
        "--paths",
        "50",
        "--pairs",
        "32",
        "--bootstrap-reps",
        "20",
    ]
    assert main(args) == 0
    assert (out / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        main(args)
    manifest = json.loads((out / "manifest.json").read_text())
    import hashlib

    for name, digest in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_mc_zero_hits_do_not_imply_certainty():
    from synthetic_market.path_validation import binomial_mc_interval

    lo, hi = binomial_mc_interval(0, 2000)
    assert lo == 0
    assert 0.0018 < hi < 0.0020
    lo2, hi2 = binomial_mc_interval(2000, 2000)
    assert hi2 == 1 and lo2 == pytest.approx(1 - hi)
    with pytest.raises(ValueError):
        binomial_mc_interval(2, 1)


def test_recovery_restricted_mean_keeps_censored_trajectories():
    from synthetic_market.path_validation import _score_forecast

    # First path recovers at t=1, second at t=3, third never recovers by H=4.
    paths = np.array([[0.2, 0, 0, 0], [0.02, 0.02, 0.1, 0], [-0.02, 0, 0, 0]])
    cfg = PathConfig(horizons=(4,), train_window=500, pairs=16)
    row = _score_forecast(paths, paths[-1], 0.02, 0.1, cfg, 500, 4)
    assert row["recovery_restricted_mean"] == pytest.approx((1 + 3 + 4) / 3)
    assert row["recovery_restricted_actual"] == 4
    assert row["recovery_actual"] == 0 and row["recovery_right_censored"]
    not_eligible = _score_forecast(paths, paths[-1], 0.02, 0, cfg, 500, 4)
    assert not_eligible["recovery_restricted_mean"] is None


def test_full_evaluation_future_values_cannot_change_unmatured_weights():
    data = fixture_data(620)
    cfg = PathConfig(
        horizons=(21,),
        train_window=500,
        stride=7,
        paths=50,
        neighbors=50,
        pairs=16,
        bootstrap_reps=10,
    )
    original, audit = evaluate_paths(data, cfg, "2010-01-01")
    altered = data.close.copy()
    # Same information through issuance at index 528. Change every later price.
    altered.iloc[529:] *= np.exp(np.linspace(0.2, 1.0, len(altered) - 529))[:, None]
    changed, audit2 = evaluate_paths(MarketData(altered, metadata=data.metadata), cfg, "2010-01-01")
    for a, b in zip(audit["audit"], audit2["audit"], strict=True):
        if a["origin_index"] <= 528:
            assert a["weights"] == b["weights"]
    keys = ["origin_index", "horizon", "method", "max_drawdown_q95", "terminal_return_q50"]
    pd.testing.assert_frame_equal(
        original.loc[original.origin_index <= 528, keys].reset_index(drop=True),
        changed.loc[changed.origin_index <= 528, keys].reset_index(drop=True),
    )


def test_calibration_failure_is_visible_and_programming_error_is_not_swallowed(monkeypatch):
    import synthetic_market.path_validation as module
    from synthetic_market.model import CalibrationError

    data = fixture_data(600)
    cfg = PathConfig(
        horizons=(21,), train_window=500, paths=50, neighbors=50, pairs=16, bootstrap_reps=10
    )

    def rejected(*args, **kwargs):
        raise CalibrationError("deliberate rejected fit")

    monkeypatch.setattr(module, "fit_market", rejected)
    frame, audit = evaluate_paths(data, cfg, "2010-01-01")
    assert audit["requested_origins"] > 0 and audit["complete_origins"] == 0
    assert set(frame.method) == {"iid", "block", "analog", "analog_area"}
    assert not frame.complete_expert_set.any()
    assert "deliberate rejected fit" in audit["audit"][0]["status"]["fhs"]["error"]
    with pytest.raises(ValueError, match="no complete"):
        summarize_paths(frame, cfg, "2010-01-01")

    def programming_error(*args, **kwargs):
        raise ValueError("unexpected implementation failure")

    monkeypatch.setattr(module, "fit_market", programming_error)
    with pytest.raises(ValueError, match="unexpected implementation"):
        issue_forecasts(data, cfg, 500)


def test_local_rng_isolation():
    data = fixture_data(550)
    cfg = PathConfig(horizons=(21,), train_window=500, paths=50, neighbors=50)
    np.random.seed(1234)
    before = np.random.get_state()
    issue_forecasts(data, cfg, 550)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_provenance_mismatch_rejected_before_output(tmp_path):
    data = fixture_data(580)
    csv, metadata, output = tmp_path / "input.csv", tmp_path / "metadata.json", tmp_path / "out"
    data.close.to_csv(csv, index_label="Date")
    metadata.write_text(json.dumps({"price_basis": "raw"}))
    with pytest.raises(ValueError, match="price-basis"):
        main(
            [
                "--csv",
                str(csv),
                "--metadata",
                str(metadata),
                "--price-basis",
                "adjusted",
                "--out",
                str(output),
            ]
        )
    assert not output.exists()


def test_input_mutation_during_run_refuses_publication(tmp_path, monkeypatch):
    import synthetic_market.path_validation as module

    data = fixture_data(580)
    csv, output = tmp_path / "input.csv", tmp_path / "out"
    data.close.to_csv(csv, index_label="Date")
    real_evaluate = module.evaluate_paths

    def mutate_after_evaluation(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        csv.write_text(csv.read_text() + "\n")
        return result

    monkeypatch.setattr(module, "evaluate_paths", mutate_after_evaluation)
    with pytest.raises(RuntimeError, match="changed during run"):
        main(
            [
                "--csv",
                str(csv),
                "--price-basis",
                "adjusted",
                "--out",
                str(output),
                "--start",
                "2010-01-01",
                "--evaluation-start",
                "2010-01-01",
                "--train-window",
                "500",
                "--horizons",
                "5",
                "21",
                "--paths",
                "50",
                "--pairs",
                "16",
                "--bootstrap-reps",
                "10",
            ]
        )
    assert not output.exists()


def test_incomplete_or_duplicate_summary_rows_are_rejected():
    data = fixture_data(580)
    cfg = PathConfig(
        horizons=(21,), train_window=500, paths=50, neighbors=50, pairs=16, bootstrap_reps=10
    )
    frame, _ = evaluate_paths(data, cfg, "2010-01-01")
    with pytest.raises(ValueError, match="available methods"):
        summarize_paths(frame.iloc[1:], cfg, "2010-01-01")
    with pytest.raises(ValueError, match="duplicate"):
        summarize_paths(pd.concat([frame, frame.iloc[:1]]), cfg, "2010-01-01")


def test_forward_drawdown_peak_and_trough_provenance():
    data = fixture_data(580)
    cfg = PathConfig(
        horizons=(21,), train_window=500, paths=50, neighbors=50, pairs=16, bootstrap_reps=10
    )
    frame, _ = evaluate_paths(data, cfg, "2010-01-01")
    for row in frame.loc[frame.method == "fhs"].itertuples():
        a = data.close.loc[row.max_drawdown_peak_date, "SPY"]
        b = data.close.loc[row.max_drawdown_trough_date, "SPY"]
        assert 1 - b / a == pytest.approx(row.max_drawdown_actual)
        assert (
            row.origin
            <= row.max_drawdown_peak_date
            <= row.max_drawdown_trough_date
            <= row.target_end
        )
