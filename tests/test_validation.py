"""Formula, chronology, state-update, and reproducibility tests for evaluation."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from synthetic_market import validation
from synthetic_market.data import MarketData, make_demo_data
from synthetic_market.model import ModelConfig
from synthetic_market.validation import (
    crps_empirical,
    empirical_tail_statistics,
    energy_score,
    evaluate_holdout,
    variogram_score,
)


def test_crps_matches_pairwise_definition_and_known_example():
    samples = np.array([0.0, 1.0, 3.0])
    expected = np.mean(np.abs(samples - 2.0)) - 0.5 * np.mean(
        np.abs(samples[:, None] - samples[None, :])
    )
    assert crps_empirical(samples, 2.0) == pytest.approx(2.0 / 3.0)
    assert crps_empirical(samples, 2.0) == pytest.approx(expected)
    matrix = np.column_stack([samples, samples * 2 + 7])
    assert crps_empirical(matrix, [2.0, 11.0]) == pytest.approx([expected, 2 * expected])
    assert crps_empirical(np.full(5, 8.0), 3.0) == 5.0


def test_energy_complete_pairs_equal_definition_and_univariate_crps():
    samples = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0]])
    observation = np.array([2.0, 1.0])
    left, right = np.indices((len(samples), len(samples)))
    pairs = (left.ravel(), right.ravel())
    expected = np.linalg.norm(samples - observation, axis=1).mean() - 0.5 * np.mean(
        np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)
    )
    assert energy_score(samples, observation, pairs) == pytest.approx(expected)
    assert energy_score(samples[:, 0], 2.0, pairs) == pytest.approx(
        crps_empirical(samples[:, 0], 2.0)
    )


def test_variogram_known_value_and_single_asset():
    assert variogram_score([[0.0, 0.0], [1.0, 2.0]], [2.0, 1.0]) == pytest.approx(0.25)
    assert variogram_score(np.array([1.0, 2.0]), 3.0) is None


def test_fractional_tail_mass_and_ties_are_exact():
    result = empirical_tail_statistics([-4.0, -2.0, 0.0, 2.0], alpha=0.375)
    assert result["quantile_return"] == -2.0
    assert result["expected_shortfall_return"] == pytest.approx(-10.0 / 3.0)
    assert result["es_loss"] == pytest.approx(10.0 / 3.0)
    assert result["strict_tail_probability"] == 0.25
    assert result["quantile_boundary_probability"] == 0.25
    assert (
        empirical_tail_statistics([-4.0, -2.0, 0.0, 2.0], 0.1)["expected_shortfall_return"] == -4.0
    )
    assert empirical_tail_statistics([-5.0, -5.0, 1.0], 0.5)["expected_shortfall_return"] == -5.0
    positive = empirical_tail_statistics([0.01, 0.02], 0.1)
    assert positive["var_loss"] == -0.01  # Do not silently floor negative VaR at zero.


@pytest.mark.parametrize(
    "call",
    [
        lambda: crps_empirical([], 0.0),
        lambda: crps_empirical([1.0, np.nan], 0.0),
        lambda: crps_empirical([[1.0, 2.0]], [1.0]),
        lambda: energy_score([1.0, 2.0], 0.0, ([0], [2])),
        lambda: energy_score([1.0, 2.0], 0.0, ([0.1], [1])),
        lambda: variogram_score([[1.0, 2.0]], [0.0, 0.0], order=0),
        lambda: variogram_score([[1.0, 2.0]], [0.0, 0.0], order="0.5"),
        lambda: empirical_tail_statistics([1.0, 2.0], alpha=0),
        lambda: empirical_tail_statistics([1.0, 2.0], alpha=1),
        lambda: empirical_tail_statistics([1.0, 2.0], alpha=True),
        lambda: empirical_tail_statistics([1.0, 2.0], alpha="0.1"),
        lambda: empirical_tail_statistics([np.inf, 2.0], alpha=0.1),
    ],
)
def test_invalid_metric_inputs_fail(call):
    with pytest.raises(ValueError):
        call()


@pytest.fixture
def controlled_data():
    rng = np.random.default_rng(1701)
    returns = rng.normal(0, 0.01, size=(15, 2))
    prices = 100 * np.exp(np.vstack([np.zeros((1, 2)), np.cumsum(returns, axis=0)]))
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return MarketData(pd.DataFrame(prices, index=dates, columns=["A", "B"]))


@pytest.fixture
def controlled_fit(monkeypatch):
    fitted_histories = []

    def fit(training, config=None):
        fitted_histories.append(training.close.copy(deep=True))
        returns = np.log(training.close / training.close.shift(1)).iloc[1:]
        innovations = (returns - returns.mean()) / returns.std(ddof=0)
        dimensions = returns.shape[1]
        result = SimpleNamespace(
            data=training,
            assets=training.assets,
            residuals=innovations,
            means=np.full(dimensions, 0.0003),
            omega=np.full(dimensions, 0.0001),
            alpha=np.full(dimensions, 0.1),
            gamma=np.full(dimensions, 0.2),
            beta=np.full(dimensions, 0.5),
            last_variance=np.full(dimensions, 0.0002),
            last_residual=np.full(dimensions, 0.02),
        )
        result.summary = lambda: {"data_sha256": training.fingerprint()}
        return result

    monkeypatch.setattr(validation, "fit_market", fit)
    return fitted_histories


def test_holdout_has_no_leakage_and_updates_only_after_observation(controlled_data, controlled_fit):
    split = controlled_data.close.index[10]
    initial = evaluate_holdout(controlled_data, split, seed=5, energy_pairs=64)
    changed_close = controlled_data.close.copy()
    changed_close.iloc[11:] *= np.array([1.15, 0.91])
    changed = evaluate_holdout(MarketData(changed_close), split, seed=5, energy_pairs=64)
    assert len(controlled_fit) == 2
    for fitted_history in controlled_fit:
        assert fitted_history.index[-1] == split
        assert len(fitted_history) == 11
    assert initial["training"]["data_sha256"] == changed["training"]["data_sha256"]
    first = initial["per_observation"][0]
    changed_first = changed["per_observation"][0]
    for field in ["conditional_variance", "mean", "marginal_quantiles"]:
        assert first["model"][field] == changed_first["model"][field]
    for alpha in ["0.05", "0.01"]:
        assert (
            first["model"]["portfolio"][alpha]["quantile_return"]
            == changed_first["model"]["portfolio"][alpha]["quantile_return"]
        )
    assert (
        initial["per_observation"][1]["model"]["conditional_variance"]
        != changed["per_observation"][1]["model"]["conditional_variance"]
    )
    assert initial["test"]["observations"] == 5
    assert first["actual_log_returns"]["A"] == pytest.approx(
        np.log(controlled_data.close.A.iloc[11] / controlled_data.close.A.iloc[10])
    )
    h1 = 0.0001 + 0.1 * 0.02**2 + 0.5 * 0.0002
    assert first["model"]["conditional_variance"]["A"] == pytest.approx(h1)
    e1 = first["actual_log_returns"]["A"] - 0.0003
    h2 = 0.0001 + (0.1 + 0.2 * (e1 < 0)) * e1**2 + 0.5 * h1
    assert initial["per_observation"][1]["model"]["conditional_variance"]["A"] == (
        pytest.approx(h2)
    )


def test_holdout_scores_use_all_rows_and_common_mean(controlled_data, controlled_fit):
    split = controlled_data.close.index[10]
    report = evaluate_holdout(
        controlled_data, split, weights={"B": 0.75, "A": 0.25}, seed=5, energy_pairs=64
    )
    train = controlled_data.close.iloc[:11]
    raw = np.log(train / train.shift(1)).iloc[1:].to_numpy()
    baseline = raw - raw.mean(axis=0) + 0.0003
    record = report["per_observation"][0]
    actual = np.array(list(record["actual_log_returns"].values()))
    assert record["baseline"]["crps"] == pytest.approx(
        dict(zip(["A", "B"], crps_empirical(baseline, actual), strict=True))
    )
    portfolio = np.expm1(baseline) @ np.array([0.25, 0.75])
    assert record["baseline"]["portfolio"]["0.05"]["quantile_return"] == pytest.approx(
        np.quantile(portfolio, 0.05, method="inverted_cdf")
    )
    assert record["actual_portfolio_return"] == pytest.approx(
        np.expm1(actual) @ np.array([0.25, 0.75])
    )
    assert report["weights"] == {"A": 0.25, "B": 0.75}
    assert report["portfolio"]["model"]["0.01"]["expected_violations_nominal"] == 0.05
    assert "conditional_variance" not in record["baseline"]
    assert len(report["coexceedances"]["model"]["0.05"]) == 1
    assert report["coexceedances"]["model"]["0.05"][0]["assets"] == ["A", "B"]


def test_rng_is_local_repeatable_and_does_not_change_forecasts(controlled_data, controlled_fit):
    split = controlled_data.close.index[10]
    before_data = controlled_data.fingerprint()
    np.random.seed(724)
    state_before = np.random.get_state()
    first = evaluate_holdout(controlled_data, split, seed=4, energy_pairs=31)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    second = evaluate_holdout(controlled_data, split, seed=4, energy_pairs=31)
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(
        second, allow_nan=False, sort_keys=True
    )
    different_pairs = evaluate_holdout(controlled_data, split, seed=19, energy_pairs=7)
    assert first["metrics_model"]["crps_mean"] == different_pairs["metrics_model"]["crps_mean"]
    assert first["portfolio"] == different_pairs["portfolio"]
    assert first["coexceedances"] == different_pairs["coexceedances"]
    assert controlled_data.fingerprint() == before_data


def test_one_asset_variogram_and_coexceedance_are_not_applicable(controlled_data, controlled_fit):
    data = MarketData(controlled_data.close[["A"]])
    report = evaluate_holdout(data, data.close.index[10], energy_pairs=11)
    assert report["metrics_model"]["variogram_score_mean"] is None
    assert report["per_observation"][0]["model"]["variogram_score"] is None
    assert report["coexceedances"]["model"]["0.01"] == []


def test_joint_tail_probability_uses_joint_rows_not_independent_marginals(controlled_fit):
    rng = np.random.default_rng(51)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=101)))
    dates = pd.date_range("2020-01-01", periods=101)
    data = MarketData(pd.DataFrame({"A": closes, "B": 2 * closes}, index=dates))
    report = evaluate_holdout(data, dates[90], energy_pairs=10)
    joint = report["coexceedances"]["model"]["0.05"][0]
    # The inverse CDF's fifth order statistic leaves four of 90 rows strictly
    # below it. Perfectly coincident ranks imply joint probability 4/90, not its square.
    assert joint["mean_predicted_joint_probability"] == pytest.approx(4 / 90)
    assert joint["expected_joint_violations"] == pytest.approx(10 * 4 / 90)


def test_descriptive_undercoverage_flag_for_31_of_251_observations(controlled_fit):
    rng = np.random.default_rng(812)
    train_returns = rng.normal(0.0, 0.005, size=(300, 2))
    test_returns = np.zeros((251, 2))
    test_returns[:31] = -0.10
    returns = np.vstack([train_returns, test_returns])
    closes = 100 * np.exp(np.vstack([np.zeros((1, 2)), np.cumsum(returns, axis=0)]))
    dates = pd.date_range("2020-01-01", periods=len(closes))
    data = MarketData(pd.DataFrame(closes, index=dates, columns=["A", "B"]))
    report = evaluate_holdout(data, dates[300], energy_pairs=7)
    coverage = report["portfolio"]["baseline"]["0.05"]
    assert coverage["violations"] == 31
    assert coverage["observations"] == 251
    assert coverage["observed_to_nominal_ratio"] == pytest.approx(31 / 12.55)
    flag = next(
        entry
        for entry in report["diagnostic_flags"]
        if entry["method"] == "baseline" and entry["alpha"] == 0.05
    )
    assert flag == {
        "severity": "warning",
        "rule": "descriptive_undercoverage",
        "method": "baseline",
        "alpha": 0.05,
        "ratio": pytest.approx(2.470119522),
        "expectedn": 12.55,
        "observedn": 31,
    }
    assert any("Material undercoverage" in item for item in report["warnings"])
    assert all(entry["alpha"] != 0.01 for entry in report["diagnostic_flags"])


def test_undercoverage_heuristic_does_not_claim_significance_for_small_counts():
    nominal_ratio, empirical_ratio, flag = validation._coverage_diagnostic(
        "model", 0.01, 5, 2.51, 0.0
    )
    assert nominal_ratio == pytest.approx(5 / 2.51)
    assert empirical_ratio is None
    assert flag is None
    assert validation._coverage_diagnostic("model", 0.05, 25, 12.55, 12.0)[2] is None
    assert validation._coverage_diagnostic("model", 0.05, 20, 10.0, 9.9)[2] is not None


@pytest.mark.parametrize(
    "weights",
    [
        {"A": 1.0},
        {"A": 0.5, "B": 0.5, "C": 0.0},
        {"A": 1.1, "B": -0.1},
        {"A": 0.2, "B": 0.2},
        {"A": np.nan, "B": 0.0},
        {"A": True, "B": False},
        {"A": "0.5", "B": "0.5"},
    ],
)
def test_invalid_weights_fail_before_fitting(controlled_data, controlled_fit, weights):
    with pytest.raises(ValueError, match="weights"):
        evaluate_holdout(controlled_data, controlled_data.close.index[10], weights=weights)
    assert not controlled_fit


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"energy_pairs": 0}])
def test_invalid_evaluation_settings(controlled_data, controlled_fit, kwargs):
    with pytest.raises(ValueError):
        evaluate_holdout(controlled_data, controlled_data.close.index[10], **kwargs)


def test_no_test_period_and_invalid_split_fail(controlled_data, controlled_fit):
    with pytest.raises(ValueError, match="No observations"):
        evaluate_holdout(controlled_data, controlled_data.close.index[-1])
    with pytest.raises(ValueError, match="split_date"):
        evaluate_holdout(controlled_data, pd.NaT)


def test_real_fit_integration_is_json_serializable_and_keeps_training_boundary():
    data = make_demo_data(n_obs=420, seed=1729)
    split = data.close.index[409]
    report = evaluate_holdout(
        data, split, config=ModelConfig(volatility="garch"), seed=42, energy_pairs=128
    )
    assert report["training"]["return_observations"] == 409
    assert report["test"]["observations"] == 10
    assert report["training"]["data_sha256"] == data.subset(end=split).fingerprint()
    assert report["model_summary"]["training_end"] == split.date().isoformat()
    assert np.isfinite(report["metrics_model"]["crps_mean"])
    assert np.isfinite(report["metrics_baseline"]["energy_score_mean"])
    json.dumps(report, allow_nan=False)
