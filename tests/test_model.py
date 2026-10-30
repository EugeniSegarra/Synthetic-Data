"""Regression and numerical-contract tests, independent of market downloads."""

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_market.data import MarketData, make_demo_data
from synthetic_market.model import (
    CalibrationError,
    FittedMarket,
    ModelConfig,
    SimulationError,
    fit_market,
    next_variance,
)


@pytest.fixture(scope="module")
def fitted():
    return fit_market(make_demo_data(n_obs=1000, seed=42))


def manual_model():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    data = MarketData(pd.DataFrame({"A": [98.0, 99.0, 100.0]}, index=dates))
    return FittedMarket(
        data,
        ModelConfig(),
        pd.DataFrame({"A": [-1.0, 1.0]}, index=dates[1:]),
        means=np.array([0.001]),
        omega=np.array([0.01]),
        alpha=np.array([0.1]),
        gamma=np.array([0.2]),
        beta=np.array([0.5]),
        last_variance=np.array([0.04]),
        last_residual=np.array([-0.1]),
        historical_sigma=pd.DataFrame({"A": [0.2, 0.2]}, index=dates[1:]),
        fit_details={},
    )


def test_gjr_exact_sign_and_lag():
    assert next_variance(0.01, 0.1, 0.2, 0.5, 0.04, 0.1) == pytest.approx(0.031)
    assert next_variance(0.01, 0.1, 0.2, 0.5, 0.04, -0.1) == pytest.approx(0.033)


def test_simulation_matches_manual_recurrence_all_steps():
    m = manual_model()
    result = m.simulate(n_paths=4, horizon=7, seed=7)
    for path in range(4):
        variance, residual, price = 0.04, -0.1, 100.0
        for step in range(7):
            variance = 0.01 + (0.1 + 0.2 * (residual < 0)) * residual**2 + 0.5 * variance
            z = [-1.0, 1.0][result.sampled_rows[path, step]]
            residual = np.sqrt(variance) * z
            price *= np.exp(0.001 + residual)
            assert result.sigma[path, step, 0] ** 2 == pytest.approx(variance)
            assert result.close[path, step, 0] == pytest.approx(price)
    assert result.close.shape == (4, 7, 1)
    assert not np.any(result.close[:, 0, 0] == 100.0)


def test_one_future_observation_is_a_new_move():
    result = manual_model().simulate(n_paths=1, horizon=1, seed=0)
    assert result.close[0, 0, 0] != 100
    assert result.to_frame().index.names == ["Path", "Step", "Asset"]
    assert result.to_frame().index.get_level_values("Step").tolist() == [1]


def test_joint_rows_and_ohlc_templates_are_identical(fitted):
    r = fitted.simulate(n_paths=20, horizon=15, seed=71)
    expected_z = fitted.residuals.to_numpy()[r.sampled_rows]
    np.testing.assert_allclose((r.log_returns - fitted.means) / r.sigma, expected_z, atol=1e-12)
    previous = np.concatenate(
        [
            np.broadcast_to(fitted.data.close.iloc[-1].to_numpy(), (20, 1, 3)),
            r.close[:, :-1],
        ],
        axis=1,
    )
    template = fitted.ohlc_templates[r.sampled_rows]
    np.testing.assert_allclose(
        np.log(r.open / previous) / r.sigma, template[:, :, :, 0], atol=1e-11
    )
    np.testing.assert_allclose(
        np.log(r.high / np.maximum(r.open, r.close)) / r.sigma, template[:, :, :, 1], atol=1e-11
    )
    np.testing.assert_allclose(
        np.log(np.minimum(r.open, r.close) / r.low) / r.sigma, template[:, :, :, 2], atol=1e-11
    )
    np.testing.assert_allclose(
        np.log(r.open / previous) + np.log(r.close / r.open), r.log_returns, atol=1e-14
    )
    assert np.all(r.high >= np.maximum(r.open, r.close))
    assert np.all(r.low <= np.minimum(r.open, r.close))
    assert np.all(r.low > 0)


def test_every_bootstrap_innovation_has_the_required_moments(fitted):
    z = fitted.residuals.to_numpy()
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose((z * z).mean(axis=0), 1, atol=1e-14)
    empirical = fitted.beta + fitted.alpha + fitted.gamma * np.mean(z * z * (z < 0), axis=0)
    assert np.all(empirical < fitted.config.persistence_limit)
    for asset, value in zip(fitted.assets, empirical):
        assert fitted.fit_details[asset]["empirical_persistence"] == pytest.approx(value)


def test_reproducible_and_does_not_change_global_rng_or_fitted_state(fitted):
    np.random.seed(41)
    state = np.random.get_state()
    initial_variance = fitted.last_variance.copy()
    a = fitted.simulate(10, 7, seed=12)
    fitted.simulate(15, 3, seed=999)
    fitted.summary()
    b = fitted.simulate(10, 7, seed=12)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    np.testing.assert_array_equal(a.close, b.close)
    np.testing.assert_array_equal(a.open, b.open)
    np.testing.assert_array_equal(initial_variance, fitted.last_variance)
    assert not np.array_equal(a.close, fitted.simulate(10, 7, seed=13).close)


def test_price_scale_equivariance_including_ohlc(fitted):
    data = fitted.data
    larger = MarketData(data.close * 10, data.ohlc * 10, data.metadata)
    m2 = fit_market(larger, fitted.config)
    a, b = fitted.simulate(3, 8, seed=2), m2.simulate(3, 8, seed=2)
    # Optimisation roundoff may affect fitted coefficients, but not nominal units.
    for field in ["open", "high", "low", "close"]:
        np.testing.assert_allclose(getattr(b, field), getattr(a, field) * 10, rtol=2e-5)


def test_constant_asset_is_an_explicit_error_not_silent_universe_change():
    data = make_demo_data(n_obs=500)
    close = data.close.copy()
    close["FLAT"] = 100.0
    with pytest.raises(CalibrationError, match="FLAT.*constant"):
        fit_market(MarketData(close))


def test_univariate_fit_and_close_only_output():
    data = make_demo_data(n_obs=600)
    single = MarketData(data.close.iloc[:, :1], metadata=data.metadata)
    m = fit_market(single, ModelConfig(volatility="garch"))
    result = m.simulate(4, 3)
    assert result.close.shape == (4, 3, 1)
    assert result.open is None
    assert "Open" not in result.to_frame()


def test_fit_copies_caller_data(fitted):
    data = make_demo_data(n_obs=600)
    m = fit_market(data, ModelConfig(volatility="garch"))
    expected = m.simulate(3, 3).close
    data.close.iloc[-1] *= 4
    np.testing.assert_array_equal(m.simulate(3, 3).close, expected)


@pytest.mark.parametrize(
    "n_paths,horizon,seed",
    [(0, 1, 1), (1, 0, 1), (1.5, 1, 1), (True, 2, 1), (1, 2, -1), (1, 2, 1.1)],
)
def test_invalid_simulation_arguments(n_paths, horizon, seed):
    with pytest.raises(ValueError):
        manual_model().simulate(n_paths, horizon, seed)


def test_overflow_fails_loudly_instead_of_clipping():
    m = manual_model()
    m.means[0] = 1000
    with pytest.raises(SimulationError, match="not clipped"):
        m.simulate(1, 1)


def test_model_summary_is_finite_json(fitted):
    serialized = json.dumps(fitted.summary(), allow_nan=False)
    assert "empirical_persistence" in serialized
    assert "data_sha256" in serialized


def test_no_artificial_extreme_dependence_when_resampling_independent_rows():
    # A known, independent heavy-tail population, deliberately identical to the
    # original copula counterexample; the empirical joint tail is preserved.
    rng = np.random.default_rng(424242)
    z = rng.standard_t(5, size=(20000, 2))
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    dates = pd.bdate_range("1900-01-01", periods=len(z) + 1)
    data = MarketData(pd.DataFrame(100.0, index=dates, columns=["A", "B"]))
    unit = np.ones(2)
    zero = np.zeros(2)
    m = FittedMarket(
        data,
        ModelConfig(),
        pd.DataFrame(z, index=dates[1:], columns=data.assets),
        means=zero,
        omega=unit,
        alpha=zero,
        gamma=zero,
        beta=zero,
        last_variance=unit,
        last_residual=zero,
        historical_sigma=pd.DataFrame(1.0, index=dates[1:], columns=data.assets),
        fit_details={},
    )
    samples = m.simulate(n_paths=200000, horizon=1, seed=1030).log_returns[:, 0]
    cut = np.quantile(z, 0.01, axis=0)
    actual_p = np.mean((z[:, 0] < cut[0]) & (z[:, 1] < cut[1]))
    observed_p = np.mean((samples[:, 0] < cut[0]) & (samples[:, 1] < cut[1]))
    # Sampling error conditional on the empirical distribution, not a claim
    # that a finite training sample has exactly the population tail probability.
    se = np.sqrt(actual_p * (1 - actual_p) / len(samples))
    assert abs(observed_p - actual_p) <= 5 * se + 1 / len(samples)
    assert observed_p < 0.0004
