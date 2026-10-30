"""Independent adversarial checks of scientific contracts, not live-provider uptime.

The hand-built fitted objects deliberately isolate simulation from estimation. The
integration fixture then exercises the actual arch calibration on known data.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from synthetic_market.data import MarketData, make_demo_data
from synthetic_market.model import CalibrationError, FittedMarket, ModelConfig, fit_market


def _controlled_market():
    # Every pair of twenty equiprobable shocks occurs once: exactly independent
    # finite population, including in its tails. Marginal moments are 0 and 1.
    levels = np.arange(20, dtype=float)
    levels = (levels - levels.mean()) / levels.std(ddof=0)
    shocks = np.asarray(list(itertools.product(levels, repeat=2)))
    dates = pd.bdate_range("2019-01-01", periods=len(shocks) + 1)
    close = pd.DataFrame({"A": 100.0, "B": 400.0}, index=dates)
    data = MarketData(close, metadata={"price_basis": "synthetic"})
    residuals = pd.DataFrame(shocks, index=dates[1:], columns=["A", "B"])
    sigma = pd.DataFrame(0.01, index=dates[1:], columns=["A", "B"])
    return FittedMarket(
        data=data,
        config=ModelConfig(),
        residuals=residuals,
        means=np.zeros(2),
        omega=np.full(2, 0.0001),
        alpha=np.zeros(2),
        gamma=np.zeros(2),
        beta=np.zeros(2),
        last_variance=np.full(2, 0.0001),
        last_residual=np.zeros(2),
        historical_sigma=sigma,
        fit_details={},
    )


def test_independent_heavy_or_light_marginals_do_not_create_joint_tail_dependence():
    model = _controlled_market()
    # A strictly increasing nonlinear transformation changes marginal tail shape
    # while preserving independence of the complete Cartesian product.
    residuals = np.sign(model.residuals) * np.abs(model.residuals) ** 3
    model.residuals = residuals / residuals.std(ddof=0)
    result = model.simulate(n_paths=200_000, horizon=1, seed=73471, include_ohlc=False)
    z = result.log_returns[:, 0] / result.sigma[:, 0]
    unique = np.sort(model.residuals["A"].unique())
    threshold = (unique[0] + unique[1]) / 2
    both = np.logical_and(z[:, 0] < threshold, z[:, 1] < threshold)
    true_joint_probability = 0.05**2
    standard_error = np.sqrt(true_joint_probability * (1 - true_joint_probability) / len(z))
    assert abs(both.mean() - true_joint_probability) < 5 * standard_error


def test_perfect_joint_tail_dependence_is_not_destroyed():
    model = _controlled_market()
    model.residuals["B"] = model.residuals["A"]
    result = model.simulate(n_paths=300, horizon=8, seed=17, include_ohlc=False)
    np.testing.assert_array_equal(result.log_returns[:, :, 0], result.log_returns[:, :, 1])


def test_first_step_uses_signed_last_residual_with_valid_negative_gamma():
    model = _controlled_market()
    model.omega = np.full(2, 0.1)
    model.alpha = np.full(2, 0.3)
    model.gamma = np.full(2, -0.2)
    model.beta = np.full(2, 0.4)
    model.last_variance = np.full(2, 4.0)
    model.last_residual = np.array([-2.0, 2.0])
    result = model.simulate(n_paths=9, horizon=1, seed=18, include_ohlc=False)
    # Hand calculation: 0.1 + (0.3 - 0.2) * 4 + 0.4 * 4 = 2.1;
    # positive residual: 0.1 + 0.3 * 4 + 0.4 * 4 = 2.9.
    np.testing.assert_allclose(result.sigma[:, 0] ** 2, np.tile([2.1, 2.9], (9, 1)))
    assert np.all(result.close[:, 0] != model.data.close.iloc[-1].to_numpy())


def test_asymmetric_empirical_persistence_cannot_be_replaced_by_gamma_over_two(monkeypatch):
    import synthetic_market.model as implementation

    # The standardized pool [-sqrt(3), 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    # has negative second moment 3/4, not 1/2. The same fitted coefficients
    # yield persistence 1.00 empirically versus 0.95 under false symmetry.
    series = pd.Series(np.tile([-0.03, 0.01, 0.01, 0.01], 100))
    fit = SimpleNamespace(
        convergence_flag=0,
        bic=1.0,
        params=pd.Series({"omega": 0.01, "alpha[1]": 0.1, "gamma[1]": 0.2, "beta[1]": 0.75}),
        conditional_volatility=np.ones(len(series)),
    )
    monkeypatch.setattr(
        implementation,
        "arch_model",
        lambda *args, **kwargs: SimpleNamespace(fit=lambda **options: fit),
    )
    with pytest.raises(CalibrationError, match="empirical-innovation persistence"):
        implementation._fit_asset(series, "SKEWED", ModelConfig(volatility="gjr"))


def test_tiny_negative_gjr_shock_coefficient_is_rejected_not_excused_by_tolerance(monkeypatch):
    import synthetic_market.model as implementation

    # These coefficients pass a tolerance of -1e-12 on alpha+gamma and have
    # positive empirical persistence, yet the next variance is negative for
    # the final observed residual -0.03. Positivity is a sign constraint.
    series = pd.Series(np.tile([0.01, 0.01, 0.01, -0.03], 100))
    fit = SimpleNamespace(
        convergence_flag=0,
        bic=1.0,
        params=pd.Series(
            {
                "omega": 1e-16,
                "alpha[1]": 0.0,
                "gamma[1]": -1e-13,
                "beta[1]": 1e-13,
            }
        ),
        conditional_volatility=np.ones(len(series)),
    )
    monkeypatch.setattr(
        implementation,
        "arch_model",
        lambda *args, **kwargs: SimpleNamespace(fit=lambda **options: fit),
    )
    with pytest.raises(CalibrationError, match="positive variances"):
        implementation._fit_asset(series, "NEGATIVE_COEFFICIENT", ModelConfig(volatility="gjr"))


def test_every_ohlc_component_uses_the_identical_historical_day_as_return():
    model = _controlled_market()
    rows = np.arange(len(model.residuals), dtype=float)
    templates = np.empty((len(rows), 2, 3))
    templates[:, :, 0] = ((rows % 7 - 3) * 0.01)[:, None]
    templates[:, :, 1] = (0.02 + rows * 1e-6)[:, None]
    templates[:, :, 2] = (0.03 + rows * 2e-6)[:, None]
    model.ohlc_templates = templates
    result = model.simulate(n_paths=12, horizon=9, seed=37)
    prior = np.concatenate(
        [np.broadcast_to(model.data.close.iloc[-1].to_numpy(), (12, 1, 2)), result.close[:, :-1]],
        axis=1,
    )
    drawn = templates[result.sampled_rows]
    np.testing.assert_allclose(np.log(result.open / prior) / result.sigma, drawn[:, :, :, 0])
    np.testing.assert_allclose(
        np.log(result.high / np.maximum(result.open, result.close)) / result.sigma,
        drawn[:, :, :, 1],
    )
    np.testing.assert_allclose(
        np.log(np.minimum(result.open, result.close) / result.low) / result.sigma,
        drawn[:, :, :, 2],
    )
    gap = np.log(result.open / prior)
    body = np.log(result.close / result.open)
    np.testing.assert_allclose(gap + body, result.log_returns, atol=2e-15)


def test_ohlc_selection_diagnostic_calls_and_global_rng_do_not_change_close_paths():
    model = _controlled_market()
    model.ohlc_templates = np.full((len(model.residuals), 2, 3), 0.1)
    np.random.seed(123)
    state = np.random.get_state()
    first = model.simulate(n_paths=8, horizon=7, seed=11, include_ohlc=True)
    model.simulate(n_paths=19, horizon=5, seed=987, include_ohlc=False)
    second = model.simulate(n_paths=8, horizon=7, seed=11, include_ohlc=False)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    np.testing.assert_array_equal(first.close, second.close)
    np.testing.assert_array_equal(first.sampled_rows, second.sampled_rows)


@pytest.fixture(scope="module")
def calibrated_demo():
    data = make_demo_data(1000, seed=1729)
    return data, fit_market(data)


def test_fitting_copies_inputs_and_asset_permutation_preserves_economics(calibrated_demo):
    data, original = calibrated_demo
    expected = original.simulate(n_paths=8, horizon=15, seed=707)
    reordered_assets = list(reversed(data.assets))
    reordered_close = data.close[reordered_assets]
    order = pd.MultiIndex.from_product(
        [data.close.index, reordered_assets], names=["Date", "Asset"]
    )
    copied = MarketData(reordered_close, data.ohlc.reindex(order), deepcopy(data.metadata))
    reordered = fit_market(copied)
    copied.close.iloc[-1, :] *= 123
    actual = reordered.simulate(n_paths=8, horizon=15, seed=707)
    np.testing.assert_allclose(actual.close[:, :, ::-1], expected.close, rtol=1e-10)
    np.testing.assert_allclose(actual.open[:, :, ::-1], expected.open, rtol=1e-10)
    np.testing.assert_array_equal(actual.sampled_rows, expected.sampled_rows)


def test_actual_calibration_and_ohlc_are_invariant_to_large_unit_changes(calibrated_demo):
    data, original = calibrated_demo
    units = np.array([1e5, 1e-4, 13.0])
    scaled_close = data.close * units
    scaled_ohlc = data.ohlc.copy()
    for j, asset in enumerate(data.assets):
        scaled_ohlc.loc[(slice(None), asset), :] *= units[j]
    scaled = fit_market(MarketData(scaled_close, scaled_ohlc, data.metadata))
    first = original.simulate(n_paths=10, horizon=20, seed=314)
    second = scaled.simulate(n_paths=10, horizon=20, seed=314)
    for name in ["close", "open", "high", "low"]:
        np.testing.assert_allclose(getattr(second, name) / units, getattr(first, name), rtol=2e-6)
    np.testing.assert_allclose(second.log_returns, first.log_returns, atol=1e-7)
    # A tolerance on the full price can conceal a units bug in small candle
    # wicks. Compare the dimensionless excursions themselves as well.
    first_upper = np.log(first.high / np.maximum(first.open, first.close))
    second_upper = np.log(second.high / np.maximum(second.open, second.close))
    first_lower = np.log(np.minimum(first.open, first.close) / first.low)
    second_lower = np.log(np.minimum(second.open, second.close) / second.low)
    np.testing.assert_allclose(second_upper, first_upper, rtol=2e-5, atol=1e-11)
    np.testing.assert_allclose(second_lower, first_lower, rtol=2e-5, atol=1e-11)
