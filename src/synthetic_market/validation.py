"""Chronological, fixed-parameter one-step evaluation of empirical forecasts.

Forecasts precede each test observation. The observed return then updates the
volatility state; parameters and the training innovation distribution stay fixed.
Scores, coverage, and joint-tail counts are diagnostics, not a certification.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .data import MarketData
from .model import ModelConfig, fit_market, next_variance


def _sample_matrix(samples, observation):
    values = np.asarray(samples, dtype=float)
    one_dimension = values.ndim == 1
    if one_dimension:
        values = values[:, None]
    actual = np.atleast_1d(np.asarray(observation, dtype=float))
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("samples must be a nonempty sample-by-variable array")
    if actual.shape != (values.shape[1],):
        raise ValueError("observation must have one value per sample column")
    if not np.isfinite(values).all() or not np.isfinite(actual).all():
        raise ValueError("samples and observation must be finite")
    return values, actual, one_dimension


def crps_empirical(samples, observation):
    """Exact CRPS for an equally weighted empirical distribution.

    Accept a vector of samples or a sample-by-variable matrix. The result is a
    float or one score per variable, respectively. Sorting costs O(n log n) and
    avoids constructing all n-squared pairwise absolute differences.
    """
    values, actual, one_dimension = _sample_matrix(samples, observation)
    size = len(values)
    ordered = np.sort(values, axis=0)
    coefficients = 2.0 * np.arange(1, size + 1) - size - 1.0
    result = (
        np.mean(np.abs(values - actual), axis=0)
        - np.sum(coefficients[:, None] * ordered, axis=0) / size**2
    )
    return float(result[0]) if one_dimension else result


def energy_score(samples, observation, pair_indices):
    """Energy score with an exact first term and sampled independent pairs.

    ``pair_indices`` contains two equal-length integer vectors. Uniform iid
    sampling with replacement gives an unbiased Monte Carlo estimate of the
    second term for the finite empirical distribution. Reuse the same pairs for
    competing models to reduce comparison noise. A noisy estimate can be
    negative; it is not clipped to conceal Monte Carlo error.
    """
    values, actual, _ = _sample_matrix(samples, observation)
    if not isinstance(pair_indices, (tuple, list)) or len(pair_indices) != 2:
        raise ValueError("pair_indices must contain two integer index vectors")
    left, right = (np.asarray(index) for index in pair_indices)
    if (
        left.ndim != 1
        or right.ndim != 1
        or not len(left)
        or left.shape != right.shape
        or not np.issubdtype(left.dtype, np.integer)
        or not np.issubdtype(right.dtype, np.integer)
        or (left < 0).any()
        or (right < 0).any()
        or (left >= len(values)).any()
        or (right >= len(values)).any()
    ):
        raise ValueError("pair_indices must be nonempty, equally sized, and in range")
    first = np.linalg.norm(values - actual, axis=1).mean()
    second = np.linalg.norm(values[left] - values[right], axis=1).mean()
    return float(first - 0.5 * second)


def variogram_score(samples, observation, order=0.5):
    """Mean variogram score over all unordered variable pairs.

    Equal weights sum to one. Returns None for a single variable, where no
    cross-variable pair exists. Chunking bounds the temporary sample/pair array.
    """
    values, actual, _ = _sample_matrix(samples, observation)
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, Real)
        or not np.isfinite(order)
        or not 0 < order < 2
    ):
        raise ValueError("order must be finite and strictly between zero and two")
    if values.shape[1] == 1:
        return None
    left, right = np.triu_indices(values.shape[1], k=1)
    total = 0.0
    for start in range(0, len(left), 256):
        i, j = left[start : start + 256], right[start : start + 256]
        observed = np.abs(actual[i] - actual[j]) ** order
        predicted = np.mean(np.abs(values[:, i] - values[:, j]) ** order, axis=0)
        total += float(np.sum((observed - predicted) ** 2))
    return total / len(left)


def empirical_tail_statistics(samples, alpha=0.05):
    """Lower return quantile and exact empirical expected shortfall.

    ES is the average over the worst alpha probability mass. Fractional mass at
    the boundary is included, including when alpha is below one sample's mass.
    The quantile uses the inverse empirical CDF. All values are returns: losses
    are reported separately as their negatives, without flooring at zero.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("tail samples must be a nonempty, finite vector")
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("alpha must be finite and strictly between zero and one")
    ordered = np.sort(values)
    mass = len(ordered) * float(alpha)
    whole = int(np.floor(mass))
    fraction = mass - whole
    # Divide before summing so a finite convex average need not overflow simply
    # because its unnormalized sum exceeds floating-point range.
    es = float(np.sum(ordered[:whole] / mass))
    if fraction:
        es += (fraction / mass) * float(ordered[whole])
    if not np.isfinite(es):
        raise ValueError("Expected-shortfall calculation exceeded floating-point range")
    quantile = float(np.quantile(ordered, alpha, method="inverted_cdf"))
    return {
        "alpha": float(alpha),
        "quantile_return": quantile,
        "expected_shortfall_return": es,
        "var_loss": -quantile,
        "es_loss": -es,
        "strict_tail_probability": float(np.mean(values < quantile)),
        "quantile_boundary_probability": float(np.mean(values == quantile)),
    }


def _weights(assets, weights):
    if weights is None:
        return np.full(len(assets), 1.0 / len(assets))
    if not isinstance(weights, Mapping) or set(weights) != set(assets):
        raise ValueError("weights must name every asset exactly once, with no extra assets")
    if any(
        isinstance(weights[asset], (bool, np.bool_)) or not isinstance(weights[asset], Real)
        for asset in assets
    ):
        raise ValueError("weights must be real numeric values, not strings or booleans")
    try:
        values = np.array([weights[asset] for asset in assets], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be finite nonnegative numbers summing to one") from exc
    if values.shape != (len(assets),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be finite nonnegative numbers summing to one")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("long-only weights must sum to one")
    return values / total  # Remove roundoff only, after explicitly checking the budget.


def _portfolio_returns(log_returns, weights):
    # A one-period portfolio return is the weighted sum of SIMPLE asset returns.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.expm1(log_returns) @ weights
    if not np.isfinite(result).all():
        raise ValueError("Portfolio returns overflowed; no values were clipped")
    return result


def _named(assets, values):
    return {asset: float(value) for asset, value in zip(assets, values, strict=True)}


def _coverage_diagnostic(method, alpha, observed, expected_nominal, expected_empirical):
    """Describe undercoverage without an independent-Bernoulli significance test.

    At least ten nominal expected breaches and twice the nominal breach count is
    an explicit reporting heuristic, not a statistical critical value. It does
    not estimate, recalibrate, select, or change any predictive distribution.
    An absent flag does not establish calibration, especially in rare tails.
    """
    nominal_ratio = float(observed / expected_nominal)
    empirical_ratio = float(observed / expected_empirical) if expected_empirical > 0 else None
    flag = None
    if expected_nominal >= 10 and nominal_ratio >= 2:
        flag = {
            "severity": "warning",
            "rule": "descriptive_undercoverage",
            "method": method,
            "alpha": float(alpha),
            "ratio": nominal_ratio,
            "expectedn": float(expected_nominal),
            "observedn": int(observed),
        }
    return nominal_ratio, empirical_ratio, flag


def evaluate_holdout(
    data: MarketData,
    split_date,
    config: ModelConfig | None = None,
    weights: dict[str, float] | None = None,
    seed: int = 42,
    energy_pairs: int = 2048,
) -> dict[str, Any]:
    """Evaluate a train-once model sequentially on dates after ``split_date``.

    The inclusive training interval contains all prices at or before the split.
    The first test return includes the last training close. Each forecast uses
    every training residual row, so marginal scores and quantiles have no Monte
    Carlo sampling noise. Only the energy score's second term uses random pairs.

    The baseline jointly resamples raw training log-return vectors, recentered
    to the fitted model's mean. Portfolio weights are long-only, sum to one, and
    refer to each one-period forecast in a common reporting currency. No FX
    conversion, transaction costs, or trade execution is inferred.
    """
    if not isinstance(data, MarketData):
        raise ValueError("data must be MarketData")
    data.validate()
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(energy_pairs, (bool, np.bool_))
        or not isinstance(energy_pairs, (int, np.integer))
        or energy_pairs < 1
    ):
        raise ValueError("energy_pairs must be a positive integer")
    asset_names = data.assets
    portfolio_weights = _weights(asset_names, weights)
    try:
        split = pd.Timestamp(split_date)
        if pd.isna(split):
            raise ValueError("split_date cannot be missing")
        test_mask = data.close.index > split
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "split_date must be a valid timestamp compatible with the data timezone"
        ) from exc
    if not test_mask.any():
        raise ValueError("No observations occur after split_date")
    training = data.subset(end=split)
    model = fit_market(training, config=config)
    if tuple(model.assets) != tuple(asset_names):
        raise ValueError("Fitted asset order differs from the input; no reordering is implicit")
    means = np.asarray(model.means, dtype=float).copy()
    innovations = model.residuals.to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        all_returns = np.log(data.close / data.close.shift(1)).iloc[1:]
        training_returns = np.log(training.close / training.close.shift(1)).iloc[1:]
    test_returns = all_returns.loc[all_returns.index > split]
    raw_training = training_returns.to_numpy(dtype=float)
    actual_returns = test_returns.to_numpy(dtype=float)
    if not np.isfinite(raw_training).all() or not np.isfinite(actual_returns).all():
        raise ValueError("Log-return calculation produced non-finite values")
    if innovations.shape != raw_training.shape:
        raise ValueError("The fitted innovation panel must retain every training return")
    baseline = raw_training - raw_training.mean(axis=0) + means
    size, dimensions = innovations.shape
    rng = np.random.default_rng(seed)
    sampled_pairs = (
        rng.integers(0, size, size=int(energy_pairs)),
        rng.integers(0, size, size=int(energy_pairs)),
    )
    pair_left, pair_right = np.triu_indices(dimensions, k=1)
    alpha_levels = (0.05, 0.01)
    labels = ("model", "baseline")
    accumulators = {
        label: {"crps": np.zeros(dimensions), "energy": 0.0, "variogram": 0.0} for label in labels
    }
    portfolio_totals = {
        label: {
            str(alpha): {
                "violations": 0,
                "expected_strict": 0.0,
                "quantile_sum": 0.0,
                "es_sum": 0.0,
            }
            for alpha in alpha_levels
        }
        for label in labels
    }
    joint_totals = {
        label: {
            str(alpha): {
                "observed": np.zeros(len(pair_left), dtype=np.int64),
                "expected": np.zeros(len(pair_left)),
            }
            for alpha in alpha_levels
        }
        for label in labels
    }
    variance = np.asarray(model.last_variance, dtype=float).copy()
    residual = np.asarray(model.last_residual, dtype=float).copy()
    observations = []
    for date, realized in zip(test_returns.index, actual_returns, strict=True):
        # The state available before seeing this test return determines the forecast.
        with np.errstate(over="ignore", invalid="ignore"):
            forecast_variance = next_variance(
                model.omega, model.alpha, model.gamma, model.beta, variance, residual
            )
            forecast = means + np.sqrt(forecast_variance) * innovations
        if (
            not np.isfinite(forecast_variance).all()
            or np.any(forecast_variance <= 0)
            or not np.isfinite(forecast).all()
        ):
            raise ValueError("Non-finite or nonpositive forecast state; no values were clipped")
        realized_portfolio = float(_portfolio_returns(realized, portfolio_weights))
        record = {
            "date": date.isoformat(),
            "actual_log_returns": _named(asset_names, realized),
            "actual_portfolio_return": realized_portfolio,
        }
        for label, distribution in (("model", forecast), ("baseline", baseline)):
            crps = crps_empirical(distribution, realized)
            energy = energy_score(distribution, realized, sampled_pairs)
            variogram = variogram_score(distribution, realized)
            if (
                not np.isfinite(crps).all()
                or not np.isfinite(energy)
                or (variogram is not None and not np.isfinite(variogram))
            ):
                raise ValueError("A scoring calculation overflowed; no score was clipped")
            accumulators[label]["crps"] += crps
            accumulators[label]["energy"] += energy
            if variogram is not None:
                accumulators[label]["variogram"] += variogram
            forecast_portfolio = _portfolio_returns(distribution, portfolio_weights)
            detail = {
                "mean": _named(asset_names, means),
                "crps": _named(asset_names, crps),
                "energy_score": energy,
                "variogram_score": variogram,
                "portfolio": {},
                "marginal_quantiles": {},
            }
            if label == "model":
                detail["conditional_variance"] = _named(asset_names, forecast_variance)
            for alpha in alpha_levels:
                key = str(alpha)
                tail = empirical_tail_statistics(forecast_portfolio, alpha)
                violation = realized_portfolio < tail["quantile_return"]
                tail["violation"] = bool(violation)
                detail["portfolio"][key] = tail
                totals = portfolio_totals[label][key]
                totals["violations"] += int(violation)
                totals["expected_strict"] += tail["strict_tail_probability"]
                totals["quantile_sum"] += tail["quantile_return"]
                totals["es_sum"] += tail["expected_shortfall_return"]
                thresholds = np.quantile(distribution, alpha, axis=0, method="inverted_cdf")
                detail["marginal_quantiles"][key] = _named(asset_names, thresholds)
                if len(pair_left):
                    sampled_breaches = distribution < thresholds
                    actual_breaches = realized < thresholds
                    joint = joint_totals[label][key]
                    joint["observed"] += actual_breaches[pair_left] & actual_breaches[pair_right]
                    # Chunking prevents an n_samples by all_asset_pairs temporary.
                    for start in range(0, len(pair_left), 256):
                        indices = slice(start, start + 256)
                        joint["expected"][indices] += np.mean(
                            sampled_breaches[:, pair_left[indices]]
                            & sampled_breaches[:, pair_right[indices]],
                            axis=0,
                        )
            record[label] = detail
        observations.append(record)
        variance = np.asarray(forecast_variance, dtype=float).copy()
        residual = realized - means

    count = len(observations)
    metrics = {}
    portfolio_report = {}
    joint_report = {}
    diagnostic_flags = []
    for label in labels:
        totals = accumulators[label]
        mean_crps = totals["crps"] / count
        metrics[label] = {
            "crps_mean": float(mean_crps.mean()),
            "crps_by_asset": _named(asset_names, mean_crps),
            "energy_score_mean": float(totals["energy"] / count),
            "variogram_score_mean": float(totals["variogram"] / count) if dimensions > 1 else None,
        }
        portfolio_report[label] = {}
        joint_report[label] = {}
        for alpha in alpha_levels:
            key = str(alpha)
            totals = portfolio_totals[label][key]
            nominal_ratio, empirical_ratio, flag = _coverage_diagnostic(
                label, alpha, totals["violations"], count * alpha, totals["expected_strict"]
            )
            if flag is not None:
                diagnostic_flags.append(flag)
            portfolio_report[label][key] = {
                "alpha": alpha,
                "observations": count,
                "violations": int(totals["violations"]),
                "violation_rate": float(totals["violations"] / count),
                "coverage": float(1.0 - totals["violations"] / count),
                "nominal_coverage": 1.0 - alpha,
                "expected_violations_nominal": count * alpha,
                "expected_violations_empirical_strict": float(totals["expected_strict"]),
                "observed_to_nominal_ratio": nominal_ratio,
                "observed_to_empirical_expected_ratio": empirical_ratio,
                "mean_quantile_return": float(totals["quantile_sum"] / count),
                "mean_expected_shortfall_return": float(totals["es_sum"] / count),
                "mean_var_loss": float(-totals["quantile_sum"] / count),
                "mean_es_loss": float(-totals["es_sum"] / count),
            }
            joint = joint_totals[label][key]
            joint_report[label][key] = [
                {
                    "assets": [asset_names[i], asset_names[j]],
                    "observations": count,
                    "observed_joint_violations": int(joint["observed"][pair]),
                    "observed_joint_rate": float(joint["observed"][pair] / count),
                    "expected_joint_violations": float(joint["expected"][pair]),
                    "mean_predicted_joint_probability": float(joint["expected"][pair] / count),
                }
                for pair, (i, j) in enumerate(zip(pair_left, pair_right, strict=True))
            ]
    caveats = [
        "Scores and tail counts are diagnostics, not an automatic validation pass.",
        "Parameters and the innovation law are frozen; only the observed volatility state updates.",
        "Energy scores use a Monte Carlo second term; paired draws are shared across models and dates.",
        "No independent-binomial p-values or dependence-robust confidence intervals are claimed.",
        "The baseline shares the model's log-return mean; neither estimates investment alpha.",
        "OHLC, transaction costs, intraday execution, and multi-period paths are not validated here.",
        "Portfolio returns assume all asset returns are already in a common reporting currency.",
    ]
    for flag in diagnostic_flags:
        caveats.append(
            f"Material undercoverage for {flag['method']} at the {flag['alpha']:.0%} "
            f"portfolio tail: {flag['observedn']} observed breaches versus "
            f"{flag['expectedn']:g} nominal expected ({flag['ratio']:.2f}x). "
            "This is a descriptive heuristic, not a statistical test; probabilities "
            "are not certified."
        )
    for alpha in alpha_levels:
        if count * alpha < 20:
            caveats.append(
                f"The {alpha:.0%} tail has only {count * alpha:g} nominal expected breaches "
                f"in {count} test observations; rare-event conclusions are sample-limited."
            )
    return {
        "schema_version": 1,
        "evaluation": "fixed_parameter_sequential_one_step",
        "split_date": split.isoformat(),
        "training": {
            "start": training.close.index[0].isoformat(),
            "end": training.close.index[-1].isoformat(),
            "price_observations": len(training.close),
            "return_observations": len(raw_training),
            "data_sha256": training.fingerprint(),
        },
        "test": {
            "start": test_returns.index[0].isoformat(),
            "end": test_returns.index[-1].isoformat(),
            "observations": count,
        },
        "assets": list(asset_names),
        "weights": _named(asset_names, portfolio_weights),
        "model_summary": model.summary(),
        "metrics_model": metrics["model"],
        "metrics_baseline": metrics["baseline"],
        "portfolio": portfolio_report,
        "coexceedances": joint_report,
        "diagnostic_flags": diagnostic_flags,
        "per_observation": observations,
        "methodology": {
            "seed": int(seed),
            "energy_pairs": int(energy_pairs),
            "marginal_forecast": "All equally weighted training innovation rows; no simulation subsample",
            "energy_second_term": "Uniform iid pairs with replacement; common pairs for model and baseline",
            "variogram_order": 0.5,
            "variogram_weights": "Equal over unordered asset pairs; weights sum to one",
            "scores_units": "Decimal log returns; variogram is squared differences of order 0.5",
            "portfolio_units": "Decimal simple return: expm1(asset log returns) @ weights",
            "quantile_method": "inverted_cdf",
            "tail_rule": "Strict return < empirical quantile; boundary probability is reported separately",
            "expected_shortfall": "Worst alpha empirical probability mass, including fractional boundary mass",
            "coexceedance_rule": "Observed and predictive events use the same model-specific marginal thresholds",
            "undercoverage_flag": (
                "Descriptive heuristic: nominal expected breaches >= 10 and "
                "observed/nominal >= 2; no significance claim and no forecast recalibration"
            ),
            "baseline": "Joint raw training log-return bootstrap, recentered to the fitted model mean",
        },
        "warnings": caveats,
    }
