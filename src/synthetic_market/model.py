"""Two-stage filtered historical simulation with joint daily OHLC templates.

No parameter in this module is inferred from a validation/future period. Innovations
are sampled as complete cross-asset rows, not independently by column. Empirical
support is finite; this is a conditional risk model, not a return predictor or an
intraday execution simulator.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from .data import MarketData


class CalibrationError(ValueError):
    """A valid, stable volatility model could not be estimated."""


class SimulationError(ValueError):
    """A scenario cannot be represented safely; nothing is silently clipped."""


@dataclass(frozen=True)
class ModelConfig:
    volatility: str = "auto"
    mean: str = "zero"
    min_observations: int = 250
    persistence_limit: float = 0.9999

    def __post_init__(self):
        if self.volatility not in {"auto", "garch", "gjr"}:
            raise ValueError("volatility must be auto, garch or gjr")
        if self.mean not in {"zero", "constant"}:
            raise ValueError("mean must be zero or constant (daily log-return mean)")
        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, int)
            or self.min_observations < 100
        ):
            raise ValueError("min_observations must be an integer >= 100")
        if not np.isfinite(self.persistence_limit) or not 0 < self.persistence_limit < 1:
            raise ValueError("persistence_limit must be strictly between zero and one")


def next_variance(omega, alpha, gamma, beta, prev_variance, prev_residual):
    """GJR(1,1) recursion, in decimal log-return variance units.

    For gamma=0 this is GARCH(1,1). Negative gamma is valid only if
    alpha+gamma>=0. Arrays broadcast across independent scenario paths.
    """
    return (
        omega
        + (alpha + gamma * (np.asarray(prev_residual) < 0)) * np.square(prev_residual)
        + beta * prev_variance
    )


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_exp(log_values):
    # Explicit rejection preserves the model: no return winsorisation or price clipping.
    if not np.isfinite(log_values).all() or np.any(np.abs(log_values) > 700):
        raise SimulationError(
            "Scenario exceeds floating-point price range. Reduce horizon or review the "
            "calibration; the path was not clipped."
        )
    return np.exp(log_values)


def _diagnostic_acf(values, lag):
    """Undefined correlations (such as constant squared shocks) are JSON null."""
    left, right = np.asarray(values)[:-lag], np.asarray(values)[lag:]
    if not len(left) or np.std(left) == 0 or np.std(right) == 0:
        return None
    correlation = float(np.corrcoef(left, right)[0, 1])
    return correlation if np.isfinite(correlation) else None


@dataclass
class ScenarioResult:
    assets: tuple[str, ...]
    close: np.ndarray
    log_returns: np.ndarray
    sigma: np.ndarray
    sampled_rows: np.ndarray
    origin: str
    source_dates: tuple[str, ...]
    seed: int
    open: np.ndarray | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        """Long format; Step counts future observations, not fictional exchange dates."""
        n_paths, horizon, n_assets = self.close.shape
        index = pd.MultiIndex.from_product(
            [range(n_paths), range(1, horizon + 1), self.assets],
            names=["Path", "Step", "Asset"],
        )
        fields = {}
        if self.open is not None:
            fields.update(Open=self.open.ravel(), High=self.high.ravel(), Low=self.low.ravel())
        fields.update(
            Close=self.close.ravel(), LogReturn=self.log_returns.ravel(), Sigma=self.sigma.ravel()
        )
        frame = pd.DataFrame(fields, index=index)
        frame["ResampledDate"] = np.repeat(
            np.asarray(self.source_dates)[self.sampled_rows].ravel(), n_assets
        )
        return frame


@dataclass
class FittedMarket:
    data: MarketData
    config: ModelConfig
    residuals: pd.DataFrame
    means: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray
    last_variance: np.ndarray
    last_residual: np.ndarray
    historical_sigma: pd.DataFrame
    fit_details: dict
    ohlc_templates: np.ndarray | None = None

    @property
    def assets(self):
        return tuple(self.residuals.columns)

    def summary(self):
        return {
            "method": "joint_filtered_historical_simulation",
            "config": asdict(self.config),
            "assets": list(self.assets),
            "price_basis": self.data.metadata.get("price_basis", "unspecified"),
            "source": self.data.metadata,
            "training_start": str(self.data.close.index[0].date()),
            "training_end": str(self.data.close.index[-1].date()),
            "training_returns": len(self.residuals),
            "data_sha256": self.data.fingerprint(),
            "fit": self.fit_details,
            "ohlc_available": self.ohlc_templates is not None,
            "limitations": [
                "Finite empirical innovation support; no unseen standardized tail extrapolation.",
                "Frozen cross-asset empirical dependence, conditional on the training period.",
                "Zero log-return drift does not imply zero arithmetic expected return.",
                "OHLC contains no intraday ordering, liquidity, volume or execution model.",
                "Parameter and model uncertainty are not represented by drawing more paths.",
            ],
        }

    def simulate(self, n_paths=1000, horizon=252, seed=42, include_ohlc=True):
        """Draw paths from a fresh local RNG; calls never mutate the fitted state.

        The first returned observation includes one new return. Current prices
        are the initial condition and are not returned as a fake future date.
        """
        _positive_integer(n_paths, "n_paths")
        _positive_integer(horizon, "horizon")
        if not isinstance(include_ohlc, (bool, np.bool_)):
            raise ValueError("include_ohlc must be boolean")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if n_paths * horizon * len(self.assets) > 10_000_000:
            raise ValueError("More than 10 million path-step-asset cells; run smaller batches")
        rng = np.random.default_rng(seed)
        rows = rng.integers(0, len(self.residuals), size=(n_paths, horizon))
        z = self.residuals.to_numpy(dtype=float)
        shape = (n_paths, horizon, len(self.assets))
        close, returns, sigma = (np.empty(shape) for _ in range(3))
        has_ohlc = include_ohlc and self.ohlc_templates is not None
        opened, high, low = (
            tuple(np.empty(shape) for _ in range(3)) if has_ohlc else (None, None, None)
        )
        log_previous = np.broadcast_to(
            np.log(self.data.close.iloc[-1].to_numpy(dtype=float)), (n_paths, len(self.assets))
        ).copy()
        variance = np.broadcast_to(self.last_variance, log_previous.shape).copy()
        residual = np.broadcast_to(self.last_residual, log_previous.shape).copy()
        for step in range(horizon):
            with np.errstate(over="ignore", invalid="ignore"):
                variance = next_variance(
                    self.omega, self.alpha, self.gamma, self.beta, variance, residual
                )
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise SimulationError("Non-finite/non-positive simulated conditional variance")
            sigma_t = np.sqrt(variance)
            residual = sigma_t * z[rows[:, step]]
            ret = self.means + residual
            log_close = log_previous + ret
            close[:, step] = _finite_exp(log_close)
            returns[:, step] = ret
            sigma[:, step] = sigma_t
            if has_ohlc:
                # All assets and all candle components use the SAME historical day.
                template = self.ohlc_templates[rows[:, step]]
                log_open = log_previous + sigma_t * template[:, :, 0]
                log_high = np.maximum(log_open, log_close) + sigma_t * template[:, :, 1]
                log_low = np.minimum(log_open, log_close) - sigma_t * template[:, :, 2]
                opened[:, step] = _finite_exp(log_open)
                high[:, step] = _finite_exp(log_high)
                low[:, step] = _finite_exp(log_low)
            log_previous = log_close
        return ScenarioResult(
            assets=self.assets,
            close=close,
            log_returns=returns,
            sigma=sigma,
            sampled_rows=rows,
            origin=str(self.data.close.index[-1].date()),
            source_dates=tuple(str(x.date()) for x in self.residuals.index),
            seed=int(seed),
            open=opened,
            high=high,
            low=low,
        )


def _fit_asset(series, asset, config):
    if not np.isfinite(series.to_numpy()).all() or series.std(ddof=0) < 1e-10:
        raise CalibrationError(
            f"{asset}: constant or degenerate returns; no asset was silently dropped"
        )
    candidates = ["garch", "gjr"] if config.volatility == "auto" else [config.volatility]
    accepted, rejected = [], {}
    for kind in candidates:
        try:
            model = arch_model(
                series * 100.0,
                mean="Zero" if config.mean == "zero" else "Constant",
                vol="GARCH",
                p=1,
                o=int(kind == "gjr"),
                q=1,
                dist="normal",
                rescale=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(disp="off", show_warning=False, options={"maxiter": 2000})
            if fit.convergence_flag != 0 or not np.isfinite(fit.bic):
                raise CalibrationError(f"optimizer failed: {fit.optimization_result.message}")
            p = fit.params
            mean = float(p.get("mu", 0.0)) / 100.0
            omega = float(p["omega"]) / 10000.0
            alpha, beta = float(p["alpha[1]"]), float(p["beta[1]"])
            gamma = float(p.get("gamma[1]", 0.0))
            if not np.isfinite([mean, omega, alpha, gamma, beta]).all():
                raise CalibrationError("non-finite parameters")
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + gamma < 0:
                raise CalibrationError("GJR coefficients do not guarantee positive variances")
            cond_sigma = np.asarray(fit.conditional_volatility, dtype=float) / 100.0
            raw_residual = series.to_numpy(dtype=float) - mean
            raw_z = raw_residual / cond_sigma
            if not np.isfinite(raw_z).all() or np.any(cond_sigma <= 0):
                raise CalibrationError("non-finite filter or standardized residuals")
            center, scale = float(raw_z.mean()), float(raw_z.std(ddof=0))
            if scale <= 1e-10:
                raise CalibrationError("degenerate empirical innovations")
            z = (raw_z - center) / scale
            neg_second = float(np.mean(z**2 * (z < 0)))
            persistence = beta + alpha * float(np.mean(z**2)) + gamma * neg_second
            if not 0 <= persistence < config.persistence_limit:
                raise CalibrationError(
                    f"empirical-innovation persistence {persistence:.6f} is not below "
                    f"{config.persistence_limit}; no coefficients were clipped"
                )
            details = {
                "selected": kind,
                "bic": float(fit.bic),
                "converged": True,
                "mean_daily_log": mean,
                "omega": omega,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "empirical_persistence": persistence,
                "negative_second_moment": neg_second,
                "raw_standardized_residual_mean": center,
                "raw_standardized_residual_std": scale,
                "residual_centering_and_scaling": "sample mean, population standard deviation",
                "innovation_min": float(z.min()),
                "innovation_max": float(z.max()),
                "warnings": [str(w.message) for w in caught],
                "residual_acf": {str(lag): _diagnostic_acf(z, lag) for lag in [1, 5, 10, 20]},
                "squared_residual_acf": {
                    str(lag): _diagnostic_acf(z**2, lag) for lag in [1, 5, 10, 20]
                },
            }
            accepted.append((float(fit.bic), details, z, cond_sigma, raw_residual))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rejected[kind] = str(exc)
    if not accepted:
        raise CalibrationError(f"{asset}: no admissible model. {rejected}")
    accepted.sort(key=lambda candidate: candidate[0])
    best = accepted[0]
    best[1]["candidate_bic"] = {c[1]["selected"]: c[0] for c in accepted}
    best[1]["rejected_candidates"] = rejected
    return best[1:]


def fit_market(data: MarketData, config: ModelConfig | None = None) -> FittedMarket:
    """Fit to exactly the supplied history; data cleaning/selection must be explicit."""
    config = config or ModelConfig()
    # Copy the input: changing a caller's frame after fitting cannot change scenarios.
    from copy import deepcopy

    data = MarketData(
        data.close.copy(deep=True),
        None if data.ohlc is None else data.ohlc.copy(deep=True),
        deepcopy(data.metadata),
    )
    log_returns = np.log(data.close / data.close.shift(1)).iloc[1:]
    if len(log_returns) < config.min_observations:
        raise CalibrationError(
            f"Need at least {config.min_observations} return observations; got {len(log_returns)}"
        )
    details, zs, sigmas, final_residuals = {}, {}, {}, []
    for asset in data.close.columns:
        d, z, sigma, raw_residual = _fit_asset(log_returns[asset], asset, config)
        details[asset], zs[asset], sigmas[asset] = d, z, sigma
        final_residuals.append(raw_residual[-1])
    residuals = pd.DataFrame(zs, index=log_returns.index)
    historical_sigma = pd.DataFrame(sigmas, index=log_returns.index)
    params = {
        key: np.array([d[key] for d in details.values()])
        for key in ["omega", "alpha", "gamma", "beta"]
    }
    templates = None
    if data.ohlc is not None:
        templates = np.empty((len(log_returns), len(data.close.columns), 3))
        for j, asset in enumerate(data.close.columns):
            bars = data.ohlc.xs(asset, level="Asset").reindex(log_returns.index)
            sigma = historical_sigma[asset].to_numpy()
            prior = data.close[asset].shift(1).reindex(log_returns.index).to_numpy()
            opened, high, low, close = (
                bars[c].to_numpy() for c in ["Open", "High", "Low", "Close"]
            )
            templates[:, j, 0] = np.log(opened / prior) / sigma
            templates[:, j, 1] = np.maximum(np.log(high / np.maximum(opened, close)) / sigma, 0.0)
            templates[:, j, 2] = np.maximum(np.log(np.minimum(opened, close) / low) / sigma, 0.0)
        if not np.isfinite(templates).all():
            raise CalibrationError("OHLC template construction produced non-finite values")
    return FittedMarket(
        data=data,
        config=config,
        residuals=residuals,
        means=np.array([d["mean_daily_log"] for d in details.values()]),
        last_variance=historical_sigma.iloc[-1].to_numpy() ** 2,
        last_residual=np.array(final_residuals),
        historical_sigma=historical_sigma,
        fit_details=details,
        ohlc_templates=templates,
        **params,
    )
