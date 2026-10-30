"""Causal volatility / innovation-shape / drift ablations on daily closes.

HARCH uses squared DAILY returns. It is neither HAR-RV nor Component-GARCH.
State pools have exact weighted zero mean/unit variance: they change shape,
not an undocumented extra mean or scale. Drift is an explicit separate factor.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
from arch import arch_model

from ..data import MarketData
from ..model import CalibrationError, ModelConfig, SimulationError, fit_market, next_variance

CANDIDATES = tuple(
    f"{v}_{p}_{d}" for v, p, d in product(("fhs", "harch"), ("global", "state"), ("zero", "shrink"))
)
BASELINE = "fhs_global_zero"
LAGS = (1, 5, 22)


def _positive(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _series(values, minimum=1):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError(f"need a finite one-dimensional history of length >= {minimum}")
    return x


def shrunk_drift(history, prior_sessions=5040, prior_annual_log=0.0):
    """Expanding log mean with an explicit pseudo-sample, no positivity clipping.

    This is a regularized mean assumption, not an estimated excess-return premium.
    The caller supplies exactly the prefix available at issuance.
    """
    r = _series(history)
    _positive(prior_sessions, "prior_sessions", 0)
    if not np.isfinite(prior_annual_log):
        raise ValueError("prior_annual_log must be finite")
    mu = (r.sum() + prior_sessions * prior_annual_log / 252) / (len(r) + prior_sessions)
    if not np.isfinite(mu):
        raise ValueError("drift calculation overflowed")
    return float(mu)


def causal_states(history):
    """State attached to shock t uses returns through t-1 only; -1 is unavailable."""
    r = _series(history, 253)
    cs = np.r_[0.0, np.cumsum(r * r)]
    t = np.arange(252, len(r))
    short = (cs[t] - cs[t - 21]) / 21
    long = (cs[t] - cs[t - 252]) / 252
    result = np.full(len(r), -1, dtype=int)
    result[t] = 2 * (short > long).astype(int) + (r[t - 1] < 0).astype(int)
    return result


@dataclass
class InnovationPool:
    z: np.ndarray
    cdf: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    negative_moments: np.ndarray
    counts: np.ndarray
    effective_sizes: np.ndarray

    @classmethod
    def fit(cls, residuals, history, pseudocount=500.0):
        z, r = _series(residuals, 253), _series(history, 253)
        if len(z) != len(r) or not np.isfinite(pseudocount) or pseudocount <= 0:
            raise ValueError("invalid pool history or pseudocount")
        states = causal_states(r)
        weights = np.empty((4, len(z)))
        counts = np.array([(states == s).sum() for s in range(4)])
        for s, n in enumerate(counts):
            local = n / (n + pseudocount)
            weights[s] = (1 - local) / len(z)
            if n:
                weights[s, states == s] += local / n
        centers = weights @ z
        centered = z[None, :] - centers[:, None]
        scales = np.sqrt(np.sum(weights * centered**2, axis=1))
        if not np.isfinite(scales).all() or np.any(scales <= 1e-12):
            raise CalibrationError("degenerate conditional innovation pool")
        standardized = centered / scales[:, None]
        neg = np.sum(weights * standardized**2 * (standardized < 0), axis=1)
        cdf = np.cumsum(weights, axis=1)
        cdf[:, -1] = 1.0  # roundoff at the normalized CDF endpoint, not price clipping
        return cls(z.copy(), cdf, centers, scales, neg, counts, 1 / np.sum(weights**2, axis=1))

    def draw(self, uniforms, states):
        u, s = np.asarray(uniforms), np.asarray(states)
        if u.shape != s.shape or u.ndim != 1 or not np.isfinite(u).all():
            raise ValueError("pool draws need matching finite one-dimensional arrays")
        if (
            np.any((u < 0) | (u >= 1))
            or not np.issubdtype(s.dtype, np.integer)
            or np.any((s < 0) | (s > 3))
        ):
            raise ValueError("uniforms must be in [0,1) and integer states in 0..3")
        result = np.empty(len(u))
        for state in range(4):
            mask = s == state
            rows = np.searchsorted(self.cdf[state], u[mask], side="right")
            result[mask] = (self.z[rows] - self.centers[state]) / self.scales[state]
        return result


@dataclass
class RiskFilter:
    kind: str
    omega: float
    coefficients: np.ndarray
    z: np.ndarray
    history: np.ndarray
    last_variance: float
    detail: dict

    def simulate(self, uniforms, pool=None):
        """Uniforms are steps x paths. Prefixes are identical when horizon is extended."""
        u = np.asarray(uniforms, dtype=float)
        if (
            u.ndim != 2
            or min(u.shape) < 1
            or not np.isfinite(u).all()
            or np.any((u < 0) | (u >= 1))
        ):
            raise ValueError("uniforms must be finite steps x paths in [0,1)")
        horizon, n = u.shape
        buffer = np.broadcast_to(self.history[-252:], (n, 252)).copy()
        square = buffer * buffer
        sums = {lag: square[:, -lag:].sum(axis=1) for lag in (1, 5, 21, 22, 252)}
        variance = np.full(n, self.last_variance)
        residual = np.full(n, self.history[-1])
        result, sigmas = np.empty((n, horizon)), np.empty((n, horizon))
        for t in range(horizon):
            state = 2 * (sums[21] / 21 > sums[252] / 252).astype(int) + (residual < 0).astype(int)
            if self.kind == "fhs":
                a, g, b = self.coefficients
                variance = next_variance(self.omega, a, g, b, variance, residual)
            else:
                variance = self.omega + sum(
                    c * sums[lag] / lag for c, lag in zip(self.coefficients, LAGS, strict=True)
                )
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise SimulationError("invalid research conditional variance; no clipping")
            innovation = (
                self.z[np.floor(u[t] * len(self.z)).astype(int)]
                if pool is None
                else pool.draw(u[t], state)
            )
            residual = np.sqrt(variance) * innovation
            with np.errstate(over="ignore", invalid="ignore"):
                new_square = residual**2
                for lag in sums:
                    sums[lag] += new_square - square[:, (t + 252 - lag) % 252]
                square[:, t % 252] = new_square
            # Small negative rolling sums can only be floating cancellation; recompute exactly.
            if any(np.any(v < 0) for v in sums.values()):
                for lag in sums:
                    indices = (t + 1 - np.arange(1, lag + 1)) % 252
                    sums[lag] = square[:, indices].sum(axis=1)
            result[:, t], sigmas[:, t] = residual, np.sqrt(variance)
        if not np.isfinite(result).all() or not np.isfinite(sigmas).all():
            raise SimulationError("non-finite research paths")
        return result, sigmas


def _fit_harch(r):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = arch_model(r * 100, mean="Zero", vol="HARCH", p=list(LAGS), rescale=False).fit(
            disp="off", show_warning=False, options={"maxiter": 2000}
        )
    if fit.convergence_flag or not np.isfinite(fit.bic):
        raise CalibrationError("HARCH optimization did not converge")
    omega = float(fit.params["omega"]) / 10000
    coefficients = np.array([fit.params[f"alpha[{lag}]"] for lag in LAGS])
    if (
        not np.isfinite(coefficients).all()
        or not np.isfinite(omega)
        or omega <= 0
        or np.any(coefficients < 0)
        or coefficients.sum() >= 0.9999
    ):
        raise CalibrationError("HARCH coefficients violate positivity/stability")
    sigma = np.asarray(fit.conditional_volatility) / 100
    raw = r / sigma
    if not np.isfinite(raw).all() or np.std(raw) <= 1e-12:
        raise CalibrationError("invalid HARCH residuals")
    z = (raw - raw.mean()) / raw.std(ddof=0)
    detail = {
        "kind": "HARCH_daily_squared_returns",
        "lags": list(LAGS),
        "omega": omega,
        "coefficients": coefficients.tolist(),
        "persistence": float(coefficients.sum()),
        "raw_z_center": float(raw.mean()),
        "raw_z_scale": float(raw.std()),
        "warnings": [str(w.message) for w in caught],
    }
    return RiskFilter("harch", omega, coefficients, z, r.copy(), float(sigma[-1] ** 2), detail)


@dataclass
class FactorialModel:
    filters: dict
    pools: dict
    drift: float
    errors: dict
    origin: str

    def simulate(self, n_paths=2000, horizon=252, seed=20261030):
        _positive(n_paths, "n_paths")
        _positive(horizon, "horizon")
        _positive(seed, "seed", 0)
        if n_paths * horizon > 5_000_000:
            raise ValueError("more than five million path-step cells; use smaller batches")
        uniforms = np.random.default_rng(seed).random((horizon, n_paths))
        paths, errors = {}, dict(self.errors)
        for kind, fitted in self.filters.items():
            for law in ("global", "state"):
                pool = None if law == "global" else self.pools.get(kind)
                if law == "state" and pool is None:
                    continue
                try:
                    risk, _ = fitted.simulate(uniforms, pool)
                    for drift in ("zero", "shrink"):
                        name = f"{kind}_{law}_{drift}"
                        ret = risk.copy() if drift == "zero" else risk + self.drift
                        if (
                            not np.isfinite(ret).all()
                            or np.max(np.abs(np.cumsum(ret, axis=1))) > 700
                        ):
                            raise SimulationError("research prices exceed exponential range")
                        paths[name] = ret
                except SimulationError as exc:
                    errors[f"{kind}_{law}"] = str(exc)
        return paths, errors

    def summary(self):
        return {
            "origin": self.origin,
            "drift_daily_log": self.drift,
            "drift_annual_log": self.drift * 252,
            "errors": self.errors,
            "filters": {k: v.detail for k, v in self.filters.items()},
            "state_pools": {
                k: {
                    "counts": v.counts.tolist(),
                    "effective_sizes": v.effective_sizes.tolist(),
                    "negative_second_moments": v.negative_moments.tolist(),
                }
                for k, v in self.pools.items()
            },
        }


def fit_factorial(prefix: MarketData, window=1260, prior_sessions=5040):
    """Fit only a supplied prefix; no function argument accepts future returns."""
    _positive(window, "window", 253)
    prefix.validate()
    if prefix.close.shape[1] != 1 or len(prefix.close) <= window:
        raise ValueError("need one asset and a full training window")
    training = MarketData(prefix.close.iloc[-window - 1 :].copy(), metadata=dict(prefix.metadata))
    r = np.diff(np.log(training.close.iloc[:, 0].to_numpy()))
    entire = np.diff(np.log(prefix.close.iloc[:, 0].to_numpy()))
    drift = shrunk_drift(entire, prior_sessions=prior_sessions)
    filters, errors, pools = {}, {}, {}
    try:
        base = fit_market(training, ModelConfig(mean="zero"))
        filters["fhs"] = RiskFilter(
            "fhs",
            float(base.omega[0]),
            np.array([base.alpha[0], base.gamma[0], base.beta[0]]),
            base.residuals.iloc[:, 0].to_numpy().copy(),
            r.copy(),
            float(base.last_variance[0]),
            base.fit_details[base.assets[0]],
        )
    except CalibrationError as exc:
        errors["fhs"] = str(exc)
    try:
        filters["harch"] = _fit_harch(r)
    except CalibrationError as exc:
        errors["harch"] = str(exc)
    for kind, fitted in filters.items():
        pool = InnovationPool.fit(fitted.z, r)
        if kind == "fhs":
            a, g, b = fitted.coefficients
            persistence = a + b + g * pool.negative_moments
            if np.any(persistence >= 0.9999) or np.any(persistence < 0):
                errors["fhs_state"] = "a state violates GJR variance stability: " + str(
                    persistence.tolist()
                )
                continue
        pools[kind] = pool
    return FactorialModel(filters, pools, drift, errors, str(prefix.close.index[-1].date()))
