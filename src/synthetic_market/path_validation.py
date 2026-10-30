"""Daily path forecast evaluation and explicitly experimental challenger models.

No model is promoted by this module. All forecasts are issued from prefixes;
path losses enter the online mixture only after their entire horizon matures.
The primary statistical unit is the forecast origin, never a simulated path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from .data import MarketData, read_csv
from .model import CalibrationError, ModelConfig, SimulationError, fit_market
from .validation import crps_empirical

METHODS = ("fhs", "iid", "block", "analog", "analog_area", "mixture")
EXPERTS = ("fhs", "block", "analog_area")
FEATURES = (
    "terminal_return",
    "realized_volatility",
    "max_drawdown",
    "underwater_fraction",
    "longest_underwater",
    "drawdown_area",
)


def _int(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _returns(values):
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or min(result.shape) == 0 or not np.isfinite(result).all():
        raise ValueError("log returns must be a nonempty finite paths-by-steps array")
    with np.errstate(over="ignore", invalid="ignore"):
        cumulative = np.cumsum(result, axis=1)
    if not np.isfinite(cumulative).all() or np.max(np.abs(cumulative)) > 700:
        raise ValueError("cumulative log returns exceed a safe exponential range")
    return result


def _seed(seed, *keys):
    _int(seed, "seed", 0)
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1, dtype=np.uint64)[0])


def path_features(log_returns, annualization=252):
    """Forward drawdown resets its known high-water mark to the origin price.

    Time zero IS included in the peak search. Underwater durations are observed
    runs within this horizon, not completed episode lengths or recovery times.
    Realized volatility is a close-to-close RMS proxy, not intraday realized RV.
    """
    _int(annualization, "annualization")
    r = _returns(log_returns)
    logs = np.c_[np.zeros(len(r)), np.cumsum(r, axis=1)]
    if not np.isfinite(logs).all() or np.max(np.abs(logs)) > 700:
        raise ValueError("cumulative log returns exceed a safe exponential range")
    peaks = np.maximum.accumulate(logs, axis=1)
    dd = -np.expm1(logs[:, 1:] - peaks[:, 1:])
    below = dd > 1e-12
    run = np.zeros(len(r), dtype=np.int64)
    longest = run.copy()
    for col in below.T:
        run = (run + 1) * col
        longest = np.maximum(longest, run)
    return {
        "terminal_return": np.expm1(logs[:, -1]),
        "realized_volatility": np.sqrt(annualization * np.mean(r * r, axis=1)),
        "max_drawdown": dd.max(axis=1),
        "underwater_fraction": below.mean(axis=1),
        "longest_underwater": longest.astype(float),
        "drawdown_area": dd.mean(axis=1),
    }


def first_passage_cdf(log_returns, barrier=0.10, kind="drawdown"):
    """Empirical P(first passage <= t), for t=1..H, including unreached paths.

    drawdown: loss from an evolving forward peak, including time zero.
    origin_loss: loss relative to the fixed origin price.
    No-hit means right-censored at H; it is never treated as recovery at H.
    """
    if isinstance(barrier, (bool, np.bool_)) or not np.isfinite(barrier) or not 0 < barrier < 1:
        raise ValueError("barrier must lie strictly between zero and one")
    r = _returns(log_returns)
    logs = np.c_[np.zeros(len(r)), r.cumsum(axis=1)]
    if kind == "drawdown":
        log_loss = np.maximum.accumulate(logs, axis=1)[:, 1:] - logs[:, 1:]
    elif kind == "origin_loss":
        log_loss = -logs[:, 1:]
    else:
        raise ValueError("kind must be drawdown or origin_loss")
    hit = log_loss >= -np.log1p(-barrier) - 1e-12
    ever = np.maximum.accumulate(hit, axis=1)
    return ever.mean(axis=0)


def recovery_cdf(log_returns, initial_log_gap):
    """P(recover the already known pre-origin peak by t).

    initial_log_gap=log(known_peak/current_price). Gap zero is already recovered
    at issuance. Incomplete recoveries remain non-events, not fake H-day events.
    """
    if not np.isfinite(initial_log_gap) or initial_log_gap < 0:
        raise ValueError("initial_log_gap must be finite and nonnegative")
    r = _returns(log_returns)
    if initial_log_gap <= 1e-12:
        return np.ones(r.shape[1])
    return np.maximum.accumulate(r.cumsum(axis=1) >= initial_log_gap - 1e-12, axis=1).mean(axis=0)


def signed_area(increments, scale):
    """Exact level-2 antisymmetric areas for piecewise-linear augmented paths.

    Channels: normalized time, cumulative return, cumulative squared return.
    The two areas are (t,return) and (t,quadratic variation). This is NOT the
    full/infinite signature kernel. The full daily vector is retained below.
    """
    r = _returns(increments)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    h = r.shape[1]
    time_left = np.arange(h, dtype=float) / h
    dr = r / (scale * np.sqrt(h))
    dv = r * r / (scale * scale * h)
    ret_left = dr.cumsum(axis=1) - dr
    var_left = dv.cumsum(axis=1) - dv
    return np.c_[
        0.5 * np.sum(time_left * dr - ret_left / h, axis=1),
        0.5 * np.sum(time_left * dv - var_left / h, axis=1),
    ]


def path_embedding(log_returns, scale):
    """Injective full-grid embedding plus exact level-2 areas, training scale only."""
    r = _returns(log_returns)
    area = signed_area(r, scale)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        embedded = np.c_[r.cumsum(axis=1) / (scale * r.shape[1]), area]
    if not np.isfinite(embedded).all():
        raise ValueError("path embedding exceeds floating-point range")
    return embedded


def kernel_path_score(samples, observation, scale, seed=42, pairs=4096):
    """Bounded Gaussian-mixture kernel score on complete daily paths.

    0.5 E k(X,X') - E k(X,y) + 0.5 k(y,y), lower is better. X,X' are
    distinct iid ensemble members: an unbiased U-pair estimate of the latent
    forecast score, not its finite-ensemble V score. Finite estimates can be
    slightly negative. Fixed bandwidths 0.5,1,2 are not tuned on outcomes.
    The RBF on the injective finite-grid embedding is characteristic; the
    areas alone would NOT identify every path distribution.
    """
    r = _returns(samples)
    y = _returns(observation)
    if len(r) < 2 or y.shape != (1, r.shape[1]):
        raise ValueError("need >=2 paths and one observation with the same horizon")
    _int(pairs, "pairs")
    x, target = path_embedding(r, scale), path_embedding(y, scale)[0]
    rng = np.random.default_rng(_seed(seed, 71))
    left = rng.integers(len(x), size=pairs)
    offset = rng.integers(1, len(x), size=pairs)
    right = (left + offset) % len(x)
    d_xy = np.sum((x - target) ** 2, axis=1)
    d_xx = np.sum((x[left] - x[right]) ** 2, axis=1)
    k_xy = np.mean([np.exp(-d_xy / (2 * b * b)).mean() for b in (0.5, 1.0, 2.0)])
    k_xx = np.mean([np.exp(-d_xx / (2 * b * b)).mean() for b in (0.5, 1.0, 2.0)])
    return float(0.5 * k_xx - k_xy + 0.5)


def stationary_indices(n, n_paths, horizon, mean_block, seed):
    """Circular stationary bootstrap with geometric block lengths (mean L)."""
    for value, name in ((n, "n"), (n_paths, "n_paths"), (horizon, "horizon")):
        _int(value, name)
    if not np.isfinite(mean_block) or mean_block < 1:
        raise ValueError("mean_block must be finite and >=1")
    rng = np.random.default_rng(seed)
    rows = rng.integers(n, size=(n_paths, horizon))
    restart = rng.random((n_paths, horizon)) < 1.0 / mean_block
    for t in range(1, horizon):
        rows[:, t] = np.where(restart[:, t], rows[:, t], (rows[:, t - 1] + 1) % n)
    return rows


def _states(r, positions, with_area):
    """Causal state at each return endpoint; only <= position enters a feature."""
    states, scales = [], []
    for end in positions:
        hist = r[end - 251 : end + 1]
        vol = np.sqrt(np.mean(hist * hist))
        short = np.sqrt(np.mean(hist[-21:] ** 2))
        middle = np.sqrt(np.mean(hist[-63:] ** 2))
        if not np.isfinite([vol, short, middle]).all() or min(vol, short, middle) <= 1e-12:
            raise ValueError("degenerate historical scale in analogue state")
        logs = np.r_[0.0, hist.cumsum()]
        values = [
            np.log(short / vol),
            np.log(middle / vol),
            hist[-63:].sum() / (vol * np.sqrt(63)),
            (logs.max() - logs[-1]) / (vol * np.sqrt(252)),
        ]
        if with_area:
            # Scale-free placement of variation: late vs early within the window.
            qv = hist[-63:] ** 2
            mid_time = (np.arange(63) + 0.5) / 63
            values.append(float(np.dot(mid_time - 0.5, qv) / qv.sum()))
        states.append(values)
        scales.append(short)
    return np.asarray(states), np.asarray(scales)


def analogue_paths(history, horizon, n_paths, seed=42, neighbors=100, with_area=True):
    """Conditional analogue continuations, an experimental meteorological transfer.

    Donor outcomes must be FULLY observed in history. Neighbors use preceding
    state only. Whole continuations preserve their observed temporal ordering.
    Scale by current/donor past-21-day RMS and remove the exact weighted
    lead-wise donor mean, keeping zero expected log drift at every horizon.
    """
    r = np.asarray(history, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError("history must be a finite univariate return series")
    for v, name in ((horizon, "horizon"), (n_paths, "n_paths"), (neighbors, "neighbors")):
        _int(v, name)
    donors = np.arange(251, len(r) - horizon)
    if len(donors) < neighbors:
        raise ValueError("not enough fully matured historical analogue continuations")
    states, scales = _states(r, np.r_[donors, len(r) - 1], with_area)
    location = np.median(states[:-1], axis=0)
    spread = np.median(np.abs(states[:-1] - location), axis=0)
    spread = np.where(spread > 1e-10, spread, 1.0)
    distance = np.sum(((states[:-1] - states[-1]) / spread) ** 2, axis=1)
    chosen = np.argsort(distance, kind="stable")[:neighbors]
    bandwidth2 = max(float(distance[chosen[-1]]), 1e-12)
    probability = softmax(-distance[chosen] / (2 * bandwidth2))
    donor_rows = donors[chosen, None] + 1 + np.arange(horizon)[None, :]
    continuations = r[donor_rows] * (scales[-1] / scales[chosen])[:, None]
    continuations = continuations - probability @ continuations
    rng = np.random.default_rng(seed)
    selection = rng.choice(neighbors, n_paths, p=probability)
    if not np.isfinite(continuations).all():
        raise ValueError("analogue continuations exceed floating-point range")
    # This is non-overlap, NOT statistical independence. Donor weights alone
    # can obscure a pool containing many copies of one historical episode.
    disjoint, last_end = 0, -1
    for donor in np.sort(donors[chosen]):
        if donor >= last_end:
            disjoint += 1
            last_end = donor + horizon
    return continuations[selection], {
        "donor_count": len(donors),
        "neighbors": neighbors,
        "maximum_disjoint_donor_outcomes": disjoint,
        "effective_weighted_neighbors": float(1.0 / np.sum(probability**2)),
        "maximum_donor_outcome_index": int(donor_rows.max()),
        "history_last_index": len(r) - 1,
        "scale_ratio_min": float((scales[-1] / scales[chosen]).min()),
        "scale_ratio_max": float((scales[-1] / scales[chosen]).max()),
        "signature_area_feature": bool(with_area),
        "mean": "zero exact weighted lead-wise conditional log mean",
    }


@dataclass(frozen=True)
class PathConfig:
    horizons: tuple[int, ...] = (21, 63, 126)
    train_window: int = 1260
    stride: int = 21
    paths: int = 2000
    seed: int = 20261030
    neighbors: int = 100
    block_length: int = 10
    pairs: int = 4096
    bootstrap_reps: int = 2000
    mix_learning_rate: float = 1.0
    mix_forgetting: float = 0.97

    def __post_init__(self):
        for name in (
            "train_window",
            "stride",
            "paths",
            "neighbors",
            "block_length",
            "pairs",
            "bootstrap_reps",
        ):
            _int(getattr(self, name), name)
        _int(self.seed, "seed", 0)
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be strictly increasing positive integers")
        for h in self.horizons:
            _int(h, "horizon")
        if self.paths < 50 or self.train_window < max(self.horizons) + 252 + self.neighbors:
            raise ValueError("need >=50 paths and enough training history for all donors")
        if self.paths * max(self.horizons) > 5_000_000:
            raise ValueError("more than five million path-step cells")
        if not np.isfinite(self.mix_learning_rate) or self.mix_learning_rate <= 0:
            raise ValueError("mix_learning_rate must be positive")
        if not np.isfinite(self.mix_forgetting) or not 0 < self.mix_forgetting <= 1:
            raise ValueError("mix_forgetting must be in (0,1]")


class DelayedMixture:
    """Discounted exponential experts: only full-horizon matured losses update weights.

    This is an explicit heuristic adaptation, not a claimed regret theorem for
    nonstationary financial returns. The default FHS engine is unchanged.
    """

    def __init__(self, learning_rate=1.0, forgetting=0.97):
        if (
            not np.isfinite(learning_rate)
            or learning_rate <= 0
            or not np.isfinite(forgetting)
            or not 0 < forgetting <= 1
        ):
            raise ValueError("invalid mixture learning rate or forgetting")
        self.learning_rate, self.forgetting = learning_rate, forgetting
        self.log_weights = np.zeros(len(EXPERTS))
        self.pending = deque()
        self.last_origin = -1
        self.last_matured = None
        self.updates = 0

    def weights(self, origin):
        _int(origin, "origin", 0)
        if origin < self.last_origin:
            raise ValueError("origins must be chronological")
        self.last_origin = origin
        while self.pending and self.pending[0][0] <= origin:
            end, loss = self.pending.popleft()
            self.log_weights = self.forgetting * self.log_weights - self.learning_rate * loss
            self.log_weights -= self.log_weights.max()
            self.last_matured = end
            self.updates += 1
        return softmax(self.log_weights)

    def enqueue(self, maturity, losses):
        _int(maturity, "maturity", 0)
        loss = np.asarray(losses, dtype=float)
        if loss.shape != (len(EXPERTS),) or not np.isfinite(loss).all():
            raise ValueError("need a finite loss for every expert")
        if maturity <= self.last_origin or (self.pending and maturity <= self.pending[-1][0]):
            raise ValueError("queued maturities must be strictly increasing and in the future")
        self.pending.append((maturity, loss.copy()))


def issue_forecasts(training: MarketData, config: PathConfig, origin_index: int):
    """Separate causal issuing function: it has no access to future observations."""
    if len(training.close.columns) != 1:
        raise ValueError("path study is explicitly single-asset; select one asset")
    r = np.diff(np.log(training.close.iloc[:, 0].to_numpy()))
    h = max(config.horizons)
    draws, status = {}, {}
    try:
        fitted = fit_market(training, ModelConfig(mean="zero"))
        draws["fhs"] = fitted.simulate(
            config.paths, h, _seed(config.seed, origin_index, 1), include_ohlc=False
        ).log_returns[:, :, 0]
        status["fhs"] = fitted.summary()["fit"]
    except (CalibrationError, SimulationError) as exc:
        status["fhs"] = {"error": str(exc)}
    centered = r - r.mean()
    for method, length, key in (("iid", 1, 2), ("block", config.block_length, 3)):
        indices = stationary_indices(
            len(r), config.paths, h, length, _seed(config.seed, origin_index, key)
        )
        draws[method] = centered[indices]
    for method, area, key in (("analog", False, 4), ("analog_area", True, 5)):
        draws[method], status[method] = analogue_paths(
            r, h, config.paths, _seed(config.seed, origin_index, key), config.neighbors, area
        )
    return draws, status


def binomial_mc_interval(events, trials):
    """Wilson 95% interval for independent simulator draws, NOT model uncertainty.

    Unlike plug-in standard errors, zero simulated hits do not imply certainty.
    This describes Monte Carlo resolution conditional on the issued generator.
    """
    _int(trials, "trials")
    _int(events, "events", 0)
    if events > trials:
        raise ValueError("events cannot exceed trials")
    z = 1.959963984540054
    p = events / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (
        0.0 if events == 0 else float(max(0.0, center - radius)),
        1.0 if events == trials else float(min(1.0, center + radius)),
    )


def _score_forecast(samples, actual, scale, log_gap, config, origin, horizon):
    predicted, observed = path_features(samples), path_features(actual)
    record = {}
    for name in FEATURES:
        x, y = predicted[name], float(observed[name][0])
        q = np.quantile(x, [0.05, 0.5, 0.9, 0.95], method="inverted_cdf")
        rng = np.random.default_rng(_seed(config.seed, origin, horizon, 100 + FEATURES.index(name)))
        record.update(
            {
                f"{name}_actual": y,
                f"{name}_q05": float(q[0]),
                f"{name}_q50": float(q[1]),
                f"{name}_q90": float(q[2]),
                f"{name}_q95": float(q[3]),
                f"{name}_crps": float(crps_empirical(x, y)),
                f"{name}_coverage90": float(q[0] <= y <= q[3]),
                f"{name}_above_q95": float(y > q[3]),
                f"{name}_below_q05": float(y < q[0]),
                f"{name}_rank": float(
                    (np.sum(x < y) + rng.random() * (np.sum(x == y) + 1)) / (len(x) + 1)
                ),
            }
        )
    record["drawdown_twcrps10"] = float(
        crps_empirical(
            np.maximum(predicted["max_drawdown"], 0.10), max(observed["max_drawdown"][0], 0.10)
        )
    )
    for level in (0.05, 0.10, 0.20):
        tag = str(int(level * 100))
        prob, event = first_passage_cdf(samples, level), first_passage_cdf(actual, level)
        record[f"drawdown_hit{tag}_prob"] = float(prob[-1])
        record[f"drawdown_hit{tag}_actual"] = float(event[-1])
        record[f"drawdown_hit{tag}_brier"] = float((prob[-1] - event[-1]) ** 2)
        record[f"drawdown_hit{tag}_integrated_brier"] = float(np.mean((prob - event) ** 2))
        record[f"drawdown_hit{tag}_mc_se"] = float(
            np.sqrt(prob[-1] * (1 - prob[-1]) / len(samples))
        )
        low, high = binomial_mc_interval(int(round(prob[-1] * len(samples))), len(samples))
        record[f"drawdown_hit{tag}_mc_lower95"] = low
        record[f"drawdown_hit{tag}_mc_upper95"] = high
    record["origin_loss10_prob"] = float(first_passage_cdf(samples, 0.10, "origin_loss")[-1])
    record["origin_loss10_actual"] = float(first_passage_cdf(actual, 0.10, "origin_loss")[-1])
    record["kernel_path_score"] = kernel_path_score(
        samples, actual, scale, _seed(config.seed, origin, horizon, 99), config.pairs
    )
    record["initial_log_gap"] = float(log_gap)
    if log_gap > 1e-12:
        probability, event = recovery_cdf(samples, log_gap), recovery_cdf(actual, log_gap)
        record.update(
            recovery_prob=float(probability[-1]),
            recovery_actual=float(event[-1]),
            recovery_integrated_brier=float(np.mean((probability - event) ** 2)),
            recovery_restricted_mean=float(1.0 + np.sum(1.0 - probability[:-1])),
            recovery_restricted_actual=float(1.0 + np.sum(1.0 - event[:-1])),
            recovery_right_censored=bool(event[-1] == 0),
        )
    else:
        record.update(
            recovery_prob=None,
            recovery_actual=None,
            recovery_integrated_brier=None,
            recovery_restricted_mean=None,
            recovery_restricted_actual=None,
            recovery_right_censored=None,
        )
    return record


def evaluate_paths(data: MarketData, config=None, start="2000-01-01", progress=False):
    """Rolling fixed-window forecasts on a common origin grid for every horizon.

    Trajectories are open-loop: no realized future return enters a simulated
    path. Administrative right truncation excludes origins without Hmax future
    observations, never treats incomplete targets as complete. All failures are
    recorded; comparisons use an explicit common-method set at each origin.
    """
    config = config or PathConfig()
    if len(data.close.columns) != 1:
        raise ValueError("select exactly one asset")
    date = pd.Timestamp(start)
    if pd.isna(date) or date.tz is not None or date != date.normalize():
        raise ValueError("start must be a midnight date without timezone")
    n, h = len(data.close), max(config.horizons)
    first = max(config.train_window, int(data.close.index.searchsorted(date)))
    origins = np.arange(first, n - h, config.stride, dtype=int)
    if not len(origins):
        raise ValueError("no complete origins after training window and requested start")
    r = np.diff(np.log(data.close.iloc[:, 0].to_numpy()))
    logs = np.log(data.close.iloc[:, 0].to_numpy())
    peaks = np.maximum.accumulate(logs)
    learner = DelayedMixture(config.mix_learning_rate, config.mix_forgetting)
    rows, audit = [], []
    for step, origin in enumerate(origins):
        weights = learner.weights(int(origin))
        training = MarketData(
            data.close.iloc[origin - config.train_window : origin + 1].copy(),
            metadata=dict(data.metadata),
        )
        draws, status = issue_forecasts(training, config, int(origin))
        complete = all(m in draws for m in EXPERTS)
        if complete:
            rng = np.random.default_rng(_seed(config.seed, int(origin), 6))
            labels = rng.choice(len(EXPERTS), config.paths, p=weights)
            source = rng.integers(config.paths, size=config.paths)
            draws["mixture"] = np.stack([draws[m] for m in EXPERTS])[labels, source]
        actual = r[origin : origin + h]
        scale = float(np.std(r[origin - config.train_window : origin], ddof=0))
        log_gap = float(peaks[origin] - logs[origin])
        losses = {}
        for method, forecast in draws.items():
            for horizon in config.horizons:
                record = {
                    "origin_index": int(origin),
                    "origin": str(data.close.index[origin].date()),
                    "target_end": str(data.close.index[origin + horizon].date()),
                    "horizon": horizon,
                    "method": method,
                    "complete_expert_set": complete,
                }
                actual_logs = np.r_[0.0, actual[:horizon].cumsum()]
                actual_drawdown = np.maximum.accumulate(actual_logs) - actual_logs
                trough = int(np.argmax(actual_drawdown))
                peak = int(np.argmax(actual_logs[: trough + 1]))
                record["max_drawdown_peak_date"] = str(data.close.index[origin + peak].date())
                record["max_drawdown_trough_date"] = str(data.close.index[origin + trough].date())
                record.update(
                    _score_forecast(
                        forecast[:, :horizon],
                        actual[:horizon],
                        scale,
                        log_gap,
                        config,
                        int(origin),
                        horizon,
                    )
                )
                if horizon == h and method in EXPERTS:
                    losses[method] = record["kernel_path_score"]
                rows.append(record)
        if complete:
            learner.enqueue(int(origin + h), [losses[m] for m in EXPERTS])
        audit.append(
            {
                "origin": str(data.close.index[origin].date()),
                "origin_index": int(origin),
                "train_start": str(training.close.index[0].date()),
                "train_end": str(training.close.index[-1].date()),
                "weights": dict(zip(EXPERTS, map(float, weights), strict=True)),
                "matured_updates": learner.updates,
                "last_matured_index": learner.last_matured,
                "complete_expert_set": complete,
                "status": status,
            }
        )
        if progress and (step % 20 == 0 or step == len(origins) - 1):
            print(
                f"Completed origin {step + 1}/{len(origins)}: {data.close.index[origin].date()}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    metadata = {
        "config": asdict(config),
        "input_sha256": data.fingerprint(),
        "source": data.metadata,
        "observed_start": str(data.close.index[0].date()),
        "observed_end": str(data.close.index[-1].date()),
        "input_observations": n,
        "requested_start": str(date.date()),
        "requested_origins": len(origins),
        "complete_origins": int(sum(a["complete_expert_set"] for a in audit)),
        "excluded_incomplete_tail_observations": h,
        "interpretation": "Retrospective rolling-origin pseudo-out-of-sample, not prospective or immutable point-in-time evidence; zero log mean, daily adjusted prices; no intraday extrema.",
        "audit": audit,
    }
    return frame, metadata


def block_mean_interval(values, mean_block, reps=2000, seed=42):
    """Percentile stationary-bootstrap interval for a dependent-origin mean.

    Approximate and pointwise under suitable stationarity/mixing; overlapping
    returns and changing regimes prevent a universal finite-sample guarantee.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("need >=2 finite ordered observations")
    index = stationary_indices(len(x), _int(reps, "reps"), len(x), mean_block, seed)
    boot = x[index].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "mean": float(x.mean()),
        "lower95": float(low),
        "upper95": float(high),
        "origins": len(x),
        "mean_block_origins": float(mean_block),
    }


def summarize_paths(frame, config, evaluation_start="2015-01-01"):
    date = pd.Timestamp(evaluation_start)
    if pd.isna(date) or date.tz is not None or date != date.normalize():
        raise ValueError("evaluation_start must be a midnight date without timezone")
    evaluation_start = str(date.date())
    if frame.duplicated(["origin_index", "horizon", "method"]).any():
        raise ValueError("duplicate origin/horizon/method records")
    selected = frame.loc[(frame.origin >= evaluation_start) & frame.complete_expert_set].copy()
    for _, group in selected.groupby(["origin_index", "horizon"]):
        if set(group.method) != set(METHODS):
            raise ValueError("complete-origin flag disagrees with available methods")
    if selected.empty:
        raise ValueError("no complete evaluation origins")
    metrics = [f"{f}_crps" for f in FEATURES] + [
        "drawdown_twcrps10",
        "kernel_path_score",
        "drawdown_hit10_brier",
        "drawdown_hit10_integrated_brier",
    ]
    aggregate, comparisons, coverage = [], [], []
    for h in config.horizons:
        sub = selected.loc[selected.horizon == h]
        overlap = max(1, math.ceil(h / config.stride))
        for method in METHODS:
            group = sub.loc[sub.method == method].sort_values("origin_index")
            if len(group) < 2:
                continue
            row = {"method": method, "horizon": h, "origins": len(group)}
            row.update({m: float(group[m].mean()) for m in metrics})
            row.update(
                {f"{f}_coverage90": float(group[f"{f}_coverage90"].mean()) for f in FEATURES}
            )
            row.update({f"{f}_above_q95": float(group[f"{f}_above_q95"].mean()) for f in FEATURES})
            row.update(
                {
                    f"hit{t}_forecast": float(group[f"drawdown_hit{t}_prob"].mean())
                    for t in (5, 10, 20)
                }
            )
            row.update(
                {
                    f"hit{t}_observed": float(group[f"drawdown_hit{t}_actual"].mean())
                    for t in (5, 10, 20)
                }
            )
            row["drawdown_q95_breach_origins"] = int(group.max_drawdown_above_q95.sum())
            row["drawdown_q95_breach_peak_dates"] = sorted(
                group.loc[group.max_drawdown_above_q95 > 0, "max_drawdown_peak_date"]
                .unique()
                .tolist()
            )
            eligible = group.loc[group.recovery_prob.notna()]
            row["recovery_eligible_origins"] = len(eligible)
            for key in ("recovery_prob", "recovery_actual", "recovery_integrated_brier"):
                row[key] = float(eligible[key].mean()) if len(eligible) else None
            aggregate.append(row)
            for f in ("terminal_return", "realized_volatility", "max_drawdown"):
                values = group[f"{f}_coverage90"].to_numpy()
                cov = block_mean_interval(
                    values, 2 * overlap, config.bootstrap_reps, _seed(config.seed, h, 20)
                )
                phases = [values[p::overlap] for p in range(overlap) if len(values[p::overlap])]
                coverage.append(
                    {
                        "method": method,
                        "horizon": h,
                        "feature": f,
                        **cov,
                        "nonoverlap_phase_counts": [len(a) for a in phases],
                        "nonoverlap_phase_coverages": [float(a.mean()) for a in phases],
                    }
                )
        comparison_pairs = [(m, "fhs") for m in METHODS if m != "fhs"] + [("analog_area", "analog")]
        for method, reference in comparison_pairs:
            for metric in (
                "max_drawdown_crps",
                "drawdown_twcrps10",
                "kernel_path_score",
                "drawdown_hit10_integrated_brier",
            ):
                pivot = sub.pivot(
                    index="origin_index", columns="method", values=metric
                ).sort_index()
                differences = (pivot[method] - pivot[reference]).to_numpy()
                for mult in (1, 2, 4):
                    comparisons.append(
                        {
                            "method": method,
                            "reference": reference,
                            "horizon": h,
                            "metric": metric,
                            **block_mean_interval(
                                differences,
                                mult * overlap,
                                config.bootstrap_reps,
                                _seed(config.seed, h, mult, 21),
                            ),
                        }
                    )
    return {
        "evaluation_start": evaluation_start,
        "requested_evaluation_origins": int(
            frame.loc[frame.origin >= evaluation_start, "origin_index"].nunique()
        ),
        "complete_evaluation_origins": int(selected.origin_index.nunique()),
        "aggregate": aggregate,
        "paired_differences": comparisons,
        "coverage": coverage,
        "inference": "Pointwise approximate stationary-bootstrap intervals; three predeclared block lengths. Not multiplicity-adjusted, not finite-sample certificates. Phase views are not independent replications.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--asset", default="SPY")
    parser.add_argument("--price-basis", choices=["adjusted", "raw"], required=True)
    parser.add_argument("--metadata", type=Path, help="Optional provider input_metadata.json")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--evaluation-start", default="2015-01-01")
    parser.add_argument("--horizons", type=int, nargs="+", default=[21, 63, 126])
    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--stride", type=int, default=21)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = PathConfig(
        horizons=tuple(args.horizons),
        train_window=args.train_window,
        stride=args.stride,
        paths=args.paths,
        seed=args.seed,
        pairs=args.pairs,
        bootstrap_reps=args.bootstrap_reps,
    )
    source_paths = {"input": args.csv}
    source_paths.update({path.name: path for path in Path(__file__).parent.glob("*.py")})
    if args.metadata is not None:
        source_paths["provider_metadata"] = args.metadata
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()
    }
    source = read_csv(args.csv, price_basis=args.price_basis)
    if args.metadata is not None:
        provenance = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            raise ValueError("provider metadata must be a JSON object")
        if provenance.get("price_basis", args.price_basis) != args.price_basis:
            raise ValueError("price-basis disagrees with provider metadata")
        for key, position in (("observed_start", 0), ("observed_end", -1)):
            if key in provenance and pd.Timestamp(provenance[key]) != source.close.index[position]:
                raise ValueError(f"{key} disagrees with provider metadata")
        source.metadata["provider_snapshot"] = provenance
    if args.asset not in source.close:
        parser.error("requested asset absent from CSV")
    data = MarketData(source.close[[args.asset]], metadata=source.metadata)
    from .cli import _new_output_directory

    with _new_output_directory(args.out) as staging:
        frame, meta = evaluate_paths(data, config, args.start, progress=True)
        summary = summarize_paths(frame, config, args.evaluation_start)
        frame.to_csv(staging / "path_forecasts.csv", index=False, float_format="%.17g")
        (staging / "path_metadata.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8"
        )
        (staging / "path_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        import importlib.metadata

        packages = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "arch")}
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.iterdir()}
        if any(
            hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("input or source code changed during run; refusing publication")
        manifest = {
            "source_hashes": source_hashes,
            "argv": vars(args)
            | {
                "csv": str(args.csv),
                "out": str(args.out),
                "metadata": None if args.metadata is None else str(args.metadata),
            },
            "config": asdict(config),
            "input_file_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
            "input_fingerprint": meta["input_sha256"],
            "software_versions": packages,
            "files": files,
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(f"Wrote path study to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
