"""Rolling factorial experiment with matured-only model selection and paired inference."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import MarketData
from ..path_validation import _score_forecast, _seed, stationary_indices
from .generators import BASELINE, CANDIDATES, _positive, fit_factorial

METHODS = CANDIDATES + ("delayed_selector",)


@dataclass(frozen=True)
class ResearchConfig:
    horizons: tuple[int, ...] = (21, 63, 126, 252)
    train_window: int = 1260
    stride: int = 21
    paths: int = 2000
    seed: int = 20261030
    pairs: int = 4096
    bootstrap_reps: int = 2000
    prior_sessions: int = 5040

    def __post_init__(self):
        for key in ("train_window", "stride", "paths", "pairs", "bootstrap_reps"):
            _positive(getattr(self, key), key)
        _positive(self.seed, "seed", 0)
        _positive(self.prior_sessions, "prior_sessions", 0)
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        for h in self.horizons:
            _positive(h, "horizon")
        if 126 not in self.horizons or self.train_window < 253 or self.paths < 50:
            raise ValueError(
                "require 126-session primary target, >=253 training returns, >=50 paths"
            )
        if self.paths * max(self.horizons) > 5_000_000:
            raise ValueError("more than five million path-step cells")


class MaturedSelector:
    """Rolling inner model choice; no gradient or outcome arrives before maturity."""

    def __init__(self, minimum=24, memory=60):
        _positive(minimum, "minimum")
        _positive(memory, "memory", minimum)
        self.minimum = minimum
        self.pending = deque()
        self.available = deque(maxlen=memory)
        self.last_origin = -1
        self.last_maturity = None

    def choose(self, origin):
        _positive(origin, "origin", 0)
        if origin <= self.last_origin:
            raise ValueError("selector origins must strictly increase")
        self.last_origin = origin
        while self.pending and self.pending[0][0] <= origin:
            maturity, scores = self.pending.popleft()
            self.available.append(scores)
            self.last_maturity = maturity
        if len(self.available) < self.minimum:
            return BASELINE
        losses = np.mean(np.stack(self.available), axis=0)
        return CANDIDATES[int(np.argmin(losses))]

    def enqueue(self, maturity, scores):
        _positive(maturity, "maturity", 0)
        s = np.asarray(scores, dtype=float)
        if s.shape != (len(CANDIDATES),) or not np.isfinite(s).all():
            raise ValueError("need one finite score per candidate")
        if maturity <= self.last_origin or (self.pending and maturity <= self.pending[-1][0]):
            raise ValueError("maturity must be increasing and strictly after current origin")
        self.pending.append((maturity, s.copy()))


def _date(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tz is not None or t != t.normalize():
        raise ValueError("dates must be timezone-naive midnight labels")
    return t


def origin_state(returns):
    r = np.asarray(returns, dtype=float)
    annual = float(np.sqrt(252 * np.mean(r[-252:] ** 2)))
    short = float(np.sqrt(252 * np.mean(r[-21:] ** 2)))
    return {
        "initial_vol252": annual,
        "initial_vol21": short,
        "initial_vol_group": "low" if annual < 0.15 else "medium" if annual < 0.25 else "high",
        "initial_vol_direction": "rising" if short > annual else "falling",
        "initial_trend": "positive" if r[-63:].sum() >= 0 else "negative",
    }


def evaluate(data: MarketData, config=None, start="2000-01-01", progress=False):
    config = config or ResearchConfig()
    data.validate()
    if data.close.shape[1] != 1:
        raise ValueError("one asset per research experiment")
    if data.close.index[-1] > _date("2026-10-30"):
        raise ValueError("data exceed research information cutoff 2026-10-30")
    first = max(config.train_window, int(data.close.index.searchsorted(_date(start))))
    origins = list(range(first, len(data.close) - min(config.horizons), config.stride))
    if not origins:
        raise ValueError("no complete forecast targets")
    logs = np.log(data.close.iloc[:, 0].to_numpy())
    returns, peaks = np.diff(logs), np.maximum.accumulate(logs)
    selector = MaturedSelector()
    rows, metadata = [], []
    for count, index in enumerate(origins):
        chosen = selector.choose(index)
        prefix = MarketData(data.close.iloc[: index + 1].copy(), metadata=dict(data.metadata))
        fitted = fit_factorial(prefix, config.train_window, config.prior_sessions)
        draws, errors = fitted.simulate(
            config.paths, max(config.horizons), _seed(config.seed, index, 201)
        )
        horizons = [h for h in config.horizons if index + h < len(logs)]
        actual = returns[index : index + max(horizons)]
        state = origin_state(returns[:index])
        gap = float(peaks[index] - logs[index])
        state["initial_drawdown"] = float(-np.expm1(-gap))
        state["initial_underwater"] = "deep" if state["initial_drawdown"] >= 0.10 else "shallow"
        scale = float(np.std(returns[index - config.train_window : index], ddof=0))
        common = all(m in draws for m in CANDIDATES)
        scored, losses = {}, {}
        for method, forecast in draws.items():
            for h in horizons:
                path = np.r_[0.0, actual[:h].cumsum()]
                trough = int(np.argmax(np.maximum.accumulate(path) - path))
                peak = int(np.argmax(path[: trough + 1]))
                row = {
                    "asset": str(data.close.columns[0]),
                    "origin_index": index,
                    "origin": str(data.close.index[index].date()),
                    "target_end": str(data.close.index[index + h].date()),
                    "horizon": h,
                    "method": method,
                    "common_candidate_set": common,
                    "max_drawdown_peak_date": str(data.close.index[index + peak].date()),
                    "max_drawdown_trough_date": str(data.close.index[index + trough].date()),
                    "drift_annual_log": fitted.drift * 252 if method.endswith("shrink") else 0.0,
                    **state,
                }
                row.update(
                    _score_forecast(forecast[:, :h], actual[:h], scale, gap, config, index, h)
                )
                alarm = int(row["drawdown_hit10_prob"] >= 1 / 6)
                event = int(row["drawdown_hit10_actual"])
                row.update(
                    alarm=alarm,
                    false_alarm=int(alarm and not event),
                    missed_event=int(event and not alarm),
                    decision_loss=float(alarm * (1 - event) + 5 * (1 - alarm) * event),
                )
                rows.append(row)
                scored[(method, h)] = row
                if h == 126:
                    losses[method] = row["max_drawdown_crps"]
        for h in horizons:
            if (chosen, h) in scored:
                rows.append(
                    {**scored[(chosen, h)], "method": "delayed_selector", "selected": chosen}
                )
            else:
                errors["delayed_selector"] = (
                    f"chosen candidate unavailable: {chosen}; no replacement"
                )
        meta = {
            "origin_index": index,
            "origin": str(data.close.index[index].date()),
            "training_start": str(data.close.index[index - config.train_window].date()),
            "training_end": str(data.close.index[index].date()),
            "available_horizons": horizons,
            "complete_candidate_set": common,
            "selected": chosen,
            "selector_matured_count": len(selector.available),
            "selector_last_maturity": selector.last_maturity,
            "errors": errors,
            "fit": fitted.summary(),
        }
        metadata.append(meta)
        if len(losses) == len(CANDIDATES):
            selector.enqueue(index + 126, [losses[m] for m in CANDIDATES])
        if progress and (count % 10 == 0 or count == len(origins) - 1):
            print(
                f"{data.close.columns[0]} origin {count + 1}/{len(origins)} {meta['origin']} errors={len(errors)}",
                flush=True,
            )
    return pd.DataFrame(rows), metadata


def simultaneous_intervals(differences, block_length, reps=2000, seed=42):
    """Approximate common-block max-standardized intervals, not finite-sample proof.

    Columns are paired candidate-minus-reference scores on the SAME origins.
    Bootstrap centering and a max statistic address this declared comparison family.
    """
    x = np.asarray(differences, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or not np.isfinite(x).all():
        raise ValueError("finite common-origin differences required")
    _positive(reps, "reps", 20)
    indices = stationary_indices(len(x), reps, len(x), block_length, seed)
    means = x[indices].mean(axis=1)
    center, se = x.mean(axis=0), means.std(axis=0, ddof=1)
    active = se > 1e-15
    maximum = (
        np.max(np.abs((means[:, active] - center[active]) / se[active]), axis=1)
        if active.any()
        else np.zeros(reps)
    )
    critical = float(np.quantile(maximum, 0.95))
    return center, center - critical * se, center + critical * se


def summarize(frame, config, evaluation_start="2015-01-01"):
    if frame.empty or frame.duplicated(["origin_index", "horizon", "method"]).any():
        raise ValueError("empty/duplicate research forecast records")
    if not set(frame.method).issubset(METHODS):
        raise ValueError("unknown method")
    sub = frame.loc[frame.origin >= str(_date(evaluation_start).date())].copy()
    if sub.empty:
        raise ValueError("no evaluation origins")
    aggregates, conditional, comparisons, phases = [], [], [], []
    metrics = [
        c for c in frame if c.endswith(("_crps", "_brier", "coverage90", "above_q95", "below_q05"))
    ]
    metrics += [
        "kernel_path_score",
        "drawdown_twcrps10",
        "drawdown_hit10_prob",
        "drawdown_hit10_actual",
        "recovery_prob",
        "recovery_actual",
        "decision_loss",
        "false_alarm",
        "missed_event",
    ]
    metrics = list(dict.fromkeys(metrics))
    for (h, method), group in sub.groupby(["horizon", "method"], sort=True):
        row = {
            "horizon": int(h),
            "method": method,
            "origins": len(group),
            **{m: None if group[m].isna().all() else float(group[m].mean()) for m in metrics},
        }
        row["recovery_origins"] = int(group.recovery_prob.notna().sum())
        row["always_alert_loss"] = float((1 - group.drawdown_hit10_actual).mean())
        row["never_alert_loss"] = float((5 * group.drawdown_hit10_actual).mean())
        row["crisis_peak_dates"] = sorted(
            group.loc[group.max_drawdown_above_q95 > 0, "max_drawdown_peak_date"].unique().tolist()
        )
        aggregates.append(row)
        for field in (
            "initial_vol_group",
            "initial_vol_direction",
            "initial_trend",
            "initial_underwater",
        ):
            for value, part in group.groupby(field):
                conditional.append(
                    {
                        "horizon": int(h),
                        "method": method,
                        "state_variable": field,
                        "state": str(value),
                        "origins": len(part),
                        **{
                            m: float(part[m].mean())
                            for m in (
                                "max_drawdown_crps",
                                "max_drawdown_coverage90",
                                "drawdown_hit10_prob",
                                "drawdown_hit10_actual",
                                "terminal_return_rank",
                            )
                        },
                    }
                )
        overlap = math.ceil(h / config.stride)
        ordered = group.sort_values("origin_index")
        # Phase labels refer to the original forecast grid, not reindexed surviving rows.
        grid_position = ((ordered.origin_index - frame.origin_index.min()) // config.stride).astype(
            int
        )
        for phase in range(overlap):
            part = ordered.loc[(grid_position % overlap) == phase]
            if len(part):
                phases.append(
                    {
                        "method": method,
                        "horizon": int(h),
                        "phase": phase,
                        "origins": len(part),
                        "coverage90": float(part.max_drawdown_coverage90.mean()),
                    }
                )
    for h in config.horizons:
        group = sub.loc[sub.horizon == h]
        pivot = (
            group.pivot(index="origin_index", columns="method", values="max_drawdown_crps")
            .reindex(columns=METHODS)
            .dropna()
            .sort_index()
        )
        if len(pivot) < 3:
            continue
        names = [m for m in METHODS if m != BASELINE]
        diff = pivot[names].to_numpy() - pivot[[BASELINE]].to_numpy()
        for multiplier in (1, 2, 4):
            length = multiplier * math.ceil(h / config.stride)
            mean, lower, upper = simultaneous_intervals(
                diff, length, config.bootstrap_reps, _seed(config.seed, h, multiplier, 202)
            )
            for i, method in enumerate(names):
                base_score = float(pivot[BASELINE].mean())
                deterioration = {}
                for metric in (
                    "terminal_return_crps",
                    "realized_volatility_crps",
                    "kernel_path_score",
                ):
                    p = group.pivot(index="origin_index", columns="method", values=metric).loc[
                        pivot.index
                    ]
                    deterioration[metric] = float(p[method].mean() / p[BASELINE].mean() - 1)
                passed = (
                    mean[i] <= -0.05 * base_score
                    and upper[i] < 0
                    and max(deterioration.values()) <= 0.02
                )
                comparisons.append(
                    {
                        "horizon": h,
                        "method": method,
                        "block_length": length,
                        "common_origins": len(pivot),
                        "mean_difference": float(mean[i]),
                        "relative_change": float(mean[i] / base_score),
                        "simultaneous_lower95": float(lower[i]),
                        "simultaneous_upper95": float(upper[i]),
                        "guardrail_relative_changes": deterioration,
                        "numeric_screen_pass": bool(passed),
                        "default_promotion": False,
                    }
                )
    return {
        "evaluation_start": evaluation_start,
        "aggregate": aggregates,
        "conditional": conditional,
        "simultaneous_comparisons": comparisons,
        "nonoverlap_phases": phases,
        "default_promoted": False,
        "caveats": [
            "Known history: exploratory pseudo-out-of-sample, not untouched validation.",
            "Aggregate uses all available origins per model; comparisons use common origins.",
            "Approximate simultaneous bootstrap intervals require dependence assumptions.",
            "Phase views and global-market crisis observations are not independent replications.",
            "Alarm costs are abstract predeclared decision losses, not trading P&L.",
        ],
    }
