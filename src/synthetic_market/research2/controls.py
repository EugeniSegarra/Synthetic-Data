"""Separate controls for uncertainty, event recalibration and probability-free stress."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..path_validation import path_features, stationary_indices
from .generators import BASELINE, _positive, _series, shrunk_drift


def isotonic_fit(probabilities, events):
    """Least-squares monotone calibration by weighted pool-adjacent-violators.

    Equal forecast probabilities are aggregated before fitting. Returned values
    interpolate monotonically, with endpoint extrapolation. No calibration guarantee
    for financial distribution shift is claimed.
    """
    p, y = _series(probabilities), _series(events)
    if len(p) != len(y) or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("probabilities must be [0,1]; outcomes binary and aligned")
    x, inverse = np.unique(p, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    sums = np.bincount(inverse, weights=y)
    blocks = []
    for i in range(len(x)):
        blocks.append([i, i + 1, counts[i], sums[i]])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.empty(len(x))
    for a, b, n, total in blocks:
        fitted[a:b] = total / n
    return x, fitted


def delayed_recalibration(frame, memory=60, minimum=24, pseudocount=200):
    """Event-probability comparator only, NEVER a recalibrated joint price law."""
    _positive(memory, "memory")
    _positive(minimum, "minimum")
    _positive(pseudocount, "pseudocount")
    if memory < minimum:
        raise ValueError("memory must be >= minimum")
    records = []
    base = frame.loc[frame.method == BASELINE].copy()
    if base.duplicated(["origin_index", "horizon"]).any():
        raise ValueError("duplicate baseline origins")
    for h, group in base.groupby("horizon"):
        group = group.sort_values("origin_index")
        for row in group.itertuples():
            available = group.loc[group.origin_index + h <= row.origin_index].tail(memory)
            p = float(row.drawdown_hit10_prob)
            if len(available) >= minimum:
                x, y = isotonic_fit(available.drawdown_hit10_prob, available.drawdown_hit10_actual)
                calibrated = float(np.interp(p, x, y))
                weight = len(available) / (len(available) + pseudocount)
                q = (1 - weight) * p + weight * calibrated
            else:
                q = p
            records.append(
                {
                    "origin": row.origin,
                    "origin_index": row.origin_index,
                    "horizon": int(h),
                    "target_end": row.target_end,
                    "raw_probability": p,
                    "recalibrated_probability": q,
                    "event": int(row.drawdown_hit10_actual),
                    "raw_brier": (p - row.drawdown_hit10_actual) ** 2,
                    "recalibrated_brier": (q - row.drawdown_hit10_actual) ** 2,
                    "matured_count": len(available),
                    "last_matured_index": None
                    if available.empty
                    else int((available.origin_index + h).max()),
                }
            )
    return pd.DataFrame(records)


def bootstrap_drift(history, reps=200, mean_block=21, prior_sessions=5040, seed=42):
    """Block-bootstrap sensitivity of the regularized mean, NOT all parameter risk.

    Resampling the observed series assumes local stationarity and does not add
    new economic regimes. No Gaussian or Bayesian posterior claim is made.
    """
    r = _series(history, 253)
    _positive(reps, "reps", 20)
    indices = stationary_indices(len(r), reps, len(r), mean_block, seed)
    estimates = np.array([shrunk_drift(r[row], prior_sessions) for row in indices])
    return estimates


def stress_paths(horizon=126):
    """Declared hypothetical scenarios; values are log returns; no probabilities."""
    _positive(horizon, "horizon", 10)
    shock = np.zeros(horizon)
    shock[0] = np.log(0.8)
    clustered = np.zeros(horizon)
    clustered[:5] = np.log(0.96)
    recovery = clustered.copy()
    recovery[5:] = (-clustered.sum()) / (horizon - 5)
    return {
        "instant_20pct_loss": shock,
        "five_consecutive_4pct_losses": clustered,
        "five_4pct_losses_then_full_recovery": recovery,
    }


def stress_report(horizon=126):
    return {
        name: {
            "probability": None,
            "kind": "hypothetical_not_estimated",
            **{k: float(v[0]) for k, v in path_features(path).items()},
        }
        for name, path in stress_paths(horizon).items()
    }
