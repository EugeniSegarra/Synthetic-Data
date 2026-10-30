"""Strict daily market-data contracts, explicit Yahoo adjustment, and an offline demo.

Dates must be timezone-naive midnight session-date labels, not raw timestamps.
Map timestamps to the appropriate local trading session explicitly before loading.
These utilities do not infer a trading calendar, forward-fill prices, reconstruct
a historical index universe, or hide missing assets.
Yahoo's adjusted OHLC are adjusted price series, not historical executable quotes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OHLC_FIELDS = ("Open", "High", "Low", "Close")


def _dates(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Prices require a DatetimeIndex of observation dates.")
    if index.tz is not None:
        raise ValueError(
            "Daily data require timezone-naive session-date labels. Convert timestamps to "
            "the appropriate local trading session dates explicitly before supplying data."
        )
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Observation dates must be nonmissing, unique, and increasing.")
    if index.normalize().has_duplicates:
        raise ValueError(
            "Daily data permit only one observation per calendar date, not intraday rows."
        )
    if not index.equals(index.normalize()):
        raise ValueError(
            "Daily data require midnight session-date labels, not raw intraday timestamps. "
            "Map timestamps to local trading session dates explicitly before supplying data."
        )
    return index


def _numeric_prices(frame: pd.DataFrame, label: str) -> np.ndarray:
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise ValueError(
            f"{label} must contain real numeric prices, not strings, booleans, or complex numbers."
        )
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label} must contain only finite, strictly positive prices.")
    return values


@dataclass
class MarketData:
    """A complete common-date panel, with optional observed daily OHLC.

    ``close`` has timezone-naive midnight session dates as rows and ordered asset
    names as columns. User timestamps are never silently normalized. ``ohlc`` has
    a (Date, Asset) MultiIndex in the exact date-major, asset-column order of
    ``close``. Inputs are defensively copied. The exposed pandas objects remain
    mutable; call ``validate`` after editing, or construct a fresh MarketData.
    """

    close: pd.DataFrame
    ohlc: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.close, pd.DataFrame):
            raise ValueError("close must be a pandas DataFrame.")
        if self.ohlc is not None and not isinstance(self.ohlc, pd.DataFrame):
            raise ValueError("ohlc must be a pandas DataFrame or None.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        self.close = self.close.copy(deep=True)
        self.ohlc = None if self.ohlc is None else self.ohlc.copy(deep=True)
        self.metadata = deepcopy(self.metadata)
        self.metadata.setdefault("price_basis", "unspecified")
        self.metadata.setdefault("frequency", "daily_observations")
        self.validate()
        self.close = self.close.astype(float)
        self.close.index.name = "Date"
        self.close.columns.name = "Asset"
        if self.ohlc is not None:
            self.ohlc = self.ohlc.astype(float)

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.close.columns)

    def validate(self) -> None:
        """Reject malformed panels; never silently sort, fill, or remove assets."""
        index = _dates(self.close.index)
        if len(index) < 2 or self.close.shape[1] < 1:
            raise ValueError("At least two observations and one asset are required.")
        if isinstance(self.close.columns, pd.MultiIndex) or self.close.columns.has_duplicates:
            raise ValueError("Asset columns must be a unique one-dimensional index.")
        if any(not isinstance(a, str) or not a.strip() for a in self.close.columns):
            raise ValueError("Every asset name must be a nonempty string.")
        close_values = _numeric_prices(self.close, "Close")
        for key in ("price_basis", "frequency"):
            if not isinstance(self.metadata.get(key), str):
                raise ValueError(f"metadata[{key!r}] must be a string.")
        if self.ohlc is None:
            return
        if tuple(self.ohlc.columns) != OHLC_FIELDS:
            raise ValueError(f"OHLC columns must be exactly {OHLC_FIELDS}, in that order.")
        expected = pd.MultiIndex.from_product([index, self.close.columns], names=["Date", "Asset"])
        if not isinstance(self.ohlc.index, pd.MultiIndex) or not self.ohlc.index.equals(expected):
            raise ValueError("OHLC index must be the complete ordered (Date, Asset) panel.")
        if self.ohlc.index.names != ["Date", "Asset"]:
            raise ValueError("OHLC index levels must be named Date and Asset.")
        values = _numeric_prices(self.ohlc, "OHLC")
        opens, highs, lows, closes = values.T
        upper_endpoint = np.maximum(opens, closes)
        lower_endpoint = np.minimum(opens, closes)
        upper_tolerance = 1e-10 * np.maximum(highs, upper_endpoint)
        lower_tolerance = 1e-10 * np.maximum(lows, lower_endpoint)
        if np.any(highs + upper_tolerance < upper_endpoint) or np.any(
            lows - lower_tolerance > lower_endpoint
        ):
            raise ValueError("OHLC violates Low <= min(Open, Close) <= max(Open, Close) <= High.")
        if not np.allclose(closes, close_values.reshape(-1), rtol=1e-10, atol=0.0):
            raise ValueError("OHLC Close does not agree with the close-price panel.")

    def subset(self, start: Any = None, end: Any = None) -> MarketData:
        """Return a copied inclusive date interval, retaining economic metadata."""
        close = self.close.loc[start:end].copy()
        ohlc = None
        if self.ohlc is not None:
            wanted = pd.MultiIndex.from_product(
                [close.index, close.columns], names=["Date", "Asset"]
            )
            ohlc = self.ohlc.reindex(wanted)
        metadata = deepcopy(self.metadata)
        if len(close):
            metadata["observed_start"] = close.index[0].isoformat()
            metadata["observed_end"] = close.index[-1].isoformat()
        return MarketData(close, ohlc, metadata)

    def fingerprint(self) -> str:
        """SHA-256 of values, dates, asset order, OHLC, and economic convention.

        File paths, request times, and provider diagnostics are provenance, not
        economic contents, and deliberately do not change this fingerprint.
        """
        self.validate()
        digest = hashlib.sha256()
        convention = {
            "format": 1,
            "assets": self.assets,
            "timezone": str(self.close.index.tz),
            "price_basis": self.metadata["price_basis"],
            "frequency": self.metadata["frequency"],
            "has_ohlc": self.ohlc is not None,
        }
        digest.update(json.dumps(convention, sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.asarray(self.close.index.as_unit("ns").asi8, dtype="<i8").tobytes())
        digest.update(np.asarray(self.close, dtype="<f8").tobytes(order="C"))
        if self.ohlc is not None:
            digest.update(np.asarray(self.ohlc, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()


def _check_alignment(alignment: str) -> None:
    if alignment not in {"strict", "intersection"}:
        raise ValueError("alignment must be 'strict' or the explicit 'intersection'.")


def _assemble(
    close: pd.DataFrame,
    ohlc: pd.DataFrame | None,
    metadata: dict[str, Any],
    alignment: str,
) -> MarketData:
    _check_alignment(alignment)
    _dates(close.index)
    if close.empty:
        raise ValueError("No observations were supplied.")
    usable_assets = close.notna().any(axis=0)
    missing_assets = close.columns[~usable_assets].tolist()
    if missing_assets:
        raise ValueError(
            f"No usable prices for requested assets: {missing_assets}. No assets were dropped."
        )
    valid = close.notna().all(axis=1).to_numpy()
    if ohlc is not None:
        complete_ohlc = ohlc.notna().all(axis=1).to_numpy().reshape(len(close), close.shape[1])
        no_ohlc = close.columns[~complete_ohlc.any(axis=0)].tolist()
        if no_ohlc:
            raise ValueError(f"No complete OHLC observations for assets: {no_ohlc}.")
        valid &= complete_ohlc.all(axis=1)
    dropped = close.index[~valid]
    if len(dropped) and alignment == "strict":
        preview = [x.isoformat() for x in dropped[:5]]
        raise ValueError(
            f"Missing observations on {len(dropped)} dates (first dates: {preview}). "
            "Use alignment='intersection' explicitly to discard incomplete dates."
        )
    if alignment == "intersection" and valid.any():
        retained_positions = np.flatnonzero(valid)
        internal_missing = ~valid[retained_positions[0] : retained_positions[-1] + 1]
        if internal_missing.any():
            raise ValueError(
                "Intersection would bridge missing sessions inside the retained history. "
                "Only leading/trailing incomplete dates may be trimmed; supply a complete "
                "common daily history instead of converting multi-session returns into one-day returns."
            )
    metadata = deepcopy(metadata)
    metadata["alignment"] = {
        "policy": alignment,
        "input_dates": len(close),
        "retained_dates": int(valid.sum()),
        "dropped_dates_count": len(dropped),
        "dropped_dates": [x.isoformat() for x in dropped],
    }
    close = close.loc[valid]
    if ohlc is not None:
        wanted = pd.MultiIndex.from_product([close.index, close.columns], names=["Date", "Asset"])
        ohlc = ohlc.reindex(wanted)
    if len(close):
        metadata["observed_start"] = close.index[0].isoformat()
        metadata["observed_end"] = close.index[-1].isoformat()
    return MarketData(close, ohlc, metadata)


def read_csv(
    path: str | Path, *, price_basis: str = "unspecified", alignment: str = "strict"
) -> MarketData:
    """Read wide ``Date,A,B,...`` closes or long ``Date,Asset,Open,High,Low,Close``.

    Date fields must use exactly YYYY-MM-DD: no timestamp, offset, or timezone
    abbreviation is inferred or discarded. Long data must be ordered by date;
    within each date assets are canonically
    arranged in their first-occurrence order. No missing data are filled.
    Explicit intersection may trim incomplete edges, never an internal session.
    Dates absent from every asset cannot be identified without an exchange
    calendar; the caller must supply a complete common-session daily history.
    """
    _check_alignment(alignment)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
        raise ValueError("CSV field names must be nonempty and unique.")
    string_fields = {name: "string" for name in ("Date", "Asset") if name in header}
    frame = pd.read_csv(path, float_precision="round_trip", dtype=string_fields)
    if frame.columns.has_duplicates or "Date" not in frame:
        raise ValueError("CSV needs a Date column and unique field names.")
    raw_dates = frame.pop("Date")
    if not raw_dates.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=False).all():
        raise ValueError(
            "CSV Date values must be timezone-naive session-date labels in exact YYYY-MM-DD "
            "format. Convert timestamps to the appropriate local trading session dates "
            "explicitly before supplying data; no timezone is inferred or discarded."
        )
    try:
        dates = pd.DatetimeIndex(
            pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="raise"), name="Date"
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "CSV Date values must be valid calendar dates in exact YYYY-MM-DD format."
        ) from exc
    if dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError("CSV dates must be nonmissing and increasing (date-major for long data).")
    metadata = {
        "source": "csv",
        "filename": Path(path).name,
        "price_basis": price_basis,
        "frequency": "daily_observations",
    }
    if "Asset" not in frame:
        if dates.has_duplicates:
            raise ValueError("Wide CSV dates must be unique.")
        frame.index = dates
        return _assemble(frame, None, metadata, alignment)
    if not set(OHLC_FIELDS).issubset(frame.columns):
        raise ValueError("Long CSV needs Asset, Open, High, Low, and Close columns.")
    labels = frame.pop("Asset")
    if labels.isna().any() or any(not isinstance(a, str) or not a.strip() for a in labels):
        raise ValueError("CSV asset names must be nonempty strings.")
    assets = list(dict.fromkeys(labels))
    panel_index = pd.MultiIndex.from_arrays([dates, labels], names=["Date", "Asset"])
    if panel_index.has_duplicates:
        raise ValueError("Long CSV contains duplicate Date/Asset observations.")
    ohlc = frame.loc[:, list(OHLC_FIELDS)].copy()
    ohlc.index = panel_index
    observation_dates = dates.unique()
    wanted = pd.MultiIndex.from_product([observation_dates, assets], names=["Date", "Asset"])
    ohlc = ohlc.reindex(wanted)
    close = ohlc["Close"].unstack("Asset").reindex(columns=assets)
    return _assemble(close, ohlc, metadata, alignment)


def write_csv(data: MarketData, path: str | Path) -> None:
    """Write exact round-trippable numeric values; metadata remains separate."""
    data.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = data.close if data.ohlc is None else data.ohlc
    frame.reset_index().to_csv(path, index=False, float_format="%.17g")


def download_yahoo(
    tickers: list[str], start: str, end: str | None = None, *, alignment: str = "strict"
) -> MarketData:
    """Download adjusted daily OHLC; ``end`` follows Yahoo's exclusive convention.

    This is a convenience research source, not a point-in-time institutional feed.
    ``auto_adjust=True`` is explicit. All four prices share the same adjustment;
    these are not reconstructed executable historical quotes. Network/provider
    failures and a missing ticker are errors, never synthetic fallbacks.
    Intersection may trim incomplete edges, but never bridge missing sessions.
    """
    _check_alignment(alignment)
    if not isinstance(tickers, (list, tuple)) or not tickers:
        raise ValueError("tickers must be a nonempty list of explicit symbols.")
    if any(not isinstance(t, str) or not t.strip() or t != t.strip() for t in tickers):
        raise ValueError("Ticker symbols must be nonempty strings without outer whitespace.")
    tickers = list(tickers)
    if len(set(tickers)) != len(tickers):
        raise ValueError("Ticker symbols must be unique.")
    start_date = pd.Timestamp(start)
    if pd.isna(start_date):
        raise ValueError("start must be a valid date.")
    if end is not None and (pd.isna(pd.Timestamp(end)) or pd.Timestamp(end) <= start_date):
        raise ValueError("end must be later than start (exclusive).")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Yahoo download requires the optional yfinance dependency.") from exc
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            back_adjust=False,
            repair=False,
            actions=False,
            keepna=True,
            group_by="column",
            multi_level_index=True,
            threads=False,
            progress=False,
            prepost=False,
        )
    except Exception as exc:
        raise RuntimeError("Yahoo download failed; no data were generated or substituted.") from exc
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(
            "Yahoo returned no observations. Check symbols, dates, and provider availability."
        )
    raw = raw.copy()
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("Yahoo returned an unsupported date index.")
    # Daily exchange-local dates are labels, not UTC instants. Retain the labels.
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    _dates(raw.index)
    raw.index.name = "Date"
    asset_frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        if raw.columns.nlevels != 2 or raw.columns.has_duplicates:
            raise ValueError("Yahoo returned an unsupported or duplicate column schema.")
        candidates = [
            level
            for level in (0, 1)
            if set(OHLC_FIELDS).issubset(set(raw.columns.get_level_values(level)))
        ]
        if len(candidates) != 1:
            raise ValueError("Cannot identify Yahoo OHLC field level.")
        field_level = candidates[0]
        asset_level = 1 - field_level
        available = set(raw.columns.get_level_values(asset_level))
        missing = [a for a in tickers if a not in available]
        if missing:
            raise ValueError(f"Yahoo omitted requested assets: {missing}. No assets were dropped.")
        for asset in tickers:
            fields = raw.xs(asset, axis=1, level=asset_level)
            if not set(OHLC_FIELDS).issubset(fields.columns):
                raise ValueError(f"Yahoo returned incomplete OHLC fields for {asset}.")
            asset_frames.append(fields.loc[:, list(OHLC_FIELDS)])
    else:
        if (
            len(tickers) != 1
            or not set(OHLC_FIELDS).issubset(raw.columns)
            or raw.columns.has_duplicates
        ):
            raise ValueError("Yahoo returned an unsupported single-level column schema.")
        asset_frames.append(raw.loc[:, list(OHLC_FIELDS)])
    values = np.stack([f.to_numpy(dtype=float) for f in asset_frames], axis=1)
    panel_index = pd.MultiIndex.from_product([raw.index, tickers], names=["Date", "Asset"])
    ohlc = pd.DataFrame(values.reshape(-1, 4), index=panel_index, columns=OHLC_FIELDS)
    close = pd.DataFrame(values[:, :, 3], index=raw.index, columns=tickers)
    metadata = {
        "source": "Yahoo Finance via yfinance",
        "provider_version": getattr(yf, "__version__", "unknown"),
        "price_basis": "adjusted",
        "frequency": "daily_observations",
        "auto_adjust": True,
        "requested_tickers": tickers,
        "requested_start": start,
        "requested_end_exclusive": end,
        "interpretation": "Adjusted OHLC research series; not executable historical quotes.",
    }
    return _assemble(close, ohlc, metadata, alignment)


def make_demo_data(n_obs: int = 1600, seed: int = 1729) -> MarketData:
    """Generate a reproducible, explicitly synthetic three-asset daily fixture.

    The known DGP is correlated Gaussian GJR(1,1), plus relative/logarithmic
    overnight gaps and excursions. It exercises the software; it is neither a
    downloaded market sample nor evidence of empirical market realism.
    """
    if isinstance(n_obs, bool) or not isinstance(n_obs, (int, np.integer)) or n_obs < 2:
        raise ValueError("n_obs must be an integer of at least two.")
    rng = np.random.default_rng(seed)
    assets = ["DEMO_EQUITY", "DEMO_BONDS", "DEMO_GOLD"]
    dates = pd.bdate_range("2010-01-04", periods=n_obs, name="Date")
    correlation = np.array([[1.0, -0.20, 0.15], [-0.20, 1.0, 0.10], [0.15, 0.10, 1.0]])
    chol = np.linalg.cholesky(correlation)
    target_variance = np.square([0.012, 0.005, 0.009])
    alpha, gamma, beta = 0.05, 0.06, 0.88
    omega = target_variance * (1 - alpha - gamma / 2 - beta)
    mean = np.array([0.0002, 0.00005, 0.0001])
    variance = target_variance.copy()
    previous_residual = np.zeros(3)
    values = np.zeros((n_obs, 3, 4))
    previous_close = np.array([100.0, 80.0, 120.0])
    for t in range(n_obs):
        if t:
            variance = (
                omega
                + (alpha + gamma * (previous_residual < 0)) * previous_residual**2
                + beta * variance
            )
        sigma = np.sqrt(variance)
        residual = sigma * (rng.standard_normal(3) @ chol.T)
        log_return = mean + residual
        gap = 0.25 * log_return + 0.20 * sigma * rng.standard_normal(3)
        opens = previous_close * np.exp(gap)
        closes = previous_close * np.exp(log_return)
        upper = 0.35 * sigma * np.abs(rng.standard_normal(3))
        lower = 0.35 * sigma * np.abs(rng.standard_normal(3))
        highs = np.maximum(opens, closes) * np.exp(upper)
        lows = np.minimum(opens, closes) * np.exp(-lower)
        values[t] = np.column_stack([opens, highs, lows, closes])
        previous_close = closes
        previous_residual = residual
    index = pd.MultiIndex.from_product([dates, assets], names=["Date", "Asset"])
    ohlc = pd.DataFrame(values.reshape(-1, 4), index=index, columns=OHLC_FIELDS)
    close = pd.DataFrame(values[:, :, 3], index=dates, columns=assets)
    metadata = {
        "source": "SYNTHETIC DEMO — not observed market data",
        "price_basis": "synthetic",
        "frequency": "daily_observations",
        "seed": int(seed),
        "demo_dgp": "Correlated Gaussian GJR(1,1) with logarithmic OHLC excursions",
        "demo_calendar": "Illustrative weekday labels; not an exchange calendar",
        "demo_parameters": {"alpha": alpha, "gamma": gamma, "beta": beta},
        "observed_start": dates[0].isoformat(),
        "observed_end": dates[-1].isoformat(),
    }
    return MarketData(close, ohlc, metadata)
