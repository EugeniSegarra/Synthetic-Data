"""Data-boundary and offline-demo tests; no network calls are made here."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from synthetic_market.data import (
    OHLC_FIELDS,
    MarketData,
    download_yahoo,
    make_demo_data,
    read_csv,
    write_csv,
)


@pytest.fixture
def simple_close():
    return pd.DataFrame(
        {"B": [100.0, 101.0, 99.0], "A": [70.0, 71.0, 72.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )


def test_contract_copies_inputs_and_preserves_asset_order(simple_close):
    metadata = {"price_basis": "adjusted", "nested": {"value": 1}}
    data = MarketData(simple_close, metadata=metadata)
    assert data.assets == ("B", "A")
    simple_close.iloc[0, 0] = 999.0
    metadata["nested"]["value"] = 2
    assert data.close.iloc[0, 0] == 100.0
    assert data.metadata["nested"]["value"] == 1
    assert data.close.index.name == "Date"


@pytest.mark.parametrize(
    "kind", ["unsorted", "duplicates", "nan", "zero", "inf", "string", "bool", "complex"]
)
def test_bad_prices_are_rejected(simple_close, kind):
    frame = simple_close.copy()
    if kind == "unsorted":
        frame = frame.iloc[::-1]
    elif kind == "duplicates":
        frame.index = [frame.index[0]] * len(frame)
    elif kind in {"nan", "zero", "inf"}:
        frame.iloc[1, 0] = {"nan": np.nan, "zero": 0.0, "inf": np.inf}[kind]
    elif kind == "string":
        frame["B"] = ["100", "101", "99"]
    elif kind == "complex":
        frame["B"] = np.array([100 + 1j, 101 + 2j, 99 + 1j])
    else:
        frame["B"] = True
    with pytest.raises(ValueError):
        MarketData(frame)


def test_one_asset_and_constant_prices_are_legal_data(simple_close):
    one = simple_close[["B"]] * 0 + 100
    assert MarketData(one).assets == ("B",)


def test_duplicate_and_nonstring_asset_names_are_rejected(simple_close):
    simple_close.columns = ["A", "A"]
    with pytest.raises(ValueError, match="unique"):
        MarketData(simple_close)
    simple_close.columns = ["A", 1]
    with pytest.raises(ValueError, match="string"):
        MarketData(simple_close)


def test_intraday_rows_on_same_date_are_rejected(simple_close):
    simple_close.index = pd.DatetimeIndex(
        ["2020-01-01 09:00", "2020-01-01 16:00", "2020-01-02 16:00"]
    )
    with pytest.raises(ValueError, match="one observation per calendar date"):
        MarketData(simple_close)


@pytest.mark.parametrize("timezone", ["UTC", "America/New_York"])
def test_timezone_aware_dates_are_rejected_without_silent_normalization(simple_close, timezone):
    simple_close.index = pd.date_range("2024-03-09", periods=3, tz=timezone)
    with pytest.raises(ValueError, match="timezone-naive session-date labels"):
        MarketData(simple_close)


def test_midday_daily_timestamps_require_explicit_session_date_conversion(simple_close):
    simple_close.index = pd.date_range("2024-03-09 12:00:00", periods=3)
    with pytest.raises(ValueError, match="midnight session-date labels"):
        MarketData(simple_close)


def test_csv_timezone_offsets_across_dst_are_rejected_explicitly(tmp_path):
    path = tmp_path / "dst.csv"
    path.write_text(
        "Date,A\n2024-03-09 00:00:00-05:00,100\n"
        "2024-03-10 00:00:00-05:00,101\n2024-03-11 00:00:00-04:00,102\n"
    )
    with pytest.raises(ValueError, match="timezone-naive session-date labels"):
        read_csv(path)


@pytest.mark.parametrize(
    "suffix", [" 00:00:00 EST", " 00:00:00 CET", " GMT", " PST", "Z", "z", " 12:00:00", " 00:00:00"]
)
def test_csv_date_whitelist_rejects_timezone_abbreviations_and_timestamps(tmp_path, suffix):
    path = tmp_path / "timezone.csv"
    path.write_text(f"Date,A\n2020-01-01{suffix},100\n2020-01-02{suffix},101\n")
    with pytest.raises(ValueError, match="exact YYYY-MM-DD"):
        read_csv(path)


def test_bad_ohlc_order_and_close_are_rejected():
    data = make_demo_data(10)
    invalid = data.ohlc.copy()
    invalid.iloc[0, invalid.columns.get_loc("High")] = invalid.iloc[0]["Low"]
    with pytest.raises(ValueError, match="OHLC violates"):
        MarketData(data.close, invalid)
    inconsistent = data.ohlc.copy()
    inconsistent["Close"] *= 1.000001
    # Keep extrema permissive, so this isolates Close agreement.
    inconsistent["High"] *= 2
    inconsistent["Low"] /= 2
    with pytest.raises(ValueError, match="does not agree"):
        MarketData(data.close, inconsistent)
    with pytest.raises(ValueError, match="complete ordered"):
        MarketData(data.close, data.ohlc.iloc[::-1])
    huge_high = data.ohlc.copy()
    huge_high.iloc[0, huge_high.columns.get_loc("High")] = 1e20
    huge_high.iloc[0, huge_high.columns.get_loc("Low")] = 2 * huge_high.iloc[0]["Open"]
    with pytest.raises(ValueError, match="OHLC violates"):
        MarketData(data.close, huge_high)


def test_demo_is_deterministic_valid_and_does_not_touch_global_rng():
    np.random.seed(71)
    before = np.random.get_state()
    first = make_demo_data(150, seed=123)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    second = make_demo_data(150, seed=123)
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != make_demo_data(150, seed=124).fingerprint()
    assert "SYNTHETIC" in first.metadata["source"]
    assert first.ohlc.shape == (450, 4)
    assert (first.ohlc.High >= first.ohlc[["Open", "Close"]].max(axis=1)).all()
    assert (first.ohlc.Low <= first.ohlc[["Open", "Close"]].min(axis=1)).all()
    assert (first.ohlc > 0).all().all()


@pytest.mark.parametrize("n_obs", [0, 1, -1, True, 10.5])
def test_invalid_demo_size_rejected(n_obs):
    with pytest.raises(ValueError):
        make_demo_data(n_obs)


def test_subset_is_inclusive_and_copied():
    data = make_demo_data(30)
    sub = data.subset(data.close.index[4], data.close.index[9])
    assert len(sub.close) == 6
    assert len(sub.ohlc) == 18
    pd.testing.assert_frame_equal(sub.close, data.close.iloc[4:10])
    sub.close.iloc[0, 0] = 9999
    assert data.close.iloc[4, 0] != 9999
    with pytest.raises(ValueError, match="two observations"):
        data.subset(data.close.index[0], data.close.index[0])


def test_fingerprint_tracks_data_and_convention_not_filename():
    data = make_demo_data(30)
    before = data.fingerprint()
    data.metadata["filename"] = "elsewhere.csv"
    data.metadata["request_time"] = "future"
    assert data.fingerprint() == before
    data.metadata["price_basis"] = "adjusted"
    assert data.fingerprint() != before
    altered = MarketData(data.close * 1.01, metadata=data.metadata)
    assert altered.fingerprint() != MarketData(data.close, metadata=data.metadata).fingerprint()


def test_fingerprint_normalizes_datetime_resolution(simple_close):
    first = MarketData(simple_close)
    second = simple_close.copy()
    second.index = second.index.as_unit("s")
    assert first.fingerprint() == MarketData(second).fingerprint()


@pytest.mark.parametrize("has_ohlc", [True, False])
def test_csv_round_trip_preserves_exact_prices(tmp_path, has_ohlc):
    original = make_demo_data(25)
    if not has_ohlc:
        original = MarketData(original.close, metadata=original.metadata)
    path = tmp_path / "panel.csv"
    write_csv(original, path)
    reread = read_csv(path, price_basis="synthetic")
    pd.testing.assert_frame_equal(original.close, reread.close, check_freq=False)
    if has_ohlc:
        pd.testing.assert_frame_equal(original.ohlc, reread.ohlc, check_freq=False)
    assert reread.fingerprint() == original.fingerprint()


def test_csv_missing_dates_require_explicit_intersection(tmp_path):
    path = tmp_path / "gaps.csv"
    path.write_text("Date,A,B\n2020-01-01,10,\n2020-01-02,11,21\n2020-01-03,12,22\n")
    with pytest.raises(ValueError, match="intersection"):
        read_csv(path)
    result = read_csv(path, alignment="intersection")
    assert len(result.close) == 2
    assert result.assets == ("A", "B")
    assert result.metadata["alignment"]["dropped_dates_count"] == 1


def test_csv_intersection_cannot_bridge_internal_missing_sessions(tmp_path):
    path = tmp_path / "gaps.csv"
    path.write_text("Date,A,B\n2020-01-01,10,20\n2020-01-02,11,\n2020-01-03,12,22\n")
    with pytest.raises(ValueError, match="would bridge missing sessions"):
        read_csv(path, alignment="intersection")


def test_csv_intersection_trims_both_edges_only(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text(
        "Date,A,B\n2020-01-01,10,\n2020-01-02,11,21\n2020-01-03,12,22\n2020-01-04,,23\n"
    )
    result = read_csv(path, alignment="intersection")
    assert list(result.close.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert result.metadata["alignment"]["dropped_dates_count"] == 2


def test_numeric_asset_strings_keep_leading_zeroes(tmp_path):
    path = tmp_path / "numeric_names.csv"
    path.write_text(
        "Date,Asset,Open,High,Low,Close\n2020-01-01,000001,10,11,9,10\n2020-01-02,000001,10,11,9,10.5\n"
    )
    result = read_csv(path)
    assert result.assets == ("000001",)
    assert result.close.index[0] == pd.Timestamp("2020-01-01")


def test_compact_numeric_dates_are_rejected_not_interpreted_as_nanoseconds(tmp_path):
    path = tmp_path / "numeric_dates.csv"
    path.write_text("Date,A\n20200101,100\n20200102,101\n")
    with pytest.raises(ValueError, match="exact YYYY-MM-DD"):
        read_csv(path)


@pytest.mark.parametrize(
    "text",
    [
        "Date,A,A\n2020-01-01,1,2\n2020-01-02,3,4\n",
        "Date,,A\n2020-01-01,1,2\n2020-01-02,3,4\n",
        "Date,A\n2020-01-02,1\n2020-01-01,2\n",
        "Date,A\n2020-01-01,1\n2020-01-01,2\n",
        "Date,Asset,Open,High,Low,Close\n2020-01-01,A,1,2,0.5,1\n2020-01-01,A,1,2,0.5,1\n",
    ],
)
def test_bad_csv_headers_dates_and_duplicates_fail(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_csv(path)


def test_empty_asset_never_silently_removed(tmp_path):
    path = tmp_path / "empty_asset.csv"
    path.write_text("Date,A,B\n2020-01-01,10,\n2020-01-02,11,\n2020-01-03,12,\n")
    with pytest.raises(ValueError, match="No usable prices"):
        read_csv(path, alignment="intersection")


def _mock_yahoo_frame():
    data = make_demo_data(10)
    # A typical yfinance schema has field first and ticker second.
    frame = data.ohlc.reset_index()
    original_assets = ["DEMO_EQUITY", "DEMO_BONDS", "DEMO_GOLD"]
    frame["Asset"] = frame.Asset.map(dict(zip(original_assets, ["AAA", "BBB", "CCC"])))
    raw = frame.pivot(index="Date", columns="Asset", values=list(OHLC_FIELDS))
    return raw


def test_yahoo_explicit_adjustment_schema_and_requested_asset_order(monkeypatch):
    calls = []
    raw = _mock_yahoo_frame()

    def download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return raw

    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(download=download, __version__="test")
    )
    result = download_yahoo(["CCC", "AAA"], "2020-01-01", "2021-01-01")
    assert result.assets == ("CCC", "AAA")
    assert result.metadata["price_basis"] == "adjusted"
    assert result.metadata["requested_end_exclusive"] == "2021-01-01"
    assert calls[0][1]["auto_adjust"] is True
    assert calls[0][1]["threads"] is False
    assert calls[0][1]["prepost"] is False
    assert calls[0][1]["actions"] is False
    assert calls[0][1]["repair"] is False
    assert calls[0][1]["interval"] == "1d"
    result.validate()


@pytest.mark.parametrize("layout", ["field_first", "ticker_first", "single"])
def test_yahoo_schema_variants(monkeypatch, layout):
    raw = _mock_yahoo_frame()
    tickers = ["AAA"] if layout == "single" else ["AAA", "BBB"]
    if layout == "ticker_first":
        raw = raw.swaplevel(0, 1, axis=1)
    elif layout == "single":
        raw = raw.xs("AAA", axis=1, level=1)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: raw))
    result = download_yahoo(tickers, "2020-01-01")
    assert result.assets == tuple(tickers)


def test_yahoo_no_synthetic_fallback_and_no_missing_assets(monkeypatch):
    raw = _mock_yahoo_frame()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: raw))
    with pytest.raises(ValueError, match="omitted requested assets"):
        download_yahoo(["AAA", "NOT_PRESENT"], "2020-01-01")
    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
    )
    with pytest.raises(ValueError, match="no observations"):
        download_yahoo(["AAA"], "2020-01-01")


def test_yahoo_missing_observations_record_intersection(monkeypatch):
    raw = _mock_yahoo_frame()
    raw.loc[raw.index[0], ("Open", "AAA")] = np.nan
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: raw))
    with pytest.raises(ValueError, match="Missing observations"):
        download_yahoo(["AAA", "BBB"], "2020-01-01")
    result = download_yahoo(["AAA", "BBB"], "2020-01-01", alignment="intersection")
    assert len(result.close) == len(raw) - 1
    assert result.metadata["alignment"]["dropped_dates_count"] == 1


def test_yahoo_cannot_bridge_missing_internal_ohlc_observations(monkeypatch):
    raw = _mock_yahoo_frame()
    raw.loc[raw.index[3], ("Open", "AAA")] = np.nan
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: raw))
    with pytest.raises(ValueError, match="would bridge missing sessions"):
        download_yahoo(["AAA", "BBB"], "2020-01-01", alignment="intersection")


@pytest.mark.parametrize("tickers", [[], ["AAA", "AAA"], [" AAA"], [None]])
def test_invalid_ticker_requests_rejected_before_network(tickers):
    with pytest.raises(ValueError):
        download_yahoo(tickers, "2020-01-01")
