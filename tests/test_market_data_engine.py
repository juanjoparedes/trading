from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from trading_agent.data.cache import FileSystemCache
from trading_agent.data.engine import DataEngine
from trading_agent.data.exceptions import (
    DataOutsideRequestedRangeError,
    DataValidationError,
    EmptyDataError,
    InvalidDateRangeError,
    SymbolNotFoundError,
)
from trading_agent.data.models import DailyDataRequest
from trading_agent.data.normalization import normalize_ohlcv
from trading_agent.data.providers import MarketDataProvider
from trading_agent.data.quality import check_data_quality


def raw_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02"],
            "Open": [102.0, 100.0],
            "High": [104.0, 103.0],
            "Low": [101.0, 99.0],
            "Close": [103.0, 102.0],
            "Volume": [1_100, 1_000],
        }
    )


@dataclass
class FakeProvider(MarketDataProvider):
    data: pd.DataFrame
    cache_key: str = "fake-v1"
    calls: int = 0

    def get_daily_data(self, request: DailyDataRequest) -> pd.DataFrame:
        self.calls += 1
        frames = [normalize_ohlcv(self.data, symbol=symbol) for symbol in request.symbols]
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def test_normalization_produces_expected_columns() -> None:
    normalized = normalize_ohlcv(raw_bars(), symbol="spy")

    assert normalized.columns.tolist() == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert normalized["symbol"].tolist() == ["SPY", "SPY"]


def test_normalization_orders_dates_ascending() -> None:
    normalized = normalize_ohlcv(raw_bars(), symbol="SPY")

    assert normalized["date"].tolist() == sorted(normalized["date"].tolist())


def test_normalization_rejects_duplicate_symbol_and_date() -> None:
    duplicate = pd.concat([raw_bars(), raw_bars().iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="Duplicate symbol"):
        normalize_ohlcv(duplicate, symbol="SPY")


def test_normalization_returns_numeric_ohlc_and_volume() -> None:
    normalized = normalize_ohlcv(raw_bars(), symbol="SPY")

    assert all(pd.api.types.is_numeric_dtype(normalized[column]) for column in ["open", "high", "low", "close", "volume"])


def test_empty_frame_raises_clear_error() -> None:
    with pytest.raises(EmptyDataError, match="empty"):
        normalize_ohlcv(pd.DataFrame(), symbol="SPY")


def test_invalid_request_dates_are_rejected() -> None:
    with pytest.raises(InvalidDateRangeError, match="YYYY-MM-DD"):
        DailyDataRequest.create(["SPY"], "01/01/2024", "2024-02-01")


@pytest.mark.parametrize(
    ("start", "end"),
    [("2024-01-01", "2024-01-01"), ("2024-02-01", "2024-01-01")],
)
def test_equal_or_reversed_request_dates_are_rejected(start: str, end: str) -> None:
    with pytest.raises(InvalidDateRangeError, match="earlier"):
        DailyDataRequest.create(["SPY"], start, end)


def test_symbol_without_data_is_reported() -> None:
    class MissingSymbolProvider(MarketDataProvider):
        cache_key = "missing-v1"

        def get_daily_data(self, request: DailyDataRequest) -> pd.DataFrame:
            raise SymbolNotFoundError("No data returned for symbol(s): MISSING.")

    engine = DataEngine(MissingSymbolProvider())
    with pytest.raises(SymbolNotFoundError, match="MISSING"):
        engine.get_daily_data(symbols=["MISSING"], start="2024-01-01", end="2024-02-01")


def test_engine_normalization_path_has_no_network_dependency() -> None:
    provider = FakeProvider(raw_bars())
    engine = DataEngine(provider)

    result = engine.get_daily_data(symbols=["SPY", "QQQ"], start="2024-01-01", end="2024-02-01")

    assert provider.calls == 1
    assert result["symbol"].tolist() == ["QQQ", "QQQ", "SPY", "SPY"]


def test_cache_prevents_identical_request_from_calling_provider_twice(tmp_path) -> None:
    provider = FakeProvider(raw_bars())
    engine = DataEngine(provider, cache=FileSystemCache(tmp_path / "cache"))

    first = engine.get_daily_data(symbols=["SPY"], start="2024-01-01", end="2024-02-01")
    second = engine.get_daily_data(symbols=["SPY"], start="2024-01-01", end="2024-02-01")

    assert provider.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_quality_report_flags_inconsistent_ohlc() -> None:
    data = normalize_ohlcv(raw_bars(), symbol="SPY")
    data.loc[0, "high"] = 90.0

    report = check_data_quality(data)

    assert not report.valid
    assert any("high" in issue for issue in report.issues)


@pytest.mark.parametrize("outside_date", ["2023-12-31", "2025-01-01", "2025-01-02"])
def test_engine_rejects_data_outside_half_open_request_range(outside_date: str) -> None:
    data = raw_bars()
    data.loc[0, "Date"] = outside_date
    engine = DataEngine(FakeProvider(data))

    with pytest.raises(DataOutsideRequestedRangeError, match="outside the requested"):
        engine.get_daily_data(symbols=["SPY"], start="2024-01-01", end="2025-01-01")


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Volume"])
@pytest.mark.parametrize("infinite_value", [float("inf"), float("-inf")])
def test_normalization_rejects_infinite_ohlcv_values(column: str, infinite_value: float) -> None:
    data = raw_bars()
    data[column] = data[column].astype(float)
    data.loc[0, column] = infinite_value

    with pytest.raises(DataValidationError, match="finite"):
        normalize_ohlcv(data, symbol="SPY")


def test_normalization_represents_daily_dates_without_time_or_timezone() -> None:
    normalized = normalize_ohlcv(raw_bars(), symbol="SPY")

    assert str(normalized["date"].dtype) == "datetime64[ns]"
    assert normalized["date"].dt.tz is None
    assert (normalized["date"].dt.time == pd.Timestamp("00:00").time()).all()


def test_timezone_aware_midnight_preserves_provider_local_trading_date() -> None:
    data = raw_bars()
    data.loc[0, "Date"] = "2024-01-03 00:00:00-05:00"

    normalized = normalize_ohlcv(data, symbol="SPY")

    assert normalized.loc[1, "date"] == pd.Timestamp("2024-01-03")
    assert normalized["date"].dt.tz is None


def test_intraday_timestamp_is_rejected_for_daily_contract() -> None:
    data = raw_bars()
    data.loc[0, "Date"] = "2024-01-03 15:30:00"

    with pytest.raises(DataValidationError, match="intraday"):
        normalize_ohlcv(data, symbol="SPY")
