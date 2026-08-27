from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.data.exceptions import DataValidationError
from trading_agent.indicators import IndicatorConfig, IndicatorsEngine


def bars(symbol: str = "AAPL", closes: list[float] | None = None) -> pd.DataFrame:
    closes = closes or [100.0, 102.0, 101.0, 103.0, 105.0, 104.0]
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        }
    )


def calculate(data: pd.DataFrame, *configs: str | IndicatorConfig) -> pd.DataFrame:
    return IndicatorsEngine().calculate(data, indicators=list(configs))


def test_sma_known_value_window_and_warmup() -> None:
    result = calculate(bars(closes=[1, 2, 3, 4]), IndicatorConfig("sma", 3))

    assert result["sma_3"].isna().tolist() == [True, True, False, False]
    assert result["sma_3"].tolist()[2:] == [2.0, 3.0]


def test_ema_uses_recursive_adjust_false_initialization() -> None:
    result = calculate(bars(closes=[1, 2, 3, 4]), IndicatorConfig("ema", 3))

    assert result["ema_3"].isna().tolist() == [True, True, False, False]
    assert result.loc[2, "ema_3"] == pytest.approx(2.25)
    assert result.loc[3, "ema_3"] == pytest.approx(3.125)


def test_rsi_wilder_known_values_and_warmup() -> None:
    result = calculate(bars(closes=[1, 2, 1, 3, 2]), IndicatorConfig("rsi", 2))

    assert result["rsi_2"].isna().tolist()[:2] == [True, True]
    assert result.loc[2, "rsi_2"] == pytest.approx(50.0)
    assert result.loc[3, "rsi_2"] == pytest.approx(83.3333333333)


def test_momentum_and_returns_known_values() -> None:
    result = calculate(bars(closes=[100, 110, 121]), IndicatorConfig("momentum", 2), "returns")

    assert np.isnan(result.loc[0, "returns"])
    assert result["returns"].tolist()[1:] == pytest.approx([0.10, 0.10])
    assert result.loc[2, "momentum_2"] == pytest.approx(0.21)


def test_atr_uses_true_range_and_wilder_smoothing() -> None:
    data = bars(closes=[9, 11, 10])
    data["high"] = [10, 13, 14]
    data["low"] = [8, 10, 9]
    result = calculate(data, IndicatorConfig("atr", 2))

    # True ranges are 2, 4, and 5; ATR(2) is 3 then (3 + 5) / 2 = 4.
    assert result["atr_2"].isna().tolist() == [True, False, False]
    assert result["atr_2"].tolist()[1:] == pytest.approx([3.0, 4.0])


@pytest.mark.parametrize(
    "config",
    [
        IndicatorConfig("sma", 2),
        IndicatorConfig("ema", 2),
        IndicatorConfig("rsi", 2),
        IndicatorConfig("momentum", 2),
        IndicatorConfig("atr", 2),
        IndicatorConfig("returns"),
    ],
)
def test_indicators_are_independent_per_symbol(config: IndicatorConfig) -> None:
    aapl = bars("AAPL", [1, 2, 3])
    spy = bars("SPY", [100, 200, 300])
    result = calculate(pd.concat([aapl, spy], ignore_index=True), config)
    aapl_only = calculate(aapl, config)
    column = config.column_name()

    pd.testing.assert_series_equal(
        result.loc[result["symbol"] == "AAPL", column].reset_index(drop=True),
        aapl_only[column],
        check_names=False,
    )


def test_unsorted_input_matches_sorted_calculation() -> None:
    ordered = pd.concat([bars("AAPL"), bars("SPY")], ignore_index=True)
    unordered = ordered.sample(frac=1, random_state=7).reset_index(drop=True)

    expected = calculate(ordered, IndicatorConfig("momentum", 2))
    actual = calculate(unordered, IndicatorConfig("momentum", 2))

    pd.testing.assert_frame_equal(actual, expected)


def test_anti_lookahead_value_does_not_change_when_future_rows_are_added() -> None:
    initial = bars(closes=[100, 102, 104, 106])
    extended = pd.concat([initial, bars(closes=[100, 102, 104, 106, 10, 1])], ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")

    before = calculate(initial, IndicatorConfig("sma", 3))
    after = calculate(extended, IndicatorConfig("sma", 3))

    assert after.loc[after["date"] == pd.Timestamp("2024-01-04"), "sma_3"].iloc[0] == before.loc[3, "sma_3"]


def test_duplicate_symbol_and_date_is_rejected() -> None:
    data = pd.concat([bars(), bars().iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="Duplicate"):
        calculate(data, "sma")


@pytest.mark.parametrize(
    "invalid_data",
    [
        lambda: bars().drop(columns="close"),
        lambda: bars().assign(close="not-a-number"),
        lambda: bars().assign(high=lambda frame: frame["low"] - 1),
        lambda: bars().assign(close=np.nan),
        lambda: bars().assign(volume=np.nan),
        lambda: bars().assign(date="not-a-date"),
    ],
)
def test_invalid_normalized_input_is_rejected(invalid_data) -> None:
    with pytest.raises((DataValidationError, ValueError)):
        calculate(invalid_data(), "sma")


def test_daily_input_rejects_timezone_and_intraday_dates() -> None:
    timezone_aware = bars()
    timezone_aware["date"] = timezone_aware["date"].dt.tz_localize("America/New_York")
    with pytest.raises(ValueError, match="timezone-naive"):
        calculate(timezone_aware, "sma")

    intraday = bars()
    intraday["date"] = intraday["date"].map(lambda timestamp: timestamp.replace(hour=1))
    with pytest.raises(ValueError, match="intraday"):
        calculate(intraday, "sma")


def test_calculation_is_reproducible() -> None:
    data = bars()
    first = calculate(data, "sma", "ema", "rsi", "momentum", "atr", "returns")
    second = calculate(data, "sma", "ema", "rsi", "momentum", "atr", "returns")

    pd.testing.assert_frame_equal(first, second)
