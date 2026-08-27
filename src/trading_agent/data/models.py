"""Typed request models for the market-data layer."""

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

from trading_agent.data.exceptions import InvalidDateRangeError

Timeframe = Literal["1d"]


def parse_iso_date(value: date | datetime | str, field_name: str) -> date:
    """Parse a daily date, preserving the source calendar date without timezone conversion."""
    if isinstance(value, datetime):
        if value.timetz().replace(tzinfo=None) != time.min:
            raise InvalidDateRangeError(
                f"{field_name} must be a daily date or a midnight datetime without intraday time."
            )
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise InvalidDateRangeError(f"{field_name} must be a date or ISO date string.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise InvalidDateRangeError(f"{field_name} must use ISO format YYYY-MM-DD; received {value!r}.") from error


@dataclass(frozen=True, slots=True)
class DailyDataRequest:
    """A provider-independent request for daily historical bars."""

    symbols: tuple[str, ...]
    start: date
    end: date
    timeframe: Timeframe = "1d"

    @classmethod
    def create(
        cls,
        symbols: list[str] | tuple[str, ...],
        start: date | datetime | str,
        end: date | datetime | str,
        timeframe: Timeframe = "1d",
    ) -> "DailyDataRequest":
        normalized_symbols = tuple(symbol.strip().upper() for symbol in symbols if symbol.strip())
        if not normalized_symbols:
            raise ValueError("At least one non-empty symbol is required.")
        if len(set(normalized_symbols)) != len(normalized_symbols):
            raise ValueError("Symbols must be unique within a request.")
        if timeframe != "1d":
            raise ValueError(f"Unsupported timeframe {timeframe!r}; only '1d' is available.")

        start_date = parse_iso_date(start, "start")
        end_date = parse_iso_date(end, "end")
        if start_date >= end_date:
            raise InvalidDateRangeError("start must be earlier than end.")
        return cls(normalized_symbols, start_date, end_date, timeframe)
