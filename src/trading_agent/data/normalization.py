"""Conversion of provider responses to the normalized OHLCV contract."""

import pandas as pd

from trading_agent.data.exceptions import DataValidationError, EmptyDataError
from trading_agent.data.quality import REQUIRED_COLUMNS, require_valid_data

_COLUMN_ALIASES = {
    "date": "date",
    "datetime": "date",
    "symbol": "symbol",
    "ticker": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def normalize_ohlcv(raw_data: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """Return sorted, typed OHLCV data, rejecting malformed inputs explicitly.

    Inputs must contain the usual OHLCV fields and either a date column or a
    datetime index. A symbol may be supplied for providers that return one
    ticker per response.
    """
    if raw_data.empty:
        raise EmptyDataError("Cannot normalize an empty market-data response.")

    data = raw_data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        raise DataValidationError("Multi-index columns must be flattened by the provider first.")
    if "date" not in {str(column).lower() for column in data.columns}:
        data = data.reset_index()

    renamed = {column: _COLUMN_ALIASES.get(str(column).lower(), str(column).lower()) for column in data.columns}
    data = data.rename(columns=renamed)
    if symbol is not None:
        data["symbol"] = symbol.strip().upper()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}.")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    data["date"] = _normalize_daily_dates(data["date"])

    data["symbol"] = data["symbol"].astype("string").str.strip().str.upper()
    if data["symbol"].eq("").any():
        raise DataValidationError("symbol must not be blank.")

    for column in ("open", "high", "low", "close", "volume"):
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"{column} contains non-numeric values.") from error

    data = data.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    require_valid_data(data)
    return data


def _normalize_daily_dates(values: pd.Series) -> pd.Series:
    """Convert daily source dates to timezone-naive midnight timestamps.

    A timezone-aware midnight timestamp keeps its provider-local calendar date;
    it is never converted through UTC. Any non-midnight timestamp is rejected
    because this package's contract represents daily bars, not intraday data.
    """
    normalized: list[pd.Timestamp] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise DataValidationError("date contains invalid values.") from error
        if pd.isna(timestamp):
            raise DataValidationError("date contains invalid values.")
        if timestamp.hour or timestamp.minute or timestamp.second or timestamp.microsecond or timestamp.nanosecond:
            raise DataValidationError("date must represent a daily date without an intraday time component.")
        # ``date()`` intentionally preserves the timestamp's local calendar day.
        normalized.append(pd.Timestamp(timestamp.date()))
    return pd.Series(normalized, index=values.index, dtype="datetime64[ns]")
