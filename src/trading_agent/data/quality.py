"""Validation for the normalized daily OHLCV contract."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from trading_agent.data.exceptions import DataOutsideRequestedRangeError, DataValidationError

REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Structured result of basic normalized-data quality checks."""

    valid: bool
    issues: tuple[str, ...]


def check_data_quality(data: pd.DataFrame) -> DataQualityReport:
    """Check the normalized OHLCV contract without mutating ``data``."""
    issues: list[str] = []
    if data.empty:
        issues.append("DataFrame is empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}.")
        return DataQualityReport(False, tuple(issues))

    if data.loc[:, REQUIRED_COLUMNS].isna().any().any():
        issues.append("Required OHLCV values must not be missing.")

    if not pd.api.types.is_datetime64_any_dtype(data["date"]):
        issues.append("date must have a datetime dtype.")
    if not pd.api.types.is_string_dtype(data["symbol"]):
        issues.append("symbol must have a string dtype.")

    for column in (*PRICE_COLUMNS, "volume"):
        if not pd.api.types.is_numeric_dtype(data[column]):
            issues.append(f"{column} must have a numeric dtype.")

    numeric_columns = (*PRICE_COLUMNS, "volume")
    if all(pd.api.types.is_numeric_dtype(data[column]) for column in numeric_columns):
        if not np.isfinite(data.loc[:, numeric_columns].to_numpy(dtype=float)).all():
            issues.append("OHLCV numeric values must be finite; +inf and -inf are not allowed.")

    if data.duplicated(["symbol", "date"]).any():
        issues.append("Duplicate symbol + date observations are not allowed.")

    if not data.empty and all(pd.api.types.is_numeric_dtype(data[column]) for column in numeric_columns):
        if (data.loc[:, PRICE_COLUMNS] < 0).any().any():
            issues.append("OHLC prices must be non-negative.")
        if (data["volume"] < 0).any():
            issues.append("volume must be non-negative.")
        if (data["high"] < data["low"]).any():
            issues.append("high must be greater than or equal to low.")
        if ((data["open"] < data["low"]) | (data["open"] > data["high"])).any():
            issues.append("open must fall between low and high.")
        if ((data["close"] < data["low"]) | (data["close"] > data["high"])).any():
            issues.append("close must fall between low and high.")

    return DataQualityReport(not issues, tuple(issues))


def require_valid_data(data: pd.DataFrame) -> None:
    """Raise a diagnostic error if the normalized OHLCV contract is invalid."""
    report = check_data_quality(data)
    if not report.valid:
        raise DataValidationError(" ".join(report.issues))


def require_within_date_range(data: pd.DataFrame, *, start: date, end: date) -> None:
    """Reject observations outside the request's half-open ``[start, end)`` range."""
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    out_of_range = (data["date"] < start_timestamp) | (data["date"] >= end_timestamp)
    if out_of_range.any():
        offending_dates = ", ".join(
            data.loc[out_of_range, "date"].dt.strftime("%Y-%m-%d").unique().tolist()
        )
        raise DataOutsideRequestedRangeError(
            "Provider returned observations outside the requested "
            f"[{start.isoformat()}, {end.isoformat()}) range: {offending_dates}."
        )
