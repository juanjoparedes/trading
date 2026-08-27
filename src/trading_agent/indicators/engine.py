"""Reproducible, provider-independent daily technical indicators."""

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from trading_agent.data.quality import require_valid_data

IndicatorName = Literal["sma", "ema", "rsi", "momentum", "atr", "returns"]
_DEFAULT_WINDOWS: dict[IndicatorName, int | None] = {
    "sma": 20,
    "ema": 20,
    "rsi": 14,
    "momentum": 10,
    "atr": 14,
    "returns": None,
}


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    """Configuration for one indicator calculation."""

    name: IndicatorName
    window: int | None = None

    def resolved_window(self) -> int | None:
        """Return the configured window or the documented indicator default."""
        window = self.window if self.window is not None else _DEFAULT_WINDOWS[self.name]
        if self.name == "returns":
            if self.window is not None:
                raise ValueError("returns does not accept a window.")
            return None
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError(f"{self.name} window must be a positive integer.")
        return window

    def column_name(self) -> str:
        """Return the stable output column name for this configuration."""
        window = self.resolved_window()
        return self.name if window is None else f"{self.name}_{window}"


class IndicatorsEngine:
    """Calculate daily indicators independently for every symbol.

    The engine is deterministic and only uses each symbol's rows through the
    current timestamp. It performs no acquisition, signaling, or trading.
    """

    def calculate(
        self,
        data: pd.DataFrame,
        *,
        indicators: Sequence[str | IndicatorConfig],
    ) -> pd.DataFrame:
        """Return sorted normalized data augmented with requested indicator columns."""
        result = self._prepare_input(data)
        configs = self._resolve_configs(indicators)
        for config in configs:
            self._add_indicator(result, config)
        return result

    @staticmethod
    def _prepare_input(data: pd.DataFrame) -> pd.DataFrame:
        require_valid_data(data)
        dates = data["date"]
        if dates.dt.tz is not None:
            raise ValueError("Indicator input date must be timezone-naive for the daily contract.")
        if not dates.dt.normalize().eq(dates).all():
            raise ValueError("Indicator input date must not contain an intraday time component.")
        return data.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True).copy()

    @staticmethod
    def _resolve_configs(indicators: Sequence[str | IndicatorConfig]) -> list[IndicatorConfig]:
        configs: list[IndicatorConfig] = []
        for indicator in indicators:
            config = IndicatorConfig(indicator) if isinstance(indicator, str) else indicator
            if not isinstance(config, IndicatorConfig):
                raise TypeError("Indicators must be names or IndicatorConfig instances.")
            try:
                config.resolved_window()
            except KeyError as error:
                raise ValueError(f"Unsupported indicator {config.name!r}.") from error
            configs.append(config)
        if len({config.column_name() for config in configs}) != len(configs):
            raise ValueError("Each requested indicator output column must be unique.")
        return configs

    @staticmethod
    def _add_indicator(data: pd.DataFrame, config: IndicatorConfig) -> None:
        column = config.column_name()
        window = config.resolved_window()
        values = pd.Series(np.nan, index=data.index, dtype=float)
        for _, group in data.groupby("symbol", sort=False):
            if config.name == "sma":
                calculated = group["close"].rolling(window, min_periods=window).mean()
            elif config.name == "ema":
                calculated = group["close"].ewm(span=window, adjust=False, min_periods=window).mean()
            elif config.name == "rsi":
                calculated = _rsi(group["close"], window)
            elif config.name == "momentum":
                calculated = group["close"] / group["close"].shift(window) - 1.0
            elif config.name == "atr":
                calculated = _atr(group, window)
            elif config.name == "returns":
                calculated = group["close"].pct_change(fill_method=None)
            else:  # pragma: no cover - configuration is checked before calculation.
                raise ValueError(f"Unsupported indicator {config.name!r}.")
            values.loc[group.index] = calculated.astype(float)
        data[column] = values


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing seeded by the first complete arithmetic average."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    first_valid = values.rolling(period, min_periods=period).mean()
    first_index = first_valid.first_valid_index()
    if first_index is None:
        return result
    result.loc[first_index] = first_valid.loc[first_index]
    positions = values.index.get_loc(first_index)
    for position in range(positions + 1, len(values)):
        current_index = values.index[position]
        previous_index = values.index[position - 1]
        result.loc[current_index] = (result.loc[previous_index] * (period - 1) + values.loc[current_index]) / period
    return result


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI using close-to-close gains and losses."""
    delta = close.diff()
    average_gain = _wilder_average(delta.clip(lower=0), period)
    average_loss = _wilder_average((-delta).clip(lower=0), period)
    relative_strength = average_gain / average_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)


def _atr(data: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR from current range and the previous close."""
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return _wilder_average(true_range, period)
