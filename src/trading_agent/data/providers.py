"""Provider interface and the initial yfinance adapter."""

from abc import ABC, abstractmethod

import pandas as pd

from trading_agent.data.exceptions import EmptyDataError, ProviderUnavailableError, SymbolNotFoundError
from trading_agent.data.models import DailyDataRequest
from trading_agent.data.normalization import normalize_ohlcv


class MarketDataProvider(ABC):
    """An adapter that retrieves historical market data for a data request."""

    cache_key: str

    @abstractmethod
    def get_daily_data(self, request: DailyDataRequest) -> pd.DataFrame:
        """Return normalized daily OHLCV data for every requested symbol."""


class YFinanceProvider(MarketDataProvider):
    """Historical-data adapter that encapsulates the yfinance dependency."""

    cache_key = "yfinance-v1"

    def get_daily_data(self, request: DailyDataRequest) -> pd.DataFrame:
        try:
            import yfinance as yf

            response = yf.download(
                tickers=list(request.symbols),
                start=request.start.isoformat(),
                end=request.end.isoformat(),
                interval=request.timeframe,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as error:
            raise ProviderUnavailableError("yfinance could not retrieve historical market data.") from error

        if response.empty:
            raise EmptyDataError("yfinance returned no data for the requested symbols and date range.")

        frames: list[pd.DataFrame] = []
        missing_symbols: list[str] = []
        for symbol in request.symbols:
            if isinstance(response.columns, pd.MultiIndex):
                if symbol not in response.columns.get_level_values(0):
                    missing_symbols.append(symbol)
                    continue
                symbol_data = response[symbol].dropna(how="all")
            elif len(request.symbols) == 1:
                symbol_data = response.dropna(how="all")
            else:
                missing_symbols.append(symbol)
                continue

            if symbol_data.empty:
                missing_symbols.append(symbol)
            else:
                frames.append(normalize_ohlcv(symbol_data, symbol=symbol))

        if missing_symbols:
            raise SymbolNotFoundError(f"No data returned for symbol(s): {', '.join(missing_symbols)}.")
        if not frames:
            raise EmptyDataError("yfinance returned no usable daily bars.")
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
