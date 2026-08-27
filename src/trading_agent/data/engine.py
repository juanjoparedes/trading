"""Provider-independent data engine."""

import pandas as pd

from trading_agent.data.cache import FileSystemCache
from trading_agent.data.models import DailyDataRequest
from trading_agent.data.providers import MarketDataProvider
from trading_agent.data.quality import require_valid_data, require_within_date_range


class DataEngine:
    """Fetch normalized daily bars through an injected provider, with optional cache."""

    def __init__(self, provider: MarketDataProvider, cache: FileSystemCache | None = None) -> None:
        self.provider = provider
        self.cache = cache

    def get_daily_data(
        self,
        *,
        symbols: list[str] | tuple[str, ...],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch an exact daily request, returning the normalized OHLCV contract."""
        request = DailyDataRequest.create(symbols, start, end)
        if self.cache is not None:
            cached = self.cache.get(self.provider.cache_key, request)
            if cached is not None:
                require_valid_data(cached)
                require_within_date_range(cached, start=request.start, end=request.end)
                return cached

        data = self.provider.get_daily_data(request)
        require_valid_data(data)
        require_within_date_range(data, start=request.start, end=request.end)
        if self.cache is not None:
            self.cache.set(self.provider.cache_key, request, data)
        return data
