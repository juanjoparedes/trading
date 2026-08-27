"""Provider-agnostic market-data acquisition and normalization."""

from trading_agent.data.engine import DataEngine
from trading_agent.data.models import DailyDataRequest
from trading_agent.data.providers import MarketDataProvider, YFinanceProvider

__all__ = ["DataEngine", "DailyDataRequest", "MarketDataProvider", "YFinanceProvider"]
