# Trading Agent

## Purpose

Trading Agent is being built progressively as a research and algorithmic-trading system for studying strategies, running backtests, managing risk, and later supporting paper trading.

The initial objective is **not** to guarantee profitability. The system is intended to evaluate, scientifically, whether evidence exists for a statistical edge in specific strategies.

## Current milestone

**Milestone 1 — Market Data Engine**

## Phases

1. Market Scanner
2. Strategy Engine
3. Backtesting
4. Risk Engine
5. Paper Trading
6. Dashboard
7. AI Agent Layer
8. Live Trading

## Planned initial configuration

- **Market:** United States equities and ETFs
- **Initial timeframe:** daily
- **Initial virtual capital:** USD 10,000
- **Initial target risk:** 0.5% per trade

These variables will be configurable in a later milestone.

## Architecture

```text
Market Data
     ↓
Data Engine
     ↓
Market Scanner
     ↓
Strategy Engine
     ↓
Signal Engine
     ↓
Risk Engine
     ↓
Execution
     ↓
Paper Trading / Live Trading
     ↓
Trade Journal
     ↓
Dashboard
     ↓
AI Agent Layer
```

Live Trading remains blocked until validation criteria have been satisfied and explicit authorization has been received.

## Market Data Engine

The market-data layer separates data acquisition from the rest of the system:

```text
Market Data Provider → Data Engine → Normalized Market Data
```

`MarketDataProvider` is the provider interface. The initial implementation,
`YFinanceProvider`, is an adapter around `yfinance` intended for historical
research only. `yfinance` is not imported outside that adapter, so it can be
replaced later (for example, by another provider) without coupling indicators,
strategies, or backtesting to it.

### Normalized format

Every returned row uses these fields: `date`, `symbol`, `open`, `high`, `low`,
`close`, and `volume`. `date` is a timezone-naive daily trading date at
midnight; timestamps with an intraday time are rejected. For a timezone-aware
midnight input, its provider-local calendar date is preserved without a UTC
conversion. Prices and volume are finite numeric values. Data is sorted by
symbol and date, and duplicate `symbol` + `date` rows are rejected. The engine
enforces the half-open request interval `[start, end)`: it includes `start`,
excludes `end`, and rejects provider rows outside that range. It also rejects
missing fields, missing values, invalid dates, non-numeric or infinite values,
negative volume, and inconsistent OHLC ranges rather than silently changing
source data.

### Installation and use

Install the project dependencies, then inject a provider into the engine. The
cache location is deliberately supplied by configuration rather than embedded
in the data logic.

```bash
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m pytest -q
```

```python
from pathlib import Path

from trading_agent.data import DataEngine, YFinanceProvider
from trading_agent.data.cache import FileSystemCache

engine = DataEngine(YFinanceProvider(), cache=FileSystemCache(Path("data/cache")))
bars = engine.get_daily_data(
    symbols=["SPY", "QQQ", "AAPL"],
    start="2024-01-01",
    end="2025-01-01",
)
```

### Cache

`FileSystemCache` stores normalized DataFrames locally as pickle files. A cache
entry is used only for an exact match of provider identity, symbols, timeframe,
start, and end. It is reproducible and disposable; deleting its directory only
causes a fresh provider request. The cache is ignored by Git.

### Known limitations

The initial provider depends on yfinance and its upstream data availability,
which may be delayed, unavailable, or subject to provider changes. It is an
implementation choice for research, not a trading or execution connection.
The default test suite uses synthetic data and does not require Internet access.
