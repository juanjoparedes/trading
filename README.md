# Trading Agent

## Purpose

Trading Agent is being built progressively as a research and algorithmic-trading system for studying strategies, running backtests, managing risk, and later supporting paper trading.

The initial objective is **not** to guarantee profitability. The system is intended to evaluate, scientifically, whether evidence exists for a statistical edge in specific strategies.

## Current milestone

**Milestone 0 — Project Foundation**

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
