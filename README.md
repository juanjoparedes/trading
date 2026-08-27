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

## Indicators Engine

The Indicators Engine accepts only normalized daily OHLCV data and returns
mathematical indicator values. It neither fetches data nor produces signals,
orders, or risk decisions. Calculations are independently grouped by `symbol`,
sorted by `date`, and only use current or prior observations.

Available indicators are `sma`, `ema`, `rsi`, `momentum`, `atr`, and `returns`.
Their defaults are SMA/EMA 20, RSI/ATR 14, and momentum 10 periods. Windowed
SMA and EMA use `min_periods=window`; momentum uses an N-period lag; and the
first return is `NaN`. EMA uses pandas' recursive `adjust=False` convention,
with its first visible value withheld until a full window. RSI and ATR use
Wilder smoothing: an initial arithmetic average over the first full period,
then `(previous × (period - 1) + current) / period`. RSI is 100 for a positive
average gain with zero average loss, and 50 when both are zero.

Warm-up `NaN` values are expected and are not filled. Input-data `NaN` values
remain invalid. Use configurable `IndicatorConfig` values for non-default
windows:

```python
from trading_agent.indicators import IndicatorConfig, IndicatorsEngine

result = IndicatorsEngine().calculate(
    ohlcv_data,
    indicators=["returns", IndicatorConfig("sma", window=20), IndicatorConfig("rsi", window=14)],
)
```

## Market Scanner

The Market Scanner identifies, for an explicit universe of symbols and a
single `evaluation_date`, which symbols pass a configured set of objective
filters. It does not fetch data, does not import a provider, does not rank
or score candidates, and does not generate buy/sell signals — those remain
later milestones.

```text
Normalized OHLCV
       ↓
filter: date <= evaluation_date
       ↓
IndicatorsEngine (only the declared indicators)
       ↓
MarketScanner (hard filters, soft conditions)
       ↓
ScanReport
```

### Symbol normalization

`ScannerConfig.universe` is always normalized to uppercase, matching what
`normalize_ohlcv` already guarantees for `data["symbol"]`. `scan()` relies on
both sides using that same convention to match a symbol against the
universe, so it validates it explicitly: a non-uppercase `symbol` in `data`
is rejected with `DataValidationError` naming the offending value(s), rather
than silently reported as `missing_symbol_data`. Data produced by
`DataEngine` already satisfies this; the check only matters for a
hand-built or future data source that skips normalization.

### Anti-look-ahead

The scanner trims the input to `date <= evaluation_date` *before* handing it
to the `IndicatorsEngine`, for every symbol. No indicator value used in a
decision can ever be influenced by a row dated after `evaluation_date`. This
holds for every supported indicator, not only moving averages.

### `evaluation_date` and `as_of_date`

For each symbol, `as_of_date` is the latest available session with
`date <= evaluation_date`. If `evaluation_date` itself is not a session in
the data (a weekend, a holiday, or simply absent for that symbol), the
previous available session is used. A later session is never used. Both
`requested_evaluation_date` and `as_of_date` are preserved on every result.

### Staleness

`ScannerConfig.max_staleness_days` bounds how old `as_of_date` may be
relative to `evaluation_date`. Exceeding it does not reject the symbol; it
marks the result `insufficient_data` with `reason_code="stale_data"`.

### Warm-up

If any declared indicator is still `NaN` at `as_of_date` (insufficient
history for its window), the result is `insufficient_data` with
`reason_code="insufficient_indicator_history"`. Warm-up values are never
filled, interpolated, or substituted.

### Filters

`HardFilter` supports `>=`, `<=`, `>`, `<` against either a numeric
`threshold` (e.g. `close >= 10`) or another field via `compare_field` (e.g.
`close > sma_20`, `ema_20 > sma_20`). A filter's `field` and `compare_field`
must be a base OHLCV column or an indicator explicitly declared in
`indicator_requirements` — this is validated when the `ScannerConfig` is
constructed, not at scan time. There is no general expression language.

`SoftCondition` records a metric's value for context (e.g. `rsi_14`). Soft
conditions never reject a symbol.

### Result states

Each `SymbolScanResult.status` is exactly one of `candidate`, `rejected`, or
`insufficient_data`. Each `FilterEvaluation.status` is `passed`, `failed`, or
`unavailable`. Reason codes are structured, not narrative text: at minimum
`filter_passed`, `filter_failed`, `missing_indicator`,
`insufficient_indicator_history`, `stale_data`, and `missing_symbol_data`.

### Determinism

The same OHLCV data, universe, `ScannerConfig`, and `evaluation_date` always
produce the same `ScanReport`; results are ordered by `symbol`.
`ScannerConfig.config_id` is a deterministic SHA-256 hash of the
configuration's semantic content: changing any material field (`version`,
`universe`, `indicator_requirements`, a filter's or soft condition's own
values, `max_staleness_days`, `warmup_policy`) changes it. `config_id` is
deterministic and invariant to the declaration order of `hard_filters` and
`soft_conditions`, as long as the semantic content is the same — each is
canonicalized into a stable order (by `filter_id`, and by `field`,
respectively) purely for identity purposes before hashing. This does not
affect filter evaluation order or the order presented in a `ScanReport`.

```python
from trading_agent.indicators import IndicatorConfig
from trading_agent.indicators.engine import IndicatorsEngine
from trading_agent.scanner import HardFilter, MarketScanner, ScannerConfig

config = ScannerConfig.create(
    universe=["AAPL", "SPY", "QQQ"],
    indicator_requirements=[IndicatorConfig("sma", 20)],
    hard_filters=[HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_20")],
    max_staleness_days=2,
)
report = MarketScanner(IndicatorsEngine()).scan(ohlcv_data, config=config, evaluation_date="2024-06-28")
```

## Opportunity Scorer

The Opportunity Scorer takes a `ScanReport` already produced by the
`MarketScanner` and orders its `candidate` symbols by a single declared
metric. It does not fetch or compute market data, does not import `pandas`,
`DataEngine`, or `IndicatorsEngine`, does not re-evaluate filters, and does
not depend on `ScannerConfig` in any way — its only input is the `ScanReport`
value itself.

```text
ScanReport (from MarketScanner)
       ↓
keep only status == "candidate"
       ↓
OpportunityScorer (ScorerConfig: metric, direction)
       ↓
ScoreReport (ranked results + excluded)
```

### Candidate selection

Only results with `status == "candidate"` are considered. `rejected` and
`insufficient_data` results are neither ranked nor reported as excluded —
they are simply not part of this stage's concern, since the Scanner already
recorded why they were set aside.

### Metric and exclusion

`ScorerConfig.metric` names a key expected in `SymbolScanResult.metrics`
(never `soft_conditions`). A candidate is excluded — never ranked, never
imputed a value such as zero — when the metric is absent from `metrics`
(`reason_code="metric_missing"`), present but `None`
(`reason_code="metric_none"`), or a float NaN (`reason_code="metric_nan"`).
Whether the metric will actually be present cannot be verified when
`ScorerConfig` is constructed, since it depends on a future `ScanReport`; it
is checked only when `score()` runs.

### Ranking and determinism

`ScorerConfig.direction` is `"asc"` or `"desc"`. Ranks are ordinal
(`1, 2, 3, ...`) with no gaps or repeats, ordered by the metric and broken by
`symbol` ascending on exact ties. The result never depends on the order of
`report.results` — the same `ScanReport` and `ScorerConfig` always produce
the same `ScoreReport`. `ScorerConfig.config_id` is a deterministic SHA-256
hash over `{version, metric, direction}`, independent of any
`ScannerConfig.config_id`.

### Traceability

Every `ScoredCandidate` and `ExcludedCandidate` keeps its original
`SymbolScanResult` as `source_result`, unmutated. `ScoreReport` also carries
`source_scanner_config_id` and `source_scanner_version` (copied from the
input `ScanReport`), so a `ScoreReport` is always traceable back to the exact
scan that produced it.

```python
from trading_agent.scorer import OpportunityScorer, ScorerConfig

scorer_config = ScorerConfig.create(metric="rsi_14", direction="desc")
score_report = OpportunityScorer().score(report, config=scorer_config)
```

## Strategy

The Strategy layer takes a `ScoreReport` already produced by the
`OpportunityScorer` and decides, for each ranked candidate, whether it falls
within a configured limit. It does not import `pandas`, `DataEngine`,
`IndicatorsEngine`, `MarketScanner`, or `ScannerConfig`, does not depend on
`ScorerConfig`, and does not compute direction, position size, stop-loss,
take-profit, or place any order — this is deliberately the smallest possible
first version, limited to turning scored opportunities into decisions.

```text
ScoreReport (from OpportunityScorer)
       ↓
consider only report.results (ranked candidates; report.excluded is ignored)
       ↓
StrategyEngine (StrategyConfig: max_candidates)
       ↓
StrategyReport (one StrategyDecision per ranked candidate)
```

### Decision rule

Only `report.results` is considered — `report.excluded` entries were never
ranked, so they never produce a decision. A candidate with
`rank <= StrategyConfig.max_candidates` is `"enter"`
(`reason_code="selected_top_ranked"`); otherwise it is `"no_action"`
(`reason_code="rank_exceeds_max_candidates"`). There is no third action, no
direction, and no sizing — those remain later milestones.

### Determinism and traceability

`StrategyConfig.config_id` is a deterministic SHA-256 hash over
`{version, max_candidates}`, independent of any `ScorerConfig.config_id` or
`ScannerConfig.config_id`. `StrategyEngine.decide()` never depends on the
order of `report.results`: decisions are always returned ordered by rank
ascending. Every `StrategyDecision` keeps its original `ScoredCandidate` as
`source_candidate`, unmutated, so a decision is always traceable back to the
scored candidate — and, through it, to the original scan result — that
produced it.

```python
from trading_agent.strategies import StrategyConfig, StrategyEngine

strategy_config = StrategyConfig.create(max_candidates=5)
strategy_report = StrategyEngine().decide(score_report, config=strategy_config)
```
