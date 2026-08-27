"""Deterministic, explainable Market Scanner.

The scanner answers one question per symbol, as of one evaluation date: does
this symbol pass the configured hard filters, given only data available up to
that date? It never fetches data, never ranks or scores candidates, and never
produces a trading signal — those are later milestones.

Anti-look-ahead is structural, not incidental: the input is always trimmed to
``date <= evaluation_date`` *before* the Indicators Engine ever sees it, so no
indicator value at ``as_of_date`` can depend on a row from the future.
"""

from __future__ import annotations

import operator as operator_module
from datetime import date
from typing import Callable

import pandas as pd

from trading_agent.data.exceptions import DataValidationError
from trading_agent.data.models import parse_iso_date
from trading_agent.data.quality import require_valid_data
from trading_agent.indicators.engine import IndicatorsEngine
from trading_agent.scanner.config import HardFilter, ScannerConfig, SoftCondition
from trading_agent.scanner.models import (
    DatasetMetadata,
    FilterEvaluation,
    ScanReport,
    SoftConditionValue,
    SymbolScanResult,
)

SCANNER_VERSION = "1"

_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator_module.ge,
    "<=": operator_module.le,
    ">": operator_module.gt,
    "<": operator_module.lt,
}


class MarketScanner:
    """Identify which symbols in a universe pass configured filters, as of a date."""

    def __init__(self, indicators_engine: IndicatorsEngine) -> None:
        self._indicators_engine = indicators_engine

    def scan(
        self,
        data: pd.DataFrame,
        *,
        config: ScannerConfig,
        evaluation_date: date | str,
    ) -> ScanReport:
        """Evaluate every symbol in ``config.universe`` as of ``evaluation_date``.

        ``data`` must already satisfy the normalized OHLCV contract, which
        includes uppercase ``symbol`` values (as produced by
        :func:`trading_agent.data.normalization.normalize_ohlcv`). A symbol
        matched against ``config.universe`` (itself always uppercase) relies
        on that guarantee, so a non-uppercase ``symbol`` is treated as a
        contract violation and rejected outright rather than silently
        surfacing as a missing-data result. Rows dated after
        ``evaluation_date`` are never used to compute anything, for any
        symbol.
        """
        require_valid_data(data)
        _require_normalized_symbols(data)
        requested_evaluation_date = parse_iso_date(evaluation_date, "evaluation_date")
        evaluation_timestamp = pd.Timestamp(requested_evaluation_date)

        history = data.loc[data["date"] <= evaluation_timestamp]

        results = [
            self._scan_symbol(
                symbol=symbol,
                history=history.loc[history["symbol"] == symbol],
                config=config,
                requested_evaluation_date=requested_evaluation_date,
                evaluation_timestamp=evaluation_timestamp,
            )
            for symbol in config.universe
        ]
        results.sort(key=lambda result: result.symbol)

        metadata = DatasetMetadata(
            symbols_requested=len(config.universe),
            symbols_evaluated=len(results),
            evaluation_date=requested_evaluation_date,
            scanner_version=SCANNER_VERSION,
            config_id=config.config_id,
        )
        return ScanReport(
            requested_evaluation_date=requested_evaluation_date,
            config_id=config.config_id,
            dataset_metadata=metadata,
            results=tuple(results),
        )

    def _scan_symbol(
        self,
        *,
        symbol: str,
        history: pd.DataFrame,
        config: ScannerConfig,
        requested_evaluation_date: date,
        evaluation_timestamp: pd.Timestamp,
    ) -> SymbolScanResult:
        required_indicators = config.required_indicator_columns()

        if history.empty:
            return _insufficient_result(
                symbol=symbol,
                requested_evaluation_date=requested_evaluation_date,
                as_of_date=None,
                reason_code="missing_symbol_data",
                note="No observations at or before the requested evaluation_date.",
                config=config,
                required_indicators=required_indicators,
            )

        as_of_timestamp = history["date"].max()
        as_of_date = as_of_timestamp.date()
        staleness_days = (evaluation_timestamp - as_of_timestamp).days

        if staleness_days > config.max_staleness_days:
            return _insufficient_result(
                symbol=symbol,
                requested_evaluation_date=requested_evaluation_date,
                as_of_date=as_of_date,
                reason_code="stale_data",
                note=(
                    f"Most recent observation is {staleness_days} day(s) before evaluation_date; "
                    f"max_staleness_days is {config.max_staleness_days}."
                ),
                config=config,
                required_indicators=required_indicators,
            )

        augmented = self._indicators_engine.calculate(history, indicators=config.indicator_requirements)
        as_of_row = augmented.loc[augmented["date"] == as_of_timestamp].iloc[-1]

        missing_indicators = [
            column
            for column in required_indicators
            if column not in as_of_row.index or pd.isna(as_of_row[column])
        ]
        if missing_indicators:
            return _insufficient_result(
                symbol=symbol,
                requested_evaluation_date=requested_evaluation_date,
                as_of_date=as_of_date,
                reason_code="insufficient_indicator_history",
                note=f"Indicator(s) not yet available at as_of_date: {', '.join(missing_indicators)}.",
                config=config,
                required_indicators=required_indicators,
            )

        passed_filters, failed_filters = _evaluate_hard_filters(config.hard_filters, as_of_row)
        soft_conditions = _evaluate_soft_conditions(config.soft_conditions, as_of_row)
        metrics = _collect_metrics(config, as_of_row)

        status = "rejected" if failed_filters else "candidate"
        return SymbolScanResult(
            symbol=symbol,
            requested_evaluation_date=requested_evaluation_date,
            as_of_date=as_of_date,
            status=status,
            passed_filters=passed_filters,
            failed_filters=failed_filters,
            soft_conditions=soft_conditions,
            metrics=metrics,
            required_indicators=required_indicators,
            data_quality_notes=_staleness_note(requested_evaluation_date, as_of_date),
        )


def _require_normalized_symbols(data: pd.DataFrame) -> None:
    """Reject ``data`` if any ``symbol`` value is not already uppercase.

    The normalized OHLCV contract (``normalize_ohlcv``) always produces
    uppercase symbols, and ``ScannerConfig.universe`` is normalized the same
    way. Matching a symbol against the universe depends on both sides using
    that same convention; if ``data`` were left un-normalized, a legitimate
    symbol would silently fail to match and be reported as
    ``missing_symbol_data`` instead of surfacing the real problem, which is
    that the input never went through the established normalization.
    """
    symbols = data["symbol"]
    offending = sorted(symbols[symbols != symbols.str.upper()].unique())
    if offending:
        raise DataValidationError(
            "symbol values must already be uppercase, per the normalized OHLCV contract "
            f"(see normalize_ohlcv); found non-uppercase symbol(s): {', '.join(offending)}."
        )


def _staleness_note(requested_evaluation_date: date, as_of_date: date) -> tuple[str, ...]:
    if as_of_date == requested_evaluation_date:
        return ()
    return (f"as_of_date {as_of_date.isoformat()} precedes requested_evaluation_date "
            f"{requested_evaluation_date.isoformat()}.",)


def _insufficient_result(
    *,
    symbol: str,
    requested_evaluation_date: date,
    as_of_date: date | None,
    reason_code: str,
    note: str,
    config: ScannerConfig,
    required_indicators: tuple[str, ...],
) -> SymbolScanResult:
    """Build an insufficient_data result, recording every declared filter as unavailable."""
    unavailable = tuple(
        FilterEvaluation(
            filter_id=hard_filter.filter_id,
            status="unavailable",
            observed_value=None,
            operator=hard_filter.operator,
            threshold=None,
            reason_code=reason_code,  # type: ignore[arg-type]
        )
        for hard_filter in config.hard_filters
    )
    soft_conditions = tuple(
        SoftConditionValue(field=condition.field, value=None) for condition in config.soft_conditions
    )
    return SymbolScanResult(
        symbol=symbol,
        requested_evaluation_date=requested_evaluation_date,
        as_of_date=as_of_date,
        status="insufficient_data",
        passed_filters=(),
        failed_filters=unavailable,
        soft_conditions=soft_conditions,
        metrics={},
        required_indicators=required_indicators,
        data_quality_notes=(note,),
    )


def _evaluate_hard_filters(
    hard_filters: tuple[HardFilter, ...],
    row: pd.Series,
) -> tuple[tuple[FilterEvaluation, ...], tuple[FilterEvaluation, ...]]:
    passed: list[FilterEvaluation] = []
    failed: list[FilterEvaluation] = []
    for hard_filter in hard_filters:
        evaluation = _evaluate_hard_filter(hard_filter, row)
        (passed if evaluation.status == "passed" else failed).append(evaluation)
    return tuple(passed), tuple(failed)


def _evaluate_hard_filter(hard_filter: HardFilter, row: pd.Series) -> FilterEvaluation:
    observed = row.get(hard_filter.field)
    if observed is None or pd.isna(observed):
        return FilterEvaluation(
            filter_id=hard_filter.filter_id,
            status="unavailable",
            observed_value=None,
            operator=hard_filter.operator,
            threshold=None,
            reason_code="missing_indicator",
        )

    if hard_filter.compare_field is not None:
        threshold = row.get(hard_filter.compare_field)
        if threshold is None or pd.isna(threshold):
            return FilterEvaluation(
                filter_id=hard_filter.filter_id,
                status="unavailable",
                observed_value=float(observed),
                operator=hard_filter.operator,
                threshold=None,
                reason_code="missing_indicator",
            )
    else:
        threshold = hard_filter.threshold

    passed = _OPERATORS[hard_filter.operator](float(observed), float(threshold))
    return FilterEvaluation(
        filter_id=hard_filter.filter_id,
        status="passed" if passed else "failed",
        observed_value=float(observed),
        operator=hard_filter.operator,
        threshold=float(threshold),
        reason_code="filter_passed" if passed else "filter_failed",
    )


def _evaluate_soft_conditions(
    soft_conditions: tuple[SoftCondition, ...],
    row: pd.Series,
) -> tuple[SoftConditionValue, ...]:
    values: list[SoftConditionValue] = []
    for condition in soft_conditions:
        observed = row.get(condition.field)
        value = None if observed is None or pd.isna(observed) else float(observed)
        values.append(SoftConditionValue(field=condition.field, value=value))
    return tuple(values)


def _collect_metrics(config: ScannerConfig, row: pd.Series) -> dict[str, float | None]:
    """Collect declared metrics plus ``close``, which is always available.

    ``close`` is the as-of-date reference price for the symbol. It is
    included unconditionally — independent of whether it was declared in
    ``hard_filters``, ``soft_conditions``, or ``indicator_requirements`` —
    so a downstream consumer (e.g. a future Strategy) always has a reference
    price without needing direct OHLCV access. It comes from the same
    already-trimmed, already-validated ``row`` every other metric comes
    from, so it carries the same anti-look-ahead and per-symbol isolation
    guarantees as any other field collected here.
    """
    fields = set(config.required_indicator_columns())
    fields.update(hard_filter.field for hard_filter in config.hard_filters)
    fields.update(
        hard_filter.compare_field for hard_filter in config.hard_filters if hard_filter.compare_field is not None
    )
    fields.update(condition.field for condition in config.soft_conditions)
    fields.add("close")
    metrics: dict[str, float | None] = {}
    for field_name in sorted(fields):
        value = row.get(field_name)
        metrics[field_name] = None if value is None or pd.isna(value) else float(value)
    return metrics
