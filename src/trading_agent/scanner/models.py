"""Structured, explainable results produced by the Market Scanner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

ScanStatus = Literal["candidate", "rejected", "insufficient_data"]
FilterStatus = Literal["passed", "failed", "unavailable"]
ReasonCode = Literal[
    "filter_passed",
    "filter_failed",
    "missing_indicator",
    "insufficient_indicator_history",
    "stale_data",
    "missing_symbol_data",
]


@dataclass(frozen=True, slots=True)
class FilterEvaluation:
    """The outcome of evaluating one hard filter for one symbol."""

    filter_id: str
    status: FilterStatus
    observed_value: float | None
    operator: str
    threshold: float | None
    reason_code: ReasonCode


@dataclass(frozen=True, slots=True)
class SoftConditionValue:
    """A recorded, non-eliminating metric for one symbol."""

    field: str
    value: float | None


@dataclass(frozen=True, slots=True)
class SymbolScanResult:
    """The complete, auditable outcome for one symbol in one scan."""

    symbol: str
    requested_evaluation_date: date
    as_of_date: date | None
    status: ScanStatus
    passed_filters: tuple[FilterEvaluation, ...]
    failed_filters: tuple[FilterEvaluation, ...]
    soft_conditions: tuple[SoftConditionValue, ...]
    metrics: dict[str, float | None]
    required_indicators: tuple[str, ...]
    data_quality_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Audit metadata describing the run, independent of any single symbol."""

    symbols_requested: int
    symbols_evaluated: int
    evaluation_date: date
    scanner_version: str
    config_id: str


@dataclass(frozen=True, slots=True)
class ScanReport:
    """The full, deterministic output of one :class:`MarketScanner` run."""

    requested_evaluation_date: date
    config_id: str
    dataset_metadata: DatasetMetadata
    results: tuple[SymbolScanResult, ...]
