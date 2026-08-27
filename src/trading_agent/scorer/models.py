"""Structured, explainable results produced by the Opportunity Scorer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from trading_agent.scanner.models import SymbolScanResult

ExclusionReasonCode = Literal["metric_missing", "metric_none", "metric_nan"]


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One ranked candidate, traceable back to its source scan result."""

    symbol: str
    rank: int
    metric_used: str
    observed_value: float
    source_result: SymbolScanResult


@dataclass(frozen=True, slots=True)
class ExcludedCandidate:
    """A ``candidate``-status symbol that could not be ranked, and why.

    ``reason_code`` is exactly one of: ``metric_missing`` (the declared
    metric is not a key of ``source_result.metrics`` at all), ``metric_none``
    (the key exists but its value is ``None``), or ``metric_nan`` (the value
    is a float NaN). A missing metric is never treated as zero or otherwise
    imputed.
    """

    symbol: str
    reason_code: ExclusionReasonCode
    source_result: SymbolScanResult


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """The full, deterministic output of one :class:`OpportunityScorer` run."""

    source_scanner_config_id: str
    source_scanner_version: str
    scorer_config_id: str
    evaluation_date: date
    results: tuple[ScoredCandidate, ...]
    excluded: tuple[ExcludedCandidate, ...]
