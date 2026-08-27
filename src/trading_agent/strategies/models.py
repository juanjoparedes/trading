"""Structured, explainable results produced by the Strategy layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from trading_agent.scorer.models import ScoredCandidate

StrategyAction = Literal["enter", "no_action"]
StrategyReasonCode = Literal["selected_top_ranked", "rank_exceeds_max_candidates"]


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """One deterministic decision for one scored candidate.

    This is deliberately not a trading signal or an order: it records only
    whether the candidate falls within the configured limit, never a
    direction, size, stop, or target. Those remain later milestones.
    """

    symbol: str
    action: StrategyAction
    reason_code: StrategyReasonCode
    source_candidate: ScoredCandidate


@dataclass(frozen=True, slots=True)
class StrategyReport:
    """The full, deterministic output of one :class:`StrategyEngine` run."""

    source_scanner_config_id: str
    source_scorer_config_id: str
    strategy_config_id: str
    evaluation_date: date
    decisions: tuple[StrategyDecision, ...]
