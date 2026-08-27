"""Structured, explainable results produced by the Risk layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from trading_agent.strategies.models import StrategyDecision

RiskStatus = Literal["approved", "rejected_by_risk"]
RiskReasonCode = Literal["within_risk_limit", "risk_limit_exceeded"]


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """One deterministic risk verdict for one ``"enter"`` strategy decision.

    This is deliberately not a sized position or an order: it records only
    whether the decision falls within the configured capacity limit, never a
    direction, size, stop, target, or capital amount. Those remain later
    milestones.
    """

    symbol: str
    status: RiskStatus
    reason_code: RiskReasonCode
    source_decision: StrategyDecision


@dataclass(frozen=True, slots=True)
class RiskReport:
    """The full, deterministic output of one :class:`RiskEngine` run."""

    source_scanner_config_id: str
    source_scorer_config_id: str
    source_strategy_config_id: str
    risk_config_id: str
    evaluation_date: date
    decisions: tuple[RiskDecision, ...]
