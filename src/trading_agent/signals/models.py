"""Structured, explainable results produced by the Signal layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from trading_agent.risk.models import RiskDecision

TradeDirection = Literal["long"]
SignalReasonCode = Literal["approved_by_risk"]


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """One deterministic signal for one ``"approved"`` risk decision.

    This is deliberately not a sized position or an order: it records only
    that the decision was approved by Risk and assigns it the configured
    trade direction, never a size, stop, target, or capital amount. Those
    remain later milestones. Only ``"approved"`` risk decisions ever produce
    a ``SignalDecision`` — a ``"rejected_by_risk"`` decision never does.
    """

    symbol: str
    trade_direction: TradeDirection
    reason_code: SignalReasonCode
    source_decision: RiskDecision


@dataclass(frozen=True, slots=True)
class SignalReport:
    """The full, deterministic output of one :class:`SignalEngine` run."""

    source_scanner_config_id: str
    source_scorer_config_id: str
    source_strategy_config_id: str
    source_risk_config_id: str
    signal_config_id: str
    evaluation_date: date
    decisions: tuple[SignalDecision, ...]
