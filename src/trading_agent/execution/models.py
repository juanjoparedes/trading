"""Structured, explainable results produced by the Execution layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from trading_agent.portfolio.models import TargetPosition

OrderSide = Literal["buy"]
OrderStatus = Literal["simulated"]


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """One deterministic, simulated order intent for one target position.

    This is deliberately not a real order: ``status`` is always
    ``"simulated"`` — it never claims that anything was actually sent to a
    broker or filled. ``side`` is derived exclusively from
    ``source_position.source_signal.trade_direction`` (only ``"long"`` is
    supported today, so ``side`` is always ``"buy"``); no short side exists
    in this version. No order type, price, broker, execution timestamp, or
    fill detail is recorded here — those remain later milestones.
    """

    symbol: str
    side: OrderSide
    quantity: int
    status: OrderStatus
    source_position: TargetPosition


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """The full, deterministic output of one :class:`ExecutionEngine` run."""

    source_scanner_config_id: str
    source_scorer_config_id: str
    source_strategy_config_id: str
    source_risk_config_id: str
    source_signal_config_id: str
    source_portfolio_config_id: str
    execution_config_id: str
    evaluation_date: date
    orders: tuple[OrderIntent, ...]
