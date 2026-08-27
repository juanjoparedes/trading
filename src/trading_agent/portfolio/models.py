"""Structured, explainable results produced by the Portfolio layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from trading_agent.signals.models import SignalDecision

PortfolioExclusionReasonCode = Literal["price_unavailable", "allocation_below_one_share"]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """A caller-supplied, dated set of reference prices.

    Portfolio never reaches into ``source_*`` references to read a price —
    those exist for traceability, not as a market-data API. A price used by
    :class:`~trading_agent.portfolio.engine.PortfolioEngine` always comes
    from here. ``as_of_date`` must equal the ``evaluation_date`` of the
    ``SignalReport`` being processed; this is what keeps anti-look-ahead
    intact at the one point where Portfolio needs a real market price.
    """

    as_of_date: date
    prices: dict[str, float]


@dataclass(frozen=True, slots=True)
class PortfolioExclusion:
    """A signal that did not produce a target position, and why.

    ``reason_code`` is exactly one of: ``price_unavailable`` (the symbol is
    not a key of ``market_snapshot.prices``) or ``allocation_below_one_share``
    (the symbol has a price, but ``floor(ideal_allocation / price) == 0``).
    An excluded signal never recovers or redistributes any capital.
    """

    symbol: str
    reason_code: PortfolioExclusionReasonCode
    source_signal: SignalDecision


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """One deterministic target position for one signal.

    This is deliberately not a filled position or an order: ``quantity`` is
    always a positive integer (never fractional, never zero — a zero
    quantity is a :class:`PortfolioExclusion` instead), and ``target_value``
    is the value actually represented by that whole-share quantity
    (``quantity * price_used``), never the intermediate ideal allocation.
    No stop, target, capital commitment beyond this, or execution detail is
    recorded here — those remain later milestones.
    """

    symbol: str
    quantity: int
    target_value: float
    source_signal: SignalDecision


@dataclass(frozen=True, slots=True)
class PortfolioReport:
    """The full, deterministic output of one :class:`PortfolioEngine` run."""

    source_scanner_config_id: str
    source_scorer_config_id: str
    source_strategy_config_id: str
    source_risk_config_id: str
    source_signal_config_id: str
    portfolio_config_id: str
    evaluation_date: date
    target_positions: tuple[TargetPosition, ...]
    excluded: tuple[PortfolioExclusion, ...]
    unallocated_cash: float
