"""Deterministic, explainable Signal layer.

The signal engine answers one question given an already-computed
``RiskReport``: among the ``"approved"`` risk decisions, what is the trade
direction of each one? It never fetches or computes market data, never
re-evaluates risk, never sizes a position, and never places an order —
anti-look-ahead is structural here too, since this component has no access
path to any market data at all.
"""

from __future__ import annotations

from trading_agent.risk.models import RiskReport
from trading_agent.signals.config import SignalConfig
from trading_agent.signals.models import SignalDecision, SignalReport


class SignalEngine:
    """Turn ``"approved"`` risk decisions from a :class:`RiskReport` into signals."""

    def generate(self, report: RiskReport, *, config: SignalConfig) -> SignalReport:
        """Generate one :class:`SignalDecision` for every ``"approved"`` risk decision.

        Only decisions with ``status == "approved"`` are considered;
        ``"rejected_by_risk"`` decisions never produce a ``SignalDecision``
        and never appear in the returned ``SignalReport`` — they simply were
        never a candidate for a signal. Considered decisions are ordered by
        ``(source_decision.source_candidate.rank, symbol)`` before signals are
        generated, so the result never depends on the order of
        ``report.decisions`` and stays deterministic even if two decisions
        shared the same rank. Every generated ``SignalDecision`` carries
        ``config.trade_direction`` — the only trade direction this version
        supports is ``"long"``.
        """
        approved_decisions = [decision for decision in report.decisions if decision.status == "approved"]
        approved_decisions.sort(
            key=lambda decision: (decision.source_decision.source_candidate.rank, decision.symbol)
        )

        signal_decisions = tuple(
            SignalDecision(
                symbol=decision.symbol,
                trade_direction=config.trade_direction,
                reason_code="approved_by_risk",
                source_decision=decision,
            )
            for decision in approved_decisions
        )

        return SignalReport(
            source_scanner_config_id=report.source_scanner_config_id,
            source_scorer_config_id=report.source_scorer_config_id,
            source_strategy_config_id=report.source_strategy_config_id,
            source_risk_config_id=report.risk_config_id,
            signal_config_id=config.config_id,
            evaluation_date=report.evaluation_date,
            decisions=signal_decisions,
        )
