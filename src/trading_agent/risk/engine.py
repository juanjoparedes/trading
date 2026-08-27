"""Deterministic, explainable Risk layer.

The risk gate answers one question given an already-computed
``StrategyReport``: among the ``"enter"`` decisions, how many can proceed
under a static capacity limit? It never fetches or computes market data,
never sizes a position, never computes a stop or target, and never places an
order — anti-look-ahead is structural here too, since this component has no
access path to any market data at all.
"""

from __future__ import annotations

from trading_agent.risk.config import RiskConfig
from trading_agent.risk.models import RiskDecision, RiskReport
from trading_agent.strategies.models import StrategyReport


class RiskEngine:
    """Gate ``"enter"`` decisions from a :class:`StrategyReport` by a static capacity limit."""

    def evaluate(self, report: StrategyReport, *, config: RiskConfig) -> RiskReport:
        """Approve or reject every ``"enter"`` decision in ``report``.

        Only decisions with ``action == "enter"`` are considered;
        ``"no_action"`` decisions never produce a ``RiskDecision`` and never
        appear in the returned ``RiskReport`` — they simply were never a
        candidate for risk approval. Considered decisions are ordered by
        ``(source_candidate.rank, symbol)`` before the limit is applied, so
        the result never depends on the order of ``report.decisions`` and
        stays deterministic even if two decisions shared the same rank. The
        first ``config.max_approved_decisions`` are ``"approved"``; the rest
        are ``"rejected_by_risk"``. No other selection criterion is used.
        """
        enter_decisions = [decision for decision in report.decisions if decision.action == "enter"]
        enter_decisions.sort(key=lambda decision: (decision.source_candidate.rank, decision.symbol))

        risk_decisions = tuple(
            RiskDecision(
                symbol=decision.symbol,
                status="approved" if position < config.max_approved_decisions else "rejected_by_risk",
                reason_code=(
                    "within_risk_limit" if position < config.max_approved_decisions else "risk_limit_exceeded"
                ),
                source_decision=decision,
            )
            for position, decision in enumerate(enter_decisions)
        )

        return RiskReport(
            source_scanner_config_id=report.source_scanner_config_id,
            source_scorer_config_id=report.source_scorer_config_id,
            source_strategy_config_id=report.strategy_config_id,
            risk_config_id=config.config_id,
            evaluation_date=report.evaluation_date,
            decisions=risk_decisions,
        )
