"""Deterministic, explainable Execution layer.

The execution engine answers one question given an already-computed
``PortfolioReport``: what would a simulated order for each target position
look like? It never fetches or computes market data, never navigates
``source_*`` references to find a price, never recomputes ``quantity``, and
never places a real order — anti-look-ahead is structural here too, since
this component has no access path to any market data at all.
"""

from __future__ import annotations

from trading_agent.execution.config import ExecutionConfig
from trading_agent.execution.exceptions import ExecutionError
from trading_agent.execution.models import ExecutionReport, OrderIntent, OrderSide
from trading_agent.portfolio.models import PortfolioReport, TargetPosition


class ExecutionEngine:
    """Turn target positions from a :class:`PortfolioReport` into simulated order intents."""

    def prepare(self, portfolio_report: PortfolioReport, *, config: ExecutionConfig) -> ExecutionReport:
        """Prepare one :class:`OrderIntent` for every target position in ``portfolio_report``.

        Every ``TargetPosition`` in ``portfolio_report.target_positions``
        produces exactly one ``OrderIntent`` — the relationship is 1:1, and
        no new exclusion state is introduced, since a ``TargetPosition`` that
        already reached this layer is, by construction of
        :class:`~trading_agent.portfolio.engine.PortfolioEngine`, always a
        valid position with ``quantity >= 1``. ``side`` is derived
        exclusively from ``source_signal.trade_direction`` (only ``"long"``
        is supported today, so ``side`` is always ``"buy"``); ``quantity``
        is copied unchanged from ``TargetPosition.quantity``, never
        recomputed. Considered positions are ordered by ``(rank, symbol)``
        before processing, so the result never depends on the order of
        ``portfolio_report.target_positions``.
        """
        positions = sorted(portfolio_report.target_positions, key=_position_sort_key)

        orders = tuple(
            OrderIntent(
                symbol=position.symbol,
                side=_derive_side(position.source_signal.trade_direction),
                quantity=position.quantity,
                status="simulated",
                source_position=position,
            )
            for position in positions
        )

        return ExecutionReport(
            source_scanner_config_id=portfolio_report.source_scanner_config_id,
            source_scorer_config_id=portfolio_report.source_scorer_config_id,
            source_strategy_config_id=portfolio_report.source_strategy_config_id,
            source_risk_config_id=portfolio_report.source_risk_config_id,
            source_signal_config_id=portfolio_report.source_signal_config_id,
            source_portfolio_config_id=portfolio_report.portfolio_config_id,
            execution_config_id=config.config_id,
            evaluation_date=portfolio_report.evaluation_date,
            orders=orders,
        )


def _derive_side(trade_direction: str) -> OrderSide:
    """Translate a signal's trade direction into an order side.

    Only ``"long"`` is supported in this version, and it always maps to
    ``"buy"``. No short side exists — a ``trade_direction`` other than
    ``"long"`` cannot be produced by any real ``SignalEngine`` today, but
    this still fails loudly rather than silently defaulting, consistent with
    how every other layer treats an unsupported value it is not built to
    handle.
    """
    if trade_direction == "long":
        return "buy"
    raise ExecutionError(
        f"Unsupported trade_direction {trade_direction!r}; only 'long' is supported in this version."
    )


def _position_sort_key(position: TargetPosition) -> tuple[int, str]:
    signal_decision = position.source_signal
    risk_decision = signal_decision.source_decision
    strategy_decision = risk_decision.source_decision
    rank = strategy_decision.source_candidate.rank
    return (rank, position.symbol)
