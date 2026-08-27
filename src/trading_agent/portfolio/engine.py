"""Deterministic, explainable Portfolio layer.

The portfolio engine answers one question given an already-computed
``SignalReport`` and a dated ``MarketSnapshot``: how many whole shares of
each signaled symbol would an equal-weight allocation of a fixed capital
buy, as of the same evaluation date? It never fetches or computes market
data itself, never re-evaluates risk or signal generation, never sizes by
risk-per-trade or volatility, and never places an order — anti-look-ahead
at this layer rests entirely on ``market_snapshot.as_of_date`` matching
``signal_report.evaluation_date``, which this engine enforces before doing
anything else.
"""

from __future__ import annotations

import math

from trading_agent.portfolio.config import PortfolioConfig
from trading_agent.portfolio.exceptions import PortfolioSnapshotDateMismatchError
from trading_agent.portfolio.models import MarketSnapshot, PortfolioExclusion, PortfolioReport, TargetPosition
from trading_agent.signals.models import SignalDecision, SignalReport


class PortfolioEngine:
    """Turn signals from a :class:`SignalReport` into an equal-weight target portfolio."""

    def build(
        self,
        signal_report: SignalReport,
        *,
        config: PortfolioConfig,
        market_snapshot: MarketSnapshot,
    ) -> PortfolioReport:
        """Build a :class:`PortfolioReport` for every signal in ``signal_report``.

        ``market_snapshot.as_of_date`` must equal ``signal_report.evaluation_date``
        exactly; any mismatch, earlier or later, raises
        :class:`PortfolioSnapshotDateMismatchError` before any allocation is
        computed. Every signal is either a :class:`TargetPosition` (a symbol
        with a price in ``market_snapshot`` that resolves to at least one
        whole share) or a :class:`PortfolioExclusion` (``price_unavailable``
        or ``allocation_below_one_share``) — a signal never simply
        disappears. ``ideal_allocation`` is ``config.initial_capital``
        divided by the number of signals, computed exactly once and never
        recomputed after an exclusion, so excluded signals never free up
        capital for the remaining ones. Considered decisions are ordered by
        ``(rank, symbol)`` before processing, so the result never depends on
        the order of ``signal_report.decisions``. ``unallocated_cash`` is
        ``config.initial_capital`` minus the sum of every ``target_value``
        actually allocated — it is a computed result of this run, never
        persisted or carried over from a prior one.
        """
        if market_snapshot.as_of_date != signal_report.evaluation_date:
            raise PortfolioSnapshotDateMismatchError(
                f"market_snapshot.as_of_date ({market_snapshot.as_of_date.isoformat()}) must equal "
                f"signal_report.evaluation_date ({signal_report.evaluation_date.isoformat()})."
            )

        decisions = sorted(signal_report.decisions, key=_decision_sort_key)
        number_of_signals = len(decisions)

        target_positions: list[TargetPosition] = []
        excluded: list[PortfolioExclusion] = []

        if number_of_signals > 0:
            ideal_allocation = config.initial_capital / number_of_signals

            for decision in decisions:
                price = market_snapshot.prices.get(decision.symbol)
                if price is None:
                    excluded.append(
                        PortfolioExclusion(
                            symbol=decision.symbol,
                            reason_code="price_unavailable",
                            source_signal=decision,
                        )
                    )
                    continue

                quantity = math.floor(ideal_allocation / price)
                if quantity == 0:
                    excluded.append(
                        PortfolioExclusion(
                            symbol=decision.symbol,
                            reason_code="allocation_below_one_share",
                            source_signal=decision,
                        )
                    )
                    continue

                target_positions.append(
                    TargetPosition(
                        symbol=decision.symbol,
                        quantity=quantity,
                        target_value=quantity * price,
                        source_signal=decision,
                    )
                )

        unallocated_cash = config.initial_capital - sum(
            position.target_value for position in target_positions
        )

        return PortfolioReport(
            source_scanner_config_id=signal_report.source_scanner_config_id,
            source_scorer_config_id=signal_report.source_scorer_config_id,
            source_strategy_config_id=signal_report.source_strategy_config_id,
            source_risk_config_id=signal_report.source_risk_config_id,
            source_signal_config_id=signal_report.signal_config_id,
            portfolio_config_id=config.config_id,
            evaluation_date=signal_report.evaluation_date,
            target_positions=tuple(target_positions),
            excluded=tuple(excluded),
            unallocated_cash=unallocated_cash,
        )


def _decision_sort_key(decision: SignalDecision) -> tuple[int, str]:
    risk_decision = decision.source_decision
    strategy_decision = risk_decision.source_decision
    rank = strategy_decision.source_candidate.rank
    return (rank, decision.symbol)
