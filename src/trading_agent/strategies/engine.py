"""Deterministic, explainable Strategy layer.

The strategy answers one question given an already-computed ``ScoreReport``:
among the ranked candidates, which ones fall within the configured limit?
It never fetches or computes market data, never re-ranks candidates, never
sizes a position, and never places an order — anti-look-ahead is structural
here too, since this component has no access path to any market data at all.
"""

from __future__ import annotations

from trading_agent.scorer.models import ScoreReport
from trading_agent.strategies.config import StrategyConfig
from trading_agent.strategies.models import StrategyDecision, StrategyReport


class StrategyEngine:
    """Turn ranked candidates from a :class:`ScoreReport` into strategy decisions."""

    def decide(self, report: ScoreReport, *, config: StrategyConfig) -> StrategyReport:
        """Decide ``"enter"`` vs ``"no_action"`` for every ranked candidate in ``report``.

        Only ``report.results`` (already-ranked candidates) are considered;
        ``report.excluded`` entries never produce a decision, since they were
        never ranked in the first place. The result never depends on the
        order of ``report.results`` — decisions are always returned ordered
        by rank ascending, with ``symbol`` ascending as a secondary key so
        the order stays deterministic even if a malformed ``ScoreReport``
        contained duplicate ranks (``OpportunityScorer`` never produces
        those, but this method does not rely on that guarantee).
        """
        decisions = [
            StrategyDecision(
                symbol=candidate.symbol,
                action="enter" if candidate.rank <= config.max_candidates else "no_action",
                reason_code=(
                    "selected_top_ranked"
                    if candidate.rank <= config.max_candidates
                    else "rank_exceeds_max_candidates"
                ),
                source_candidate=candidate,
            )
            for candidate in report.results
        ]
        decisions.sort(key=lambda decision: (decision.source_candidate.rank, decision.source_candidate.symbol))

        return StrategyReport(
            source_scanner_config_id=report.source_scanner_config_id,
            source_scorer_config_id=report.scorer_config_id,
            strategy_config_id=config.config_id,
            evaluation_date=report.evaluation_date,
            decisions=tuple(decisions),
        )
