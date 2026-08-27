"""Deterministic, explainable Opportunity Scorer.

The scorer answers one question given an already-computed ``ScanReport``:
among the symbols marked ``candidate``, in what order should they be
considered, according to a single declared metric? It never fetches or
computes market data, never re-evaluates filters, and never reaches beyond
the ``ScanReport`` it is given — anti-look-ahead is structural here too,
since this component has no access path to any market data at all.
"""

from __future__ import annotations

import math

from trading_agent.scanner.models import ScanReport
from trading_agent.scorer.config import ScorerConfig
from trading_agent.scorer.models import ExcludedCandidate, ScoredCandidate, ScoreReport


class OpportunityScorer:
    """Rank ``candidate`` symbols from a :class:`ScanReport` by one metric."""

    def score(self, report: ScanReport, *, config: ScorerConfig) -> ScoreReport:
        """Rank every ``status == "candidate"`` result in ``report`` by ``config.metric``.

        A candidate is excluded — never ranked, never imputed a value — when
        the metric is absent from ``source_result.metrics``, present but
        ``None``, or a float NaN. Ranking is ordinal (1, 2, 3, ...) with no
        gaps or repeats, ordered by the metric (descending or ascending per
        ``config.direction``) and broken by ``symbol`` ascending on exact
        ties. The result never depends on the order of ``report.results``.
        """
        rankable: list[tuple[str, float]] = []
        excluded: list[ExcludedCandidate] = []

        for result in report.results:
            if result.status != "candidate":
                continue

            if config.metric not in result.metrics:
                excluded.append(
                    ExcludedCandidate(
                        symbol=result.symbol,
                        reason_code="metric_missing",
                        source_result=result,
                    )
                )
                continue

            value = result.metrics[config.metric]
            if value is None:
                excluded.append(
                    ExcludedCandidate(
                        symbol=result.symbol,
                        reason_code="metric_none",
                        source_result=result,
                    )
                )
                continue

            if isinstance(value, float) and math.isnan(value):
                excluded.append(
                    ExcludedCandidate(
                        symbol=result.symbol,
                        reason_code="metric_nan",
                        source_result=result,
                    )
                )
                continue

            rankable.append((result.symbol, float(value)))

        if config.direction == "desc":
            rankable.sort(key=lambda item: (-item[1], item[0]))
        else:
            rankable.sort(key=lambda item: (item[1], item[0]))

        results_by_symbol = {result.symbol: result for result in report.results}
        scored = tuple(
            ScoredCandidate(
                symbol=symbol,
                rank=rank,
                metric_used=config.metric,
                observed_value=value,
                source_result=results_by_symbol[symbol],
            )
            for rank, (symbol, value) in enumerate(rankable, start=1)
        )
        excluded.sort(key=lambda item: item.symbol)

        return ScoreReport(
            source_scanner_config_id=report.config_id,
            source_scanner_version=report.dataset_metadata.scanner_version,
            scorer_config_id=config.config_id,
            evaluation_date=report.requested_evaluation_date,
            results=scored,
            excluded=tuple(excluded),
        )
