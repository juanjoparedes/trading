"""Opportunity Scorer: deterministic ranking of Market Scanner candidates."""

from trading_agent.scorer.config import ScorerConfig
from trading_agent.scorer.engine import OpportunityScorer
from trading_agent.scorer.exceptions import ScorerConfigError, ScorerError
from trading_agent.scorer.models import ExcludedCandidate, ScoredCandidate, ScoreReport

__all__ = [
    "ExcludedCandidate",
    "OpportunityScorer",
    "ScoreReport",
    "ScoredCandidate",
    "ScorerConfig",
    "ScorerConfigError",
    "ScorerError",
]
