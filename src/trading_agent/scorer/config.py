"""Deterministic, explicit configuration for the Opportunity Scorer.

``ScorerConfig`` is intentionally independent of ``ScannerConfig``: it does
not import it, reference it, or share identity with it. The scorer's only
input at run time is a ``ScanReport`` (a value), never the configuration that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from trading_agent.scorer.exceptions import ScorerConfigError

Direction = Literal["asc", "desc"]

_DIRECTIONS: tuple[Direction, ...] = ("asc", "desc")


@dataclass(frozen=True, slots=True)
class ScorerConfig:
    """Immutable, explicit configuration for one :class:`OpportunityScorer` run.

    Construct through :meth:`ScorerConfig.create`, which validates only what
    is structurally verifiable at this point: that ``metric`` is non-blank
    and ``direction`` is one of the supported values. Whether ``metric``
    actually exists in a given ``ScanReport`` cannot be known until
    :meth:`OpportunityScorer.score` runs against a real report, so that check
    is deferred there rather than performed here.
    """

    version: str
    metric: str
    direction: Direction

    @classmethod
    def create(cls, *, metric: str, direction: Direction, version: str = "1") -> "ScorerConfig":
        if not metric.strip():
            raise ScorerConfigError("metric must be non-empty.")
        if direction not in _DIRECTIONS:
            raise ScorerConfigError(
                f"Unsupported direction {direction!r}; expected one of {_DIRECTIONS}."
            )
        return cls(version=version, metric=metric.strip(), direction=direction)

    @property
    def config_id(self) -> str:
        """A deterministic identifier derived from this configuration's semantic content.

        The same ``(version, metric, direction)`` always yields the same id;
        changing any one of them changes it. This id is entirely independent
        of any ``ScannerConfig.config_id`` — the two identity spaces never mix.
        """
        canonical = {
            "version": self.version,
            "metric": self.metric,
            "direction": self.direction,
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
