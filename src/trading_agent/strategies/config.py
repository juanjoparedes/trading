"""Deterministic, explicit configuration for the Strategy layer.

``StrategyConfig`` is intentionally independent of ``ScorerConfig`` and
``ScannerConfig``: it does not import either, reference either, or share
identity with either. Strategy's only input at run time is a ``ScoreReport``
(a value), never the configurations that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from trading_agent.strategies.exceptions import StrategyConfigError


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Immutable, explicit configuration for one :class:`StrategyEngine` run.

    Construct through :meth:`StrategyConfig.create`, which validates that
    ``max_candidates`` is a positive integer. This is deliberately the
    smallest possible configuration: it says nothing about direction,
    position sizing, stop-loss, take-profit, or execution — those remain
    later milestones.
    """

    version: str
    max_candidates: int

    @classmethod
    def create(cls, *, max_candidates: int, version: str = "1") -> "StrategyConfig":
        if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates <= 0:
            raise StrategyConfigError("max_candidates must be a positive integer.")
        return cls(version=version, max_candidates=max_candidates)

    @property
    def config_id(self) -> str:
        """A deterministic identifier derived from this configuration's semantic content.

        The same ``(version, max_candidates)`` always yields the same id;
        changing either changes it. This id is entirely independent of any
        ``ScorerConfig.config_id`` or ``ScannerConfig.config_id``.
        """
        canonical = {"version": self.version, "max_candidates": self.max_candidates}
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
