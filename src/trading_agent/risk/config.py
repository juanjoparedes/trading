"""Deterministic, explicit configuration for the Risk layer.

``RiskConfig`` is intentionally independent of ``StrategyConfig``,
``ScorerConfig``, and ``ScannerConfig``: it does not import any of them,
reference any of them, or share identity with any of them. Risk's only
input at run time is a ``StrategyReport`` (a value), never the
configurations that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from trading_agent.risk.exceptions import RiskConfigError


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Immutable, explicit configuration for one :class:`RiskEngine` run.

    Construct through :meth:`RiskConfig.create`, which validates that
    ``max_approved_decisions`` is a positive integer. This is deliberately
    the smallest possible configuration: a static capacity gate. It says
    nothing about direction, position sizing, stop-loss, take-profit,
    capital, real exposure, or execution — those remain later milestones.
    """

    version: str
    max_approved_decisions: int

    @classmethod
    def create(cls, *, max_approved_decisions: int, version: str = "1") -> "RiskConfig":
        if (
            not isinstance(max_approved_decisions, int)
            or isinstance(max_approved_decisions, bool)
            or max_approved_decisions <= 0
        ):
            raise RiskConfigError("max_approved_decisions must be a positive integer.")
        return cls(version=version, max_approved_decisions=max_approved_decisions)

    @property
    def config_id(self) -> str:
        """A deterministic identifier derived from this configuration's semantic content.

        The same ``(version, max_approved_decisions)`` always yields the
        same id; changing either changes it. This id is entirely independent
        of any ``StrategyConfig.config_id``, ``ScorerConfig.config_id``, or
        ``ScannerConfig.config_id``.
        """
        canonical = {"version": self.version, "max_approved_decisions": self.max_approved_decisions}
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
