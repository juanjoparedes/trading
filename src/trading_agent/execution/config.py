"""Deterministic, explicit configuration for the Execution layer.

``ExecutionConfig`` is intentionally independent of ``PortfolioConfig``,
``SignalConfig``, ``RiskConfig``, ``StrategyConfig``, ``ScorerConfig``, and
``ScannerConfig``: it does not import any of them, reference any of them, or
share identity with any of them. Execution's only input at run time is a
``PortfolioReport`` (a value), never the configurations that produced it.

This is deliberately the smallest possible configuration: a single
``version`` field, with no business parameter at all. That absence is not an
omission — broker selection, order type, slippage, commissions, fills, and
real execution are all explicitly out of scope for this version, so there is
no decision left for this config to carry. It still exists, and still has
its own ``config_id``, because a config (and the identity it provides) is
mandatory for every layer in this architecture, independent of how much or
how little it currently parameterizes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Immutable, explicit configuration for one :class:`ExecutionEngine` run."""

    version: str = "1"

    @classmethod
    def create(cls, *, version: str = "1") -> "ExecutionConfig":
        return cls(version=version)

    @property
    def config_id(self) -> str:
        """A deterministic identifier derived from this configuration's semantic content.

        The same ``version`` always yields the same id; changing it changes
        the id. This id is entirely independent of any
        ``PortfolioConfig.config_id``, ``SignalConfig.config_id``,
        ``RiskConfig.config_id``, ``StrategyConfig.config_id``,
        ``ScorerConfig.config_id``, or ``ScannerConfig.config_id``.
        """
        canonical = {"version": self.version}
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
