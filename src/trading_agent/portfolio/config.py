"""Deterministic, explicit configuration for the Portfolio layer.

``PortfolioConfig`` is intentionally independent of ``SignalConfig``,
``RiskConfig``, ``StrategyConfig``, ``ScorerConfig``, and ``ScannerConfig``:
it does not import any of them, reference any of them, or share identity
with any of them. Portfolio's only inputs at run time are a
``SignalReport`` and a ``MarketSnapshot`` (values), never the configurations
that produced the former.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from trading_agent.portfolio.exceptions import PortfolioConfigError

AllocationPolicy = Literal["equal_weight"]

_ALLOCATION_POLICIES: tuple[AllocationPolicy, ...] = ("equal_weight",)


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Immutable, explicit configuration for one :class:`PortfolioEngine` run.

    Construct through :meth:`PortfolioConfig.create`, which validates that
    ``initial_capital`` is a positive number and ``allocation_policy`` is one
    of the supported values. This is deliberately the smallest possible
    configuration: it says nothing about risk-per-trade, ATR or volatility
    sizing, stop-loss, take-profit, commissions, slippage, fills, orders, or
    execution — those remain later milestones. Only ``"equal_weight"`` is
    supported in this version.
    """

    version: str
    initial_capital: float
    allocation_policy: AllocationPolicy

    @classmethod
    def create(
        cls,
        *,
        initial_capital: float,
        allocation_policy: str,
        version: str = "1",
    ) -> "PortfolioConfig":
        if (
            not isinstance(initial_capital, (int, float))
            or isinstance(initial_capital, bool)
            or initial_capital <= 0
        ):
            raise PortfolioConfigError("initial_capital must be a positive number.")
        if allocation_policy not in _ALLOCATION_POLICIES:
            raise PortfolioConfigError(
                f"Unsupported allocation_policy {allocation_policy!r}; "
                f"expected one of {_ALLOCATION_POLICIES}."
            )
        return cls(version=version, initial_capital=float(initial_capital), allocation_policy=allocation_policy)

    @property
    def config_id(self) -> str:
        """A deterministic identifier derived from this configuration's semantic content.

        The same ``(version, initial_capital, allocation_policy)`` always
        yields the same id; changing any of them changes it. This id is
        entirely independent of any ``SignalConfig.config_id``,
        ``RiskConfig.config_id``, ``StrategyConfig.config_id``,
        ``ScorerConfig.config_id``, or ``ScannerConfig.config_id``.
        """
        canonical = {
            "version": self.version,
            "initial_capital": self.initial_capital,
            "allocation_policy": self.allocation_policy,
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
