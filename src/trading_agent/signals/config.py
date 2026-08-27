"""Deterministic, explicit configuration for the Signal layer.

``SignalConfig`` is intentionally independent of ``RiskConfig``,
``StrategyConfig``, ``ScorerConfig``, and ``ScannerConfig``: it does not
import any of them, reference any of them, or share identity with any of
them. Signal's only input at run time is a ``RiskReport`` (a value), never
the configurations that produced it.

The field is named ``trade_direction`` (not ``direction``) to avoid any
confusion with ``ScorerConfig.direction``, which means the ranking order
(``"asc"``/``"desc"``) and has nothing to do with the direction of a trade.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from trading_agent.signals.exceptions import SignalConfigError

TradeDirection = Literal["long"]

_TRADE_DIRECTIONS: tuple[TradeDirection, ...] = ("long",)


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Immutable, explicit configuration for one :class:`SignalEngine` run.

    Construct through :meth:`SignalConfig.create`, which validates that
    ``trade_direction`` is one of the supported values. This is deliberately
    the smallest possible configuration: it says nothing about position
    sizing, capital, exposure, stop-loss, take-profit, portfolio state,
    orders, or execution — those remain later milestones. Only ``"long"``
    is supported in this version; short selling is not implemented.
    """

    version: str
    trade_direction: TradeDirection

    @classmethod
    def create(cls, *, trade_direction: str, version: str = "1") -> "SignalConfig":
        if trade_direction not in _TRADE_DIRECTIONS:
            raise SignalConfigError(
                f"Unsupported trade_direction {trade_direction!r}; "
                f"expected one of {_TRADE_DIRECTIONS}."
            )
        return cls(version=version, trade_direction=trade_direction)

    @property
    def config_id(self) -> str:
        """A deterministic identifier derived from this configuration's semantic content.

        The same ``(version, trade_direction)`` always yields the same id;
        changing either changes it. This id is entirely independent of any
        ``RiskConfig.config_id``, ``StrategyConfig.config_id``,
        ``ScorerConfig.config_id``, or ``ScannerConfig.config_id``.
        """
        canonical = {"version": self.version, "trade_direction": self.trade_direction}
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
