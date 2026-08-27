"""Exceptions raised by the strategy layer."""


class StrategyError(Exception):
    """Base class for strategy failures."""


class StrategyConfigError(StrategyError):
    """Raised when a :class:`StrategyConfig` is structurally invalid."""
