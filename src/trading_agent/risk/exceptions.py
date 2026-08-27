"""Exceptions raised by the risk layer."""


class RiskError(Exception):
    """Base class for risk-management failures."""


class RiskConfigError(RiskError):
    """Raised when a :class:`RiskConfig` is structurally invalid."""
