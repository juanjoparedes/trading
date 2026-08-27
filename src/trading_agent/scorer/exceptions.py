"""Exceptions raised by the opportunity-scorer layer."""


class ScorerError(Exception):
    """Base class for opportunity-scorer failures."""


class ScorerConfigError(ScorerError):
    """Raised when a :class:`ScorerConfig` is structurally invalid."""
