"""Exceptions raised by the signal layer."""


class SignalError(Exception):
    """Base class for signal-generation failures."""


class SignalConfigError(SignalError):
    """Raised when a :class:`SignalConfig` is structurally invalid."""
