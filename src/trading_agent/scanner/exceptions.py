"""Exceptions raised by the market-scanner layer."""


class ScannerError(Exception):
    """Base class for market-scanner failures."""


class ScannerConfigError(ScannerError):
    """Raised when a :class:`ScannerConfig` is structurally invalid."""
