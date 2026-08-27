"""Exceptions raised by market-data components."""


class MarketDataError(Exception):
    """Base class for market-data failures."""


class InvalidDateRangeError(MarketDataError):
    """Raised when a request has malformed or inverted dates."""


class EmptyDataError(MarketDataError):
    """Raised when a provider or data set returns no observations."""


class SymbolNotFoundError(MarketDataError):
    """Raised when one or more requested symbols have no data."""


class DataValidationError(MarketDataError):
    """Raised when normalized OHLCV data does not meet quality requirements."""


class DataOutsideRequestedRangeError(DataValidationError):
    """Raised when a provider returns observations outside the requested range."""


class ProviderUnavailableError(MarketDataError):
    """Raised when a provider cannot be reached or cannot serve a request."""
