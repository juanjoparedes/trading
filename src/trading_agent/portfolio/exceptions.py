"""Exceptions raised by the portfolio layer."""


class PortfolioError(Exception):
    """Base class for portfolio failures."""


class PortfolioConfigError(PortfolioError):
    """Raised when a :class:`PortfolioConfig` is structurally invalid."""


class PortfolioSnapshotDateMismatchError(PortfolioError):
    """Raised when ``market_snapshot.as_of_date`` does not equal ``signal_report.evaluation_date``."""
