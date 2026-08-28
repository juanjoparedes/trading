"""Exceptions raised by the execution layer."""


class ExecutionError(Exception):
    """Base class for execution failures."""


class ExecutionConfigError(ExecutionError):
    """Raised when an :class:`ExecutionConfig` is structurally invalid."""
