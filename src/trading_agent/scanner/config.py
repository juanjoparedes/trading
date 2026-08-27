"""Deterministic, explicit configuration for the Market Scanner.

``ScannerConfig`` never calculates anything itself. It only declares what the
scanner is allowed to look at (a universe, a set of required indicators) and
how to judge each symbol (hard filters, soft conditions, staleness policy).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Sequence

from trading_agent.indicators.engine import IndicatorConfig
from trading_agent.scanner.exceptions import ScannerConfigError

Operator = Literal[">=", "<=", ">", "<"]
WarmupPolicy = Literal["reject_insufficient"]

_OPERATORS: tuple[Operator, ...] = (">=", "<=", ">", "<")

#: Columns always present on normalized OHLCV data, independent of any
#: indicator the scanner was asked to compute.
BASE_FIELDS: frozenset[str] = frozenset({"open", "high", "low", "close", "volume"})


@dataclass(frozen=True, slots=True)
class HardFilter:
    """An objective, configurable pass/fail condition.

    Exactly one of ``threshold`` (a numeric constant) or ``compare_field``
    (another column, e.g. ``sma_20``) must be supplied — this is what allows
    both ``close >= 10`` and ``close > sma_20`` style filters without a
    general expression language.
    """

    filter_id: str
    field: str
    operator: Operator
    threshold: float | None = None
    compare_field: str | None = None

    def __post_init__(self) -> None:
        if not self.filter_id.strip():
            raise ScannerConfigError("HardFilter.filter_id must be non-empty.")
        if not self.field.strip():
            raise ScannerConfigError(f"HardFilter {self.filter_id!r}: field must be non-empty.")
        if self.operator not in _OPERATORS:
            raise ScannerConfigError(
                f"HardFilter {self.filter_id!r}: unsupported operator {self.operator!r}; "
                f"expected one of {_OPERATORS}."
            )
        has_threshold = self.threshold is not None
        has_compare_field = self.compare_field is not None
        if has_threshold == has_compare_field:
            raise ScannerConfigError(
                f"HardFilter {self.filter_id!r}: exactly one of threshold or compare_field is required."
            )
        if has_threshold and not isinstance(self.threshold, (int, float)):
            raise ScannerConfigError(f"HardFilter {self.filter_id!r}: threshold must be numeric.")


@dataclass(frozen=True, slots=True)
class SoftCondition:
    """A metric recorded for context. Soft conditions never reject a symbol."""

    field: str

    def __post_init__(self) -> None:
        if not self.field.strip():
            raise ScannerConfigError("SoftCondition.field must be non-empty.")


def _normalize_universe(universe: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(symbol.strip().upper() for symbol in universe if symbol.strip())
    if not normalized:
        raise ScannerConfigError("universe must contain at least one non-empty symbol.")
    if len(set(normalized)) != len(normalized):
        raise ScannerConfigError("universe symbols must be unique.")
    return normalized


def _normalize_indicator_requirements(
    indicators: Sequence[str | IndicatorConfig],
) -> tuple[IndicatorConfig, ...]:
    resolved: list[IndicatorConfig] = []
    for indicator in indicators:
        config = indicator if isinstance(indicator, IndicatorConfig) else IndicatorConfig(indicator)
        try:
            config.resolved_window()
        except (KeyError, ValueError) as error:
            raise ScannerConfigError(f"Invalid indicator requirement {indicator!r}: {error}") from error
        resolved.append(config)
    column_names = [config.column_name() for config in resolved]
    if len(set(column_names)) != len(column_names):
        raise ScannerConfigError("Each indicator requirement must produce a unique column name.")
    return tuple(resolved)


@dataclass(frozen=True, slots=True)
class ScannerConfig:
    """Immutable, explicit configuration for one :class:`MarketScanner` run.

    Construct through :meth:`ScannerConfig.create`, which normalizes inputs
    (uppercasing symbols, wrapping bare indicator names) and validates that
    every filter and soft condition only references a field the scanner will
    actually have available: a base OHLCV column or a declared indicator.
    """

    version: str
    universe: tuple[str, ...]
    indicator_requirements: tuple[IndicatorConfig, ...]
    hard_filters: tuple[HardFilter, ...] = field(default_factory=tuple)
    soft_conditions: tuple[SoftCondition, ...] = field(default_factory=tuple)
    max_staleness_days: int = 0
    warmup_policy: WarmupPolicy = "reject_insufficient"

    @classmethod
    def create(
        cls,
        *,
        universe: Sequence[str],
        indicator_requirements: Sequence[str | IndicatorConfig] = (),
        hard_filters: Sequence[HardFilter] = (),
        soft_conditions: Sequence[SoftCondition] = (),
        max_staleness_days: int = 0,
        warmup_policy: WarmupPolicy = "reject_insufficient",
        version: str = "1",
    ) -> "ScannerConfig":
        if not isinstance(max_staleness_days, int) or isinstance(max_staleness_days, bool) or max_staleness_days < 0:
            raise ScannerConfigError("max_staleness_days must be a non-negative integer.")
        if warmup_policy != "reject_insufficient":
            raise ScannerConfigError(
                f"Unsupported warmup_policy {warmup_policy!r}; only 'reject_insufficient' is implemented."
            )

        normalized_universe = _normalize_universe(universe)
        normalized_indicators = _normalize_indicator_requirements(indicator_requirements)
        available_fields = BASE_FIELDS | {config.column_name() for config in normalized_indicators}

        normalized_hard_filters = tuple(hard_filters)
        normalized_soft_conditions = tuple(soft_conditions)
        _require_known_fields(normalized_hard_filters, normalized_soft_conditions, available_fields)

        filter_ids = [item.filter_id for item in normalized_hard_filters]
        if len(set(filter_ids)) != len(filter_ids):
            raise ScannerConfigError("hard_filters must have unique filter_id values.")

        return cls(
            version=version,
            universe=normalized_universe,
            indicator_requirements=normalized_indicators,
            hard_filters=normalized_hard_filters,
            soft_conditions=normalized_soft_conditions,
            max_staleness_days=max_staleness_days,
            warmup_policy=warmup_policy,
        )

    @property
    def config_id(self) -> str:
        """A deterministic identifier derived from this configuration's semantic content.

        The same configuration always yields the same id, and changing any
        material field (version, universe, indicator_requirements, a filter's
        or soft condition's own values, max_staleness_days, warmup_policy)
        changes it. The *declaration order* of ``hard_filters`` and
        ``soft_conditions`` is not material — each is canonicalized into a
        stable order (by ``filter_id`` and by ``field`` respectively) before
        hashing, so reordering them alone never changes the id. Evaluation
        order and the order presented in a ``ScanReport`` are untouched by
        this — this canonicalization exists solely for identity purposes.
        """
        canonical = {
            "version": self.version,
            "universe": list(self.universe),
            "indicator_requirements": [asdict(config) for config in self.indicator_requirements],
            "hard_filters": sorted(
                (asdict(hard_filter) for hard_filter in self.hard_filters),
                key=lambda item: item["filter_id"],
            ),
            "soft_conditions": sorted(
                (asdict(condition) for condition in self.soft_conditions),
                key=lambda item: item["field"],
            ),
            "max_staleness_days": self.max_staleness_days,
            "warmup_policy": self.warmup_policy,
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def required_indicator_columns(self) -> tuple[str, ...]:
        """Column names the Indicators Engine must produce for this config."""
        return tuple(config.column_name() for config in self.indicator_requirements)


def _require_known_fields(
    hard_filters: Sequence[HardFilter],
    soft_conditions: Sequence[SoftCondition],
    available_fields: frozenset[str],
) -> None:
    for hard_filter in hard_filters:
        if hard_filter.field not in available_fields:
            raise ScannerConfigError(
                f"HardFilter {hard_filter.filter_id!r} references unknown field {hard_filter.field!r}; "
                "it must be a base OHLCV column or a declared indicator requirement."
            )
        if hard_filter.compare_field is not None and hard_filter.compare_field not in available_fields:
            raise ScannerConfigError(
                f"HardFilter {hard_filter.filter_id!r} references unknown compare_field "
                f"{hard_filter.compare_field!r}; it must be a base OHLCV column or a declared "
                "indicator requirement."
            )
    for condition in soft_conditions:
        if condition.field not in available_fields:
            raise ScannerConfigError(
                f"SoftCondition references unknown field {condition.field!r}; it must be a base "
                "OHLCV column or a declared indicator requirement."
            )
