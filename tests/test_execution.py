"""Tests for the Execution layer.

These tests construct ``PortfolioReport`` / ``TargetPosition`` /
``SignalDecision`` / ``RiskDecision`` / ``StrategyDecision`` /
``ScoredCandidate`` / ``SymbolScanResult`` values directly rather than
running the real
``MarketScanner``/``OpportunityScorer``/``StrategyEngine``/``RiskEngine``/
``SignalEngine``/``PortfolioEngine`` pipeline — Execution's contract is with
the *shape* of a ``PortfolioReport``, not with how it was produced, and the
execution package must never import pandas, ``DataEngine``,
``IndicatorsEngine``, ``MarketScanner``, ``ScannerConfig``, ``ScorerConfig``,
``StrategyConfig``, ``RiskConfig``, ``SignalConfig``, or ``PortfolioConfig``
to build one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_agent.execution import (
    ExecutionConfig,
    ExecutionEngine,
    ExecutionReport,
    OrderIntent,
)
from trading_agent.portfolio.models import PortfolioReport, TargetPosition
from trading_agent.risk.models import RiskDecision
from trading_agent.scanner.models import SymbolScanResult
from trading_agent.scorer.models import ScoredCandidate
from trading_agent.signals.models import SignalDecision
from trading_agent.strategies.models import StrategyDecision

EVALUATION_DATE = date(2024, 6, 28)


def _source_result(symbol: str) -> SymbolScanResult:
    return SymbolScanResult(
        symbol=symbol,
        requested_evaluation_date=EVALUATION_DATE,
        as_of_date=EVALUATION_DATE,
        status="candidate",
        passed_filters=(),
        failed_filters=(),
        soft_conditions=(),
        metrics={"rsi_14": 55.0, "close": 100.0},
        required_indicators=(),
        data_quality_notes=(),
    )


def _candidate(symbol: str, rank: int, value: float = 1.0) -> ScoredCandidate:
    return ScoredCandidate(
        symbol=symbol,
        rank=rank,
        metric_used="rsi_14",
        observed_value=value,
        source_result=_source_result(symbol),
    )


def _strategy_decision(symbol: str, rank: int) -> StrategyDecision:
    return StrategyDecision(
        symbol=symbol,
        action="enter",
        reason_code="selected_top_ranked",
        source_candidate=_candidate(symbol, rank),
    )


def _risk_decision(symbol: str, rank: int) -> RiskDecision:
    return RiskDecision(
        symbol=symbol,
        status="approved",
        reason_code="within_risk_limit",
        source_decision=_strategy_decision(symbol, rank),
    )


def _signal_decision(symbol: str, rank: int) -> SignalDecision:
    return SignalDecision(
        symbol=symbol,
        trade_direction="long",
        reason_code="approved_by_risk",
        source_decision=_risk_decision(symbol, rank),
    )


def _target_position(symbol: str, rank: int, quantity: int = 5, target_value: float = 500.0) -> TargetPosition:
    return TargetPosition(
        symbol=symbol,
        quantity=quantity,
        target_value=target_value,
        source_signal=_signal_decision(symbol, rank),
    )


def _portfolio_report(
    target_positions: tuple[TargetPosition, ...],
    *,
    source_scanner_config_id: str = "scanner-cfg-abc",
    source_scorer_config_id: str = "scorer-cfg-xyz",
    source_strategy_config_id: str = "strategy-cfg-123",
    source_risk_config_id: str = "risk-cfg-789",
    source_signal_config_id: str = "signal-cfg-456",
    portfolio_config_id: str = "portfolio-cfg-321",
    evaluation_date: date = EVALUATION_DATE,
    unallocated_cash: float = 0.0,
) -> PortfolioReport:
    return PortfolioReport(
        source_scanner_config_id=source_scanner_config_id,
        source_scorer_config_id=source_scorer_config_id,
        source_strategy_config_id=source_strategy_config_id,
        source_risk_config_id=source_risk_config_id,
        source_signal_config_id=source_signal_config_id,
        portfolio_config_id=portfolio_config_id,
        evaluation_date=evaluation_date,
        target_positions=target_positions,
        excluded=(),
        unallocated_cash=unallocated_cash,
    )


def _config(version: str = "1") -> ExecutionConfig:
    return ExecutionConfig.create(version=version)


# --- 1. Basic contract ----------------------------------------------------------


def test_execution_accepts_a_valid_portfolio_report_and_returns_an_execution_report():
    report = _portfolio_report((_target_position("AAPL", 1),))
    result = ExecutionEngine().prepare(report, config=_config())
    assert isinstance(result, ExecutionReport)
    assert len(result.orders) == 1


# --- 2. Empty PortfolioReport ------------------------------------------------------


def test_empty_portfolio_report_produces_no_orders():
    report = _portfolio_report(())
    result = ExecutionEngine().prepare(report, config=_config())
    assert result.orders == ()


# --- 3. One position / multiple positions -------------------------------------------


def test_one_position_produces_exactly_one_order():
    report = _portfolio_report((_target_position("AAPL", 1),))
    result = ExecutionEngine().prepare(report, config=_config())
    assert len(result.orders) == 1
    assert result.orders[0].symbol == "AAPL"


def test_multiple_positions_produce_one_order_each():
    report = _portfolio_report(
        (_target_position("AAPL", 1), _target_position("MSFT", 2), _target_position("QQQ", 3))
    )
    result = ExecutionEngine().prepare(report, config=_config())
    assert len(result.orders) == 3
    assert {order.symbol for order in result.orders} == {"AAPL", "MSFT", "QQQ"}


# --- 4. Determinism -----------------------------------------------------------


def test_prepare_is_deterministic_across_repeated_calls():
    report = _portfolio_report((_target_position("AAPL", 1), _target_position("MSFT", 2)))
    config = _config()
    engine = ExecutionEngine()
    assert engine.prepare(report, config=config) == engine.prepare(report, config=config)


# --- 5. Order independence -----------------------------------------------------


def test_prepare_result_is_independent_of_target_positions_input_order():
    ordered = _portfolio_report(
        (_target_position("AAPL", 1), _target_position("MSFT", 2), _target_position("QQQ", 3))
    )
    reversed_report = _portfolio_report(tuple(reversed(ordered.target_positions)))
    config = _config()
    engine = ExecutionEngine()
    assert engine.prepare(ordered, config=config) == engine.prepare(reversed_report, config=config)


def test_orders_are_ordered_by_rank_ascending():
    shuffled = _portfolio_report(
        (_target_position("QQQ", 3), _target_position("AAPL", 1), _target_position("MSFT", 2))
    )
    result = ExecutionEngine().prepare(shuffled, config=_config())
    ranks = [
        order.source_position.source_signal.source_decision.source_decision.source_candidate.rank
        for order in result.orders
    ]
    assert ranks == [1, 2, 3]


def test_duplicate_ranks_break_ties_by_symbol_ascending_deterministically():
    """A malformed PortfolioReport (PortfolioEngine never produces duplicate
    ranks among target positions coming from a real SignalReport) must still
    yield a deterministic, order-independent result: ties on rank are broken
    by symbol ascending, never by input order."""
    forward = _portfolio_report(
        (_target_position("MSFT", 1), _target_position("AAPL", 1), _target_position("QQQ", 2))
    )
    reversed_input = _portfolio_report(tuple(reversed(forward.target_positions)))
    config = _config()
    engine = ExecutionEngine()

    result_forward = engine.prepare(forward, config=config)
    result_reversed = engine.prepare(reversed_input, config=config)

    assert engine.prepare(forward, config=config) == result_forward
    assert result_forward == result_reversed
    assert [order.symbol for order in result_forward.orders] == ["AAPL", "MSFT", "QQQ"]


# --- 6. quantity and symbol conserved exactly ---------------------------------------


def test_quantity_is_conserved_exactly_from_target_position():
    report = _portfolio_report((_target_position("AAPL", 1, quantity=7),))
    result = ExecutionEngine().prepare(report, config=_config())
    assert result.orders[0].quantity == 7


def test_symbol_is_conserved_exactly_from_target_position():
    report = _portfolio_report((_target_position("MSFT", 1),))
    result = ExecutionEngine().prepare(report, config=_config())
    assert result.orders[0].symbol == "MSFT"


# --- 7. side == "buy" and absence of short ------------------------------------------


def test_side_is_buy_for_a_long_trade_direction():
    report = _portfolio_report((_target_position("AAPL", 1),))
    result = ExecutionEngine().prepare(report, config=_config())
    assert result.orders[0].side == "buy"


def test_order_side_field_admits_no_value_other_than_buy_structurally():
    # Every order produced from the current pipeline is "buy"; no code path
    # in this version can produce anything else, since SignalConfig only
    # ever supports "long".
    report = _portfolio_report(
        (_target_position("AAPL", 1), _target_position("MSFT", 2), _target_position("QQQ", 3))
    )
    result = ExecutionEngine().prepare(report, config=_config())
    assert {order.side for order in result.orders} == {"buy"}


# --- 8. status == "simulated" ------------------------------------------------------


def test_status_is_always_simulated():
    report = _portfolio_report((_target_position("AAPL", 1),))
    result = ExecutionEngine().prepare(report, config=_config())
    assert result.orders[0].status == "simulated"


# --- 9. Traceability -------------------------------------------------------------


def test_order_intent_preserves_exact_reference_to_the_original_target_position():
    position = _target_position("AAPL", 1)
    report = _portfolio_report((position,))
    result = ExecutionEngine().prepare(report, config=_config())
    assert result.orders[0].source_position is position
    assert result.orders[0].source_position.source_signal is position.source_signal


# --- 10. Propagation of the 7 config ids and evaluation_date ------------------------


def test_execution_report_carries_all_seven_upstream_and_own_config_ids():
    report = _portfolio_report(
        (_target_position("AAPL", 1),),
        source_scanner_config_id="a-specific-scanner-id",
        source_scorer_config_id="a-specific-scorer-id",
        source_strategy_config_id="a-specific-strategy-id",
        source_risk_config_id="a-specific-risk-id",
        source_signal_config_id="a-specific-signal-id",
        portfolio_config_id="a-specific-portfolio-id",
    )
    config = _config()
    result = ExecutionEngine().prepare(report, config=config)
    assert result.source_scanner_config_id == "a-specific-scanner-id"
    assert result.source_scorer_config_id == "a-specific-scorer-id"
    assert result.source_strategy_config_id == "a-specific-strategy-id"
    assert result.source_risk_config_id == "a-specific-risk-id"
    assert result.source_signal_config_id == "a-specific-signal-id"
    assert result.source_portfolio_config_id == "a-specific-portfolio-id"
    assert result.execution_config_id == config.config_id


def test_execution_report_carries_evaluation_date():
    report = _portfolio_report((_target_position("AAPL", 1),))
    result = ExecutionEngine().prepare(report, config=_config())
    assert result.evaluation_date == EVALUATION_DATE


# --- 11. ExecutionConfig independence and determinism --------------------------------


def test_execution_config_has_no_coupling_to_other_layer_configs():
    import trading_agent.execution.config as config_module

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "from trading_agent.portfolio" not in source
    assert "from trading_agent.signals" not in source
    assert "from trading_agent.risk" not in source
    assert "from trading_agent.strategies" not in source
    assert "from trading_agent.scorer" not in source
    assert "from trading_agent.scanner" not in source


def test_config_id_is_deterministic_for_identical_configs():
    first = ExecutionConfig.create(version="1")
    second = ExecutionConfig.create(version="1")
    assert first.config_id == second.config_id


def test_config_id_changes_when_version_changes():
    base = ExecutionConfig.create(version="1")
    changed = ExecutionConfig.create(version="2")
    assert base.config_id != changed.config_id


def test_config_id_is_independent_of_other_layer_config_ids():
    config = ExecutionConfig.create()
    assert config.config_id not in {
        "portfolio-cfg-321", "signal-cfg-456", "risk-cfg-789", "strategy-cfg-123", "scorer-cfg-xyz", "scanner-cfg-abc",
    }


def test_execution_config_default_version_is_one():
    assert ExecutionConfig().version == "1"
    assert ExecutionConfig.create().version == "1"


# --- 12. Immutability and absence of side effects ------------------------------------


def test_input_portfolio_report_and_positions_are_not_mutated():
    position = _target_position("AAPL", 1)
    report = _portfolio_report((position,))
    report_before = report
    position_before = position

    ExecutionEngine().prepare(report, config=_config())

    assert report == report_before
    assert position == position_before
    assert report.target_positions[0] is position


def test_prepare_performs_no_side_effects_beyond_returning_a_value():
    report = _portfolio_report((_target_position("AAPL", 1),))
    config = _config()
    engine = ExecutionEngine()
    result = engine.prepare(report, config=config)
    result_again = engine.prepare(report, config=config)
    assert result == result_again


# --- 13. Structural independence: forbidden imports -----------------------------------


def test_execution_package_never_imports_forbidden_dependencies():
    import trading_agent.execution.config as config_module
    import trading_agent.execution.engine as engine_module
    import trading_agent.execution.exceptions as exceptions_module
    import trading_agent.execution.models as models_module

    forbidden_import_substrings = (
        "import pandas",
        "import numpy",
        "import trading_agent.data",
        "from trading_agent.data",
        "import trading_agent.indicators",
        "from trading_agent.indicators",
        "from trading_agent.scanner",
        "import trading_agent.scanner",
        "from trading_agent.scorer",
        "import trading_agent.scorer",
        "from trading_agent.strategies",
        "import trading_agent.strategies",
        "from trading_agent.risk",
        "import trading_agent.risk",
        "from trading_agent.signals",
        "import trading_agent.signals",
        "from trading_agent.portfolio.config",
        "import trading_agent.portfolio.config",
        "from trading_agent.portfolio.engine",
        "import trading_agent.portfolio.engine",
        "import yfinance",
    )
    for module in (config_module, engine_module, exceptions_module, models_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in forbidden_import_substrings:
            assert forbidden not in source, f"{module.__name__} must not reference {forbidden!r}"
        assert "MarketScanner(" not in source
        assert "ScannerConfig(" not in source
        assert "ScorerConfig(" not in source
        assert "StrategyConfig(" not in source
        assert "RiskConfig(" not in source
        assert "SignalConfig(" not in source
        assert "PortfolioConfig(" not in source
        assert "MarketSnapshot(" not in source


def test_execution_engine_never_reads_source_signal_metrics_for_price():
    import trading_agent.execution.engine as engine_module

    source = Path(engine_module.__file__).read_text(encoding="utf-8")
    assert "source_result" not in source
    assert ".metrics" not in source
    assert "metrics[" not in source
    assert "market_snapshot" not in source.lower()


# --- 14. Absence of Market Data / broker / real execution / P&L / fills --------------


def test_order_intent_has_no_order_type_price_broker_or_fill_fields():
    fields = set(OrderIntent.__dataclass_fields__)
    assert fields == {"symbol", "side", "quantity", "status", "source_position"}


def test_execution_report_has_no_extra_fields():
    fields = set(ExecutionReport.__dataclass_fields__)
    assert fields == {
        "source_scanner_config_id",
        "source_scorer_config_id",
        "source_strategy_config_id",
        "source_risk_config_id",
        "source_signal_config_id",
        "source_portfolio_config_id",
        "execution_config_id",
        "evaluation_date",
        "orders",
    }


def test_execution_config_has_only_version_field():
    fields = set(ExecutionConfig.__dataclass_fields__)
    assert fields == {"version"}


# --- Full-value equality ------------------------------------------------------------


def test_full_execution_report_equality_between_independent_constructions():
    position = _target_position("AAPL", 1, quantity=5, target_value=500.0)
    report = _portfolio_report(
        (position,),
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        source_strategy_config_id="cfg-c",
        source_risk_config_id="cfg-d",
        source_signal_config_id="cfg-e",
        portfolio_config_id="cfg-f",
    )
    config = _config()
    engine = ExecutionEngine()

    first = engine.prepare(report, config=config)
    second = engine.prepare(report, config=config)

    assert first == second
    assert first == ExecutionReport(
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        source_strategy_config_id="cfg-c",
        source_risk_config_id="cfg-d",
        source_signal_config_id="cfg-e",
        source_portfolio_config_id="cfg-f",
        execution_config_id=config.config_id,
        evaluation_date=EVALUATION_DATE,
        orders=(
            OrderIntent(
                symbol="AAPL",
                side="buy",
                quantity=5,
                status="simulated",
                source_position=position,
            ),
        ),
    )
