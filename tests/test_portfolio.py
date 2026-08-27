"""Tests for the Portfolio layer.

These tests construct ``SignalReport`` / ``SignalDecision`` / ``RiskDecision``
/ ``StrategyDecision`` / ``ScoredCandidate`` / ``SymbolScanResult`` values
directly rather than running the real
``MarketScanner``/``OpportunityScorer``/``StrategyEngine``/``RiskEngine``/
``SignalEngine`` pipeline — Portfolio's contract is with the *shape* of a
``SignalReport`` plus an explicit ``MarketSnapshot``, not with how either was
produced, and the portfolio package must never import pandas,
``DataEngine``, ``IndicatorsEngine``, ``MarketScanner``, ``ScannerConfig``,
``ScorerConfig``, ``StrategyConfig``, ``RiskConfig``, or ``SignalConfig`` to
build one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_agent.portfolio import (
    MarketSnapshot,
    PortfolioConfig,
    PortfolioConfigError,
    PortfolioEngine,
    PortfolioExclusion,
    PortfolioReport,
    PortfolioSnapshotDateMismatchError,
    TargetPosition,
)
from trading_agent.risk.models import RiskDecision
from trading_agent.scanner.models import SymbolScanResult
from trading_agent.scorer.models import ScoredCandidate
from trading_agent.signals.models import SignalDecision
from trading_agent.strategies.models import StrategyDecision

EVALUATION_DATE = date(2024, 6, 28)
OTHER_DATE = date(2024, 6, 27)


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


def _signal(symbol: str, rank: int) -> SignalDecision:
    return SignalDecision(
        symbol=symbol,
        trade_direction="long",
        reason_code="approved_by_risk",
        source_decision=_risk_decision(symbol, rank),
    )


def _signal_report(
    decisions: tuple[SignalDecision, ...],
    *,
    source_scanner_config_id: str = "scanner-cfg-abc",
    source_scorer_config_id: str = "scorer-cfg-xyz",
    source_strategy_config_id: str = "strategy-cfg-123",
    source_risk_config_id: str = "risk-cfg-789",
    signal_config_id: str = "signal-cfg-456",
    evaluation_date: date = EVALUATION_DATE,
):
    from trading_agent.signals.models import SignalReport

    return SignalReport(
        source_scanner_config_id=source_scanner_config_id,
        source_scorer_config_id=source_scorer_config_id,
        source_strategy_config_id=source_strategy_config_id,
        source_risk_config_id=source_risk_config_id,
        signal_config_id=signal_config_id,
        evaluation_date=evaluation_date,
        decisions=decisions,
    )


def _snapshot(prices: dict[str, float], *, as_of_date: date = EVALUATION_DATE) -> MarketSnapshot:
    return MarketSnapshot(as_of_date=as_of_date, prices=prices)


def _config(initial_capital: float = 1000.0, allocation_policy: str = "equal_weight") -> PortfolioConfig:
    return PortfolioConfig.create(initial_capital=initial_capital, allocation_policy=allocation_policy)


# --- 1. Basic contract ----------------------------------------------------------


def test_portfolio_accepts_a_valid_signal_report_and_returns_a_portfolio_report():
    report = _signal_report((_signal("AAPL", 1),))
    result = PortfolioEngine().build(report, config=_config(), market_snapshot=_snapshot({"AAPL": 100.0}))
    assert isinstance(result, PortfolioReport)
    assert len(result.target_positions) == 1


# --- 2. Equal-weight -------------------------------------------------------------


def test_equal_weight_ideal_allocation_is_capital_divided_by_number_of_signals():
    report = _signal_report((_signal("AAPL", 1), _signal("MSFT", 2)))
    snapshot = _snapshot({"AAPL": 100.0, "MSFT": 100.0})
    result = PortfolioEngine().build(report, config=_config(initial_capital=1000.0), market_snapshot=snapshot)
    by_symbol = {position.symbol: position for position in result.target_positions}
    # ideal_allocation = 1000 / 2 = 500; 500 / 100 = 5 shares each.
    assert by_symbol["AAPL"].quantity == 5
    assert by_symbol["MSFT"].quantity == 5


# --- 3. Floor rounding -------------------------------------------------------------


def test_quantity_uses_floor_rounding_never_up():
    report = _signal_report((_signal("AAPL", 1),))
    # ideal_allocation = 1000; price = 300 -> 3.333... shares -> floor to 3.
    result = PortfolioEngine().build(report, config=_config(1000.0), market_snapshot=_snapshot({"AAPL": 300.0}))
    assert result.target_positions[0].quantity == 3


# --- 4. target_value ---------------------------------------------------------------


def test_target_value_is_quantity_times_price_used_not_ideal_allocation():
    report = _signal_report((_signal("AAPL", 1),))
    result = PortfolioEngine().build(report, config=_config(1000.0), market_snapshot=_snapshot({"AAPL": 300.0}))
    position = result.target_positions[0]
    assert position.quantity == 3
    assert position.target_value == 900.0  # 3 * 300, not the 1000.0 ideal allocation.


# --- 5. unallocated_cash -----------------------------------------------------------


def test_unallocated_cash_is_capital_minus_sum_of_target_values():
    report = _signal_report((_signal("AAPL", 1),))
    result = PortfolioEngine().build(report, config=_config(1000.0), market_snapshot=_snapshot({"AAPL": 300.0}))
    assert result.unallocated_cash == 100.0  # 1000 - 900


def test_unallocated_cash_is_full_capital_when_all_signals_are_excluded():
    report = _signal_report((_signal("AAPL", 1),))
    result = PortfolioEngine().build(report, config=_config(1000.0), market_snapshot=_snapshot({}))
    assert result.unallocated_cash == 1000.0


def test_unallocated_cash_is_deterministic():
    report = _signal_report((_signal("AAPL", 1), _signal("MSFT", 2)))
    snapshot = _snapshot({"AAPL": 300.0, "MSFT": 70.0})
    config = _config(1000.0)
    engine = PortfolioEngine()
    first = engine.build(report, config=config, market_snapshot=snapshot)
    second = engine.build(report, config=config, market_snapshot=snapshot)
    assert first.unallocated_cash == second.unallocated_cash


# --- 6. price_unavailable exclusion -------------------------------------------------


def test_signal_without_a_price_is_excluded_not_positioned():
    report = _signal_report((_signal("AAPL", 1), _signal("MSFT", 2)))
    snapshot = _snapshot({"AAPL": 100.0})  # MSFT has no price.
    result = PortfolioEngine().build(report, config=_config(), market_snapshot=snapshot)
    assert len(result.target_positions) == 1
    assert result.target_positions[0].symbol == "AAPL"
    assert len(result.excluded) == 1
    assert result.excluded[0].symbol == "MSFT"
    assert result.excluded[0].reason_code == "price_unavailable"


# --- 7. allocation_below_one_share exclusion ----------------------------------------


def test_allocation_below_one_share_is_excluded_not_positioned_with_zero_quantity():
    report = _signal_report((_signal("AAPL", 1),))
    # ideal_allocation = 100; price = 1000 -> 0.1 shares -> floor to 0 -> excluded.
    result = PortfolioEngine().build(report, config=_config(100.0), market_snapshot=_snapshot({"AAPL": 1000.0}))
    assert result.target_positions == ()
    assert len(result.excluded) == 1
    assert result.excluded[0].symbol == "AAPL"
    assert result.excluded[0].reason_code == "allocation_below_one_share"


# --- 8. No redistribution after exclusions ------------------------------------------


def test_ideal_allocation_is_not_recomputed_after_an_exclusion():
    # Three signals, capital 900 -> ideal_allocation = 300 each.
    # MSFT has no price and is excluded; AAPL and QQQ must still use 300,
    # not 450 (900 / 2 remaining signals).
    report = _signal_report((_signal("AAPL", 1), _signal("MSFT", 2), _signal("QQQ", 3)))
    snapshot = _snapshot({"AAPL": 100.0, "QQQ": 100.0})
    result = PortfolioEngine().build(report, config=_config(900.0), market_snapshot=snapshot)
    by_symbol = {position.symbol: position for position in result.target_positions}
    # If ideal_allocation were recomputed to 450, quantity would be 4 (floor(450/100)).
    # With the correct, un-recomputed 300, quantity must be 3 (floor(300/100)).
    assert by_symbol["AAPL"].quantity == 3
    assert by_symbol["QQQ"].quantity == 3
    assert {exclusion.symbol for exclusion in result.excluded} == {"MSFT"}


# --- 9. as_of_date validation --------------------------------------------------------


def test_snapshot_date_earlier_than_evaluation_date_raises():
    report = _signal_report((_signal("AAPL", 1),), evaluation_date=EVALUATION_DATE)
    snapshot = _snapshot({"AAPL": 100.0}, as_of_date=OTHER_DATE)
    with pytest.raises(PortfolioSnapshotDateMismatchError):
        PortfolioEngine().build(report, config=_config(), market_snapshot=snapshot)


def test_snapshot_date_later_than_evaluation_date_raises():
    report = _signal_report((_signal("AAPL", 1),), evaluation_date=OTHER_DATE)
    snapshot = _snapshot({"AAPL": 100.0}, as_of_date=EVALUATION_DATE)
    with pytest.raises(PortfolioSnapshotDateMismatchError):
        PortfolioEngine().build(report, config=_config(), market_snapshot=snapshot)


def test_snapshot_date_equal_to_evaluation_date_does_not_raise():
    report = _signal_report((_signal("AAPL", 1),), evaluation_date=EVALUATION_DATE)
    snapshot = _snapshot({"AAPL": 100.0}, as_of_date=EVALUATION_DATE)
    result = PortfolioEngine().build(report, config=_config(), market_snapshot=snapshot)
    assert isinstance(result, PortfolioReport)


# --- 10. Determinism --------------------------------------------------------------


def test_build_is_deterministic_across_repeated_calls():
    report = _signal_report((_signal("AAPL", 1), _signal("MSFT", 2)))
    snapshot = _snapshot({"AAPL": 100.0, "MSFT": 50.0})
    config = _config()
    engine = PortfolioEngine()
    assert engine.build(report, config=config, market_snapshot=snapshot) == engine.build(
        report, config=config, market_snapshot=snapshot
    )


# --- 11. Order independence --------------------------------------------------------


def test_build_result_is_independent_of_signal_report_decisions_input_order():
    ordered = _signal_report((_signal("AAPL", 1), _signal("MSFT", 2), _signal("QQQ", 3)))
    reversed_report = _signal_report(tuple(reversed(ordered.decisions)))
    snapshot = _snapshot({"AAPL": 100.0, "MSFT": 100.0, "QQQ": 100.0})
    config = _config()
    engine = PortfolioEngine()
    assert engine.build(ordered, config=config, market_snapshot=snapshot) == engine.build(
        reversed_report, config=config, market_snapshot=snapshot
    )


# --- 12. Duplicate ranks break ties by symbol ---------------------------------------


def test_duplicate_ranks_break_ties_by_symbol_ascending_deterministically():
    """A malformed SignalReport (SignalEngine never produces duplicate ranks
    among "approved" decisions coming from a real RiskReport) must still
    yield a deterministic, order-independent result: ties on rank are broken
    by symbol ascending, never by input order."""
    forward = _signal_report((_signal("MSFT", 1), _signal("AAPL", 1), _signal("QQQ", 2)))
    reversed_input = _signal_report(tuple(reversed(forward.decisions)))
    snapshot = _snapshot({"MSFT": 100.0, "AAPL": 100.0, "QQQ": 100.0})
    config = _config()
    engine = PortfolioEngine()

    result_forward = engine.build(forward, config=config, market_snapshot=snapshot)
    result_reversed = engine.build(reversed_input, config=config, market_snapshot=snapshot)

    assert engine.build(forward, config=config, market_snapshot=snapshot) == result_forward
    assert result_forward == result_reversed
    assert [position.symbol for position in result_forward.target_positions] == ["AAPL", "MSFT", "QQQ"]


# --- 13. Traceability -------------------------------------------------------------


def test_target_position_preserves_exact_reference_to_the_original_signal_decision():
    decision = _signal("AAPL", 1)
    report = _signal_report((decision,))
    result = PortfolioEngine().build(report, config=_config(), market_snapshot=_snapshot({"AAPL": 100.0}))
    assert result.target_positions[0].source_signal is decision
    assert result.target_positions[0].source_signal.source_decision is decision.source_decision


def test_portfolio_exclusion_preserves_exact_reference_to_the_original_signal_decision():
    decision = _signal("AAPL", 1)
    report = _signal_report((decision,))
    result = PortfolioEngine().build(report, config=_config(), market_snapshot=_snapshot({}))
    assert result.excluded[0].source_signal is decision


# --- 14. Propagation of the 6 config ids and evaluation_date ------------------------


def test_portfolio_report_carries_all_six_upstream_and_own_config_ids():
    report = _signal_report(
        (_signal("AAPL", 1),),
        source_scanner_config_id="a-specific-scanner-id",
        source_scorer_config_id="a-specific-scorer-id",
        source_strategy_config_id="a-specific-strategy-id",
        source_risk_config_id="a-specific-risk-id",
        signal_config_id="a-specific-signal-id",
    )
    config = _config()
    result = PortfolioEngine().build(report, config=config, market_snapshot=_snapshot({"AAPL": 100.0}))
    assert result.source_scanner_config_id == "a-specific-scanner-id"
    assert result.source_scorer_config_id == "a-specific-scorer-id"
    assert result.source_strategy_config_id == "a-specific-strategy-id"
    assert result.source_risk_config_id == "a-specific-risk-id"
    assert result.source_signal_config_id == "a-specific-signal-id"
    assert result.portfolio_config_id == config.config_id


def test_portfolio_report_carries_evaluation_date():
    report = _signal_report((_signal("AAPL", 1),))
    result = PortfolioEngine().build(report, config=_config(), market_snapshot=_snapshot({"AAPL": 100.0}))
    assert result.evaluation_date == EVALUATION_DATE


# --- 15. Empty input ---------------------------------------------------------------


def test_empty_signal_report_produces_no_positions_no_exclusions_and_full_unallocated_cash():
    report = _signal_report(())
    result = PortfolioEngine().build(report, config=_config(1000.0), market_snapshot=_snapshot({}))
    assert result.target_positions == ()
    assert result.excluded == ()
    assert result.unallocated_cash == 1000.0


# --- 16. PortfolioConfig validation --------------------------------------------------


def test_initial_capital_must_be_positive():
    with pytest.raises(PortfolioConfigError):
        PortfolioConfig.create(initial_capital=0, allocation_policy="equal_weight")
    with pytest.raises(PortfolioConfigError):
        PortfolioConfig.create(initial_capital=-1, allocation_policy="equal_weight")


def test_initial_capital_must_be_numeric_not_bool():
    with pytest.raises(PortfolioConfigError):
        PortfolioConfig.create(initial_capital=True, allocation_policy="equal_weight")  # type: ignore[arg-type]


def test_allocation_policy_must_be_a_supported_value():
    with pytest.raises(PortfolioConfigError):
        PortfolioConfig.create(initial_capital=1000.0, allocation_policy="market_cap_weight")


# --- 17. config_id determinism and sensitivity ---------------------------------------


def test_config_id_is_deterministic_for_identical_configs():
    first = PortfolioConfig.create(initial_capital=1000.0, allocation_policy="equal_weight", version="1")
    second = PortfolioConfig.create(initial_capital=1000.0, allocation_policy="equal_weight", version="1")
    assert first.config_id == second.config_id


def test_config_id_changes_when_initial_capital_changes():
    base = PortfolioConfig.create(initial_capital=1000.0, allocation_policy="equal_weight")
    changed = PortfolioConfig.create(initial_capital=2000.0, allocation_policy="equal_weight")
    assert base.config_id != changed.config_id


def test_config_id_changes_when_version_changes():
    base = PortfolioConfig.create(initial_capital=1000.0, allocation_policy="equal_weight", version="1")
    changed = PortfolioConfig.create(initial_capital=1000.0, allocation_policy="equal_weight", version="2")
    assert base.config_id != changed.config_id


# --- 18. Immutability ---------------------------------------------------------------


def test_input_signal_report_and_decisions_are_not_mutated():
    decision = _signal("AAPL", 1)
    report = _signal_report((decision,))
    report_before = report
    decision_before = decision

    PortfolioEngine().build(report, config=_config(), market_snapshot=_snapshot({"AAPL": 100.0}))

    assert report == report_before
    assert decision == decision_before
    assert report.decisions[0] is decision


def test_build_performs_no_side_effects_beyond_returning_a_value():
    report = _signal_report((_signal("AAPL", 1),))
    config = _config()
    snapshot = _snapshot({"AAPL": 100.0})
    engine = PortfolioEngine()
    result = engine.build(report, config=config, market_snapshot=snapshot)
    result_again = engine.build(report, config=config, market_snapshot=snapshot)
    assert result == result_again


# --- 19. Structural independence: forbidden imports (AST-adjacent grep) --------------


def test_portfolio_package_never_imports_forbidden_dependencies():
    import trading_agent.portfolio.config as config_module
    import trading_agent.portfolio.engine as engine_module
    import trading_agent.portfolio.exceptions as exceptions_module
    import trading_agent.portfolio.models as models_module

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
        "from trading_agent.signals.config",
        "import trading_agent.signals.config",
        "from trading_agent.signals.engine",
        "import trading_agent.signals.engine",
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


def test_portfolio_config_has_no_coupling_to_other_layer_configs():
    import trading_agent.portfolio.config as config_module

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "from trading_agent.signals" not in source
    assert "from trading_agent.risk" not in source
    assert "from trading_agent.strategies" not in source
    assert "from trading_agent.scorer" not in source
    assert "from trading_agent.scanner" not in source


def test_portfolio_engine_never_reads_source_result_metrics_for_price():
    import trading_agent.portfolio.engine as engine_module

    source = Path(engine_module.__file__).read_text(encoding="utf-8")
    assert "source_result" not in source
    assert ".metrics" not in source
    assert "metrics[" not in source


# --- 20. No Execution/Risk/Market-Data responsibilities -----------------------------


def test_target_position_has_no_order_execution_or_sizing_beyond_quantity_and_value():
    fields = set(TargetPosition.__dataclass_fields__)
    assert fields == {"symbol", "quantity", "target_value", "source_signal"}


def test_portfolio_exclusion_has_no_extra_fields():
    fields = set(PortfolioExclusion.__dataclass_fields__)
    assert fields == {"symbol", "reason_code", "source_signal"}


def test_market_snapshot_has_no_extra_fields():
    fields = set(MarketSnapshot.__dataclass_fields__)
    assert fields == {"as_of_date", "prices"}


def test_portfolio_report_has_no_extra_fields():
    fields = set(PortfolioReport.__dataclass_fields__)
    assert fields == {
        "source_scanner_config_id",
        "source_scorer_config_id",
        "source_strategy_config_id",
        "source_risk_config_id",
        "source_signal_config_id",
        "portfolio_config_id",
        "evaluation_date",
        "target_positions",
        "excluded",
        "unallocated_cash",
    }


# --- Full-value equality ------------------------------------------------------------


def test_full_portfolio_report_equality_between_independent_constructions():
    decision = _signal("AAPL", 1)
    report = _signal_report(
        (decision,),
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        source_strategy_config_id="cfg-c",
        source_risk_config_id="cfg-d",
        signal_config_id="cfg-e",
    )
    config = _config(1000.0)
    snapshot = _snapshot({"AAPL": 100.0})
    engine = PortfolioEngine()

    first = engine.build(report, config=config, market_snapshot=snapshot)
    second = engine.build(report, config=config, market_snapshot=snapshot)

    assert first == second
    assert first == PortfolioReport(
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        source_strategy_config_id="cfg-c",
        source_risk_config_id="cfg-d",
        source_signal_config_id="cfg-e",
        portfolio_config_id=config.config_id,
        evaluation_date=EVALUATION_DATE,
        target_positions=(
            TargetPosition(
                symbol="AAPL",
                quantity=10,
                target_value=1000.0,
                source_signal=decision,
            ),
        ),
        excluded=(),
        unallocated_cash=0.0,
    )
