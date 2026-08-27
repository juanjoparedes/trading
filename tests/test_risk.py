"""Tests for the Risk layer.

These tests construct ``StrategyReport`` / ``StrategyDecision`` /
``ScoredCandidate`` / ``SymbolScanResult`` values directly rather than
running the real ``MarketScanner``/``OpportunityScorer``/``StrategyEngine``
pipeline — Risk's contract is with the *shape* of a ``StrategyReport``, not
with how one was produced, and the risk package must never import pandas,
``DataEngine``, ``IndicatorsEngine``, ``MarketScanner``, ``ScannerConfig``,
``ScorerConfig``, or ``StrategyConfig`` to build one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_agent.risk import RiskConfig, RiskConfigError, RiskDecision, RiskEngine, RiskReport
from trading_agent.scanner.models import SymbolScanResult
from trading_agent.scorer.models import ScoredCandidate
from trading_agent.strategies.models import StrategyDecision, StrategyReport

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


def _enter(symbol: str, rank: int) -> StrategyDecision:
    return StrategyDecision(
        symbol=symbol,
        action="enter",
        reason_code="selected_top_ranked",
        source_candidate=_candidate(symbol, rank),
    )


def _no_action(symbol: str, rank: int) -> StrategyDecision:
    return StrategyDecision(
        symbol=symbol,
        action="no_action",
        reason_code="rank_exceeds_max_candidates",
        source_candidate=_candidate(symbol, rank),
    )


def _strategy_report(
    decisions: tuple[StrategyDecision, ...],
    *,
    source_scanner_config_id: str = "scanner-cfg-abc",
    source_scorer_config_id: str = "scorer-cfg-xyz",
    strategy_config_id: str = "strategy-cfg-123",
) -> StrategyReport:
    return StrategyReport(
        source_scanner_config_id=source_scanner_config_id,
        source_scorer_config_id=source_scorer_config_id,
        strategy_config_id=strategy_config_id,
        evaluation_date=EVALUATION_DATE,
        decisions=decisions,
    )


def _config(max_approved_decisions: int = 2) -> RiskConfig:
    return RiskConfig.create(max_approved_decisions=max_approved_decisions)


# --- 1. Accepts a valid StrategyReport / basic behavior ----------------------


def test_risk_accepts_a_valid_strategy_report_and_returns_a_risk_report():
    report = _strategy_report((_enter("AAPL", 1),))
    result = RiskEngine().evaluate(report, config=_config())
    assert isinstance(result, RiskReport)
    assert len(result.decisions) == 1


def test_decisions_within_the_limit_are_approved_and_beyond_it_are_rejected():
    report = _strategy_report((_enter("AAPL", 1), _enter("MSFT", 2), _enter("QQQ", 3)))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=2))
    by_symbol = {decision.symbol: decision for decision in result.decisions}
    assert by_symbol["AAPL"].status == "approved"
    assert by_symbol["AAPL"].reason_code == "within_risk_limit"
    assert by_symbol["MSFT"].status == "approved"
    assert by_symbol["QQQ"].status == "rejected_by_risk"
    assert by_symbol["QQQ"].reason_code == "risk_limit_exceeded"


def test_exactly_at_the_limit_is_approved():
    report = _strategy_report((_enter("AAPL", 1), _enter("MSFT", 2)))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=2))
    assert all(decision.status == "approved" for decision in result.decisions)


def test_one_past_the_limit_is_rejected():
    report = _strategy_report((_enter("AAPL", 1), _enter("MSFT", 2), _enter("QQQ", 3)))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=2))
    by_symbol = {decision.symbol: decision for decision in result.decisions}
    assert by_symbol["QQQ"].status == "rejected_by_risk"


# --- 2. Determinism ------------------------------------------------------------


def test_evaluate_is_deterministic_across_repeated_calls():
    report = _strategy_report((_enter("AAPL", 1), _enter("MSFT", 2)))
    config = _config()
    engine = RiskEngine()
    assert engine.evaluate(report, config=config) == engine.evaluate(report, config=config)


# --- 3. Order independence ------------------------------------------------------


def test_evaluate_result_is_independent_of_report_decisions_input_order():
    ordered = _strategy_report((_enter("AAPL", 1), _enter("MSFT", 2), _enter("QQQ", 3)))
    reversed_report = _strategy_report(tuple(reversed(ordered.decisions)))
    config = _config(max_approved_decisions=2)
    engine = RiskEngine()
    assert engine.evaluate(ordered, config=config) == engine.evaluate(reversed_report, config=config)


def test_decisions_are_ordered_by_rank_ascending_before_the_limit_is_applied():
    shuffled = _strategy_report((_enter("QQQ", 3), _enter("AAPL", 1), _enter("MSFT", 2)))
    result = RiskEngine().evaluate(shuffled, config=_config(max_approved_decisions=3))
    assert [decision.source_decision.source_candidate.rank for decision in result.decisions] == [1, 2, 3]


def test_duplicate_ranks_break_ties_by_symbol_ascending_deterministically():
    """A malformed StrategyReport (StrategyEngine never produces duplicate
    ranks among "enter" decisions coming from a real ScoreReport) must still
    yield a deterministic, order-independent result: ties on rank are broken
    by symbol ascending, never by input order."""
    forward = _strategy_report((_enter("MSFT", 1), _enter("AAPL", 1), _enter("QQQ", 2)))
    reversed_input = _strategy_report(tuple(reversed(forward.decisions)))
    config = _config(max_approved_decisions=3)
    engine = RiskEngine()

    result_forward = engine.evaluate(forward, config=config)
    result_reversed = engine.evaluate(reversed_input, config=config)

    assert engine.evaluate(forward, config=config) == result_forward
    assert result_forward == result_reversed
    assert [decision.symbol for decision in result_forward.decisions] == ["AAPL", "MSFT", "QQQ"]


def test_duplicate_ranks_with_a_tight_limit_approve_by_symbol_ascending():
    report = _strategy_report((_enter("MSFT", 1), _enter("AAPL", 1)))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=1))
    by_symbol = {decision.symbol: decision for decision in result.decisions}
    assert by_symbol["AAPL"].status == "approved"
    assert by_symbol["MSFT"].status == "rejected_by_risk"


# --- 4. Traceability -------------------------------------------------------------


def test_risk_decision_preserves_exact_reference_to_the_original_strategy_decision():
    decision = _enter("AAPL", 1)
    report = _strategy_report((decision,))
    result = RiskEngine().evaluate(report, config=_config())
    assert result.decisions[0].source_decision is decision
    assert result.decisions[0].source_decision.source_candidate is decision.source_candidate


def test_risk_report_carries_source_scanner_scorer_and_strategy_config_ids():
    report = _strategy_report(
        (_enter("AAPL", 1),),
        source_scanner_config_id="a-specific-scanner-id",
        source_scorer_config_id="a-specific-scorer-id",
        strategy_config_id="a-specific-strategy-id",
    )
    result = RiskEngine().evaluate(report, config=_config())
    assert result.source_scanner_config_id == "a-specific-scanner-id"
    assert result.source_scorer_config_id == "a-specific-scorer-id"
    assert result.source_strategy_config_id == "a-specific-strategy-id"


def test_risk_report_carries_risk_config_id_and_evaluation_date():
    config = _config(max_approved_decisions=5)
    report = _strategy_report((_enter("AAPL", 1),))
    result = RiskEngine().evaluate(report, config=config)
    assert result.risk_config_id == config.config_id
    assert result.evaluation_date == EVALUATION_DATE


# --- 5-9. Structural independence -------------------------------------------------


def test_risk_package_never_imports_forbidden_dependencies():
    import trading_agent.risk.config as config_module
    import trading_agent.risk.engine as engine_module
    import trading_agent.risk.exceptions as exceptions_module
    import trading_agent.risk.models as models_module

    forbidden_import_substrings = (
        "import pandas",
        "import trading_agent.data",
        "from trading_agent.data",
        "import trading_agent.indicators",
        "from trading_agent.indicators",
        "from trading_agent.scanner",
        "import trading_agent.scanner",
        "from trading_agent.scorer",
        "import trading_agent.scorer",
        "from trading_agent.strategies.config",
        "import trading_agent.strategies.config",
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


def test_risk_config_has_no_coupling_to_strategy_scorer_or_scanner_config():
    import trading_agent.risk.config as config_module

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "from trading_agent.strategies" not in source
    assert "from trading_agent.scorer" not in source
    assert "from trading_agent.scanner" not in source


# --- 10-11. No orders, no execution, no sizing/direction ----------------------


def test_risk_decision_has_no_order_sizing_or_direction_fields():
    decision_fields = set(RiskDecision.__dataclass_fields__)
    assert decision_fields == {"symbol", "status", "reason_code", "source_decision"}


def test_evaluate_performs_no_side_effects_beyond_returning_a_value():
    report = _strategy_report((_enter("AAPL", 1),))
    config = _config()
    engine = RiskEngine()
    result = engine.evaluate(report, config=config)
    result_again = engine.evaluate(report, config=config)
    assert result == result_again


# --- 12. Input immutability -------------------------------------------------------


def test_input_strategy_report_and_decisions_are_not_mutated():
    decision = _enter("AAPL", 1)
    report = _strategy_report((decision,))
    report_before = report
    decision_before = decision

    RiskEngine().evaluate(report, config=_config())

    assert report == report_before
    assert decision == decision_before
    assert report.decisions[0] is decision


# --- 13. Empty StrategyReport ------------------------------------------------------


def test_empty_strategy_report_produces_no_risk_decisions():
    report = _strategy_report(())
    result = RiskEngine().evaluate(report, config=_config())
    assert result.decisions == ()


# --- 14. Insufficient / boundary counts ----------------------------------------


def test_fewer_enter_decisions_than_the_limit_are_all_approved():
    report = _strategy_report((_enter("AAPL", 1), _enter("MSFT", 2)))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=10))
    assert all(decision.status == "approved" for decision in result.decisions)


def test_single_enter_decision_with_limit_of_one():
    report = _strategy_report((_enter("AAPL", 1),))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=1))
    assert result.decisions[0].status == "approved"


# --- 7 (mandatory). no_action never reaches RiskReport --------------------------


def test_no_action_decisions_never_produce_a_risk_decision():
    report = _strategy_report((_enter("AAPL", 1), _no_action("MSFT", 2), _no_action("QQQ", 3)))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=10))
    assert len(result.decisions) == 1
    assert result.decisions[0].symbol == "AAPL"
    assert {decision.symbol for decision in result.decisions}.isdisjoint({"MSFT", "QQQ"})


def test_strategy_report_with_only_no_action_decisions_produces_no_risk_decisions():
    report = _strategy_report((_no_action("AAPL", 1), _no_action("MSFT", 2)))
    result = RiskEngine().evaluate(report, config=_config())
    assert result.decisions == ()


def test_no_action_decisions_do_not_consume_risk_capacity():
    # A no_action entry ranked ahead of an enter entry must not "use up" a
    # slot in the risk limit — only enter decisions are ever counted.
    report = _strategy_report((_no_action("MSFT", 1), _enter("AAPL", 2)))
    result = RiskEngine().evaluate(report, config=_config(max_approved_decisions=1))
    assert len(result.decisions) == 1
    assert result.decisions[0].symbol == "AAPL"
    assert result.decisions[0].status == "approved"


# --- RiskConfig validation ---------------------------------------------------------


def test_max_approved_decisions_must_be_positive():
    with pytest.raises(RiskConfigError):
        RiskConfig.create(max_approved_decisions=0)
    with pytest.raises(RiskConfigError):
        RiskConfig.create(max_approved_decisions=-1)


def test_max_approved_decisions_must_be_an_integer():
    with pytest.raises(RiskConfigError):
        RiskConfig.create(max_approved_decisions=1.5)  # type: ignore[arg-type]
    with pytest.raises(RiskConfigError):
        RiskConfig.create(max_approved_decisions=True)  # type: ignore[arg-type]


# --- config_id determinism and sensitivity ------------------------------------------


def test_config_id_is_deterministic_for_identical_configs():
    first = RiskConfig.create(max_approved_decisions=3, version="1")
    second = RiskConfig.create(max_approved_decisions=3, version="1")
    assert first.config_id == second.config_id


def test_config_id_changes_when_max_approved_decisions_changes():
    base = RiskConfig.create(max_approved_decisions=3)
    changed = RiskConfig.create(max_approved_decisions=4)
    assert base.config_id != changed.config_id


def test_config_id_changes_when_version_changes():
    base = RiskConfig.create(max_approved_decisions=3, version="1")
    changed = RiskConfig.create(max_approved_decisions=3, version="2")
    assert base.config_id != changed.config_id


# --- Full-value equality ------------------------------------------------------------


def test_full_risk_report_equality_between_independent_constructions():
    decision = _enter("AAPL", 1)
    report = _strategy_report(
        (decision,),
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        strategy_config_id="cfg-c",
    )
    config = _config(max_approved_decisions=1)
    engine = RiskEngine()

    first = engine.evaluate(report, config=config)
    second = engine.evaluate(report, config=config)

    assert first == second
    assert first == RiskReport(
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        source_strategy_config_id="cfg-c",
        risk_config_id=config.config_id,
        evaluation_date=EVALUATION_DATE,
        decisions=(
            RiskDecision(
                symbol="AAPL",
                status="approved",
                reason_code="within_risk_limit",
                source_decision=decision,
            ),
        ),
    )
