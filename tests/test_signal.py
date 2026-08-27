"""Tests for the Signal layer.

These tests construct ``RiskReport`` / ``RiskDecision`` / ``StrategyDecision``
/ ``ScoredCandidate`` / ``SymbolScanResult`` values directly rather than
running the real ``MarketScanner``/``OpportunityScorer``/``StrategyEngine``/
``RiskEngine`` pipeline — Signal's contract is with the *shape* of a
``RiskReport``, not with how one was produced, and the signals package must
never import pandas, ``DataEngine``, ``IndicatorsEngine``, ``MarketScanner``,
``ScannerConfig``, ``ScorerConfig``, ``StrategyConfig``, or ``RiskConfig`` to
build one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_agent.risk.models import RiskDecision, RiskReport
from trading_agent.scanner.models import SymbolScanResult
from trading_agent.scorer.models import ScoredCandidate
from trading_agent.signals import (
    SignalConfig,
    SignalConfigError,
    SignalDecision,
    SignalEngine,
    SignalReport,
)
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


def _approved(symbol: str, rank: int) -> RiskDecision:
    return RiskDecision(
        symbol=symbol,
        status="approved",
        reason_code="within_risk_limit",
        source_decision=_strategy_decision(symbol, rank),
    )


def _rejected(symbol: str, rank: int) -> RiskDecision:
    return RiskDecision(
        symbol=symbol,
        status="rejected_by_risk",
        reason_code="risk_limit_exceeded",
        source_decision=_strategy_decision(symbol, rank),
    )


def _risk_report(
    decisions: tuple[RiskDecision, ...],
    *,
    source_scanner_config_id: str = "scanner-cfg-abc",
    source_scorer_config_id: str = "scorer-cfg-xyz",
    source_strategy_config_id: str = "strategy-cfg-123",
    risk_config_id: str = "risk-cfg-789",
) -> RiskReport:
    return RiskReport(
        source_scanner_config_id=source_scanner_config_id,
        source_scorer_config_id=source_scorer_config_id,
        source_strategy_config_id=source_strategy_config_id,
        risk_config_id=risk_config_id,
        evaluation_date=EVALUATION_DATE,
        decisions=decisions,
    )


def _config(trade_direction: str = "long") -> SignalConfig:
    return SignalConfig.create(trade_direction=trade_direction)


# --- 1. Accepts a valid RiskReport / basic behavior --------------------------


def test_signal_accepts_a_valid_risk_report_and_returns_a_signal_report():
    report = _risk_report((_approved("AAPL", 1),))
    result = SignalEngine().generate(report, config=_config())
    assert isinstance(result, SignalReport)
    assert len(result.decisions) == 1


def test_approved_decisions_produce_a_signal_with_the_configured_trade_direction():
    report = _risk_report((_approved("AAPL", 1),))
    result = SignalEngine().generate(report, config=_config(trade_direction="long"))
    assert result.decisions[0].trade_direction == "long"
    assert result.decisions[0].reason_code == "approved_by_risk"


# --- 2. Determinism -----------------------------------------------------------


def test_generate_is_deterministic_across_repeated_calls():
    report = _risk_report((_approved("AAPL", 1), _approved("MSFT", 2)))
    config = _config()
    engine = SignalEngine()
    assert engine.generate(report, config=config) == engine.generate(report, config=config)


# --- 3. Order independence -----------------------------------------------------


def test_generate_result_is_independent_of_report_decisions_input_order():
    ordered = _risk_report((_approved("AAPL", 1), _approved("MSFT", 2), _approved("QQQ", 3)))
    reversed_report = _risk_report(tuple(reversed(ordered.decisions)))
    config = _config()
    engine = SignalEngine()
    assert engine.generate(ordered, config=config) == engine.generate(reversed_report, config=config)


def test_decisions_are_ordered_by_rank_ascending():
    shuffled = _risk_report((_approved("QQQ", 3), _approved("AAPL", 1), _approved("MSFT", 2)))
    result = SignalEngine().generate(shuffled, config=_config())
    ranks = [decision.source_decision.source_decision.source_candidate.rank for decision in result.decisions]
    assert ranks == [1, 2, 3]


def test_duplicate_ranks_break_ties_by_symbol_ascending_deterministically():
    """A malformed RiskReport (RiskEngine never produces duplicate ranks
    among "approved" decisions coming from a real StrategyReport) must still
    yield a deterministic, order-independent result: ties on rank are broken
    by symbol ascending, never by input order."""
    forward = _risk_report((_approved("MSFT", 1), _approved("AAPL", 1), _approved("QQQ", 2)))
    reversed_input = _risk_report(tuple(reversed(forward.decisions)))
    config = _config()
    engine = SignalEngine()

    result_forward = engine.generate(forward, config=config)
    result_reversed = engine.generate(reversed_input, config=config)

    assert engine.generate(forward, config=config) == result_forward
    assert result_forward == result_reversed
    assert [decision.symbol for decision in result_forward.decisions] == ["AAPL", "MSFT", "QQQ"]


# --- 4. Traceability -------------------------------------------------------------


def test_signal_decision_preserves_exact_reference_to_the_original_risk_decision():
    decision = _approved("AAPL", 1)
    report = _risk_report((decision,))
    result = SignalEngine().generate(report, config=_config())
    assert result.decisions[0].source_decision is decision
    assert result.decisions[0].source_decision.source_decision is decision.source_decision


def test_signal_report_carries_source_scanner_scorer_strategy_and_risk_config_ids():
    report = _risk_report(
        (_approved("AAPL", 1),),
        source_scanner_config_id="a-specific-scanner-id",
        source_scorer_config_id="a-specific-scorer-id",
        source_strategy_config_id="a-specific-strategy-id",
        risk_config_id="a-specific-risk-id",
    )
    result = SignalEngine().generate(report, config=_config())
    assert result.source_scanner_config_id == "a-specific-scanner-id"
    assert result.source_scorer_config_id == "a-specific-scorer-id"
    assert result.source_strategy_config_id == "a-specific-strategy-id"
    assert result.source_risk_config_id == "a-specific-risk-id"


def test_signal_report_carries_signal_config_id_and_evaluation_date():
    config = _config()
    report = _risk_report((_approved("AAPL", 1),))
    result = SignalEngine().generate(report, config=config)
    assert result.signal_config_id == config.config_id
    assert result.evaluation_date == EVALUATION_DATE


# --- 5-9. Structural independence -------------------------------------------------


def test_signals_package_never_imports_forbidden_dependencies():
    import trading_agent.signals.config as config_module
    import trading_agent.signals.engine as engine_module
    import trading_agent.signals.exceptions as exceptions_module
    import trading_agent.signals.models as models_module

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
        "from trading_agent.strategies",
        "import trading_agent.strategies",
        "from trading_agent.risk.config",
        "import trading_agent.risk.config",
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


def test_signal_config_has_no_coupling_to_risk_strategy_scorer_or_scanner_config():
    import trading_agent.signals.config as config_module

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "from trading_agent.risk" not in source
    assert "from trading_agent.strategies" not in source
    assert "from trading_agent.scorer" not in source
    assert "from trading_agent.scanner" not in source


# --- 10-11. No sizing, capital, exposure, orders, or execution ------------------


def test_signal_decision_has_no_sizing_capital_exposure_or_order_fields():
    decision_fields = set(SignalDecision.__dataclass_fields__)
    assert decision_fields == {"symbol", "trade_direction", "reason_code", "source_decision"}


def test_generate_performs_no_side_effects_beyond_returning_a_value():
    report = _risk_report((_approved("AAPL", 1),))
    config = _config()
    engine = SignalEngine()
    result = engine.generate(report, config=config)
    result_again = engine.generate(report, config=config)
    assert result == result_again


# --- 12. Input immutability -------------------------------------------------------


def test_input_risk_report_and_decisions_are_not_mutated():
    decision = _approved("AAPL", 1)
    report = _risk_report((decision,))
    report_before = report
    decision_before = decision

    SignalEngine().generate(report, config=_config())

    assert report == report_before
    assert decision == decision_before
    assert report.decisions[0] is decision


# --- 13. Empty RiskReport ------------------------------------------------------


def test_empty_risk_report_produces_no_signal_decisions():
    report = _risk_report(())
    result = SignalEngine().generate(report, config=_config())
    assert result.decisions == ()


# --- 14. rejected_by_risk decisions never reach SignalReport --------------------


def test_rejected_by_risk_decisions_never_produce_a_signal_decision():
    report = _risk_report((_approved("AAPL", 1), _rejected("MSFT", 2), _rejected("QQQ", 3)))
    result = SignalEngine().generate(report, config=_config())
    assert len(result.decisions) == 1
    assert result.decisions[0].symbol == "AAPL"
    assert {decision.symbol for decision in result.decisions}.isdisjoint({"MSFT", "QQQ"})


def test_risk_report_with_only_rejected_decisions_produces_no_signal_decisions():
    report = _risk_report((_rejected("AAPL", 1), _rejected("MSFT", 2)))
    result = SignalEngine().generate(report, config=_config())
    assert result.decisions == ()


def test_rejected_decisions_do_not_appear_between_approved_ones_after_ordering():
    # A rejected entry ranked ahead of an approved entry must not appear in
    # the output at all, and must not affect the ordering of the approved one.
    report = _risk_report((_rejected("MSFT", 1), _approved("AAPL", 2)))
    result = SignalEngine().generate(report, config=_config())
    assert len(result.decisions) == 1
    assert result.decisions[0].symbol == "AAPL"


# --- SignalConfig validation ---------------------------------------------------------


def test_trade_direction_must_be_a_supported_value():
    with pytest.raises(SignalConfigError):
        SignalConfig.create(trade_direction="short")
    with pytest.raises(SignalConfigError):
        SignalConfig.create(trade_direction="")
    with pytest.raises(SignalConfigError):
        SignalConfig.create(trade_direction="buy")


def test_trade_direction_long_is_accepted():
    config = SignalConfig.create(trade_direction="long")
    assert config.trade_direction == "long"


# --- config_id determinism and sensitivity ------------------------------------------


def test_config_id_is_deterministic_for_identical_configs():
    first = SignalConfig.create(trade_direction="long", version="1")
    second = SignalConfig.create(trade_direction="long", version="1")
    assert first.config_id == second.config_id


def test_config_id_changes_when_version_changes():
    base = SignalConfig.create(trade_direction="long", version="1")
    changed = SignalConfig.create(trade_direction="long", version="2")
    assert base.config_id != changed.config_id


def test_config_id_is_independent_of_risk_strategy_scorer_and_scanner_config_ids():
    config = SignalConfig.create(trade_direction="long")
    assert config.config_id not in {"risk-cfg-789", "strategy-cfg-123", "scorer-cfg-xyz", "scanner-cfg-abc"}


# --- Full-value equality ------------------------------------------------------------


def test_full_signal_report_equality_between_independent_constructions():
    decision = _approved("AAPL", 1)
    report = _risk_report(
        (decision,),
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        source_strategy_config_id="cfg-c",
        risk_config_id="cfg-d",
    )
    config = _config()
    engine = SignalEngine()

    first = engine.generate(report, config=config)
    second = engine.generate(report, config=config)

    assert first == second
    assert first == SignalReport(
        source_scanner_config_id="cfg-a",
        source_scorer_config_id="cfg-b",
        source_strategy_config_id="cfg-c",
        source_risk_config_id="cfg-d",
        signal_config_id=config.config_id,
        evaluation_date=EVALUATION_DATE,
        decisions=(
            SignalDecision(
                symbol="AAPL",
                trade_direction="long",
                reason_code="approved_by_risk",
                source_decision=decision,
            ),
        ),
    )
