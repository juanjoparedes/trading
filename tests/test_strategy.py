"""Tests for the Strategy layer.

These tests construct ``ScoreReport`` / ``ScoredCandidate`` / ``ExcludedCandidate``
/ ``SymbolScanResult`` values directly rather than running the real
``MarketScanner``/``OpportunityScorer`` pipeline — Strategy's contract is with
the *shape* of a ``ScoreReport``, not with how one was produced, and the
strategies package must never import pandas, ``DataEngine``,
``IndicatorsEngine``, or ``MarketScanner`` to build one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import get_args

import pytest

from trading_agent.scanner.models import SymbolScanResult
from trading_agent.scorer.models import ExcludedCandidate, ScoredCandidate, ScoreReport
from trading_agent.strategies import (
    StrategyConfig,
    StrategyConfigError,
    StrategyDecision,
    StrategyEngine,
    StrategyReport,
)
from trading_agent.strategies.models import StrategyAction

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


def _excluded(symbol: str) -> ExcludedCandidate:
    return ExcludedCandidate(symbol=symbol, reason_code="metric_none", source_result=_source_result(symbol))


def _score_report(
    results: tuple[ScoredCandidate, ...] = (),
    excluded: tuple[ExcludedCandidate, ...] = (),
    *,
    source_scanner_config_id: str = "scanner-cfg-abc",
    scorer_config_id: str = "scorer-cfg-xyz",
) -> ScoreReport:
    return ScoreReport(
        source_scanner_config_id=source_scanner_config_id,
        source_scanner_version="1",
        scorer_config_id=scorer_config_id,
        evaluation_date=EVALUATION_DATE,
        results=results,
        excluded=excluded,
    )


def _config(max_candidates: int = 2) -> StrategyConfig:
    return StrategyConfig.create(max_candidates=max_candidates)


# --- 1. Accepts a valid ScoreReport / basic behavior ------------------------


def test_strategy_accepts_a_valid_score_report_and_returns_a_strategy_report():
    report = _score_report((_candidate("AAPL", 1),))
    result = StrategyEngine().decide(report, config=_config())
    assert isinstance(result, StrategyReport)
    assert len(result.decisions) == 1


def test_candidates_within_the_limit_are_entered_and_beyond_it_are_not():
    report = _score_report((_candidate("AAPL", 1), _candidate("MSFT", 2), _candidate("QQQ", 3)))
    result = StrategyEngine().decide(report, config=_config(max_candidates=2))
    by_symbol = {decision.symbol: decision for decision in result.decisions}
    assert by_symbol["AAPL"].action == "enter"
    assert by_symbol["AAPL"].reason_code == "selected_top_ranked"
    assert by_symbol["MSFT"].action == "enter"
    assert by_symbol["QQQ"].action == "no_action"
    assert by_symbol["QQQ"].reason_code == "rank_exceeds_max_candidates"


def test_rank_exactly_at_the_limit_is_entered():
    report = _score_report((_candidate("AAPL", 3),))
    result = StrategyEngine().decide(report, config=_config(max_candidates=3))
    assert result.decisions[0].action == "enter"


def test_rank_one_past_the_limit_is_no_action():
    report = _score_report((_candidate("AAPL", 4),))
    result = StrategyEngine().decide(report, config=_config(max_candidates=3))
    assert result.decisions[0].action == "no_action"


# --- 2. Determinism ----------------------------------------------------------


def test_decide_is_deterministic_across_repeated_calls():
    report = _score_report((_candidate("AAPL", 1), _candidate("MSFT", 2)))
    config = _config()
    engine = StrategyEngine()
    assert engine.decide(report, config=config) == engine.decide(report, config=config)


# --- 3. Order independence ---------------------------------------------------


def test_decide_result_is_independent_of_report_results_input_order():
    ordered = _score_report((_candidate("AAPL", 1), _candidate("MSFT", 2), _candidate("QQQ", 3)))
    reversed_report = _score_report(tuple(reversed(ordered.results)), source_scanner_config_id="scanner-cfg-abc", scorer_config_id="scorer-cfg-xyz")
    config = _config(max_candidates=2)
    engine = StrategyEngine()
    assert engine.decide(ordered, config=config) == engine.decide(reversed_report, config=config)


def test_decisions_are_always_returned_ordered_by_rank_ascending():
    shuffled = _score_report((_candidate("QQQ", 3), _candidate("AAPL", 1), _candidate("MSFT", 2)))
    result = StrategyEngine().decide(shuffled, config=_config(max_candidates=3))
    assert [decision.source_candidate.rank for decision in result.decisions] == [1, 2, 3]


def test_duplicate_ranks_break_ties_by_symbol_ascending_deterministically() -> None:
    """A malformed ScoreReport (OpportunityScorer never produces duplicate
    ranks) must still yield a deterministic, order-independent result: ties
    on rank are broken by symbol ascending, never by input order."""
    forward = _score_report((_candidate("MSFT", 1), _candidate("AAPL", 1), _candidate("QQQ", 2)))
    reversed_input = _score_report(
        tuple(reversed(forward.results)),
        source_scanner_config_id="scanner-cfg-abc",
        scorer_config_id="scorer-cfg-xyz",
    )
    config = _config(max_candidates=3)
    engine = StrategyEngine()

    result_forward = engine.decide(forward, config=config)
    result_reversed = engine.decide(reversed_input, config=config)

    # 1. Deterministic: calling decide() again on the same input yields the same result.
    assert engine.decide(forward, config=config) == result_forward
    # 2. Order independence: input order must not change the StrategyReport.
    assert result_forward == result_reversed
    # 3. Tie-break by symbol ascending among the rank-1 pair.
    assert [decision.symbol for decision in result_forward.decisions] == ["AAPL", "MSFT", "QQQ"]


# --- 4. Traceability ----------------------------------------------------------


def test_decision_preserves_exact_reference_to_the_original_scored_candidate():
    candidate = _candidate("AAPL", 1)
    report = _score_report((candidate,))
    result = StrategyEngine().decide(report, config=_config())
    # Identity, not just value equality: a future regression that replaced
    # source_candidate with an equal-valued copy must fail this test.
    assert result.decisions[0].source_candidate is candidate
    assert result.decisions[0].source_candidate.source_result is candidate.source_result


def test_strategy_report_carries_source_scanner_and_scorer_config_ids():
    report = _score_report(
        (_candidate("AAPL", 1),),
        source_scanner_config_id="a-specific-scanner-id",
        scorer_config_id="a-specific-scorer-id",
    )
    result = StrategyEngine().decide(report, config=_config())
    assert result.source_scanner_config_id == "a-specific-scanner-id"
    assert result.source_scorer_config_id == "a-specific-scorer-id"


def test_strategy_report_carries_strategy_config_id_and_evaluation_date():
    config = _config(max_candidates=5)
    report = _score_report((_candidate("AAPL", 1),))
    result = StrategyEngine().decide(report, config=config)
    assert result.strategy_config_id == config.config_id
    assert result.evaluation_date == EVALUATION_DATE


# --- 5-9. Structural independence -------------------------------------------


def test_strategy_package_never_imports_forbidden_dependencies():
    import trading_agent.strategies.config as config_module
    import trading_agent.strategies.engine as engine_module
    import trading_agent.strategies.exceptions as exceptions_module
    import trading_agent.strategies.models as models_module

    forbidden_import_substrings = (
        "import pandas",
        "import trading_agent.data",
        "from trading_agent.data",
        "import trading_agent.indicators",
        "from trading_agent.indicators",
        "from trading_agent.scanner.engine",
        "from trading_agent.scanner.config",
        "import trading_agent.scanner.engine",
        "import trading_agent.scanner.config",
        "import yfinance",
    )
    for module in (config_module, engine_module, exceptions_module, models_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in forbidden_import_substrings:
            assert forbidden not in source, f"{module.__name__} must not reference {forbidden!r}"
        # No real usage of MarketScanner/ScannerConfig as code symbols (docstring
        # prose mentioning the names for documentation purposes is fine).
        assert "MarketScanner(" not in source
        assert "ScannerConfig(" not in source


def test_strategy_config_has_no_coupling_to_scorer_or_scanner_config():
    import trading_agent.strategies.config as config_module

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "from trading_agent.scorer" not in source
    assert "from trading_agent.scanner" not in source


# --- 10-11. No orders, no execution ------------------------------------------


def test_strategy_decision_has_no_order_or_execution_fields():
    decision_fields = set(StrategyDecision.__dataclass_fields__)
    assert decision_fields == {"symbol", "action", "reason_code", "source_candidate"}
    # StrategyAction is a Literal["enter", "no_action"]; there is no order,
    # size, price, direction, or execution-status value anywhere in the model.
    assert set(get_args(StrategyAction)) == {"enter", "no_action"}


def test_decide_performs_no_side_effects_beyond_returning_a_value():
    report = _score_report((_candidate("AAPL", 1),))
    config = _config()
    result = StrategyEngine().decide(report, config=config)
    # Calling decide() twice with the same immutable inputs must be
    # side-effect-free: the second call must not observe any state left by
    # the first (no global counters, no caching that could mask a bug).
    result_again = StrategyEngine().decide(report, config=config)
    assert result == result_again


# --- 12. Input immutability ---------------------------------------------------


def test_input_score_report_and_candidates_are_not_mutated():
    candidate = _candidate("AAPL", 1)
    report = _score_report((candidate,))
    report_before = report
    candidate_before = candidate

    StrategyEngine().decide(report, config=_config())

    assert report == report_before
    assert candidate == candidate_before
    assert report.results[0] is candidate  # identity preserved, never replaced


# --- 13. Empty ScoreReport ----------------------------------------------------


def test_empty_score_report_produces_no_decisions():
    report = _score_report((), ())
    result = StrategyEngine().decide(report, config=_config())
    assert result.decisions == ()


# --- 14. Insufficient / boundary candidate counts -----------------------------


def test_fewer_candidates_than_max_candidates_are_all_entered():
    report = _score_report((_candidate("AAPL", 1), _candidate("MSFT", 2)))
    result = StrategyEngine().decide(report, config=_config(max_candidates=10))
    assert all(decision.action == "enter" for decision in result.decisions)


def test_single_candidate_report_with_limit_of_one():
    report = _score_report((_candidate("AAPL", 1),))
    result = StrategyEngine().decide(report, config=_config(max_candidates=1))
    assert result.decisions[0].action == "enter"


# --- 15. Excluded candidates present in ScoreReport ---------------------------


def test_excluded_candidates_never_produce_a_decision():
    report = _score_report((_candidate("AAPL", 1),), (_excluded("MSFT"), _excluded("QQQ")))
    result = StrategyEngine().decide(report, config=_config())
    assert len(result.decisions) == 1
    assert result.decisions[0].symbol == "AAPL"
    assert {decision.symbol for decision in result.decisions}.isdisjoint({"MSFT", "QQQ"})


def test_score_report_with_only_excluded_candidates_produces_no_decisions():
    report = _score_report((), (_excluded("AAPL"), _excluded("MSFT")))
    result = StrategyEngine().decide(report, config=_config())
    assert result.decisions == ()


# --- StrategyConfig validation -------------------------------------------------


def test_max_candidates_must_be_positive():
    with pytest.raises(StrategyConfigError):
        StrategyConfig.create(max_candidates=0)
    with pytest.raises(StrategyConfigError):
        StrategyConfig.create(max_candidates=-1)


def test_max_candidates_must_be_an_integer():
    with pytest.raises(StrategyConfigError):
        StrategyConfig.create(max_candidates=1.5)  # type: ignore[arg-type]
    with pytest.raises(StrategyConfigError):
        StrategyConfig.create(max_candidates=True)  # type: ignore[arg-type]


# --- config_id determinism and sensitivity -------------------------------------


def test_config_id_is_deterministic_for_identical_configs():
    first = StrategyConfig.create(max_candidates=3, version="1")
    second = StrategyConfig.create(max_candidates=3, version="1")
    assert first.config_id == second.config_id


def test_config_id_changes_when_max_candidates_changes():
    base = StrategyConfig.create(max_candidates=3)
    changed = StrategyConfig.create(max_candidates=4)
    assert base.config_id != changed.config_id


def test_config_id_changes_when_version_changes():
    base = StrategyConfig.create(max_candidates=3, version="1")
    changed = StrategyConfig.create(max_candidates=3, version="2")
    assert base.config_id != changed.config_id


# --- Full-value equality --------------------------------------------------------


def test_full_strategy_report_equality_between_independent_constructions():
    candidate = _candidate("AAPL", 1)
    report = _score_report((candidate,), source_scanner_config_id="cfg-x", scorer_config_id="cfg-y")
    config = _config(max_candidates=1)
    engine = StrategyEngine()

    first = engine.decide(report, config=config)
    second = engine.decide(report, config=config)

    assert first == second
    assert first == StrategyReport(
        source_scanner_config_id="cfg-x",
        source_scorer_config_id="cfg-y",
        strategy_config_id=config.config_id,
        evaluation_date=EVALUATION_DATE,
        decisions=(
            StrategyDecision(
                symbol="AAPL",
                action="enter",
                reason_code="selected_top_ranked",
                source_candidate=candidate,
            ),
        ),
    )
