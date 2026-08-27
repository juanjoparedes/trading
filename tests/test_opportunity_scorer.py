"""Tests for the Opportunity Scorer.

These tests construct ``ScanReport`` / ``SymbolScanResult`` values directly
rather than running the real ``MarketScanner`` — the scorer's contract is
with the *shape* of a ``ScanReport``, not with how one was produced, and the
scorer package must never import pandas, ``DataEngine``, or
``IndicatorsEngine`` to build one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_agent.scanner.models import (
    DatasetMetadata,
    ScanReport,
    SymbolScanResult,
)
from trading_agent.scorer import (
    ExcludedCandidate,
    OpportunityScorer,
    ScoreReport,
    ScoredCandidate,
    ScorerConfig,
    ScorerConfigError,
)

EVALUATION_DATE = date(2024, 6, 28)


def _result(
    symbol: str,
    *,
    status: str = "candidate",
    metrics: dict[str, float | None] | None = None,
    as_of_date: date | None = EVALUATION_DATE,
) -> SymbolScanResult:
    return SymbolScanResult(
        symbol=symbol,
        requested_evaluation_date=EVALUATION_DATE,
        as_of_date=as_of_date,
        status=status,
        passed_filters=(),
        failed_filters=(),
        soft_conditions=(),
        metrics=metrics if metrics is not None else {},
        required_indicators=(),
        data_quality_notes=(),
    )


def _report(
    results: tuple[SymbolScanResult, ...],
    *,
    config_id: str = "scanner-config-abc",
    scanner_version: str = "1",
) -> ScanReport:
    metadata = DatasetMetadata(
        symbols_requested=len(results),
        symbols_evaluated=len(results),
        evaluation_date=EVALUATION_DATE,
        scanner_version=scanner_version,
        config_id=config_id,
    )
    return ScanReport(
        requested_evaluation_date=EVALUATION_DATE,
        config_id=config_id,
        dataset_metadata=metadata,
        results=results,
    )


def _config(metric: str = "rsi_14", direction: str = "desc") -> ScorerConfig:
    return ScorerConfig.create(metric=metric, direction=direction)


# --- Basic ranking behavior -------------------------------------------------


def test_zero_candidates_produce_empty_results_and_empty_excluded():
    report = _report((_result("AAPL", status="rejected"), _result("MSFT", status="insufficient_data")))
    score_report = OpportunityScorer().score(report, config=_config())
    assert score_report.results == ()
    assert score_report.excluded == ()


def test_single_candidate_is_ranked_first():
    report = _report((_result("AAPL", metrics={"rsi_14": 55.0}),))
    score_report = OpportunityScorer().score(report, config=_config())
    assert len(score_report.results) == 1
    assert score_report.results[0].symbol == "AAPL"
    assert score_report.results[0].rank == 1
    assert score_report.results[0].observed_value == 55.0
    assert score_report.results[0].metric_used == "rsi_14"


def test_multiple_candidates_are_ranked_by_metric_descending():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": 40.0}),
            _result("MSFT", metrics={"rsi_14": 80.0}),
            _result("QQQ", metrics={"rsi_14": 60.0}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config(direction="desc"))
    ranked_symbols = [candidate.symbol for candidate in score_report.results]
    assert ranked_symbols == ["MSFT", "QQQ", "AAPL"]
    assert [candidate.rank for candidate in score_report.results] == [1, 2, 3]


def test_ascending_direction_ranks_lowest_metric_first():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": 40.0}),
            _result("MSFT", metrics={"rsi_14": 80.0}),
            _result("QQQ", metrics={"rsi_14": 60.0}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config(direction="asc"))
    ranked_symbols = [candidate.symbol for candidate in score_report.results]
    assert ranked_symbols == ["AAPL", "QQQ", "MSFT"]


def test_descending_direction_ranks_highest_metric_first():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": 40.0}),
            _result("MSFT", metrics={"rsi_14": 80.0}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config(direction="desc"))
    assert score_report.results[0].symbol == "MSFT"
    assert score_report.results[1].symbol == "AAPL"


def test_exact_tie_breaks_by_symbol_ascending_regardless_of_direction():
    report = _report(
        (
            _result("MSFT", metrics={"rsi_14": 50.0}),
            _result("AAPL", metrics={"rsi_14": 50.0}),
        )
    )
    desc_report = OpportunityScorer().score(report, config=_config(direction="desc"))
    asc_report = OpportunityScorer().score(report, config=_config(direction="asc"))
    assert [c.symbol for c in desc_report.results] == ["AAPL", "MSFT"]
    assert [c.symbol for c in asc_report.results] == ["AAPL", "MSFT"]


def test_ranks_are_sequential_without_gaps_or_repeats_even_with_ties():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": 50.0}),
            _result("MSFT", metrics={"rsi_14": 50.0}),
            _result("QQQ", metrics={"rsi_14": 50.0}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config())
    assert [c.rank for c in score_report.results] == [1, 2, 3]


# --- Non-candidate statuses are neither ranked nor excluded -----------------


def test_rejected_symbols_are_excluded_from_both_results_and_excluded_list():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": 50.0}),
            _result("MSFT", status="rejected", metrics={"rsi_14": 90.0}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config())
    assert [c.symbol for c in score_report.results] == ["AAPL"]
    assert score_report.excluded == ()


def test_insufficient_data_symbols_are_excluded_from_both_results_and_excluded_list():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": 50.0}),
            _result("MSFT", status="insufficient_data", metrics={}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config())
    assert [c.symbol for c in score_report.results] == ["AAPL"]
    assert score_report.excluded == ()


# --- Metric absence / None / NaN --------------------------------------------


def test_candidate_missing_the_metric_key_is_excluded_with_metric_missing():
    report = _report((_result("AAPL", metrics={"sma_20": 100.0}),))
    score_report = OpportunityScorer().score(report, config=_config(metric="rsi_14"))
    assert score_report.results == ()
    assert len(score_report.excluded) == 1
    assert score_report.excluded[0].symbol == "AAPL"
    assert score_report.excluded[0].reason_code == "metric_missing"


def test_candidate_with_none_metric_value_is_excluded_with_metric_none():
    report = _report((_result("AAPL", metrics={"rsi_14": None}),))
    score_report = OpportunityScorer().score(report, config=_config(metric="rsi_14"))
    assert score_report.results == ()
    assert score_report.excluded[0].reason_code == "metric_none"


def test_candidate_with_nan_metric_value_is_excluded_with_metric_nan():
    report = _report((_result("AAPL", metrics={"rsi_14": float("nan")}),))
    score_report = OpportunityScorer().score(report, config=_config(metric="rsi_14"))
    assert score_report.results == ()
    assert score_report.excluded[0].reason_code == "metric_nan"


def test_excluded_symbols_are_never_imputed_a_zero_value():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": None}),
            _result("MSFT", metrics={"rsi_14": 0.0}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config(metric="rsi_14"))
    assert [c.symbol for c in score_report.results] == ["MSFT"]
    assert score_report.results[0].observed_value == 0.0
    assert [e.symbol for e in score_report.excluded] == ["AAPL"]


def test_excluded_list_is_sorted_by_symbol_regardless_of_input_order():
    report = _report(
        (
            _result("QQQ", metrics={"rsi_14": None}),
            _result("AAPL", metrics={"rsi_14": None}),
            _result("MSFT", metrics={"rsi_14": None}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config())
    assert [e.symbol for e in score_report.excluded] == ["AAPL", "MSFT", "QQQ"]


# --- Determinism and order-independence -------------------------------------


def test_scoring_is_deterministic_across_repeated_calls():
    report = _report(
        (
            _result("AAPL", metrics={"rsi_14": 40.0}),
            _result("MSFT", metrics={"rsi_14": 80.0}),
        )
    )
    config = _config()
    scorer = OpportunityScorer()
    assert scorer.score(report, config=config) == scorer.score(report, config=config)


def test_scoring_result_is_independent_of_report_results_input_order():
    ordered = _report(
        (
            _result("AAPL", metrics={"rsi_14": 40.0}),
            _result("MSFT", metrics={"rsi_14": 80.0}),
            _result("QQQ", metrics={"rsi_14": 60.0}),
        )
    )
    reversed_report = _report(tuple(reversed(ordered.results)))
    config = _config()
    scorer = OpportunityScorer()
    assert scorer.score(ordered, config=config) == scorer.score(reversed_report, config=config)


def test_source_result_is_preserved_unmutated_on_scored_candidates():
    original = _result("AAPL", metrics={"rsi_14": 55.0})
    report = _report((original,))
    score_report = OpportunityScorer().score(report, config=_config())
    assert score_report.results[0].source_result == original


def test_source_result_is_preserved_unmutated_on_excluded_candidates():
    original = _result("AAPL", metrics={"rsi_14": None})
    report = _report((original,))
    score_report = OpportunityScorer().score(report, config=_config())
    assert score_report.excluded[0].source_result == original


# --- Traceability fields -----------------------------------------------------


def test_source_scanner_config_id_is_copied_from_the_report():
    report = _report((_result("AAPL", metrics={"rsi_14": 1.0}),), config_id="a-specific-scanner-id")
    score_report = OpportunityScorer().score(report, config=_config())
    assert score_report.source_scanner_config_id == "a-specific-scanner-id"


def test_source_scanner_version_is_copied_from_dataset_metadata():
    report = _report((_result("AAPL", metrics={"rsi_14": 1.0}),), scanner_version="7")
    score_report = OpportunityScorer().score(report, config=_config())
    assert score_report.source_scanner_version == "7"


def test_scorer_config_id_matches_the_config_used():
    config = _config(metric="rsi_14", direction="asc")
    report = _report((_result("AAPL", metrics={"rsi_14": 1.0}),))
    score_report = OpportunityScorer().score(report, config=config)
    assert score_report.scorer_config_id == config.config_id


def test_evaluation_date_is_copied_from_requested_evaluation_date():
    report = _report((_result("AAPL", metrics={"rsi_14": 1.0}),))
    score_report = OpportunityScorer().score(report, config=_config())
    assert score_report.evaluation_date == EVALUATION_DATE


# --- ScorerConfig validation --------------------------------------------------


def test_invalid_direction_is_rejected_at_construction():
    with pytest.raises(ScorerConfigError):
        ScorerConfig.create(metric="rsi_14", direction="ascending")  # type: ignore[arg-type]


def test_blank_metric_is_rejected_at_construction():
    with pytest.raises(ScorerConfigError):
        ScorerConfig.create(metric="   ", direction="desc")


def test_empty_metric_is_rejected_at_construction():
    with pytest.raises(ScorerConfigError):
        ScorerConfig.create(metric="", direction="desc")


# --- config_id determinism and sensitivity ------------------------------------


def test_config_id_is_deterministic_for_identical_configs():
    first = ScorerConfig.create(metric="rsi_14", direction="desc", version="1")
    second = ScorerConfig.create(metric="rsi_14", direction="desc", version="1")
    assert first.config_id == second.config_id


def test_config_id_changes_when_version_changes():
    base = ScorerConfig.create(metric="rsi_14", direction="desc", version="1")
    changed = ScorerConfig.create(metric="rsi_14", direction="desc", version="2")
    assert base.config_id != changed.config_id


def test_config_id_changes_when_metric_changes():
    base = ScorerConfig.create(metric="rsi_14", direction="desc")
    changed = ScorerConfig.create(metric="sma_20", direction="desc")
    assert base.config_id != changed.config_id


def test_config_id_changes_when_direction_changes():
    base = ScorerConfig.create(metric="rsi_14", direction="desc")
    changed = ScorerConfig.create(metric="rsi_14", direction="asc")
    assert base.config_id != changed.config_id


# --- Structural independence --------------------------------------------------


def test_scorer_package_never_imports_pandas_or_data_or_indicators_engines():
    import trading_agent.scorer.config as config_module
    import trading_agent.scorer.engine as engine_module
    import trading_agent.scorer.exceptions as exceptions_module
    import trading_agent.scorer.models as models_module

    forbidden_substrings = ("pandas", "trading_agent.data", "trading_agent.indicators", "yfinance")
    for module in (config_module, engine_module, exceptions_module, models_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in forbidden_substrings:
            assert forbidden not in source, f"{module.__name__} must not reference {forbidden!r}"


def test_scorer_config_has_no_coupling_to_scanner_config():
    import trading_agent.scorer.config as config_module

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "import trading_agent.scanner" not in source
    assert "from trading_agent.scanner" not in source


# --- Full-value equality -------------------------------------------------------


def test_full_score_report_equality_between_independent_constructions():
    report = _report((_result("AAPL", metrics={"rsi_14": 55.0}),), config_id="cfg-x", scanner_version="3")
    config = _config(metric="rsi_14", direction="desc")
    scorer = OpportunityScorer()

    first = scorer.score(report, config=config)
    second = scorer.score(report, config=config)

    assert first == second
    assert first == ScoreReport(
        source_scanner_config_id="cfg-x",
        source_scanner_version="3",
        scorer_config_id=config.config_id,
        evaluation_date=EVALUATION_DATE,
        results=(
            ScoredCandidate(
                symbol="AAPL",
                rank=1,
                metric_used="rsi_14",
                observed_value=55.0,
                source_result=_result("AAPL", metrics={"rsi_14": 55.0}),
            ),
        ),
        excluded=(),
    )


def test_mixed_report_partitions_candidates_rejected_insufficient_and_excluded_correctly():
    report = _report(
        (
            _result("AAPL", status="candidate", metrics={"rsi_14": 70.0}),
            _result("MSFT", status="rejected", metrics={"rsi_14": 99.0}),
            _result("QQQ", status="insufficient_data", metrics={}),
            _result("SPY", status="candidate", metrics={"rsi_14": None}),
            _result("TLT", status="candidate", metrics={}),
        )
    )
    score_report = OpportunityScorer().score(report, config=_config(metric="rsi_14"))
    assert [c.symbol for c in score_report.results] == ["AAPL"]
    excluded_by_symbol = {e.symbol: e.reason_code for e in score_report.excluded}
    assert excluded_by_symbol == {"SPY": "metric_none", "TLT": "metric_missing"}
