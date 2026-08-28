"""End-to-end integration tests for the full trading pipeline.

These tests exercise the real production engines for every layer from
Scorer through Execution -- ``OpportunityScorer``, ``StrategyEngine``,
``RiskEngine``, ``SignalEngine``, ``PortfolioEngine``, and
``ExecutionEngine`` -- chained together exactly as a real caller would use
them:

    ScoreReport = OpportunityScorer().score(scan_report, config=...)
    StrategyReport = StrategyEngine().decide(score_report, config=...)
    RiskReport = RiskEngine().evaluate(strategy_report, config=...)
    SignalReport = SignalEngine().generate(risk_report, config=...)
    PortfolioReport = PortfolioEngine().build(signal_report, config=..., market_snapshot=...)
    ExecutionReport = ExecutionEngine().prepare(portfolio_report, config=...)

The goal is not to re-test each layer's internal logic again (that is
already covered exhaustively by that layer's own test file) -- it is to
demonstrate that the real contracts these layers were each audited against
independently actually fit together when chained for real.

Consistent with every other test file in this repository (``test_opportunity_scorer.py``,
``test_strategy.py``, ``test_risk.py``, ``test_signal.py``, ``test_portfolio.py``,
``test_execution.py``), the pipeline's starting point (``ScanReport`` /
``SymbolScanResult``) is constructed directly rather than by running the
real ``MarketScanner`` over a pandas ``DataFrame`` and ``IndicatorsEngine``:
Scanner's own internal behavior is already exhaustively covered by
``tests/test_market_scanner.py``, and this file's contract is with the
*shape* Scanner hands to Scorer, not with how a ``ScanReport`` was produced.
From ``ScoredCandidate`` (rank) through ``OrderIntent``, every value in the
main scenario is produced exclusively by its real engine -- no
``TargetPosition`` or ``OrderIntent`` is ever hand-constructed for the main
scenario.

One scenario (duplicate ranks, see below) hand-builds a ``ScoreReport``
directly instead of starting from a ``ScanReport``, for the same reason
every duplicate-rank test in ``test_strategy.py``/``test_risk.py``/
``test_signal.py``/``test_portfolio.py``/``test_execution.py`` does: a real
``OpportunityScorer`` run can never itself produce duplicate ranks (ranking
is strictly ordinal), so the only way to exercise the documented defensive
``(rank, symbol)`` tie-break that Strategy, Risk, Signal, Portfolio, and
Execution all implement is to hand-construct a malformed-but-contract-valid
``ScoreReport`` and run the five real downstream engines against it.
"""

from __future__ import annotations

from datetime import date

from trading_agent.execution import ExecutionConfig, ExecutionEngine, ExecutionReport
from trading_agent.portfolio import MarketSnapshot, PortfolioConfig, PortfolioEngine, PortfolioReport
from trading_agent.risk import RiskConfig, RiskEngine, RiskReport
from trading_agent.scanner import DatasetMetadata, ScanReport, ScannerConfig, SymbolScanResult
from trading_agent.scorer import OpportunityScorer, ScoreReport, ScoredCandidate, ScorerConfig
from trading_agent.signals import SignalConfig, SignalEngine, SignalReport
from trading_agent.strategies import StrategyConfig, StrategyEngine, StrategyReport

EVALUATION_DATE = date(2024, 6, 28)


# --- Layer configs -----------------------------------------------------------


def _scanner_config(universe: tuple[str, ...]) -> ScannerConfig:
    return ScannerConfig.create(universe=universe)


def _scorer_config() -> ScorerConfig:
    return ScorerConfig.create(metric="rsi_14", direction="desc")


def _strategy_config(max_candidates: int) -> StrategyConfig:
    return StrategyConfig.create(max_candidates=max_candidates)


def _risk_config(max_approved_decisions: int) -> RiskConfig:
    return RiskConfig.create(max_approved_decisions=max_approved_decisions)


def _signal_config() -> SignalConfig:
    return SignalConfig.create(trade_direction="long")


def _portfolio_config(initial_capital: float) -> PortfolioConfig:
    return PortfolioConfig.create(initial_capital=initial_capital, allocation_policy="equal_weight")


def _execution_config() -> ExecutionConfig:
    return ExecutionConfig.create()


# --- Scanner-output construction (see module docstring) ----------------------


def _scan_result(symbol: str, *, rsi_14: float, status: str = "candidate") -> SymbolScanResult:
    return SymbolScanResult(
        symbol=symbol,
        requested_evaluation_date=EVALUATION_DATE,
        as_of_date=EVALUATION_DATE,
        status=status,
        passed_filters=(),
        failed_filters=(),
        soft_conditions=(),
        metrics={"rsi_14": rsi_14},
        required_indicators=(),
        data_quality_notes=(),
    )


def _scan_report(results: tuple[SymbolScanResult, ...], *, scanner_config: ScannerConfig) -> ScanReport:
    metadata = DatasetMetadata(
        symbols_requested=len(results),
        symbols_evaluated=len(results),
        evaluation_date=EVALUATION_DATE,
        scanner_version="1",
        config_id=scanner_config.config_id,
    )
    return ScanReport(
        requested_evaluation_date=EVALUATION_DATE,
        config_id=scanner_config.config_id,
        dataset_metadata=metadata,
        results=results,
    )


# --- The real, chained pipeline ----------------------------------------------


def _run_pipeline_from_score_report(
    score_report: ScoreReport,
    *,
    strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    signal_config: SignalConfig,
    portfolio_config: PortfolioConfig,
    execution_config: ExecutionConfig,
    market_snapshot: MarketSnapshot,
) -> tuple[StrategyReport, RiskReport, SignalReport, PortfolioReport, ExecutionReport]:
    strategy_report = StrategyEngine().decide(score_report, config=strategy_config)
    risk_report = RiskEngine().evaluate(strategy_report, config=risk_config)
    signal_report = SignalEngine().generate(risk_report, config=signal_config)
    portfolio_report = PortfolioEngine().build(
        signal_report, config=portfolio_config, market_snapshot=market_snapshot
    )
    execution_report = ExecutionEngine().prepare(portfolio_report, config=execution_config)
    return strategy_report, risk_report, signal_report, portfolio_report, execution_report


def _run_full_pipeline(
    scan_results: tuple[SymbolScanResult, ...],
    *,
    scanner_config: ScannerConfig,
    scorer_config: ScorerConfig,
    strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    signal_config: SignalConfig,
    portfolio_config: PortfolioConfig,
    execution_config: ExecutionConfig,
    market_snapshot: MarketSnapshot,
) -> tuple[ScanReport, ScoreReport, StrategyReport, RiskReport, SignalReport, PortfolioReport, ExecutionReport]:
    scan_report = _scan_report(scan_results, scanner_config=scanner_config)
    score_report = OpportunityScorer().score(scan_report, config=scorer_config)
    rest = _run_pipeline_from_score_report(
        score_report,
        strategy_config=strategy_config,
        risk_config=risk_config,
        signal_config=signal_config,
        portfolio_config=portfolio_config,
        execution_config=execution_config,
        market_snapshot=market_snapshot,
    )
    return (scan_report, score_report, *rest)


# --- Shared main-scenario fixture data ---------------------------------------
#
# Three candidates with distinct rsi_14 values (unique ranks 1, 2, 3 under
# direction="desc"), enough capital and prices for every signal to resolve to
# a whole-share position with zero unallocated cash, chosen so the expected
# quantities are exact integers with no floating-point ambiguity.

_MAIN_SYMBOLS = ("AAPL", "MSFT", "GOOG")
_MAIN_RSI = {"AAPL": 80.0, "MSFT": 60.0, "GOOG": 40.0}
_MAIN_PRICES = {"AAPL": 100.0, "MSFT": 50.0, "GOOG": 20.0}
_MAIN_INITIAL_CAPITAL = 30_000.0
_MAIN_EXPECTED_QUANTITY = {"AAPL": 100, "MSFT": 200, "GOOG": 500}


def _main_scan_results() -> tuple[SymbolScanResult, ...]:
    return tuple(_scan_result(symbol, rsi_14=_MAIN_RSI[symbol]) for symbol in _MAIN_SYMBOLS)


def _main_market_snapshot() -> MarketSnapshot:
    return MarketSnapshot(as_of_date=EVALUATION_DATE, prices=dict(_MAIN_PRICES))


def _run_main_scenario(scan_results: tuple[SymbolScanResult, ...] | None = None):
    scanner_config = _scanner_config(_MAIN_SYMBOLS)
    scorer_config = _scorer_config()
    strategy_config = _strategy_config(max_candidates=3)
    risk_config = _risk_config(max_approved_decisions=3)
    signal_config = _signal_config()
    portfolio_config = _portfolio_config(initial_capital=_MAIN_INITIAL_CAPITAL)
    market_snapshot = _main_market_snapshot()

    reports = _run_full_pipeline(
        scan_results if scan_results is not None else _main_scan_results(),
        scanner_config=scanner_config,
        scorer_config=scorer_config,
        strategy_config=strategy_config,
        risk_config=risk_config,
        signal_config=signal_config,
        portfolio_config=portfolio_config,
        execution_config=_execution_config(),
        market_snapshot=market_snapshot,
    )
    configs = {
        "scanner": scanner_config,
        "scorer": scorer_config,
        "strategy": strategy_config,
        "risk": risk_config,
        "signal": signal_config,
        "portfolio": portfolio_config,
        "execution": _execution_config(),
    }
    return reports, configs


# --- 1. The full chain runs and produces the expected orders -----------------


def test_full_pipeline_runs_all_six_engines_and_produces_three_orders():
    (scan_report, score_report, strategy_report, risk_report, signal_report, portfolio_report, execution_report), _ = (
        _run_main_scenario()
    )

    assert isinstance(scan_report, ScanReport)
    assert isinstance(score_report, ScoreReport)
    assert isinstance(strategy_report, StrategyReport)
    assert isinstance(risk_report, RiskReport)
    assert isinstance(signal_report, SignalReport)
    assert isinstance(portfolio_report, PortfolioReport)
    assert isinstance(execution_report, ExecutionReport)

    assert [candidate.symbol for candidate in score_report.results] == ["AAPL", "MSFT", "GOOG"]
    assert [decision.action for decision in strategy_report.decisions] == ["enter", "enter", "enter"]
    assert [decision.status for decision in risk_report.decisions] == ["approved", "approved", "approved"]
    assert [decision.trade_direction for decision in signal_report.decisions] == ["long", "long", "long"]
    assert len(portfolio_report.target_positions) == 3
    assert len(execution_report.orders) == 3
    assert [order.symbol for order in execution_report.orders] == ["AAPL", "MSFT", "GOOG"]


# --- 2. End-to-end traceability by reference identity -------------------------


def test_end_to_end_traceability_by_identity():
    scan_results = _main_scan_results()
    (scan_report, score_report, strategy_report, risk_report, signal_report, portfolio_report, execution_report), _ = (
        _run_main_scenario(scan_results)
    )

    scan_results_by_symbol = {result.symbol: result for result in scan_results}
    candidates_by_symbol = {candidate.symbol: candidate for candidate in score_report.results}
    strategy_by_symbol = {decision.symbol: decision for decision in strategy_report.decisions}
    risk_by_symbol = {decision.symbol: decision for decision in risk_report.decisions}
    signal_by_symbol = {decision.symbol: decision for decision in signal_report.decisions}
    positions_by_symbol = {position.symbol: position for position in portfolio_report.target_positions}

    assert len(execution_report.orders) == 3
    for order in execution_report.orders:
        symbol = order.symbol

        target_position = positions_by_symbol[symbol]
        assert order.source_position is target_position

        signal_decision = signal_by_symbol[symbol]
        assert target_position.source_signal is signal_decision

        risk_decision = risk_by_symbol[symbol]
        assert signal_decision.source_decision is risk_decision

        strategy_decision = strategy_by_symbol[symbol]
        assert risk_decision.source_decision is strategy_decision

        scored_candidate = candidates_by_symbol[symbol]
        assert strategy_decision.source_candidate is scored_candidate

        original_scan_result = scan_results_by_symbol[symbol]
        assert scored_candidate.source_result is original_scan_result


# --- 3. Config IDs propagate to the final report and match the real configs --


def test_execution_report_config_ids_match_the_real_configs_used():
    (_, _, _, _, _, _, execution_report), configs = _run_main_scenario()

    assert execution_report.source_scanner_config_id == configs["scanner"].config_id
    assert execution_report.source_scorer_config_id == configs["scorer"].config_id
    assert execution_report.source_strategy_config_id == configs["strategy"].config_id
    assert execution_report.source_risk_config_id == configs["risk"].config_id
    assert execution_report.source_signal_config_id == configs["signal"].config_id
    assert execution_report.source_portfolio_config_id == configs["portfolio"].config_id
    assert execution_report.execution_config_id == configs["execution"].config_id


# --- 4. evaluation_date flows unchanged through every intermediate report ----


def test_evaluation_date_is_identical_across_every_report():
    scan_report, score_report, strategy_report, risk_report, signal_report, portfolio_report, execution_report = (
        _run_main_scenario()[0]
    )

    assert scan_report.requested_evaluation_date == EVALUATION_DATE
    assert score_report.evaluation_date == EVALUATION_DATE
    assert strategy_report.evaluation_date == EVALUATION_DATE
    assert risk_report.evaluation_date == EVALUATION_DATE
    assert signal_report.evaluation_date == EVALUATION_DATE
    assert portfolio_report.evaluation_date == EVALUATION_DATE
    assert execution_report.evaluation_date == EVALUATION_DATE


# --- 5. The full pipeline is independent of scan-result input order ----------


def test_full_pipeline_result_is_independent_of_scan_result_input_order():
    ordered = tuple(_scan_result(symbol, rsi_14=_MAIN_RSI[symbol]) for symbol in ("AAPL", "MSFT", "GOOG"))
    reordered = tuple(_scan_result(symbol, rsi_14=_MAIN_RSI[symbol]) for symbol in ("GOOG", "AAPL", "MSFT"))

    execution_report_1 = _run_main_scenario(ordered)[0][-1]
    execution_report_2 = _run_main_scenario(reordered)[0][-1]

    assert [order.symbol for order in execution_report_1.orders] == ["AAPL", "MSFT", "GOOG"]
    assert [order.symbol for order in execution_report_2.orders] == ["AAPL", "MSFT", "GOOG"]
    assert execution_report_1.orders == execution_report_2.orders


# --- 6. Duplicate ranks break ties by symbol end-to-end -----------------------
#
# See the module docstring: a real OpportunityScorer run can never itself
# produce duplicate ranks, so this scenario hand-builds a ScoreReport (the
# same technique used by the duplicate-rank test in every downstream layer's
# own test file) and runs the five real downstream engines against it.


def test_duplicate_ranks_break_ties_by_symbol_end_to_end():
    scanner_config = _scanner_config(("AAPL", "MSFT", "GOOG"))
    scorer_config = _scorer_config()

    candidate_msft = ScoredCandidate(
        symbol="MSFT",
        rank=1,
        metric_used="rsi_14",
        observed_value=80.0,
        source_result=_scan_result("MSFT", rsi_14=80.0),
    )
    candidate_aapl = ScoredCandidate(
        symbol="AAPL",
        rank=1,
        metric_used="rsi_14",
        observed_value=80.0,
        source_result=_scan_result("AAPL", rsi_14=80.0),
    )
    candidate_goog = ScoredCandidate(
        symbol="GOOG",
        rank=2,
        metric_used="rsi_14",
        observed_value=40.0,
        source_result=_scan_result("GOOG", rsi_14=40.0),
    )
    score_report = ScoreReport(
        source_scanner_config_id=scanner_config.config_id,
        source_scanner_version="1",
        scorer_config_id=scorer_config.config_id,
        evaluation_date=EVALUATION_DATE,
        results=(candidate_msft, candidate_aapl, candidate_goog),
        excluded=(),
    )

    _, _, _, _, execution_report = _run_pipeline_from_score_report(
        score_report,
        strategy_config=_strategy_config(max_candidates=3),
        risk_config=_risk_config(max_approved_decisions=3),
        signal_config=_signal_config(),
        portfolio_config=_portfolio_config(initial_capital=_MAIN_INITIAL_CAPITAL),
        execution_config=_execution_config(),
        market_snapshot=_main_market_snapshot(),
    )

    # rank 1 tie between MSFT and AAPL must resolve to AAPL, MSFT (symbol
    # ascending); GOOG (rank 2) must come last, regardless of the hand-built
    # ScoreReport's own declaration order (MSFT, AAPL, GOOG).
    assert [order.symbol for order in execution_report.orders] == ["AAPL", "MSFT", "GOOG"]


# --- 7. Quantity is conserved from Portfolio to Execution, and is correct ----


def test_quantity_is_conserved_from_portfolio_to_execution_and_matches_equal_weight_math():
    (_, _, _, _, _, portfolio_report, execution_report), _ = _run_main_scenario()

    positions_by_symbol = {position.symbol: position for position in portfolio_report.target_positions}
    orders_by_symbol = {order.symbol: order for order in execution_report.orders}

    assert set(orders_by_symbol) == set(_MAIN_EXPECTED_QUANTITY)
    for symbol, expected_quantity in _MAIN_EXPECTED_QUANTITY.items():
        target_position = positions_by_symbol[symbol]
        order = orders_by_symbol[symbol]
        assert target_position.quantity == expected_quantity
        assert order.quantity == target_position.quantity


# --- 8. side and status arise from the real pipeline, never hand-set --------


def test_every_order_has_side_buy_and_status_simulated():
    (_, _, _, _, _, _, execution_report), _ = _run_main_scenario()

    assert len(execution_report.orders) == 3
    assert all(order.side == "buy" for order in execution_report.orders)
    assert all(order.status == "simulated" for order in execution_report.orders)


# --- 9. An empty scan (no candidates) propagates cleanly to no orders -------


def test_scanner_with_no_candidates_propagates_to_an_empty_execution_report():
    scan_results = (_scan_result("AAPL", rsi_14=80.0, status="rejected"),)

    reports = _run_full_pipeline(
        scan_results,
        scanner_config=_scanner_config(("AAPL",)),
        scorer_config=_scorer_config(),
        strategy_config=_strategy_config(max_candidates=3),
        risk_config=_risk_config(max_approved_decisions=3),
        signal_config=_signal_config(),
        portfolio_config=_portfolio_config(initial_capital=_MAIN_INITIAL_CAPITAL),
        execution_config=_execution_config(),
        market_snapshot=MarketSnapshot(as_of_date=EVALUATION_DATE, prices={}),
    )
    scan_report, score_report, strategy_report, risk_report, signal_report, portfolio_report, execution_report = reports

    assert score_report.results == ()
    assert strategy_report.decisions == ()
    assert risk_report.decisions == ()
    assert signal_report.decisions == ()
    assert portfolio_report.target_positions == ()
    assert portfolio_report.unallocated_cash == _MAIN_INITIAL_CAPITAL
    assert execution_report.orders == ()


# --- 10. No phantom orders: exact 1:1 relationship with target positions ----


def test_no_phantom_orders_exact_one_to_one_with_target_positions():
    (_, _, _, _, _, portfolio_report, execution_report), _ = _run_main_scenario()

    assert len(execution_report.orders) == len(portfolio_report.target_positions) == 3

    for order in execution_report.orders:
        matches = [tp for tp in portfolio_report.target_positions if tp is order.source_position]
        assert len(matches) == 1

    for target_position in portfolio_report.target_positions:
        matches = [order for order in execution_report.orders if order.source_position is target_position]
        assert len(matches) == 1
