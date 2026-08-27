from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_agent.data.exceptions import DataValidationError
from trading_agent.indicators import IndicatorConfig, IndicatorsEngine
from trading_agent.scanner import (
    HardFilter,
    MarketScanner,
    ScannerConfig,
    ScannerConfigError,
    SoftCondition,
)

SCANNER_SRC = Path(__file__).resolve().parents[1] / "src" / "trading_agent" / "scanner"


def make_ohlcv(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Business-day OHLCV bars with a fixed +-1 high/low band around close."""
    dates = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        }
    )


def combine(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def new_scanner() -> MarketScanner:
    return MarketScanner(IndicatorsEngine())


# ---------------------------------------------------------------------------
# 1-3: hard filter outcomes (pass / single failure / multiple failures)
# ---------------------------------------------------------------------------


def test_symbol_passing_all_filters_is_a_candidate() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104, 105, 106, 107])
    config = ScannerConfig.create(
        universe=["AAPL"],
        indicator_requirements=[IndicatorConfig("sma", 3)],
        hard_filters=[
            HardFilter(filter_id="min_price", field="close", operator=">=", threshold=50.0),
            HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_3"),
        ],
    )
    report = new_scanner().scan(aapl, config=config, evaluation_date=aapl["date"].max())

    (result,) = report.results
    assert result.status == "candidate"
    assert result.failed_filters == ()
    assert {evaluation.filter_id for evaluation in result.passed_filters} == {"min_price", "uptrend"}


def test_symbol_failing_one_hard_filter_is_rejected() -> None:
    spy = make_ohlcv("SPY", [110, 109, 108, 107, 106, 105, 104, 103])
    config = ScannerConfig.create(
        universe=["SPY"],
        indicator_requirements=[IndicatorConfig("sma", 3)],
        hard_filters=[
            HardFilter(filter_id="min_price", field="close", operator=">=", threshold=50.0),
            HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_3"),
        ],
    )
    report = new_scanner().scan(spy, config=config, evaluation_date=spy["date"].max())

    (result,) = report.results
    assert result.status == "rejected"
    assert [evaluation.filter_id for evaluation in result.failed_filters] == ["uptrend"]
    assert [evaluation.filter_id for evaluation in result.passed_filters] == ["min_price"]


def test_symbol_failing_multiple_hard_filters_reports_all_of_them() -> None:
    penny = make_ohlcv("PENNY", [10, 9, 8, 7, 6, 5, 4, 3])
    config = ScannerConfig.create(
        universe=["PENNY"],
        indicator_requirements=[IndicatorConfig("sma", 3)],
        hard_filters=[
            HardFilter(filter_id="min_price", field="close", operator=">=", threshold=50.0),
            HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_3"),
        ],
    )
    report = new_scanner().scan(penny, config=config, evaluation_date=penny["date"].max())

    (result,) = report.results
    assert result.status == "rejected"
    assert {evaluation.filter_id for evaluation in result.failed_filters} == {"min_price", "uptrend"}
    assert result.passed_filters == ()


# ---------------------------------------------------------------------------
# 4, 16, 23: warm-up / insufficient indicator history, across indicator types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "indicator, insufficient_length",
    [
        (IndicatorConfig("sma", 3), 2),
        (IndicatorConfig("ema", 3), 2),
        (IndicatorConfig("atr", 3), 2),
        (IndicatorConfig("rsi", 3), 3),
        (IndicatorConfig("momentum", 3), 3),
        (IndicatorConfig("returns"), 1),
    ],
)
def test_indicator_warmup_nan_yields_insufficient_data(
    indicator: IndicatorConfig, insufficient_length: int
) -> None:
    closes = [100.0 + i for i in range(insufficient_length)]
    data = make_ohlcv("WARM", closes)
    config = ScannerConfig.create(universe=["WARM"], indicator_requirements=[indicator])

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    assert result.status == "insufficient_data"
    assert all(evaluation.reason_code == "insufficient_indicator_history" for evaluation in result.failed_filters)
    assert indicator.column_name() in result.required_indicators


# ---------------------------------------------------------------------------
# 5, 8: missing / absent symbol data
# ---------------------------------------------------------------------------


def test_symbol_absent_from_dataset_is_insufficient_data() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102])
    config = ScannerConfig.create(universe=["AAPL", "QQQ"])

    report = new_scanner().scan(aapl, config=config, evaluation_date=aapl["date"].max())

    qqq = next(result for result in report.results if result.symbol == "QQQ")
    assert qqq.status == "insufficient_data"
    assert qqq.as_of_date is None
    assert all(evaluation.reason_code == "missing_symbol_data" for evaluation in qqq.failed_filters)


def test_evaluation_date_before_all_data_is_insufficient_data() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102], start="2024-01-01")
    config = ScannerConfig.create(universe=["AAPL"])

    report = new_scanner().scan(aapl, config=config, evaluation_date="2023-12-25")

    (result,) = report.results
    assert result.status == "insufficient_data"
    assert result.as_of_date is None


# ---------------------------------------------------------------------------
# Symbol normalization: data["symbol"] must already be uppercase, matching
# normalize_ohlcv's guarantee and ScannerConfig.universe's own normalization.
# A mismatch is a contract violation, not a "no data" business outcome.
# ---------------------------------------------------------------------------


def test_scan_rejects_lowercase_symbol_even_when_universe_matches_case_insensitively() -> None:
    lowercase_aapl = make_ohlcv("aapl", [100, 101, 102])
    config = ScannerConfig.create(universe=["AAPL"])  # normalized to uppercase internally

    with pytest.raises(DataValidationError, match="uppercase"):
        new_scanner().scan(lowercase_aapl, config=config, evaluation_date=lowercase_aapl["date"].max())


def test_scan_rejects_mixed_case_inconsistency_across_symbols_and_names_the_offender() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102])  # already normalized
    inconsistent_spy = make_ohlcv("Spy", [200, 201, 202])  # not normalized
    data = combine(aapl, inconsistent_spy)
    config = ScannerConfig.create(universe=["AAPL", "SPY"])

    with pytest.raises(DataValidationError, match="Spy") as excinfo:
        new_scanner().scan(data, config=config, evaluation_date=aapl["date"].max())
    # The already-correct AAPL must not be named as an offender.
    assert "AAPL" not in str(excinfo.value).split("symbol(s):")[-1]


def test_scan_still_treats_symbols_independently_once_all_are_correctly_normalized() -> None:
    # Guards against the new upfront check accidentally coupling symbols
    # together (e.g. validating the whole frame in a way that leaks one
    # symbol's data into another's result) now that scan() does one more
    # whole-frame pass before the existing per-symbol slicing.
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104, 105, 106, 107])
    spy = make_ohlcv("SPY", [110, 109, 108, 107, 106, 105, 104, 103])
    config = ScannerConfig.create(
        universe=["AAPL", "SPY"],
        indicator_requirements=[IndicatorConfig("sma", 3)],
        hard_filters=[HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_3")],
    )

    joint_report = new_scanner().scan(combine(aapl, spy), config=config, evaluation_date=aapl["date"].max())
    solo_aapl_report = new_scanner().scan(aapl, config=config, evaluation_date=aapl["date"].max())

    by_symbol = {result.symbol: result for result in joint_report.results}
    assert by_symbol["AAPL"].status == "candidate"
    assert by_symbol["SPY"].status == "rejected"
    assert by_symbol["AAPL"].metrics == solo_aapl_report.results[0].metrics


# ---------------------------------------------------------------------------
# 6, 7: evaluation_date on a trading day vs. a weekend
# ---------------------------------------------------------------------------


def test_evaluation_date_on_a_trading_day_uses_that_exact_session() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104], start="2024-01-01")  # Mon-Fri
    config = ScannerConfig.create(universe=["AAPL"], max_staleness_days=3)
    trading_day = aapl["date"].iloc[2]  # Wednesday

    report = new_scanner().scan(aapl, config=config, evaluation_date=trading_day)

    (result,) = report.results
    assert result.as_of_date == trading_day.date()
    assert result.requested_evaluation_date == trading_day.date()


def test_evaluation_date_on_a_weekend_falls_back_to_last_session() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104], start="2024-01-01")  # Mon-Fri
    last_friday = aapl["date"].max()
    weekend_date = last_friday + pd.Timedelta(days=2)  # Sunday
    config = ScannerConfig.create(universe=["AAPL"], max_staleness_days=3)

    report = new_scanner().scan(aapl, config=config, evaluation_date=weekend_date)

    (result,) = report.results
    assert result.as_of_date == last_friday.date()
    assert result.requested_evaluation_date == weekend_date.date()
    assert result.status != "insufficient_data"


# ---------------------------------------------------------------------------
# 9, 10, 22: anti-look-ahead, across every indicator the scanner supports
# ---------------------------------------------------------------------------


def test_anti_lookahead_holds_for_every_supported_indicator() -> None:
    stable_closes = [100.0, 101.0, 99.0, 102.0, 103.0, 104.0]
    future_shock = [10.0, 500.0, 5.0, 999.0]
    dataset_a = make_ohlcv("AAPL", stable_closes)
    dataset_b = combine(dataset_a, make_ohlcv("AAPL", future_shock, start="2024-01-15"))
    evaluation_date = dataset_a["date"].max()

    indicators = [
        IndicatorConfig("sma", 3),
        IndicatorConfig("ema", 3),
        IndicatorConfig("rsi", 3),
        IndicatorConfig("momentum", 3),
        IndicatorConfig("atr", 3),
        "returns",
    ]
    config = ScannerConfig.create(universe=["AAPL"], indicator_requirements=indicators)
    scanner = new_scanner()

    report_a = scanner.scan(dataset_a, config=config, evaluation_date=evaluation_date)
    report_b = scanner.scan(dataset_b, config=config, evaluation_date=evaluation_date)

    assert report_a.results == report_b.results


# ---------------------------------------------------------------------------
# 11: staleness
# ---------------------------------------------------------------------------


def test_stale_data_is_insufficient_data_not_rejected() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104], start="2024-01-01")  # last session Fri Jan 5
    stale_evaluation_date = "2024-01-10"  # following Wednesday
    config = ScannerConfig.create(universe=["AAPL"], max_staleness_days=0)

    report = new_scanner().scan(aapl, config=config, evaluation_date=stale_evaluation_date)

    (result,) = report.results
    assert result.status == "insufficient_data"
    assert all(evaluation.reason_code == "stale_data" for evaluation in result.failed_filters)


def test_staleness_within_policy_is_not_flagged() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104], start="2024-01-01")
    weekend_date = aapl["date"].max() + pd.Timedelta(days=2)
    config = ScannerConfig.create(universe=["AAPL"], max_staleness_days=3)

    report = new_scanner().scan(aapl, config=config, evaluation_date=weekend_date)

    (result,) = report.results
    assert result.status != "insufficient_data"


def test_staleness_exactly_at_the_boundary_is_not_stale() -> None:
    # Policy: staleness_days <= max_staleness_days is allowed; only strictly
    # greater than the limit is insufficient_data.
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104], start="2024-01-01")  # last session Fri Jan 5
    evaluation_date = aapl["date"].max() + pd.Timedelta(days=2)  # Sunday Jan 7: staleness_days == 2
    config = ScannerConfig.create(universe=["AAPL"], max_staleness_days=2)

    report = new_scanner().scan(aapl, config=config, evaluation_date=evaluation_date)

    (result,) = report.results
    assert result.status != "insufficient_data"
    assert result.as_of_date == aapl["date"].max().date()


def test_staleness_one_day_past_the_boundary_is_stale() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104], start="2024-01-01")  # last session Fri Jan 5
    evaluation_date = aapl["date"].max() + pd.Timedelta(days=2)  # same Sunday: staleness_days == 2
    config = ScannerConfig.create(universe=["AAPL"], max_staleness_days=1)  # one day tighter than the boundary

    report = new_scanner().scan(aapl, config=config, evaluation_date=evaluation_date)

    (result,) = report.results
    assert result.status == "insufficient_data"
    assert all(evaluation.reason_code == "stale_data" for evaluation in result.failed_filters)


# ---------------------------------------------------------------------------
# 12, 13, 18: multi-symbol scans, and isolation between symbols
# ---------------------------------------------------------------------------


def test_multiple_symbols_are_evaluated_independently() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104, 105, 106, 107])
    spy = make_ohlcv("SPY", [110, 109, 108, 107, 106, 105, 104, 103])
    data = combine(aapl, spy)
    config = ScannerConfig.create(
        universe=["AAPL", "SPY", "QQQ"],
        indicator_requirements=[IndicatorConfig("sma", 3)],
        hard_filters=[HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_3")],
    )

    report = new_scanner().scan(data, config=config, evaluation_date=aapl["date"].max())
    by_symbol = {result.symbol: result for result in report.results}

    assert by_symbol["AAPL"].status == "candidate"
    assert by_symbol["SPY"].status == "rejected"
    assert by_symbol["QQQ"].status == "insufficient_data"


def test_symbols_do_not_contaminate_each_others_indicator_values() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104, 105, 106, 107])
    spy = make_ohlcv("SPY", [110, 109, 108, 107, 106, 105, 104, 103])
    config = ScannerConfig.create(universe=["AAPL"], indicator_requirements=[IndicatorConfig("sma", 3)])
    evaluation_date = aapl["date"].max()

    solo_report = new_scanner().scan(aapl, config=config, evaluation_date=evaluation_date)
    joint_report = new_scanner().scan(combine(aapl, spy), config=config, evaluation_date=evaluation_date)

    assert solo_report.results[0].metrics == joint_report.results[0].metrics


# ---------------------------------------------------------------------------
# 14, 15: price filters and indicator filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operator, threshold, expect_pass",
    [(">=", 104.0, True), (">=", 200.0, False), ("<=", 104.0, True), (">", 104.0, False), ("<", 104.0, False)],
)
def test_price_hard_filter_operators(operator: str, threshold: float, expect_pass: bool) -> None:
    data = make_ohlcv("AAPL", [100, 101, 102, 103, 104])
    config = ScannerConfig.create(
        universe=["AAPL"],
        hard_filters=[HardFilter(filter_id="price", field="close", operator=operator, threshold=threshold)],
    )

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    evaluation = result.passed_filters[0] if expect_pass else result.failed_filters[0]
    assert evaluation.filter_id == "price"


@pytest.mark.parametrize(
    "operator, expect_pass",
    [(">=", True), ("<=", True), (">", False), ("<", False)],
)
def test_price_hard_filter_at_exact_equality_boundary(operator: str, expect_pass: bool) -> None:
    # close on the last session is exactly 104.0; threshold == 104.0 as well.
    data = make_ohlcv("AAPL", [100, 101, 102, 103, 104])
    config = ScannerConfig.create(
        universe=["AAPL"],
        hard_filters=[HardFilter(filter_id="boundary", field="close", operator=operator, threshold=104.0)],
    )

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    evaluation = result.passed_filters[0] if expect_pass else result.failed_filters[0]
    assert evaluation.filter_id == "boundary"
    assert evaluation.observed_value == pytest.approx(104.0)
    assert evaluation.threshold == pytest.approx(104.0)
    assert evaluation.status == ("passed" if expect_pass else "failed")


def test_indicator_hard_filter_uses_computed_indicator_value() -> None:
    # Matches the documented rsi_2 Wilder values for closes=[1,2,1,3,2]: 50.0, then 83.3333...
    data = make_ohlcv("IND", [1, 2, 1, 3, 2])
    config = ScannerConfig.create(
        universe=["IND"],
        indicator_requirements=[IndicatorConfig("rsi", 2)],
        hard_filters=[
            HardFilter(filter_id="rsi_too_hot", field="rsi_2", operator="<=", threshold=80.0),
            HardFilter(filter_id="rsi_generous", field="rsi_2", operator="<=", threshold=90.0),
        ],
    )
    evaluation_date = data["date"].iloc[3]  # rsi_2 == 83.3333...

    report = new_scanner().scan(data, config=config, evaluation_date=evaluation_date)

    (result,) = report.results
    assert [evaluation.filter_id for evaluation in result.failed_filters] == ["rsi_too_hot"]
    assert [evaluation.filter_id for evaluation in result.passed_filters] == ["rsi_generous"]
    assert result.failed_filters[0].observed_value == pytest.approx(83.3333333333, rel=1e-9)


# ---------------------------------------------------------------------------
# 16: indicator-vs-indicator comparison
# ---------------------------------------------------------------------------


def test_indicator_vs_indicator_comparison() -> None:
    uptrend = make_ohlcv("UP", [100, 101, 102, 103, 104, 105, 106])
    downtrend = make_ohlcv("DOWN", [106, 105, 104, 103, 102, 101, 100])
    config = ScannerConfig.create(
        universe=["UP", "DOWN"],
        indicator_requirements=[IndicatorConfig("ema", 3), IndicatorConfig("sma", 3)],
        hard_filters=[HardFilter(filter_id="ema_above_sma", field="ema_3", operator=">", compare_field="sma_3")],
    )

    report = new_scanner().scan(combine(uptrend, downtrend), config=config, evaluation_date=uptrend["date"].max())
    by_symbol = {result.symbol: result for result in report.results}

    assert by_symbol["UP"].status == "candidate"
    assert by_symbol["DOWN"].status == "rejected"


# ---------------------------------------------------------------------------
# 17: soft conditions never eliminate candidates
# ---------------------------------------------------------------------------


def test_soft_conditions_are_recorded_but_never_reject() -> None:
    data = make_ohlcv("IND", [1, 2, 1, 3, 2])
    config = ScannerConfig.create(
        universe=["IND"],
        indicator_requirements=[IndicatorConfig("rsi", 2)],
        hard_filters=[HardFilter(filter_id="min_price", field="close", operator=">=", threshold=0.0)],
        soft_conditions=[SoftCondition(field="rsi_2")],
    )
    evaluation_date = data["date"].iloc[3]  # rsi_2 == 83.3333... (an "overbought" reading)

    report = new_scanner().scan(data, config=config, evaluation_date=evaluation_date)

    (result,) = report.results
    assert result.status == "candidate"
    assert len(result.soft_conditions) == 1
    assert result.soft_conditions[0].field == "rsi_2"
    assert result.soft_conditions[0].value == pytest.approx(83.3333333333, rel=1e-9)


# ---------------------------------------------------------------------------
# 18: reason codes
# ---------------------------------------------------------------------------


def test_reason_codes_are_structured_and_specific() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102, 103, 104, 105, 106, 107])
    config = ScannerConfig.create(
        universe=["AAPL", "GHOST"],
        indicator_requirements=[IndicatorConfig("sma", 3)],
        hard_filters=[
            HardFilter(filter_id="min_price", field="close", operator=">=", threshold=50.0),
            HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_3"),
        ],
    )

    report = new_scanner().scan(aapl, config=config, evaluation_date=aapl["date"].max())
    by_symbol = {result.symbol: result for result in report.results}

    assert {evaluation.reason_code for evaluation in by_symbol["AAPL"].passed_filters} == {"filter_passed"}
    assert {evaluation.reason_code for evaluation in by_symbol["GHOST"].failed_filters} == {"missing_symbol_data"}


# ---------------------------------------------------------------------------
# 19: config_id determinism, sensitivity to material changes, and invariance
# to the declaration order of hard_filters / soft_conditions.
# ---------------------------------------------------------------------------


def _reference_config(**overrides: object) -> ScannerConfig:
    defaults: dict = dict(
        universe=["AAPL", "SPY"],
        indicator_requirements=[IndicatorConfig("sma", 20)],
        hard_filters=[
            HardFilter(filter_id="min_price", field="close", operator=">=", threshold=10.0),
            HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_20"),
        ],
        soft_conditions=[SoftCondition(field="close"), SoftCondition(field="sma_20")],
        max_staleness_days=1,
        version="1",
    )
    defaults.update(overrides)
    return ScannerConfig.create(**defaults)


def test_config_id_is_deterministic_for_identical_configuration() -> None:
    assert _reference_config().config_id == _reference_config().config_id


def test_config_id_changes_when_a_filter_threshold_changes() -> None:
    changed_filters = [
        HardFilter(filter_id="min_price", field="close", operator=">=", threshold=999.0),
        HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_20"),
    ]
    assert _reference_config().config_id != _reference_config(hard_filters=changed_filters).config_id


def test_config_id_changes_when_universe_changes() -> None:
    assert _reference_config().config_id != _reference_config(universe=["AAPL", "SPY", "QQQ"]).config_id


def test_config_id_changes_when_indicator_requirements_change() -> None:
    # Hard filters/soft conditions here only reference base OHLCV fields, so the
    # indicator requirement can vary freely without breaking field validation.
    base = ScannerConfig.create(
        universe=["AAPL"],
        indicator_requirements=[IndicatorConfig("sma", 20)],
        hard_filters=[HardFilter(filter_id="min_price", field="close", operator=">=", threshold=10.0)],
    )
    changed = ScannerConfig.create(
        universe=["AAPL"],
        indicator_requirements=[IndicatorConfig("sma", 50)],
        hard_filters=[HardFilter(filter_id="min_price", field="close", operator=">=", threshold=10.0)],
    )
    assert base.config_id != changed.config_id


def test_config_id_changes_when_max_staleness_days_changes() -> None:
    assert _reference_config().config_id != _reference_config(max_staleness_days=5).config_id


def test_config_id_changes_when_version_changes() -> None:
    assert _reference_config().config_id != _reference_config(version="2").config_id


def test_config_id_is_invariant_to_hard_filter_declaration_order() -> None:
    forward = [
        HardFilter(filter_id="min_price", field="close", operator=">=", threshold=10.0),
        HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_20"),
    ]
    reversed_filters = list(reversed(forward))

    config_forward = _reference_config(hard_filters=forward)
    config_reversed = _reference_config(hard_filters=reversed_filters)

    assert config_forward.config_id == config_reversed.config_id
    # The evaluation order itself is untouched by canonicalization.
    assert [item.filter_id for item in config_forward.hard_filters] == ["min_price", "uptrend"]
    assert [item.filter_id for item in config_reversed.hard_filters] == ["uptrend", "min_price"]


def test_config_id_is_invariant_to_soft_condition_declaration_order() -> None:
    forward = [SoftCondition(field="close"), SoftCondition(field="sma_20")]
    reversed_conditions = list(reversed(forward))

    config_forward = _reference_config(soft_conditions=forward)
    config_reversed = _reference_config(soft_conditions=reversed_conditions)

    assert config_forward.config_id == config_reversed.config_id
    assert [item.field for item in config_reversed.soft_conditions] == ["sma_20", "close"]


# ---------------------------------------------------------------------------
# 20, 21: reproducibility and deterministic ordering
# ---------------------------------------------------------------------------


def test_scan_is_reproducible_for_identical_inputs() -> None:
    data = make_ohlcv("AAPL", [100, 101, 102, 103, 104, 105, 106, 107])
    config = ScannerConfig.create(
        universe=["AAPL"],
        indicator_requirements=[IndicatorConfig("sma", 3)],
        hard_filters=[HardFilter(filter_id="uptrend", field="close", operator=">", compare_field="sma_3")],
    )
    scanner = new_scanner()

    first = scanner.scan(data, config=config, evaluation_date=data["date"].max())
    second = scanner.scan(data, config=config, evaluation_date=data["date"].max())

    assert first == second


def test_results_are_ordered_deterministically_by_symbol() -> None:
    aapl = make_ohlcv("AAPL", [100, 101, 102])
    spy = make_ohlcv("SPY", [100, 101, 102])
    config = ScannerConfig.create(universe=["SPY", "AAPL", "QQQ"])

    report = new_scanner().scan(combine(aapl, spy), config=config, evaluation_date=aapl["date"].max())

    assert [result.symbol for result in report.results] == ["AAPL", "QQQ", "SPY"]


# ---------------------------------------------------------------------------
# 22: multiple indicators combined
# ---------------------------------------------------------------------------


def test_multiple_indicators_are_all_computed_and_reported() -> None:
    data = make_ohlcv("AAPL", [100, 101, 99, 102, 103, 104, 105, 106])
    indicators = [
        IndicatorConfig("sma", 3),
        IndicatorConfig("ema", 3),
        IndicatorConfig("rsi", 3),
        IndicatorConfig("momentum", 3),
        IndicatorConfig("atr", 3),
    ]
    config = ScannerConfig.create(universe=["AAPL"], indicator_requirements=indicators)

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    assert result.status == "candidate"
    expected_columns = {"sma_3", "ema_3", "rsi_3", "momentum_3", "atr_3"}
    assert set(result.required_indicators) == expected_columns
    assert expected_columns <= set(result.metrics)
    assert all(result.metrics[column] is not None for column in expected_columns)


# ---------------------------------------------------------------------------
# 24: the scanner never imports yfinance or a provider
# ---------------------------------------------------------------------------


def test_scanner_package_does_not_reference_yfinance_or_providers() -> None:
    for source_file in SCANNER_SRC.glob("*.py"):
        content = source_file.read_text()
        assert "yfinance" not in content, f"{source_file} must not reference yfinance."
        assert "providers" not in content, f"{source_file} must not couple to a data provider."


# ---------------------------------------------------------------------------
# ScannerConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_hard_filter_referencing_undeclared_indicator() -> None:
    with pytest.raises(ScannerConfigError, match="unknown field"):
        ScannerConfig.create(
            universe=["AAPL"],
            hard_filters=[HardFilter(filter_id="rsi_check", field="rsi_14", operator=">=", threshold=30.0)],
        )


def test_config_rejects_soft_condition_referencing_undeclared_indicator() -> None:
    with pytest.raises(ScannerConfigError, match="unknown field"):
        ScannerConfig.create(universe=["AAPL"], soft_conditions=[SoftCondition(field="momentum_20")])


def test_config_rejects_empty_universe() -> None:
    with pytest.raises(ScannerConfigError):
        ScannerConfig.create(universe=[])


def test_config_rejects_duplicate_filter_ids() -> None:
    with pytest.raises(ScannerConfigError, match="unique"):
        ScannerConfig.create(
            universe=["AAPL"],
            hard_filters=[
                HardFilter(filter_id="dup", field="close", operator=">=", threshold=1.0),
                HardFilter(filter_id="dup", field="close", operator="<=", threshold=2.0),
            ],
        )


def test_hard_filter_requires_exactly_one_of_threshold_or_compare_field() -> None:
    with pytest.raises(ScannerConfigError):
        HardFilter(filter_id="bad", field="close", operator=">=", threshold=1.0, compare_field="sma_3")
    with pytest.raises(ScannerConfigError):
        HardFilter(filter_id="bad", field="close", operator=">=")


# ---------------------------------------------------------------------------
# metrics["close"] is unconditionally available (MEDIUM finding remediation)
# ---------------------------------------------------------------------------


def test_metrics_close_is_present_even_when_never_declared_in_config() -> None:
    data = make_ohlcv("AAPL", [100.0, 101.0, 102.0, 103.0, 104.0])
    config = ScannerConfig.create(universe=["AAPL"])  # no hard_filters, soft_conditions, or indicators

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    assert result.status == "candidate"
    assert result.metrics["close"] == 104.0


def test_metrics_close_matches_the_as_of_date_row_exactly() -> None:
    data = make_ohlcv("AAPL", [100.0, 101.0, 999.0])
    config = ScannerConfig.create(universe=["AAPL"])
    evaluation_date = data["date"].iloc[1]  # stop before the 999.0 session

    report = new_scanner().scan(data, config=config, evaluation_date=evaluation_date)

    (result,) = report.results
    assert result.as_of_date == data["date"].iloc[1].date()
    assert result.metrics["close"] == 101.0


def test_metrics_close_never_reflects_a_future_session() -> None:
    stable_closes = [100.0, 101.0, 99.0, 102.0, 103.0, 104.0]
    future_shock = [10.0, 500.0, 5.0, 999.0]
    dataset_a = make_ohlcv("AAPL", stable_closes)
    dataset_b = combine(dataset_a, make_ohlcv("AAPL", future_shock, start="2024-01-15"))
    evaluation_date = dataset_a["date"].max()
    config = ScannerConfig.create(universe=["AAPL"])
    scanner = new_scanner()

    report_a = scanner.scan(dataset_a, config=config, evaluation_date=evaluation_date)
    report_b = scanner.scan(dataset_b, config=config, evaluation_date=evaluation_date)

    assert report_a.results == report_b.results
    assert report_a.results[0].metrics["close"] == stable_closes[-1]
    assert report_a.results[0].metrics["close"] not in future_shock


def test_metrics_close_is_independent_between_symbols() -> None:
    aapl = make_ohlcv("AAPL", [100.0, 101.0, 102.0])
    spy = make_ohlcv("SPY", [400.0, 401.0, 402.0])
    config = ScannerConfig.create(universe=["AAPL", "SPY"])
    evaluation_date = aapl["date"].max()

    solo_aapl_report = new_scanner().scan(aapl, config=config, evaluation_date=evaluation_date)
    joint_report = new_scanner().scan(combine(aapl, spy), config=config, evaluation_date=evaluation_date)

    by_symbol = {result.symbol: result for result in joint_report.results}
    assert by_symbol["AAPL"].metrics["close"] == 102.0
    assert by_symbol["SPY"].metrics["close"] == 402.0
    assert by_symbol["AAPL"].metrics["close"] == solo_aapl_report.results[0].metrics["close"]


def test_metrics_close_still_present_and_unchanged_when_explicitly_requested() -> None:
    data = make_ohlcv("AAPL", [100.0, 101.0, 102.0, 103.0])
    config = ScannerConfig.create(
        universe=["AAPL"],
        hard_filters=[HardFilter(filter_id="min_price", field="close", operator=">=", threshold=50.0)],
        soft_conditions=[SoftCondition(field="close")],
    )

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    assert result.status == "candidate"
    assert result.metrics["close"] == 103.0
    assert result.soft_conditions[0].field == "close"
    assert result.soft_conditions[0].value == 103.0
    assert result.passed_filters[0].observed_value == 103.0


def test_metrics_close_is_not_fabricated_for_missing_symbol_data() -> None:
    data = make_ohlcv("AAPL", [100.0, 101.0], start="2024-02-01")
    config = ScannerConfig.create(universe=["AAPL"])
    evaluation_date = "2024-01-01"  # entirely before any observation

    report = new_scanner().scan(data, config=config, evaluation_date=evaluation_date)

    (result,) = report.results
    assert result.status == "insufficient_data"
    assert result.metrics == {}


def test_metrics_close_is_not_fabricated_for_stale_data() -> None:
    aapl = make_ohlcv("AAPL", [100.0, 101.0, 102.0], start="2024-01-01")
    stale_evaluation_date = "2024-01-10"
    config = ScannerConfig.create(universe=["AAPL"], max_staleness_days=0)

    report = new_scanner().scan(aapl, config=config, evaluation_date=stale_evaluation_date)

    (result,) = report.results
    assert result.status == "insufficient_data"
    assert result.metrics == {}


def test_metrics_close_is_not_fabricated_when_warmup_is_insufficient() -> None:
    data = make_ohlcv("AAPL", [100.0, 101.0])
    config = ScannerConfig.create(universe=["AAPL"], indicator_requirements=[IndicatorConfig("sma", 5)])

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    assert result.status == "insufficient_data"
    assert result.metrics == {}


def test_metrics_close_is_present_for_rejected_symbols_too() -> None:
    data = make_ohlcv("AAPL", [100.0, 99.0, 98.0, 97.0])
    config = ScannerConfig.create(
        universe=["AAPL"],
        hard_filters=[HardFilter(filter_id="min_price", field="close", operator=">=", threshold=1000.0)],
    )

    report = new_scanner().scan(data, config=config, evaluation_date=data["date"].max())

    (result,) = report.results
    assert result.status == "rejected"
    assert result.metrics["close"] == 97.0
