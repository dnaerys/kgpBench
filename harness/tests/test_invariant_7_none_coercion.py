"""Invariant 7 — `None` is never silently coerced to a number in the analysis layer.

`float("nan")` serialises to JSON `null` and reads back as `None`, and `None` is
not the framework's unscored sentinel: `_eval/task/results.py:510` tests only for
NaN, and `value_to_float` (`scorer/_metric.py:205-255`) falls through to
``return 0.0``. So `recompute_metrics` over a stored log turns every unreachable
check into a failed check.

The harness does not use the framework's aggregation for anything reported. This
file asserts that our own code cannot make the same mistake, and — separately —
pins the framework behaviour we are working around, so that a framework bump that
fixes it shows up as a failing test rather than as a silent change of meaning.
"""

from __future__ import annotations

import math

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import recompute_metrics
from mock_harness import clean_model, mock_task

from genomics_harness import (
    Outcome,
    judge_verdict_rates,
    outcome_of_score_value,
    read_log,
    verdict_table,
)
from genomics_harness.log_reading import _NUMERIC_OUTCOME

# -- the decoder ----------------------------------------------------------


def test_none_is_unreachable_not_zero():
    assert outcome_of_score_value(None) is Outcome.NOT_REACHABLE
    assert outcome_of_score_value(None) is not Outcome.NOT_PERFORMED


def test_unreachable_has_no_numeric_value_at_all():
    """The lookup that would put NaN into a mean does not have a key for it."""
    assert Outcome.NOT_REACHABLE not in _NUMERIC_OUTCOME
    with pytest.raises(KeyError):
        _NUMERIC_OUTCOME[Outcome.NOT_REACHABLE]


@pytest.mark.parametrize("value", ["", "0", "null", "None", [], {}, 0.25, -0.0])
def test_nothing_else_decodes_by_accident(value):
    if value == -0.0:
        assert outcome_of_score_value(value) is Outcome.NOT_PERFORMED  # -0.0 == 0.0
        return
    with pytest.raises(ValueError):
        outcome_of_score_value(value)


# -- the aggregation ------------------------------------------------------


def _cells(**by_check):
    from genomics_harness import Completeness, VerdictCell

    return [
        VerdictCell(
            sample_id="s",
            epoch=epoch,
            judge="j",
            check_id=check_id,
            outcome=outcome_of_score_value(value),
            completeness=Completeness.CLEAN,
        )
        for check_id, values in by_check.items()
        for epoch, value in enumerate(values, start=1)
    ]


def test_unreachable_verdicts_leave_the_denominator():
    rates = judge_verdict_rates(_cells(A1=[1.0, None, 1.0]))
    assert rates["A1"].scored == 2
    assert rates["A1"].not_reachable == 1
    assert rates["A1"].rate == 1.0


def test_zero_is_a_failure_and_stays_in_the_denominator():
    rates = judge_verdict_rates(_cells(A1=[1.0, 0.0]))
    assert rates["A1"].scored == 2
    assert rates["A1"].not_reachable == 0
    assert rates["A1"].rate == 0.5


def test_an_all_unreachable_check_has_no_rate_rather_than_a_rate_of_zero():
    """Same failure one level up: reporting 0.0 says the model failed something
    it was never asked."""
    rates = judge_verdict_rates(_cells(A1=[None, None, None]))
    assert rates["A1"].rate is None
    assert rates["A1"].rate != 0.0
    assert rates["A1"].scored == 0


def test_partial_credit_is_explicit():
    rates = judge_verdict_rates(_cells(A1=[1.0, 0.5, 0.0]))
    assert rates["A1"].rate == pytest.approx(0.5)


def test_an_unrecognised_score_value_raises_instead_of_being_averaged():
    from genomics_harness import Completeness, VerdictCell  # noqa: F401

    with pytest.raises(ValueError):
        _cells(A1=[1.0, "probably"])


# -- the round trip, end to end -------------------------------------------


def _stored_log(instances, tmp_path):
    log = inspect_eval(
        mock_task(instances, epochs=3),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    return read_log(log.location)


def test_our_analysis_of_a_stored_log_matches_the_in_run_meaning(instances, tmp_path):
    """The rates read back off disk are the rates the run actually produced."""
    stored = _stored_log(instances, tmp_path)
    rates = judge_verdict_rates(verdict_table(stored))

    # instance one reaches A1 and B1, instance two reaches D1; three epochs each
    assert rates["A1"].scored == 3
    assert rates["A1"].not_reachable == 3
    assert rates["A1"].rate == 1.0
    assert rates["D1"].scored == 3
    assert rates["D1"].not_reachable == 3
    assert rates["D1"].rate == 1.0


def test_no_stored_null_is_ever_counted_as_a_failure(instances, tmp_path):
    stored = _stored_log(instances, tmp_path)
    cells = verdict_table(stored)
    nulls = sum(
        1
        for sample in stored.samples or []
        for score in (sample.scores or {}).values()
        for value in score.value.values()
        if value is None
    )
    unreachable = sum(1 for c in cells if c.outcome is Outcome.NOT_REACHABLE)
    not_performed = sum(1 for c in cells if c.outcome is Outcome.NOT_PERFORMED)
    assert nulls == unreachable == 9
    assert not_performed == 0


def test_framework_recompute_metrics_still_coerces_none_to_zero(instances, tmp_path):
    """Pins the framework behaviour the harness routes around.

    If this fails, upstream has changed how a stored `null` is aggregated. That
    is good news, but it means `design/foundations.md` and the handoff's §6.2
    invariant 1 need re-reading before anything relies on the new behaviour.
    """
    stored = _stored_log(instances, tmp_path)

    in_run_means = {
        score.name: score.metrics["mean"].value for score in (stored.results.scores)
    }
    unscored = {score.name: score.unscored_samples for score in stored.results.scores}
    assert unscored["A1"] == 3
    assert in_run_means["A1"] == 1.0

    recompute_metrics(stored)
    after = {score.name: score.metrics["mean"].value for score in stored.results.scores}
    after_unscored = {
        score.name: score.unscored_samples for score in stored.results.scores
    }

    # the unreachable half is now scored, and scored as failure
    assert after_unscored["A1"] == 0
    assert after["A1"] == pytest.approx(0.5)
    assert after["A1"] != in_run_means["A1"]

    # and our own reading of the same log is unmoved
    assert judge_verdict_rates(verdict_table(stored))["A1"].rate == 1.0


def test_nan_in_memory_and_none_on_disk_agree(instances, tmp_path):
    stored = _stored_log(instances, tmp_path)
    for sample in stored.samples or []:
        for score in (sample.scores or {}).values():
            for value in score.value.values():
                if value is None:
                    assert outcome_of_score_value(value) is outcome_of_score_value(
                        float("nan")
                    )
                else:
                    assert not math.isnan(value)
