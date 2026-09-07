"""The log-reading layer — the raw table, the pooled view, and the shared rules.

`mockllm/model` throughout; no network, no keys.

**This module has no panel and reports no number.** The per-evaluation
aggregation it used to cover retired at v40 with `analysis.py`, and
``test_report.py`` carries every rule that moved with it (§14.34, §0 v40). What
is asserted here is the layer `genomics_harness.report` and
`genomics_harness.scoring` both read stored logs through, and the raw readers
below it that answer what a single judge returned.

Five properties:

1. **The raw table** flattens one log's dict-valued scores and refuses a scalar.
   It is the unresolved view: its ``not_reachable`` is the score value's own
   reading, and nothing reported is built on it.
2. **The pooled per-judge view** — not a reported number, kept because it is the
   only reader that works on a log whose scorer does not follow the judge
   metadata contract, and because runtime invariant 7 is asserted through it.
3. **The resolution rule**, on hand-built votes: two agreeing voters resolve, two
   differing voters split, a lone verdict counts, and no voter is a
   non-observation whose cause is reachability rather than silence.
4. **One judge's stored record, validated** — the cross-check that makes the
   reachability carrier checkable rather than merely stated. `report.build_matrices`
   is the caller, so a record that contradicts itself must raise here and not
   there.
5. **The module computes every number it reports itself** — a source control, not
   a behavioural one.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from mock_harness import (
    MOCK_CATALOGUE,
    clean_model,
    dense_judge,
    mock_task,
    truncating_model,
)
from mock_judging import (
    CATALOGUE,
    JudgeRecorder,
    generation_log,
    judge_model,
    verdict_reply,
)

from genomics_harness import log_reading
from genomics_harness.instances import InstanceSet
from genomics_harness.judge import JUDGE_NAMES
from genomics_harness.judge_prompt import JUDGE_PROMPT_VERSION
from genomics_harness.log_reading import (
    REACHABILITY_KEY,
    JudgeReading,
    Resolution,
    VerdictCell,
    judge_reading,
    judge_verdict_rates,
    read_log,
    resolve_cell,
    verdict_table,
)
from genomics_harness.provenance import JudgeConfig
from genomics_harness.scoring import run_judges
from genomics_harness.trajectory import Completeness
from genomics_harness.verdicts import Outcome

KIN = "a1b2c3d4e5f60718"
PANEL_ID = "0f1e2d3c4b5a6978"
TRUNC = "9c8b7a6d5e4f3021"

KIN_REACHABLE = ["B6", "B5", "B1", "D4"]
PANEL_REACHABLE = ["A1"]

FULL_VERDICTS = {
    "B6": "performed",
    "B5": "partial",
    "B1": "not_performed",
    "D4": "performed",
    "A1": "performed",
}
"""One reply covering every check any instance reaches, so the same canned judge
works on every sample."""


def _configs(**efforts: str) -> tuple[JudgeConfig, ...]:
    """A panel of three mock judges, at the efforts named."""
    return tuple(
        JudgeConfig(
            name=name,
            model="mockllm/model",
            reasoning_effort=efforts.get(name, "xhigh"),  # type: ignore[arg-type]
            prompt_version=JUDGE_PROMPT_VERSION,
        )
        for name in JUDGE_NAMES
    )


def _score(
    generation: EvalLog,
    scoring_dir: Path,
    replies: Mapping[str, Mapping[str, str] | str],
    *,
    configs: tuple[JudgeConfig, ...] | None = None,
) -> dict[str, str]:
    """Judge one generation log with a chosen reply per judge.

    Args:
        generation: The log to score.
        scoring_dir: Where the scoring logs go.
        replies: Judge name to either a verdict mapping or a raw reply string —
            a string that is not a verdict object is how a whole-reply parse
            failure is produced.
        configs: The panel to record in the sidecar.

    Returns:
        ``ScoringRun.scoring_logs``.
    """
    recorder = JudgeRecorder()
    roles: dict[str, Any] = {
        name: judge_model(
            recorder,
            verdict_reply(reply) if isinstance(reply, Mapping) else reply,
        )
        for name, reply in replies.items()
    }
    run = run_judges(
        generation,
        catalogue=CATALOGUE,
        scoring_dir=scoring_dir,
        configs=configs or _configs(),
        model="mockllm/model",
        model_roles=roles,
    )
    return run.scoring_logs


def _judge_metadata(log: EvalLog, judge: str, sample_id: str) -> dict[str, Any]:
    """One judge's `Score.metadata` on one stored trajectory, for mutation.

    The tests that check a refusal have to corrupt the record the refusal reads,
    and this is the one place that reaches into a stored score to do it.
    """
    sample = next(s for s in log.samples or [] if str(s.id) == sample_id)
    score = (sample.scores or {})[judge]
    assert score.metadata is not None
    return score.metadata


# -- 1. the raw table ------------------------------------------------------


def test_the_table_has_one_row_per_trajectory_judge_check(instances, tmp_path):
    log = inspect_eval(
        mock_task(instances, epochs=2),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    stored = read_log(log.location)
    cells = verdict_table(stored)
    assert len(cells) == 2 * 2 * len(MOCK_CATALOGUE)  # instances × epochs × checks
    assert {c.judge for c in cells} == {"dense_judge"}


def test_two_judges_produce_two_rows_per_check(instances, tmp_path):
    log = inspect_eval(
        mock_task(
            instances,
            scorers=[
                dense_judge("judge_a", MOCK_CATALOGUE.to_arg()),
                dense_judge("judge_b", MOCK_CATALOGUE.to_arg()),
            ],
            name="two-judges",
        ),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    stored = read_log(log.location)
    cells = verdict_table(stored)
    judges = {c.judge for c in cells}
    assert judges == {"dense_judge", "dense_judge1"}, (
        "score keys derive from the factory name, not its arguments — three "
        "separately named factories are needed for self-describing keys"
    )


def test_a_scalar_score_is_refused(instances, tmp_path):
    """The table only makes sense over dict-valued scores."""
    log = inspect_eval(
        mock_task(instances),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    stored: EvalLog = read_log(log.location)
    stored.samples[0].scores["dense_judge"].value = 1.0
    with pytest.raises(ValueError, match="not a dict of per-check verdicts"):
        verdict_table(stored)


def test_the_raw_table_does_not_separate_the_three_causes_of_a_null(
    instances: InstanceSet, tmp_path: Path
) -> None:
    """Why nothing reported is built on :class:`VerdictCell`.

    A truncated trajectory's cells and an unreachable check's cells read the
    same `not_reachable` off the score value alone. The separation lives in
    `Score.metadata`, which the panel reads and this view does not.
    """
    log = inspect_eval(
        mock_task(instances, epochs=1, name="raw-nulls"),
        model=truncating_model(),
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    cells = verdict_table(read_log(log.location))
    assert cells, "the fixture produced no verdicts; this test measures nothing"
    unreachable = [c for c in cells if c.outcome is Outcome.NOT_REACHABLE]
    assert all(c.excluded_reason is not None for c in cells), (
        "the fixture is meant to be a truncated run"
    )
    # the raw view offers exactly one reading of a null, and it is not a cause
    assert all(not c.reachable for c in unreachable)


# readers, renamed from `check_rates` / `disagreement_rates` so that the plain
# names belong to the reported pair. Every assertion below is the one it carried
# under the old name; runtime invariant 7 is asserted through the same functions
# in `test_invariant_7_none_coercion.py`.


def _cell_for(
    check_id,
    outcome,
    *,
    judge="j1",
    sample="s1",
    epoch=1,
    completeness=Completeness.CLEAN,
):
    return VerdictCell(
        sample_id=sample,
        epoch=epoch,
        judge=judge,
        check_id=check_id,
        outcome=outcome,
        completeness=completeness,
    )


def test_incomplete_trajectories_are_excluded_by_default():
    cells = [
        _cell_for("A1", Outcome.PERFORMED),
        _cell_for(
            "A1", Outcome.NOT_PERFORMED, epoch=2, completeness=Completeness.TRUNCATED
        ),
    ]
    rates = judge_verdict_rates(cells)
    assert rates["A1"].scored == 1
    assert rates["A1"].excluded_incomplete == 1
    assert rates["A1"].rate == 1.0


def test_incomplete_trajectories_can_be_included_deliberately():
    cells = [
        _cell_for("A1", Outcome.PERFORMED),
        _cell_for(
            "A1", Outcome.NOT_PERFORMED, epoch=2, completeness=Completeness.TRUNCATED
        ),
    ]
    rates = judge_verdict_rates(cells, exclude_incomplete=False)
    assert rates["A1"].scored == 2
    assert rates["A1"].rate == 0.5


def test_only_a_clean_trajectory_is_scored():
    """Replaces `test_a_continued_trajectory_is_still_scoreable`, which asserted
    that a `CONTINUED` trajectory counted toward the rate.

    That was the load-bearing consequence of the continuation policy and the one
    the reversal is about: a continuation that confabulates produces a trajectory
    that looks complete and **gets scored**, carrying reasoning steps that never
    happened (§0 v7). Every outcome but `CLEAN` is now excluded, and each is
    counted under its own cause.
    """
    for completeness in Completeness:
        rates = judge_verdict_rates(
            [_cell_for("A1", Outcome.PERFORMED, completeness=completeness)]
        )
        scored = completeness is Completeness.CLEAN
        assert rates["A1"].scored == (1 if scored else 0), completeness
        assert rates["A1"].excluded_incomplete == (0 if scored else 1), completeness


def test_rates_can_be_computed_per_judge():
    cells = [
        _cell_for("A1", Outcome.PERFORMED, judge="a"),
        _cell_for("A1", Outcome.NOT_PERFORMED, judge="b"),
    ]
    assert judge_verdict_rates(cells, judges=["a"])["A1"].rate == 1.0
    assert judge_verdict_rates(cells, judges=["b"])["A1"].rate == 0.0
    assert judge_verdict_rates(cells)["A1"].rate == 0.5


# -- 2. Decision A, on hand-built votes ------------------------------------


@pytest.mark.parametrize(
    ("outcome", "band"),
    [(Outcome.PERFORMED, 1.0), (Outcome.PARTIAL, 0.5), (Outcome.NOT_PERFORMED, 0.0)],
)
def test_two_agreeing_voters_are_unanimous_and_carry_the_band(
    outcome: Outcome, band: float
) -> None:
    resolution, value = resolve_cell({"a": outcome, "b": outcome}, reachable=True)
    assert resolution is Resolution.RESOLVED
    assert value == band


def test_two_voters_that_differ_are_a_split_and_carry_no_band() -> None:
    resolution, band = resolve_cell(
        {"a": Outcome.PERFORMED, "b": Outcome.PARTIAL, "c": Outcome.PERFORMED},
        reachable=True,
    )
    assert resolution is Resolution.SPLIT
    assert band is None


def test_a_lone_verdict_counts(tmp_path: Path) -> None:
    """The two-voter floor is **withdrawn** (§0 v34).

    It was written under a fixed panel of three, and under it a single-judge
    evaluation reports nothing at all. One judge reading one trajectory is an
    observation; the panel size is carried beside the count so a number resting
    on one judge for half its cells says so.

    Asserted on the rule and on the vocabulary, because the withdrawn rule was
    expressed in both: a category that still exists is a category something can
    still be routed to.
    """
    resolution, band = resolve_cell({"a": Outcome.PERFORMED}, reachable=True)
    assert resolution is Resolution.RESOLVED
    assert band == 1.0
    assert not hasattr(Resolution, "UNDER_OBSERVED"), (
        "the withdrawn category is still reachable"
    )
    assert "under_observed" not in {r.value for r in Resolution}


def test_no_voter_on_a_reachable_check_is_unobserved_reachable() -> None:
    resolution, band = resolve_cell({}, reachable=True)
    assert resolution is Resolution.UNOBSERVED_REACHABLE
    assert band is None


def test_no_voter_on_an_unreachable_check_is_unreachable() -> None:
    """The same empty vote dict, two different facts — which is why
    ``reachable`` is an argument and not something read off the votes."""
    resolution, band = resolve_cell({}, reachable=False)
    assert resolution is Resolution.UNREACHABLE
    assert band is None


# -- 4. one judge's stored record, validated -------------------------------
#
# `judge_reading` was only ever exercised through `verdict_panel` before v40.
# With the panel retired its caller is `report.build_matrices`, so the record
# validation is asserted here on its own — the unit it is, rather than through
# whichever layer happens to call it (§15).


def _one_judge(tmp_path: Path, judge: str = "judge_opus_a") -> tuple[EvalLog, Any]:
    """One scored log and the KIN sample inside it."""
    locations = _score(
        generation_log(tmp_path),
        tmp_path / "scoring",
        dict.fromkeys(JUDGE_NAMES, FULL_VERDICTS),
    )
    log = read_log(locations[judge])
    sample = next(s for s in log.samples or [] if str(s.id) == KIN)
    return log, sample


def test_a_valid_record_reads_its_four_facts(tmp_path: Path) -> None:
    """The happy path, stated so the refusals below are refusals of something."""
    _, sample = _one_judge(tmp_path)
    reading = judge_reading("judge_opus_a", sample, CATALOGUE, "where")

    assert isinstance(reading, JudgeReading)
    assert reading.judge == "judge_opus_a"
    assert set(reading.outcomes) == set(CATALOGUE.ids), "dense over the catalogue"
    assert set(reading.reachable) == set(KIN_REACHABLE)
    assert reading.missing == frozenset()
    assert not reading.parse_failed


def test_reachability_is_read_from_the_carrier_and_not_from_the_nulls(
    tmp_path: Path,
) -> None:
    """The carrier is the source; the NaN pattern is only ever a cross-check.

    Inferring reachability from which verdicts came back null would read a
    judge's silence as a property of the task.
    """
    log, sample = _one_judge(tmp_path)
    reading = judge_reading("judge_opus_a", sample, CATALOGUE, "where")
    assert set(reading.reachable) == set(KIN_REACHABLE)

    unreachable = set(CATALOGUE.ids) - set(KIN_REACHABLE)
    assert unreachable, "the fixture instance must leave some check unreachable"
    for check_id in unreachable:
        assert reading.outcomes[check_id] is Outcome.NOT_REACHABLE

    # and the carrier, not the nulls, is what moves the answer
    _judge_metadata(log, "judge_opus_a", KIN)[REACHABILITY_KEY] = list(KIN_REACHABLE)[
        :2
    ]
    with pytest.raises(ValueError, match="contradicts its own metadata"):
        judge_reading("judge_opus_a", sample, CATALOGUE, "where")


def test_a_record_with_no_reachability_carrier_is_refused(tmp_path: Path) -> None:
    """A parse failure still carries it, so its absence is a broken judge."""
    log, sample = _one_judge(tmp_path)
    del _judge_metadata(log, "judge_opus_a", KIN)[REACHABILITY_KEY]
    with pytest.raises(ValueError, match=REACHABILITY_KEY):
        judge_reading("judge_opus_a", sample, CATALOGUE, "where")


def test_a_judge_whose_verdicts_contradict_its_own_metadata_is_refused(
    tmp_path: Path,
) -> None:
    """`missing_verdicts` says the check was not answered; the value says it was.

    One of the two is wrong and nothing on disk says which, so the cross-check
    raises rather than picking. This is also what makes the carrier claim
    checkable rather than merely stated.
    """
    log, sample = _one_judge(tmp_path)
    _judge_metadata(log, "judge_opus_a", KIN)["missing_verdicts"] = ["B6"]
    with pytest.raises(ValueError, match="contradicts its own metadata"):
        judge_reading("judge_opus_a", sample, CATALOGUE, "where")


def test_a_score_not_dense_over_the_catalogue_is_refused(tmp_path: Path) -> None:
    """A ragged key set silently drops a check or errors the run, depending on
    which sample hits it first (§5)."""
    log, sample = _one_judge(tmp_path)
    score = (sample.scores or {})["judge_opus_a"]
    assert isinstance(score.value, dict)
    score.value.pop("B6")
    with pytest.raises(ValueError, match="not dense over catalogue"):
        judge_reading("judge_opus_a", sample, CATALOGUE, "where")


def test_an_absent_score_is_refused_rather_than_treated_as_silence(
    tmp_path: Path,
) -> None:
    """ "The judge failed" and "the judge never ran" are different facts (§5)."""
    _, sample = _one_judge(tmp_path)
    with pytest.raises(ValueError, match="has no score"):
        judge_reading("judge_sonnet_xhigh", sample, CATALOGUE, "where")


# -- 5. the module computes every number it reports itself -----------------


def test_the_log_reading_layer_reads_no_framework_aggregation() -> None:
    """A control on the module, not on one run of it.

    §13.4: over a stored log the framework's aggregation turns every unreachable
    and every excluded check into a scored zero, because NaN reads back as
    ``None`` and `value_to_float` falls through to ``0.0``. A behavioural test
    shows this layer did not use it *this time*; this shows it cannot. Modelled
    on the spend readers' control in ``test_scoring_driver.py``, with the anchor
    asserted first so a rename disables the control loudly.
    """
    source = Path(log_reading.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert {
        "verdict_table",
        "resolve_cell",
        "judge_reading",
        "VerdictCell",
    } <= defined, (
        "the log-reading layer's entry points were renamed; this control no "
        "longer covers them"
    )

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for aggregate in (
        "recompute_metrics",
        "EvalResults",
        "EvalScore",
        "EvalMetric",
        "value_to_float",
        "metric",
        "mean",
    ):
        assert aggregate not in imported, (
            f"the log-reading layer imports {aggregate!r}; it computes every "
            "reported number itself"
        )

    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for aggregate in ("results", "metrics", "reductions"):
        assert aggregate not in attributes, (
            f"the log-reading layer reads `.{aggregate}`, which is the framework's "
            "own aggregation over a log whose nulls it read as zeros"
        )
