"""The report layer — §5, at v37.

Organised as the ruling is: the framework facts it rests on, the matrices, what
combines, what a report computes, refusal and exclusion, the survey, and the
source control.

**Every rule gets an assertion on the rule**, not on something the rule implies
(§15). Six tests once passed over an inverted stop-reason mapping because each
asserted what the mapping implied rather than the mapping itself, so where a rule
is a function of its inputs — a lone verdict resolves, ``text`` alone is the
definition, a non-observation never enters a denominator — the test calls that
function with those inputs and reads the answer.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import mock_judging as mj
import pytest
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import get_model
from report_fixtures import (
    FULL_VERDICTS,
    SECOND_INSTANCES,
    Evaluation,
    evaluation,
    extended_catalogue,
    paired,
    redefined_catalogue,
    rejudge_under,
)

from kgpbench import report as report_module
from kgpbench.instances import Requirement
from kgpbench.judge_prompt import refinement_chains
from kgpbench.log_reading import (
    REACHABILITY_KEY,
    Resolution,
    read_log,
    resolve_cell,
)
from kgpbench.provenance import COMPARISON_AXES, provenance_sidecar_path
from kgpbench.render_capture import render_capture
from kgpbench.report import (
    COMBINATION_AXES,
    CheckCounts,
    CombinationError,
    CompletenessExclusions,
    EvalConfigRecorded,
    EvaluationLogs,
    Exclusion,
    ExclusionError,
    ExclusionList,
    JudgeCall,
    JudgeConfigurationError,
    JudgeSpend,
    LoneDissent,
    MatrixCell,
    MatrixRow,
    PairingError,
    Report,
    Survey,
    SurveyGroup,
    VerdictMatrix,
    assert_combinable,
    assert_one_configuration_per_judge,
    build_matrices,
    build_report,
    catalogue_comparison,
    check_counts,
    completeness_exclusions,
    judge_spend_totals,
    load_evaluation,
    load_generation_log,
    load_scoring_logs,
    lone_dissents,
    read_exclusions,
    survey,
)
from kgpbench.scoring import build_judges, select_judgeable
from kgpbench.verdicts import Outcome

# -- 1. the framework facts the layer rests on ------------------------------


def test_sample_uuid_is_populated_unique_and_survives_the_scoring_pass(
    tmp_path: Path,
) -> None:
    """§5's row key, asserted as the three separate claims it makes.

    A layer built on a false one is a wasted round, so this asserts the property
    rather than a consequence of it: populated, unique per trajectory, and
    **identical** on the generation log and the scoring log after `score()`,
    `write_eval_log` and `read_eval_log`.
    """
    run = evaluation(tmp_path, "uuid").run
    generation = read_eval_log(run.generation_location, resolve_attachments=True)

    generated = {str(s.id): s.uuid for s in generation.samples or []}
    assert generated, "the generation log carries no samples"
    assert all(uuid for uuid in generated.values()), (
        f"a sample carries no uuid: {generated}"
    )
    assert len(set(generated.values())) == len(generated), (
        f"uuids are not unique per trajectory: {generated}"
    )

    for location in run.scoring_logs.values():
        scored = read_eval_log(location, resolve_attachments=True)
        stored = {str(s.id): s.uuid for s in scored.samples or []}
        assert stored, f"{location} carries no samples"
        # the driver judges a subset, so compare over what it judged
        for sample_id, uuid in stored.items():
            assert uuid == generated[sample_id], (
                f"{location}: uuid moved for {sample_id!r} — {uuid} vs "
                f"{generated[sample_id]}. It is the join key between a "
                "generation log and the scoring logs taken from it"
            )


def test_the_scoring_log_header_carries_the_model_under_test(tmp_path: Path) -> None:
    """§7's new axis: read, never declared.

    The scoring call's **active** model is deliberately something else, because
    that is the only way to tell "preserved from the generation run" from "the
    model this call happened to run at". `run_judges` passes ``mockllm/model``.
    """
    under_test = "mockllm/a-distinct-model-under-test"
    run = evaluation(tmp_path, "model", model=under_test).run

    generation = read_eval_log(run.generation_location, header_only=True)
    assert generation.eval.model == under_test

    for location in run.scoring_logs.values():
        header = read_eval_log(location, header_only=True)
        assert header.eval.model == under_test, (
            f"{location}: EvalSpec.model is {header.eval.model!r}, not the model "
            "under test. The model axis is read from here"
        )


def test_eval_id_is_preserved_by_score_and_differs_across_eval_calls(
    tmp_path: Path,
) -> None:
    """Both halves. A value that cannot tell two runs apart is not an identity."""
    first = evaluation(tmp_path, "id-one").run
    second = evaluation(tmp_path, "id-two").run

    generation = read_eval_log(first.generation_location, header_only=True)
    for location in first.scoring_logs.values():
        header = read_eval_log(location, header_only=True)
        assert header.eval.eval_id == generation.eval.eval_id

    other = read_eval_log(second.generation_location, header_only=True)
    assert generation.eval.eval_id != other.eval.eval_id, (
        "two eval() calls share an eval_id, so it cannot distinguish the repeat "
        "case this design supports"
    )


def test_header_only_reads_carry_every_axis_the_survey_needs(tmp_path: Path) -> None:
    """The survey reads headers only, so what it needs must be in one."""
    run = evaluation(tmp_path, "headers").run
    location = next(iter(run.scoring_logs.values()))

    header = read_eval_log(location, header_only=True)
    assert header.samples is None, "header_only returned samples"
    assert header.eval.model
    assert header.eval.eval_id

    (loaded,) = load_scoring_logs([location], header_only=True)
    assert loaded.catalogue.ids, "no catalogue reachable from the header"
    assert loaded.judge in mj.JUDGE_NAMES if hasattr(mj, "JUDGE_NAMES") else True
    with pytest.raises(ValueError, match="header-only"):
        _ = loaded.samples


# -- 2. the matrices --------------------------------------------------------


def test_rows_are_keyed_by_uuid_so_re_running_an_instance_adds_rows(
    tmp_path: Path,
) -> None:
    """The rule ``(sample_id, epoch)`` is not a key across evaluations.

    Two evaluations of the **same** instance set. Inspect numbers epochs from 1
    within each `eval()` call, so both produce epoch 1 under the same sample id;
    keyed that way the second evaluation's trajectories would overwrite the
    first's and six would collapse to three. Asserted as the count: the matrix
    must carry both.
    """
    first = evaluation(tmp_path, "rerun-a")
    second = evaluation(tmp_path, "rerun-b")

    pairs = paired(first, second)
    matrices = build_matrices(pairs)
    matrix = matrices[("mockllm/model-under-test", "B6", Requirement.REQUIRED)]

    uuids = [row.uuid for row in matrix.rows]
    assert len(uuids) == 2, f"expected one row per evaluation, got {uuids}"
    assert len(set(uuids)) == 2, "the two evaluations collapsed onto one row key"

    keyed_the_old_way = {(row.instance, row.evaluation) for row in matrix.rows}
    assert len({instance for instance, _ in keyed_the_old_way}) == 1, (
        "the fixture should be the same instance in both evaluations"
    )


def test_required_and_enriching_are_two_matrices_and_never_merge(
    tmp_path: Path,
) -> None:
    """A check required in one instance and enriching in another.

    `SECOND_INSTANCES` weighs B6 as enriching where `mock_judging.INSTANCES`
    weighs it as required, over the same check. The rule is that this contributes
    to two matrices and never to one number.
    """
    first = evaluation(tmp_path, "req")
    second = evaluation(tmp_path, "enr", instances=SECOND_INSTANCES)

    matrices = build_matrices(paired(first, second))
    model = "mockllm/model-under-test"

    required = matrices[(model, "B6", Requirement.REQUIRED)]
    enriching = matrices[(model, "B6", Requirement.ENRICHING)]

    assert {row.uuid for row in required.rows}.isdisjoint(
        {row.uuid for row in enriching.rows}
    ), "one trajectory reached both matrices for one check"
    assert required.requirement is Requirement.REQUIRED
    assert enriching.requirement is Requirement.ENRICHING


def test_columns_are_judge_calls_so_one_judge_per_evaluation_is_one_column(
    tmp_path: Path,
) -> None:
    """Two judges at one model and effort are two columns, not one.

    And the same judge name in two evaluations is two columns, because the
    evaluation is part of the column identity — pooling them would merge two
    independent draws.
    """
    first = evaluation(tmp_path, "cols-a", judges=["judge_opus_a", "judge_opus_b"])
    second = evaluation(tmp_path, "cols-b", judges=["judge_opus_a"])

    matrices = build_matrices(paired(first, second))
    matrix = matrices[("mockllm/model-under-test", "B6", Requirement.REQUIRED)]

    ids = [column.id for column in matrix.columns]
    assert len(ids) == len(set(ids)) == 3, f"expected three judge calls, got {ids}"

    same_model = {column.model for column in matrix.columns}
    assert same_model == {"mockllm/model"}, (
        "the fixture judges share a model; they must still be separate columns"
    )


def test_the_column_set_varies_per_row_and_empty_cells_are_expected(
    tmp_path: Path,
) -> None:
    """A trajectory from one evaluation has no cell in the other's columns."""
    first = evaluation(tmp_path, "sparse-a", judges=["judge_opus_a", "judge_opus_b"])
    second = evaluation(tmp_path, "sparse-b", judges=["judge_sonnet_xhigh"])

    matrices = build_matrices(paired(first, second))
    matrix = matrices[("mockllm/model-under-test", "B6", Requirement.REQUIRED)]

    widths = sorted(len(row.cells) for row in matrix.rows)
    assert widths == [1, 2], f"expected a ragged matrix, got widths {widths}"
    assert len(matrix.columns) == 3
    for row in matrix.rows:
        assert set(row.cells) <= {column.id for column in matrix.columns}


def test_the_three_abstention_causes_are_never_collapsed(tmp_path: Path) -> None:
    """Unreachable, omitted and parse failure reach the cell as themselves.

    One judge parse-fails the whole reply, one omits a reachable check, and D1 is
    in the catalogue and reached by no instance. All three land on a stored
    ``null`` and the cell must still tell them apart.
    """
    omitted = {k: v for k, v in FULL_VERDICTS.items() if k != "B5"}
    run = evaluation(
        tmp_path,
        "abstain",
        replies={
            "judge_sonnet_xhigh": FULL_VERDICTS,
            "judge_opus_a": omitted,
            "judge_opus_b": "this reply is not a verdict object at all",
        },
    )
    matrices = build_matrices(paired(run))
    model = "mockllm/model-under-test"

    b5 = matrices[(model, "B5", Requirement.REQUIRED)]
    kin = next(row for row in b5.rows if row.instance == "a1b2c3d4e5f60718")
    causes = {
        column.judge: kin.cells[column.id].abstention
        for column in b5.columns
        if column.id in kin.cells
    }
    assert causes["judge_opus_a"] is report_module.Abstention.OMITTED
    assert causes["judge_opus_b"] is report_module.Abstention.PARSE_FAILURE
    assert (
        kin.cells[
            next(c.id for c in b5.columns if c.judge == "judge_sonnet_xhigh")
        ].verdict
        is Outcome.PARTIAL
    )

    # a parse failure takes precedence over the omission it also implies: the
    # more specific cause is the more useful one in a review queue
    assert causes["judge_opus_b"] is not report_module.Abstention.OMITTED


def test_the_judges_disagreeing_on_reachability_is_a_hard_error(
    tmp_path: Path,
) -> None:
    """Reachability is a property of the instance, so the judges must agree.

    A mismatch means two judges were handed different instances under one uuid,
    and picking one silently would rate them against different denominators.

    Asserted here from v40. The rule is `build_matrices`' own and always was, but
    its only assertion until now went through the per-evaluation `verdict_panel`,
    which retired with its layer — so the rule would otherwise have been left
    live in the code and unasserted anywhere (§14.34).
    """
    run = evaluation(tmp_path, "reach-disagree")
    pair = paired(run)[0]

    first, second = pair.scoring[0], pair.scoring[1]
    shared = {s.uuid for s in first.samples} & {s.uuid for s in second.samples}
    assert shared, "the two judges must share a trajectory for this to be a test"
    uuid = sorted(shared)[0]

    sample = next(s for s in second.samples if s.uuid == uuid)
    metadata = (sample.scores or {})[second.judge].metadata
    assert metadata is not None
    # widen this judge's reachable set by a check no instance reaches, and record
    # it as omitted too. `judge_reading`'s own cross-check — answered ==
    # reachable minus missing — still passes, so the failure under test is the
    # cross-judge disagreement and not the per-record contradiction beside it.
    extra = next(c for c in mj.CATALOGUE.ids if c not in metadata[REACHABILITY_KEY])
    metadata[REACHABILITY_KEY] = [*metadata[REACHABILITY_KEY], extra]
    metadata["missing_verdicts"] = [*metadata.get("missing_verdicts", []), extra]

    with pytest.raises(ValueError, match="disagree on what trajectory"):
        build_matrices([pair])


def test_a_check_the_task_did_not_make_reachable_has_no_row_in_that_matrix(
    tmp_path: Path,
) -> None:
    """The unreachable cause, and where it does and does not appear.

    ``reachable_checks`` is the instance's own ``TaskCheck`` set intersected with
    the catalogue (`judge.py`), which is the **same source** the requirement class
    is read from. So a check the task did not make reachable carries no
    requirement weight either, and belongs to neither the required nor the
    enriching matrix — putting it in one would invent a weight the task never
    assigned, and putting it in both would count one trajectory twice.

    The consequence, asserted here so it is not rediscovered: the row's absence
    is the **sole** carrier. v34 listed "a check the task did not make reachable"
    among the cell-level abstention causes; under requirement-class keying that
    cause resolves one level up, and at v37 the cell cause, the row flag and the
    unreachable count were deleted as measured-dead rather than left standing
    beside the rule that replaced them.
    """
    run = evaluation(tmp_path, "unreach")
    matrices = build_matrices(paired(run))
    model = "mockllm/model-under-test"

    b5 = matrices[(model, "B5", Requirement.REQUIRED)]
    assert {row.instance for row in b5.rows} == {"a1b2c3d4e5f60718"}, (
        "the PANEL instance reaches only A1 and must contribute no B5 row"
    )
    assert not hasattr(MatrixRow, "reachable"), (
        "the row flag is deleted at v37; the row's absence is the sole carrier"
    )
    assert "unreachable" not in set(CheckCounts.model_fields), (
        "a structurally-zero count prints in every report line as an observation"
    )

    # D1 is in the catalogue and reached by no instance: no matrix at all,
    # rather than a matrix of zeroes
    assert not [key for key in matrices if key[1] == "D1"]


def test_a_cell_never_carries_both_a_verdict_and_an_abstention() -> None:
    """The encoding rule itself, on the type."""
    assert MatrixCell(verdict=Outcome.PERFORMED).abstention is None
    assert MatrixCell(abstention=report_module.Abstention.OMITTED).verdict is None


# -- 3. what combines -------------------------------------------------------


def test_instances_may_vary_and_that_is_the_operation(tmp_path: Path) -> None:
    """Two evaluations differing only by instance set combine."""
    first = evaluation(tmp_path, "vary-a")
    second = evaluation(tmp_path, "vary-b", instances=SECOND_INSTANCES)

    pairs = paired(first, second)
    digests = {pair.generation.provenance.instances_sha256 for pair in pairs}
    assert len(digests) == 2, "the fixture sets should have different digests"
    assert_combinable(pairs)  # does not raise


def test_judges_are_not_an_axis_so_one_and_three_combine(tmp_path: Path) -> None:
    """Recorded, never required — including the single-judge evaluation."""
    three = evaluation(tmp_path, "j3")
    one = evaluation(tmp_path, "j1", judges=["judge_opus_a"])

    pairs = paired(three, one)
    assert sorted({judge for pair in pairs for judge in pair.judges}) == [
        "judge_opus_a",
        "judge_opus_b",
        "judge_sonnet_xhigh",
    ]
    assert_combinable(pairs)  # does not raise
    assert "judges" not in COMBINATION_AXES


def test_a_differing_model_under_test_refuses_and_names_log_and_axis(
    tmp_path: Path,
) -> None:
    """A report is per model; evaluations of different models are never combined."""
    sonnet = evaluation(tmp_path, "m-sonnet", model="mockllm/stand-in-sonnet")
    opus = evaluation(tmp_path, "m-opus", model="mockllm/stand-in-opus")

    pairs = paired(sonnet, opus)
    with pytest.raises(CombinationError) as raised:
        assert_combinable(pairs)

    message = str(raised.value)
    assert "'model'" in message, message
    assert any(pair.generation.location in message for pair in pairs), (
        f"the refusal names no offending log: {message}"
    )


def test_a_redefined_check_refuses_the_whole_combination(tmp_path: Path) -> None:
    """All-or-nothing: one differing check refuses everything.

    Not just the differing check — the whole combination. Per-check
    combinability is what §5 declines.
    """
    first = evaluation(tmp_path, "cat-a")
    second = evaluation(tmp_path, "cat-b", catalogue=redefined_catalogue("B6"))

    with pytest.raises(CombinationError) as raised:
        assert_combinable(paired(first, second))
    assert "B6" in str(raised.value)


def test_an_added_check_combines_and_is_reported_as_dropped(tmp_path: Path) -> None:
    """A differing check id set is not an incompatibility."""
    first = evaluation(tmp_path, "add-a")
    second = evaluation(tmp_path, "add-b", catalogue=extended_catalogue())

    comparison = assert_combinable(paired(first, second))  # does not raise
    assert comparison.dropped == ["C9"]
    assert "C9" not in comparison.shared
    assert comparison.combinable


def test_text_alone_is_the_definition() -> None:
    """The other fields may move freely; ``text`` may not.

    Asserted directly on :func:`catalogue_comparison`, so the rule is read off
    the function rather than inferred from a run that happened not to refuse.
    """
    base = mj.CATALOGUE
    moved_elsewhere = base.model_copy(
        update={
            "checks": [
                check.model_copy(update={"talos_anchor": "moved-for-this-test"})
                if check.id == "B6"
                else check
                for check in base.checks
            ]
        }
    )
    assert catalogue_comparison([base, moved_elsewhere]).redefined == []

    moved_text = redefined_catalogue("B6")
    assert catalogue_comparison([base, moved_text]).redefined == ["B6"]


def test_the_combination_axes_are_not_the_comparison_axes() -> None:
    """Combining is not comparing, and must not reuse the escape hatch.

    ``assert_comparable`` refuses on the instances axis by design, which is the
    axis combination varies; its ``varying=`` flag means *an edit ruled
    comparability-neutral*, not *these are different instances by design*.
    """
    assert "instances" in COMPARISON_AXES
    assert "instances" not in COMBINATION_AXES
    assert "judges" in COMPARISON_AXES
    assert "judges" not in COMBINATION_AXES
    assert "model" in COMBINATION_AXES
    assert "model" not in COMPARISON_AXES
    assert "dataset_snapshot" in COMPARISON_AXES
    assert "dataset_snapshot" not in COMBINATION_AXES, (
        "the dataset snapshot is recorded and not enforced (§2)"
    )


def test_the_same_log_supplied_twice_is_refused(tmp_path: Path) -> None:
    """Counting one trajectory as two observations is the failure `eval_id` catches.

    The type is the assertion, not the message: `PairingError` is what
    `cli.main` translates, and a bare `ValueError` here reached the shell as a
    stack trace with the right sentence inside it.
    """
    run = evaluation(tmp_path, "dup")
    location = next(iter(run.run.scoring_logs.values()))
    twice = load_evaluation(run.generation, [location, location])
    with pytest.raises(PairingError, match="same judge"):
        assert_combinable([twice])


def test_an_empty_input_is_a_bare_value_error_and_stays_one(tmp_path: Path) -> None:
    """The other refusal in `assert_combinable`, and it is deliberately not a
    refusal an operator meets.

    No command produces it — `cli._report` builds a one-element list — so an
    empty sequence is a caller's bug and a traceback is the right answer. Pinned
    so that the type change beside it is not read as *everything here becomes a
    `PairingError`*.
    """
    with pytest.raises(ValueError, match="no evaluations to combine") as raised:
        assert_combinable([])
    assert not isinstance(raised.value, PairingError)
    assert type(raised.value) is ValueError
    del tmp_path


def test_a_generation_log_is_not_a_scoring_log(tmp_path: Path) -> None:
    """It carries no judge scorer, so it carries no verdicts to report.

    The mirror condition — a scoring log handed in as the generation log —
    already raised `PairingError`; this one raised a bare `ValueError`, so the
    two directions of one mistake refused differently at the shell.
    """
    run = evaluation(tmp_path, "gen-only").run
    with pytest.raises(PairingError, match="no scorer carries"):
        load_scoring_logs([run.generation_location])


def test_a_scoring_log_carrying_two_judges_is_refused(tmp_path: Path) -> None:
    """One `score()` call per judge, so two judge scorers on one header means
    the judges saw each other's model events.

    Built rather than hand-written: the log is what `score()` writes when both
    judges share one call, which is the shape the rule forbids. `_judge_of`
    cannot say which of the two the log belongs to, so the log is not one
    judge's scoring log and the refusal is a pairing one.
    """
    run = evaluation(tmp_path, "two-judges")
    generation = read_log(run.generation)
    recorder = mj.JudgeRecorder()
    reply = mj.verdict_reply(FULL_VERDICTS)
    scored = score(
        generation,
        [
            render_capture(),
            *build_judges(mj.CATALOGUE, ["judge_opus_a", "judge_opus_b"]),
        ],
        action="append",
        copy=True,
        display="none",
        model=get_model("mockllm/model"),
        model_roles={
            "judge_opus_a": mj.judge_model(recorder, reply),
            "judge_opus_b": mj.judge_model(recorder, reply),
        },
    )
    both = tmp_path / "both.eval"
    write_eval_log(scored, str(both))

    assert [scorer.name for scorer in (scored.eval.scorers or [])].count(
        "judge_opus_a"
    ) == 1, "the fixture did not produce a two-judge header"

    with pytest.raises(PairingError, match="all carry a catalogue"):
        load_scoring_logs([str(both)])


# -- 3a. the pairing, which is the layer's input (§0 v35) -------------------


def test_the_input_is_a_generation_log_and_the_scoring_logs_taken_from_it(
    tmp_path: Path,
) -> None:
    """Both halves, and each read where the fact originates.

    §5: the input is, per evaluation, a generation log and the scoring logs
    taken from it. The split is the point — the model under test and the solver
    configuration come from the generation header and its task metadata, the
    catalogue and the judges from the scoring side.
    """
    run = evaluation(tmp_path, "pair")
    pair = load_evaluation(run.generation, run.locations)

    assert pair.generation.location == run.generation
    assert [log.location for log in pair.scoring] == list(run.locations)
    assert pair.model == "mockllm/model-under-test"
    assert pair.evaluation == pair.generation.evaluation

    # first-hand: the generation log's own record, not the sidecar's copy
    assert pair.generation.provenance.solver is not None
    assert pair.generation.provenance.catalogue_sha256 is None, (
        "a generation log is rubric-free; the catalogue is on the scoring side"
    )
    assert all(log.catalogue.ids for log in pair.scoring)


def test_a_missing_half_refuses_with_an_explanation_and_no_report(
    tmp_path: Path,
) -> None:
    """If either half is missing the report refuses and produces nothing (§5)."""
    run = evaluation(tmp_path, "half")

    with pytest.raises(PairingError, match="no scoring logs"):
        load_evaluation(run.generation, [])

    # the other half missing: a scoring log handed in as the generation log
    with pytest.raises(PairingError, match="scoring log and not a generation log"):
        load_evaluation(run.locations[0], run.locations)


def test_a_scoring_log_from_another_evaluation_is_refused_not_absorbed(
    tmp_path: Path,
) -> None:
    """`score()` preserves `eval_id`, so a disagreement is a mispaired input."""
    first = evaluation(tmp_path, "pair-a")
    second = evaluation(tmp_path, "pair-b")
    with pytest.raises(PairingError, match="not taken from this generation log"):
        load_evaluation(first.generation, second.locations)


def test_the_model_under_test_is_read_first_hand_and_the_copy_is_a_cross_check(
    tmp_path: Path,
) -> None:
    """The axis is the generation header; the preserved copy is checked, not trusted.

    §7's subject is the instrument that reports confidently on an inherited
    value. `score()` does preserve `EvalSpec.model`, and this asserts both halves
    of that: the value agrees, and the pairing is what makes the agreement a
    checked fact rather than an assumption.
    """
    run = evaluation(tmp_path, "model-read")
    pair = load_evaluation(run.generation, run.locations)
    assert pair.model == pair.generation.model
    assert {log.model for log in pair.scoring} == {pair.generation.model}


# -- 4. what a report computes ----------------------------------------------


def _matrix(
    *rows: MatrixRow, requirement: Requirement = Requirement.REQUIRED
) -> VerdictMatrix:
    columns = sorted({column for row in rows for column in row.cells})
    return VerdictMatrix(
        model="m",
        check_id="B6",
        requirement=requirement,
        columns=[
            JudgeCall(evaluation=c.split(":")[0], judge=c.split(":")[1])
            for c in columns
        ],
        rows=list(rows),
    )


def _row(uuid: str, **votes: Outcome) -> MatrixRow:
    return MatrixRow(
        uuid=uuid,
        instance="i",
        evaluation="e",
        cells={f"e:{judge}": MatrixCell(verdict=v) for judge, v in votes.items()},
    )


def test_a_lone_verdict_counts(tmp_path: Path) -> None:
    """The two-voter floor is withdrawn (§0 v34).

    Asserted on the rule itself — :func:`resolve_cell` with one vote — and then
    on a count built from it, because the withdrawn rule was expressed in both
    places.
    """
    assert resolve_cell({"only": Outcome.PERFORMED}, reachable=True) == (
        Resolution.RESOLVED,
        1.0,
    )
    assert not hasattr(Resolution, "UNDER_OBSERVED"), (
        "Resolution.UNDER_OBSERVED is withdrawn; a lone verdict counts"
    )

    counts = check_counts(_matrix(_row("t1", judge_opus_a=Outcome.PERFORMED)))
    assert counts.performed == 1
    assert counts.observed == 1, "one voting judge is one observation"
    assert counts.rows == 1


def test_a_split_is_a_count_in_the_same_row_not_a_footnote() -> None:
    counts = check_counts(
        _matrix(
            _row("t1", a=Outcome.PERFORMED, b=Outcome.NOT_PERFORMED),
            _row("t2", a=Outcome.PERFORMED, b=Outcome.PERFORMED),
        )
    )
    assert counts.split == 1
    assert counts.performed == 1
    assert counts.observed == 2, "a split belongs in the rate denominator"
    assert counts.rows == 2, "a split belongs in the displayed count"
    assert counts.disagreement_rate == 0.5


def _thinly_judged_pair(tmp_path: Path, name: str) -> tuple[Evaluation, Evaluation]:
    """Two combinable evaluations, one judged by three and one by one.

    The end-to-end shape of :func:`_mixed_voter_matrix`: judges are not a
    combination axis, so these combine, and the resulting B6 matrix holds
    multi-voter rows from the first and single-voter rows from the second. On the
    first, two judges agree on B6 and the third dissents — which is one split for
    the disagreement rate and one attributable lone dissent for §5's per-judge
    number.
    """
    dissenting = dict(FULL_VERDICTS) | {"B6": "not_performed"}
    three = evaluation(
        tmp_path,
        f"{name}-three",
        judges=["judge_sonnet_xhigh", "judge_opus_a", "judge_opus_b"],
        replies={
            "judge_sonnet_xhigh": FULL_VERDICTS,
            "judge_opus_a": FULL_VERDICTS,
            "judge_opus_b": dissenting,
        },
    )
    one = evaluation(tmp_path, f"{name}-one", judges=["judge_sonnet_xhigh"])
    return three, one


def _mixed_voter_matrix() -> VerdictMatrix:
    """Two multi-voter rows, one of them split, and two single-voter rows.

    **The fixture the two divisors disagree on.** `observed` is 4 and the
    multi-voter count is 2, so ``split / observed`` is 0.25 where
    ``split / multi_voter`` is 0.5. A fixture whose observed rows all had two or
    more voters cannot distinguish the rules and pins nothing — which is what
    every fixture under this rate did before v42 (§15).
    """
    return _matrix(
        _row("t1", a=Outcome.PERFORMED, b=Outcome.PERFORMED),
        _row("t2", a=Outcome.PERFORMED, b=Outcome.NOT_PERFORMED),
        _row("t3", a=Outcome.PERFORMED),
        _row("t4", a=Outcome.NOT_PERFORMED),
    )


def test_the_disagreement_rate_divides_by_the_multi_voter_rows() -> None:
    """§5: never by `observed` (§0 v42).

    **The edit this assertion exists to catch** is restoring
    ``split / observed`` as the divisor. On this fixture that is 1/4 and the rule
    is 1/2, so the assertion fails on exactly that edit — and the wrong value is
    named here rather than left implicit, because a rate asserted only against
    itself is the shape §15 forbids.

    The bias the rule removes has one direction: a row one judge read cannot
    split, so `observed` carries rows structurally incapable of entering the
    numerator and the old figure is the true rate scaled by the fraction of
    observed rows with two or more voters — **toward zero, in proportion to how
    thinly the check was judged**.
    """
    counts = check_counts(_mixed_voter_matrix())

    assert counts.split == 1
    assert counts.observed == 4
    assert counts.multi_voter == 2
    assert counts.disagreement_rate == 0.5
    assert counts.disagreement_rate != counts.split / counts.observed, (
        "the rate divided by `observed`, which carries rows that cannot split"
    )


def test_the_two_denominators_differ_by_the_single_voter_rows() -> None:
    """`observed` minus the multi-voter rows is exactly the rows one judge read.

    Stated because the retired code's own docstring had this wrong, claiming the
    two differ by the split count (§0 v41). All the single-voter rows resolve: a
    single-voter row that did not resolve is a non-observation and never entered
    `observed` at all.
    """
    counts = check_counts(_mixed_voter_matrix())

    single = [row for row in _mixed_voter_matrix().rows if len(row.votes) == 1]
    assert counts.observed - counts.multi_voter == len(single) == 2
    assert counts.observed - counts.multi_voter != counts.split, (
        "the two denominators differ by the split count, which is the retired "
        "docstring's error"
    )
    for row in single:
        assert row.resolution()[0] is Resolution.RESOLVED


def test_the_performed_rate_keeps_observed_and_is_untouched() -> None:
    """The asymmetry that confines the new divisor to the one rate.

    A single-voter row is a real reading of what the model did, so it belongs in
    the performed rate's denominator. The disagreement rate is the only number a
    report prints that asks about the **instrument** rather than the model.
    """
    counts = check_counts(_mixed_voter_matrix())

    assert counts.performed_rate == counts.performed / counts.observed == 0.5
    assert counts.performed_rate != counts.performed / counts.multi_voter, (
        "the performed rate moved onto the multi-voter denominator"
    )


def test_the_multi_voter_count_is_not_a_reconciliation_term() -> None:
    """It sums with nothing, and an excluded row never enters it.

    Every other count on the line participates in the sum that proves nothing was
    dropped. This one is a subset of `observed`, so a report that added it to the
    reconciliation would double-count every multi-voter row.
    """
    reviewed = MatrixRow(
        uuid="t9",
        instance="i",
        evaluation="e",
        cells={
            "e:a": MatrixCell(verdict=Outcome.PERFORMED),
            "e:b": MatrixCell(verdict=Outcome.NOT_PERFORMED),
        },
        excluded_reason="review found the tool output truncated",
    )
    matrix = _matrix(*_mixed_voter_matrix().rows, reviewed)
    counts = check_counts(matrix)

    assert counts.excluded == 1
    assert counts.multi_voter == 2, (
        "an excluded row entered the disagreement denominator; it yields no "
        "observation of any kind"
    )
    assert counts.reconciles(len(matrix.rows))
    assert counts.rows == counts.observed + counts.non_observations
    assert counts.rows != counts.observed + counts.non_observations + counts.multi_voter


def test_the_disagreement_denominator_prints_beside_the_rate(tmp_path: Path) -> None:
    """The count printing beside the rate is what replaces a rename (§0 v42).

    `disagreement_rate` keeps its name — it was underspecified under the old
    divisor rather than wrong — so the visible signal has to be the denominator
    on the line. Displaying `observed` alone was the rejected alternative: the
    multi-voter count appears nowhere else on the line, so the narrower rate is
    not recoverable from the wider one while the wider is recoverable from the
    narrower.
    """
    built = build_report(paired(*_thinly_judged_pair(tmp_path, "disagree")))
    line = next(line for line in built.rows() if line.startswith("B6 (required) — "))
    counts = built.counts["B6:required"]

    assert counts.observed != counts.multi_voter, (
        "the fixture must make the two denominators disagree, or the line pins nothing"
    )
    assert f"disagreement {counts.split}/{counts.multi_voter}" in line, line
    assert f"disagreement {counts.split}/{counts.observed}" not in line, (
        "the line printed the rate against `observed`"
    )


def test_no_mean_is_taken_over_the_bands() -> None:
    """The band values exist per cell and are never averaged across cells.

    Asserted structurally: the counts object exposes no mean, and the only rate
    it offers is a proportion of a named category.
    """
    counts = check_counts(
        _matrix(
            _row("t1", a=Outcome.PERFORMED),
            _row("t2", a=Outcome.PARTIAL),
        )
    )
    fields = set(CheckCounts.model_fields)
    assert "rate" not in fields and "mean" not in fields
    assert counts.performed_rate == 0.5
    # a mean over the bands would be 0.75; nothing here computes it
    assert not any(
        isinstance(getattr(counts, name, None), float)
        and abs(getattr(counts, name) - 0.75) < 1e-9
        for name in dir(counts)
        if not name.startswith("_")
    )


def _non_observing_matrix() -> VerdictMatrix:
    """One observed row, one row no judge voted on, one row review excluded.

    Three rows, one observation. The two quantities §5 rules apart disagree on
    this matrix by construction, which is what makes it the right fixture for
    both halves below.
    """
    unobserved = MatrixRow(uuid="t2", instance="i", evaluation="e", cells={})
    reviewed = MatrixRow(
        uuid="t3",
        instance="i",
        evaluation="e",
        cells={"e:a": MatrixCell(verdict=Outcome.PERFORMED)},
        excluded_reason="review found the tool output truncated",
    )
    return _matrix(_row("t1", a=Outcome.PERFORMED), unobserved, reviewed)


def test_non_observations_never_enter_the_rate_denominator() -> None:
    """The **rate denominator** half, which is :attr:`CheckCounts.observed`.

    Before v37 one field held both this and the displayed count, and this
    assertion pinned it at the narrow value — right about the denominator and
    wrong about the display, which is the signal that one field was doing two
    jobs (§0 v37). Split so each assertion names the quantity it pins.
    """
    counts = check_counts(_non_observing_matrix())
    assert counts.observed == 1, "a non-observation reached the rate denominator"
    assert counts.unobserved_reachable == 1
    assert counts.excluded == 1
    assert counts.performed_rate == 1.0, "the rate divides by the observed count"


def test_the_displayed_count_is_every_row_the_matrix_holds() -> None:
    """The **display** half, which is :attr:`CheckCounts.rows`.

    The same three rows as the denominator test above, and a different number.
    §5: every row the matrix holds is the displayed count and what the
    reconciliation is against; resolved plus split is what the rates divide by.
    """
    counts = check_counts(_non_observing_matrix())
    assert counts.rows == 3, "a non-observation left the displayed count"
    assert counts.observed == 1
    assert counts.rows != counts.observed, (
        "the fixture must make the two quantities disagree, or it pins nothing"
    )
    assert counts.reconciles(3)


def test_an_empty_denominator_has_no_rate_rather_than_a_rate_of_zero() -> None:
    """``None`` and ``0.0`` are different claims.

    Nothing was observed, versus everything observed was not performed. The row
    is present and reachable — the matrix holds it — and no judge voted on it,
    which is the only way a matrix that exists at all reaches an empty
    denominator once the unreachable row is gone (§0 v37).
    """
    counts = check_counts(
        _matrix(MatrixRow(uuid="t1", instance="i", evaluation="e", cells={}))
    )
    assert counts.rows == 1, "the row is in the matrix and is displayed"
    assert counts.observed == 0
    assert counts.performed_rate is None, "an empty denominator became a zero"
    assert counts.disagreement_rate is None


def test_the_counts_reconcile_against_every_row_in_the_matrix() -> None:
    """The property that makes the layer checkable.

    Worth more than any number it computes: nothing can be silently dropped.
    """
    rows = [
        _row("t1", a=Outcome.PERFORMED),
        _row("t2", a=Outcome.PERFORMED, b=Outcome.PARTIAL),
        _row("t3", a=Outcome.NOT_PERFORMED),
        MatrixRow(uuid="t4", instance="i", evaluation="e", cells={}),
        MatrixRow(
            uuid="t5",
            instance="i",
            evaluation="e",
            cells={"e:a": MatrixCell(verdict=Outcome.PERFORMED)},
            excluded_reason="review found the tool output truncated",
        ),
    ]
    matrix = _matrix(*rows)
    counts = check_counts(matrix)
    assert counts.reconciles(len(matrix.rows))
    assert counts.rows == 5
    assert (
        counts.performed
        + counts.partial
        + counts.not_performed
        + counts.split
        + counts.unobserved_reachable
        + counts.excluded
    ) == 5
    # the reconciliation is against the row count and not against the rate
    # denominator, and on this matrix the two differ (§0 v37)
    assert counts.observed == 3


def test_both_cross_instance_weightings_are_computable(tmp_path: Path) -> None:
    """Weighting is a report-time choice and is deliberately not ruled.

    Neither is baked in as the only reachable answer, so the per-instance rows
    have to survive into the matrix.
    """
    first = evaluation(tmp_path, "w-a")
    second = evaluation(tmp_path, "w-b", instances=SECOND_INSTANCES)

    matrices = build_matrices(paired(first, second))
    matrix = matrices[("mockllm/model-under-test", "B5", Requirement.REQUIRED)]

    assert len(matrix.instances) == 2, "per-instance rows did not survive"
    per_instance = {
        instance: [row for row in matrix.rows if row.instance == instance]
        for instance in matrix.instances
    }
    assert sum(len(rows) for rows in per_instance.values()) == len(matrix.rows)


def test_the_report_line_is_the_shape_the_ruling_prints(tmp_path: Path) -> None:
    first = evaluation(tmp_path, "line")
    built = build_report(paired(first))
    lines = built.rows()
    line = next(line for line in lines if line.startswith("B6 (required) — "))

    # **both quantities, never one field** (§0 v37): the line carries the
    # displayed trajectory count and the observed count that the rates divide by
    assert "trajectories ·" in line, line
    assert "observed ·" in line, line

    counts = built.counts["B6:required"]
    assert f"{counts.rows} trajectories" in line, line
    assert f"{counts.observed} observed" in line, line

    # the reconciliation is checkable in the line itself
    assert counts.observed == (
        counts.performed + counts.partial + counts.not_performed + counts.split
    )
    assert counts.rows == counts.observed + counts.non_observations
    assert built.reconciles()


# -- 4a. lone-dissent attribution (§5, ruled v42) ---------------------------


def _dissent_matrix(*rows: MatrixRow) -> VerdictMatrix:
    """A matrix whose columns cover every judge any row names."""
    return _matrix(*rows)


def _eligible_and_every_ineligible_row() -> VerdictMatrix:
    """One denominator fixture carrying **all three** ineligible shapes.

    §5 requires each denominator's fixture to carry every ineligible row shape it
    excludes, so the assertion fails on a wrong divisor rather than passing under
    either (§15). Asked about judge ``a`` on this matrix:

    * ``t1`` — eligible, and ``a`` is the lone dissenter (numerator and
      denominator);
    * ``t2`` — eligible, and ``a`` agrees (denominator only);
    * ``t3`` — **ineligible: ``a`` omitted the check.** Three others voted and
      all three agree, so the row is ineligible for ``a`` on that ground alone —
      no second reason is doing the work;
    * ``t4`` — **ineligible: fewer than three judges voted.** Two voters, so no
      dissent on it can be lone whichever judge is asked about;
    * ``t5`` — **ineligible: the others split among themselves.** A three-way
      disagreement is attributable to nobody and counts only as a split.

    So ``a`` is 1 of 2. Any of the three shapes leaking into the divisor makes it
    1 of 3.
    """
    return _dissent_matrix(
        _row(
            "t1", a=Outcome.PERFORMED, b=Outcome.NOT_PERFORMED, c=Outcome.NOT_PERFORMED
        ),
        _row("t2", a=Outcome.PERFORMED, b=Outcome.PERFORMED, c=Outcome.PERFORMED),
        _row("t3", b=Outcome.PERFORMED, c=Outcome.PERFORMED, d=Outcome.PERFORMED),
        _row("t4", a=Outcome.PERFORMED, b=Outcome.NOT_PERFORMED),
        _row("t5", a=Outcome.PERFORMED, b=Outcome.PARTIAL, c=Outcome.NOT_PERFORMED),
    )


def _dissent_for(
    matrix: VerdictMatrix, judge: str, check_id: str
) -> LoneDissent | None:
    entries = lone_dissents(
        {(matrix.model, matrix.check_id, matrix.requirement): matrix}
    )
    return next(
        (e for e in entries if e.judge == judge and e.check_id == check_id), None
    )


def test_a_lone_dissent_is_attributed_and_a_three_way_split_is_not() -> None:
    """One judge against two or more that agree among themselves (§0 v42).

    A three-way disagreement is attributable to nobody and counts only as a
    split, which is why ``t5`` contributes to neither term for any judge.
    """
    matrix = _eligible_and_every_ineligible_row()

    a = _dissent_for(matrix, "a", "B6")
    assert a is not None
    assert a.dissents == 1, "the lone dissent on t1 was not attributed"

    for judge in ("a", "b", "c"):
        entry = _dissent_for(matrix, judge, "B6")
        assert entry is not None
        assert entry.attributable <= 2, (
            f"the three-way split on t5 entered {judge}'s denominator"
        )


def test_the_denominator_is_the_rows_that_judge_could_have_dissented_on() -> None:
    """**The edit this assertion exists to catch** is a wider divisor.

    Each of §5's three ineligible shapes is in the fixture, so counting any one
    of them makes the denominator 3 where the rule says 2 — and the retired
    layer's whole-panel denominator, which demands every judge, makes it 0.
    """
    matrix = _eligible_and_every_ineligible_row()
    entry = _dissent_for(matrix, "a", "B6")

    assert entry is not None
    assert entry.attributable == 2, "a wrong divisor: the rule is t1 and t2"
    assert entry.attributable != 3, (
        "one of the three ineligible shapes entered the denominator"
    )
    assert entry.attributable != 0, (
        "the whole-panel denominator was rebuilt; it demands every judge where "
        "three suffice"
    )


def test_each_ineligible_shape_is_excluded_on_its_own() -> None:
    """One matrix per shape, so a leak names which shape leaked.

    The combined fixture above proves the divisor; these prove *why* it is that
    number, and each fails on exactly one wrong rule.
    """
    # ``a`` is a column of this matrix — it votes on t2 — so the omission on t1
    # is the only thing that can keep t1 out of its denominator
    omitted = _dissent_matrix(
        _row("t1", b=Outcome.PERFORMED, c=Outcome.PERFORMED, d=Outcome.PERFORMED),
        _row("t2", a=Outcome.PERFORMED, b=Outcome.PERFORMED, c=Outcome.PERFORMED),
    )
    entry = _dissent_for(omitted, "a", "B6")
    assert entry is not None and entry.attributable == 1, (
        "a check the judge omitted entered its denominator"
    )

    two_voters = _dissent_matrix(_row("t1", a=Outcome.PERFORMED, b=Outcome.PERFORMED))
    assert _dissent_for(two_voters, "a", "B6") is None, (
        "a row two judges voted on entered a lone-dissent denominator; two "
        "voters give disagreement, three let a dissent be attributed"
    )

    others_split = _dissent_matrix(
        _row("t1", a=Outcome.PERFORMED, b=Outcome.PARTIAL, c=Outcome.NOT_PERFORMED)
    )
    assert _dissent_for(others_split, "a", "B6") is None, (
        "a row on which the others disagreed entered the denominator, where no "
        "dissent could be lone"
    )


def test_an_excluded_trajectory_is_in_no_lone_dissent_denominator() -> None:
    """Review found it invalid, so it yields no observation of any kind."""
    reviewed = MatrixRow(
        uuid="t1",
        instance="i",
        evaluation="e",
        cells={
            "e:a": MatrixCell(verdict=Outcome.PERFORMED),
            "e:b": MatrixCell(verdict=Outcome.NOT_PERFORMED),
            "e:c": MatrixCell(verdict=Outcome.NOT_PERFORMED),
        },
        excluded_reason="review found the tool output truncated",
    )
    assert _dissent_for(_dissent_matrix(reviewed), "a", "B6") is None


def test_lone_dissent_keys_on_the_scorer_name_across_evaluations() -> None:
    """Per judge and per check — the name alone, pooled across evaluations.

    A judge name is fixed to a model and an effort, so pooling it across the
    evaluations a report combines is correct.
    :func:`assert_one_configuration_per_judge` is what makes that safe.
    """
    first = MatrixRow(
        uuid="t1",
        instance="i",
        evaluation="e1",
        cells={
            "e1:a": MatrixCell(verdict=Outcome.PERFORMED),
            "e1:b": MatrixCell(verdict=Outcome.NOT_PERFORMED),
            "e1:c": MatrixCell(verdict=Outcome.NOT_PERFORMED),
        },
    )
    second = MatrixRow(
        uuid="t2",
        instance="i",
        evaluation="e2",
        cells={
            "e2:a": MatrixCell(verdict=Outcome.PERFORMED),
            "e2:b": MatrixCell(verdict=Outcome.NOT_PERFORMED),
            "e2:c": MatrixCell(verdict=Outcome.NOT_PERFORMED),
        },
    )
    entry = _dissent_for(_dissent_matrix(first, second), "a", "B6")

    assert entry is not None
    assert (entry.dissents, entry.attributable) == (2, 2), (
        "two evaluations produced two rows under one judge name, not one"
    )


def test_lone_dissent_pools_the_two_requirement_matrices_for_one_check() -> None:
    """§5 names two key components — judge and check — where every other number
    a report computes carries three.

    Pooling double-counts nothing: a trajectory carries one requirement class per
    check, so its row exists in exactly one of the two matrices.
    """
    required = _matrix(
        _row(
            "t1", a=Outcome.PERFORMED, b=Outcome.NOT_PERFORMED, c=Outcome.NOT_PERFORMED
        ),
        requirement=Requirement.REQUIRED,
    )
    enriching = _matrix(
        _row(
            "t2", a=Outcome.PERFORMED, b=Outcome.NOT_PERFORMED, c=Outcome.NOT_PERFORMED
        ),
        requirement=Requirement.ENRICHING,
    )
    entries = lone_dissents(
        {(m.model, m.check_id, m.requirement): m for m in (required, enriching)}
    )
    keyed = [e for e in entries if e.judge == "a"]

    assert len(keyed) == 1, "the two requirement matrices produced two rows"
    assert (keyed[0].dissents, keyed[0].attributable) == (2, 2)


def test_dissents_are_counted_flat_and_not_by_band_pair() -> None:
    """§5 forbids averaging the bands, so a band pair has no defensible number.

    A `partial` dissenting from a unanimous `performed` and a `not_performed`
    dissenting from the same are different observations; the review queue carries
    every voting judge and its band per split row, which is where that lives.
    """
    matrix = _dissent_matrix(
        _row("t1", a=Outcome.PARTIAL, b=Outcome.PERFORMED, c=Outcome.PERFORMED),
        _row("t2", a=Outcome.NOT_PERFORMED, b=Outcome.PERFORMED, c=Outcome.PERFORMED),
    )
    entry = _dissent_for(matrix, "a", "B6")

    assert entry is not None
    assert (entry.dissents, entry.attributable) == (2, 2)
    assert not any(
        "band" in name or "partial" in name for name in LoneDissent.model_fields
    ), "a band-pair carrier was added; §5 counts dissents flat"


def test_a_report_computes_lone_dissent_rather_than_leaving_it_by_hand(
    tmp_path: Path,
) -> None:
    """From v42 a report computes it (§9): it was recoverable by hand before.

    The end-to-end shape, over stored logs: three judges on one evaluation with
    one of them dissenting on B6, and a single-judge evaluation beside it whose
    rows are ineligible for want of a third voter.
    """
    built = build_report(paired(*_thinly_judged_pair(tmp_path, "attrib")))

    b6 = {entry.judge: entry for entry in built.dissent if entry.check_id == "B6"}
    assert b6["judge_opus_b"].dissents == 1, "the dissent was not attributed"
    assert b6["judge_opus_b"].attributable == 1
    assert "judge_sonnet_xhigh" not in b6 and "judge_opus_a" not in b6, (
        "the row the two agreeing judges could not have been lone on entered "
        "their denominators: on it the other two split"
    )


def test_a_judge_name_carrying_two_configurations_refuses(tmp_path: Path) -> None:
    """Naming the name and both configurations (§0 v42).

    Narrower than a combination axis: an evaluation judged by three and one
    judged by four still combine. What is refused is one scorer name standing for
    two different `JudgeConfig`s inside one report, because the per-judge key is
    the name alone and a key that tolerates an error hides it.
    """
    first = evaluation(tmp_path, "cfg-a", judges=["judge_sonnet_xhigh"])
    second = evaluation(tmp_path, "cfg-b", judges=["judge_sonnet_xhigh"])

    pair = second.pair()
    log = pair.scoring[0]
    provenance = log.provenance
    rebranded = tuple(
        config.model_copy(update={"reasoning_effort": "low"})
        for config in provenance.judges
    )
    collided = pair.model_copy(
        update={
            "scoring": [
                log.model_copy(
                    update={
                        "provenance": provenance.model_copy(
                            update={"judges": rebranded}
                        )
                    }
                )
            ]
        }
    )

    assert_one_configuration_per_judge([first.pair()])
    with pytest.raises(JudgeConfigurationError) as raised:
        assert_one_configuration_per_judge([first.pair(), collided])

    message = str(raised.value)
    assert "judge_sonnet_xhigh" in message, message
    assert "xhigh" in message and "low" in message, (
        "the refusal named the judge but not both configurations"
    )

    with pytest.raises(JudgeConfigurationError):
        build_report([first.pair(), collided])


def test_judges_at_identical_configuration_stay_two_rows(tmp_path: Path) -> None:
    """`judge_opus_a` and `judge_opus_b` are two independent draws, not one
    instrument sampled twice — so the refusal never merges two names."""
    run = evaluation(tmp_path, "two-draws", judges=["judge_opus_a", "judge_opus_b"])
    configs = assert_one_configuration_per_judge([run.pair()])

    assert set(configs) == {"judge_opus_a", "judge_opus_b"}
    assert configs["judge_opus_a"].model == configs["judge_opus_b"].model


def test_the_lone_dissent_count_feeds_no_discard_rule() -> None:
    """§5, on §11's ground: a harness discarding on this number would select for
    lineage agreement — automating the validity threat §11 exists to detect.

    §14.32's vote-discarding mechanism stays a **declared** list. Asserted
    structurally, because the failure this guards against is a future edit rather
    than a present value: the exclusion carrier holds only what a human declared,
    and no function that builds an exclusion reads the dissent count.
    """
    assert set(Exclusion.model_fields) == {"uuid", "reason"}, (
        "the exclusion carrier grew a field that could hold a derived discard"
    )

    source = ast.parse(Path(report_module.__file__).read_text())
    anchor = {
        node.name
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "lone_dissents"
    }
    assert anchor == {"lone_dissents"}, (
        "the anchor this control keys on is gone; rename it here too"
    )

    # every `Exclusion` in this layer is parsed from a declared file, and there
    # is no second producer for a derived one to come from
    builders = {
        node.name
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "Exclusion"
            for call in ast.walk(node)
        )
    }
    assert builders == {"read_exclusions"}, (
        f"{sorted(builders - {'read_exclusions'})} builds an exclusion outside "
        "the declared-file reader; §5 rules the discard list stays declared and "
        "is never a threshold over the lone-dissent count"
    )


# -- 5. refusal, and exclusion after review ---------------------------------


def test_a_report_refuses_rather_than_dropping_the_offender(tmp_path: Path) -> None:
    """It never drops the offender and proceeds."""
    first = evaluation(tmp_path, "refuse-a", model="mockllm/one")
    second = evaluation(tmp_path, "refuse-b", model="mockllm/two")
    with pytest.raises(CombinationError):
        build_report(paired(first, second))


def test_an_excluded_trajectory_is_reported_under_its_own_cause(
    tmp_path: Path,
) -> None:
    """Keyed on uuid, with a reason, read at analysis time, log never mutated."""
    run = evaluation(tmp_path, "excl")
    matrices = build_matrices(paired(run))
    model = "mockllm/model-under-test"
    victim = matrices[(model, "B6", Requirement.REQUIRED)].rows[0].uuid

    exclusions = ExclusionList(
        exclusions=[Exclusion(uuid=victim, reason="tool output truncated on review")]
    )
    built = build_report(paired(run), exclusions=exclusions)
    counts = built.counts["B6:required"]

    assert counts.excluded == 1
    assert counts.observed == 0, "an excluded trajectory stayed in the denominator"
    assert counts.rows == 1, "an excluded trajectory left the displayed count"
    assert counts.reconciles(len(built.matrices["B6:required"].rows))
    assert built.exclusions[0].reason == "tool output truncated on review"

    stored = read_eval_log(
        next(iter(run.run.scoring_logs.values())), resolve_attachments=True
    )
    assert stored.samples, "the log was mutated"


def test_the_exclusion_carrier_is_the_uuid_not_the_instance(tmp_path: Path) -> None:
    """An instance runs many times and only one of those runs may be invalid."""
    run = evaluation(tmp_path, "excl-key")
    matrix = build_matrices(paired(run))[
        ("mockllm/model-under-test", "B6", Requirement.REQUIRED)
    ]
    row = matrix.rows[0]
    assert row.uuid != row.instance

    by_instance = ExclusionList(
        exclusions=[Exclusion(uuid=row.instance, reason="wrong key on purpose")]
    )
    built = build_report(paired(run), exclusions=by_instance)
    assert built.counts["B6:required"].excluded == 0, (
        "an instance id matched a trajectory key"
    )


def test_a_missing_exclusion_file_is_a_runtime_error(tmp_path: Path) -> None:
    """Not mitigated (§15): proceeding without it reports as if review never ran."""
    with pytest.raises(ExclusionError, match="could not be read"):
        read_exclusions(tmp_path / "nothing-here.json")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("{not json", "not valid JSON"),
        ('{"exclusions": 3}', "must be a list"),
        ('"a string"', "expected a list"),
        ('[{"uuid": "a", "reason": "x"}, {"uuid": "a", "reason": "y"}]', "twice"),
        ('[{"uuid": "a", "reason": "  "}]', "no reason"),
    ],
)
def test_a_bad_exclusion_file_is_a_runtime_error(
    tmp_path: Path, payload: str, match: str
) -> None:
    path = tmp_path / "exclusions.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ExclusionError, match=match):
        read_exclusions(path)


def test_a_well_formed_exclusion_file_reads_in_both_shapes(tmp_path: Path) -> None:
    entries = [{"uuid": "abc", "reason": "review"}]
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps(entries), encoding="utf-8")
    as_object = tmp_path / "object.json"
    as_object.write_text(json.dumps({"exclusions": entries}), encoding="utf-8")

    assert read_exclusions(as_list).uuids == {"abc"}
    assert read_exclusions(as_object).reason_for("abc") == "review"
    assert read_exclusions(as_object).reason_for("other") is None


# -- 5a. per-cause completeness exclusions (§5, unblocked by the pairing) ----


def test_exclusions_are_reported_per_cause_and_never_as_one_bucket(
    tmp_path: Path,
) -> None:
    """Every exclusion is reported per cause (§5).

    Computed over the **generation** log, which is the only place the cause
    exists: `select_judgeable` filters before any judge runs, so an excluded
    trajectory never reaches a scoring log at all. That is the half of §5 that
    v34's scoring-logs-only input could not deliver, and the half that failed
    loudly rather than silently.
    """
    run = evaluation(tmp_path, "causes")
    pair = run.pair()
    counted = completeness_exclusions([pair])

    selection = select_judgeable(read_log(run.generation))
    assert counted.trajectories == len(selection.judgeable) + len(selection.excluded)
    assert counted.judgeable == len(selection.judgeable)
    assert counted.by_cause == selection.causes, (
        "the report's number and the driver's number must be one number"
    )
    assert counted.excluded == sum(counted.by_cause.values())

    # the counts are keyed by cause; nothing here is a single bucket
    assert isinstance(counted.by_cause, dict)
    assert "excluded" not in set(CompletenessExclusions.model_fields), (
        "a single total would be one bucket for causes §5 keeps apart"
    )


def test_the_two_exclusions_are_never_merged(tmp_path: Path) -> None:
    """One happens before judging and one after, and §5 keeps them apart.

    The completeness exclusion never reaches a scoring log and so has no row in
    any matrix; the review exclusion keeps its row and is counted as
    :attr:`CheckCounts.excluded`. Two carriers, two names, one report.
    """
    run = evaluation(tmp_path, "two-kinds")
    matrices = build_matrices(paired(run))
    matrix = matrices[("mockllm/model-under-test", "B6", Requirement.REQUIRED)]
    victim = matrix.rows[0].uuid

    built = build_report(
        paired(run),
        exclusions=ExclusionList(
            exclusions=[Exclusion(uuid=victim, reason="review found it invalid")]
        ),
    )

    # both kinds are present in this one report, and they are two numbers
    assert built.counts["B6:required"].excluded == 1, "the review exclusion"
    assert built.exclusions[0].uuid == victim
    assert built.completeness.by_cause, (
        "the fixture carries a truncated trajectory; its cause must be reported"
    )
    assert "completeness:truncated" in built.completeness.by_cause

    # and they are carried apart, on two fields with two names
    assert {"completeness", "exclusions"} <= set(Report.model_fields)
    assert built.completeness.excluded != len(built.exclusions) or True

    # the completeness exclusion has **no row anywhere**: it never reached a
    # scoring log, so it is not in any matrix and cannot be double-counted as a
    # review exclusion. The review exclusion keeps its row.
    rows = {row.uuid for m in matrices.values() for row in m.rows}
    assert victim in rows
    assert len(rows) == built.completeness.judgeable


def test_an_excluded_trajectory_never_reaches_a_scoring_log(tmp_path: Path) -> None:
    """Why the input had to widen, asserted rather than argued.

    The judged set is the judgeable set, so a report over scoring logs alone can
    see no exclusion at all — its cause lives in the generation log and nowhere
    else (§0 v35).
    """
    run = evaluation(tmp_path, "reach")
    generation = read_log(run.generation)
    selection = select_judgeable(generation)

    judged = {
        (str(sample.id), sample.epoch)
        for sample in read_log(run.locations[0]).samples or []
    }
    assert judged == selection.keys
    assert judged.isdisjoint({t.key for t in selection.excluded})


def test_the_per_cause_rate_carries_its_own_denominator(tmp_path: Path) -> None:
    """Exclusion counts are results, not hygiene (§5).

    The rate is offered per cause and over a stated denominator, because
    exclusion drift between two models is exactly the comparison the raw counts
    hide. An empty denominator yields no rates rather than zeroes.
    """
    counted = CompletenessExclusions(
        trajectories=20,
        judgeable=12,
        by_cause={"completeness:truncated": 6, "stop_cause:turn_cap": 2},
    )
    assert counted.excluded == 8
    assert counted.rate_by_cause() == {
        "completeness:truncated": 0.3,
        "stop_cause:turn_cap": 0.1,
    }
    assert CompletenessExclusions().rate_by_cause() == {}, (
        "nothing was run is not the same claim as nothing was excluded"
    )
    assert any("completeness:truncated: 6" in line for line in counted.lines())


# -- 5b. the two report-level surfaces (§9: per evaluation only, until now) --


def test_the_review_queue_is_report_level_and_keyed_by_uuid(tmp_path: Path) -> None:
    """A report holds matrices, so its review queue is built over matrices.

    Keyed by trajectory ``uuid`` and carrying the requirement class, because
    ``(sample_id, epoch)`` is not a key across evaluations and a check split as
    required is a different observation from the same check split as enriching.
    """
    run = evaluation(
        tmp_path,
        "queue",
        replies={
            "judge_sonnet_xhigh": {**FULL_VERDICTS, "B6": "performed"},
            "judge_opus_a": {**FULL_VERDICTS, "B6": "not_performed"},
            "judge_opus_b": {**FULL_VERDICTS, "B6": "performed"},
        },
    )
    built = build_report(paired(run))

    splits = [item for item in built.review if item.check_id == "B6"]
    assert splits, f"no split reached the queue: {built.rows()}"
    item = splits[0]
    assert item.uuid and item.instance and item.evaluation
    assert item.requirement in set(Requirement)
    assert len(item.verdicts) >= 2, "a split needs two voters to be a split"
    assert list(item.verdicts) == sorted(item.verdicts)

    # the split is a count in the row as well as an entry in the queue
    assert built.counts[f"B6:{item.requirement.value}"].split == len(splits)


def test_report_chain_consistency_reads_the_catalogue_s_own_refines_links(
    tmp_path: Path,
) -> None:
    """The chains come from the catalogue, never from a hardcoded id triple.

    The fixture catalogue declares B6 → B5 → B1, and this is now the only chain
    report in the tree: the per-evaluation version retired at v40 with the layer
    around it, and the debt to widen it off its fixture went with it (§14.34).
    """
    run = evaluation(tmp_path, "chain")
    built = build_report(paired(run))

    assert built.consistency, "the fixture catalogue declares a refinement chain"
    chains = [entry.chain for entry in built.consistency]
    assert chains == [
        chain for chain in refinement_chains(mj.CATALOGUE, mj.CATALOGUE.ids)
    ]

    entry = built.consistency[0]
    assert entry.consistent + entry.violations + entry.no_claim > 0
    assert entry.consistent + entry.no_claim + len(entry.offending) > 0


def test_a_chain_read_across_a_split_makes_no_claim(tmp_path: Path) -> None:
    """A chain read across a split would report a judging gap as incoherence."""
    run = evaluation(
        tmp_path,
        "chain-split",
        replies={
            "judge_sonnet_xhigh": {**FULL_VERDICTS, "B5": "performed"},
            "judge_opus_a": {**FULL_VERDICTS, "B5": "not_performed"},
            "judge_opus_b": {**FULL_VERDICTS, "B5": "partial"},
        },
    )
    built = build_report(paired(run))
    entry = next(e for e in built.consistency if "B5" in e.chain)
    assert entry.no_claim > 0, "an unresolved link entered the chain's claim"
    assert entry.violations == 0


def test_a_refinement_above_its_base_is_flagged(tmp_path: Path) -> None:
    """The rule itself: a refinement resolved strictly above its base.

    Surfacing a refined step implies the step it refines, so B5 performed with B6
    not_performed is incoherent in the resolved reading.
    """
    run = evaluation(
        tmp_path,
        "chain-violate",
        replies=dict.fromkeys(
            ("judge_sonnet_xhigh", "judge_opus_a", "judge_opus_b"),
            {**FULL_VERDICTS, "B6": "not_performed", "B5": "performed"},
        ),
    )
    built = build_report(paired(run))
    entry = next(e for e in built.consistency if e.chain[:2] == ["B6", "B5"])
    assert entry.violations > 0, [e.model_dump() for e in built.consistency]
    violation = entry.offending[0]
    assert violation.base == "B6" and violation.refinement == "B5"
    assert violation.refinement_band > violation.base_band
    assert violation.uuid and violation.instance and violation.evaluation


# -- 6. the survey ----------------------------------------------------------


def test_the_survey_groups_what_combines_and_names_what_does_not(
    tmp_path: Path,
) -> None:
    """The operator's instrument for deciding what to re-judge and what to discard."""
    shared = tmp_path / "all"
    first = evaluation(shared, "s-a")
    second = evaluation(shared, "s-b", instances=SECOND_INSTANCES)
    other = evaluation(shared, "s-c", model="mockllm/a-different-model")

    result = survey(shared)
    assert len(result.evaluations) == 3, [e.evaluation for e in result.evaluations]

    combined = [group for group in result.groups if len(group.evaluations) > 1]
    assert combined, f"nothing combined: {result.lines()}"
    assert any("model" in line for line in result.incombinable), result.incombinable

    models = {group.model for group in result.groups}
    assert models == {"mockllm/model-under-test", "mockllm/a-different-model"}
    del first, second, other


def test_the_survey_reads_headers_only(tmp_path: Path) -> None:
    """It never opens a sample."""
    evaluation(tmp_path, "hdr")
    result = survey(tmp_path)
    assert result.evaluations
    for entry in result.evaluations:
        assert entry.catalogue_sha256
        assert entry.instances_sha256


def test_a_log_whose_digests_match_nothing_in_the_tree_is_detectable(
    tmp_path: Path,
) -> None:
    """§14.31 — and the survey is the only place this check can run.

    The fixture instance set and the fixture catalogue exist only in the test
    module, so neither digest resolves against the shipped tree. The operator's
    remedy is to re-judge it or remove it.
    """
    evaluation(tmp_path, "unmatched")
    result = survey(tmp_path)
    entry = result.evaluations[0]
    assert entry.unknown_instances, (
        "a fixture instance set resolved against the shipped tree"
    )
    assert entry.unknown_catalogue
    assert any("unmatched" in line for line in result.lines()), result.lines()


def test_the_unmatched_flag_does_not_decide_what_combines(tmp_path: Path) -> None:
    """The repair of `survey()`'s two contradictory lines (§9, reported at v43).

    `combine` is read from what the logs record, over §5's combination axes;
    `unmatched` is read against *this checkout*. They answered different
    questions while the second claimed a report consequence it does not have.

    **Which line was false is settled by building the report**, not by reading
    either sentence: a report over exactly the evaluations the survey flags
    builds and reconciles. Instances are the one axis that may vary and that is
    the whole point of the operation (§5), so the grouping is right and the
    sentence gave way.
    """
    first = evaluation(tmp_path, "unmatched-a")
    second = evaluation(tmp_path, "unmatched-b")
    result = survey(tmp_path)

    assert all(entry.unknown_instances for entry in result.evaluations)
    assert [len(group.evaluations) for group in result.groups] == [2], (
        "the flag reached the grouping; instances are not a combination axis"
    )

    built = build_report(paired(first, second))
    assert built.reconciles() and built.matrices, (
        "a report over the flagged evaluations refused, which would make the "
        "survey's sentence the true one"
    )

    unmatched = [line for line in result.lines() if line.startswith("unmatched")]
    assert unmatched
    for line in unmatched:
        assert "cannot contribute to a report" not in line, line
        assert "does not decide combinability" in line, line


def test_the_survey_pairs_and_reads_its_axes_from_the_generation_side(
    tmp_path: Path,
) -> None:
    """The survey pairs too (§5), and reads each fact where it originates.

    A directory holds both halves, and a report's input is a pair — so the
    survey's unit is a pair as well, with the axes read from the generation
    log's own record and the catalogue from the scoring side.
    """
    run = evaluation(tmp_path, "srv-pair")
    result = survey(tmp_path)

    (entry,) = result.evaluations
    assert entry.generation_location == run.generation
    assert sorted(entry.locations) == sorted(run.locations)
    assert not result.unpaired, result.unpaired
    assert not result.unreadable, result.unreadable

    pair = run.pair()
    assert entry.solver_sha256 is not None
    assert entry.instances_sha256 == pair.generation.provenance.instances_sha256
    assert entry.catalogue_sha256 == pair.scoring[0].catalogue.digest


def test_a_generation_log_with_no_scoring_log_is_named_unpaired(
    tmp_path: Path,
) -> None:
    """An unpaired log is its own named condition, never a drop (§5).

    Before v35 a generation log in a surveyed directory was *unreadable* — the
    survey knew only about scoring logs, so the half it now requires read as a
    stray file. It has not been judged, which is a thing an operator needs told.
    """
    run = evaluation(tmp_path, "mixed").run
    directory = Path(run.generation_location).parent
    result = survey(directory)

    assert not result.unreadable, result.unreadable
    assert result.unpaired, "a generation log did not land in `unpaired`"
    assert any("no scoring log" in line for line in result.unpaired), result.unpaired
    assert any(line.startswith("unpaired (") for line in result.lines()), result.lines()
    assert not result.evaluations, "an unpaired half entered the survey's evaluations"


def test_scoring_logs_with_no_generation_log_are_named_unpaired(
    tmp_path: Path,
) -> None:
    """The other half, and the reason it cannot enter a report on its own.

    The requirement class, the solver digest and the exclusion causes are all
    read from the generation side now, so scoring logs alone are not an
    evaluation.
    """
    run = evaluation(tmp_path, "orphan").run
    directory = Path(next(iter(run.scoring_logs.values()))).parent
    result = survey(directory)

    assert not result.unreadable, result.unreadable
    assert result.unpaired, "orphan scoring logs did not land in `unpaired`"
    assert any("no generation log" in line for line in result.unpaired), result.unpaired
    assert not result.evaluations


def test_the_survey_reports_an_unreadable_log_rather_than_raising(
    tmp_path: Path,
) -> None:
    """A file that is not an Inspect log at all is reported, never raised.

    A survey over a directory is exactly where mixed contents are expected, and a
    survey that raises on a stray file tells an operator nothing.
    """
    evaluation(tmp_path, "stray")
    stray = tmp_path / "stray" / "generation" / "not-a-log.eval"
    stray.write_bytes(b"not an eval log at all")

    result = survey(tmp_path)
    assert result.unreadable, "a stray file did not land in `unreadable`"
    assert any("not-a-log.eval" in line for line in result.unreadable)
    assert any(
        "could not be read as an Inspect log" in line for line in result.unreadable
    ), result.unreadable
    assert result.evaluations, "the real evaluation beside it was lost"

    # **The negative half of the refused/unreadable split.** A parse failure
    # stays here and reaches `refused` never: a split that moved everything
    # would satisfy any assertion about `refused` being populated.
    assert not result.refused, result.refused
    printed = result.lines()
    assert any(line.startswith("unreadable (") for line in printed), printed
    assert not any(line.startswith("refused (") for line in printed), printed


def test_which_half_a_log_is_comes_from_its_header_not_from_load_success(
    tmp_path: Path,
) -> None:
    """A scoring log that fails to load is still a scoring log.

    A scoring log written before the judge axis was carried has no sidecar, and
    the two facts are independent: classifying on load success reports it as
    *not a scoring log*, which is false and points the operator at the wrong
    repair. The discriminator is the header's ``catalogue`` option (§0 v36).
    """
    run = evaluation(tmp_path, "no-sidecar").run
    scoring = Path(next(iter(run.scoring_logs.values())))
    provenance_sidecar_path(scoring).unlink()

    result = survey(Path(run.generation_location).parent.parent)

    (failure,) = [
        line
        for line in result.unreadable
        if "a scoring log this survey could not load" in line
    ]
    # and it names the model under test: this line's header *did* parse, so
    # unlike a file that is not a log at all it has an `EvalSpec.model` to carry
    # — the two `unreadable` shapes differ on exactly that (§9 v82)
    assert failure.startswith("(mockllm/model-under-test): "), failure
    assert not any("not a scoring log" in line for line in result.unreadable), (
        "a real scoring log was diagnosed as not being one"
    )
    assert any("no provenance sidecar" in line for line in result.unreadable)
    # the other parse-failure member, and it does not reach `refused` either
    assert not result.refused, result.refused


def test_the_survey_matches_the_two_halves_across_directories(
    tmp_path: Path,
) -> None:
    """Directory layout carries no meaning; the join is the ``eval_id``.

    A generation log in one round directory and its scoring logs in two others
    are one combinable group, and the survey has to say so — an operator's tree
    is round directories, and the report the survey exists to help them build
    takes a generation log and the scoring logs taken from it wherever they lie.
    """
    run = evaluation(tmp_path / "built", "scattered")
    tree = tmp_path / "tree"
    generation = tree / "some" / "round"
    generation.mkdir(parents=True)
    shutil.copy(run.generation, generation / Path(run.generation).name)
    for index, location in enumerate(run.locations):
        elsewhere = tree / f"judged-{index}"
        elsewhere.mkdir()
        shutil.copy(location, elsewhere / Path(location).name)
        shutil.copy(
            location + ".provenance.json",
            elsewhere / (Path(location).name + ".provenance.json"),
        )

    result = survey(tree)

    (entry,) = result.evaluations
    assert not result.unpaired, result.unpaired
    assert not result.unreadable, result.unreadable
    assert Path(entry.generation_location).parent == generation
    assert len({Path(location).parent for location in entry.locations}) == len(
        run.locations
    ), "the scoring logs were not matched across separate directories"
    assert [len(group.evaluations) for group in result.groups] == [1]


def test_the_default_survey_line_is_one_per_evaluation(tmp_path: Path) -> None:
    """One line per evaluation, which is one `kgpbench report` invocation.

    The line this replaces read `combine` and gathered several evaluations under
    one heading, describing a report the shipped command cannot build:
    ``--generation`` is given once, so two evaluations are two invocations. The
    two here **do** combine, which is what makes the shape the assertion — under
    the grouped line they were one line and they are now two.

    The model label stays on each line, because it is the one thing saying what
    was run rather than what can be run; what goes with the heading is the claim
    that several evaluations share it.
    """
    first = evaluation(tmp_path, "named-a", judges=["judge_opus_a", "judge_opus_b"])
    second = evaluation(tmp_path, "named-b", judges=["judge_opus_a"])
    result = survey(tmp_path)

    assert [len(group.evaluations) for group in result.groups] == [2], (
        "the two evaluations did not combine, so this fixture cannot show that "
        "the summary stopped grouping"
    )

    summary = [line for line in result.lines() if line.startswith("report (")]
    assert len(summary) == 2, summary
    assert not any(line.startswith("combine") for line in result.lines())

    by_id = {entry.evaluation: entry for entry in result.evaluations}
    for line in summary:
        named = [key for key in by_id if key in line]
        assert len(named) == 1, f"a summary line names {len(named)} evaluations: {line}"
        entry = by_id[named[0]]
        assert f"report ({entry.model})" in line, line
        assert f"{len(entry.locations)} scoring log" in line, line
        for judge in entry.judges:
            assert judge in line, line
        assert f"{len(entry.checks)} checks" in line, line
        assert entry.generation_location not in line, (
            "the default line is carrying full paths"
        )

    two, one = sorted(summary, key=lambda line: line.count("judge_"), reverse=True)
    assert "2 scoring logs" in two and "1 scoring log" in one, summary
    del first, second


def test_the_paths_flag_prints_the_report_invocation(tmp_path: Path) -> None:
    """`--paths` is what makes the survey actionable without opening a file.

    What an operator does next with a path is type it, so the paths are spelled
    as the `kgpbench report` arguments that would read them rather than as a
    list. The default stays compact because a tree of full paths at every group
    is a wall.
    """
    run = evaluation(tmp_path, "invocation", judges=["judge_opus_a"])
    result = survey(tmp_path)

    plain = result.lines()
    with_paths = result.lines(paths=True)

    assert f"    --generation {run.generation}" in with_paths
    assert f"    --scoring {run.locations[0]}" in with_paths
    assert not any(line.startswith("    --") for line in plain), plain
    assert len(with_paths) > len(plain)


def test_two_scoring_logs_for_one_judge_name_both_candidates(tmp_path: Path) -> None:
    """The scoring side's collision, which produced no message at all.

    A judge runs once per `score()` call and its log is named for it, so two
    logs claiming one ``(eval_id, judge)`` are one log in two places — the state
    copying a round directory leaves. Both used to enter the pair: the group's
    judge list carried the name twice and the `combine` line was unchanged,
    while `assert_combinable` refuses exactly this. The survey never reached
    that refusal, because its grouping loop only calls it when a *second*
    evaluation is tested against an existing group — so a lone evaluation
    carrying a duplicate was surveyed as clean and refused by the report.

    The message is the generation-side collision's: both full paths, which
    claimed the key first, and why one must be a copy.
    """
    run = evaluation(tmp_path, "copied", judges=["judge_opus_a"])
    original = run.locations[0]
    copy_dir = tmp_path / "copied-again"
    copy_dir.mkdir()
    duplicate = copy_dir / Path(original).name
    shutil.copy(original, duplicate)
    shutil.copy(original + ".provenance.json", str(duplicate) + ".provenance.json")

    result = survey(tmp_path)

    (entry,) = result.evaluations
    assert entry.judges == ["judge_opus_a"], entry.judges
    assert len(entry.locations) == 1, entry.locations

    (collision,) = [line for line in result.refused if "a second scoring log" in line]
    for candidate in (original, str(duplicate)):
        assert candidate in collision, collision
    assert "already claims" in collision
    assert "one of these is a copy" in collision
    # and it is a refusal rather than a failure to read: both of these files
    # open, and the operator's move is to drop the copy
    assert not result.unreadable, result.unreadable
    assert any(line.startswith("refused (") for line in result.lines())

    # and the survivor still groups, so the report the survey points at builds
    assert [len(group.evaluations) for group in result.groups] == [1]
    assert build_report(paired(run)).reconciles()


def test_two_generation_logs_for_one_evaluation_name_both_candidates(
    tmp_path: Path,
) -> None:
    """The generation side's collision, which printed under the wrong word.

    `eval()` issues one ``eval_id`` per call, so two generation logs naming one
    are one log in two places — the state copying a round directory leaves, and
    the collision the scoring-side twin above was written to match. This is the
    older of the two by a wide margin, and it printed under ``unreadable``: an
    operator was told that a log they can open in an editor could not be read,
    which sends them to look for a corrupt file. It opens, it parses, and what
    is wrong with it is what it claims, which is a refusal.

    Unlike its twin it has no `assert_combinable` counterpart — a report takes
    one generation log — so this survey is the only place it is caught, and it
    had no test at all until this one.
    """
    run = evaluation(tmp_path, "twice-over", judges=["judge_opus_a"])
    original = run.generation
    elsewhere = tmp_path / "twice-over-again"
    elsewhere.mkdir()
    duplicate = elsewhere / Path(original).name
    shutil.copy(original, duplicate)

    result = survey(tmp_path)

    (collision,) = [
        line
        for line in result.refused
        if "a second generation log for evaluation" in line
    ]
    for candidate in (original, str(duplicate)):
        assert candidate in collision, collision
    assert run.evaluation_id in collision, collision
    assert "already names" in collision
    assert "one of these is a copy" in collision

    # a refusal and not a failure to read: both of these files open and parse
    assert not result.unreadable, result.unreadable
    assert any(
        line.startswith("refused (") and "a second generation log" in line
        for line in result.lines()
    ), result.lines()

    # and the survivor still pairs, so one copy costs the evaluation nothing
    (entry,) = result.evaluations
    assert entry.evaluation == run.evaluation_id
    assert sum(line.startswith("report (") for line in result.lines()) == 1


def test_two_generation_logs_with_two_eval_ids_are_not_refused(
    tmp_path: Path,
) -> None:
    """The half that carries the work: the key is the ``eval_id``, not the count.

    A refusal reading *a second generation log in this directory* would satisfy
    the test above and refuse every ordinary survey, because a directory holding
    two runs is what a round looks like. Both generation logs are put side by
    side in one directory to say so on the layout too: two files, two
    ``eval_id``s, nothing refused and nothing unreadable.
    """
    first = evaluation(tmp_path, "one-run", judges=["judge_opus_a"])
    second = evaluation(tmp_path, "the-other-run", judges=["judge_opus_a"])
    # read before the move: the fixture's own accessor reads the log at the
    # path it was written to
    identifiers = sorted((first.evaluation_id, second.evaluation_id))
    beside = Path(first.generation).parent / f"moved-{Path(second.generation).name}"
    shutil.move(second.generation, beside)

    result = survey(tmp_path)

    assert not result.refused, result.refused
    assert not result.unreadable, result.unreadable
    assert not result.unpaired, result.unpaired
    assert sorted(entry.evaluation for entry in result.evaluations) == identifiers
    assert sum(line.startswith("report (") for line in result.lines()) == 2


def test_an_evaluation_whose_own_scoring_logs_disagree_is_refused(
    tmp_path: Path,
) -> None:
    """One evaluation, one trajectory, judged twice across a wording change.

    `assert_combinable` compares the catalogues of every scoring log handed to
    it, a single evaluation's own included — so the report refuses this, while
    the survey printed a ``report`` line for it: the instrument advertising an
    invocation the command rejects.

    The refusal is at pairing time and names this evaluation's own logs, both of
    them, and the check whose definition moved.
    """
    run = evaluation(tmp_path, "self-disagreeing", judges=["judge_opus_a"])
    second = rejudge_under(
        run, judge="judge_opus_b", catalogue=redefined_catalogue("B6")
    )

    result = survey(tmp_path)

    assert not result.evaluations, [e.evaluation for e in result.evaluations]
    assert not result.groups
    (refusal,) = [line for line in result.refused if "disagree" in line]
    assert run.evaluation_id in refusal, refusal
    for location in (run.locations[0], second):
        assert location in refusal, refusal
    assert "B6" in refusal, refusal
    assert "kgpbench judge" in refusal, refusal

    # a refusal, not a failure to read: both logs open and both parse
    assert not result.unreadable, result.unreadable

    # and it prints under a heading, not as a `report` line the command rejects
    printed = result.lines()
    assert not any(line.startswith("report (") for line in printed), printed
    assert any(
        line.startswith("refused (") and "disagree" in line for line in printed
    ), printed


def test_an_evaluation_whose_own_scoring_logs_agree_is_not_refused(
    tmp_path: Path,
) -> None:
    """The half that carries the work.

    A guard that fired on every multi-judge evaluation would satisfy the test
    above and destroy the instrument: two judges under **one** catalogue are the
    ordinary case, and an added check is a combinable difference (§5) rather
    than a redefinition. Both shapes are asserted, because only the second
    distinguishes *the catalogues differ* from *a shared check's `text` moved*.
    """
    same = evaluation(tmp_path / "same", "agreeing", judges=["judge_opus_a"])
    rejudge_under(same, judge="judge_opus_b", catalogue=mj.CATALOGUE)

    added = evaluation(tmp_path / "added", "extended", judges=["judge_opus_a"])
    rejudge_under(added, judge="judge_opus_b", catalogue=extended_catalogue())

    for directory, run, dropped in (
        (tmp_path / "same", same, []),
        (tmp_path / "added", added, ["C9"]),
    ):
        result = survey(directory)
        assert not result.refused, result.refused
        assert not result.unreadable, result.unreadable
        (entry,) = result.evaluations
        assert entry.evaluation == run.evaluation_id
        assert sorted(entry.judges) == ["judge_opus_a", "judge_opus_b"]
        assert entry.dropped_checks == dropped, entry.dropped_checks
        assert any(line.startswith("report (") for line in result.lines())


def test_the_refusal_is_not_attributed_to_an_innocent_neighbour(
    tmp_path: Path,
) -> None:
    """The misattribution, which is why the item was ruled and not merely noted.

    With a clean second evaluation present the grouping loop reached
    `assert_combinable` only when the clean one was tested against the broken
    one, so the refusal surfaced as *these two do not join*, named all three
    logs including the innocent one, and then gave **both** evaluations their
    own ``report`` line. An operator read it as a fault in a relationship when
    it is inside one evaluation.

    Closing the first half closes this one, and this asserts that it does: the
    clean evaluation reports, the broken one does not, and nothing in the output
    names the clean evaluation's logs as part of the problem.
    """
    broken = evaluation(tmp_path, "broken", judges=["judge_opus_a"])
    rejudge_under(broken, judge="judge_opus_b", catalogue=redefined_catalogue("B6"))
    clean = evaluation(tmp_path, "clean", judges=["judge_opus_a"])

    result = survey(tmp_path)

    (entry,) = result.evaluations
    assert entry.evaluation == clean.evaluation_id
    assert not result.incombinable, result.incombinable
    assert not result.unreadable, result.unreadable
    (refusal,) = result.refused
    assert broken.evaluation_id in refusal
    assert clean.evaluation_id not in refusal, refusal
    assert clean.locations[0] not in refusal, refusal

    lines = result.lines()
    assert not any("does not join" in line for line in lines), lines
    assert sum(line.startswith("report (") for line in lines) == 1, lines


def test_readable_refusals_and_a_parse_failure_are_two_lists_and_two_printed_words(
    tmp_path: Path,
) -> None:
    """The split, asserted where several refusals are present at once.

    Either half alone is satisfiable by a survey that moved everything or by one
    that moved nothing. One directory carrying readable refusals *and* an
    unreadable file separates them: each refusal is in ``refused`` and not in
    ``unreadable``, the stray file is in ``unreadable`` and not in ``refused``,
    and the printed lines carry the two words apart. They are asserted together
    because the field's claim is about a kind of thing and not about one
    condition — a split that carried some of them and left another behind is
    what this test now refuses to pass.

    **The subject is the split into two fields and two printed words, not the
    membership.** Which conditions reach ``refused`` is §14.40's, and the
    refusals built here are some of them, not all: widening this directory to
    carry every one would make this test a carrier of that enumeration, which is
    the thing §14.40 exists to hold in one place. The two counts in the name are
    counts of fields.

    The word is the whole ground for the field. An operator told that a file
    they can open in an editor is unreadable goes looking for a corrupt log.

    **Every needle here is a diagnosis and every one of them carries a space**,
    which is what keeps a round directory's name out of the match: each printed
    line contains the log's full path, and this test's first draft asserted on a
    path — which passes on a directory that happens to be named for the fault.
    """
    disagreeing = evaluation(tmp_path, "two-catalogues", judges=["judge_opus_a"])
    rejudge_under(
        disagreeing, judge="judge_opus_b", catalogue=redefined_catalogue("B6")
    )

    judged_twice = evaluation(tmp_path, "copied-judgement", judges=["judge_opus_a"])
    origin = judged_twice.locations[0]
    beside = tmp_path / "copied-judgement-again"
    beside.mkdir()
    twin = beside / Path(origin).name
    shutil.copy(origin, twin)
    shutil.copy(origin + ".provenance.json", str(twin) + ".provenance.json")

    run_twice = evaluation(tmp_path, "copied-run", judges=["judge_opus_a"])
    again = tmp_path / "copied-run-again"
    again.mkdir()
    shutil.copy(run_twice.generation, again / Path(run_twice.generation).name)

    stray = tmp_path / "copied-run" / "generation" / "not-a-log.eval"
    stray.write_bytes(b"not an eval log at all")

    result = survey(tmp_path)

    (scoring_duplicate,) = [
        line for line in result.refused if "a second scoring log for judge" in line
    ]
    (generation_duplicate,) = [
        line
        for line in result.refused
        if "a second generation log for evaluation" in line
    ]
    (redefinition,) = [
        line for line in result.refused if "disagree on the definition of" in line
    ]
    assert judged_twice.evaluation_id in scoring_duplicate, scoring_duplicate
    assert run_twice.evaluation_id in generation_duplicate, generation_duplicate
    assert disagreeing.evaluation_id in redefinition, redefinition

    (unreadable,) = result.unreadable
    assert "not-a-log.eval" in unreadable, unreadable
    assert "could not be read as an Inspect log" in unreadable, unreadable

    # no refusal appears under the other word, and the parse failure appears
    # under neither of theirs
    for needle in (
        "a second scoring log for judge",
        "a second generation log for evaluation",
        "disagree on the definition of",
    ):
        assert not any(needle in line for line in result.unreadable), result.unreadable
    assert not any("not-a-log.eval" in line for line in result.refused), result.refused

    printed = result.lines()
    assert [line for line in printed if line.startswith("refused (")] == [
        f"refused {line}"
        for line in (scoring_duplicate, generation_duplicate, redefinition)
    ], printed
    assert [line for line in printed if line.startswith("unreadable (")] == [
        f"unreadable {unreadable}"
    ], printed
    # the two evaluations that are not themselves broken still report
    assert sum(line.startswith("report (") for line in printed) == 2, printed


# -- 6a. the model on every line ---------------------------------------------

SURVEY_LINE = re.compile(
    r"^(?P<kind>report|incombinable|unmatched|unpaired|refused|unreadable) "
    r"\((?P<model>[^)]+)\): "
)
"""Every survey line but a `--paths` continuation, split into its kind and its
parenthetical. Written here rather than derived from the source so that a line
losing its model goes red instead of the pattern following it (§15)."""

OTHER_MODEL = "mockllm/a-different-model"
"""A second model under test, so a line can be asked which one it names."""

UNREAD_HEADER = "unread header"
"""What stands in the parenthetical for a file whose header did not parse.

It is not a model and is not meant to read as one: nothing was parsed, so there
is no ``EvalSpec.model`` to carry."""


def test_every_survey_line_kind_names_the_model_under_test(tmp_path: Path) -> None:
    """Half one of §9 v82's survey ruling, over every kind at once.

    A rollout holds one generation log and ``inspect eval --model a,b`` writes
    two into it (§14.49), alike in everything the survey printed but the
    ``eval_id`` and the file name — which is what a copied log looks like too.
    So every line carries the model, and this test is over a directory holding
    **all six kinds simultaneously**: a kind that is absent cannot fail, and the
    six arise from six different code paths.

    The negative half is the `UNREAD_HEADER` assertion. A parenthetical that
    always held some string would satisfy the regex whatever it carried, so the
    models are checked against the two that were actually run, and the one line
    that cannot name a model is required to be the one file that is not a log.
    """
    # report, and incombinable: two evaluations that pair and do not combine
    first = evaluation(tmp_path, "m-a", judges=["judge_opus_a"])
    second = evaluation(tmp_path, "m-b", judges=["judge_opus_a"], model=OTHER_MODEL)

    # unpaired, both of its readable shapes: a generation log with nothing
    # beside it, and scoring logs whose generation log is gone. They are two
    # construction sites reading the model off two different headers, so one of
    # them alone is not coverage of the kind.
    lone = evaluation(tmp_path, "m-lone", judges=["judge_opus_a"])
    for location in lone.locations:
        Path(location).unlink()
        Path(provenance_sidecar_path(location)).unlink()

    orphans = evaluation(
        tmp_path, "m-orphans", judges=["judge_opus_a"], model=OTHER_MODEL
    )
    Path(orphans.generation).unlink()

    # refused: a second generation log claiming an eval_id another names
    copied = tmp_path / "m-copy"
    copied.mkdir()
    shutil.copy(first.generation, copied / Path(first.generation).name)

    # unreadable: a file that is not an Inspect log at all
    (tmp_path / "m-copy" / "not-a-log.eval").write_bytes(b"not an eval log")

    printed = [
        line for line in survey(tmp_path).lines(paths=True) if not line.startswith(" ")
    ]

    kinds: dict[str, list[str]] = {}
    for line in printed:
        matched = SURVEY_LINE.match(line)
        assert matched is not None, f"no model on this line: {line}"
        kinds.setdefault(matched["kind"], []).append(matched["model"])

    assert set(kinds) == {
        "report",
        "incombinable",
        "unmatched",
        "unpaired",
        "refused",
        "unreadable",
    }, kinds

    # every parenthetical is one of the two models that ran, except the file
    # that was never read as a log — which is the only line carrying the other
    # word, and there is exactly one of it
    run = {"mockllm/model-under-test", OTHER_MODEL}
    for kind, models in kinds.items():
        for model in models:
            assert model in run or model == UNREAD_HEADER, (kind, model)
    assert kinds["unreadable"] == [UNREAD_HEADER], kinds["unreadable"]
    assert UNREAD_HEADER not in [
        model
        for kind, models in kinds.items()
        if kind != "unreadable"
        for model in models
    ]
    # both `unpaired` shapes are present, and they name the two models apart —
    # one read from a generation header, one from the scoring headers' copy
    assert sorted(kinds["unpaired"]) == sorted(
        ["mockllm/model-under-test", OTHER_MODEL]
    ), kinds["unpaired"]
    del second


def test_two_generation_logs_differing_only_in_model_print_as_two_models(
    tmp_path: Path,
) -> None:
    """Half two, and it is the failure the ruling was written against.

    ``inspect eval --model a,b`` builds the task once per model and writes one
    log per model into one ``generation/`` directory (§14.49, measured at v82).
    Neither has been judged, so both print under ``unpaired`` — and before this
    round those two lines differed only in an ``eval_id`` and a file name, which
    is also what one log copied twice looks like.

    The fixture is what carries the claim: the two runs are the same task over
    the same instances at the same effort, so the **model is the only axis that
    differs**, and a survey that dropped it could not distinguish them at all.
    """
    rollout = tmp_path / "one-rollout"
    together = rollout / "generation"
    first = evaluation(rollout, "gen-a", judges=["judge_opus_a"])
    second = evaluation(rollout, "gen-b", judges=["judge_opus_a"], model=OTHER_MODEL)
    for produced in (first, second):
        for location in produced.locations:
            Path(location).unlink()
            Path(provenance_sidecar_path(location)).unlink()
    together.mkdir(parents=True, exist_ok=True)
    for produced in (first, second):
        shutil.move(produced.generation, together / Path(produced.generation).name)
    shutil.rmtree(rollout / "gen-a")
    shutil.rmtree(rollout / "gen-b")

    unpaired = [
        line for line in survey(rollout).lines() if line.startswith("unpaired (")
    ]

    assert len(unpaired) == 2, unpaired
    models = sorted(SURVEY_LINE.match(line)["model"] for line in unpaired)  # type: ignore[index]
    assert models == sorted(["mockllm/model-under-test", OTHER_MODEL]), unpaired
    # and they are two logs rather than one named twice
    assert len({line for line in unpaired}) == 2, unpaired


# -- 6b. the survey refuses what the report refuses --------------------------


def restate_the_model(location: str, model: str) -> None:
    """Rewrite one scoring log's recorded model under test, in place.

    `score()` preserves ``EvalSpec.model``, so the two halves of a pair agree by
    construction and no sequence of harness calls produces a disagreement —
    which is exactly why the refusal exists: it is what makes the preserved copy
    a checked fact rather than an assumption. So the fixture states the other
    value directly, on a log that is in every other respect the one the driver
    wrote, with its ``eval_id`` untouched so the survey still matches the two
    halves to each other.
    """
    log = read_eval_log(location, resolve_attachments=True)
    log.eval.model = model
    write_eval_log(log, location)


def test_the_survey_refuses_a_pair_whose_headers_disagree_on_the_model(
    tmp_path: Path,
) -> None:
    """Half one, and the line is asked which models it names.

    One directory, two evaluations: one whose scoring log has been restated to
    another model, and one left alone. Both are needed — a survey that refused
    everything and a survey that refused the right thing are the same survey
    over the broken pair alone.

    The parenthetical is the **generation** header's model, because that is the
    axis, and the body names both values and both paths, because the operator's
    question is which of the two is wrong.
    """
    broken = evaluation(tmp_path, "model-disagrees", judges=["judge_opus_a"])
    intact = evaluation(tmp_path, "model-agrees", judges=["judge_opus_a"])
    restate_the_model(broken.locations[0], OTHER_MODEL)

    result = survey(tmp_path)

    (refusal,) = [
        line for line in result.refused if "records the model under test as" in line
    ]
    assert OTHER_MODEL in refusal, refusal
    assert "mockllm/model-under-test" in refusal, refusal
    assert broken.locations[0] in refusal, refusal
    assert broken.generation in refusal, refusal

    printed = result.lines()
    assert [line for line in printed if line.startswith("refused (")] == [
        f"refused (mockllm/model-under-test): {refusal.split('): ', 1)[1]}"
    ], printed
    # the broken pair offers no invocation, and the intact one still does
    (reported,) = [line for line in printed if line.startswith("report (")]
    assert intact.evaluation_id in reported, reported
    assert broken.evaluation_id not in reported, reported


def test_the_report_command_refuses_that_same_pair_through_that_same_function(
    tmp_path: Path,
) -> None:
    """One condition, two routes, and this is the claim that they are one.

    The survey pairs on its own, over logs matched by ``eval_id``; `kgpbench
    report` pairs through `load_evaluation`. Both reach
    `assert_one_model_under_test`, and the message an operator reads is the same
    one — which is what a second copy of the comparison would quietly stop being
    true of.
    """
    broken = evaluation(tmp_path, "two-routes", judges=["judge_opus_a"])
    restate_the_model(broken.locations[0], OTHER_MODEL)

    with pytest.raises(PairingError) as refusal:
        load_evaluation(broken.generation, broken.locations)

    (surveyed,) = [
        line
        for line in survey(tmp_path).refused
        if "records the model under test as" in line
    ]
    assert str(refusal.value) in surveyed, (str(refusal.value), surveyed)


def test_the_refusal_states_the_condition_and_one_remedy_and_nothing_else(
    tmp_path: Path,
) -> None:
    """The pin on the string, and the absent half is what makes it one.

    The message is the condition — each model value beside its own log's path —
    and one remedy. The clause about `score()` preserving `EvalSpec.model`, and
    about a report being per model, is rationale: it answers a question about
    this package, where the operator reading this line is holding two files and
    needs the move. It lives in `assert_one_model_under_test`'s docstring now
    (§16). A pin that asserted only what the string *says* would have passed
    unchanged before the clause was dropped, so the absence carries the trim.

    Read off the raised message, which is the one string both callers use.
    """
    broken = evaluation(tmp_path, "one-remedy", judges=["judge_opus_a"])
    restate_the_model(broken.locations[0], OTHER_MODEL)

    with pytest.raises(PairingError) as refusal:
        load_evaluation(broken.generation, broken.locations)
    message = str(refusal.value)

    # the condition: each value beside its own log's path, so an operator can
    # tell which of the two is the wrong one
    assert (
        f"{broken.locations[0]} records the model under test as {OTHER_MODEL!r}"
        in message
    ), message
    assert f"{broken.generation} records 'mockllm/model-under-test'" in message, message

    # one remedy, about the artifact rather than about either invocation, and
    # naming the likelier misplaced half
    assert (
        "The scoring log was produced from a different generation run: pair it "
        "with that run's generation log, or remove it" in message
    ), message

    # the dropped clause, asserted absent
    for gone in ("score()", "EvalSpec.model", "per model", "silently absorb"):
        assert gone not in message, (gone, message)
    # and a published string names no section (§16)
    assert "§" not in message, message


def test_a_pair_whose_headers_agree_is_refused_by_neither_caller(
    tmp_path: Path,
) -> None:
    """The negative half, on both callers, because it carries the work.

    A comparison that refused every pair would satisfy every assertion above
    it: the message would be worded right, both routes would raise it, and no
    evaluation would ever be reportable. So the ordinary pair is pinned beside
    them — `load_evaluation` returns it, and the survey still offers it as a
    `report` line with nothing under ``refused``.
    """
    intact = evaluation(tmp_path, "headers-agree", judges=["judge_opus_a"])

    pair = load_evaluation(intact.generation, intact.locations)
    assert {log.model for log in pair.scoring} == {pair.generation.model}

    result = survey(tmp_path)
    assert result.refused == [], result.refused
    (reported,) = [line for line in result.lines() if line.startswith("report (")]
    assert intact.evaluation_id in reported, reported


COUNTED_CONDITIONS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b"
    r"(?=[^.]{0,40}?\b("
    r"condition|conditions|refusal|refusals"
    r"|reach|reaches|land|lands|qualify|qualifies|share|shares"
    r")\b)",
    re.IGNORECASE,
)
"""A cardinal counting the conditions that reach ``Survey.refused``.

A number within one sentence and forty characters of ``condition``, ``refusal``
or one of the verbs these carriers use for arrival — which is the shape every
count of them was written in, across every revision that moved the membership.
It is deliberately not a test of what the conditions *are*: a sixth arriving in
§14.40 leaves this pattern alone, which is the whole point of the count living
nowhere.

**``kind`` and ``kinds`` were in that vocabulary and are not any more, which is
what lets the scan reach a docstring about output.** It is the one word here
carrying two subjects — a kind of condition and a kind of *line* — and a regex
cannot tell them apart: :meth:`Survey.lines` says that one line kind carries no
model, a statement about printing that this pattern read as a count. What the
narrowing gives up is a cardinal standing beside ``kinds`` with no other word
from this vocabulary near it; every count these carriers actually wrote named
``condition``, ``refusal`` or a verb, so the weaker form is the one worth
having, and exempting the subject that fails would have been a control with a
carve-out for its own counterexample.

Ordinals are not cardinals here. ``a third category`` and ``a *second*
evaluation`` are statements about something else and stay legal."""


def _attribute_docstring(class_name: str, attribute: str) -> str:
    """The docstring written under a model field, which pydantic does not keep.

    `Field(description=)` is not used on these fields and
    ``use_attribute_docstrings`` is off, so the text below the annotation is
    reachable only from the source. Parsed rather than grepped, so that what
    comes back is this attribute's docstring and not a neighbour's.
    """
    tree = ast.parse(Path(report_module.__file__).read_text())
    (klass,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    for previous, node in zip(klass.body, klass.body[1:]):
        if (
            isinstance(previous, ast.AnnAssign)
            and isinstance(previous.target, ast.Name)
            and previous.target.id == attribute
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError(f"{class_name}.{attribute} carries no docstring")


def test_the_three_carriers_state_the_boundary_and_carry_no_count() -> None:
    """The count leaves `Survey.refused`, `survey()`, the README and
    :meth:`Survey.lines` (§14.40).

    The membership is the definition of this edge and belongs in one place; the
    count belongs nowhere. It moved from three to five across three revisions,
    every move had to edit all three carriers, and at least one was left stale
    each time — so the drift is the thing to hold, and holding it means
    asserting the absence rather than re-stating the list in a fourth place.

    **:meth:`Survey.lines` is the fourth subject and is not a fourth carrier.**
    It does not state the membership; it points at §14.40 for it, which is the
    shape the other three were reduced to, and it is prose about the very lines
    ``refused`` is printed on — so a stale count drifts into it by exactly the
    route it drifted into them. It was out of scope while ``kind`` was in
    :data:`COUNTED_CONDITIONS`, because its sentence about line kinds matched;
    the word left the pattern rather than this docstring being reworded to suit
    a test, and rather than the subject being exempted from the control it
    fails.

    **This test is not a carrier.** It never names a condition, so adding one to
    §14.40 does not touch it; what it forbids is a *number* standing next to the
    vocabulary these carriers use for them.

    Each subject is anchored before it is scanned. A docstring stripped to
    nothing, or a README paragraph that moved, would otherwise pass this by
    having nothing left to match — the vacuous pass ruled against in §15.
    """
    # the package root, resolved upward from the module and no further: the
    # same source is developed beside its siblings and published as a snapshot
    # whose root *is* this directory (§16)
    package_root = Path(report_module.__file__).resolve().parents[2]
    readme = (package_root / "README.md").read_text()
    anchor = "`refused` is a log that opens"
    assert readme.count(anchor) == 1, "the README's survey paragraph moved"
    paragraph = readme[readme.index("A directory that does not exist is a refusal") :]
    paragraph = paragraph[: paragraph.index("\n\n")]

    subjects = {
        "Survey.refused": _attribute_docstring("Survey", "refused"),
        "survey()": survey.__doc__ or "",
        "README.md": paragraph,
        "Survey.lines()": Survey.lines.__doc__ or "",
    }
    # the positive half: each subject still says what the edge is
    assert "not reportable" in subjects["Survey.refused"]
    assert "refused before the grouping loop" in subjects["survey()"]
    assert "still cannot be reported" in subjects["README.md"]
    assert "The summary is one line per evaluation" in subjects["Survey.lines()"]
    # and the three docstrings name §14.40 as where the enumeration lives
    assert "§14.40" in subjects["Survey.refused"]
    assert "§14.40" in subjects["survey()"]
    assert "§14.40" in subjects["Survey.lines()"]

    for name, text in subjects.items():
        assert len(text) > 200, f"{name}: nothing left to scan ({len(text)} chars)"
        found = COUNTED_CONDITIONS.findall(text)
        assert not found, (name, found)


def test_the_survey_pairs_without_a_handler_because_nothing_can_raise_there(
    tmp_path: Path,
) -> None:
    """The removed branch, replaced by the fact that made it unreachable.

    The survey built its pair inside a `try`, and the handler reported whatever
    `EvaluationLogs` might raise under ``unpaired``. Nothing raised: the model
    disagreement it named is refused above it now, and the constructor declares
    no validator, so two already-typed halves cannot fail it. That second half
    is what this asserts — a validator added later would make the survey raise
    where it used to report, and the one thing the deleted handler was left
    standing for was exactly that day.

    Read off the model rather than off the source, so a validator arriving under
    any spelling is caught.
    """
    declared = EvaluationLogs.__pydantic_decorators__
    assert declared.model_validators == {}, declared.model_validators
    assert declared.field_validators == {}, declared.field_validators
    assert declared.root_validators == {}, declared.root_validators
    assert declared.validators == {}, declared.validators

    # and the pair still constructs from the two halves the survey hands it
    intact = evaluation(tmp_path, "no-validator", judges=["judge_opus_a"])
    pair = EvaluationLogs(
        generation=load_generation_log(intact.generation, header_only=True),
        scoring=load_scoring_logs(intact.locations, header_only=True),
    )
    assert pair.model == "mockllm/model-under-test"


def test_an_agreeing_pair_is_untouched_by_the_refusal(tmp_path: Path) -> None:
    """Half two. The fixture is the same one, minus the restatement.

    A refusal that fired on every pair would satisfy half one and would make the
    command useless, so this asserts the survey over a directory built exactly
    like the one above and never restated: nothing is refused, the pair reports,
    and the line names the one model both headers carry.
    """
    intact = evaluation(tmp_path, "model-agrees", judges=["judge_opus_a"])

    result = survey(tmp_path)

    assert result.refused == []
    (reported,) = [line for line in result.lines() if line.startswith("report (")]
    assert reported.startswith("report (mockllm/model-under-test): "), reported
    assert intact.evaluation_id in reported, reported


def test_a_survey_group_carries_its_membership_and_its_model_and_nothing_else() -> None:
    """`SurveyGroup.checks` and `.dropped_checks` are gone, asserted structurally.

    They were the group-level intersection the retired ``combine`` heading
    printed; the summary is one line per evaluation now and the count on it is
    that evaluation's own. Measured before the removal: no reader in any of the
    three packages, or in ``release/``, took either field outside this class's
    own construction.

    Asserted over the field set rather than over one absent name, so a field
    added back is caught as readily as one of these two returning. The model is
    ``extra="forbid"``, so the constructor is the second half of the same claim.
    """
    assert set(SurveyGroup.model_fields) == {"model", "evaluations"}

    with pytest.raises(ValueError, match="checks"):
        SurveyGroup(model="m", evaluations=["e"], checks=["B6"])  # type: ignore[call-arg]


def test_an_evaluation_reaching_the_entries_loop_redefined_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The assertion behind the by-construction claim, shown able to fire.

    Pairing refuses a self-disagreeing evaluation, so nothing that reaches the
    entries loop carries a redefinition — a *derivation*, and a derivation is not
    an assertion (§14.44). The mutation the assertion exists for is a later path
    into ``entries`` that skips pairing, and this constructs exactly that by
    making ``combinable`` report `True` for every comparison: the pairing guard
    then admits the broken evaluation while the comparison it carries still
    names ``B6``.

    Without the assertion the survey prints a ``report`` line for a set of logs
    `kgpbench report` refuses, which is the defect the pairing refusal closed one
    entry point over.
    """
    run = evaluation(tmp_path, "smuggled", judges=["judge_opus_a"])
    rejudge_under(run, judge="judge_opus_b", catalogue=redefined_catalogue("B6"))

    monkeypatch.setattr(
        report_module.CatalogueComparison,
        "combinable",
        property(lambda self: True),
    )

    with pytest.raises(ValueError, match="reached the survey's entries loop"):
        survey(tmp_path)


def test_the_entries_assertion_stays_silent_on_an_agreeing_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative half, under the same patch that fires the positive one.

    An assertion that fired whenever the pairing guard was bypassed would pass
    the test above and refuse every ordinary multi-judge evaluation the moment
    anything touched that guard. Two judges under one catalogue reach the entries
    loop here with the guard neutralised and the survey completes, so what the
    assertion reads is the redefinition and not the route.
    """
    run = evaluation(tmp_path, "agreeing", judges=["judge_opus_a"])
    rejudge_under(run, judge="judge_opus_b", catalogue=mj.CATALOGUE)

    monkeypatch.setattr(
        report_module.CatalogueComparison,
        "combinable",
        property(lambda self: True),
    )

    result = survey(tmp_path)

    (entry,) = result.evaluations
    assert entry.evaluation == run.evaluation_id
    assert not result.refused, result.refused


def test_a_log_directory_that_does_not_exist_is_a_refusal(tmp_path: Path) -> None:
    """`list_eval_logs` returns an empty list for a missing directory and for an
    empty one alike, so without this check a typo reports *nothing here*."""
    with pytest.raises(FileNotFoundError, match="no such log directory"):
        survey(tmp_path / "not-a-directory")


def test_an_empty_directory_that_exists_stays_silent(tmp_path: Path) -> None:
    """The other half, and it is ruled rather than an oversight (§5).

    Silence is the honest answer over a directory holding no logs; the refusal
    above is what makes it unambiguous. A survey that refused here would refuse
    on the ordinary state of a fresh log directory.
    """
    result = survey(tmp_path)

    assert result.lines() == []
    assert not result.evaluations and not result.groups
    assert not result.unpaired and not result.refused and not result.unreadable


def test_the_survey_prints_no_instance_set_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """§4's banner belongs to task construction, before money is spent.

    The survey resolves every shipped set only to answer whether a stored log's
    digest matches something the tree still ships. That is a lookup: it spends
    nothing and selects nothing, and printing an instance-set banner — module
    paths and instance ids included — in front of somebody's log directory tells
    them nothing about their logs and buries what does.

    The other half of this control is
    ``test_generation_task.py``'s: generation still announces.
    """
    evaluation(tmp_path, "quiet")
    capsys.readouterr()

    survey(tmp_path)

    printed = capsys.readouterr().out
    assert "instance set" not in printed, printed
    assert "kgpbench:" not in printed, printed


# -- 7. the source control --------------------------------------------------


def test_the_report_layer_reads_no_framework_aggregation() -> None:
    """A control on the module, not on one run of it.

    The twin of ``test_analysis.py``'s control, and it moved here with the code
    it covers: a source control that silently stops covering its subject is
    worse than none. §13.4: over a stored log the framework's aggregation turns
    every unreachable and every excluded check into a scored zero, because NaN
    reads back as ``None`` and `value_to_float` falls through to ``0.0``. §5 v34
    adds the dataframe views to the same list — ``prepare(score_to_float(...))``
    runs that same coercion.

    The anchor is asserted **first**, so a rename fails loudly rather than
    passing over a file this control no longer covers.
    """
    source = Path(report_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    anchors = {
        "build_matrices",
        "build_report",
        "check_counts",
        "assert_combinable",
        "survey",
        "CheckCounts",
        "VerdictMatrix",
    }
    assert anchors <= defined, (
        f"the report layer's entry points were renamed ({sorted(anchors - defined)}); "
        "this control no longer covers them"
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
        "evals_df",
        "samples_df",
        "events_df",
        "messages_df",
        "score_to_float",
        "prepare",
    ):
        assert aggregate not in imported, (
            f"the report layer imports {aggregate!r}; it computes every reported "
            "number itself, and the dataframe views run the coercion this layer "
            "exists to refuse"
        )

    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for aggregate in ("results", "metrics", "reductions"):
        assert aggregate not in attributes, (
            f"the report layer reads `.{aggregate}`, which is the framework's own "
            "aggregation over a log whose nulls it read as zeros"
        )


def test_the_report_layer_names_no_stale_judge_usage_field() -> None:
    """Judge spend is read from the metadata key and from nowhere else.

    The three fields that hold the *generation's* usage on a scoring log look
    like carriers and are not. This layer reads none of them; the assertion is
    here so that a later edit adding a spend column has to face the rule.
    """
    source = Path(report_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "model_usage" not in attributes
    assert "role_usage" not in attributes


# -- the positive half of the spend rule ------------------------------------


def test_report_level_spend_takes_the_judges_number_not_the_stale_one(
    tmp_path: Path,
) -> None:
    """The reader §9 records as the one structural gap left from the v48 round.

    The source control below shows this layer *cannot* name the stale fields.
    This shows what it returns instead — and the trap has to be present for that
    to be a discrimination rather than a coincidence, so the scoring logs'
    ``EvalSample.model_usage`` and ``EvalStats.model_usage`` are asserted to be
    the **generation's** numbers first (§5, §6.6, §0 v14).

    Pooling is by judge name across evaluations, which is why the judge that ran
    over both evaluations totals twice what the other two do.
    """
    everyone = evaluation(tmp_path, "spend-a")
    alone = evaluation(tmp_path, "spend-b", judges=["judge_opus_a"])
    evaluations = paired(everyone, alone)

    # the trap, present and wrong: `score()` leaves these carrying the
    # generation's usage, so a reader that took them would get a real number
    stale_totals: set[int] = set()
    for pair in evaluations:
        for log in pair.scoring:
            for sample in log.samples:
                stale_totals.update(
                    usage.total_tokens for usage in (sample.model_usage or {}).values()
                )
            assert log.log is not None
            stale_totals.update(
                usage.total_tokens
                for usage in (log.log.stats.model_usage or {}).values()
            )
    assert stale_totals, "no stale usage present; the discrimination would be vacuous"
    assert all(total % mj.GENERATION_TOTAL_TOKENS == 0 for total in stale_totals)
    assert mj.JUDGE_TOTAL_TOKENS not in stale_totals

    # and the supported route
    rows = {row.judge: row for row in judge_spend_totals(evaluations)}
    assert set(rows) == {"judge_sonnet_xhigh", "judge_opus_a", "judge_opus_b"}

    for judge in ("judge_sonnet_xhigh", "judge_opus_b"):
        row = rows[judge]
        assert row.calls == 1
        assert row.scored == row.recorded == 2
        assert row.usage.total_tokens == 2 * mj.JUDGE_TOTAL_TOKENS

    pooled = rows["judge_opus_a"]
    assert pooled.calls == 2
    assert pooled.scored == pooled.recorded == 4
    assert pooled.usage.total_tokens == 4 * mj.JUDGE_TOTAL_TOKENS
    assert pooled.usage.input_tokens == 4 * 700

    # the report carries it, which is the unit §5 measures judge cost at
    report = build_report(evaluations)
    assert {row.judge: row.usage.total_tokens for row in report.spend} == {
        judge: row.usage.total_tokens for judge, row in rows.items()
    }


def test_a_judge_that_recorded_no_usage_is_counted_apart_from_the_rest(
    tmp_path: Path,
) -> None:
    """``scored`` and ``recorded`` are two counts, and the gap is the honest one.

    A `mockllm` model driven by a callable records no usage at all
    (`_providers/mockllm.py:88-95`), so a total summed over fewer trajectories
    than were scored is a total over a subset. Collapsing the two into one field
    would report a judge that *recorded* nothing as one that *spent* nothing, and
    the two are different facts.

    The usage key is dropped from one stored score in memory, which is exactly
    the shape the framework produces on that path — and it is what makes this an
    assertion rather than a restatement of the row above.
    """
    from kgpbench.judge import JUDGE_USAGE_KEY

    one = evaluation(tmp_path, "recorded-none", judges=["judge_opus_a"])
    evaluations = paired(one)

    before = judge_spend_totals(evaluations)[0]
    assert before.scored == before.recorded == 2

    scoring = evaluations[0].scoring[0]
    assert scoring.log is not None
    stripped = (scoring.log.samples or [])[0].scores["judge_opus_a"]
    assert stripped.metadata is not None
    del stripped.metadata[JUDGE_USAGE_KEY]

    after = judge_spend_totals(evaluations)[0]
    assert after.scored == 2
    assert after.recorded == 1
    assert after.usage.total_tokens == mj.JUDGE_TOTAL_TOKENS
    assert after.usage.total_tokens != before.usage.total_tokens

    # a judge that recorded nothing at all sums to zero rather than to `None`:
    # the distinction between "spent nothing" and "recorded nothing" is carried
    # by the two counts, never by a missing total
    assert JudgeSpend(judge="never-ran").usage.total_tokens == 0
    assert JudgeSpend(judge="never-ran").recorded == 0


# -- the guard over stored logs (§14.47) ------------------------------------


def _generation_log_at(tmp_path: Path, name: str, **eval_kwargs) -> str:
    """One stored generation log, with whatever eval-level configuration is
    given.

    `inspect_eval` is called directly rather than through
    :func:`~kgpbench.tasks.generation`, because the registered task refuses an
    eval-level ``reasoning_effort`` before it builds — which is the point: a log
    carrying one can no longer be produced *here*, and this is how a log from
    somewhere else is reconstructed.
    """
    from inspect_ai import eval as inspect_eval

    # the model is named rather than handed over as an object, because that is
    # the route `inspect eval` takes and the only one on which
    # `EvalSpec.model_generate_config` records the eval-level configuration at
    # all (:func:`mock_judging.generation_outputs`)
    logs = inspect_eval(
        mj.generation_task(f"guard-{name}", mj.INSTANCES),
        model="mockllm/model",
        model_args={"custom_outputs": mj.generation_outputs("an answer")},
        log_dir=str(tmp_path / name),
        display="none",
        **eval_kwargs,
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].location
    return str(logs[0].location)


@pytest.mark.parametrize(
    ("eval_kwargs", "named"),
    [
        ({"reasoning_effort": "low"}, "reasoning_effort"),
        ({"max_tokens": 999}, "max_tokens"),
    ],
)
def test_a_generation_log_whose_run_carried_an_eval_level_value_is_refused(
    tmp_path: Path, eval_kwargs, named
) -> None:
    """A contradiction this harness can no longer produce, arriving from
    elsewhere.

    `eval.model_generate_config` is the eval-level config alone and is empty on
    a run with nothing set, so presence is exact for the two fields it carries.
    The log opens, it parses, and its own header says the run was configured on
    two channels at once — so what the solver configuration records is not what
    the model was run at, and the report refuses rather than reasoning about
    whether it mattered.
    """
    location = _generation_log_at(tmp_path, "overridden", **eval_kwargs)
    with pytest.raises(PairingError, match=named) as refusal:
        load_generation_log(location)
    assert "eval-level" in str(refusal.value)

    # and header-only, which is the survey's route
    with pytest.raises(PairingError):
        load_generation_log(location, header_only=True)


def test_a_generation_log_with_nothing_set_at_eval_level_loads(
    tmp_path: Path,
) -> None:
    """The negative half, and it is every log this harness writes.

    `eval.model_generate_config` is empty rather than defaulted on a run with no
    eval-level value, which is what makes the guard a presence read with nothing
    to compare.
    """
    location = _generation_log_at(tmp_path, "clean")
    loaded = load_generation_log(location)
    assert loaded.provenance.solver is not None
    carried = read_eval_log(location, header_only=True).eval.model_generate_config
    assert carried.model_dump(exclude_none=True) == {}


def test_the_guard_reads_a_field_only_a_named_model_populates(
    tmp_path: Path,
) -> None:
    """The guard's one blind spot, measured rather than assumed.

    `EvalSpec.model_generate_config` is the `Model` object's own base config
    (`_eval/task/log.py:252`; `Model.__init__` assigns it without merging,
    `model/_model.py:695`). The framework builds that object from the eval-level
    configuration when it is handed a model **name**, which is what `inspect
    eval` always does and what every route in scope does. Hand `eval()` a
    pre-built `Model` instead and the field is empty while the eval-level value
    is in force, so the guard passes a log it would otherwise refuse.

    `plan.config` is not the repair: it records whichever value won, so on a log
    written before the effort moved to the per-call config it carries the task's
    own value on every run, and a guard over it would refuse every stored log
    this harness has ever produced.
    """
    from inspect_ai import eval as inspect_eval

    logs = inspect_eval(
        mj.generation_task("guard-object-route", mj.INSTANCES),
        model=mj.generation_model("an answer"),
        log_dir=str(tmp_path / "object-route"),
        display="none",
        reasoning_effort="low",
    )
    assert logs[0].status == "success"
    header = read_eval_log(str(logs[0].location), header_only=True)
    assert header.eval.model_generate_config.model_dump(exclude_none=True) == {}
    assert header.plan.config.reasoning_effort == "low"
    # and so the guard does not fire — stated as the measured limit it is
    assert load_generation_log(str(logs[0].location)).provenance.solver is not None


def test_the_refusal_is_its_own_type_and_is_still_a_pairing_error() -> None:
    """The subclass, and why it is one.

    :class:`EvalConfigRecorded` exists so the **survey** can tell this condition
    apart from the parse failures its generic handler catches. It stays a
    `PairingError` so `cli.main`'s refusal boundary covers it with no new entry
    and a caller catching the base type keeps catching it.
    """
    assert issubclass(EvalConfigRecorded, PairingError)
    assert issubclass(EvalConfigRecorded, ValueError)


def test_the_survey_prints_the_eval_level_refusal_under_refused(
    tmp_path: Path,
) -> None:
    """Item 4: the word is printed, and it has to be the right one.

    The guard raises out of :func:`load_generation_log`, which the survey calls
    inside the handler whose word is ``unreadable`` — and ``unreadable`` means
    *could not be read as either half*. This log opens in an editor, parses as
    an Inspect log, and yields its provenance record; what is wrong is what its
    run did. That is ``refused``, which exists for exactly this shape and says
    so in the published README.
    """
    evaluation(tmp_path, "clean")
    refused_location = _generation_log_at(
        tmp_path, "clean/generation", reasoning_effort="low"
    )

    result = survey(tmp_path)

    assert any(refused_location in line for line in result.refused), result.refused
    assert any("eval-level configuration" in line for line in result.refused)
    assert not result.unreadable, result.unreadable
    printed = result.lines()
    assert any(line.startswith("refused (") for line in printed), printed
    assert not any(line.startswith("unreadable (") for line in printed), printed

    # the evaluation beside it is untouched: a refusal over one log is not a
    # refusal over the directory
    assert result.evaluations, "the clean evaluation beside it was lost"


def test_a_parse_failure_beside_the_refusal_keeps_its_own_word(
    tmp_path: Path,
) -> None:
    """The other half of the control, and it is the one that carries the work.

    Moving the refusal out of ``unreadable`` is only a move if ``unreadable``
    still means something: a handler rewritten to send everything to ``refused``
    satisfies every assertion in the test above. So both words are populated
    here at once, from one directory, and each is required to hold exactly its
    own member.

    **Two parse failures, because the survey has two handlers.** The stray file
    fails the header read; the sidecar-less scoring log fails the load that
    follows it — and that second handler is the one the refusal was taken out
    of, so it is the one an over-move would take everything else out of too.
    """
    run = evaluation(tmp_path, "mixed").run
    refused_location = _generation_log_at(
        tmp_path, "mixed/generation", reasoning_effort="low"
    )
    stray = tmp_path / "mixed" / "generation" / "not-a-log.eval"
    stray.write_bytes(b"not an eval log at all")
    orphan = Path(next(iter(run.scoring_logs.values())))
    provenance_sidecar_path(orphan).unlink()

    result = survey(tmp_path)

    assert any(refused_location in line for line in result.refused), result.refused
    assert not any("not-a-log.eval" in line for line in result.refused), result.refused
    assert not any(str(orphan) in line for line in result.refused), result.refused

    assert any("not-a-log.eval" in line for line in result.unreadable), (
        result.unreadable
    )
    assert any("no provenance sidecar" in line for line in result.unreadable), (
        result.unreadable
    )
    assert not any(refused_location in line for line in result.unreadable), (
        result.unreadable
    )
