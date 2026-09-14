"""The scoring driver — the filter, the three calls, and the two traps.

`mockllm/model` throughout; no network, no keys.

Six properties:

1. **Filter first, from the generation record.** Axis 1 `CLEAN`, axis 2
   `MODEL_STOPPED`, a complete carrier, the two records agreeing — and every
   exclusion counted against its cause, never one bucket (§5, §6.3).
2. **Isolation and location.** Three calls, three files, and the generation log
   byte-identical afterwards (§6.3).
3. **The bridge.** The capture re-computed under scoring equals the one
   generation recorded (§4a Stage 2).
4. **The append shape.** One scoring log carries the generation's capture, the
   re-computed capture, and one judge's verdicts, with one judge entry in
   ``eval.scorers``.
5. **The spend guard.** The reader takes the judge's usage from `Score.metadata`
   and never from the three fields that hold the generation's (§5, §0 v14).
6. **End to end**, in the shape the driver actually runs.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model, get_model
from mock_judging import (
    CATALOGUE,
    GENERATION_TOTAL_TOKENS,
    JUDGE_TOTAL_TOKENS,
    JudgeRecorder,
    generation_log,
    judge_model,
    verdict_reply,
)

from kgpbench import scoring
from kgpbench.catalogue import Catalogue
from kgpbench.judge import JUDGE_NAMES
from kgpbench.judge_prompt import JUDGE_PROMPT_VERSION
from kgpbench.provenance import (
    JudgeConfig,
    compare_provenance,
    provenance_from_metadata,
    provenance_sidecar_path,
    read_run_provenance,
)
from kgpbench.scoring import (
    CAPTURE_RESCORE_KEY,
    REFERENCE_JUDGE_PANEL,
    build_judges,
    judge_configs,
    judge_name,
    judge_spend,
    judge_usage,
    run_judges,
    scoring_metadata,
    select_judgeable,
    total_judge_spend,
)
from kgpbench.trajectory import (
    CARRIER_KEY,
    Completeness,
    StopCause,
    carried_record,
)
from kgpbench.verdicts import Outcome, outcome_of_score_value

KIN = "a1b2c3d4e5f60718"
PANEL = "0f1e2d3c4b5a6978"
TRUNC = "9c8b7a6d5e4f3021"

VERDICTS = {
    "B6": "performed",
    "B5": "partial",
    "B1": "not_performed",
    "D4": "performed",
    "A1": "performed",
}
"""One reply covering every check any instance reaches, so the same canned
judge works on every sample."""


def _roles(recorder: JudgeRecorder) -> dict[str, str | Model]:
    """One mock model per judge role, all replying with the same verdicts."""
    return {
        name: judge_model(recorder, verdict_reply(VERDICTS)) for name in JUDGE_NAMES
    }


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def clean_log(tmp_path: Path) -> EvalLog:
    """Two judgeable trajectories and one truncated, from one generation run."""
    return generation_log(tmp_path)


# -- 1. the filter ---------------------------------------------------------


def test_the_filter_keeps_only_completed_trajectories(clean_log: EvalLog) -> None:
    """§5: axis 1 `CLEAN`, axis 2 `MODEL_STOPPED`, and both records agreeing."""
    selection = select_judgeable(clean_log)

    assert sorted(ref.sample_id for ref in selection.judgeable) == sorted([KIN, PANEL])
    assert [(e.sample_id, e.reason) for e in selection.excluded] == [
        (TRUNC, "completeness:truncated")
    ]
    assert selection.causes == {"completeness:truncated": 1}
    assert len(selection) == 2


def test_a_limit_trip_leaves_a_provisional_carrier_and_is_excluded(
    tmp_path: Path,
) -> None:
    """The normal shape of a limit trip, not a bug signal (§5).

    A limit unwinds the solver by exception before its final write
    (`_eval/task/run.py:2104-2105`), so the **provisional** record is what
    reaches the log. §5 excludes it and does *not* route it to manual review:
    only two complete records in conflict is a disagreement.

    The reported cause is ``completeness:limit_tripped`` rather than
    ``carrier:provisional``, and that ordering is deliberate — a cause the trace
    can see is reported against the trace, so the per-cause counts do not depend
    on which record was read first (§5).
    """
    limited = generation_log(tmp_path, name="limited", message_limit=2)
    selection = select_judgeable(limited)
    excluded = {e.sample_id: e for e in selection.excluded}

    assert excluded[KIN].reason == "completeness:limit_tripped"
    assert excluded[KIN].completeness is Completeness.LIMIT_TRIPPED
    assert excluded[KIN].stop_cause is None, "the solver never reached its write"

    carried = carried_record(
        next(s for s in limited.samples or [] if str(s.id) == KIN).store
    )
    assert carried is not None and not carried.complete

    # PANEL answers on its first turn, so the limit does not bind there: the
    # fixture is a mixture rather than a wholesale failure
    assert [ref.sample_id for ref in selection.judgeable] == [PANEL]


def test_the_turn_cap_is_excluded_although_the_trace_reads_clean(
    tmp_path: Path,
) -> None:
    """The one cause with no trace signature, and the reason for two records (§6.3).

    The solver stops after a turn that ended at ``tool_calls``, so the
    derivation reads `CLEAN` and has nothing to see. A driver reading only the
    derivation would judge a trajectory the harness cut short.
    """
    capped = generation_log(tmp_path, name="capped", turn_cap=1)
    selection = select_judgeable(capped)

    excluded = {e.sample_id: e for e in selection.excluded}
    assert excluded[KIN].reason == "stop_cause:turn_cap"
    assert excluded[KIN].completeness is Completeness.CLEAN, (
        "the trace says clean; only the carrier knows the harness intervened"
    )
    assert excluded[KIN].stop_cause is StopCause.TURN_CAP

    # PANEL answers on its first turn, so the cap does not bind there
    assert [ref.sample_id for ref in selection.judgeable] == [PANEL]


def test_a_trajectory_with_no_carrier_at_all_is_excluded(
    clean_log: EvalLog,
) -> None:
    """§5 wants both records. `counts_toward_rate` accepts one, and the driver
    does not.

    A trajectory with no solver record cannot be shown *not* to have hit the
    turn cap, because that cause has no trace signature. Judging it would be
    counting a trajectory whose completeness is unestablished.
    """
    stripped = []
    for sample in clean_log.samples or []:
        store = dict(sample.store)
        store.pop(CARRIER_KEY, None)
        stripped.append(sample.model_copy(update={"store": store}))
    foreign = clean_log.model_copy(update={"samples": stripped})

    selection = select_judgeable(foreign)
    assert selection.judgeable == []
    assert selection.causes["carrier:absent"] == 2
    assert selection.causes["completeness:truncated"] == 1


def test_every_exclusion_is_counted_against_its_own_cause(tmp_path: Path) -> None:
    """§5 reports exclusions per cause, never as one bucket.

    Four causes gathered from three runs into one log, because a limit is
    task-level and applies to every sample in its own run, so the shapes cannot
    otherwise coexist. The filter is a function of the stored samples, so
    combining them is legitimate — it is the same input it would see had one run
    produced all four.
    """
    clean = generation_log(tmp_path, name="mixed-clean")
    limited = generation_log(tmp_path, name="mixed-limited", message_limit=2)
    capped = generation_log(tmp_path, name="mixed-capped", turn_cap=1)

    def sample(log: EvalLog, sample_id: str):  # type: ignore[no-untyped-def]
        return next(s for s in log.samples or [] if str(s.id) == sample_id)

    foreign = sample(clean, PANEL)
    foreign = foreign.model_copy(
        update={"store": {k: v for k, v in foreign.store.items() if k != CARRIER_KEY}}
    )

    combined = clean.model_copy(
        update={
            "samples": [
                sample(clean, KIN),  # judgeable
                sample(clean, TRUNC),  # completeness:truncated
                sample(limited, KIN),  # completeness:limit_tripped
                sample(capped, KIN),  # stop_cause:turn_cap
                foreign,  # carrier:absent
            ]
        }
    )

    selection = select_judgeable(combined)
    assert [ref.sample_id for ref in selection.judgeable] == [KIN]
    assert selection.causes == {
        "completeness:truncated": 1,
        "completeness:limit_tripped": 1,
        "stop_cause:turn_cap": 1,
        "carrier:absent": 1,
    }


def test_a_trace_visible_cause_is_reported_against_the_trace(
    clean_log: EvalLog,
) -> None:
    """`OUTPUT_CAP` restates a trace fact, so the exclusion names the trace (§5).

    Otherwise the per-cause counts would depend on which record a consumer read
    first: the truncated sample carries `OUTPUT_CAP` on axis 2 *and*
    `TRUNCATED` on axis 1, and only one of them may be the reported cause.
    """
    excluded = {e.sample_id: e for e in select_judgeable(clean_log).excluded}
    assert excluded[TRUNC].reason == "completeness:truncated"
    assert excluded[TRUNC].stop_cause is StopCause.OUTPUT_CAP


# -- 2. one judge per call, three files ------------------------------------


def test_three_judges_write_three_distinct_logs(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    assert list(run.scoring_logs) == list(JUDGE_NAMES)
    locations = list(run.scoring_logs.values())
    assert len(set(locations)) == 3
    for name, location in run.scoring_logs.items():
        assert Path(location).name == f"{name}.eval"
        assert Path(location).exists()


def test_the_generation_log_on_disk_is_byte_unchanged(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """`score()` deep-copies and writes nothing (§6.3).

    The trap it guards is the *write*: `score()` returns a log whose
    ``location`` still names the generation file, so a `write_eval_log` with no
    location rewrites it with the rubric in its header.
    """
    assert clean_log.location
    before = _digest(clean_log.location)

    recorder = JudgeRecorder()
    run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    assert _digest(clean_log.location) == before


def test_the_scoring_directory_may_not_be_the_generation_directory(
    clean_log: EvalLog,
) -> None:
    """§12 Principle 4: trajectory logs are rubric-free and publish separately."""
    assert clean_log.location
    with pytest.raises(ValueError, match="generation log's own directory"):
        run_judges(
            clean_log,
            catalogue=CATALOGUE,
            scoring_dir=Path(clean_log.location).parent,
            judges=build_judges(CATALOGUE),
            model="mockllm/model",
            model_roles=_roles(JudgeRecorder()),
        )


def test_an_existing_scoring_log_is_not_silently_replaced(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """A scoring log is one judge's verdicts over one corpus; replacing it loses
    the comparison it existed for.

    That sentence is this docstring's and no longer the message's. A published
    refusal names the condition and the remedy, and the reasoning moves here,
    beside the code (§14.40) — so the pin asserts the two halves that survive
    **and** the absence of the two clauses that went, which is what makes the
    old string coming back a failure rather than a wash.
    """
    recorder = JudgeRecorder()
    arguments = dict(
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )
    run_judges(clean_log, **arguments)  # type: ignore[arg-type]

    with pytest.raises(FileExistsError, match="already exist") as raised:
        run_judges(clean_log, **arguments)  # type: ignore[arg-type]

    run_judges(clean_log, overwrite=True, **arguments)  # type: ignore[arg-type]

    # the remedy this string names has to be one a shipped command can perform.
    # `--overwrite` came off `kgpbench judge` at §14.40, so naming `overwrite=`
    # here would name a Python parameter no published command can reach — which
    # is the shape §14.38 was opened on, reappearing at the moment the flag left.
    # Pinned as a literal rather than against the source that writes it, because
    # what this string is for is what it tells someone who is not us (§15).
    message = str(raised.value)
    assert "scoring logs already exist" in message
    assert "`--scoring-dir`" in message
    assert "overwrite=" not in message, (
        "the refusal offers a Python parameter as its remedy again; the remedy a "
        "published string can name is a different scoring directory"
    )
    for rationale in (
        "loses the comparison it existed for",
        "outside the harness",
    ):
        assert rationale not in message, (
            f"design rationale is back in the refusal: {rationale!r}. It belongs "
            "in `run_judges`'s docstring, beside the code"
        )


def test_the_overwrite_guard_fires_on_a_sidecar_with_no_log_beside_it(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """F2: the guard is over the pair, and the orphan half has its own message.

    `run_judges` writes a ``.eval`` and a ``.eval.provenance.json`` per judge,
    and the existence check read the ``.eval`` names alone — so a directory
    holding a sidecar with no log beside it passed a run that had asked not to
    overwrite anything, and had its sidecar silently replaced. That is the state
    deleting one file of the pair leaves, and deleting a scoring log by hand is
    the documented route since ``--overwrite`` came off the command (§14.40).

    **This is a behaviour change, not a repair**: the harness now refuses where
    it previously proceeded, in the paid path, and that is intended. Its control
    is the test below — a genuinely empty directory must still run, because a
    guard that refuses on everything passes this one.
    """
    recorder = JudgeRecorder()
    scoring = tmp_path / "scoring"
    arguments = dict(
        catalogue=CATALOGUE,
        scoring_dir=scoring,
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )
    run = run_judges(clean_log, **arguments)  # type: ignore[arg-type]
    log = Path(run.scoring_logs["judge_opus_a"])
    sidecar = provenance_sidecar_path(log)
    before = sidecar.read_bytes()

    log.unlink()  # the half-deletion, by hand, exactly as an operator does it
    assert sidecar.exists()

    with pytest.raises(FileExistsError) as raised:
        run_judges(clean_log, **arguments)  # type: ignore[arg-type]

    message = str(raised.value)
    assert str(sidecar) in message, message
    assert "no scoring log beside them" in message, message
    assert "Delete them" in message, message
    assert "scoring logs already exist" not in message, (
        "the orphan case got the log-present message; the two conditions are "
        "told apart because their remedies differ"
    )
    assert sidecar.read_bytes() == before, "the orphan sidecar was replaced anyway"


def test_the_overwrite_guard_does_not_fire_on_an_empty_scoring_directory(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The half that matters: a guard refusing on everything passes the test above.

    An empty directory that exists is the ordinary first run — `run_judges`
    creates the directory itself before the guard reads it — so the widened
    check must see neither a log nor a sidecar there and proceed.
    """
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    assert not list(scoring.iterdir())

    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=scoring,
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        model="mockllm/model",
        model_roles=_roles(JudgeRecorder()),
    )

    assert Path(run.scoring_logs["judge_opus_a"]).exists()
    assert Path(run.provenance_sidecars["judge_opus_a"]).exists()


def test_an_unrelated_file_in_the_scoring_directory_does_not_fire_the_guard(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The guard is over the two names this run would write, not over the directory.

    A second judge's pair, or a stray file, is not this judge's half-deletion.
    Without this the widened check would read *anything is here* as *the pair is
    broken*, which refuses a legitimate re-judge of one judge beside another's
    output — the shape one shared ``--scoring-dir`` is built on (§14.40).
    """
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    (scoring / "judge_opus_b.eval.provenance.json").write_text("{}", encoding="utf-8")
    (scoring / "notes.txt").write_text("not ours", encoding="utf-8")

    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=scoring,
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        model="mockllm/model",
        model_roles=_roles(JudgeRecorder()),
    )

    assert Path(run.scoring_logs["judge_opus_a"]).exists()


def test_the_overwrite_parameter_stays_reachable_from_python(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The capability is not what came off the surface — the flag is (§14.40).

    The other half of the removal: `kgpbench judge` carries no `--overwrite`
    (`test_cli.py`), and this function still takes one. A test that asserted only
    the absence would pass equally over a `run_judges` that had lost the
    parameter, which is a different and larger change than the one ruled.
    """
    recorder = JudgeRecorder()
    arguments = dict(
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )
    run = run_judges(clean_log, **arguments)  # type: ignore[arg-type]
    first = {
        name: Path(location).stat().st_mtime_ns
        for name, location in run.scoring_logs.items()
    }

    again = run_judges(clean_log, overwrite=True, **arguments)  # type: ignore[arg-type]

    assert again.scoring_logs == run.scoring_logs
    assert any(
        Path(location).stat().st_mtime_ns != first[name]
        for name, location in again.scoring_logs.items()
    ), "overwrite=True replaced nothing"


def test_overwrite_rewrites_the_sidecar_and_not_only_the_log(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """§14.40 records this as unmeasured; it is measured here and pinned.

    The hazard it names is the §7 evidence failure — an overwrite leaving a
    sidecar that names the previous pass's catalogue beside the new pass's
    verdicts. It does not arise: `write_run_provenance` is called inside the
    per-judge loop and is not conditioned on the guard, so the pair moves
    together. The second pass runs under a **different** catalogue, which is what
    makes the assertion able to fail — under one catalogue the sidecar's bytes
    would be identical whether it was rewritten or not (§15).
    """
    recorder = JudgeRecorder()
    second = Catalogue(
        release=CATALOGUE.release,
        checks=CATALOGUE.checks,
        notes="a second catalogue, so the recorded digest moves if it is rewritten",
    )
    assert second.digest != CATALOGUE.digest

    def one(catalogue: Catalogue, *, overwrite: bool) -> None:
        run_judges(
            clean_log,
            catalogue=catalogue,
            scoring_dir=tmp_path / "scoring",
            judges=build_judges(catalogue, ["judge_opus_a"]),
            model="mockllm/model",
            model_roles={"judge_opus_a": _roles(recorder)["judge_opus_a"]},
            overwrite=overwrite,
        )

    one(CATALOGUE, overwrite=False)
    log = tmp_path / "scoring" / "judge_opus_a.eval"
    assert read_run_provenance(log).catalogue_sha256 == CATALOGUE.digest

    one(second, overwrite=True)

    assert read_run_provenance(log).catalogue_sha256 == second.digest, (
        "the sidecar kept the previous pass's catalogue beside the new pass's verdicts"
    )


def test_no_judge_sees_another_judges_model_events(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """Separate calls are isolated; scorers inside one call are not (§6.6).

    Each scoring log carries exactly one judge's generation per judged sample —
    two samples here — plus the generation's own turns. A second judge's call
    landing in the first judge's log would mean the driver had put both in one
    `score()` call.
    """
    from inspect_ai.event import ModelEvent

    from kgpbench.log_reading import read_log

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    for location in run.scoring_logs.values():
        scored = read_log(location)
        for sample in scored.samples or []:
            generation_turns = len(
                [
                    event
                    for event in (
                        next(
                            s
                            for s in clean_log.samples or []
                            if str(s.id) == str(sample.id)
                        ).events
                    )
                    if isinstance(event, ModelEvent)
                ]
            )
            scoring_turns = len([e for e in sample.events if isinstance(e, ModelEvent)])
            assert scoring_turns == generation_turns + 1, (
                "exactly one judge call per judged sample per scoring log"
            )


# -- 3. the bridge ---------------------------------------------------------


def test_the_capture_recomputed_under_scoring_matches_the_generation_digest(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """§4a's bridge, measured in one pair of files.

    ``action="append"`` carries the generation's `render_capture` into the
    scoring log, so the capture the driver runs beside the judge collides with
    it and `unique_scorer_name` suffixes the second. Both are in one log and
    must agree — and the judge's own ``rendering_sha256`` must agree with them,
    because that is the digest of the bytes it actually put in its prompt.
    """
    from kgpbench.log_reading import read_log

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    generated = {
        str(sample.id): (sample.scores or {})["render_capture"].value
        for sample in clean_log.samples or []
    }

    for judge, location in run.scoring_logs.items():
        scored = read_log(location)
        for sample in scored.samples or []:
            scores = sample.scores or {}
            assert scores[CAPTURE_RESCORE_KEY].value == generated[str(sample.id)], (
                f"{judge} re-scored a different rendering for {sample.id}"
            )
            assert scores["render_capture"].value == generated[str(sample.id)]
            metadata = scores[judge].metadata or {}
            assert metadata["rendering_sha256"] == generated[str(sample.id)]


# -- 4. the append shape ---------------------------------------------------


def test_a_scoring_log_carries_verdicts_beside_the_capture(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    from kgpbench.log_reading import read_log

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE, ["judge_sonnet_xhigh"]),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    scored = read_log(run.scoring_logs["judge_sonnet_xhigh"])
    for sample in scored.samples or []:
        scores = sample.scores or {}
        assert set(scores) == {
            "render_capture",
            CAPTURE_RESCORE_KEY,
            "judge_sonnet_xhigh",
        }
        value = scores["judge_sonnet_xhigh"].value
        assert isinstance(value, dict)
        assert list(value) == CATALOGUE.ids

    judges = [
        entry.name
        for entry in scored.eval.scorers or []
        if entry.name.startswith("judge")
    ]
    assert judges == ["judge_sonnet_xhigh"], "one judge's configuration per log"


# -- 5. the spend guard ----------------------------------------------------


def test_the_spend_reader_takes_the_judges_number_not_the_stale_one(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The §7 evidence failure, guarded rather than trusted (§5, §0 v14).

    The trap has to be present for the test to mean anything: the scoring log's
    ``EvalSample.model_usage`` and ``EvalStats.model_usage`` carry the
    *generation's* numbers, unchanged by `score()`, and they are asserted here
    so that "the reader returned the judge's number" is a discrimination and not
    a coincidence.
    """
    from kgpbench.log_reading import read_log

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE, ["judge_opus_b"]),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )
    scored = read_log(run.scoring_logs["judge_opus_b"])

    # the stale fields, present and wrong: unchanged by `score()`, so still the
    # generation's numbers
    for sample in scored.samples or []:
        totals = [usage.total_tokens for usage in sample.model_usage.values()]
        assert totals and all(
            total % GENERATION_TOTAL_TOKENS == 0 and total != JUDGE_TOTAL_TOKENS
            for total in totals
        )
    assert [u.total_tokens for u in scored.stats.model_usage.values()] == [
        u.total_tokens for u in clean_log.stats.model_usage.values()
    ]

    # and the supported route, which is the judge's own
    spend = judge_spend(scored, "judge_opus_b")
    assert len(spend) == 2
    assert {usage.total_tokens for usage in spend.values() if usage} == {
        JUDGE_TOTAL_TOKENS
    }
    assert total_judge_spend(scored, "judge_opus_b").total_tokens == (
        2 * JUDGE_TOTAL_TOKENS
    )
    assert total_judge_spend(scored, "judge_opus_b").input_tokens == 1400


def test_the_spend_reader_names_no_stale_field_in_its_source() -> None:
    """A control on the reader, not on one run of it (§5, §0 v14).

    Three fields hold the generation's usage on a scoring log and all three look
    like carriers. A behavioural test shows the reader did not use them *this
    time*; this shows it cannot. The anchor is asserted first, because a rule
    that finds no function reports clean.
    """
    tree = ast.parse(Path(scoring.__file__).read_text(encoding="utf-8"))
    readers = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"judge_usage", "judge_spend", "total_judge_spend"}
    }
    assert set(readers) == {"judge_usage", "judge_spend", "total_judge_spend"}, (
        "the spend readers were renamed; this control no longer covers them"
    )

    for name, node in readers.items():
        attributes = {
            item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
        }
        for stale in ("model_usage", "role_usage", "stats"):
            assert stale not in attributes, (
                f"{name} reads `.{stale}`; on a scoring log that field holds the "
                "generation's usage, which is a real number that is not the "
                "judge's"
            )
        names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
        assert "ScoreEvent" not in names


def test_recompute_metrics_is_never_called_anywhere_in_the_harness() -> None:
    """§13.4: over a stored log it turns every unreachable and every excluded
    check into a scored zero, silently, because NaN reads back as ``None``.

    A call or an import, not a mention: `analysis.py` names it in prose, which
    is where the reason for not calling it is written down.
    """
    root = Path(scoring.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                called = (
                    target.id
                    if isinstance(target, ast.Name)
                    else target.attr
                    if isinstance(target, ast.Attribute)
                    else None
                )
                assert called != "recompute_metrics", f"{path}:{node.lineno}"
            if isinstance(node, ast.ImportFrom):
                assert "recompute_metrics" not in {a.name for a in node.names}, (
                    f"{path}:{node.lineno}"
                )


# -- 6. configuration ------------------------------------------------------


def test_the_judge_panel_is_configuration_and_matches_section_5() -> None:
    """One Sonnet 5 and two Opus 5, all three at xhigh (§5).

    Asserted on the configuration objects rather than on a scorer body: a judge
    resolves its model by role, so no provider string appears in scoring logic.
    """
    assert [config.name for config in REFERENCE_JUDGE_PANEL] == list(JUDGE_NAMES)
    assert [
        (config.model, config.reasoning_effort) for config in REFERENCE_JUDGE_PANEL
    ] == [
        ("anthropic/claude-sonnet-5", "xhigh"),
        ("anthropic/claude-opus-5", "xhigh"),
        ("anthropic/claude-opus-5", "xhigh"),
    ]
    # effort is part of a judge's identity, so it moves the digest (§7)
    sonnet = REFERENCE_JUDGE_PANEL[0]
    assert sonnet.digest != sonnet.model_copy(update={"reasoning_effort": "max"}).digest
    assert judge_configs(["judge_opus_a"]) == (REFERENCE_JUDGE_PANEL[1],)


def test_a_judges_name_comes_from_its_registered_factory() -> None:
    for name, judge in zip(JUDGE_NAMES, build_judges(CATALOGUE)):
        assert judge_name(judge) == name


# -- 7. end to end ---------------------------------------------------------


def test_end_to_end(clean_log: EvalLog, tmp_path: Path) -> None:
    """One generation log, filtered, judged three times, written three times.

    The shape §6.3 specifies, asserted as one sequence: the excluded trajectory
    was never judged, the three logs are present and distinct, the generation
    log is untouched, and every scoring log carries dense verdicts, the digest
    bridge and the judge's spend.
    """
    from kgpbench.log_reading import read_log

    assert clean_log.location
    before = _digest(clean_log.location)

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    # the filter ran, and the truncated trajectory was never rendered for a judge
    assert sorted(ref.sample_id for ref in run.selection.judgeable) == sorted(
        [KIN, PANEL]
    )
    assert [e.sample_id for e in run.selection.excluded] == [TRUNC]
    assert len(recorder.prompts) == 6, "two judged trajectories, three judges"
    assert not any("TRUNC." in prompt for prompt in recorder.prompts)
    assert run.parse_failures == []

    # three logs, distinct, and the generation log untouched
    assert len(set(run.scoring_logs.values())) == 3
    assert _digest(clean_log.location) == before

    generated = {
        str(sample.id): (sample.scores or {})["render_capture"].value
        for sample in clean_log.samples or []
    }

    for judge, location in run.scoring_logs.items():
        scored = read_log(location)
        stored_ids = [str(sample.id) for sample in scored.samples or []]
        assert TRUNC not in stored_ids
        assert len(stored_ids) == 2

        for sample in scored.samples or []:
            judged = (sample.scores or {})[judge]
            value = judged.value
            assert isinstance(value, dict)
            assert list(value) == CATALOGUE.ids
            # off disk, NaN has become JSON `null` and reads back as `None`
            # (§6.2 invariant 1) — which is why the analysis layer maps it
            # explicitly rather than letting `value_to_float` reach 0.0
            assert value["D1"] is None
            assert outcome_of_score_value(value["D1"]) is Outcome.NOT_REACHABLE

            metadata = judged.metadata or {}
            assert metadata["rendering_sha256"] == generated[str(sample.id)]
            assert metadata["catalogue_sha256"] == CATALOGUE.digest
            usage = judge_usage(judged)
            assert usage is not None and usage.total_tokens == JUDGE_TOTAL_TOKENS

    summary = scoring_metadata(run)
    assert summary["judged"] == 2
    assert summary["exclusion_causes"] == {"completeness:truncated": 1}
    assert summary["parse_failures"] == []


def test_a_parse_failure_reaches_the_run_summary(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """A judge that cannot be read goes to review, not into a rate (§5, §0 v14)."""
    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        model="mockllm/model",
        model_roles={"judge_opus_a": judge_model(recorder, "I liked it.")},
    )

    assert {failure.sample_id for failure in run.parse_failures} == {KIN, PANEL}
    assert all(failure.judge == "judge_opus_a" for failure in run.parse_failures)
    assert scoring_metadata(run)["parse_failures"]


# -- 8. the judge provenance sidecar ---------------------------------------


MOCK_CONFIGS = (
    JudgeConfig(
        name="judge_sonnet_xhigh",
        model="mockllm/model",
        reasoning_effort="xhigh",
        prompt_version=JUDGE_PROMPT_VERSION,
    ),
    JudgeConfig(
        name="judge_opus_a",
        model="mockllm/model",
        reasoning_effort="xhigh",
        prompt_version=JUDGE_PROMPT_VERSION,
    ),
    JudgeConfig(
        name="judge_opus_b",
        model="mockllm/model",
        reasoning_effort="xhigh",
        prompt_version=JUDGE_PROMPT_VERSION,
    ),
)
"""§5's panel shape against the mock: the models are what the fixture wires, so
the record is true of the run rather than of the panel it imitates."""


def test_the_scoring_run_writes_the_judge_digest_beside_each_log(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """`cleanups-august-2026.md` §5.1: the judge axis reaches no field of the log.

    `score()` takes no task metadata, so `JudgeConfig.digest` — the value
    `compare_provenance` reads on the ``judges`` axis — was computed, compared
    and never written. Asserted on **the digest that lands**, not on the file
    existing: a sidecar carrying the wrong configuration is exactly as present
    as one carrying the right one.
    """
    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        configs=MOCK_CONFIGS,
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    assert list(run.provenance_sidecars) == list(JUDGE_NAMES)
    for name, log_location in run.scoring_logs.items():
        sidecar = Path(run.provenance_sidecars[name])
        assert sidecar == provenance_sidecar_path(log_location)
        assert sidecar.exists()

        stored = read_run_provenance(sidecar)
        # the digest, per judge, in panel order — the whole point of the write
        assert [judge.digest for judge in stored.judges] == [
            config.digest for config in MOCK_CONFIGS
        ]
        # and the effort verbatim, which is what the digest is a digest of
        assert [judge.reasoning_effort for judge in stored.judges] == [
            "xhigh",
            "xhigh",
            "xhigh",
        ]

    # re-reading reconstructs the axis `compare_provenance` compares on: no
    # mismatch against the record the run returned, and a `judges` mismatch —
    # and only that — against the same record at a different effort
    reread = read_run_provenance(run.scoring_logs["judge_opus_a"])
    assert run.provenance is not None
    assert compare_provenance(reread, run.provenance) == []

    at_high = reread.model_copy(
        update={
            "judges": [
                judge.model_copy(update={"reasoning_effort": "high"})
                for judge in reread.judges
            ]
        }
    )
    assert [m.axis for m in compare_provenance(reread, at_high)] == ["judges"]


def test_the_sidecar_carries_the_generation_axes_and_this_passs_catalogue(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The five axes a comparison needs, from the two runs that own them.

    Instances, solver and renderer belong to the generation and are copied
    across unchanged; the catalogue belongs to *this* pass, because a generation
    log carries none at all — that absence is what makes a trajectory log
    rubric-free (§12 Principle 4), and it is why the scoring record cannot
    simply be the generation record.
    """
    generation = provenance_from_metadata(clean_log.eval.metadata)
    assert generation is not None
    assert generation.catalogue_sha256 is None, (
        "a generation log is rubric-free; if it carries a catalogue digest this "
        "test is measuring the wrong thing"
    )

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        configs=MOCK_CONFIGS,
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    stored = read_run_provenance(run.scoring_logs["judge_sonnet_xhigh"])
    assert stored.instances_sha256 == generation.instances_sha256
    assert stored.solver is not None and generation.solver is not None
    assert stored.solver.digest == generation.solver.digest
    assert stored.renderer_version == generation.renderer_version
    assert stored.catalogue_sha256 == CATALOGUE.digest
    assert stored.catalogue_release == CATALOGUE.release
    assert scoring_metadata(run)["provenance_sidecars"] == run.provenance_sidecars


def test_a_generation_log_with_no_provenance_record_is_refused(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """Refusing beats inventing. A sidecar naming an instances digest nobody
    supplied would compare **equal** to another run's invented one, and the
    comparison the record exists to refuse would go through."""
    stripped = clean_log.model_copy(
        update={"eval": clean_log.eval.model_copy(update={"metadata": {}})}
    )
    recorder = JudgeRecorder()
    with pytest.raises(ValueError, match="carries no provenance record"):
        run_judges(
            stripped,
            catalogue=CATALOGUE,
            scoring_dir=tmp_path / "scoring",
            judges=build_judges(CATALOGUE),
            configs=MOCK_CONFIGS,
            model="mockllm/model",
            model_roles=_roles(recorder),
        )
    assert recorder.prompts == [], (
        "the refusal must land before the first judge call, not after a panel "
        "of judge tokens has been spent"
    )


def test_an_undeclared_panel_is_recorded_from_the_roles_that_ran(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """``configs=None`` records what was wired, not what the panel says.

    The hazard is specific: every driver test here points the three judge roles
    at `mockllm/model`, and a sidecar defaulting to
    :data:`REFERENCE_JUDGE_PANEL` would record two Opus 5 calls that never
    happened. The roles are the thing in force, so they are what is recorded.
    """
    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    stored = read_run_provenance(run.scoring_logs["judge_opus_b"])
    assert [judge.name for judge in stored.judges] == list(JUDGE_NAMES)
    assert {judge.model for judge in stored.judges} == {"mockllm/model"}
    assert [judge.digest for judge in stored.judges] != [
        config.digest for config in REFERENCE_JUDGE_PANEL
    ]


def test_declared_configs_must_cover_every_judge_that_runs(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """Half a declaration is worse than none: the sidecar would carry one
    configuration the caller wrote and two the panel guessed, with nothing in it
    saying which is which."""
    recorder = JudgeRecorder()
    with pytest.raises(KeyError, match="judge_opus_b"):
        run_judges(
            clean_log,
            catalogue=CATALOGUE,
            scoring_dir=tmp_path / "scoring",
            judges=build_judges(CATALOGUE),
            configs=MOCK_CONFIGS[:2],
            model="mockllm/model",
            model_roles=_roles(recorder),
        )


# -- 9. nothing fires the reference panel implicitly (§14.42) ---------------
#
# Four refusals, each with both halves: the bad input raises, and the good input
# next to it still runs. Two rungs used to make `REFERENCE_JUDGE_PANEL` fire
# without anyone naming it, and the second wrote a sidecar declaring Anthropic
# models beside a `mockllm` transcript (§0 v62).


def test_a_run_with_no_judges_named_refuses_and_names_what_to_pass(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """``judges=`` is required, and empty is not a run.

    Two shapes, because they fail in two places. **Omitting the argument** is a
    `TypeError` from the interpreter and a `mypy --strict` error at the call
    site — the omission is unrepresentable rather than diagnosed, which is what
    stops the panel becoming a default again. **An empty sequence** is the one
    way past a required parameter to "no judges", and it used to return a
    `ScoringRun` that had judged nothing and reported success; it now refuses
    with a message naming what to pass.
    """
    recorder = JudgeRecorder()
    arguments = dict(
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    with pytest.raises(TypeError, match="judges"):
        run_judges(clean_log, **arguments)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="build_judges") as empty:
        run_judges(clean_log, judges=[], **arguments)  # type: ignore[arg-type]
    assert "no judges" in str(empty.value)

    # nothing was spent on either, and this refusal lands so early that even
    # the scoring directory's own `mkdir` has not happened
    assert recorder.prompts == []
    assert not (tmp_path / "scoring").exists()

    # and the good input beside it: one judge, named, runs
    run = run_judges(
        clean_log,
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        **arguments,  # type: ignore[arg-type]
    )
    assert list(run.scoring_logs) == ["judge_opus_a"]
    assert recorder.prompts != []


def test_partial_role_coverage_refuses_and_names_the_uncovered_judges(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The rung that recorded the reference panel for judges it never ran.

    The coverage test used to be an ``all(...)``, so **one** uncovered judge
    discarded **every** wired role and recorded the shipped Anthropic panel for
    all three. The artifact was a scoring log whose own model events said
    `mockllm/model` beside a sidecar declaring Sonnet 5 at ``xhigh`` — written to
    disk, and surviving the failure the next judge raised (§0 v62).

    So this asserts the whole of that: the refusal, that it names the uncovered
    judge, that no model was called, and that **no scoring log and no sidecar
    exist** — the last is what the old behaviour left behind.
    """
    recorder = JudgeRecorder()
    scoring = tmp_path / "scoring"
    wired = _roles(recorder)
    partial = {name: role for name, role in wired.items() if name != "judge_opus_b"}

    with pytest.raises(KeyError, match="judge_opus_b") as refusal:
        run_judges(
            clean_log,
            catalogue=CATALOGUE,
            scoring_dir=scoring,
            judges=build_judges(CATALOGUE),
            model="mockllm/model",
            model_roles=partial,
        )
    assert "judge_sonnet_xhigh" in str(refusal.value), (
        "the message names what is covered as well as what is not"
    )
    assert recorder.prompts == [], "the refusal lands before the first judge call"
    assert list(scoring.glob("*.eval")) == []
    assert list(scoring.glob("*.provenance.json")) == []

    # the good input: the same judges, covered, record the roles that ran
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=scoring,
        judges=build_judges(CATALOGUE),
        model="mockllm/model",
        model_roles=wired,
    )
    stored = read_run_provenance(run.scoring_logs["judge_opus_b"])
    assert {judge.model for judge in stored.judges} == {"mockllm/model"}


def test_running_the_two_covered_judges_is_the_supported_way_to_run_two(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The refusal has a remedy, and it is naming the judges rather than the roles.

    The partial-coverage case above is a caller who wired two roles and asked for
    three judges. Asking for the two they wired runs, records those two, and
    reaches no panel at all — which is what makes the refusal a message about the
    request rather than a limitation.
    """
    recorder = JudgeRecorder()
    wired = {
        name: judge_model(recorder, verdict_reply(VERDICTS))
        for name in ("judge_sonnet_xhigh", "judge_opus_a")
    }

    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE, list(wired)),
        model="mockllm/model",
        model_roles=wired,
    )

    assert list(run.scoring_logs) == list(wired)
    stored = read_run_provenance(run.scoring_logs["judge_opus_a"])
    assert [judge.name for judge in stored.judges] == list(wired)


def test_a_run_declaring_nothing_and_wiring_nothing_refuses(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """Rung 3 removed: with no ``configs`` and no ``model_roles`` there is
    nothing to read a configuration off, and the reference panel is not it.

    Before v62 this recorded `REFERENCE_JUDGE_PANEL` — a full Anthropic panel —
    and then failed inside the first `score()` call, because a judge resolves its
    model through a role that was never wired. The record was written first.
    """
    recorder = JudgeRecorder()
    with pytest.raises(KeyError, match="model_roles"):
        run_judges(
            clean_log,
            catalogue=CATALOGUE,
            scoring_dir=tmp_path / "scoring",
            judges=build_judges(CATALOGUE, ["judge_opus_a"]),
            model="mockllm/model",
        )
    assert recorder.prompts == []

    # the good input: the same run, with the one role it needs
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        model="mockllm/model",
        model_roles={"judge_opus_a": judge_model(recorder, verdict_reply(VERDICTS))},
    )
    assert list(run.scoring_logs) == ["judge_opus_a"]


def test_a_declaration_contradicting_the_wired_role_is_refused(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """Passing ``configs`` and ``model_roles`` together was legal and silent.

    The record followed the first and the run followed the second, so a sidecar
    naming Opus 5 at ``xhigh`` beside a `mockllm` transcript was reachable — the
    §7 evidence failure, and worse than a missing record because it is plausible.

    Both halves of the message are asserted: the declared value and the wired
    one. A message naming only one of them does not let a reader see which is
    wrong.
    """
    recorder = JudgeRecorder()
    wired = {"judge_opus_a": judge_model(recorder, verdict_reply(VERDICTS))}
    lying = (
        JudgeConfig(
            name="judge_opus_a",
            model="anthropic/claude-opus-5",
            reasoning_effort="xhigh",
        ),
    )

    with pytest.raises(ValueError, match="contradicts") as refusal:
        run_judges(
            clean_log,
            catalogue=CATALOGUE,
            scoring_dir=tmp_path / "scoring",
            judges=build_judges(CATALOGUE, ["judge_opus_a"]),
            configs=lying,
            model="mockllm/model",
            model_roles=wired,
        )
    message = str(refusal.value)
    assert "judge_opus_a" in message
    assert "anthropic/claude-opus-5" in message and "mockllm/model" in message
    assert recorder.prompts == []

    # the good input: the same declaration told the truth about the model. The
    # effort it declares is not compared, because the role does not carry one —
    # an unset knob is one nobody asked for, not one set to nothing
    truthful = (lying[0].model_copy(update={"model": "mockllm/model"}),)
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        configs=truthful,
        model="mockllm/model",
        model_roles=wired,
    )
    stored = read_run_provenance(run.scoring_logs["judge_opus_a"])
    assert [judge.model for judge in stored.judges] == ["mockllm/model"]
    assert [judge.reasoning_effort for judge in stored.judges] == ["xhigh"]


def test_a_declared_effort_contradicting_the_wired_effort_is_refused(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """The comparison reaches every field the role carries, not the model alone.

    Effort is the one that matters most: the same model at a different effort is
    a different judge (§7), and it is invisible in a transcript, so a record that
    disagrees with the request on it cannot be caught by reading the log.
    """
    recorder = JudgeRecorder()
    at_high = get_model(
        "mockllm/model",
        config=GenerateConfig(reasoning_effort="high"),
        custom_outputs=[],
        memoize=False,
    )
    wired: dict[str, str | Model] = {"judge_opus_a": at_high}
    declared = (
        JudgeConfig(
            name="judge_opus_a",
            model="mockllm/model",
            reasoning_effort="xhigh",
        ),
    )

    with pytest.raises(ValueError, match="reasoning_effort") as refusal:
        run_judges(
            clean_log,
            catalogue=CATALOGUE,
            scoring_dir=tmp_path / "scoring",
            judges=build_judges(CATALOGUE, ["judge_opus_a"]),
            configs=declared,
            model="mockllm/model",
            model_roles=wired,
        )
    assert "'xhigh'" in str(refusal.value) and "'high'" in str(refusal.value)
    assert recorder.prompts == []

    # the good input: the declaration agreeing with the role, which then runs
    agreeing = (declared[0].model_copy(update={"reasoning_effort": "high"}),)
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        configs=agreeing,
        model="mockllm/model",
        model_roles={
            "judge_opus_a": judge_model(recorder, verdict_reply(VERDICTS)),
        },
    )
    stored = read_run_provenance(run.scoring_logs["judge_opus_a"])
    assert [judge.reasoning_effort for judge in stored.judges] == ["high"]


def test_the_recorded_prompt_version_is_stamped_and_never_believed(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """`prompt_version` is not caller-supplied information.

    `build_judge_prompt` takes no version parameter and there is no other
    builder, so no judge can run at any version but
    :data:`JUDGE_PROMPT_VERSION`. A sidecar declaring ``j1`` beside scores
    recording the live version was reachable before v62; it is now stamped on
    both rungs, and a declared value is overwritten rather than carried.
    """
    recorder = JudgeRecorder()
    stale = (
        JudgeConfig(
            name="judge_opus_a",
            model="mockllm/model",
            prompt_version="j1",
        ),
    )

    declared_run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "declared",
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        configs=stale,
        model="mockllm/model",
        model_roles={"judge_opus_a": judge_model(recorder, verdict_reply(VERDICTS))},
    )
    stored = read_run_provenance(declared_run.scoring_logs["judge_opus_a"])
    assert [judge.prompt_version for judge in stored.judges] == [JUDGE_PROMPT_VERSION]

    # the sidecar and the scores now agree, which is the property the stamp buys
    from kgpbench.log_reading import read_log

    scored = read_log(declared_run.scoring_logs["judge_opus_a"])
    for sample in scored.samples or []:
        metadata = (sample.scores or {})["judge_opus_a"].metadata or {}
        assert metadata["prompt_version"] == JUDGE_PROMPT_VERSION

    # and the role rung stamps it too — a wired `Model` carries no prompt
    # version at all, so before v62 this axis was simply absent from the record
    role_run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "wired",
        judges=build_judges(CATALOGUE, ["judge_opus_a"]),
        model="mockllm/model",
        model_roles={"judge_opus_a": judge_model(recorder, verdict_reply(VERDICTS))},
    )
    from_role = read_run_provenance(role_run.scoring_logs["judge_opus_a"])
    assert [judge.prompt_version for judge in from_role.judges] == [
        JUDGE_PROMPT_VERSION
    ]
