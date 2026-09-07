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
from inspect_ai.model import Model
from mock_judging import (
    CATALOGUE,
    GENERATION_TOTAL_TOKENS,
    JUDGE_TOTAL_TOKENS,
    JudgeRecorder,
    generation_log,
    judge_model,
    verdict_reply,
)

from genomics_harness import scoring
from genomics_harness.judge import JUDGE_NAMES
from genomics_harness.judge_prompt import JUDGE_PROMPT_VERSION
from genomics_harness.provenance import (
    JudgeConfig,
    compare_provenance,
    provenance_from_metadata,
    provenance_sidecar_path,
    read_run_provenance,
)
from genomics_harness.scoring import (
    CAPTURE_RESCORE_KEY,
    DEFAULT_JUDGE_CONFIGS,
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
from genomics_harness.trajectory import (
    CARRIER_KEY,
    Completeness,
    StopCause,
    carried_record,
)
from genomics_harness.verdicts import Outcome, outcome_of_score_value

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
            model="mockllm/model",
            model_roles=_roles(JudgeRecorder()),
        )


def test_an_existing_scoring_log_is_not_silently_replaced(
    clean_log: EvalLog, tmp_path: Path
) -> None:
    """A scoring log is one judge's verdicts over one corpus; replacing it loses
    the comparison it existed for."""
    recorder = JudgeRecorder()
    arguments = dict(
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        model="mockllm/model",
        model_roles=_roles(recorder),
    )
    run_judges(clean_log, **arguments)  # type: ignore[arg-type]

    with pytest.raises(FileExistsError, match="already exist"):
        run_judges(clean_log, **arguments)  # type: ignore[arg-type]

    run_judges(clean_log, overwrite=True, **arguments)  # type: ignore[arg-type]


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

    from genomics_harness.log_reading import read_log

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
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
    from genomics_harness.log_reading import read_log

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
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
    from genomics_harness.log_reading import read_log

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
    from genomics_harness.log_reading import read_log

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
    assert [config.name for config in DEFAULT_JUDGE_CONFIGS] == list(JUDGE_NAMES)
    assert [
        (config.model, config.reasoning_effort) for config in DEFAULT_JUDGE_CONFIGS
    ] == [
        ("anthropic/claude-sonnet-5", "xhigh"),
        ("anthropic/claude-opus-5", "xhigh"),
        ("anthropic/claude-opus-5", "xhigh"),
    ]
    # effort is part of a judge's identity, so it moves the digest (§7)
    sonnet = DEFAULT_JUDGE_CONFIGS[0]
    assert sonnet.digest != sonnet.model_copy(update={"reasoning_effort": "max"}).digest
    assert judge_configs(["judge_opus_a"]) == (DEFAULT_JUDGE_CONFIGS[1],)


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
    from genomics_harness.log_reading import read_log

    assert clean_log.location
    before = _digest(clean_log.location)

    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
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
    :data:`DEFAULT_JUDGE_CONFIGS` would record two Opus 5 calls that never
    happened. The roles are the thing in force, so they are what is recorded.
    """
    recorder = JudgeRecorder()
    run = run_judges(
        clean_log,
        catalogue=CATALOGUE,
        scoring_dir=tmp_path / "scoring",
        model="mockllm/model",
        model_roles=_roles(recorder),
    )

    stored = read_run_provenance(run.scoring_logs["judge_opus_b"])
    assert [judge.name for judge in stored.judges] == list(JUDGE_NAMES)
    assert {judge.model for judge in stored.judges} == {"mockllm/model"}
    assert [judge.digest for judge in stored.judges] != [
        config.digest for config in DEFAULT_JUDGE_CONFIGS
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
            configs=MOCK_CONFIGS[:2],
            model="mockllm/model",
            model_roles=_roles(recorder),
        )
