"""What a scorer's own model call does to the trajectory record — §14.27.

Judging is a second pass, so every verdict is produced by a scorer running over a
`TaskState` and a transcript that `_eval/score.py` rebuilds from stored events.
A judge is itself a model call. This file pins what that call does to the record
it is reading.

Seven facts, all established against the pinned framework and all of them
framework mechanics rather than provider behaviour — which is why `mockllm` is
the right instrument here and not a compromise (`specification.md` §15, "a mock
confirms plumbing, never semantics"):

1. **A scorer's `ModelEvent` persists.** `_eval/score.py:485-488` slices every
   event the scorer produced off the transcript and splices it into
   `EvalSample.events`. It survives `write_eval_log` unchanged.
2. **A scorer sees its own `ModelEvent`** the moment it reads `transcript()`
   again, because `model.generate` appends to the active transcript
   (`model/_model.py:1650-1666`) and the deferred path installs the sample's
   transcript as the active one (`_eval/score.py:434-435`).
3. **Two `score()` calls against one generation log are isolated**, which
   `decisions.md` §0 v11 inferred from deep-copy plus
   reconstruct-from-stored-events and did not observe. Observed here.
4. **Scorers inside one call run sequentially and see each other**, so the
   configuration §0 v11 removed was not merely untidy.
5. **`derive_status` changes its answer** once a scorer has generated: the
   derivation takes the *final* stop reason, and the judge's turn is now final.
6. **The scorer's own span survives the write and names the scorer** — but the
   span's `parent_id` is left dangling by the append splice, so ancestry does not
   reconstruct and only the immediately enclosing span is usable as a filter.
7. **The judge's token usage is recorded nowhere the analysis layer may read.**
   `EvalSample.model_usage`, `EvalSample.role_usage` and `EvalStats.model_usage`
   are all untouched by `_run_score_task`; the only carrier is
   `ModelEvent.output.usage`, which §6.4 invariant 8 forbids the harness to read.

The consequence for the shipped design is a rule rather than a repair:
**completeness is derived over a generation log and never over a scoring log**,
and a scorer that renders must render before it generates. Fact 5 is what makes
both load-bearing.

`mockllm/model` throughout; no network, no keys, milliseconds.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval, score
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.event import (
    Event,
    EventTreeSpan,
    ModelEvent,
    ScoreEvent,
    SpanBeginEvent,
    event_tree,
)
from inspect_ai.log import EvalLog, read_eval_log, transcript, write_eval_log
from inspect_ai.model import ChatMessageUser, ModelUsage, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState
from mock_harness import metered_model, mock_agent, truncating_model

from genomics_harness.render_capture import capture, render_capture
from genomics_harness.renderer import SECTIONS
from genomics_harness.trajectory import Completeness, derive_status

# -- the trajectory under test --------------------------------------------

DATASET = MemoryDataset(
    [
        Sample(
            id="a1b2c3d4e5f60718", input="Investigate the cohort.", metadata={"gt": 42}
        )
    ]
)
"""One sample, one turn. The trajectory's content is irrelevant here; what
matters is that its true outcome is `TRUNCATED` and so is distinguishable from
anything a judge stops on."""


def _generation_task() -> Task:
    return Task(
        name="scorer_events_generation",
        dataset=DATASET,
        solver=mock_agent(),
        scorer=[render_capture()],
        version="v-scorer-events",
    )


@pytest.fixture
def generation_path(tmp_path: Path) -> str:
    """One generation run whose single turn stopped at ``max_tokens``.

    `truncating_model` is the fixture the rest of the suite uses for an
    output-cap stop; the derivation over this log reads `TRUNCATED`, which is
    what makes a flip visible.
    """
    logs = eval(
        _generation_task(),
        model=truncating_model(),  # type: ignore[no-untyped-call]
        log_dir=str(tmp_path / "generation"),
        display="none",
    )
    assert logs[0].status == "success"
    return str(logs[0].location)


def _read(path: str) -> EvalLog:
    return read_eval_log(path, resolve_attachments=True)


def _events(log: EvalLog) -> list[Event]:
    """The one sample's event stream."""
    assert log.samples is not None
    return log.samples[0].events


def _reported(log: EvalLog, scorer_name: str) -> dict[str, Any]:
    """What a probe scorer recorded, from the one sample."""
    assert log.samples is not None
    scores = log.samples[0].scores
    assert scores is not None
    metadata = scores[scorer_name].metadata
    assert metadata is not None
    return metadata


# -- readers ---------------------------------------------------------------
#
# Each reader binds exactly one event type to a name, because static invariant 8
# is scoped per function: a name the source binds to a `ModelEvent` may only be
# read through `output.stop_reason`, so a helper that also walked span begins
# under the same name would report every `begin.id` as a violation.


def _model_event_count(events: Sequence[Event]) -> int:
    return sum(1 for event in events if isinstance(event, ModelEvent))


def _model_event_span_ids(events: Sequence[Event]) -> list[str | None]:
    """The span each `ModelEvent` sits directly inside.

    Waived rather than avoided. The rule exists to keep the *trajectory-reading
    surface* off the attachment-bearing fields (§6.4); a test establishing what
    the framework writes is not that surface, and the waiver is visible in
    review, which is the point of having one.
    """
    return [
        event.span_id  # harness-invariant: allow model-event-stop-reason-only
        for event in events
        if isinstance(event, ModelEvent)
    ]


def _model_event_usage(events: Sequence[Event]) -> list[ModelUsage | None]:
    """`ModelEvent.output.usage`, the one place a judge's spend is recorded.

    Waived for the same reason, and the waiver is the finding: the harness
    cannot read this field, so what it records is unavailable to the analysis
    layer without either a waiver or a rule change.
    """
    return [
        event.output.usage  # harness-invariant: allow model-event-stop-reason-only
        for event in events
        if isinstance(event, ModelEvent)
    ]


def _sections(captured: Score) -> dict[str, str]:
    """The per-section digests off a `render_capture` `Score`."""
    metadata = captured.metadata
    assert metadata is not None
    digests: dict[str, str] = metadata["section_digests"]
    return digests


def _span_begins(events: Sequence[Event]) -> list[SpanBeginEvent]:
    return [begin for begin in events if isinstance(begin, SpanBeginEvent)]


def _span_index(events: Sequence[Event]) -> dict[str, SpanBeginEvent]:
    return {begin.id: begin for begin in _span_begins(events)}


def _top_level_span_names(log_events: Sequence[Event]) -> list[str]:
    return [
        node.name for node in event_tree(log_events) if isinstance(node, EventTreeSpan)
    ]


# -- probe scorers ---------------------------------------------------------
#
# Registered at import, like every `@scorer`. They are instruments, not judges.

CALL_ORDER: list[str] = []
"""Appended to by the ordering probes; cleared by the test that reads it."""


@scorer(name="probe_render_generate_render", metrics=[])
def probe_render_generate_render() -> Scorer:
    """Render, generate, render again, and report both renderings.

    One scorer rather than two so that the before and after readings come from
    the same transcript in the same call, which is the configuration the
    "render before generating" discipline is about.
    """

    async def score_(state: TaskState, target: Target) -> Score:
        before_events = list(transcript().events)
        before_status = derive_status(before_events)
        before = capture(state, before_events)

        output = await get_model().generate(
            input=[ChatMessageUser(content="Score this trajectory.")]
        )

        after_events = list(transcript().events)
        after_status = derive_status(after_events)
        after = capture(state, after_events)

        return Score(
            value=1.0,
            metadata={
                "model_events_before": _model_event_count(before_events),
                "model_events_after": _model_event_count(after_events),
                "completeness_before": before_status.completeness.value,
                "completeness_after": after_status.completeness.value,
                "stop_reasons_before": before_status.stop_reasons,
                "stop_reasons_after": after_status.stop_reasons,
                "digest_before": before.value,
                "digest_after": after.value,
                "sections_before": _sections(before),
                "sections_after": _sections(after),
                "judge_output_tokens": (
                    None if output.usage is None else output.usage.output_tokens
                ),
            },
        )

    return score_


def _ordering_probe(name: str) -> Scorer:
    @scorer(name=name, metrics=[])
    def factory() -> Scorer:
        async def score_(state: TaskState, target: Target) -> Score:
            CALL_ORDER.append(f"{name}:enter")
            before = _model_event_count(transcript().events)
            await get_model().generate(
                input=[ChatMessageUser(content=f"Score this, from {name}.")]
            )
            after = _model_event_count(transcript().events)
            CALL_ORDER.append(f"{name}:exit")
            return Score(
                value=1.0,
                metadata={
                    "model_events_before": before,
                    "model_events_after": after,
                },
            )

        return score_

    return factory()


@scorer(name="probe_role_judge", metrics=[])
def probe_role_judge() -> Scorer:
    """Generates through a **model role**, which is how judges are configured."""

    async def score_(state: TaskState, target: Target) -> Score:
        output = await get_model(role="judge").generate(
            input=[ChatMessageUser(content="Score this trajectory.")]
        )
        return Score(
            value=1.0,
            metadata={
                "judge_usage_total": (
                    None if output.usage is None else output.usage.total_tokens
                )
            },
        )

    return score_


def _score_once(path: str, judge: Scorer, out: Path) -> EvalLog:
    """One `score()` call against the pristine generation log, written out.

    Reads the log fresh each time, which is stricter than the shipped driver
    needs — §6.3 passes one `EvalLog` object to every judge's call, and
    `test_two_score_calls_reusing_one_log_object_cannot_see_each_other` covers
    that shape. The explicit `out` is not optional: `score()` leaves
    `log.location` pointing at the generation file, so a `write_eval_log` with
    no path rewrites it (§6.3).
    """
    scored = score(
        _read(path),
        [judge],
        action="append",
        model="mockllm/model",
        display="none",
    )
    write_eval_log(scored, str(out))
    return scored


# -- Q1: persistence -------------------------------------------------------


def test_a_scorers_model_event_is_spliced_into_the_sample_events(
    generation_path: str, tmp_path: Path
) -> None:
    """`_eval/score.py:485-488`, asserted on the event list itself.

    Stated as the count and the enclosing span rather than as anything
    downstream, so the test still fails if the splice changes shape but leaves
    the derivation unaffected (§15, "assert the fact, not its consequences").
    """
    generation = _read(generation_path)
    assert _model_event_count(_events(generation)) == 1

    scored = _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )

    returned = _events(scored)
    assert _model_event_count(returned) == 2, (
        "the scorer's own generate call is expected in the returned log"
    )

    written = _events(_read(str(tmp_path / "scoring.eval")))
    assert _model_event_count(written) == 2
    assert [type(event).__name__ for event in written] == [
        type(event).__name__ for event in returned
    ], "the written log's event sequence differs from the returned one"

    # and under what key: an `EvalSample.events` entry of type `model`, inside a
    # span named for the scorer
    spans = _span_index(written)
    span_ids = _model_event_span_ids(written)
    assert [spans[span_id].name for span_id in span_ids if span_id] == [
        "mock_agent",
        "probe_render_generate_render",
    ]


def test_the_generation_log_is_left_alone_by_scoring(
    generation_path: str, tmp_path: Path
) -> None:
    """The splice is on `score()`'s deep copy (`_eval/score.py:276-278`)."""
    _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )
    assert _model_event_count(_events(_read(generation_path))) == 1


# -- Q2: self-visibility ---------------------------------------------------


def test_a_scorer_sees_its_own_model_event(
    generation_path: str, tmp_path: Path
) -> None:
    """The discipline "render before generating" is load-bearing, not hygiene."""
    scored = _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )
    seen = _reported(scored, "probe_render_generate_render")

    assert seen["model_events_before"] == 1
    assert seen["model_events_after"] == 2


# -- Q3: isolation across calls -------------------------------------------


def test_two_score_calls_reusing_one_log_object_cannot_see_each_other(
    generation_path: str, tmp_path: Path
) -> None:
    """The shipped judging shape (§6.3): one `EvalLog` in hand, one call per judge.

    §0 v11 infers isolation from `score()` deep-copying (`_eval/score.py:276-278`)
    and the deferred path rebuilding a transcript from the stored events
    (`:434-435`), and states plainly that the composition has not been observed.
    Observed here, in the configuration the driver will actually run: the same
    object passed to every call, each written to its own location.
    """
    generation = _read(generation_path)

    written: list[EvalLog] = []
    for name in ("judge_a", "judge_b", "judge_c"):
        scored = score(
            generation,
            [probe_render_generate_render()],
            action="append",
            model="mockllm/model",
            display="none",
        )
        write_eval_log(scored, str(tmp_path / f"{name}.eval"))
        written.append(scored)

    for scored in written:
        seen = _reported(scored, "probe_render_generate_render")
        assert seen["model_events_before"] == 1
        assert seen["stop_reasons_before"] == ["max_tokens"]

    # the shared object is unchanged, and each log carries its own judge's call
    assert _model_event_count(_events(generation)) == 1
    for name in ("judge_a", "judge_b", "judge_c"):
        assert _model_event_count(_events(_read(str(tmp_path / f"{name}.eval")))) == 2


def test_two_score_calls_reading_the_log_afresh_cannot_see_each_other(
    generation_path: str, tmp_path: Path
) -> None:
    """The same, when each call re-reads the log from disk.

    Not the shipped shape — kept because it is the variant a driver would reach
    for under `eval_set` resumption or batching, and it costs milliseconds.
    """
    first = _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "judge_a.eval",
    )
    second = _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "judge_b.eval",
    )

    for scored in (first, second):
        seen = _reported(scored, "probe_render_generate_render")
        assert seen["model_events_before"] == 1
        assert seen["stop_reasons_before"] == ["max_tokens"]

    for name in ("judge_a.eval", "judge_b.eval"):
        assert _model_event_count(_events(_read(str(tmp_path / name)))) == 2


# -- Q4: ordering and concurrency within one call --------------------------


def test_scorers_in_one_call_run_in_order_and_see_each_other(
    generation_path: str,
) -> None:
    """Sequential, in list order (`_eval/score.py:449-456`).

    This is the configuration §0 v11 removed by decision. The test exists so the
    reason for the decision stays a fact rather than a recollection: under one
    call, the second scorer walks a stream one model call longer than the first,
    so a rendering that walks `ModelEvent`s is unstable *between judges within a
    single run* — which is a different failure from the one Q5 describes.
    """
    CALL_ORDER.clear()
    both = score(
        _read(generation_path),
        [_ordering_probe("probe_a"), _ordering_probe("probe_b")],
        action="append",
        model="mockllm/model",
        display="none",
    )
    assert CALL_ORDER == [
        "probe_a:enter",
        "probe_a:exit",
        "probe_b:enter",
        "probe_b:exit",
    ], "scorers within one call are expected to run sequentially, in list order"

    assert _reported(both, "probe_a")["model_events_before"] == 1
    assert _reported(both, "probe_b")["model_events_before"] == 2, (
        "the second scorer is expected to see the first scorer's model event"
    )


# -- Q5: consequence for the derivation ------------------------------------


def test_the_derivation_flips_once_a_scorer_has_generated(
    generation_path: str, tmp_path: Path
) -> None:
    """`TRUNCATED` becomes `CLEAN`, because the judge's turn is now the final one.

    This is why completeness is derived over a **generation** log. The rule is
    ours; the behaviour it guards against is the framework's.
    """
    scored = _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )
    seen = _reported(scored, "probe_render_generate_render")

    assert seen["completeness_before"] == Completeness.TRUNCATED.value
    assert seen["stop_reasons_before"] == ["max_tokens"]
    assert seen["completeness_after"] == Completeness.CLEAN.value
    assert seen["stop_reasons_after"] == ["max_tokens", "stop"]

    # and the same flip is on disk, for anyone re-deriving over a scoring log
    written = _events(_read(str(tmp_path / "scoring.eval")))
    assert derive_status(written).completeness is Completeness.CLEAN
    assert derive_status(written).stop_reasons == ["max_tokens", "stop"]

    # while the generation log still derives correctly
    generation = derive_status(_events(_read(generation_path)))
    assert generation.completeness is Completeness.TRUNCATED


# -- Q6: is the span boundary usable? --------------------------------------


def test_the_scorer_span_survives_the_write_and_names_the_scorer(
    generation_path: str, tmp_path: Path
) -> None:
    """Solver turns and scorer turns are separable by the *immediately* enclosing span."""
    _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )
    events = _events(_read(str(tmp_path / "scoring.eval")))

    spans = _span_index(events)
    kinds = [
        (spans[span_id].name, spans[span_id].type)
        for span_id in _model_event_span_ids(events)
        if span_id
    ]
    assert kinds == [
        ("mock_agent", "solver"),
        ("probe_render_generate_render", "scorer"),
    ]


def test_the_appended_scorer_spans_parent_is_dangling(
    generation_path: str, tmp_path: Path
) -> None:
    """Ancestry does not reconstruct, so an ancestor-based filter is unsound.

    `_get_updated_events` (`_eval/score.py:184-188`) re-parents the new scorer's
    `EventTreeSpan`, but `EventTreeSpan.parent_id` is a dataclass copy
    (`event/_tree.py:18-40`) and `event_sequence` re-emits the original
    `SpanBeginEvent` (`:107`), whose `parent_id` still names the scoring run's
    own ``scorers`` span — a span the splice discards. Rebuilding the tree from
    the stored events therefore lifts the scorer span to the root.
    """
    _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )
    events = _events(_read(str(tmp_path / "scoring.eval")))

    begins = {begin.name: begin for begin in _span_begins(events)}
    known = {begin.id for begin in _span_begins(events)}
    appended = begins["probe_render_generate_render"]
    assert appended.parent_id is not None
    assert appended.parent_id not in known, (
        "if this starts resolving, ancestry-based filtering became available "
        "and `solver-cleanup-august-2026.md` §6 option 3 can be reconsidered"
    )

    # which is observable in the rebuilt tree: a root sibling of `scorers`,
    # not a child of it
    assert _top_level_span_names(events) == [
        "init",
        "solvers",
        "scorers",
        "probe_render_generate_render",
    ]


# -- Q7: renderer-capture equivalence under a generating scorer ------------


def test_the_capture_digest_survives_a_generating_scorer_that_renders_first(
    generation_path: str, tmp_path: Path
) -> None:
    """The §6.3 equivalence, re-established under a scorer that calls a model.

    The delivered result re-scored with `render_capture` alone, which generates
    nothing, so it said nothing about the shipping configuration.
    """
    generation = _read(generation_path)
    assert generation.samples is not None
    scores = generation.samples[0].scores
    assert scores is not None
    inline = scores["render_capture"]

    scored = _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )
    seen = _reported(scored, "probe_render_generate_render")

    assert seen["digest_before"] == inline.value
    assert seen["sections_before"] == _sections(inline)
    assert seen["judge_output_tokens"], "the probe must actually have generated"


def test_no_section_moves_when_the_scorer_generates_before_rendering(
    generation_path: str, tmp_path: Path
) -> None:
    """What the v12 section removal bought, asserted rather than argued.

    Until v12 this test read ``moved == {"completeness"}``: that section walked
    `ModelEvent`s, a judge's own call is spliced into the stream it walks
    (`_eval/score.py:485-488`), and so a rendering was stable only while the
    scorer rendered *before* it generated. The section is withdrawn from what
    judges read (§6.4), and rule ``renderer-no-model-event`` keeps the renderer
    off the event stream entirely, so the ordering discipline no longer has a
    rendering to protect.

    It is asserted here and not deleted, because the *reason* the discipline
    existed is still live for anything else a scorer might read out of the
    transcript — `test_a_scorer_sees_its_own_model_event` is that fact — and a
    renderer that reached for a `ModelEvent` again would fail here as well as at
    the static rule.
    """
    scored = _score_once(
        generation_path,
        probe_render_generate_render(),
        tmp_path / "scoring.eval",
    )
    seen = _reported(scored, "probe_render_generate_render")

    before, after = seen["sections_before"], seen["sections_after"]
    moved = {name for name in before if before[name] != after[name]}
    assert moved == set(), "no rendering section may depend on the event stream"
    assert seen["digest_before"] == seen["digest_after"]
    assert set(before) == set(SECTIONS)

    # and the probe genuinely generated, or the equality is vacuous
    assert seen["model_events_before"] == 1
    assert seen["model_events_after"] == 2


# -- what the scoring log does not record ----------------------------------


def test_a_generation_built_from_a_callable_records_no_usage_at_all(
    generation_path: str,
) -> None:
    """Why the before-and-after pair below needs a different generation model.

    `mockllm` fills `ModelOutput.usage` in only on its iterator branch: the
    `custom_outputs=<callable>` branch returns at `_providers/mockllm.py:88-95`,
    before the ``if output.usage is None`` block at `:105-133`. Every model
    fixture in `mock_harness` except `metered_model` is built from a callable,
    so a log generated with one carries no usage in any field.

    Asserted rather than described, because an observation that a usage field is
    empty *after* an operation says nothing when the field was empty before it —
    which is what happened to two rows of `scorer-events-august-2026.md` §10.1.
    """
    generation = _read(generation_path)
    assert generation.samples is not None

    assert generation.samples[0].model_usage == {}
    assert generation.stats.model_usage == {}
    assert _model_event_usage(_events(generation)) == [None]


@pytest.fixture
def metered_generation_path(tmp_path: Path) -> str:
    """A generation run whose model reports usage, so the fields have a subject."""
    logs = eval(
        Task(
            name="scorer_events_metered",
            dataset=DATASET,
            solver=mock_agent(),
            scorer=[render_capture()],
            version="v-scorer-events-metered",
        ),
        model=metered_model(),  # type: ignore[no-untyped-call]
        log_dir=str(tmp_path / "metered"),
        display="none",
    )
    assert logs[0].status == "success"
    return str(logs[0].location)


def test_the_generation_log_records_usage_before_any_scoring_call(
    metered_generation_path: str,
) -> None:
    """The "before" half, stated on its own so the "after" half means something."""
    generation = _read(metered_generation_path)
    assert generation.samples is not None
    sample = generation.samples[0]

    assert [usage.total_tokens for usage in sample.model_usage.values()] == [33]
    assert [usage.total_tokens for usage in generation.stats.model_usage.values()] == [
        33
    ]
    usages = _model_event_usage(_events(generation))
    assert [usage.total_tokens for usage in usages if usage] == [33]

    # `role_usage` is empty because the generation run declares no roles, not
    # because scoring emptied it
    assert sample.role_usage == {}
    assert generation.stats.role_usage == {}


def test_a_judges_usage_reaches_no_field_the_harness_may_read(
    metered_generation_path: str, tmp_path: Path
) -> None:
    """`_run_score_task` writes `scores` and `events`, and no usage anywhere.

    `CLAUDE.md` §6 read `model_roles=` as attributing each judge's usage into
    `EvalSample.role_usage`. It does not, on this path: the judge spends tokens,
    its `ModelOutput` reports them, and no aggregate field moves. The only
    carrier is `ModelEvent.output.usage`, which invariant 8 forbids the harness
    to read — so the spend is on disk and out of reach.

    **The fields are stale, not empty**, which is the correction this test
    carries over `scorer-events-august-2026.md` §10.1's table. That table read
    ``{}`` on a generation whose model reported nothing. With a generation that
    does, `score()` leaves those fields holding the *generation's* numbers, so a
    consumer reading `EvalSample.model_usage` off a scoring log gets a real
    number that is not the judge's — the same failure §10.1 named for
    `ScoreEvent.model_usage` alone.
    """
    scored = score(
        _read(metered_generation_path),
        [probe_role_judge()],
        action="append",
        model="mockllm/model",
        model_roles={"judge": "mockllm/model"},
        display="none",
    )
    write_eval_log(scored, str(tmp_path / "scoring.eval"))
    written = _read(str(tmp_path / "scoring.eval"))

    spent = _reported(scored, "probe_role_judge")["judge_usage_total"]
    assert spent, "the judge must actually have spent tokens for this to mean anything"
    assert spent != 33, "the judge's spend must differ from the generation's"

    assert written.samples is not None
    sample = written.samples[0]

    # unchanged by scoring: still exactly the generation's usage
    assert [usage.total_tokens for usage in sample.model_usage.values()] == [33]
    assert [usage.total_tokens for usage in written.stats.model_usage.values()] == [33]

    # and the judge ran through a model role, which attributes nowhere
    assert sample.role_usage == {}
    assert written.stats.role_usage == {}

    # the one place the judge's own number survives
    usages = _model_event_usage(_events(written))
    assert usages[-1] is not None
    assert usages[-1].total_tokens == spent


def test_score_event_model_usage_carries_the_generations_usage(
    metered_generation_path: str, tmp_path: Path
) -> None:
    """A field holding a real number that is not the judge's.

    `_eval/score.py:472-473` sets `ScoreEvent.model_usage` from
    `resolved_sample.model_usage`, so the judge's own score event reports what
    the *generation* spent. `scorer-events-august-2026.md` §11 left this as
    narrative on the ground that nothing in the harness reads `ScoreEvent`; the
    reason to pin it anyway is that a field which is wrong is a worse trap than
    one which is absent, and §10.1's third option — reporting judge spend from
    the driver — would be built by someone reading exactly this field.

    `role_usage` is `None` rather than `{}` on the event: the constructor passes
    ``resolved_sample.role_usage or None`` and the generation's is empty.
    """
    scored = score(
        _read(metered_generation_path),
        [probe_role_judge()],
        action="append",
        model="mockllm/model",
        model_roles={"judge": "mockllm/model"},
        display="none",
    )
    write_eval_log(scored, str(tmp_path / "scoring.eval"))

    events = _events(_read(str(tmp_path / "scoring.eval")))
    judged = [
        event
        for event in events
        if isinstance(event, ScoreEvent) and event.scorer == "probe_role_judge"
    ]
    assert len(judged) == 1
    usage = judged[0].model_usage or {}
    assert [entry.total_tokens for entry in usage.values()] == [33]
    assert judged[0].role_usage is None
