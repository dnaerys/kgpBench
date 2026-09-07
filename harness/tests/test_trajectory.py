"""Trajectory completeness — the derivation, and the carrier that cross-checks it.

The empirical answer behind the round-trip tests is in `design/foundations.md`
§1: both `state.metadata` and `state.store` survive the log write, and the
classification is also derivable from the event stream, which is why the
derivation is primary.

**What this file used to assert, and why it is worth reading the diff.** Until
v4 six tests here pinned an inverted stop-reason mapping — `model_length` treated
as output-cap truncation and `max_tokens` never mentioned. They passed, so the
suite was green while `derive_status` classified every real Claude truncation as
`CLEAN`. `design/framework-questions-august-2026.md` §18 has the correction;
`design/solver-august-2026.md` lists the six.

The lesson the file now encodes: nothing asserted the *mapping*, only the branch
structure that consumed it. :func:`test_the_stop_reason_mapping_is_not_inverted`
and its neighbours assert the mapping directly, in both directions, so an
inversion fails a test rather than a round of runs.

**And a second policy has since been reversed.** Until v7 this file asserted that
a `max_tokens` turn followed by another turn was `CONTINUED` and was
**scoreable** — the trajectory counted toward the rate. Continuation is dropped on
evidence (`design_document.md` §0 v7): the replay conveys a provider summary
rather than the reasoning, so a cut inside thinking confabulates, and thinking is
80–100% of output tokens. Those tests are rewritten rather than adjusted, and each
replacement names what it replaced. A suite that goes green after a policy
reversal is evidence of nothing (§15).
"""

from __future__ import annotations

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai import score
from inspect_ai.event import ErrorEvent, ModelEvent, SampleLimitEvent
from inspect_ai.log import EvalError
from inspect_ai.model import GenerateConfig, ModelOutput
from mock_harness import (
    clean_model,
    exhausted_model,
    looping_model,
    mock_task,
    refusing_model,
    truncating_model,
)
from pydantic import ValidationError

from genomics_harness import (
    CONTENT_FILTER_STOP_REASON,
    CONTEXT_STOP_REASONS,
    INVERSION_CONTROL,
    OUTPUT_CAP_STOP_REASON,
    Completeness,
    SolverRecord,
    StopCause,
    TrajectoryStatus,
    carried_record,
    counts_toward_rate,
    derive_status,
    read_log,
    reconcile,
    trajectory_status,
)

# -- the derivation, over synthetic event streams -------------------------


def _model_event(stop_reason: str) -> ModelEvent:
    return ModelEvent(
        model="mockllm/model",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content(
            model="mockllm/model", content="x", stop_reason=stop_reason
        ),
    )


# -- the positive control against re-inversion ----------------------------


def test_the_stop_reason_mapping_is_not_inverted():
    """The control the first delivery lacked.

    `max_tokens` is the output cap and `model_length` is the context window
    (`CHANGELOG.md:2938`). Swap them and this fails on both rows: an output-cap
    stop would classify as context exhaustion, and a context-window stop as a
    recoverable truncation.
    """
    assert INVERSION_CONTROL["max_tokens"] is Completeness.TRUNCATED
    assert INVERSION_CONTROL["model_length"] is Completeness.CONTEXT_EXHAUSTED
    assert OUTPUT_CAP_STOP_REASON == "max_tokens"
    assert "model_length" in CONTEXT_STOP_REASONS
    assert OUTPUT_CAP_STOP_REASON not in CONTEXT_STOP_REASONS
    assert CONTENT_FILTER_STOP_REASON not in CONTEXT_STOP_REASONS


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    sorted((k, v) for k, v in INVERSION_CONTROL.items()),
)
def test_each_stop_reason_classifies_as_the_table_says(stop_reason, expected):
    """Every `StopReason` value (`model/_model_output.py:95-102`), one at a time.

    Table-driven against the exported mapping rather than against the branch
    structure of `derive_status`, because it was the branch structure that
    survived the inversion.
    """
    assert derive_status([_model_event(stop_reason)]).completeness is expected


def test_neither_truncation_nor_exhaustion_is_scoreable():
    """The consequence the inversion had: a cut-off trajectory being scored."""
    for reason in ("max_tokens", "model_length", "unknown"):
        assert not derive_status([_model_event(reason)]).scoreable


def test_every_stop_reason_in_the_framework_union_is_in_the_table():
    """A `StopReason` added upstream must be classified, not defaulted."""
    from typing import get_args

    from inspect_ai.model._model_output import StopReason

    assert set(get_args(StopReason)) == set(INVERSION_CONTROL)


# -- the five states ------------------------------------------------------


def test_a_single_clean_turn_is_clean():
    status = derive_status([_model_event("stop")])
    assert status.completeness is Completeness.CLEAN
    assert status.scoreable


def test_tool_call_stops_are_ordinary_stops():
    status = derive_status([_model_event("tool_calls"), _model_event("stop")])
    assert status.completeness is Completeness.CLEAN


def test_a_trailing_output_cap_stop_is_truncation():
    """Was `test_a_trailing_model_length_is_truncation` under the inversion."""
    status = derive_status([_model_event("stop"), _model_event("max_tokens")])
    assert status.completeness is Completeness.TRUNCATED
    assert not status.scoreable


def test_an_output_cap_turn_a_further_turn_follows_is_not_recovered():
    """Replaces `test_an_output_cap_turn_followed_by_another_turn_is_a_continuation`.

    That test asserted the v4 policy: a `max_tokens` turn with another turn after
    it was `CONTINUED`, and `CONTINUED` was **scoreable**. Continuation is dropped
    (§0 v7), so this shape is no longer something the solver can produce — it
    breaks on the first `max_tokens`.

    What the derivation must not do is *reward* it. A trajectory that was
    truncated and carried on regardless is the shape a confabulated continuation
    has, and if it ends cleanly the classification is `CLEAN` on the final turn
    only because the derivation reads the final turn. The `max_tokens` entry stays
    in `stop_reasons`, which is the evidence a reader needs to see it happened.
    """
    status = derive_status([_model_event("max_tokens"), _model_event("stop")])
    assert status.stop_reasons == ["max_tokens", "stop"]
    assert not hasattr(status, "continuations")


def test_two_output_cap_turns_are_truncation():
    """Was `test_a_continuation_that_never_completes_is_truncation`, which counted
    one continuation. The final turn decides and nothing is counted."""
    status = derive_status([_model_event("max_tokens"), _model_event("max_tokens")])
    assert status.completeness is Completeness.TRUNCATED
    assert status.model_turns == 2


@pytest.mark.parametrize("reason", sorted(CONTEXT_STOP_REASONS))
def test_context_exhaustion_accepts_both_values(reason):
    """`unknown` on the first-party Anthropic provider, `model_length` on Bedrock.

    Narrowing to `unknown` because that is what ships today would silently
    reclassify every exhausted trajectory the day the fork is bumped (§6.3).
    """
    status = derive_status([_model_event("tool_calls"), _model_event(reason)])
    assert status.completeness is Completeness.CONTEXT_EXHAUSTED
    assert not status.scoreable


def test_context_exhaustion_after_an_output_cap_stop_is_still_exhaustion():
    """Precedence is the final turn's stop reason; the earlier one stays visible
    in `stop_reasons` rather than being folded into the outcome."""
    status = derive_status([_model_event("max_tokens"), _model_event("unknown")])
    assert status.completeness is Completeness.CONTEXT_EXHAUSTED
    assert status.stop_reasons == ["max_tokens", "unknown"]


def test_a_content_filter_stop_is_its_own_outcome():
    """§0 v5 made `CONTENT_FILTERED` the sixth outcome and the delivered module
    never implemented it — `content_filter` mapped to `UNCLASSIFIED`, which is
    "we do not know" being used for something we do know.

    A refusal is a result: this is medical genetics, D2 and D3 present findings
    that cannot be true, and C1 and A5 reward calibrated refusal. It is still not
    scoreable — the checks it did not reach are unreachable, not failed.
    """
    status = derive_status([_model_event("content_filter")])
    assert status.completeness is Completeness.CONTENT_FILTERED
    assert not status.scoreable
    assert status.completeness is not Completeness.UNCLASSIFIED


def test_unclassified_is_reserved_for_what_is_not_known():
    """The residual keeps its meaning only if nothing knowable is bucketed into
    it. Every member of `StopReason` now maps to a named outcome."""
    assert Completeness.UNCLASSIFIED not in set(INVERSION_CONTROL.values())


def test_a_limit_event_outranks_everything_but_keeps_the_components():
    status = derive_status(
        [
            _model_event("max_tokens"),
            _model_event("stop"),
            SampleLimitEvent(type="message", message="Message limit exceeded", limit=4),
        ]
    )
    assert status.completeness is Completeness.LIMIT_TRIPPED
    assert status.limit_type == "message"
    assert status.limit_value == 4
    assert status.stop_reasons == ["max_tokens", "stop"]  # components not collapsed
    assert not status.scoreable


def test_an_error_event_outranks_a_limit():
    status = derive_status(
        [
            _model_event("stop"),
            SampleLimitEvent(type="time", message="Time limit exceeded", limit=1),
            ErrorEvent(
                error=EvalError(message="boom", traceback="", traceback_ansi="")
            ),
        ]
    )
    assert status.completeness is Completeness.ERRORED


def test_no_model_turn_is_unclassified_not_clean():
    status = derive_status([])
    assert status.completeness is Completeness.UNCLASSIFIED
    assert not status.scoreable


# -- the provisional carrier ----------------------------------------------


def test_a_provisional_record_does_not_decode_as_a_status():
    """§6.3: "structurally incapable of decoding as a terminal status".

    A consumer that skips reconciliation and reads the store directly gets an
    exception, not a plausible-looking answer.
    """
    payload = SolverRecord(complete=False).model_dump(mode="json")
    with pytest.raises(ValidationError):
        TrajectoryStatus.model_validate(payload)


def test_a_final_record_does_not_decode_as_a_status_either():
    """The same guard, on the completed record. `TrajectoryStatus` forbids extra
    fields, and the carrier has `complete`, `stop_cause` and `solver_version`."""
    payload = SolverRecord(
        complete=True,
        stop_cause=StopCause.MODEL_STOPPED,
        completeness=Completeness.CLEAN,
    ).model_dump(mode="json")
    with pytest.raises(ValidationError):
        TrajectoryStatus.model_validate(payload)


def test_reading_a_provisional_record_as_a_status_raises():
    with pytest.raises(ValueError, match="provisional"):
        SolverRecord(complete=False).as_status()


def test_a_complete_record_needs_a_completeness():
    from genomics_harness import record_final

    class _Store:
        def __init__(self):
            self.data = {}

        def set(self, key, value):
            self.data[key] = value

    with pytest.raises(ValueError, match="complete record"):
        record_final(_Store(), SolverRecord(complete=True))


# -- reconciliation -------------------------------------------------------


def _carried(completeness: Completeness, **kwargs) -> SolverRecord:
    return SolverRecord(
        complete=True,
        stop_cause=kwargs.pop("stop_cause", StopCause.MODEL_STOPPED),
        completeness=completeness,
        **kwargs,
    )


def test_reconcile_reports_agreement_as_no_disagreements():
    derived = derive_status([_model_event("stop")])
    authoritative, disagreements = reconcile(derived, _carried(Completeness.CLEAN))
    assert authoritative is derived
    assert disagreements == []


def test_reconcile_compares_completeness_and_says_so():
    """`continuations` was the second compared field and went with the
    continuation policy. `model_turns` is on both records and is deliberately
    **not** compared: the derivation counts every `ModelEvent` in the stream it is
    handed, and a judge's own generate call lands in that stream at scoring time.
    """
    assert set(SolverRecord.model_fields) >= {"completeness", "model_turns"}
    derived = derive_status([_model_event("stop"), _model_event("stop")])
    carried = _carried(Completeness.CLEAN, model_turns=1)
    assert reconcile(derived, carried)[1] == []


def test_reconcile_surfaces_a_solver_that_lied():
    derived = derive_status([_model_event("max_tokens")])
    authoritative, disagreements = reconcile(derived, _carried(Completeness.CLEAN))
    assert authoritative.completeness is Completeness.TRUNCATED
    assert any("completeness" in d for d in disagreements)


def test_reconcile_flags_a_provisional_carrier_rather_than_ignoring_it():
    derived = derive_status([_model_event("stop")])
    _, disagreements = reconcile(derived, SolverRecord(complete=False))
    assert disagreements and "provisional" in disagreements[0]


def test_reconcile_tolerates_no_carrier_at_all():
    derived = derive_status([_model_event("stop")])
    assert reconcile(derived, None) == (derived, [])


def test_reconcile_never_resolves_a_disagreement():
    """The derivation wins and the carrier is reported, never merged."""
    derived = derive_status([_model_event("unknown")])
    authoritative, disagreements = reconcile(derived, _carried(Completeness.CLEAN))
    assert authoritative.completeness is derived.completeness
    assert disagreements


# -- §5's exclusion decision ----------------------------------------------


def test_a_clean_trajectory_with_an_agreeing_carrier_counts():
    derived = derive_status([_model_event("stop")])
    assert counts_toward_rate(derived, _carried(Completeness.CLEAN)) == (True, None)


def test_a_turn_cap_stop_is_excluded_although_the_trace_looks_clean():
    """Why `counts_toward_rate` still takes both records after the removal.

    Replaces `test_a_headroom_stop_is_excluded_although_the_trace_looks_clean`,
    which asserted the same property through the pre-emptive stop. That stop is
    gone (§0 v7); the turn cap has the identical shape and is the reason the
    two-axis design survives on its own merits. The solver stops after a turn that
    ended at `tool_calls`, so the derivation says `CLEAN` and only the carrier
    knows the model was not finished.
    """
    derived = derive_status([_model_event("tool_calls")])
    assert derived.completeness is Completeness.CLEAN
    assert derived.scoreable
    counts, reason = counts_toward_rate(
        derived, _carried(Completeness.CLEAN, stop_cause=StopCause.TURN_CAP)
    )
    assert not counts
    assert reason == "stop_cause:turn_cap"


def test_a_withdrawn_stop_cause_cannot_be_named():
    """The two the removal took out. Naming either is now an AttributeError
    rather than an exclusion nobody can produce."""
    for withdrawn in ("HEADROOM", "CONTINUATION_BUDGET"):
        assert not hasattr(StopCause, withdrawn)


def test_a_trace_visible_cause_is_reported_against_the_trace():
    """`OUTPUT_CAP` and `CONTEXT_SIGNAL` are absent from the solver-incomplete set
    on purpose: the derivation already classifies both, so the exclusion reason is
    the trace fact and not the solver's assertion about it. §5 counts per cause,
    and the count must not depend on which record was consulted first."""
    derived = derive_status([_model_event("max_tokens")])
    counts, reason = counts_toward_rate(
        derived,
        _carried(Completeness.TRUNCATED, stop_cause=StopCause.OUTPUT_CAP),
    )
    assert not counts
    assert reason == "completeness:truncated"


def test_a_provisional_carrier_excludes():
    derived = derive_status([_model_event("stop")])
    counts, reason = counts_toward_rate(derived, SolverRecord(complete=False))
    assert not counts and reason == "carrier:provisional"


def test_a_disagreement_excludes():
    derived = derive_status([_model_event("stop")])
    counts, reason = counts_toward_rate(derived, _carried(Completeness.TRUNCATED))
    assert not counts and reason == "carrier:disagreement"


def test_an_incomplete_trace_excludes_whatever_the_carrier_says():
    derived = derive_status([_model_event("model_length")])
    counts, reason = counts_toward_rate(
        derived, _carried(Completeness.CONTEXT_EXHAUSTED)
    )
    assert not counts and reason == "completeness:context_exhausted"


# -- the round trip, end to end -------------------------------------------


def _run(instances, tmp_path, model, name):
    return inspect_eval(
        mock_task(instances, name=name),
        model=model,
        display="none",
        log_dir=str(tmp_path / name),
    )[0]


def test_the_carrier_survives_the_log_write(instances, tmp_path):
    """`state.store` written by the solver reaches `EvalSample.store`.

    The mechanism is a snapshot, not an event replay: `create_eval_sample` writes
    ``store=dict(state.store.items())`` (`_eval/task/run.py:2567`).
    """
    log = _run(instances, tmp_path, truncating_model(), "carrier")
    stored = read_log(log.location)
    for sample in stored.samples or []:
        carried = carried_record(sample.store)
        assert carried is not None
        assert carried.complete
        assert carried.completeness is Completeness.TRUNCATED
        assert carried.model_turns == 1  # stopped on the cap, not continued
        assert carried.stop_cause is StopCause.OUTPUT_CAP


def test_the_derivation_agrees_with_the_carrier(instances, tmp_path):
    log = _run(instances, tmp_path, truncating_model(), "agree")
    stored = read_log(log.location)
    for sample in stored.samples or []:
        derived = trajectory_status(sample)
        _, disagreements = reconcile(derived, carried_record(sample.store))
        assert disagreements == [], disagreements
        assert derived.completeness is Completeness.TRUNCATED


def test_the_harness_writes_nothing_into_the_conversation(instances, tmp_path):
    """Replaces `test_the_continuation_turn_is_identifiable_as_harness_written`,
    which asserted that the injected turn carried ``source=None`` so a judge could
    tell it apart. There is no injected turn (§0 v7, §11), so the stronger
    property holds: **exactly one user message, and it is the instance prompt.**

    `sample_messages` sets ``source="input"`` on the seeded prompt and only on it
    (`_eval/task/util.py:25-32`), so a second user message with ``source=None`` is
    exactly what a regression would look like. Truncation is the case that used to
    produce one.
    """
    log = _run(instances, tmp_path, truncating_model(), "source")
    for sample in log.samples or []:
        users = [m for m in sample.messages if m.role == "user"]
        assert len(users) == 1
        assert users[0].source == "input"
        assert not [m for m in sample.messages if m.role == "system"]


def test_a_clean_run_derives_as_clean(instances, tmp_path):
    log = _run(instances, tmp_path, clean_model(), "clean")
    stored = read_log(log.location)
    for sample in stored.samples or []:
        assert trajectory_status(sample).completeness is Completeness.CLEAN


def test_a_refusal_round_trips_as_content_filtered(instances, tmp_path):
    """Both records agree, so a refusal is excluded under its own cause rather
    than as a carrier disagreement — which is what §5's per-cause counts need."""
    log = _run(instances, tmp_path, refusing_model(), "refused")
    stored = read_log(log.location)
    for sample in stored.samples or []:
        derived = trajectory_status(sample)
        assert derived.completeness is Completeness.CONTENT_FILTERED
        carried = carried_record(sample.store)
        assert carried is not None
        assert carried.completeness is Completeness.CONTENT_FILTERED
        assert reconcile(derived, carried)[1] == []
        counts, reason = counts_toward_rate(derived, carried)
        assert not counts
        assert reason == "completeness:content_filtered"


@pytest.mark.parametrize("reason", sorted(CONTEXT_STOP_REASONS))
def test_context_exhaustion_stops_the_loop(instances, tmp_path, reason):
    """The behaviour the inversion had backwards, end to end.

    Under the inverted mapping the solver continued on `model_length`, appending
    a user turn to a conversation the provider had just refused. It stops, and the
    trajectory is classified and excluded. Nothing in this test changed at v7 —
    it is the case the withdrawn policy was already right about, and it is here to
    show that removing continuation did not disturb it.
    """
    log = _run(instances, tmp_path, exhausted_model()(reason), f"exhausted-{reason}")
    stored = read_log(log.location)
    for sample in stored.samples or []:
        derived = trajectory_status(sample)
        assert derived.completeness is Completeness.CONTEXT_EXHAUSTED
        assert derived.model_turns == 1  # stopped, not continued
        carried = carried_record(sample.store)
        assert carried is not None
        assert carried.stop_cause is StopCause.CONTEXT_SIGNAL
        assert reconcile(derived, carried)[1] == []


def test_the_scorer_sees_the_same_status_inline_and_deferred(instances, tmp_path):
    """The property the second-pass decision depends on."""
    from inspect_ai._eval.score import resolve_scorers

    log = _run(instances, tmp_path, truncating_model(), "both")
    inline = {
        (sample.id, name): s.metadata["trajectory_status"]["completeness"]
        for sample in log.samples or []
        for name, s in (sample.scores or {}).items()
    }
    assert set(inline.values()) == {"truncated"}

    stored = read_log(log.location)
    rescored = score(stored, resolve_scorers(stored), action="append", display="none")
    deferred = {
        (sample.id, name): s.metadata["trajectory_status"]["completeness"]
        for sample in rescored.samples or []
        for name, s in (sample.scores or {}).items()
        if name.endswith("1")
    }
    assert set(deferred.values()) == {"truncated"}
    assert len(deferred) == len(inline)


def test_a_limit_trip_is_visible_from_the_stored_log(instances, tmp_path):
    """A message limit unwinds the solver before its final bookkeeping runs, so
    the derivation is the only thing that gets the classification right.

    Driven by :func:`looping_model` rather than :func:`truncating_model`. Under
    the continuation policy the truncating fixture produced a second turn and the
    limit landed there, so this test reached its own subject only because
    continuation existed.
    """
    log = inspect_eval(
        mock_task(instances, name="limited"),
        model=looping_model(),
        display="none",
        message_limit=2,
        log_dir=str(tmp_path / "limited"),
    )[0]
    stored = read_log(log.location)
    assert stored.status == "success"  # a limit is not an error

    for sample in stored.samples or []:
        derived = trajectory_status(sample)
        assert derived.completeness is Completeness.LIMIT_TRIPPED
        assert derived.limit_type == "message"
        assert not derived.scoreable

        # the solver's provisional write survived the unwind, and says so
        carried = carried_record(sample.store)
        assert carried is not None
        assert not carried.complete
        assert carried.completeness is None
        _, disagreements = reconcile(derived, carried)
        assert disagreements, "a provisional carrier must not pass silently"
        assert counts_toward_rate(derived, carried)[0] is False
