"""Trajectory completeness — how a run ended, and how a judge finds out.

"Check not performed" and "trajectory never reached the point where the check
was possible" are different results. Collapsing them makes the per-check rate
wrong in a direction that penalises the model for our instance sizing.

Under second-pass judging the solver and the judge are separated by a log write,
so how completeness travels matters. Both available carriers survive that write
(`design/foundations.md` §1), and so does a derivation from the event stream.
This module implements the derivation as primary and the carrier as a
cross-check, so a solver bug shows up as a disagreement rather than as a
confident wrong flag.

The derivation reads only `ModelEvent`, `SampleLimitEvent` and `ErrorEvent`, all
of which are reconstructed by `_eval/score.py:434` before a deferred scorer runs
and are equally present in `EvalSample.events` for the analysis layer.

**The stop-reason mapping, which the first delivery had inverted.** In this
framework ``max_tokens`` is the **output cap** and ``model_length`` is the
**context window**. `CHANGELOG.md:2938` records the split verbatim: "StopReason:
Added ``model_length`` for exceeding token window and renamed ``length`` to
``max_tokens``." The consequence for this module is total — the two are
different failures with different causes, and a derivation that tests the wrong
one classifies every real Claude truncation as `CLEAN` and scores a trajectory
that stopped mid-sentence (`design/framework-questions-august-2026.md` §18,
`design/specification.md` §0 v4 and §6.3).

**Nothing is recovered.** Continuation after an output-cap truncation was built,
run against the live API and withdrawn on evidence (`specification.md` §0 v7):
the replayed thinking is a provider *summary* rather than the reasoning, so a cut
inside thinking confabulates a resumption that **looks complete and gets scored**.
A `TRUNCATED` trajectory is marked, excluded from the primary rate and reported
per cause. For a benchmark that scores reasoning steps, a rare loss is strictly
preferable to a rare fabrication.

`Completeness` therefore carries §6.3's five outcomes plus two residuals the five
do not describe; :data:`INVERSION_CONTROL` pins the mapping so a re-inversion
fails a test rather than a round of runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

from inspect_ai.event import ErrorEvent, Event, ModelEvent, SampleLimitEvent
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CARRIER_KEY",
    "CONTENT_FILTER_STOP_REASON",
    "CONTEXT_STOP_REASONS",
    "INVERSION_CONTROL",
    "OUTPUT_CAP_STOP_REASON",
    "Completeness",
    "SolverRecord",
    "StopCause",
    "TrajectoryStatus",
    "carried_record",
    "counts_toward_rate",
    "derive_status",
    "reconcile",
    "record_final",
    "record_provisional",
]

CARRIER_KEY = "trajectory_status"
"""Key under which a solver parks its own account in `state.store`.

`state.store` rather than `state.metadata` because ground truth lives in
metadata and the two should not share a namespace — see
`design/foundations.md` §1.5. Both carriers survive the log write; this is a
hygiene choice, not a correctness one, and neither is protected from the
templating hazard (`specification.md` §6.2 invariant 7).
"""

OUTPUT_CAP_STOP_REASON = "max_tokens"
"""The stop reason that means the generation reached ``max_tokens``.

Thinking included — 80–100% of output tokens across every observation
(`specification.md` §0 v7). Anthropic's stop-reason reference gives the action
as "Raise ``max_tokens`` or continue the response"
(`framework-questions-august-2026.md` §16.3); we do neither, because `max_tokens`
already sits at the model maximum and continuing confabulates. The trajectory is
`TRUNCATED` and is excluded.

Named for what the provider means by it. It was ``CONTINUABLE_STOP_REASON`` until
v7, when continuation was withdrawn — the old name asserted a policy that no
longer exists.
"""

CONTENT_FILTER_STOP_REASON = "content_filter"
"""The stop reason that means the generation was refused or filtered.

Anthropic's ``refusal`` maps here (`_providers/anthropic.py:4142-4143`). Its own
outcome rather than a residual, because a refusal on a D2 or D3 instance is a
result — see :attr:`Completeness.CONTENT_FILTERED`.
"""

CONTEXT_STOP_REASONS: frozenset[str] = frozenset({"model_length", "unknown"})
"""Stop reasons that mean the context window was the binding constraint.

Both, deliberately (§6.3). Bedrock maps `model_context_window_exceeded` to
``model_length`` (`_providers/bedrock.py:789-790`); the first-party Anthropic
provider has no case for it and it falls through to ``unknown``
(`_providers/anthropic.py:4144-4145`). They mean the same thing today and they
will still mean the same thing after the one-line upstream fix. A derivation
testing only ``unknown`` silently reclassifies every exhausted trajectory the day
the pinned `inspect_ai` release is bumped.

This survives the removal of everything built around it: the per-turn cap
arithmetic that was supposed to keep overflow off this branch is gone (§0 v7),
which makes the branch *more* reachable rather than less.

The cost, recorded rather than hidden: ``unknown`` is the fall-through for *every*
unrecognised stop reason, so this state reads "did not finish, probably the
window" rather than a clean diagnosis. `stop_details` is the plausible future home
of a cleaner signal and was ``null`` on every probe call (§6.6).
"""


class Completeness(str, Enum):
    """How a trajectory ended — axis 1, a trace fact.

    The first five are `specification.md` §6.3's outcomes. The last two are
    residuals describing endings the five do not cover, and are not scoreable
    either. **They must not be folded into the five**: `UNCLASSIFIED` means "we
    do not know", never "we know and did not keep it".

    Precedence when several apply: errored, limit_tripped, then the final turn's
    stop reason. Components — model turns, the stop-reason sequence — are
    retained whichever outcome wins, so nothing collapses.
    """

    CLEAN = "clean"
    """The final turn stopped for a reason of the model's own — ``stop`` or
    ``tool_calls``."""

    TRUNCATED = "truncated"
    """The final turn stopped at ``max_tokens``: the output cap was hit.

    Terminal. Nothing is recovered — §0 v7, and the module docstring."""

    CONTEXT_EXHAUSTED = "context_exhausted"
    """The final turn stopped at ``model_length`` or ``unknown`` — see
    :data:`CONTEXT_STOP_REASONS`."""

    LIMIT_TRIPPED = "limit_tripped"
    """A sample limit unwound the solver."""

    CONTENT_FILTERED = "content_filtered"
    """The final turn stopped at ``content_filter``.

    Its own outcome rather than a residual, because a refusal is a **result**
    (§0 v5): this is medical genetics, D2 and D3 present findings that cannot be
    true, and C1 and A5 already reward calibrated refusal. Bucketing a refusal
    with "no model turn was observed" throws the signal away.

    Not scoreable all the same — the checks a filtered trajectory did not reach
    are unreachable, not failed."""

    ERRORED = "errored"
    """An `ErrorEvent` was recorded. A residual, outside §6.3's five."""

    UNCLASSIFIED = "unclassified"
    """No model turn was observed, or the final turn's stop reason is not one the
    five outcomes describe. A residual, and deliberately not folded into `CLEAN`:
    `CLEAN` is defined as ``stop`` or ``tool_calls``, and this is neither."""


INVERSION_CONTROL: dict[str, Completeness] = {
    "stop": Completeness.CLEAN,
    "tool_calls": Completeness.CLEAN,
    "max_tokens": Completeness.TRUNCATED,
    "model_length": Completeness.CONTEXT_EXHAUSTED,
    "unknown": Completeness.CONTEXT_EXHAUSTED,
    "content_filter": Completeness.CONTENT_FILTERED,
}
"""Final stop reason to completeness, for a trajectory with no limit and no error.

Every value of `StopReason` (`model/_model_output.py:95-102`) appears exactly
once. Exported so the mapping is pinned by a test rather than by the branch
structure of :func:`derive_status` — the inversion this module was delivered with
survived a green suite because nothing asserted the mapping directly.
"""


class StopCause(str, Enum):
    """Why the *solver* stopped — axis 2, a harness fact.

    §6.3: the carrier records what the derivation cannot. `TURN_CAP` is the one
    cause with no trace signature at all; the rest name which trace signal the
    solver acted on, and exist so §5's exclusion counts are per cause rather than
    one bucket.

    `HEADROOM` and `CONTINUATION_BUDGET` were members until v7 and are
    **withdrawn with the machinery that produced them** (§0 v7). `HEADROOM` was
    the one cause the trace could not see, which is what
    :func:`counts_toward_rate` was built around; that function still reads both
    records, because `TURN_CAP` needs it too.
    """

    MODEL_STOPPED = "model_stopped"
    """The model ended its turn with no tool calls."""

    TURN_CAP = "turn_cap"
    """The solver's own turn bound was reached.

    The only cause with no trace signature: the last turn ended at ``stop`` or
    ``tool_calls`` and the derivation reads `CLEAN`. Only the carrier knows."""

    OUTPUT_CAP = "output_cap"
    """A turn stopped at ``max_tokens``, so the solver stopped.

    Nothing is continued (§0 v7). Replaces `CONTINUATION_BUDGET`, which recorded
    the same trace signal arriving after a budget that no longer exists."""

    CONTEXT_SIGNAL = "context_signal"
    """A turn stopped at ``model_length`` or ``unknown``.

    Terminal in both spellings: a ``model_length`` output is a `BadRequestError`
    converted to a synthetic `ModelOutput` whose content is the error text
    (`_providers/anthropic.py:1418-1449`), so there is nothing to build on."""

    TOOL_ERROR = "tool_error"
    """The solver abandoned the loop on a tool failure it chose not to feed back.

    :func:`~genomics_harness.solver.research_agent` never sets it. MCP errors are
    converted to `ToolError` and fed back to the model rather than raised
    (`tool/_mcp/_local.py:54-86`), and how a model responds to a failing tool is
    behaviour under test, not a reason to end a trajectory. The value exists
    because §6.3 names it and a later policy may want it."""


class TrajectoryStatus(BaseModel):
    """The completeness classification plus the components it was built from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    completeness: Completeness
    model_turns: int = 0
    stop_reasons: list[str] = Field(default_factory=list)
    """Per-turn ``ModelEvent.output.stop_reason``, in order.

    A ``max_tokens`` entry anywhere but the last position would mean a turn was
    truncated and something carried on regardless. The solver cannot produce
    that shape — it breaks on the first one — and the sequence is kept whole so
    that a foreign trajectory which does is visible rather than smoothed over."""

    limit_type: str | None = None
    """``message`` | ``time`` | ``working`` | ``token`` | ``turn`` | ``cost`` |
    ``operator`` | ``custom``."""

    limit_value: float | None = None
    limit_message: str | None = None
    source: str = "derived"
    """``derived`` from the event stream, or ``carried`` from a solver write."""

    @property
    def scoreable(self) -> bool:
        """Whether checks should be scored rather than flagged, on the trace alone.

        A trajectory that did not run to completion is a flagged trajectory, not
        a scored one: the checks it did not reach are unreachable, not failed.

        This reads the derivation only. The solver's turn cap is not in the
        trace, so §5's exclusion decision is :func:`counts_toward_rate`, which
        takes the carrier as well.
        """
        return self.completeness is Completeness.CLEAN

    def with_source(self, source: str) -> TrajectoryStatus:
        return self.model_copy(update={"source": source})


class SolverRecord(BaseModel):
    """The solver's own account of its run, parked in `state.store`.

    **Not a `TrajectoryStatus`, on purpose.** §6.3: "The provisional state is
    structurally incapable of decoding as a terminal status. Not a member of the
    same type as the five above. The carrier is a record with a separate 'solver
    completed its write' field, so a consumer that skips reconciliation fails to
    decode rather than reading a plausible-looking answer."

    Two mechanisms make that hold, and both are tested:

    * ``complete`` is required and defaults to nothing. A record written before
      the loop carries ``complete=False`` and ``completeness=None``, so there is
      no member of :class:`Completeness` to misread.
    * the payload does not validate as a :class:`TrajectoryStatus`.
      ``TrajectoryStatus`` sets ``extra="forbid"`` and requires ``completeness``,
      so ``TrajectoryStatus.model_validate(store[CARRIER_KEY])`` raises on both
      counts. A consumer that skips :func:`reconcile` gets an exception rather
      than a plausible-looking answer.

    The write happens before the loop and is updated as the loop proceeds: a
    limit unwinds the solver by exception and the live `TaskState` is recovered
    for scoring (`_eval/task/run.py:2104-2105`), so a record written at the
    bottom of the function is absent on exactly the trajectories it exists to
    describe.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    complete: bool
    """Whether the solver reached its own final write. ``False`` means the record
    is provisional and says nothing about how the trajectory ended."""

    stop_cause: StopCause | None = None
    """Why the solver stopped. ``None`` while provisional."""

    completeness: Completeness | None = None
    """The solver's view of how the *model turns* ended, for cross-checking
    against the derivation. ``None`` while provisional.

    Scoped to the same question the derivation answers, so a divergence means a
    bug in one of the two. Why the *loop* stopped is not expressed here — it is a
    fact about the solver, not about the turns, and it travels in
    :attr:`stop_cause`."""

    model_turns: int = 0
    solver_version: str | None = None
    """`SolverConfig.digest`, so a carried record names the policy that produced
    it (§7)."""

    detail: str | None = None
    """Free text for the stop, e.g. the stop reason and the cap it was measured
    against. Never a substitute for :attr:`stop_cause` — nothing parses it."""

    def as_status(self) -> TrajectoryStatus:
        """The carried view as a :class:`TrajectoryStatus`, for comparison only.

        Raises:
            ValueError: The record is provisional. Refusing here is the point:
                a provisional carrier is not an answer.
        """
        if not self.complete or self.completeness is None:
            raise ValueError(
                "solver record is provisional: the solver did not reach its "
                "final write, so it carries no completeness. Reconcile against "
                "the derivation instead of reading the carrier"
            )
        return TrajectoryStatus(
            completeness=self.completeness,
            model_turns=self.model_turns,
            source="carried",
        )


def _stop_reason(event: ModelEvent) -> str:
    """The one `ModelEvent` field the reading surface permits (§6.4 invariant 8).

    `ModelEvent.output` is typed non-optional (`event/_model.py`), and
    `ModelOutput.stop_reason` reads ``choices[0].stop_reason``, which raises on an
    output with no choices — a failed generate leaves exactly that. Guarded here
    rather than at every call site.
    """
    try:
        return str(event.output.stop_reason)
    except (AttributeError, IndexError):
        return "unknown"


def derive_status(events: Sequence[Event]) -> TrajectoryStatus:
    """Classify a trajectory from its event stream alone.

    Works unchanged on `transcript().events` inside a scorer (inline or
    deferred) and on `EvalSample.events` in the analysis layer.

    The classification is the final turn's stop reason under
    :data:`INVERSION_CONTROL`, with an `ErrorEvent` and then a `SampleLimitEvent`
    taking precedence over it. Nothing here reads the solver's carrier — the two
    records are computed independently on purpose, so that a bug in one shows up
    as a disagreement rather than as agreement (§6.3).

    Args:
        events: The trajectory's events, in order.

    Returns:
        The classification and the components behind it.
    """
    stop_reasons: list[str] = []
    model_turns = 0
    limit: SampleLimitEvent | None = None
    errored = False

    for event in events:
        if isinstance(event, ModelEvent):
            model_turns += 1
            stop_reasons.append(_stop_reason(event))
        elif isinstance(event, SampleLimitEvent) and limit is None:
            limit = event
        elif isinstance(event, ErrorEvent):
            errored = True

    if errored:
        completeness = Completeness.ERRORED
    elif limit is not None:
        completeness = Completeness.LIMIT_TRIPPED
    elif not stop_reasons:
        completeness = Completeness.UNCLASSIFIED
    else:
        completeness = INVERSION_CONTROL.get(
            stop_reasons[-1], Completeness.UNCLASSIFIED
        )

    return TrajectoryStatus(
        completeness=completeness,
        model_turns=model_turns,
        stop_reasons=stop_reasons,
        limit_type=None if limit is None else str(limit.type),
        limit_value=None if limit is None else limit.limit,
        limit_message=None if limit is None else limit.message,
        source="derived",
    )


def record_provisional(store: Any, *, solver_version: str | None = None) -> None:
    """Park the provisional carrier, before the loop runs.

    Args:
        store: `TaskState.store`.
        solver_version: `SolverConfig.digest`, when the solver has one.
    """
    store.set(
        CARRIER_KEY,
        SolverRecord(complete=False, solver_version=solver_version).model_dump(
            mode="json"
        ),
    )


def record_final(store: Any, record: SolverRecord) -> None:
    """Park the solver's final account.

    Args:
        store: `TaskState.store`.
        record: What the solver believes happened. Must be complete.

    Raises:
        ValueError: ``record`` is not complete, or carries no completeness.
    """
    if not record.complete or record.completeness is None:
        raise ValueError(
            "record_final() requires a complete record; use record_provisional()"
        )
    store.set(CARRIER_KEY, record.model_dump(mode="json"))


def carried_record(store: Any) -> SolverRecord | None:
    """Read back what the solver wrote, if anything.

    Args:
        store: `TaskState.store`, or `EvalSample.store` — a `Store` and a plain
            `dict` both answer ``.get``.

    Returns:
        The record, or ``None`` when the solver wrote nothing. A provisional
        record comes back as a provisional record, never as a status.
    """
    payload = store.get(CARRIER_KEY)
    if payload is None:
        return None
    return SolverRecord.model_validate(payload)


def reconcile(
    derived: TrajectoryStatus, carried: SolverRecord | None
) -> tuple[TrajectoryStatus, list[str]]:
    """Return the authoritative status and any disagreements with the carrier.

    The derivation wins. It is auditable from the trace, while the carrier is a
    solver's assertion about itself; a divergence is a bug in the solver or in
    the derivation, and either way it should be visible rather than resolved.

    A provisional carrier is reported as a disagreement rather than as silence:
    "the solver did not finish" is information, and it is the normal shape of a
    limit trip, where the unwind happens before the solver's final write. §5
    routes only a conflict between two **complete** records to manual review.

    Completeness is the only field compared. ``continuations`` was the second
    until v7, and went with the continuation policy; ``model_turns`` is on both
    records but is not compared, because the derivation counts every `ModelEvent`
    in the stream it is given and a judge's own generate call lands in that stream
    at scoring time.

    Args:
        derived: From :func:`derive_status`.
        carried: From :func:`carried_record`, or ``None`` when the solver wrote
            nothing.

    Returns:
        The derived status, and a list of human-readable disagreements.
    """
    if carried is None:
        return derived, []

    if not carried.complete or carried.completeness is None:
        return derived, [
            "carrier: the solver did not reach its final write, so its record is "
            f"provisional; the trace says {derived.completeness.value!r}"
        ]

    disagreements: list[str] = []
    if carried.completeness is not derived.completeness:
        disagreements.append(
            f"completeness: solver said {carried.completeness.value!r}, "
            f"trace says {derived.completeness.value!r}"
        )
    return derived, disagreements


_SOLVER_INCOMPLETE_CAUSES: frozenset[StopCause] = frozenset(
    {StopCause.TURN_CAP, StopCause.TOOL_ERROR}
)
"""Stop causes that mean the model was not finished when the solver stopped it
**and the trace does not say so**.

`OUTPUT_CAP` and `CONTEXT_SIGNAL` are absent deliberately: each names a stop
reason the derivation already classifies as unscoreable, so the exclusion is
reported against the trace fact rather than the solver's assertion about it.
Listing them here would shadow `completeness:truncated` with
`stop_cause:output_cap` and make §5's per-cause counts depend on which record
was consulted first.
"""


def counts_toward_rate(
    derived: TrajectoryStatus, carried: SolverRecord | None
) -> tuple[bool, str | None]:
    """§5's exclusion decision, over both records.

    ``scoreable`` answers it from the trace. This answers it from the trace *and*
    the carrier, which is what :attr:`StopCause.TURN_CAP` needs: the solver stops
    after a turn that ended at ``stop`` or ``tool_calls``, so the derivation reads
    `CLEAN` and has nothing to see. A model that cannot finish inside our turn cap
    is telling us something, and §5 reads that count as behaviour rather than
    hygiene.

    Args:
        derived: From :func:`derive_status`.
        carried: From :func:`carried_record`, or ``None``.

    Returns:
        Whether the trajectory's verdicts belong in the primary rate, and the
        reason for an exclusion. Every exclusion is reported as a count
        alongside the rate and is a result about the model, not hygiene (§5).
    """
    if not derived.scoreable:
        return False, f"completeness:{derived.completeness.value}"
    if carried is None:
        return True, None
    if not carried.complete:
        return False, "carrier:provisional"
    _, disagreements = reconcile(derived, carried)
    if disagreements:
        return False, "carrier:disagreement"
    if carried.stop_cause in _SOLVER_INCOMPLETE_CAUSES:
        assert carried.stop_cause is not None
        return False, f"stop_cause:{carried.stop_cause.value}"
    return True, None
