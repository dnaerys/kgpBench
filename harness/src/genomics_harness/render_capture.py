"""The renderer-capture scorer — proof that both scoring paths see one trajectory.

Judging is a second pass (`design/specification.md` §6.3), so every verdict is
produced by a scorer that ran over a `TaskState` and a transcript **rebuilt from
disk** (`_eval/score.py:413-434`). Two divergences between that rebuild and the
inline object are already known — `state.tools` is empty and `sample_limits()`
raises (§6.2 invariant 6). The risk this module addresses is a third appearing at
a framework bump, in the messages or the transcript, which is where the judges
live.

Judges cannot establish that. A judge is an LLM, so one trajectory scored twice
gives two samples from a distribution; separating judge variance from a
reconstruction failure would need repeats at every check, and the question is not
statistical. It is exact, and it is about *inputs*: does the scorer see the same
trajectory in both paths?

A scorer is the only extension point invoked with the same signature inline and
deferred. A `@hooks` `SampleEnd` callback does not run on the deferred path, and
rendering offline from an `EvalLog` is a third path that tests neither. So the
comparison is a scorer: it renders the trajectory, digests the rendering, and
returns the digest. Run it inline on the generation task and again over the
stored log with `score(log, [render_capture()], action="append")` — equal digests
mean the deferred path presented the same inputs, and the per-section digests name
the channel that diverged rather than only reporting that something did.

It takes **no factory arguments**, which is what keeps the generation log
rubric-free (§6.3, `foundations.md` §2.1): scorer arguments are captured into
`header.json` generically, whether or not scoring ran, so a rubric stays out of a
log by not being an argument — never by setting `score=False`.

**It digests the document a judge receives**, not a separate rendering of its
own. Anything else and the digest stops being evidence about what a judge read,
which is the whole of what the comparison is for. :func:`Score.value` is the
digest of :func:`~genomics_harness.renderer.render_trajectory`'s output; the
per-section digests localise a divergence to a channel and are computed over the
same blocks.

**The digest supersedes `RENDERER_VERSION`**, which detects a revision only when
someone remembers to bump it. Both are carried: the version names the revision,
the digest proves it.

The rendering itself, and the reading discipline it observes, are in
:mod:`genomics_harness.renderer`.
"""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai.event import Event
from inspect_ai.log import transcript
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from .digest import sha256_digest
from .renderer import SECTIONS, render
from .version import RENDERER_VERSION

__all__ = [
    "SECTIONS",
    "capture",
    "render_capture",
]


def capture(state: TaskState, events: Sequence[Event]) -> Score:
    """Build the capture `Score` for a rendered trajectory.

    Split out from the scorer so the comparison can be exercised without an eval.

    Args:
        state: The scorer's `TaskState`.
        events: The trajectory's events, in order.

    Returns:
        A `Score` whose value is the digest of the document a judge reads, with
        the per-section digests and lengths in metadata. ``section_lengths`` is
        also what §5 reports as the size of the reasoning channel a judge reads,
        in characters — exact, and free.
    """
    rendering = render(state, events)
    return Score(
        value=sha256_digest(rendering.text),
        metadata={
            "renderer_version": RENDERER_VERSION,
            "rendering_length": len(rendering.text),
            "section_digests": {
                name: sha256_digest(rendering.sections[name]) for name in SECTIONS
            },
            "section_lengths": {
                name: len(rendering.sections[name]) for name in SECTIONS
            },
        },
    )


@scorer(metrics=[])
def render_capture() -> Scorer:
    """Digest the trajectory rendering, inline and on re-score.

    Takes no arguments, so its `header.json` entry is a name and an empty options
    dict, and calls no model, so it costs nothing to run on every generation.
    """

    async def score(state: TaskState, target: Target) -> Score:
        return capture(state, transcript().events)

    return score
