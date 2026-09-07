"""The turn loop, its fixed output cap, and what a mock cannot establish.

**Read the docstrings before trusting a green run here.** A mock confirms
plumbing, never semantics (`design_document.md` §6.5, §15). Every test in this
file drives `mockllm/model`, so what it establishes is that the solver's branches
and its carrier behave as written when a given `stop_reason` arrives. That Claude
*produces* those values is a source finding, and for ``max_tokens`` a live one —
observed repeatedly in the probe rounds (§0 v7).

**What this file used to assert, and what replaced it.** Until v7 half of it
tested option D: the per-turn cap ``M_t = clamp(W - I_t - headroom, floor,
M_max)``, the input estimator with its cache accounting, the pre-emptive headroom
stop, and a continuation turn appended after a `max_tokens` truncation. Eight
tests are deleted rather than adjusted, because the policy they encoded is
reversed rather than refined (§0 v7, §15). Each of the tests that replaces one
names what it replaces.

One mock shape is worth naming because a provider would not produce it:
`mockllm/model` is absent from `model/_model_data/*.yml`, so `get_model_info`
returns ``None`` and **M_max** has to come from the config. That is a real finding
about the harness, not a mock artefact — a model missing from the info DB needs
`set_model_info` or an explicit `SolverConfig` — and
:func:`test_a_model_absent_from_the_info_db_raises` pins it.
"""

from __future__ import annotations

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ModelOutput, get_model, get_model_info
from inspect_ai.tool import ToolDef
from pydantic import ValidationError

from genomics_harness import (
    Completeness,
    SolverConfig,
    StopCause,
    carried_record,
    counts_toward_rate,
    derive_status,
    read_log,
    reconcile,
    research_agent,
    resolve_output_limit,
    trajectory_status,
)
from genomics_harness.solver import OutputLimitUnknown

# `mockllm/model` is absent from the info DB, so M_max is pinned. 4,000 rather
# than Claude's 128,000: the number only has to be legible in an assertion.
SMALL = SolverConfig(max_output_tokens=4_000, turn_cap=6, reasoning_effort=None)


# -- M_max is read, never assumed -----------------------------------------


@pytest.mark.parametrize(
    "model", ["anthropic/claude-opus-5", "anthropic/claude-sonnet-5"]
)
def test_the_models_under_test_are_in_the_info_db(model):
    """Both declare an output cap, so §6.3's "read, never assumed" is satisfiable
    without `set_model_info` (`_model_data/anthropic.yml:63-127`)."""
    info = get_model_info(model)
    assert info is not None
    assert info.output_tokens == 128_000

    limit = resolve_output_limit(model, SolverConfig(reasoning_effort="xhigh"))
    assert limit.max_output_tokens == 128_000
    assert limit.source == "model_info"


def test_a_model_absent_from_the_info_db_raises():
    """The behaviour option A inherits from option D unchanged, and the one part
    of the cap machinery that had to survive the removal.

    A default here would not fail — it would run, and every trajectory in the
    corpus would carry a cap nobody chose. That is the §7 failure shape exactly:
    a number produced by machinery that changed underneath it.
    """
    assert get_model_info("mockllm/model") is None
    with pytest.raises(OutputLimitUnknown, match="read, never assumed"):
        resolve_output_limit("mockllm/model", SolverConfig(reasoning_effort="xhigh"))


def test_the_config_overrides_the_info_db():
    limit = resolve_output_limit("anthropic/claude-opus-5", SMALL)
    assert limit.max_output_tokens == 4_000
    assert limit.source == "config"


def test_the_context_window_is_no_longer_a_solver_input():
    """§7's row is turn cap, `reasoning_effort`, M_max and `token_limit`.

    **W** was there for option D's arithmetic — `W - I_t - headroom` — and went
    with it (§0 v7). Naming it is now an error rather than a field that silently
    does nothing.
    """
    assert set(SolverConfig.model_fields) == {
        "turn_cap",
        "reasoning_effort",
        "max_output_tokens",
        "token_limit",
        "solver_name",
    }
    with pytest.raises(ValueError, match="context_window"):
        SolverConfig(reasoning_effort="xhigh", context_window=1_000_000)


def test_the_withdrawn_cap_fields_cannot_be_set():
    """`extra="forbid"` turns each removed knob into a loud failure. A config
    carrying `floor=32000` would otherwise validate, digest differently from one
    without it, and change nothing about the run."""
    for withdrawn in ("floor", "headroom", "chars_per_token", "continuation_budget"):
        with pytest.raises(ValueError, match=withdrawn):
            SolverConfig(reasoning_effort="xhigh", **{withdrawn: 1})


def test_reasoning_effort_is_in_the_digest():
    """§7, and §0 v6's reason for putting it there: unset, it does not change the
    trajectory — it empties the reasoning channel *and* the usage record, with
    nothing erroring (§6.6). The digest is what records which one a corpus ran
    at."""
    xhigh = SolverConfig(reasoning_effort="xhigh")
    assert xhigh.reasoning_effort == "xhigh"
    assert xhigh.digest != SolverConfig(reasoning_effort="high").digest
    assert xhigh.digest != SolverConfig(reasoning_effort=None).digest


def test_reasoning_effort_is_required_with_no_default():
    """§6.6 and §7: "``SolverConfig`` makes it a **required field with no
    default** … a default is one model's effort silently carried onto another".

    The fact asserted is the construction itself. §0 v6 asked for a static rule
    that would fail when the field was absent from a run; §0 v9 ruled that a
    required field replaces it, and a required field is only that if omitting it
    raises here — not if some downstream consumer happens to notice.
    """
    with pytest.raises(ValidationError, match="reasoning_effort"):
        SolverConfig()

    assert SolverConfig.model_fields["reasoning_effort"].is_required()


def test_making_the_field_required_moved_no_solver_digest():
    """Removing the ``"xhigh"`` default changes no digest any run recorded.

    Every construction already passed the value, so the model dump the digest
    covers is byte-identical to what it was — asserted against the literal both
    live rounds wrote to their manifests as ``solver_sha256`` rather than
    against a value recomputed here, which would move with the code it is meant
    to pin (`round-layout-findings-august-2026.md` §0).
    """
    real_run = SolverConfig(reasoning_effort="xhigh", max_output_tokens=None)
    assert real_run.digest == (
        "acac1727803f9025ea04fd225c29433522c631041f50872f3095f706353a728a"
    )

    mock_round = SolverConfig(reasoning_effort="xhigh", max_output_tokens=128_000)
    assert mock_round.short == "9fcfadc0b310beea"


# -- the loop, end to end -------------------------------------------------


async def _echo(text: str) -> str:
    """Echo a string.

    Args:
        text: What to echo.
    """
    return f"echo: {text}"


ECHO = ToolDef(_echo, name="echo").as_tool()


def _task(name: str, tools=(ECHO,), config: SolverConfig = SMALL) -> Task:
    return Task(
        name=name,
        dataset=MemoryDataset(
            samples=[Sample(input="investigate", id="i1", target="")]
        ),
        solver=research_agent(list(tools), config),
    )


def _outputs(script):
    """A `mockllm` callable driven by the number of assistant turns so far.

    Turn number comes from the conversation, never from a shared counter: epochs
    run concurrently and a counter interleaves across them (`CLAUDE.md` §6).
    """

    def outputs(input, tools, tool_choice, config):
        turn = sum(1 for m in input if m.role == "assistant")
        make = script[min(turn, len(script) - 1)]
        return make(config)

    return outputs


def _content(text: str, stop_reason: str):
    def make(config):
        return ModelOutput.from_content(
            model="mockllm/model", content=text, stop_reason=stop_reason
        )

    return make


def _tool_call():
    def make(config):
        return ModelOutput.for_tool_call(
            model="mockllm/model", tool_name="echo", tool_arguments={"text": "hi"}
        )

    return make


def _run(task: Task, script, tmp_path, name: str):
    model = get_model("mockllm/model", custom_outputs=_outputs(script), memoize=False)
    return inspect_eval(
        task, model=model, display="none", log_dir=str(tmp_path / name)
    )[0]


def test_a_clean_run_with_one_tool_call(tmp_path):
    log = _run(
        _task("solver-clean"),
        [_tool_call(), _content("done", "stop")],
        tmp_path,
        "clean",
    )
    stored = read_log(log.location)
    sample = (stored.samples or [])[0]

    derived = trajectory_status(sample)
    assert derived.completeness is Completeness.CLEAN
    assert derived.stop_reasons == ["tool_calls", "stop"]

    carried = carried_record(sample.store)
    assert carried is not None
    assert carried.complete
    assert carried.stop_cause is StopCause.MODEL_STOPPED
    assert reconcile(derived, carried)[1] == []
    assert counts_toward_rate(derived, carried) == (True, None)

    functions = [e.function for e in sample.events if e.event == "tool"]
    assert functions == ["echo"]


def test_an_output_cap_stop_ends_the_trajectory(tmp_path):
    """Replaces `test_an_output_cap_stop_is_continued_once` and
    `test_the_continuation_budget_is_a_hard_stop`.

    Both asserted the v4 policy: the first that a `max_tokens` turn was followed
    by a harness-written user message and a second generation, the second that a
    third one was refused. The reversal is not that the budget is now zero — it is
    that **there is no mechanism**, because what a continuation replays is a
    provider summary rather than the reasoning, and a cut inside thinking
    confabulates a resumption that looks complete enough to score (§0 v7).

    So: one model turn, `TRUNCATED` on both records, excluded, and **no second
    request** — which is what the script's clean second entry is here to catch.
    """
    log = _run(
        _task("solver-truncated"),
        [_content("cut off mid-", "max_tokens"), _content("never asked", "stop")],
        tmp_path,
        "truncated",
    )
    stored = read_log(log.location)
    sample = (stored.samples or [])[0]

    derived = trajectory_status(sample)
    assert derived.completeness is Completeness.TRUNCATED
    assert derived.model_turns == 1
    assert derived.stop_reasons == ["max_tokens"]

    carried = carried_record(sample.store)
    assert carried is not None
    assert carried.stop_cause is StopCause.OUTPUT_CAP
    assert carried.completeness is Completeness.TRUNCATED
    assert carried.detail and "max_tokens" in carried.detail
    assert reconcile(derived, carried)[1] == []
    assert counts_toward_rate(derived, carried) == (False, "completeness:truncated")


def test_nothing_is_appended_to_the_conversation(tmp_path):
    """§11, and the strongest form of it the harness has ever been able to state.

    The continuation turn was the last thing the harness wrote into the model's
    conversation, and it is gone (§0 v7). A truncated trajectory is where it used
    to appear, so that is where this looks: one user message, the instance prompt,
    carrying `source="input"` — which `sample_messages` sets on the seeded prompt
    and only on it (`_eval/task/util.py:25-32`).
    """
    log = _run(
        _task("solver-noinject"),
        [_content("cut off mid-", "max_tokens")],
        tmp_path,
        "noinject",
    )
    sample = (log.samples or [])[0]

    users = [m for m in sample.messages if m.role == "user"]
    assert len(users) == 1
    assert users[0].text == "investigate"
    assert users[0].source == "input"
    assert not [m for m in sample.messages if m.role == "system"]


@pytest.mark.parametrize("reason", ["model_length", "unknown"])
def test_context_exhaustion_stops_the_loop_at_once(tmp_path, reason):
    """Terminal in both spellings. On a real provider a `model_length` output is
    an API error converted to a synthetic `ModelOutput` whose content is the error
    text (`_providers/anthropic.py:1418-1449`).

    The mock supplies the stop reason directly, so this establishes the branch,
    not that Claude emits it — item 4.13.
    """
    log = _run(
        _task(f"solver-exhausted-{reason}"),
        [_content("context window exceeded", reason)],
        tmp_path,
        f"exhausted-{reason}",
    )
    stored = read_log(log.location)
    sample = (stored.samples or [])[0]

    derived = trajectory_status(sample)
    assert derived.completeness is Completeness.CONTEXT_EXHAUSTED
    assert derived.model_turns == 1

    carried = carried_record(sample.store)
    assert carried is not None
    assert carried.stop_cause is StopCause.CONTEXT_SIGNAL
    assert reconcile(derived, carried)[1] == []


def test_a_refusal_is_classified_rather_than_bucketed(tmp_path):
    """`content_filter` reaches the loop's ordinary "no tool calls" exit, so the
    carrier has to name it explicitly or it says `CLEAN` where the derivation says
    `CONTENT_FILTERED` — and §5 would read a routine refusal as a solver bug."""
    log = _run(
        _task("solver-refused"),
        [_content("I can't help with that.", "content_filter")],
        tmp_path,
        "refused",
    )
    stored = read_log(log.location)
    sample = (stored.samples or [])[0]

    derived = trajectory_status(sample)
    assert derived.completeness is Completeness.CONTENT_FILTERED

    carried = carried_record(sample.store)
    assert carried is not None
    assert carried.completeness is Completeness.CONTENT_FILTERED
    assert carried.stop_cause is StopCause.MODEL_STOPPED
    assert reconcile(derived, carried)[1] == []
    assert counts_toward_rate(derived, carried) == (
        False,
        "completeness:content_filtered",
    )


def test_the_turn_cap_stops_the_loop(tmp_path):
    """The one stop cause with no trace signature, and now the only reason
    `counts_toward_rate` needs the carrier at all."""
    log = _run(
        _task("solver-turncap", config=SMALL.model_copy(update={"turn_cap": 3})),
        [_tool_call()],
        tmp_path,
        "turncap",
    )
    stored = read_log(log.location)
    sample = (stored.samples or [])[0]

    derived = trajectory_status(sample)
    assert derived.model_turns == 3
    assert derived.completeness is Completeness.CLEAN  # the trace cannot tell
    carried = carried_record(sample.store)
    assert carried is not None
    assert carried.stop_cause is StopCause.TURN_CAP
    assert reconcile(derived, carried)[1] == []
    assert counts_toward_rate(derived, carried) == (False, "stop_cause:turn_cap")


def test_the_cap_on_the_wire_is_m_max_on_every_turn(tmp_path):
    """Replaces `test_the_per_turn_cap_reaches_the_request`, which asserted that
    the number varied with the conversation.

    `custom_outputs` as a callable is the last point before provider-specific
    serialisation, so the `GenerateConfig` it receives is what the request
    carries. The number is M_max, identical on every turn, and it comes from the
    info DB or an explicit config — never a literal.
    """
    seen: list[int | None] = []

    def outputs(input, tools, tool_choice, config):
        seen.append(config.max_tokens)
        turn = sum(1 for m in input if m.role == "assistant")
        if turn == 0:
            return ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="echo",
                tool_arguments={"text": "hi"},
            )
        return ModelOutput.from_content(
            model="mockllm/model", content="done", stop_reason="stop"
        )

    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    inspect_eval(
        _task("solver-cap-on-wire"),
        model=model,
        display="none",
        log_dir=str(tmp_path / "cap"),
    )
    assert seen == [4_000, 4_000]


def test_the_per_call_config_does_not_drop_reasoning_effort(tmp_path):
    """§14.21, settled offline. The solver passes `max_tokens` per call while
    `reasoning_effort` is set once at task level, and if the merge *replaced*
    rather than overlaid, every generation call would carry no effort — and by
    §6.6 that means an entire corpus with a blank reasoning channel and a usage
    record saying the model barely thought. Silently.

    It overlays. `GenerateConfig.merge` (`model/_generate_config.py:379-387`)
    copies `self` and writes each key from `other` **only when `other`'s value is
    not None**, and `_resolve_config` (`model/_model.py:1607-1629`) merges the
    active generate config in first for the active model, then the passed config
    on top. So the task's `reasoning_effort` survives and the solver's
    `max_tokens` wins. `CLAUDE.md` §7.13 and §7.28 co-cite that line range because
    they describe its two branches, not because one of them is wrong.

    Asserted on the composed request body rather than on the merge in isolation,
    because it is the composed body the model would receive.
    """
    seen: list[tuple[str | None, int | None]] = []

    def outputs(input, tools, tool_choice, config):
        seen.append((config.reasoning_effort, config.max_tokens))
        return ModelOutput.from_content(
            model="mockllm/model", content="done", stop_reason="stop"
        )

    task = _task("solver-effort-survives")
    task.config = GenerateConfig(reasoning_effort="xhigh")
    inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=outputs, memoize=False),
        display="none",
        log_dir=str(tmp_path / "effort"),
    )
    assert seen == [("xhigh", 4_000)]


def test_the_provisional_carrier_survives_a_limit(tmp_path):
    """A limit unwinds the solver by exception, so the final write never runs and
    the provisional record is what reaches the log
    (`_eval/task/run.py:2104-2105`)."""
    model = get_model(
        "mockllm/model", custom_outputs=_outputs([_tool_call()]), memoize=False
    )
    log = inspect_eval(
        _task("solver-limit"),
        model=model,
        display="none",
        message_limit=3,
        log_dir=str(tmp_path / "limit"),
    )[0]
    stored = read_log(log.location)
    assert stored.status == "success"  # a limit is not an error

    sample = (stored.samples or [])[0]
    carried = carried_record(sample.store)
    assert carried is not None
    assert not carried.complete
    assert carried.completeness is None
    assert carried.solver_version == SMALL.digest

    derived = trajectory_status(sample)
    assert derived.completeness is Completeness.LIMIT_TRIPPED
    assert reconcile(derived, carried)[1], "a provisional carrier must not pass"


def test_the_solver_runs_with_no_tools_at_all(tmp_path):
    log = _run(
        _task("solver-notools", tools=()),
        [_content("done", "stop")],
        tmp_path,
        "notools",
    )
    stored = read_log(log.location)
    assert derive_status((stored.samples or [])[0].events).completeness is (
        Completeness.CLEAN
    )
