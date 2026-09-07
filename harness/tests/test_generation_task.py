"""The generation task — what it carries, what it must not, and one live run.

Three things are being established, in order of how expensive they are to get
wrong later.

**The header stays rubric-free and ground-truth-free.** `score=False` is not the
mechanism (§7.25); what keeps a rubric out is that no scorer takes arguments. The
same rule applies one level up and is easier to miss: `@task` captures *every*
parameter including defaults into `EvalSpec.task_args`
(`_eval/registry.py:158-160`, `_eval/task/log.py:245-246`), so an `InstanceSet`
passed as a task argument would put every prompt and its ground truth in
``header.json``.

**The name cannot collide with a scoring task.** The scorer list is not an input
to `task_identifier`, so under `eval_set` a scoring task is treated as already
satisfied by a generation task with the same name — run B ran no model, wrote no
log, and returned run A's log reporting success (§13.2).

**The whole thing runs.** One live pass against `https://db.dnaerys.org/mcp` with
`mockllm/model`, marked ``live_mcp`` and skippable. It exercises
`mcp_connection`, `ToolSource` resolution and a real tool result through
`execute_tools`. It establishes nothing about model behaviour — the model is a
mock.
"""

from __future__ import annotations

import json

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput

from genomics_harness import (
    GENERATION_TASK_PREFIX,
    SET_CONFIG_ENV,
    Completeness,
    SolverConfig,
    StopCause,
    assert_generation_only,
    build_generation_task,
    carried_record,
    genomics_generation,
    read_log,
    reconcile,
    resolve_set,
    trajectory_status,
)

MCP_URL = "https://db.dnaerys.org/mcp"

# mockllm/model is absent from the info DB (see test_solver.py), so a run has to
# pin M_max explicitly. `reasoning_effort=None` because a mock has no reasoning
# channel to empty; on a real model that setting is the failure in §6.6.
MOCK_SOLVER_CONFIG = SolverConfig(
    max_output_tokens=16_000,
    turn_cap=4,
    reasoning_effort=None,
)


def _task(instances, **kwargs):
    return build_generation_task(
        instances,
        mcp_url=MCP_URL,
        solver_config=MOCK_SOLVER_CONFIG,
        **kwargs,
    )


# -- what the task carries ------------------------------------------------


def test_the_name_marks_it_as_generation(instances):
    built = _task(instances, instance_set="s1")
    assert built.name == f"{GENERATION_TASK_PREFIX}-s1"
    assert_generation_only(built)


def test_a_task_named_like_a_scoring_task_is_rejected(instances):
    """The positive control for :func:`assert_generation_only`.

    `Task.name` is read-only, so the wrong name is built rather than assigned.
    """
    from inspect_ai import Task

    from genomics_harness import build_dataset, render_capture

    scoring_shaped = Task(
        name="genomics-scoring-repeats3",
        dataset=build_dataset(instances),
        scorer=[render_capture()],
    )
    with pytest.raises(AssertionError, match="task_identifier"):
        assert_generation_only(scoring_shaped)


def test_a_generation_task_with_a_rubric_bearing_scorer_is_rejected(instances):
    """The second half of the same control: the name is right, the scorer is not."""
    from inspect_ai import Task
    from inspect_ai.scorer import Score, Target, scorer
    from inspect_ai.solver import TaskState

    from genomics_harness import build_dataset

    @scorer(metrics=[])
    def rubric_bearing(catalogue: list[str]):
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value=float(len(catalogue)))

        return score

    built = Task(
        name=f"{GENERATION_TASK_PREFIX}-oops",
        dataset=build_dataset(instances),
        scorer=[rubric_bearing(["A1", "B1"])],
    )
    with pytest.raises(AssertionError, match="header.json"):
        assert_generation_only(built)


def test_the_only_scorer_is_the_renderer_capture(instances):
    from inspect_ai._util.registry import registry_info

    built = _task(instances)
    assert built.scorer is not None
    assert len(built.scorer) == 1
    assert registry_info(built.scorer[0]).name.endswith("render_capture")


def test_version_and_metadata_come_from_the_instance_digest(instances):
    built = _task(instances)
    assert built.version == instances.version
    assert built.metadata is not None
    assert built.metadata["instances_sha256"] == instances.digest
    assert built.metadata["solver_sha256"] == MOCK_SOLVER_CONFIG.digest


def test_the_solver_config_travels_in_task_metadata(instances):
    """§7's row, and nothing else. The continuation message, the continuation
    budget, `floor`, `headroom`, `chars_per_token` and the context window were all
    in this payload until v7 and are withdrawn with the machinery (§0 v7)."""
    built = _task(instances)
    provenance = (built.metadata or {})["provenance"]
    solver = provenance["solver"]
    assert solver["turn_cap"] == 4
    assert solver["max_output_tokens"] == 16_000
    assert solver["reasoning_effort"] is None
    assert set(solver) == {
        "turn_cap",
        "reasoning_effort",
        "max_output_tokens",
        "token_limit",
        "solver_name",
    }


def test_reasoning_effort_is_carried_beside_the_digest_and_not_inside_only(
    instances,
):
    """§7's recorded-not-built item, now carried.

    The failure it exists for: effort is a per-model choice, so two models at
    two efforts differ on ``solver_sha256`` and the analysis layer meets a
    digest difference it cannot decompose — on the axis where the headline
    result *is* the per-model comparison. A digest cannot be taken apart, so the
    remedy is a second carrier beside it.

    **Beside, not inside**, and that is asserted by value: this test would have
    caught a build that satisfied the requirement by adding a field to
    ``SolverConfig`` or to ``RunProvenance``, either of which moves a digest
    that §7 pins.
    """
    config = SolverConfig(max_output_tokens=16_000, reasoning_effort="xhigh")
    built = build_generation_task(instances, mcp_url=MCP_URL, solver_config=config)
    metadata = built.metadata or {}

    assert metadata["reasoning_effort"] == "xhigh"
    assert metadata["solver_sha256"] == config.digest

    # the two carriers, and the point of having both: the digest still covers
    # effort — so it still refuses — while the lifted field is what a layer can
    # hold explicitly
    at_high = config.model_copy(update={"reasoning_effort": "high"})
    assert at_high.digest != config.digest
    other = build_generation_task(instances, mcp_url=MCP_URL, solver_config=at_high)
    assert (other.metadata or {})["reasoning_effort"] == "high"

    # nothing already recorded moved: neither digest §7 pins gains a field
    assert built.version == instances.version
    assert set(metadata["provenance"]["solver"]) == {
        "turn_cap",
        "reasoning_effort",
        "max_output_tokens",
        "token_limit",
        "solver_name",
    }


def test_no_solver_config_leaves_the_effort_key_absent_rather_than_none(instances):
    """Absent, never ``None``: an unset effort is the value that silently empties
    the reasoning channel (§6.6), so a task with no solver record must not read
    as one that ran at no effort."""
    from genomics_harness.provenance import RunProvenance

    metadata = RunProvenance.build(instances, solver=None).to_task_metadata()
    assert "reasoning_effort" not in metadata
    assert "solver_sha256" not in metadata


def test_a_cap_change_changes_the_digest(instances):
    """§7: change what the model is permitted to generate and trajectories change
    with nothing else changing. The digest is what makes that visible.

    Was asserted on ``floor``, a per-turn-cap constant that no longer exists.
    """
    other = MOCK_SOLVER_CONFIG.model_copy(update={"max_output_tokens": 999})
    assert other.digest != MOCK_SOLVER_CONFIG.digest
    assert _task(instances).metadata["solver_sha256"] != other.digest


def test_the_effort_on_the_task_config_is_the_one_in_the_digest(instances):
    """Replaces `test_editing_the_continuation_message_changes_the_digest`.

    `reasoning_effort` was a `build_generation_task` parameter separate from the
    `SolverConfig`, so the value the run used and the value the digest recorded
    could differ — on the one axis where a wrong value produces no error at all
    (§6.6). There is now one source for both.
    """
    config = SolverConfig(max_output_tokens=16_000, reasoning_effort="xhigh")
    built = build_generation_task(instances, mcp_url=MCP_URL, solver_config=config)
    assert built.config.reasoning_effort == "xhigh"
    assert built.metadata["solver_sha256"] == config.digest
    assert (
        config.digest != config.model_copy(update={"reasoning_effort": "high"}).digest
    )


def test_the_registered_task_resolves_its_set_and_stays_generation_only(
    fixture_modules, monkeypatch
):
    """What this file owns about the registered task: it is a generation task,
    and its ``version`` is the digest of whatever set it resolved.

    The set comes from a **fixture** config through the environment override,
    never from the shipped one: `data/instance-sets.toml` is operator-maintained
    and ships empty in a harness-only release (§16), so a test naming a set in it
    fails on ordinary use. The composition layer's own surface — which arguments
    the `@task` captures, and what each failure raises — is asserted in
    `test_composition.py`, over the same fixtures.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    resolved = resolve_set("one", announce=False)

    built = genomics_generation(
        mcp_url=MCP_URL,
        set="one",
        repeats=2,
        instance_set="named",
        reasoning_effort=None,
    )

    assert built.version == resolved.version
    assert (built.metadata or {})["instances_sha256"] == resolved.digest
    assert_generation_only(built)


# -- the header, from a real run ------------------------------------------


def test_the_header_carries_no_rubric_and_no_ground_truth(instances, tmp_path):
    """The property §12 Principle 4 depends on, checked against the bytes.

    Ground truth lives in `Sample.metadata` and reaches `EvalSample.metadata`,
    which is what a judge needs — the claim here is narrower and is about
    ``header.json``: the task's own arguments and its scorer options.
    """
    log = inspect_eval(
        _task(instances, repeats=1),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "header"),
    )[0]
    stored = read_log(log.location)

    assert stored.eval.task_args == {}
    assert stored.eval.task_args_passed == {}
    for scorer in stored.eval.scorers or []:
        assert scorer.options == {}

    # the derivation string that would appear if ground truth had leaked
    header = json.dumps(stored.eval.model_dump(mode="json"))
    assert "derivation" not in header
    assert "ground_truth" not in header


def test_passing_the_instance_set_as_a_task_argument_would_leak_it(instances, tmp_path):
    """The hazard the module docstring names, demonstrated against a written log.

    `extract_named_params(task_type, True, ...)` (`_eval/registry.py:158-160`)
    captures every parameter, defaults included, and `EvalSpec.task_args` is
    written from it (`_eval/task/log.py:245-246`). This test runs the leaky
    shape and reads the ground truth back out of the header. It is why
    :func:`genomics_generation` takes a name.
    """
    from inspect_ai import Task, task

    from genomics_harness import build_dataset

    @task
    def leaky(instance_set):
        return Task(name="leaky", dataset=build_dataset(instance_set))

    log = inspect_eval(
        leaky(instances),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "leaky"),
    )[0]
    header = json.dumps(read_log(log.location).eval.model_dump(mode="json"))
    assert "derivation" in header  # ground truth, in header.json


# -- the live run ---------------------------------------------------------


def _live_outputs(input, tools, tool_choice, config):
    """One tool call, then an answer. Turn number comes from the conversation.

    Never from a shared counter: epochs run concurrently and a counter
    interleaves across them (`CLAUDE.md` §6).
    """
    turn = sum(1 for m in input if m.role == "assistant")
    if turn == 0:
        return ModelOutput.for_tool_call(
            model="mockllm/model", tool_name="listSuperpopulations", tool_arguments={}
        )
    return ModelOutput.from_content(
        model="mockllm/model",
        content="chr7:99,000-101,000",
        stop_reason="stop",
    )


@pytest.mark.live_mcp
def test_the_task_runs_against_the_live_mcp_server(instances, tmp_path):
    """One end-to-end pass: the tool surface resolved, one real call, one answer.

    `mockllm` drives the turns, so this establishes the wiring — `mcp_connection`
    held open across the loop, the `ToolSource` resolved once into `state.tools`,
    a real MCP result reaching `execute_tools` and the transcript — and nothing
    about how a model behaves.

    The model is the **string** ``"mockllm/model"``, with the scripted outputs
    supplied through ``model_args`` as a *callable*. Passing
    ``get_model(custom_outputs=[...])`` would put `ModelOutput` objects into
    ``model_args``, each with a fresh message id, which makes a task's `eval_set`
    identity differ on every construction (§6.6). A callable does not survive the
    log at all — ``model_args`` reads back as ``{"custom_outputs": None}``.
    """
    log = inspect_eval(
        _task(instances, repeats=1),
        model="mockllm/model",
        model_args={"custom_outputs": _live_outputs},
        display="none",
        log_dir=str(tmp_path / "live"),
    )[0]

    assert log.status == "success", log.error
    stored = read_log(log.location)
    for sample in stored.samples or []:
        tool_events = [e for e in sample.events if e.event == "tool"]
        assert tool_events, "no tool call reached the server"
        assert tool_events[0].function == "listSuperpopulations"
        assert tool_events[0].error is None, tool_events[0].error
        assert "AFR" in str(tool_events[0].result)

        derived = trajectory_status(sample)
        assert derived.completeness is Completeness.CLEAN
        carried = carried_record(sample.store)
        assert carried is not None
        assert carried.stop_cause is StopCause.MODEL_STOPPED
        assert reconcile(derived, carried)[1] == []

        # one seeded prompt, verbatim, and nothing else the harness wrote
        seeded = [m for m in sample.messages if m.source == "input"]
        assert len(seeded) == 1
