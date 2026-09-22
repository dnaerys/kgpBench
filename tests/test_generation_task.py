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
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from kgpbench import (
    GENERATION_TASK_PREFIX,
    REASONING_EFFORTS,
    SET_CONFIG_ENV,
    Completeness,
    CompositionError,
    ReasoningEffort,
    SolverConfig,
    StopCause,
    assert_generation_only,
    build_generation_task,
    carried_record,
    generation,
    read_log,
    reconcile,
    resolve_set,
    trajectory_status,
)
from kgpbench.eval_config import SOLVER_OWNED_OPTIONS, EvalConfigCollision
from kgpbench.provenance import DECLINED_EFFORTS

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
    built = _task(instances)
    assert built.name == f"{GENERATION_TASK_PREFIX}-test-instances"
    assert_generation_only(built)


def test_a_task_named_like_a_scoring_task_is_rejected(instances):
    """The positive control for :func:`assert_generation_only`.

    `Task.name` is read-only, so the wrong name is built rather than assigned.
    """
    from inspect_ai import Task

    from kgpbench import build_dataset, render_capture

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

    from kgpbench import build_dataset

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
    from kgpbench.provenance import RunProvenance

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


def test_the_builder_puts_no_effort_on_the_task_config(instances):
    """Replaces `test_the_effort_on_the_task_config_is_the_one_in_the_digest`,
    which replaced `test_editing_the_continuation_message_changes_the_digest`.

    `reasoning_effort` was a `build_generation_task` parameter separate from the
    `SolverConfig`, so the value the run used and the value the digest recorded
    could differ — on the one axis where a wrong value produces no error at all
    (§6.6). There is now one source for both, and since September 2026 one
    carrier: the effort is set on the solver's **per-call** config, which wins
    the merge, and the task config carries nothing at all. The task config is
    merged under the eval-level arguments, so a value left there is a value
    `--reasoning-effort` overrides while every record here keeps this one
    (§7, §14.45).

    What survives the move is the digest half, asserted against the
    configuration's own value rather than against anything the task carries.
    """
    config = SolverConfig(max_output_tokens=16_000, reasoning_effort="xhigh")
    built = build_generation_task(instances, mcp_url=MCP_URL, solver_config=config)
    assert built.config.reasoning_effort is None
    assert built.config.max_tokens is None
    assert built.metadata["solver_sha256"] == config.digest
    assert built.metadata["reasoning_effort"] == "xhigh"
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

    built = generation(
        mcp_url=MCP_URL,
        set="one",
        repeats=2,
        reasoning_effort="xhigh",
    )

    assert built.version == resolved.version
    assert (built.metadata or {})["instances_sha256"] == resolved.digest
    assert_generation_only(built)


# -- the operator's two arguments -----------------------------------------


def test_the_registered_task_refuses_an_effort_nobody_named(
    fixture_modules, monkeypatch
):
    """Both halves. Omission raises; a named effort still builds.

    ``reasoning_effort`` carried ``= "xhigh"`` until September 2026, which put
    back on the registered rung the someone-forgot case `SolverConfig` had made
    unrepresentable by requiring the field: the run went ahead at an effort
    nobody chose, and the digest recorded it as a choice. The negative half is
    what says this is a control on the *default* and not on the argument —
    restore the default and only the first assertion moves.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )

    with pytest.raises(TypeError, match="reasoning_effort"):
        generation(mcp_url=MCP_URL, set="one")  # type: ignore[call-arg]

    built = generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh")
    assert (built.metadata or {})["reasoning_effort"] == "xhigh"


def test_the_registered_task_declares_no_unset_effort(fixture_modules, monkeypatch):
    """``None`` is gone from the declared type, and ``"none"`` is still in it.

    They are different requests, and the annotation is the wrong place to
    separate them. Python ``None`` sends no ``thinking`` field at all: the model
    reasons, the summary channel comes back empty and the usage record says it
    barely thought, with nothing erroring (§6.6). The string ``"none"`` asks for
    no thinking explicitly and is a value `GenerateConfig` takes — which is why
    `ReasoningEffort` still carries it: the alias mirrors the framework's own
    type, and `JudgeConfig` is typed over the same alias.

    So what declines ``"none"`` is `SolverConfig`, one level down, where every
    route into a run passes. This test pins the annotation; the refusal has its
    own tests below.
    """
    import typing

    hints = typing.get_type_hints(generation.__wrapped__)
    assert hints["reasoning_effort"] is ReasoningEffort
    assert type(None) not in typing.get_args(hints["reasoning_effort"])
    assert "none" in typing.get_args(hints["reasoning_effort"])

    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    with pytest.raises(ValueError, match="not an effort this harness runs"):
        generation(mcp_url=MCP_URL, set="one", reasoning_effort="none")


def test_the_registered_task_refuses_a_null_effort_from_the_shell(
    fixture_modules, monkeypatch
):
    """Both halves. ``None`` raises and names what it takes; a value still runs.

    The annotation does not close this and never could: ``-T`` values are parsed
    with `yaml.safe_load` and are never compared against a task's annotations
    (`_cli/util.py:213-227`), so ``-T reasoning_effort=null`` — and the bare
    ``-T reasoning_effort=``, which YAML reads as null too — arrived here as
    Python ``None`` and went through to `SolverConfig`, which takes ``None`` on
    purpose for its own callers. On a real model that is §6.6's silent double
    falsification at full price.

    The negative half carries the work and is the larger part of this test: every
    one of the efforts this task accepts still builds a task at that effort.
    Remove the check and only the first assertion moves.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )

    with pytest.raises(ValueError, match="reasoning_effort is null") as refusal:
        generation(mcp_url=MCP_URL, set="one", reasoning_effort=None)  # type: ignore[arg-type]

    # it names what it will accept, and no longer recommends an effort the
    # harness declines — that clause was an unverified claim about one provider
    # in a string that publishes to all of them, and is false here besides
    message = str(refusal.value)
    for effort in REASONING_EFFORTS:
        assert effort in message, effort
    assert "'none'" not in message
    assert "none" not in message.split("efforts this task accepts:")[1]

    # and the negative half: every accepted value still builds, none excepted
    for effort in REASONING_EFFORTS:
        built = generation(mcp_url=MCP_URL, set="one", reasoning_effort=effort)
        assert (built.metadata or {})["reasoning_effort"] == effort


def test_the_null_effort_refusal_precedes_the_set_resolution(monkeypatch):
    """It costs not even a config read, let alone a model call.

    `resolve_set` would raise `CompositionError` on this environment — there is
    no config at the pointed-at path — so the refusal arriving as `ValueError`
    is the evidence that nothing downstream of it ran.
    """
    monkeypatch.setenv(SET_CONFIG_ENV, "/nonexistent/instance-sets.toml")

    with pytest.raises(ValueError, match="reasoning_effort is null") as refusal:
        generation(mcp_url=MCP_URL, set="one", reasoning_effort=None)  # type: ignore[arg-type]
    assert not isinstance(refusal.value, CompositionError)


def test_the_declined_effort_refusal_precedes_the_set_resolution(monkeypatch):
    """Both halves, and the second is the one that can fail wrongly.

    The declined-effort refusal is `SolverConfig`'s own, and until this round
    the configuration was constructed *inside* the `build_generation_task` call
    — where `resolve_set(set)` is a positional argument, so Python evaluated the
    set first. A declined effort therefore reached the operator only after the
    config file had been read and a set resolved, while the null refusal one
    line above cost nothing: two claims about the same argument answered in two
    different orders (§9 v76). The configuration is now built before the call.

    The environment points at a config that does not exist, so `resolve_set`
    raises `CompositionError` if it is reached at all. **Half one**: with a
    declined effort the refusal is the effort's, which is only possible if
    nothing downstream ran. **Half two**: with an accepted effort the same
    unknown set still refuses on the set — without it, a hoist that skipped
    `resolve_set` entirely, or one placed after a swallowed error, would pass
    half one and go unnoticed.

    The message is `SolverConfig`'s and this round did not touch it, which the
    `match=` pins from the outside.
    """
    monkeypatch.setenv(SET_CONFIG_ENV, "/nonexistent/instance-sets.toml")

    with pytest.raises(ValueError, match="not an effort this harness runs") as declined:
        generation(mcp_url=MCP_URL, set="no-such-set", reasoning_effort="none")
    assert not isinstance(declined.value, CompositionError)

    with pytest.raises(CompositionError, match="no instance-set config"):
        generation(mcp_url=MCP_URL, set="no-such-set", reasoning_effort="xhigh")


def test_the_accepted_efforts_are_the_type_less_a_named_exclusion(
    fixture_modules, monkeypatch
):
    """:data:`REASONING_EFFORTS` is derived, so the refusals cannot drift.

    Two lists is the failure this shape exists to prevent. A tuple written out
    in full would be a second place to edit when the framework's own literal
    grows — and worse than stale: a level the framework added would be declined
    here silently, with nobody noticing. Derived-minus-an-exclusion makes a new
    level arrive **accepted**, and leaves exactly one place where a decision to
    decline one is written down.
    """
    import ast
    import inspect
    import typing

    import kgpbench.provenance as provenance_module

    declared = set(typing.get_args(ReasoningEffort))
    assert DECLINED_EFFORTS < declared, "a declined effort the type never had"
    assert set(REASONING_EFFORTS) == declared - DECLINED_EFFORTS
    assert None not in REASONING_EFFORTS

    # **The values agreeing is not the fact.** A tuple written out in full has
    # the same six members today and differs only on a level the framework has
    # not added yet, so a value assertion passes over exactly the defect this
    # shape exists to prevent — measured, by writing that tuple and watching
    # this test stay green (§15: assert the fact, not its consequences). The
    # fact is that the tuple is *derived*, and it is read off the source.
    assigned = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(provenance_module)))
        if isinstance(node, ast.AnnAssign | ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REASONING_EFFORTS"
            for target in (
                [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            )
        )
    ]
    assert len(assigned) == 1, "REASONING_EFFORTS is assigned once, in provenance.py"
    expression = ast.dump(assigned[0].value)  # type: ignore[arg-type]
    assert "get_args" in expression, "the accepted set is not derived from the type"
    assert "DECLINED_EFFORTS" in expression, (
        "the exclusion is not named in the derivation"
    )

    # each one is a value `SolverConfig` takes, which is what "accepts" means
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    for effort in REASONING_EFFORTS:
        assert SolverConfig(reasoning_effort=effort).reasoning_effort == effort


# -- an effort the harness declines ---------------------------------------
#
# `none` suppresses reasoning, and this harness scores per reasoning step rather
# than per final answer (§1, §5), so a run at it produces counts that compute
# and a report that reconciles over an experiment nobody wanted — discovered
# after it is paid for. The refusal is on `SolverConfig` because that is what
# every route into a run constructs, and the four tests below are the routes
# rather than an argument that they are covered.


# **No literal `SolverConfig.digest` is pinned here, and that is a decision.** A
# table of every accepted effort's digest, pasted from the tree before a change,
# stood here until September 2026. It fires on every legitimate field addition —
# the digest is over `model_dump`, so a new field moves all seven at once — and
# in that time it caught no defect. What it was guarding is guarded better by the
# per-round identity fence, which measures the same values at both ends of a
# round against the tree it started on rather than against a literal one edit
# can be talked into refreshing. Re-adding a literal table here adds a second
# place to edit and no coverage.


def test_the_solver_configuration_refuses_a_declined_effort():
    """The refusal, at the one place every route passes through.

    Each declined effort raises, and the message names the condition and the
    remedy. That every *accepted* effort still constructs is the negative half
    and is asserted above, over the derived vocabulary itself.
    """
    for declined in sorted(DECLINED_EFFORTS):
        with pytest.raises(ValueError, match="not an effort this harness runs") as ref:
            SolverConfig(reasoning_effort=declined)  # type: ignore[arg-type]
        message = str(ref.value)
        assert declined in message  # it names the condition
        for effort in REASONING_EFFORTS:
            assert effort in message, effort  # and the remedy


def test_the_refusal_binds_the_two_task_builders(fixture_modules, monkeypatch):
    """Route 1 and route 2 — the registered task, and the direct builder.

    Both halves per route. The registered task reaches the refusal through the
    `SolverConfig` it builds from a ``-T`` scalar; the direct builder takes the
    configuration ready-made, so what it is protected by is that the argument
    cannot be constructed. That asymmetry is the ruling: a check on the task
    would have closed the first route and left the second open.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )

    with pytest.raises(ValueError, match="not an effort this harness runs"):
        generation(mcp_url=MCP_URL, set="one", reasoning_effort="none")
    built = generation(mcp_url=MCP_URL, set="one", reasoning_effort="minimal")
    assert (built.metadata or {})["reasoning_effort"] == "minimal"

    with pytest.raises(ValueError, match="not an effort this harness runs"):
        build_generation_task(
            resolve_set("one"),
            mcp_url=MCP_URL,
            solver_config=SolverConfig(reasoning_effort="none"),
        )
    direct = build_generation_task(
        resolve_set("one"),
        mcp_url=MCP_URL,
        solver_config=SolverConfig(reasoning_effort="minimal"),
    )
    assert (direct.metadata or {})["reasoning_effort"] == "minimal"


def test_the_refusal_binds_a_direct_solver_call():
    """Route 3 — `research_agent`, called with nothing between it and a caller.

    It takes a `SolverConfig` and has taken a required one since September 2026,
    so the declined effort is unrepresentable in its argument list. The positive
    half is that the solver still builds at an accepted one.
    """
    from kgpbench import research_agent

    with pytest.raises(ValueError, match="not an effort this harness runs"):
        research_agent(None, SolverConfig(reasoning_effort="none"))
    assert research_agent(None, SolverConfig(reasoning_effort="minimal")) is not None


@pytest.mark.live_mcp
@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_every_accepted_effort_still_runs_end_to_end(
    effort, fixture_modules, monkeypatch, tmp_path
):
    """The negative half of the control, run rather than constructed.

    Six efforts, six `mockllm` evaluations, each one a whole registered-task run
    that has to succeed and record the effort it ran at — in ``task_args``, in
    the lifted `reasoning_effort` key, and inside the solver digest. It named
    `Task.config` as a fourth carrier until September 2026, when the effort
    moved to the solver's per-call config and the task config was left empty
    (§14.45); the assertion below never read it. A refusal that caught one of
    these would be caught here rather than on a paid round.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    built = generation(
        mcp_url=MCP_URL,
        set="one",
        repeats=1,
        reasoning_effort=effort,  # type: ignore[arg-type]
        max_output_tokens=16_000,
    )
    log = inspect_eval(
        built,
        model="mockllm/model",
        model_args={"custom_outputs": _mock_answer},
        display="none",
        log_dir=str(tmp_path / f"effort-{effort}"),
    )[0]

    assert log.status == "success", log.error
    stored = read_log(log.location)
    assert stored.eval.task_args["reasoning_effort"] == effort
    assert (stored.eval.metadata or {})["reasoning_effort"] == effort
    assert (stored.eval.metadata or {})["solver_sha256"] == SolverConfig(
        reasoning_effort=effort,  # type: ignore[arg-type]
        max_output_tokens=16_000,
    ).digest


def test_repeats_defaults_to_one_epoch(fixture_modules, monkeypatch):
    """Both entry points, and both halves.

    ``repeats`` defaulted to ``3`` until September 2026 — §8's run plan, which
    says what a *round* should run rather than what a *default* should supply,
    so an operator who named nothing paid three times. One is the framework's own
    default (``--epochs``, "defaults to 1") and the only number this package can
    supply without spending on a caller's behalf. §8 is unchanged.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )

    built = generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh")
    assert built.epochs == 1

    direct = build_generation_task(
        resolve_set("one", announce=False),
        mcp_url=MCP_URL,
        solver_config=MOCK_SOLVER_CONFIG,
    )
    assert direct.epochs == 1

    # the negative half: naming a count is still what sets it, and reduction
    # stays suppressed either way, so a per-repeat row is never collapsed
    named = generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh", repeats=7)
    assert named.epochs == 7
    assert named.epochs_reducer == []
    assert built.epochs_reducer == []


def test_the_repeats_default_moves_no_digest(fixture_modules, monkeypatch):
    """A count is not a provenance axis, and changing its default moves nothing.

    Epochs are a task-level setting, never part of `SolverConfig`, so the solver
    digest, the instance digest and ``version`` are all unmoved by this round's
    change of default — asserted by value against the three a run that names its
    arguments carries.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    resolved = resolve_set("one", announce=False)
    one = generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh", repeats=1)
    three = generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh", repeats=3)

    for built in (one, three):
        assert (built.metadata or {})["solver_sha256"] == SolverConfig(
            reasoning_effort="xhigh"
        ).digest
        assert (built.metadata or {})["instances_sha256"] == resolved.digest
        assert built.version == resolved.version


def test_the_pin_is_the_operator_half_of_m_max(fixture_modules, monkeypatch):
    """Supplied, it makes a model the info DB does not carry runnable.

    `resolve_output_limit` has always preferred the config's pin to the DB and
    refused when neither supplies a value (§6.3). Until September 2026 nothing
    on this rung could set the pin, so the operator half of that precedence was
    unreachable from a shell and `mockllm/model` — absent from the DB — could
    not be run through the registered task at all.
    """
    from kgpbench.solver import OutputLimitUnknown, resolve_output_limit

    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    pinned = generation(
        mcp_url=MCP_URL,
        set="one",
        reasoning_effort="xhigh",
        max_output_tokens=16_000,
    )
    config = SolverConfig(reasoning_effort="xhigh", max_output_tokens=16_000)
    assert (pinned.metadata or {})["solver_sha256"] == config.digest

    limit = resolve_output_limit("mockllm/model", config)
    assert limit.max_output_tokens == 16_000
    assert limit.source == "config"

    # and the negative half: without it, the same model still refuses
    with pytest.raises(OutputLimitUnknown):
        resolve_output_limit("mockllm/model", SolverConfig(reasoning_effort="xhigh"))


def test_omitting_the_pin_moves_no_digest(fixture_modules, monkeypatch):
    """The new argument is inert when it is not used, asserted by value.

    ``max_output_tokens=None`` is what `SolverConfig` already defaulted to, so a
    run that omits the argument digests exactly as the same run digested before
    the argument existed. ``version`` is the instance digest and never saw it.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    resolved = resolve_set("one", announce=False)
    built = generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh")

    assert (built.metadata or {})["solver_sha256"] == SolverConfig(
        reasoning_effort="xhigh"
    ).digest
    assert built.version == resolved.version
    assert (built.metadata or {})["instances_sha256"] == resolved.digest

    # supplying it is not inert, which is what says the assertion above has work
    # to do: a run at a different cap is a different run
    other = generation(
        mcp_url=MCP_URL, set="one", reasoning_effort="xhigh", max_output_tokens=16_000
    )
    assert (other.metadata or {})["solver_sha256"] != (built.metadata or {})[
        "solver_sha256"
    ]
    assert other.version == built.version


@pytest.mark.live_mcp
def test_the_registered_task_runs_under_mockllm_with_the_pin(
    fixture_modules, monkeypatch, tmp_path
):
    """The free rung of §14.2's staircase, reachable from the registered task.

    No MCP server is contacted: ``mcp_url`` is a `ToolSource` the solver resolves
    inside `mcp_connection`, and a mock that calls no tool never reaches it —
    this test therefore asserts the *task* runs, and the live tool surface is the
    ``live_mcp`` test at the bottom of this file.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    log = inspect_eval(
        generation(
            mcp_url=MCP_URL,
            set="one",
            repeats=1,
            reasoning_effort="xhigh",
            max_output_tokens=16_000,
        ),
        model="mockllm/model",
        model_args={"custom_outputs": _mock_answer},
        display="none",
        log_dir=str(tmp_path / "registered"),
    )[0]

    assert log.status == "success", log.error
    stored = read_log(log.location)
    for sample in stored.samples or []:
        assert trajectory_status(sample).completeness is Completeness.CLEAN
        carried = carried_record(sample.store)
        assert carried is not None
        assert carried.stop_cause is StopCause.MODEL_STOPPED


@pytest.mark.live_mcp
def test_the_registered_task_args_are_the_five_scalars(
    fixture_modules, monkeypatch, tmp_path
):
    """What this round moves, measured on a written log rather than reasoned about.

    `@task` captures every parameter *including defaults* into
    `EvalSpec.task_args` (`_eval/registry.py:158-160`), so a parameter name is a
    key of a published mapping. Before September 2026 the keys were ``mcp_url``,
    ``set``, ``repeats`` and ``reasoning_effort``; ``max_output_tokens`` is the
    fifth and the one this round adds. Values are scalars: the assertion below
    would fail if an `InstanceSet` ever reached the signature.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    log = inspect_eval(
        generation(
            mcp_url=MCP_URL,
            set="one",
            repeats=1,
            reasoning_effort="xhigh",
            max_output_tokens=16_000,
        ),
        model="mockllm/model",
        model_args={"custom_outputs": _mock_answer},
        display="none",
        log_dir=str(tmp_path / "args"),
    )[0]
    stored = read_log(log.location)

    assert stored.eval.task_args == {
        "mcp_url": MCP_URL,
        "reasoning_effort": "xhigh",
        "set": "one",
        "repeats": 1,
        "max_output_tokens": 16_000,
    }
    header = json.dumps(stored.eval.model_dump(mode="json"))
    assert "derivation" not in header
    assert "ground_truth" not in header


def _mock_answer(input, tools, tool_choice, config):
    """One answer, no tool call — so nothing reaches the MCP endpoint."""
    return ModelOutput.from_content(
        model="mockllm/model", content="chr7:99,000-101,000", stop_reason="stop"
    )


# -- the header, from a real run ------------------------------------------


@pytest.mark.live_mcp
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
    :func:`generation` takes a name.
    """
    from inspect_ai import Task, task

    from kgpbench import build_dataset

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


# -- the eval-level collision, site 1: inside the `@task` body ---------------


def _with_eval_config(**fields: object):
    """Put an eval-level `GenerateConfig` in force for the duration of a test.

    The framework's own route is `init_active_model`, which `eval_resolve_tasks`
    calls with the eval-level config **before** it invokes the task function
    (`_eval/eval.py:788-794`, `:1900`). A test that wants to exercise the read
    without an `eval()` around it sets the same context variable the same way;
    `test_the_eval_level_refusal_writes_no_log_directory` is the end-to-end half
    that establishes the two are the same channel.
    """
    from inspect_ai.model._generate_config import (
        active_generate_config_context_var,
    )

    return active_generate_config_context_var.set(GenerateConfig(**fields))


def _reset_eval_config(token) -> None:
    from inspect_ai.model._generate_config import (
        active_generate_config_context_var,
    )

    active_generate_config_context_var.reset(token)


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"reasoning_effort": "low"}, ["--reasoning-effort"]),
        # the value AGREES with the task's, and is still refused: presence is
        # the trigger, and an equality rule would pass this one silently while
        # a stale INSPECT_EVAL_REASONING_EFFORT held the same value (§14.45)
        ({"reasoning_effort": "xhigh"}, ["--reasoning-effort"]),
        # currently inert on the request — the solver's per-call `max_tokens`
        # wins the merge — and refused anyway, because inertness is a property
        # of today's merge order rather than of the configuration
        ({"max_tokens": 999}, ["--max-tokens"]),
        (
            {"reasoning_effort": "xhigh", "max_tokens": 999},
            ["--reasoning-effort", "--max-tokens"],
        ),
    ],
)
def test_an_eval_level_effort_or_max_tokens_is_refused(
    fields, named, fixture_modules, monkeypatch
):
    """Presence, not disagreement, on every one of the four shapes.

    The second row is the case the ruling turns on: the eval-level value is the
    task's own, so nothing contradicts anything, and the run is refused all the
    same.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    token = _with_eval_config(**fields)
    try:
        with pytest.raises(EvalConfigCollision) as refusal:
            generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh")
    finally:
        _reset_eval_config(token)

    message = str(refusal.value)
    for flag in named:
        assert flag in message, flag


def test_no_eval_level_configuration_builds_the_task(fixture_modules, monkeypatch):
    """The negative half, and it is the ordinary invocation.

    Nothing at eval level, so the read is empty and the task builds — including
    at an effort and an M_max the operator named through ``-T``, which is the
    shape closest to the refused one and must not trip it.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    built = generation(
        mcp_url=MCP_URL, set="one", reasoning_effort="xhigh", max_output_tokens=16_000
    )
    assert (built.metadata or {})["reasoning_effort"] == "xhigh"
    assert built.config.reasoning_effort is None


def test_the_eval_level_refusal_precedes_the_set_resolution(monkeypatch):
    """It costs not even a config read, and it comes before the null-effort one.

    `resolve_set` would raise `CompositionError` on this environment — there is
    no config at the pointed-at path — and ``reasoning_effort=None`` would raise
    `ValueError`, so an `EvalConfigCollision` is the evidence that neither ran.
    An operator who passed a flag *and* a bad ``-T`` is told about the flag.
    """
    monkeypatch.setenv(SET_CONFIG_ENV, "/nonexistent/instance-sets.toml")
    token = _with_eval_config(reasoning_effort="low")
    try:
        with pytest.raises(EvalConfigCollision):
            generation(mcp_url=MCP_URL, set="one", reasoning_effort=None)  # type: ignore[arg-type]
    finally:
        _reset_eval_config(token)


def test_the_eval_level_refusal_names_the_carriers_and_not_the_operator(
    fixture_modules, monkeypatch
):
    """§16 on the message: condition, remedy, carriers — and no claim about who
    typed what.

    The same value arrives from `INSPECT_EVAL_REASONING_EFFORT` and from a
    ``--generate-config`` file on the identical channel, so a refusal saying
    *you passed this* would name a flag an operator never typed. Pinned as
    literals rather than against the constants the source interpolates, so a
    rename of either moves this assertion too (§15).
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    token = _with_eval_config(reasoning_effort="low")
    try:
        with pytest.raises(EvalConfigCollision) as refusal:
            generation(mcp_url=MCP_URL, set="one", reasoning_effort="xhigh")
    finally:
        _reset_eval_config(token)

    message = str(refusal.value)
    assert "the eval-level configuration carries" in message
    assert "reasoning_effort='low'" in message
    assert "-T reasoning_effort=" in message
    assert "--reasoning-effort" in message
    assert "INSPECT_EVAL_REASONING_EFFORT" in message
    assert "--generate-config" in message
    assert "--run-config" in message
    # no claim about the operator, and no section reference: the document does
    # not publish (§16)
    for forbidden in ("you ", "typed", "command line", "§"):
        assert forbidden not in message, forbidden


def test_the_eval_level_refusal_writes_no_log_directory(
    fixture_modules, monkeypatch, tmp_path
):
    """The footprint, measured through `eval()` rather than asserted.

    The read is inside the task body, which runs before any log directory is
    created, so the refusal leaves nothing at all on disk — the cheaper of the
    two sites this ruling uses, and the difference an operator meets.

    The model is `mockllm/model`, named here rather than left to a default: this
    test asserts a refusal, and under the mutation that removes the refusal the
    run must still reach no provider (§15).
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    log_dir = tmp_path / "never-written"
    with pytest.raises(EvalConfigCollision):
        inspect_eval(
            "kgpbench/generation",
            task_args={
                "mcp_url": MCP_URL,
                "set": "one",
                "reasoning_effort": "xhigh",
                "max_output_tokens": 16_000,
            },
            model="mockllm/model",
            model_args={"custom_outputs": _mock_answer},
            reasoning_effort="low",
            display="none",
            log_dir=str(log_dir),
        )
    assert not log_dir.exists()


def test_the_refused_scope_is_the_four_fields_solver_config_owns():
    """The scope, held against the ruling and against `SolverConfig` itself.

    Four options, named by §14.45 as literals here rather than derived: the
    framework's own spellings cannot be computed from anything on this side, and
    a fifth field acquiring an option is a change that should fail this rather
    than pass it silently. What *is* derived is the other half — every
    ``solver_field`` must be a real `SolverConfig` field, so a rename on that
    side cannot leave a dead entry behind.

    ``solver_name`` is the one `SolverConfig` field with no entry: ``--solver``
    names it and is already closed, because `research_agent` takes two required
    positional arguments and the framework's construction by registry name
    supplies neither. ``--epochs`` has no entry **here** because ``repeats`` is a
    task argument rather than a `SolverConfig` field; it is refused all the same,
    under its own rule and from its own tuple
    (:data:`~kgpbench.eval_config.EPOCH_OPTIONS`), and the two scopes are held
    apart by `test_the_refused_epoch_scope_is_the_count_and_the_reducer`.
    """
    assert [option.flag for option in SOLVER_OWNED_OPTIONS] == [
        "--reasoning-effort",
        "--max-tokens",
        "--token-limit",
        "--turn-limit",
    ]
    assert [option.solver_field for option in SOLVER_OWNED_OPTIONS] == [
        "reasoning_effort",
        "max_output_tokens",
        "token_limit",
        "turn_cap",
    ]
    for option in SOLVER_OWNED_OPTIONS:
        assert option.solver_field in SolverConfig.model_fields, option.solver_field
    assert set(SolverConfig.model_fields) - {
        option.solver_field for option in SOLVER_OWNED_OPTIONS
    } == {"solver_name"}


# -- the eval-level collision, the epoch pair -------------------------------
#
# `repeats` is a task argument rather than a `SolverConfig` field, so the four
# options above cannot cover it, and the eval-level count is never readable on
# its own at any point inside a run: the `EvalConfig` that would carry it is
# built after the task function has already returned, and the reconciliation
# that follows assigns in both directions (`_eval/run.py:252-256`). What it
# leaves is the **effective** count on `Task.epochs`, on the object the builder
# returned — so the builder keeps an `EpochClaim` holding what it declared, and
# the solver compares the two.
#
# The count is refused on **inequality** and the reducer on **presence**, and
# the difference is the whole of what these tests hold apart.


def _epoch_run(built, tmp_path, name: str, **eval_kwargs):
    """One `mockllm` run of a built task, with an eval-level epoch setting.

    The model is `mockllm/model`, named here rather than left to a default.
    These tests assert a refusal, and under the mutation that removes it the run
    must still reach no provider: every generation it would then perform is
    answered by the mock (§15).
    """
    calls: list[int] = []

    def outputs(input, tools, tool_choice, config):
        calls.append(1)
        return ModelOutput.from_content(
            model="mockllm/model", content="chr7:99,000-101,000", stop_reason="stop"
        )

    model = get_model("mockllm/model", custom_outputs=outputs, memoize=False)
    assert model.name == "model" and model.api.model_name == "model"
    log = inspect_eval(
        built,
        model=model,
        display="none",
        log_dir=str(tmp_path / name),
        **eval_kwargs,
    )[0]
    return log, calls


@pytest.mark.parametrize(
    ("repeats", "epochs"),
    [(3, 5), (3, 1), (1, 3)],
)
def test_an_eval_level_epoch_count_that_differs_is_refused(
    repeats, epochs, instances, tmp_path
):
    """Three shapes of the same disagreement: more, fewer, and more-than-one
    against the default.

    ``--epochs`` wins the sample count while ``task_args.repeats`` goes stale
    (§7), so a run of five is recorded as a run of three by the one carrier a
    report reads. The refusal lands on the solver's first statement, so the
    sample errors with **no model event at all** — asserted below rather than
    inferred from the status.
    """
    log, calls = _epoch_run(
        _task(instances, repeats=repeats),
        tmp_path,
        f"epochs-{repeats}-{epochs}",
        epochs=epochs,
    )

    assert log.status == "error", log.status
    assert calls == [], "the refusal must precede every model call"
    errors = [sample.error.message for sample in (log.samples or []) if sample.error]
    assert errors, "the refusal reaches the operator as an errored sample"
    for message in errors:
        assert f"an epoch count of {epochs}" in message, message
        assert f"this task declares {repeats}" in message, message
        assert "--epochs" in message
        assert "INSPECT_EVAL_EPOCHS" in message


@pytest.mark.live_mcp
def test_an_eval_level_epoch_count_equal_to_the_task_runs_clean(instances, tmp_path):
    """The negative half, and it is the case that looks like a contradiction.

    ``--epochs 3`` against ``repeats=3`` is two claimants naming the same
    number: nothing is stale, the run is what the operator asked for either way,
    and there is nothing to refuse. This is the one row that distinguishes
    inequality from presence — the rule the four solver-owned options use — so a
    guard written as *an eval-level count is set* fires here and this test is
    what says so.
    """
    log, calls = _epoch_run(
        _task(instances, repeats=3), tmp_path, "epochs-agree", epochs=3
    )
    assert log.status == "success", log.error
    assert len(log.samples or []) == 3 * len(instances.instances)
    assert calls, "the run reached the model"


@pytest.mark.live_mcp
def test_no_eval_level_epoch_count_runs_clean(instances, tmp_path):
    """The baseline, and the second half of the same statement.

    With no eval-level count the framework copies the task's across
    (`_eval/run.py:252-256`), so the comparison reads equal and needs no branch
    of its own. A guard written over *presence of a reconciled value* would fire
    on every ordinary run, this one included.
    """
    log, calls = _epoch_run(_task(instances, repeats=2), tmp_path, "epochs-none")
    assert log.status == "success", log.error
    assert len(log.samples or []) == 2 * len(instances.instances)
    assert calls, "the run reached the model"


def test_an_eval_level_epoch_reducer_is_refused(instances, tmp_path):
    """Presence, and with the count *agreeing* so that only the reducer can fire.

    A reducer over per-epoch scores is the one thing this harness cannot have:
    ``Epochs(repeats, [])`` exists to suppress it, and §5 keeps every repeat as
    its own row.

    It is not inert, which was measured rather than assumed. ``log.reductions``
    is empty on a generation run under an eval-level ``mean``, which reads as
    this task's declaration surviving — and it does not: the eval-level reducer
    replaces ``Task.epochs_reducer`` outright and reaches
    ``EvalSpec.config.epochs_reducer``, and the only reason nothing collapses is
    that `render_capture`'s score value is a string. Against a numeric score the
    same configuration produces a reduction.
    """
    from inspect_ai import Epochs

    log, calls = _epoch_run(
        _task(instances, repeats=3),
        tmp_path,
        "epochs-reducer",
        epochs=Epochs(3, "mean"),
    )

    assert log.status == "error", log.status
    assert calls == [], "the refusal must precede every model call"
    errors = [sample.error.message for sample in (log.samples or []) if sample.error]
    assert errors
    for message in errors:
        assert "the epoch reducer 'mean'" in message, message
        assert "--epochs-reducer" in message
        assert "INSPECT_EVAL_EPOCHS_REDUCER" in message
        # the count agreed, so nothing about a count may appear
        assert "epoch count" not in message, message


@pytest.mark.live_mcp
def test_a_suppressed_eval_level_epoch_reducer_is_indistinguishable_and_runs(
    instances, tmp_path
):
    """``--no-epochs-reducer`` is not refused, and this is the measurement that
    says why rather than an omission.

    It reaches `eval()` as ``Epochs(n, [])`` (`_cli/eval.py:1901-1908`), and
    ``[]`` is exactly what this task declares — so the reconciliation writes the
    value that was already there. Every carrier reads the same **value** under it
    as under no option at all, which is what the assertions below hold: the run
    is clean, and ``EvalSpec.config.epochs_reducer`` is identical across the two.

    What *would* distinguish it is object identity — the list written over the
    task's is a different ``[]`` — and that is exactly the rule not written.
    Identity is a framework-internal property: were `Task.__init__` ever to copy
    the list it is handed, an identity rule would refuse every ordinary run, and
    a rule that can fail that way is worse than a gap it is honest about. The
    mutation that installs the identity rule fires this test, which is where
    that argument is kept rather than in an assertion over an ``id()``.

    ``--no-epochs-reducer`` also cannot arrive without ``--epochs``, which the
    count comparison does see.
    """
    from inspect_ai import Epochs

    suppressed, calls = _epoch_run(
        _task(instances, repeats=3),
        tmp_path,
        "epochs-no-reducer",
        epochs=Epochs(3, []),
    )
    assert suppressed.status == "success", suppressed.error
    assert calls, "the run reached the model"

    plain, _ = _epoch_run(_task(instances, repeats=3), tmp_path, "epochs-plain")
    assert plain.status == "success", plain.error

    assert (
        read_log(suppressed.location).eval.config.epochs_reducer
        == read_log(plain.location).eval.config.epochs_reducer
        == []
    )


def test_both_task_builders_bind_the_epoch_claim(
    instances, fixture_modules, monkeypatch, tmp_path
):
    """The control that the guard is installed rather than merely written.

    The comparison needs a claim bound to the built task, and a builder that
    stopped binding one would leave `assert_no_epoch_collision` with nothing to
    compare and no assertion anywhere would move. Both published routes are
    exercised — the direct builder and the registered task — and each must
    refuse.
    """
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )

    direct, direct_calls = _epoch_run(
        _task(instances, repeats=2), tmp_path, "bind-direct", epochs=4
    )
    assert direct.status == "error", direct.status
    assert direct_calls == []

    registered, registered_calls = _epoch_run(
        generation(
            mcp_url=MCP_URL,
            set="one",
            repeats=2,
            reasoning_effort="xhigh",
            max_output_tokens=16_000,
        ),
        tmp_path,
        "bind-registered",
        epochs=4,
    )
    assert registered.status == "error", registered.status
    assert registered_calls == []
    for log in (direct, registered):
        assert all(
            "an epoch count of 4 where this task declares 2" in sample.error.message
            for sample in (log.samples or [])
            if sample.error
        )


def test_the_epoch_refusal_names_the_carriers_and_not_the_operator(instances, tmp_path):
    """§16 on the message: condition, remedy, carriers — and no claim about who
    typed what.

    ``INSPECT_EVAL_EPOCHS`` sets the same field as ``--epochs`` on the identical
    channel, so a refusal saying *you passed this* would name an option an
    operator never typed. Pinned as literals rather than against the constants
    the source interpolates (§15).
    """
    log, _ = _epoch_run(
        _task(instances, repeats=3), tmp_path, "epochs-message", epochs=5
    )
    message = next(
        sample.error.message for sample in (log.samples or []) if sample.error
    )

    assert "the eval-level configuration carries" in message
    assert "an epoch count of 5 where this task declares 3" in message
    assert "this task owns this setting" in message
    assert "-T repeats=" in message
    assert "--epochs" in message
    assert "INSPECT_EVAL_EPOCHS" in message
    assert "--run-config" in message
    for forbidden in ("you ", "typed", "command line", "§"):
        assert forbidden not in message, forbidden


def test_the_refused_epoch_scope_is_the_count_and_the_reducer():
    """The scope, held as literals, and the one option deliberately absent.

    The framework's own spellings cannot be computed from this side. What is
    derived is that the two tuples partition :data:`EPOCH_OPTIONS`, so an entry
    added to neither half cannot appear in the whole.

    ``--no-epochs-reducer`` is **not** here and its absence is measured rather
    than chosen: it arrives as ``Epochs(n, [])``, which is what this task
    declares, so it leaves no difference anywhere to read
    (`test_a_suppressed_eval_level_epoch_reducer_is_indistinguishable_and_runs`).
    """
    from kgpbench.eval_config import (
        EPOCH_COUNT_OPTIONS,
        EPOCH_OPTIONS,
        EPOCH_REDUCER_OPTIONS,
        SOLVER_OWNED_OPTIONS,
    )

    assert [option.flag for option in EPOCH_COUNT_OPTIONS] == ["--epochs"]
    assert [option.flag for option in EPOCH_REDUCER_OPTIONS] == ["--epochs-reducer"]
    assert EPOCH_OPTIONS == EPOCH_COUNT_OPTIONS + EPOCH_REDUCER_OPTIONS
    assert "--no-epochs-reducer" not in [option.flag for option in EPOCH_OPTIONS]

    # and the two scopes stay apart: neither names a `SolverConfig` field, which
    # is what kept `--epochs` out of the four in the first place
    assert not {option.flag for option in EPOCH_OPTIONS} & {
        option.flag for option in SOLVER_OWNED_OPTIONS
    }
