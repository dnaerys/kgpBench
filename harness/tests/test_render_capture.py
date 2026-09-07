"""The renderer-capture scorer, and the equivalence it exists to establish.

Three properties, in order of what they protect:

1. **The digest is equal inline and on re-score.** That is the whole point — it
   is the evidence that the deferred path presents a scorer the same trajectory
   the inline path did (`design/specification.md` §6.3). Per-section digests are
   compared too, so a future divergence names its channel.
2. **A string-valued `Score` coexists with dict-valued judges.** §14.15 recorded
   this as inferred rather than established.
3. **Its header entry carries an empty options dict**, which is what keeps a
   generation log rubric-free (§12 Principle 4, `foundations.md` §2.1).

One test here is not about `render_capture` at all: `score()` does not write to
disk, and the publication split depends on it. It lives here because it needs the
same fixture.

`mockllm/model` throughout; no network, no keys.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from inspect_ai import Epochs, Task, eval, score
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, execute_tools, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef

from genomics_harness.render_capture import render_capture
from genomics_harness.renderer import SECTIONS, render_sections
from genomics_harness.version import RENDERER_VERSION

CHECKS = ["A1", "B1", "D1"]


# -- a trajectory with something in every channel --------------------------


async def lookup_gene(symbol: str) -> str:
    """Look up a gene.

    Args:
        symbol: Gene symbol.
    """
    # long enough that the result would be attachment-pooled if tool results were
    # pooled at all — they are not (`log/_condense.py:436-444`), which is part of
    # what makes the two paths agree
    return f"gene:{symbol} chr7:100000-100400 " + ("detail " * 40)


GENE_TOOL = ToolDef(lookup_gene).as_tool()


def tool_calling_model():
    """One tool-calling turn carrying reasoning, then a final answer.

    The reasoning block has the *mock's* shape, not a provider's — a real
    Anthropic block puts the signature blob in `.reasoning` and the readable text
    in `.summary` (`foundations.md` §4.1). The renderer reads `.text`, which is
    correct for both, and only a source-level rule can hold that (invariant 4).
    """

    def outputs(input, tools, tool_choice, config):
        turn = sum(1 for message in input if message.role == "assistant")
        if turn == 0:
            output = ModelOutput.for_tool_call(
                "mockllm/model",
                tool_name="lookup_gene",
                tool_arguments={"symbol": "GENE1"},
            )
            output.choices[0].message.content = [
                ContentReasoning(
                    reasoning="SIG-BLOB",
                    summary="Look the gene up before answering.",
                    redacted=True,
                ),
                ContentText(text="Looking that up."),
            ]
            return output
        return ModelOutput.from_content(
            "mockllm/model", content="chr7:100000.", stop_reason="stop"
        )

    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


@solver
def tool_using_agent() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        state.tools = [GENE_TOOL]
        for _ in range(4):
            output = await model.generate(input=state.messages, tools=state.tools)
            state.messages.append(output.message)
            state.output = output
            if output.message.tool_calls:
                messages, _ = await execute_tools(state.messages, state.tools)
                state.messages.extend(messages)
                continue
            break
        return state

    return solve


@scorer(name="judge_one", metrics={check: [mean()] for check in CHECKS})
def judge_one(catalogue: list[str]) -> Scorer:
    async def score_(state: TaskState, target: Target) -> Score:
        return Score(value={check: 1.0 for check in catalogue}, explanation="one")

    return score_


@scorer(name="judge_two", metrics={check: [mean()] for check in CHECKS})
def judge_two(catalogue: list[str], rubric_text: str = "") -> Scorer:
    async def score_(state: TaskState, target: Target) -> Score:
        return Score(value={check: 0.5 for check in catalogue}, explanation="two")

    return score_


DATASET = MemoryDataset(
    [
        Sample(
            id="a1b2c3d4e5f60718", input="Investigate the cohort.", metadata={"gt": 42}
        ),
        Sample(id="0f1e2d3c4b5a6978", input="A second question.", metadata={"gt": 7}),
    ]
)


def _generation_task(scorers=None) -> Task:
    return Task(
        name="capture_generation",
        dataset=DATASET,
        solver=tool_using_agent(),
        scorer=[render_capture()] if scorers is None else scorers,
        epochs=Epochs(2, []),
        version="v-capture-test",
    )


@pytest.fixture
def generation_log(tmp_path: Path):
    """One generation run carrying the capture scorer and nothing else."""
    logs = eval(
        _generation_task(),
        model=tool_calling_model(),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert logs[0].status == "success"
    return read_eval_log(logs[0].location, resolve_attachments=True)


# -- the equivalence -------------------------------------------------------


@pytest.mark.parametrize("action", ["overwrite", "append"])
def test_the_capture_digest_is_identical_inline_and_on_rescore(generation_log, action):
    """Both actions, because `append` is the one the design uses (§13.3).

    Under `append` the re-scored entry is uniqued by `unique_scorer_name`
    (`scorer/_scorer.py:262-269`) to `render_capture1`, so the generation run's
    own digest survives beside it — which is the property that lets a scoring log
    carry the verdicts and the capture together.
    """
    inline = {
        (sample.id, sample.epoch): sample.scores["render_capture"]
        for sample in generation_log.samples or []
    }
    assert len(inline) == 4

    rescored = score(generation_log, [render_capture()], action=action, display="none")
    name = "render_capture" if action == "overwrite" else "render_capture1"

    for sample in rescored.samples or []:
        before = inline[(sample.id, sample.epoch)]
        after = sample.scores[name]
        assert after.value == before.value, (
            f"whole-rendering digest differs for {sample.id}/{sample.epoch}"
        )
        assert (
            after.metadata["section_digests"] == before.metadata["section_digests"]
        ), "a trace channel differs between the inline and deferred paths"
        assert after.metadata["section_lengths"] == before.metadata["section_lengths"]
        assert after.metadata["rendering_length"] == before.metadata["rendering_length"]
        assert after.metadata["renderer_version"] == RENDERER_VERSION


def test_every_section_is_populated_so_the_comparison_is_not_vacuous(generation_log):
    """Equal digests over empty sections would prove nothing."""
    for sample in generation_log.samples or []:
        lengths = sample.scores["render_capture"].metadata["section_lengths"]
        assert set(lengths) == set(SECTIONS)
        assert all(length > 0 for length in lengths.values()), lengths


def test_two_trajectories_do_not_share_a_digest(generation_log):
    digests = {
        sample.scores["render_capture"].value for sample in generation_log.samples or []
    }
    # two instances, two epochs each; the epochs are identical trajectories, so
    # two distinct digests is the correct count
    assert len(digests) == 2


# -- coexistence with dict-valued judges (§14.15) --------------------------


def test_a_string_score_coexists_with_dict_scores_inline(tmp_path: Path):
    log = eval(
        _generation_task(
            [render_capture(), judge_one(CHECKS), judge_two(CHECKS, rubric_text="r")]
        ),
        model=tool_calling_model(),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]

    assert log.status == "success"
    assert log.results is not None
    value_types = {
        name: type(score_.value).__name__
        for name, score_ in log.samples[0].scores.items()
    }
    assert value_types == {
        "render_capture": "str",
        "judge_one": "dict",
        "judge_two": "dict",
    }
    # one EvalScore per judge *key*, one for the capture scorer
    rows = {(row.scorer, row.name) for row in log.results.scores}
    assert ("render_capture", "render_capture") in rows
    assert {("judge_one", check) for check in CHECKS} <= rows
    assert {("judge_two", check) for check in CHECKS} <= rows


def test_a_string_score_coexists_with_dict_scores_on_rescore(generation_log):
    rescored = score(
        generation_log,
        [judge_one(CHECKS), judge_two(CHECKS, rubric_text="r"), render_capture()],
        action="append",
        display="none",
    )

    assert rescored.results is not None
    for sample in rescored.samples or []:
        assert isinstance(sample.scores["judge_one"].value, dict)
        assert isinstance(sample.scores["judge_two"].value, dict)
        # the generation run's capture keeps the plain name; the re-scored one is
        # uniqued by `unique_scorer_name` (`scorer/_scorer.py:262-269`)
        assert isinstance(sample.scores["render_capture"].value, str)
        assert isinstance(sample.scores["render_capture1"].value, str)
        assert (
            sample.scores["render_capture1"].value
            == sample.scores["render_capture"].value
        )


# -- what keeps a generation log rubric-free -------------------------------


def test_the_capture_scorer_records_no_options(generation_log):
    scorers = generation_log.eval.scorers or []
    assert [entry.name for entry in scorers] == ["render_capture"]
    assert scorers[0].options == {}


def test_a_judge_argument_would_have_reached_the_header(generation_log):
    """The positive control: the mechanism that captures options does fire."""
    rescored = score(
        generation_log,
        [judge_two(CHECKS, rubric_text="THRESHOLDS SEPARATING PERFORMED FROM PARTIAL")],
        action="append",
        display="none",
    )
    options = {entry.name: entry.options for entry in rescored.eval.scorers or []}
    assert options["render_capture"] == {}
    assert (
        options["judge_two"]["rubric_text"]
        == "THRESHOLDS SEPARATING PERFORMED FROM PARTIAL"
    )


# -- where deferred scoring writes (§14.13) --------------------------------


def test_score_does_not_write_to_disk(tmp_path: Path):
    """`score()` returns a log; writing it is the caller's decision.

    The trajectory-log / scoring-log split depends on this: judging a generation
    log leaves it byte-identical, so the rubric-free artifact stays rubric-free.
    """
    log_dir = tmp_path / "logs"
    logs = eval(
        _generation_task(),
        model=tool_calling_model(),
        log_dir=str(log_dir),
        display="none",
    )
    path = Path(logs[0].location)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    files_before = sorted(p.name for p in log_dir.iterdir())

    log = read_eval_log(str(path), resolve_attachments=True)
    scored = score(
        log,
        [judge_two(CHECKS, rubric_text="SECRET")],
        action="append",
        display="none",
    )

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in log_dir.iterdir()) == files_before
    # and the rubric is only in the returned object
    on_disk = read_eval_log(str(path), resolve_attachments=True)
    assert [entry.name for entry in on_disk.eval.scorers or []] == ["render_capture"]
    assert "judge_two" in {entry.name for entry in scored.eval.scorers or []}


def test_the_scored_log_still_points_at_the_generation_file(tmp_path: Path):
    """`write_eval_log(scored)` with no location overwrites the generation log.

    `_file.py:283-286` falls back to `log.location`, and `score()` does not change
    it. The scoring log must be written to an explicit path.
    """
    log_dir = tmp_path / "logs"
    logs = eval(
        _generation_task(),
        model=tool_calling_model(),
        log_dir=str(log_dir),
        display="none",
    )
    path = Path(logs[0].location)
    log = read_eval_log(str(path), resolve_attachments=True)
    scored = score(
        log, [judge_two(CHECKS, rubric_text="SECRET")], action="append", display="none"
    )

    assert Path(scored.location) == path

    scoring_path = log_dir / "scoring-run.eval"
    write_eval_log(scored, str(scoring_path))

    generation = read_eval_log(str(path), resolve_attachments=True)
    scoring = read_eval_log(str(scoring_path), resolve_attachments=True)
    assert [entry.name for entry in generation.eval.scorers or []] == ["render_capture"]
    assert "judge_two" in {entry.name for entry in scoring.eval.scorers or []}


# -- rendering properties --------------------------------------------------


def test_rendering_is_stable_across_repeated_reads(generation_log):
    """No timestamps, no ids, no iteration-order dependence."""
    from inspect_ai.model import ModelName
    from inspect_ai.solver import TaskState

    sample = (generation_log.samples or [])[0]
    state = TaskState(
        model=ModelName(generation_log.eval.model),
        sample_id=sample.id,
        epoch=sample.epoch,
        input=sample.input,
        messages=sample.messages,
    )
    first = render_sections(state, sample.events)
    second = render_sections(state, sample.events)
    assert first == second
    assert set(first) == set(SECTIONS)
