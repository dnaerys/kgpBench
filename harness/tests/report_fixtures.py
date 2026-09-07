"""Stored evaluations for the report layer. `mockllm/model` throughout.

The report layer's unit is a group of evaluations, so its fixtures have to be
evaluations — several of them, on disk, with their sidecars. Everything here
builds on :mod:`mock_judging`, which owns the shipped-solver generation and the
canned judge; what this module adds is the *variation across* evaluations that
§5's axis table is about:

* **differing instances** — :data:`SECOND_INSTANCES` carries an instance with its
  own ``opaque_id``, so two evaluations combine across an axis that may vary;
* **differing catalogues** — :func:`redefined_catalogue` moves one shared check's
  ``text`` (incombinable) and :func:`extended_catalogue` adds a check
  (combinable, and the added check is dropped from the intersection). The two are
  separate functions because they are the two halves §5 needs kept apart: *a
  check was added* against *a check's definition changed*;
* **varying judge counts, including one** — :func:`evaluation` takes the judge
  names to run, so a single-judge evaluation is a first-class fixture rather than
  a degenerate case. Under v34 a lone verdict counts, and nothing here may assume
  a panel of three;
* **a differing model under test** — :func:`evaluation` takes the model string,
  which reaches ``EvalSpec.model`` on both logs.

**What these fixtures establish, and what they cannot.** They are stored Inspect
logs produced by the shipped solver, the shipped judges and the shipped driver,
so they establish the *plumbing*: that a uuid survives to the scoring log, that a
sidecar lands beside it, that the header carries the model under test and the
wire-form catalogue, and that the layer's readers find all of it where §5 says it
is. They cannot establish *semantics* — a mock confirms plumbing, never semantics
(§6.5). No judge here read a real trajectory, so nothing about verdict quality,
about how often real judges split, or about whether a check is well written is
evidence from this module. The verdicts are dictated by the test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mock_judging as mj
from inspect_ai import eval as inspect_eval
from inspect_ai.model import Model, ModelOutput, get_model

from genomics_harness import (
    Catalogue,
    Check,
    CheckGroundTruth,
    Instance,
    InstanceSet,
    Requirement,
    TaskCheck,
    Tolerance,
    read_log,
)
from genomics_harness.judge import JUDGE_NAMES
from genomics_harness.judge_prompt import JUDGE_PROMPT_VERSION
from genomics_harness.provenance import JudgeConfig
from genomics_harness.report import (
    EvaluationLogs,
    ScoringLog,
    load_evaluation,
    load_scoring_logs,
)
from genomics_harness.scoring import ScoringRun, build_judges, run_judges

__all__ = [
    "FULL_VERDICTS",
    "SECOND_INSTANCES",
    "Evaluation",
    "evaluation",
    "extended_catalogue",
    "paired",
    "redefined_catalogue",
    "scoring_logs_of",
]

FULL_VERDICTS: dict[str, str] = {
    "B6": "performed",
    "B5": "partial",
    "B1": "not_performed",
    "D4": "performed",
    "A1": "performed",
    "C9": "performed",
}
"""One reply covering every check any fixture instance reaches.

``C9`` is here for :func:`extended_catalogue`; a judge naming a check the
catalogue in force does not carry is recorded under ``checks_not_in_catalogue``
and does not become a verdict, so carrying it is harmless on the runs that do not
use it.
"""


# -- a second instance set, so the instances axis genuinely varies ----------


SECOND_PROMPT = (
    "SECOND. Establish what this cohort can and cannot show about pathogenicity."
)

SECOND_INSTANCES = InstanceSet(
    name="report-test-instances-second",
    dataset_snapshot="onekgpd-test-snapshot",
    instances=[
        Instance(
            id="5a6b7c8d9e0f1234",
            prompt=SECOND_PROMPT,
            checks=[
                # the same checks as the first set's KIN instance, at the
                # opposite requirement weight. This is the shape of a real pair
                # of instances over one substrate:
                # one ground truth, two questions, and the required/enriching
                # split differing between them — which is what makes the two
                # matrices per check something a test can actually observe.
                TaskCheck(
                    check_id="B6",
                    requirement=Requirement.ENRICHING,
                    performed="an expected count is stated and used to qualify the zero",
                    partial="an expectation is stated but never reaches the conclusion",
                    not_performed="the zero is read as informative with no expectation",
                ),
                TaskCheck(
                    check_id="B5",
                    requirement=Requirement.REQUIRED,
                    performed="the denominator is stated at the EAS level",
                    partial="the clustering is named but the computation stays cohort-wide",
                    not_performed="population structure does not reach the calculation",
                ),
            ],
            ground_truth=[
                CheckGroundTruth(
                    check_id="B6",
                    expected=0.51,
                    tolerance=(Tolerance("the expected count", 0.05),),
                    derivation="EAS n=585, carrier counts 22 and 15",
                )
            ],
            attributes={"gene": "GENE1"},
        )
    ],
)
"""A second instance set, sharing checks with :data:`mock_judging.INSTANCES` at
the opposite requirement weight.

Its one instance has its own ``opaque_id``, so it is a different instance with no
further mechanism: instance identity is a content digest (§5).
"""


# -- catalogue variants -----------------------------------------------------


def redefined_catalogue(check_id: str = "B6") -> Catalogue:
    """The fixture catalogue with one shared check's ``text`` moved.

    The incombinable case: ``text`` is what reaches the judge prompt, so a verdict
    under one definition is not a verdict under the other.
    """
    checks = [
        check.model_copy(update={"text": check.text + " (redefined for this fixture)"})
        if check.id == check_id
        else check
        for check in mj.CATALOGUE.checks
    ]
    return Catalogue(release=mj.CATALOGUE.release, checks=checks)


def extended_catalogue() -> Catalogue:
    """The fixture catalogue with one check **added**.

    The combinable case: a check that did not exist in the older evaluation is a
    missing observation, the same shape as a check an instance did not make
    reachable — reported over the intersection and named as dropped, never a
    reason to refuse.
    """
    added = Check(
        id="C9",
        group="C",
        title="An added check",
        text="the trajectory performed C9",
        ground_truth_class=mj.CATALOGUE.checks[0].ground_truth_class,
        contamination_class=mj.CATALOGUE.checks[0].contamination_class,
        construction=mj.CATALOGUE.checks[0].construction,
        genotypes=False,
        talos_anchor=None,
        in_scope=True,
        refines=None,
    )
    return Catalogue(release=mj.CATALOGUE.release, checks=[*mj.CATALOGUE.checks, added])


# -- one stored evaluation --------------------------------------------------


class Evaluation:
    """One stored evaluation: a generation log plus its judges' scoring logs.

    Both halves are the report layer's input since v35, so both are addressable
    here: :attr:`generation` is the log the trajectories were produced in, and
    :attr:`locations` are the scoring logs taken from it.
    """

    def __init__(self, run: ScoringRun, model: str) -> None:
        self.run = run
        self.model = model

    @property
    def generation(self) -> str:
        """The generation log these scoring logs were taken from."""
        return self.run.generation_location

    @property
    def locations(self) -> list[str]:
        """Every scoring log this evaluation produced."""
        return list(self.run.scoring_logs.values())

    def loaded(self) -> list[ScoringLog]:
        return load_scoring_logs(self.locations)

    def pair(self) -> EvaluationLogs:
        """This evaluation as the report layer's input: the pair."""
        return load_evaluation(self.generation, self.locations)


def _generation_model(model: str, answer: str) -> Model:
    """`mock_judging`'s scripted generation model under a chosen model string.

    The model string is the point: it reaches ``EvalSpec.model`` on the
    generation log and is preserved onto every scoring log taken from it, which
    is where the report layer reads the model under test.
    """
    scripts = mj._script(answer, kinship_call=True)

    def outputs(input: Any, tools: Any, tool_choice: Any, config: Any) -> ModelOutput:
        prompt = next(message.text for message in input if message.role == "user")
        key = prompt.split(" ", 1)[0]
        script = scripts.get(key) or scripts["KIN."]
        turn = sum(1 for message in input if message.role == "assistant")
        return script[min(turn, len(script) - 1)]

    return get_model(model, custom_outputs=outputs, memoize=False)


def evaluation(
    tmp_path: Path,
    name: str,
    *,
    model: str = "mockllm/model-under-test",
    instances: InstanceSet | None = None,
    catalogue: Catalogue | None = None,
    judges: Sequence[str] = JUDGE_NAMES,
    replies: Mapping[str, Mapping[str, str] | str] | None = None,
    answer: str = "Stratified to EAS the expectation is 0.51; three variants max.",
) -> Evaluation:
    """Generate and judge one evaluation, stored under ``tmp_path / name``.

    Args:
        tmp_path: The test's temporary directory.
        name: Subdirectory, and the task name.
        model: The model under test.
        instances: Which instances to run; defaults to `mock_judging`'s set.
        catalogue: The checks the judges score against.
        judges: Which judges to run — **one is a supported case**, and the layer
            may assume no panel size.
        replies: Judge name to a verdict mapping or a raw reply string. A string
            that is not a verdict object produces a whole-reply parse failure.
        answer: The final answer text.

    Returns:
        The stored evaluation.
    """
    instance_set = instances if instances is not None else mj.INSTANCES
    checks = catalogue if catalogue is not None else mj.CATALOGUE
    running = list(judges)

    logs = inspect_eval(
        mj.generation_task(f"report-{name}", instance_set),
        model=_generation_model(model, answer),
        log_dir=str(tmp_path / name / "generation"),
        display="none",
    )
    assert logs[0].status == "success"
    assert logs[0].location
    generation = read_log(str(logs[0].location))

    chosen = replies if replies is not None else dict.fromkeys(running, FULL_VERDICTS)
    recorder = mj.JudgeRecorder()
    roles: dict[str, Any] = {
        judge: mj.judge_model(
            recorder,
            mj.verdict_reply(reply) if isinstance(reply, Mapping) else reply,
        )
        for judge, reply in chosen.items()
    }
    configs = tuple(
        JudgeConfig(
            name=judge,
            model="mockllm/model",
            reasoning_effort="xhigh",
            prompt_version=JUDGE_PROMPT_VERSION,
        )
        for judge in running
    )
    run = run_judges(
        generation,
        catalogue=checks,
        scoring_dir=tmp_path / name / "scoring",
        judges=build_judges(checks, running),
        configs=configs,
        model="mockllm/model",
        model_roles=roles,
    )
    return Evaluation(run, model)


def scoring_logs_of(*evaluations: Evaluation) -> list[str]:
    """Every scoring log across several evaluations.

    The scoring half alone. A report reads pairs (:func:`paired`); this is what
    the tests that are about the scoring side specifically still want.
    """
    return [location for e in evaluations for location in e.locations]


def paired(*evaluations: Evaluation) -> list[EvaluationLogs]:
    """Several evaluations as the report layer's input — what a report reads.

    §5's input is, per evaluation, a generation log **and** the scoring logs
    taken from it, so this is the shape every report-level test hands in.
    """
    return [e.pair() for e in evaluations]
