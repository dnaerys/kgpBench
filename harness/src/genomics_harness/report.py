"""The report layer — the analysis over a group of combinable evaluations, for one model.

A **report** is the unit §5 v34 names (§3, §0 v34). There is no all-instances
evaluation in this design and none is planned: instances are built, evaluated and
revised one at a time, and a report accumulates over the evaluations already paid
for. Adding an instance costs one new evaluation and a re-run of the analysis —
never a re-run of the evaluations before it.

**The input is, per evaluation, a generation log and the scoring logs taken from
it** (§0 v35, superseding v34's scoring-logs-only input). Both halves are
required: **if either is missing the report refuses with an explanation and no
report is produced**. Combinability is still checked across the sidecars and
headers together and assembly still indexes by ``uuid`` — but the pairing has to
happen first, so v34's *no evaluation object is reconstructed* is withdrawn.
:class:`EvaluationLogs` is that pairing.

**Why the input widened** (§0 v35). Two statements could not both hold under the
narrower input. Exclusion is reported per cause, and
:func:`~genomics_harness.scoring.select_judgeable` filters *before* any judge
runs — so an excluded trajectory never reaches a scoring log at all and its cause
lives in the generation log and nowhere else. And the narrower input made this
layer read almost everything through a copy: the requirement class, the instance
behind a trajectory, the solver configuration and the model under test all
originate on the generation side and reach a scoring log only because ``score()``
copies the samples and the header across. That copy is a property of the append,
not a guarantee anyone stated, and §7's subject is exactly the instrument that
reports confidently on an inherited value. The exclusion counts were the half
that failed loudly; the inherited reads are the half that would have failed
silently.

**Analysis makes no framework call, because the framework has no unit above one
`eval()`** (§0 v34, verified against ``inspect_ai==0.3.252``). ``eval_results()``
takes one call's sample scores; metrics and epoch reducers both run inside a
single call. Nothing in Inspect aggregates across calls, so there is nothing here
to adopt or to diverge from. The dataframe views are not used either: ``samples_df``
reads sample *summaries*, whose metadata is abbreviated to scalar values, while
``reachable_checks``, ``missing_verdicts`` and ``judge_parse_failure`` are all
lists or dicts and all load-bearing; and ``prepare(score_to_float(...))`` runs the
same ``value_to_float`` that returns **0.0** on our stored ``null``, which is the
coercion this layer exists to refuse. A source control in ``test_report.py``
asserts both, with its anchors asserted first so a rename fails loudly rather than
passing over a file it no longer covers.

**The stored artifact is the matrices; every printed number is derived from them
at report time.** The storage decision is firm and the metric decisions are soft:
changing an aggregation rule costs a re-run of the analysis and **no judge calls**,
which is what makes an aggregation rule cheap to revisit and a lost verdict
expensive to recover.

**Three identities, and each is read rather than declared** (§7):

* the **model under test** is ``EvalSpec.model`` on the **generation** log's own
  header — a first-hand read since v35. ``score()`` does preserve it onto every
  scoring log (verified, §0 v34), and that preserved value is kept as a
  *cross-check*: :func:`load_evaluation` refuses a pairing whose two headers
  disagree, which turns an inherited value into a checked one;
* the **trajectory** is ``EvalSample.uuid`` — globally unique per sample run, and
  identical on the generation log and the scoring log (verified, §0 v34).
  ``(sample_id, epoch)`` is **not** a key across evaluations: Inspect numbers
  epochs from 1 within each ``eval()`` call, so re-running an unmodified instance
  produces a second set of epochs 1..N under the same sample id and six
  trajectories collapse to three keys;
* the **instance** is ``Sample.id``, the instance's ``opaque_id``. The instances
  axis on the sidecar is the *set* digest and is not the per-trajectory instance
  identity.

**The requirement class is read from ``Sample.metadata["instance"]``** (§0 v34),
where the instance carries its own ``TaskCheck`` relation. It is not lifted into a
``Score.metadata`` carrier the way reachability is: reachability needs an explicit
carrier because a judge must record it *before* parsing its own reply, so that a
parse failure still says what the trajectory could reach. Requirement is a property
of the instance, is written before any judge runs, and is present whatever the
judge does.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from inspect_ai.log import EvalLog, EvalSample, list_eval_logs, read_eval_log
from inspect_ai.model import ModelUsage
from pydantic import BaseModel, ConfigDict, Field

from .catalogue import Catalogue
from .instances import Instance, Requirement
from .judge_prompt import refinement_chains
from .log_reading import (
    Abstention,
    JudgeReading,
    Resolution,
    judge_reading,
    read_log,
    resolve_cell,
)
from .provenance import (
    JudgeConfig,
    RunProvenance,
    provenance_from_metadata,
    read_run_provenance,
)
from .scoring import Selection, judge_usage, select_judgeable
from .verdicts import Outcome

__all__ = [
    "CATALOGUE_OPTION",
    "COMBINATION_AXES",
    "CatalogueComparison",
    "CheckCounts",
    "CombinationError",
    "CompletenessExclusions",
    "ChainConsistency",
    "ChainViolation",
    "ReviewItem",
    "EvaluationLogs",
    "Exclusion",
    "ExclusionError",
    "ExclusionList",
    "GenerationLog",
    "JudgeCall",
    "JudgeConfigurationError",
    "JudgeSpend",
    "LoneDissent",
    "MatrixCell",
    "MatrixRow",
    "NonObservation",
    "PairingError",
    "Report",
    "ScoringLog",
    "Survey",
    "SurveyEvaluation",
    "SurveyGroup",
    "VerdictMatrix",
    "assert_combinable",
    "assert_one_configuration_per_judge",
    "build_matrices",
    "build_report",
    "catalogue_comparison",
    "check_counts",
    "completeness_exclusions",
    "chain_consistency",
    "load_evaluation",
    "load_generation_log",
    "lone_dissents",
    "load_scoring_logs",
    "judge_spend_totals",
    "read_exclusions",
    "review_queue",
    "survey",
]

CATALOGUE_OPTION = "catalogue"
"""The scorer option a judge factory carries its wire-form catalogue in.

It is how the layer finds the judge among a scoring log's scorers, and it is why
no new carrier was needed for the catalogue axis: the exact definitions that
produced each verdict are already stored in each scoring log's header (§0 v34).
The generation task's ``render_capture`` takes no arguments, which is what keeps
a generation log rubric-free — and, here, what makes "the scorer carrying a
catalogue" an unambiguous way to name the judge.
"""


# -- what the layer reads ---------------------------------------------------


class PairingError(ValueError):
    """A generation log and its scoring logs could not be paired.

    §5's input is, per evaluation, **both** halves, and if either is missing the
    report refuses with an explanation and produces nothing. Raised for a missing
    half, for a scoring log that names a different ``eval_id`` than the generation
    log it was handed with, and for two headers that disagree on the model under
    test.
    """


class GenerationLog(BaseModel):
    """One stored generation log, and the facts a report reads **first-hand**.

    The half v34 did not have. Four things a report needs originate here and
    reach a scoring log only through ``score()``'s copy: the model under test, the
    solver configuration, the instance behind each trajectory and its requirement
    class. Reading them here makes each a first-hand read (§0 v35, §7).

    A fifth thing exists only here: an excluded trajectory never reaches a
    scoring log, so the per-cause exclusion counts are computable from this log
    and from nothing else (:func:`completeness_exclusions`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    location: str
    """Where it was read from. Named in every refusal, so an operator can act."""

    model: str
    """The model under test, from ``EvalSpec.model`` on this log's own header —
    the run that produced the trajectories, read where it was written."""

    evaluation: str
    """``EvalSpec.eval_id``. Also the pairing key: ``score()`` preserves it onto
    every scoring log taken from this one (verified, §0 v34)."""

    provenance: RunProvenance
    """The run's own record, from ``Task(metadata=)``. Carries the solver
    configuration, the instances digest, the renderer version, the framework
    version and the dataset snapshot — the axes a scoring sidecar only holds a
    copy of (`scoring.scoring_provenance` builds that copy from this record)."""

    log: EvalLog | None = None
    """The full log, or ``None`` when only the header was read (:func:`survey`)."""

    def log_or_raise(self) -> EvalLog:
        """The whole log, or a refusal naming why it is not here."""
        if self.log is None:
            raise ValueError(
                f"{self.location}: read header-only; it carries no samples. "
                "`survey` reads headers; `build_report` reads whole logs"
            )
        return self.log

    @property
    def samples(self) -> list[EvalSample]:
        return list(self.log_or_raise().samples or [])


class ScoringLog(BaseModel):
    """One stored scoring log, with the facts a report reads off it.

    Assembled by :func:`load_scoring_logs`. Since v35 this log is **one half of a
    pair** and is no longer the layer's whole input. What is read here is what
    originates here: the **judge** and the **catalogue** that produced its
    verdicts, from the header; the **judges axis**, from the sidecar; and the
    **verdicts**, from the samples. :attr:`model` is still read, but as a
    cross-check against the generation header rather than as the axis
    (:func:`load_evaluation`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    location: str
    """Where it was read from. Named in every refusal, so an operator can act."""

    model: str
    """The model under test as **this** header records it — preserved by
    ``score()`` from the generation run it judged, and **not** the judge's model
    (verified, §0 v34).

    Read as a cross-check, not as the axis. The axis is
    :attr:`GenerationLog.model`; a pairing whose two headers disagree is refused
    rather than resolved, because the copy is a property of the append and not a
    guarantee anyone stated (§0 v35, §7)."""

    evaluation: str
    """``EvalSpec.eval_id``, preserved by ``score()`` and distinct across two
    ``eval()`` calls (verified, §0 v34). Carried for reporting and for catching
    the same log supplied twice; nothing rests on it."""

    judge: str
    """The judge's score key — its registered factory name."""

    catalogue: Catalogue
    """Read out of the header, where the judge factory's wire-form argument sits."""

    provenance: RunProvenance
    """The ``.provenance.json`` sidecar beside the log."""

    log: EvalLog | None = None
    """The full log, or ``None`` when only the header was read (:func:`survey`)."""

    @property
    def samples(self) -> list[EvalSample]:
        if self.log is None:
            raise ValueError(
                f"{self.location}: read header-only; it carries no samples. "
                "`survey` reads headers; `build_report` reads whole logs"
            )
        return list(self.log.samples or [])


def _carrying_catalogue(log: EvalLog) -> list[Any]:
    """The header's scorers that carry a ``catalogue`` option.

    **A judge is found in a header by carrying a ``catalogue`` option, never by
    position and never by name** (§0 v35, §0 v36). A scoring log's header carries
    ``render_capture`` **twice** — the generation task's own scorer, preserved by
    the append, and the re-run capture the driver places beside the judge (§6.3) —
    so position and name are both ambiguous. The discriminator is unambiguous for
    the same reason a generation log is catalogue-free: ``render_capture`` takes
    no factory arguments (§4a).

    It is also how a generation log is told from a scoring log: a generation log
    has no scorer carrying a catalogue at all.
    """
    return [
        scorer
        for scorer in (log.eval.scorers or [])
        if CATALOGUE_OPTION in (scorer.options or {})
    ]


def _judge_of(log: EvalLog, location: str) -> tuple[str, Catalogue]:
    """The judge scorer on a scoring log's header, and its catalogue.

    Raises:
        ValueError: No scorer carries a catalogue, or more than one does. §5
            runs **one judge per `score()` call**, so a log carrying two judge
            scorers was produced by a call that let them see each other's model
            events, and its verdicts are not the independent draws a report
            treats them as.
    """
    carrying = _carrying_catalogue(log)
    if not carrying:
        names = [scorer.name for scorer in (log.eval.scorers or [])]
        raise ValueError(
            f"{location}: no scorer carries a {CATALOGUE_OPTION!r} option, so "
            f"nothing here is a judge (scorers: {names}). A generation log is "
            "not a scoring log"
        )
    if len(carrying) > 1:
        raise ValueError(
            f"{location}: {[s.name for s in carrying]} all carry a catalogue. "
            "§5 runs one judge per score() call precisely so that judges cannot "
            "see each other's model events; this log's verdicts are not "
            "independent draws"
        )
    scorer = carrying[0]
    options = scorer.options or {}
    return scorer.name, Catalogue.of(options[CATALOGUE_OPTION])


def _local_path(location: str) -> str:
    """A location the sidecar reader can resolve.

    `list_eval_logs` yields ``file://`` URIs while `read_eval_log` accepts either,
    so the two halves of a read disagree on spelling. The sidecar sits **beside**
    the log on a filesystem, so it needs the path and not the URI — and a naive
    strip loses a slash, which is how this surfaces: a sidecar reported missing at
    ``file:/tmp/...``. Only ``file://`` is unwrapped; any other scheme is left
    alone, so a remote log fails on the sidecar read with its own message rather
    than on a mangled path.
    """
    if location.startswith("file://"):
        return url2pathname(urlparse(location).path)
    return location


def load_scoring_logs(
    locations: Iterable[str | Path], *, header_only: bool = False
) -> list[ScoringLog]:
    """Read scoring logs and their sidecars.

    Args:
        locations: Scoring log paths.
        header_only: Read headers alone. What :func:`survey` wants, and never
            enough to build a matrix.

    Returns:
        One :class:`ScoringLog` each, in the order given.

    Raises:
        FileNotFoundError: A log has no provenance sidecar. Absent is never
            treated as equal — that is the whole point of the sidecar (§0 v17).
        ValueError: A log carries no judge scorer, or carries two.
    """
    loaded: list[ScoringLog] = []
    for raw in locations:
        location = _local_path(str(raw))
        log = (
            read_eval_log(location, header_only=True)
            if header_only
            else read_log(location)
        )
        judge, catalogue = _judge_of(log, location)
        loaded.append(
            ScoringLog(
                location=location,
                model=log.eval.model,
                evaluation=log.eval.eval_id,
                judge=judge,
                catalogue=catalogue,
                provenance=read_run_provenance(location),
                log=None if header_only else log,
            )
        )
    return loaded


def load_generation_log(
    location: str | Path, *, header_only: bool = False
) -> GenerationLog:
    """Read a generation log and the provenance record in its own header.

    Args:
        location: The generation log's path.
        header_only: Read the header alone. What :func:`survey` wants, and never
            enough to read a requirement class or an exclusion cause.

    Returns:
        The log and its first-hand facts.

    Raises:
        PairingError: The log carries a judge scorer — so it is a scoring log,
            not a generation log — or it carries no provenance record in its task
            metadata, in which case nothing here can name the solver or the
            instances and a report over it would compare on invented values.
    """
    path = _local_path(str(location))
    log = read_eval_log(path, header_only=True) if header_only else read_log(path)

    carrying = _carrying_catalogue(log)
    if carrying:
        raise PairingError(
            f"{path}: {[s.name for s in carrying]} carries a "
            f"{CATALOGUE_OPTION!r} option, so this is a scoring log and not a "
            "generation log. A generation log is rubric-free by construction: "
            "its only scorer is `render_capture`, which takes no arguments (§4a)"
        )

    provenance = provenance_from_metadata(log.eval.metadata)
    if provenance is None:
        raise PairingError(
            f"{path}: the generation log carries no provenance record, so the "
            "solver configuration, the instances digest and the renderer version "
            "cannot be read first-hand. Build the generation task through "
            "`task_kwargs`, which writes it"
        )

    return GenerationLog(
        location=path,
        model=log.eval.model,
        evaluation=log.eval.eval_id,
        provenance=provenance,
        log=None if header_only else log,
    )


class EvaluationLogs(BaseModel):
    """One evaluation: a generation log and the scoring logs taken from it.

    **The pairing §5 requires** (§0 v35). Whether it is a type or a local
    grouping is implementation latitude; that it happens is not, so it is a type
    here — the survey needs the same pairing over a directory and a type is what
    lets both use one rule.

    Both halves are required, and the split of what is read from which is the
    point:

    * from the **generation** log — the model under test, the solver
      configuration, the instance behind each trajectory, its requirement class,
      and the per-cause exclusion counts;
    * from the **scoring** logs — the judge, the catalogue that produced its
      verdicts, the judges axis, and the verdicts themselves.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: GenerationLog
    scoring: list[ScoringLog]

    @property
    def evaluation(self) -> str:
        """``EvalSpec.eval_id``, read from the generation log."""
        return self.generation.evaluation

    @property
    def model(self) -> str:
        """The model under test, read first-hand from the generation header."""
        return self.generation.model

    @property
    def judges(self) -> list[str]:
        return [log.judge for log in self.scoring]


def load_evaluation(
    generation: str | Path,
    scoring: Iterable[str | Path],
    *,
    header_only: bool = False,
) -> EvaluationLogs:
    """Pair one generation log with the scoring logs taken from it.

    Args:
        generation: The generation log.
        scoring: Its scoring logs — one per judge, each with its sidecar.
        header_only: Read headers alone (:func:`survey`).

    Returns:
        The paired evaluation.

    Raises:
        PairingError: There are no scoring logs; a scoring log names a different
            ``eval_id``, so it was not taken from this generation log; or the two
            headers disagree on the model under test. Every one refuses rather
            than proceeding: §5's input is both halves, and a report over a
            mismatched pair is a number nobody can check.
        FileNotFoundError: A scoring log has no provenance sidecar.
    """
    gen = load_generation_log(generation, header_only=header_only)
    scored = load_scoring_logs(scoring, header_only=header_only)

    if not scored:
        raise PairingError(
            f"{gen.location}: no scoring logs were supplied for this generation "
            "log. §5's input is a generation log **and** the scoring logs taken "
            "from it; a generation log alone carries no verdicts, so there is "
            "nothing to report"
        )

    for log in scored:
        if log.evaluation != gen.evaluation:
            raise PairingError(
                f"{log.location} names evaluation {log.evaluation!r} and "
                f"{gen.location} names {gen.evaluation!r}, so this scoring log "
                "was not taken from this generation log. `score()` preserves "
                "`eval_id`, so a disagreement is a mispaired input and not a "
                "framework quirk"
            )
        if log.model != gen.model:
            raise PairingError(
                f"{log.location} records the model under test as {log.model!r} "
                f"and {gen.location} records {gen.model!r}. `score()` preserves "
                "`EvalSpec.model`, so a disagreement means the pair is wrong — "
                "and a report is per model, which is exactly the axis that would "
                "silently absorb it"
            )

    return EvaluationLogs(generation=gen, scoring=scored)


# -- what combines ----------------------------------------------------------


COMBINATION_AXES: tuple[str, ...] = (
    "model",
    "catalogue",
    "solver",
    "renderer",
    "framework_version",
)
"""The axes that must hold for two evaluations to combine into one report.

Deliberately **not** the same tuple as
:data:`~genomics_harness.provenance.COMPARISON_AXES`, and the difference is the
point (§0 v34). Combining is not comparing:

* **instances may vary, and that is the whole operation** — so ``instances`` is
  absent here where it is present there;
* **judges are not an axis** — recorded, never required;
* **dataset snapshot is recorded and not enforced**, until OneKGPd exposes an
  identifier (§2);
* **model is identical**, an axis added at v34 that ``compare_provenance`` has no
  entry for at all, because a report is per model.

Reusing ``assert_comparable``'s ``varying=["instances"]`` escape for combination
was weighed and rejected: that flag means something specific — *an edit to an
instance that could not have altered a verdict, ruled comparability-neutral in a
named entry* (§7) — and overloading it with *these are different instances by
design* would put two unrelated meanings behind one flag.
"""


class CombinationError(ValueError):
    """Two scoring logs do not combine into one report.

    Always names the offending log and the axis. A report **never** drops the
    offender and proceeds: an analysis costs nothing to re-run with corrected
    input, so refusing loses nothing, while a silently narrowed input set is a
    number nobody can check (§0 v34).
    """


class CatalogueComparison(BaseModel):
    """Two or more catalogues, compared check by check.

    **The comparison is per check; the verdict on it is all-or-nothing** (§0 v34).
    Those are separate statements and the coarse rule needs both. Detection has to
    run check by check over the intersection of id sets, because nothing else
    distinguishes *a check was added* — combinable — from *a check's definition
    changed* — not. What is declined is per-check **combinability**: the report
    where one check draws on six trajectories and another on three. A single
    differing check refuses the whole combination.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    shared: list[str] = Field(default_factory=list)
    """Check ids every catalogue defines — what a report runs over."""

    dropped: list[str] = Field(default_factory=list)
    """Check ids some catalogue lacks. **Not an incompatibility.** Adding a check
    is the common development edit, and it leaves an older scoring log not dense
    over the newer catalogue; a check that did not exist then is a missing
    observation, the same shape as a check an instance did not make reachable.
    Reported so a report can name what it dropped."""

    redefined: list[str] = Field(default_factory=list)
    """Shared check ids whose ``text`` differs. Any one of these refuses the
    combination.

    **``text`` alone is the definition** (§0 v34): it is what reaches the judge
    prompt and what the check means, so a change to it changes what a verdict
    asserts. Every other field may move freely — ``talos_anchor`` is carried and
    never read, ``in_scope`` and ``contamination_class`` govern which checks are
    delivered rather than what a judge is asked, and ``refines`` shapes the
    prompt's nesting without altering any single check's meaning. Comparing whole
    -check digests instead would refuse a combination over a field nothing
    consumes, which is the instances-digest over-firing of §0 v28 one level down.
    """

    @property
    def combinable(self) -> bool:
        return not self.redefined


def catalogue_comparison(catalogues: Sequence[Catalogue]) -> CatalogueComparison:
    """Compare catalogues check by check over the intersection of their id sets.

    Args:
        catalogues: One per scoring log.

    Returns:
        The shared ids, the dropped ids, and the shared ids whose ``text`` moved.
    """
    if not catalogues:
        return CatalogueComparison()

    id_sets = [set(catalogue.ids) for catalogue in catalogues]
    shared = set.intersection(*id_sets)
    dropped = set.union(*id_sets) - shared

    redefined = {
        check_id
        for check_id in shared
        if len({catalogue.get(check_id).text for catalogue in catalogues}) > 1
    }
    return CatalogueComparison(
        shared=sorted(shared), dropped=sorted(dropped), redefined=sorted(redefined)
    )


def _axis_values(evaluation: EvaluationLogs) -> dict[str, Any]:
    """This evaluation's value on every combination axis except the catalogue.

    **Read from the generation log's own record** (§0 v35). The scoring sidecar
    carries the same four values, but only because
    `scoring.scoring_provenance` copies them out of this record — and §7's
    subject is the instrument that reports confidently on an inherited value.

    The catalogue is per check and has :func:`catalogue_comparison`; these four
    are scalars and compare directly.
    """
    provenance = evaluation.generation.provenance
    solver = provenance.solver
    return {
        "model": evaluation.model,
        "solver": None if solver is None else solver.digest,
        "renderer": provenance.renderer_version,
        "framework_version": provenance.framework_version,
    }


def assert_combinable(evaluations: Sequence[EvaluationLogs]) -> CatalogueComparison:
    """Refuse a report over evaluations that do not combine, naming the offender.

    The axis rule is §5's table. Instances may vary and may coincide — that is
    the operation. Judges are not an axis. The dataset snapshot is recorded and
    not enforced. Everything else must hold.

    Args:
        evaluations: The paired evaluations a report would run over.

    Returns:
        The catalogue comparison, so a caller can name what it dropped.

    Raises:
        CombinationError: Two evaluations differ on an axis that must hold, or a
            check shared between two catalogues has a different ``text``. The
            message names the offending log's location and the axis.
        ValueError: Nothing to combine, or the same scoring log supplied twice.
    """
    if not evaluations:
        raise ValueError("no evaluations to combine")

    seen: dict[tuple[str, str], str] = {}
    for evaluation in evaluations:
        for log in evaluation.scoring:
            key = (evaluation.evaluation, log.judge)
            if key in seen:
                raise ValueError(
                    f"{log.location} and {seen[key]} are the same judge "
                    f"{log.judge!r} on the same evaluation "
                    f"{evaluation.evaluation!r}. Supplying one log twice would "
                    "count one trajectory as two observations"
                )
            seen[key] = log.location

    reference = evaluations[0]
    reference_axes = _axis_values(reference)
    for evaluation in evaluations[1:]:
        axes = _axis_values(evaluation)
        for axis in COMBINATION_AXES:
            if axis == "catalogue":
                continue
            if axes[axis] != reference_axes[axis]:
                raise CombinationError(
                    f"{evaluation.generation.location} differs from "
                    f"{reference.generation.location} on the {axis!r} axis "
                    f"({axes[axis]!r} vs {reference_axes[axis]!r}); these "
                    "evaluations do not combine into one report. Evaluations of "
                    "different models are never combined — a report is per model "
                    "— and every other axis here must hold"
                )

    catalogues = [log.catalogue for e in evaluations for log in e.scoring]
    comparison = catalogue_comparison(catalogues)
    if not comparison.combinable:
        locations = [log.location for e in evaluations for log in e.scoring]
        raise CombinationError(
            f"the scoring logs disagree on the definition of "
            f"{comparison.redefined} — a check's `text` is what reaches the judge "
            "prompt, so a verdict under one definition is not a verdict under the "
            f"other. Logs: {locations}. What incombinable "
            "costs is a re-judge, not the trajectories: point score() at the "
            "stored generation logs again under one catalogue"
        )
    return comparison


# -- exclusion after review -------------------------------------------------


class ExclusionError(ValueError):
    """The exclusion list could not be read, or does not say what it must.

    A bad or missing exclusion file is a runtime error and is **not** mitigated
    (§15). Proceeding without the list would compute a number over trajectories
    review had already found invalid, and report it as though review never
    happened.
    """


class Exclusion(BaseModel):
    """One trajectory review found invalid, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uuid: str
    """``EvalSample.uuid`` — the trajectory, not the instance and not the sample
    id. An instance runs many times and only one of those runs may be invalid."""

    reason: str
    """Free text, and required. An exclusion nobody can see is a number nobody
    can check, so this reaches the report."""


class ExclusionList(BaseModel):
    """Trajectories excluded by review, read at analysis time.

    **Where review finds a trajectory invalid, it is excluded by a declared list,
    not by editing a log** (§0 v34). Two properties, both from §7. The stored logs
    are the evidence and are never mutated — no verdict is edited, no matter what
    the judges agreed. And the exclusion is **reported**, under its own named
    cause beside the completeness causes.

    A split cell goes to review and the trajectory stays: review is per cell and
    never voids the trajectory or the evaluation around it. This list is the
    narrower thing — the trajectory itself found invalid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    exclusions: list[Exclusion] = Field(default_factory=list)

    @property
    def uuids(self) -> frozenset[str]:
        return frozenset(exclusion.uuid for exclusion in self.exclusions)

    def reason_for(self, uuid: str) -> str | None:
        """Why this trajectory was excluded, or ``None`` when it was not."""
        for exclusion in self.exclusions:
            if exclusion.uuid == uuid:
                return exclusion.reason
        return None


def read_exclusions(path: Path | str) -> ExclusionList:
    """Read the declared exclusion list.

    The file is JSON: either a list of ``{"uuid": ..., "reason": ...}`` objects,
    or an object with an ``exclusions`` key holding one.

    Args:
        path: The list's location.

    Returns:
        The parsed list.

    Raises:
        ExclusionError: The file is missing, is not readable as JSON, is not one
            of the two accepted shapes, names a trajectory twice, or carries an
            entry with no reason. Every one of these raises rather than
            degrading to an empty list: silently reading no exclusions produces
            a report that looks complete and counts trajectories review rejected.
    """
    location = Path(path)
    try:
        raw = location.read_text(encoding="utf-8")
    except OSError as error:
        raise ExclusionError(
            f"{location}: the exclusion list could not be read ({error}). It is "
            "read at analysis time and a report cannot be computed without it"
        ) from error

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ExclusionError(f"{location}: not valid JSON ({error})") from error

    if isinstance(payload, dict):
        entries = payload.get("exclusions")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ExclusionError(
            f"{location}: expected a list of exclusions or an object with an "
            f"'exclusions' key, got {type(payload).__name__}"
        )
    if not isinstance(entries, list):
        raise ExclusionError(
            f"{location}: 'exclusions' must be a list, got {type(entries).__name__}"
        )

    try:
        parsed = ExclusionList(exclusions=[Exclusion(**entry) for entry in entries])
    except (TypeError, ValueError) as error:
        raise ExclusionError(f"{location}: {error}") from error

    seen = [exclusion.uuid for exclusion in parsed.exclusions]
    duplicates = sorted({uuid for uuid in seen if seen.count(uuid) > 1})
    if duplicates:
        raise ExclusionError(
            f"{location}: trajectories excluded twice: {duplicates}. Two reasons "
            "for one exclusion is two claims, and the report can only carry one"
        )
    blank = sorted(e.uuid for e in parsed.exclusions if not e.reason.strip())
    if blank:
        raise ExclusionError(
            f"{location}: exclusions with no reason: {blank}. An exclusion nobody "
            "can see is a number nobody can check"
        )
    return parsed


# -- the stored artifact: the matrices --------------------------------------


class NonObservation(str, Enum):
    """Why a trajectory contributes no verdict to a check's counts.

    Never collapsed into one another and never into a scored zero.

    **There is no ``unreachable`` member, and its absence is the rule** (§0 v37).
    A trajectory whose task did not make a check reachable carries no requirement
    weight for it and so has **no row** in either matrix; the row's absence is the
    sole carrier and a count that is structurally zero would print in every report
    line as though it were an observation. The v34 member was measured dead and
    deleted rather than left standing beside the v35 rule that replaced it.
    """

    UNOBSERVED_REACHABLE = "unobserved_reachable"
    """Reachable, and no judge voted — every would-be voter omitted it or
    parse-failed. A result about the judges."""

    EXCLUDED = "excluded"
    """Review found the trajectory invalid and a declared list says so. The
    reconciliation denominator is the judged set **minus** what was manually
    excluded, which is why this is in scope from the first implementation rather
    than retrofitted: adding it later means revising the invariant and every
    assertion over it (§0 v34)."""


class JudgeCall(BaseModel):
    """One column: one judge's one ``score()`` call over one evaluation.

    **Columns are individual judge calls, each carrying its own identifier**, so
    two judges at the same model and effort are two columns and not one — they
    are two independent draws (§0 v34). The evaluation is part of the identity
    because the same judge name recurs in every evaluation, and pooling
    ``judge_opus_a`` across evaluations would merge two independent calls into
    one column.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation: str
    """``EvalSpec.eval_id`` of the generation run this judged."""

    judge: str
    """The judge's score key — its registered factory name."""

    model: str | None = None
    """The judge's own model, from the sidecar. Recorded, never required: judges
    are not an axis."""

    reasoning_effort: str | None = None
    """The judge's effort, from the sidecar. Recorded, never required."""

    location: str = ""
    """The scoring log this column was read from."""

    @property
    def id(self) -> str:
        """The column key: evaluation and judge, which together are unique."""
        return f"{self.evaluation}:{self.judge}"


class MatrixCell(BaseModel):
    """One judge call's reading of one check on one trajectory.

    Exactly one of :attr:`verdict` and :attr:`abstention` is set. A cell that
    does not exist at all — a judge call from another evaluation, which never saw
    this trajectory — is **absent from the row**, not a cell carrying ``None``:
    the column set varies per trajectory and empty cells are expected, not
    exceptional (§0 v34).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Outcome | None = None
    abstention: Abstention | None = None
    """Why this judge cast no vote: omitted, or parse failure. Never collapsed
    into one another and never into a scored zero.

    :attr:`~genomics_harness.log_reading.Abstention.UNREACHABLE` is a third member of
    that enum and never appears **here**. It is live one layer down, where the
    per-evaluation panel is dense over the whole catalogue; a matrix is keyed by
    requirement class, so an unreachable check has no row for a cell to hang
    from (§0 v37)."""


class MatrixRow(BaseModel):
    """One trajectory's row: what every judge call that read it said.

    **Rows are trajectories**, keyed by ``EvalSample.uuid`` (§0 v34).
    ``(sample_id, epoch)`` is not a key across evaluations — Inspect numbers
    epochs from 1 within each ``eval()`` call, so re-running an unmodified
    instance produces a second set of epochs 1..N under the same sample id and
    six trajectories collapse to three keys. Had all six run in one call the
    framework would have numbered them 1..6; because they did not, the harness
    supplies the disambiguator the framework never needed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uuid: str
    instance: str
    """``Sample.id`` — the instance's ``opaque_id``, a content digest."""

    evaluation: str

    cells: dict[str, MatrixCell] = Field(default_factory=dict)
    """Keyed by :attr:`JudgeCall.id`. A missing key is an empty cell."""

    excluded_reason: str | None = None
    """Set from the declared exclusion list. The row stays in the matrix — the
    log is the evidence and is never mutated — and contributes to
    :attr:`NonObservation.EXCLUDED` rather than to a band."""

    @property
    def votes(self) -> dict[str, Outcome]:
        """Column id to the band it voted, for the columns that voted."""
        return {
            column: cell.verdict
            for column, cell in self.cells.items()
            if cell.verdict is not None
        }

    def resolution(self) -> tuple[Resolution, float | None]:
        """How this row resolves, by the one shared rule.

        Delegates to :func:`~genomics_harness.log_reading.resolve_cell` rather than
        restating it, so "a lone verdict counts" is written down once.

        ``reachable=True`` is a constant and not a lost field (§0 v37).
        :func:`build_matrices` builds a row only where the instance carries a
        requirement class for the check, and the requirement class and the
        reachable set come from the same place — the instance in
        ``Sample.metadata``. So every row in every matrix is reachable by
        construction, and the flag the row used to carry could not take its other
        value. `resolve_cell` keeps the parameter because the per-evaluation panel
        is dense over the whole catalogue and genuinely needs it.
        """
        return resolve_cell(self.votes, reachable=True)


class VerdictMatrix(BaseModel):
    """The stored artifact: one (model, check, requirement class).

    **Two matrices per (model, check), one required and one enriching, and they
    never merge** (§0 v34; the two-rates ruling of §0 v28, delivered in this
    shape). A check that is required in one instance and enriching in another
    contributes to two matrices, never to one number. `requirement` is a per-task
    weight, and pooling them mixes *failed a step the question demanded* with
    *skipped a step it left open* under one number — which binds from the first
    instance pair onwards, two instances over one substrate disagreeing on four
    of seven checks over identical ground truth.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    check_id: str
    requirement: Requirement
    columns: list[JudgeCall] = Field(default_factory=list)
    rows: list[MatrixRow] = Field(default_factory=list)

    @property
    def instances(self) -> list[str]:
        """The distinct instances contributing rows, in first-seen order.

        Per-instance rows are kept so that whether a check is comparable across
        instances can be answered empirically rather than ruled (§0 v34), and so
        that either cross-instance weighting is computable at report time.
        """
        seen: list[str] = []
        for row in self.rows:
            if row.instance not in seen:
                seen.append(row.instance)
        return seen


def _requirements(sample: EvalSample, where: str) -> dict[str, Requirement]:
    """The requirement class per check, from ``Sample.metadata["instance"]``.

    Read off a **generation** sample since v35. The instance is written there
    before any judge runs, and a scoring log holds it only because ``score()``
    copies the samples across (§0 v35, §7).
    """
    metadata = sample.metadata or {}
    try:
        instance = Instance.from_sample_metadata(metadata)
    except KeyError as error:
        raise ValueError(
            f"{where}: {error}. The requirement class is read from the sample's "
            "own instance record, which is written before any judge runs"
        ) from error
    return {check.check_id: check.requirement for check in instance.checks}


def _generation_index(
    generation: GenerationLog,
) -> dict[str, tuple[str, dict[str, Requirement]]]:
    """Every generation trajectory by ``uuid``: its instance, and its requirements.

    The join key is ``EvalSample.uuid``, which is identical on the generation log
    and on every scoring log taken from it (verified, §0 v34) — ``score_async``
    deep-copies and reassigns only ``scores`` and ``events``.
    """
    index: dict[str, tuple[str, dict[str, Requirement]]] = {}
    for sample in generation.samples:
        uuid = sample.uuid
        if not uuid:
            raise ValueError(
                f"{generation.location}: sample {sample.id!r} epoch "
                f"{sample.epoch} carries no uuid, and the uuid is the row key "
                "across evaluations"
            )
        where = f"{generation.location}: sample {sample.id!r} epoch {sample.epoch}"
        index[uuid] = (str(sample.id), _requirements(sample, where))
    return index


def build_matrices(
    evaluations: Sequence[EvaluationLogs],
    *,
    exclusions: ExclusionList | None = None,
    checks: Sequence[str] | None = None,
) -> dict[tuple[str, str, Requirement], VerdictMatrix]:
    """Assemble the matrices from a set of paired evaluations.

    Indexes by ``uuid``: two scoring logs that judged the same trajectory
    contribute two columns to one row, and two evaluations contribute rows that
    share no column. The instance and the requirement class come from the
    **generation** log, joined on that same ``uuid``.

    Args:
        evaluations: Paired evaluations, already checked with
            :func:`assert_combinable`.
        exclusions: Trajectories review found invalid. Their rows stay in the
            matrix and count as :attr:`NonObservation.EXCLUDED`.
        checks: The check ids to build over — the intersection from
            :func:`catalogue_comparison`. ``None`` computes that intersection
            here rather than taking the first log's ids, so that the
            "every row is reachable" invariant holds on a direct call too
            (§0 v37): a check present in one judge's catalogue and absent from
            another's is not in any judge's reachable set, and taking the first
            log's ids alone would build a row for it.

    Returns:
        One matrix per (model, check, requirement class).

    Raises:
        ValueError: A judge's stored score is absent, ragged or contradicts its
            own metadata; two logs disagree on what a trajectory reached; or a
            judged trajectory has no generation sample to read its instance from.
    """
    excluded = exclusions or ExclusionList()
    check_ids = (
        list(checks)
        if checks is not None
        else catalogue_comparison(
            [log.catalogue for e in evaluations for log in e.scoring]
        ).shared
    )

    columns: dict[str, JudgeCall] = {}
    # uuid -> (instance, evaluation, model); reachability and votes accumulate
    identity: dict[str, tuple[str, str, str]] = {}
    reachable_by: dict[str, frozenset[str]] = {}
    requirement_by: dict[str, dict[str, Requirement]] = {}
    cells: dict[tuple[str, str], dict[str, MatrixCell]] = {}

    for evaluation in evaluations:
        generation = _generation_index(evaluation.generation)
        for log in evaluation.scoring:
            judge_config = next(
                (j for j in log.provenance.judges if j.name == log.judge), None
            )
            column = JudgeCall(
                evaluation=evaluation.evaluation,
                judge=log.judge,
                model=None if judge_config is None else judge_config.model,
                reasoning_effort=(
                    None if judge_config is None else judge_config.reasoning_effort
                ),
                location=log.location,
            )
            columns[column.id] = column

            for sample in log.samples:
                uuid = sample.uuid
                if not uuid:
                    raise ValueError(
                        f"{log.location}: sample {sample.id!r} epoch "
                        f"{sample.epoch} carries no uuid, and the uuid is the row "
                        "key across evaluations"
                    )
                if uuid not in generation:
                    raise ValueError(
                        f"{log.location}: trajectory {uuid} was judged but has no "
                        f"sample in {evaluation.generation.location}. The "
                        "instance and the requirement class are read from the "
                        "generation log, so a judged trajectory missing from it "
                        "is a mispaired input"
                    )
                where = f"{log.location}: sample {sample.id!r} epoch {sample.epoch}"
                reading: JudgeReading = judge_reading(
                    log.judge, sample, log.catalogue, where
                )

                instance_id, requirements = generation[uuid]
                identity.setdefault(
                    uuid, (instance_id, evaluation.evaluation, evaluation.model)
                )
                requirement_by.setdefault(uuid, requirements)

                reachable = frozenset(reading.reachable)
                if uuid in reachable_by and reachable_by[uuid] != reachable:
                    raise ValueError(
                        f"{where}: the judges disagree on what trajectory {uuid} made "
                        f"reachable ({sorted(reachable)} vs "
                        f"{sorted(reachable_by[uuid])}). Reachability is a property of "
                        "the instance, so picking one silently would rate two judges "
                        "against different denominators"
                    )
                reachable_by[uuid] = reachable

                for check_id in check_ids:
                    if check_id not in reading.outcomes:
                        continue
                    if check_id not in reachable:
                        # computed and discarded before v37, as a cell carrying an
                        # `UNREACHABLE` abstention that no row could ever hold: a row
                        # exists only where the instance carries a requirement class
                        # for the check, and that is the same set `reachable` is
                        # derived from. Skipped outright now (§0 v37).
                        continue
                    if reading.parse_failed:
                        cell = MatrixCell(abstention=Abstention.PARSE_FAILURE)
                    elif check_id in reading.missing:
                        cell = MatrixCell(abstention=Abstention.OMITTED)
                    else:
                        cell = MatrixCell(verdict=reading.outcomes[check_id])
                    cells.setdefault((uuid, check_id), {})[column.id] = cell

    matrices: dict[tuple[str, str, Requirement], VerdictMatrix] = {}
    rows_by_key: dict[tuple[str, str, Requirement], list[MatrixRow]] = {}
    columns_by_key: dict[tuple[str, str, Requirement], list[JudgeCall]] = {}

    for uuid, (instance, evaluation_id, model) in identity.items():
        requirements = requirement_by[uuid]
        for check_id in check_ids:
            requirement = requirements.get(check_id)
            if requirement is None:
                # the instance does not carry this check at all, so it has no
                # requirement class and belongs in neither matrix. The check is
                # outside this trajectory's reachable set — the same fact from
                # the other side; a row in an arbitrary matrix would invent a
                # weight. **The row's absence is the sole carrier** (§0 v37).
                continue
            key = (model, check_id, requirement)
            row = MatrixRow(
                uuid=uuid,
                instance=instance,
                evaluation=evaluation_id,
                cells=cells.get((uuid, check_id), {}),
                excluded_reason=excluded.reason_for(uuid),
            )
            rows_by_key.setdefault(key, []).append(row)
            for column_id in row.cells:
                known = columns_by_key.setdefault(key, [])
                if columns[column_id] not in known:
                    known.append(columns[column_id])

    for key, rows in rows_by_key.items():
        model, check_id, requirement = key
        matrices[key] = VerdictMatrix(
            model=model,
            check_id=check_id,
            requirement=requirement,
            columns=columns_by_key.get(key, []),
            rows=rows,
        )
    return matrices


# -- what a report computes -------------------------------------------------


class CheckCounts(BaseModel):
    """One check's counts, for one model and one requirement class.

    **Counts are primary; the denominator is displayed beside them and not
    silently divided away** (§0 v34). Counts plus the trajectory count is strictly
    more information than a rate, which is that pair with the sample size
    discarded. What a report prints is this row:

        **B6 (required)** — 7 trajectories · 6 observed · 4 performed ·
        1 partial · 0 not_performed · 1 split · 1 unobserved · 0 excluded ·
        disagreement 1/3

    **The disagreement denominator prints beside the rate and is not a
    reconciliation term** (§0 v42). One split over three rows with two or more
    voters, beside six observed. Every other number on the line participates in
    the sum that proves nothing was dropped; this one sums with nothing, and
    printing it against the rate is what *carries its own denominator* means.
    Displaying `observed` alone was the rejected alternative: the multi-voter
    count appears nowhere else on the line, so the narrower rate is not
    recoverable from the wider one while the wider is recoverable from the
    narrower.

    **The row count and the observed count are two quantities and never one
    field** (§0 v37). :attr:`rows` is every row the matrix holds: it is what the
    line displays as *trajectories* and what :meth:`reconciles` is checked
    against. :attr:`observed` is resolved plus split: it is what the rates divide
    by, because a non-observation never enters a denominator. Both rules were
    already ruled and both are right; what they cannot do is share a carrier, and
    a single field serving both roles reports one of them wrong whatever it is
    named. Before v37 this class had one property named ``trajectories`` holding
    the narrower value, and the report line printed it as the trajectory count —
    so the rename is deliberately not backwards compatible: redefining a field in
    place would have changed the meaning of every existing read without moving a
    line in any reader (§7).

    The line makes its own reconciliation checkable: the bands and the split sum
    to *observed*, and *observed* plus the non-observations sums to
    *trajectories*.

    **No mean over the bands.** Averaging ``performed=1.0, partial=0.5,
    not_performed=0.0`` asserts that a partial is worth exactly half a performed,
    and nobody can defend that coefficient. Where a single sortable number is
    wanted, :attr:`performed_rate` is the defensible one — a proportion of a named
    category rather than an average over an ordinal scale, with :attr:`partial`
    standing beside it as a diagnostic.

    **A split is a count in this row, not a footnote** (§0 v34): judges split on
    borderline trajectories, so a split silently set aside computes the numbers
    over the subset that was easy to judge.

    **Non-observations never enter a denominator.** A check every judge omitted
    and a trajectory review excluded are two distinct facts, each counted as
    itself and neither of them in :attr:`observed`. A check no trajectory reached
    is a third fact and it is **not counted here at all**: it has no row, so it
    has no numbers rather than zeroes (§0 v37).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    check_id: str
    requirement: Requirement

    performed: int = 0
    partial: int = 0
    not_performed: int = 0
    split: int = 0
    """Cells whose voters disagreed. Resolves nothing, counted as itself, and in
    the review queue."""

    unobserved_reachable: int = 0
    excluded: int = 0
    """Trajectories a declared list excluded after review — its own named cause,
    beside the completeness causes."""

    multi_voter: int = 0
    """Rows two or more judge calls voted on: the **disagreement rate's own
    denominator**, and the one count on this row that is not a reconciliation
    term (§0 v42).

    It sums with nothing. Every other count here participates in the sum that
    proves nothing was dropped; this one is a subset of :attr:`observed` — the
    rows one judge read are exactly ``observed - multi_voter``, and all of them
    resolve, because a cell with one voter cannot split. An excluded row is not
    counted here for the same reason it is not in :attr:`observed`: review found
    the trajectory invalid, so it yields no observation of any kind.
    """

    instances: list[str] = Field(default_factory=list)
    """The instances contributing rows. Both cross-instance weightings are
    computable from the stored counts, and neither is baked in: weighting is a
    report-time choice and is deliberately not ruled (§0 v34)."""

    judge_calls: int = 0
    """How many judge-call columns read this check. Carried beside the counts so
    that a number resting on one judge for half its cells says so."""

    @property
    def observed(self) -> int:
        """The **rate denominator**: every trajectory that yielded an observation.

        Resolved plus split, and deliberately **not** the row count — a
        non-observation never enters a denominator. What the line displays as the
        trajectory count is :attr:`rows` (§0 v37).
        """
        return self.performed + self.partial + self.not_performed + self.split

    @property
    def non_observations(self) -> int:
        return self.unobserved_reachable + self.excluded

    @property
    def rows(self) -> int:
        """The **displayed count**: every row the matrix holds, observed or not.

        What :meth:`reconciles` is checked against, and what the report line
        prints as *trajectories* (§0 v37).
        """
        return self.observed + self.non_observations

    @property
    def performed_rate(self) -> float | None:
        """``performed / observed``, or ``None`` on an empty denominator.

        ``None`` and ``0.0`` are different claims: nothing was observed, versus
        everything observed was not performed.
        """
        return None if not self.observed else self.performed / self.observed

    @property
    def disagreement_rate(self) -> float | None:
        """``split / multi_voter``, or ``None`` on an empty denominator.

        **It divides by the rows with two or more voters, never by**
        :attr:`observed` (§0 v42). A row one judge read *cannot* split — a cell
        with one voter always resolves — so `observed` carries rows structurally
        incapable of entering the numerator. Dividing by it yields the true rate
        multiplied by the fraction of observed rows that had two or more voters,
        and that fraction is a property of how many judges ran and of how often
        each omitted this particular check, never a property of the check. It is
        at most one, so the bias has a single direction: **toward zero, in
        proportion to how thinly the check was judged.**

        The asymmetry is what confines this rule to this one rate.
        :attr:`performed_rate` keeps `observed` and is untouched: a single-voter
        row is a real reading of what the model did. This rate is the only number
        a report prints that asks about the **instrument** rather than about the
        model, and its observable set is not *rows we got a reading on* but *rows
        on which two readings could differ*.

        The name is kept: it was underspecified under the old divisor rather than
        wrong, and what replaces a rename as the visible signal is
        :attr:`multi_voter` printing beside the rate (§0 v42).

        A statement about our check-writing rather than about the model: a check
        with a high disagreement rate is ambiguously written and is rewritten or
        withdrawn. Conservatism does not favour the wider denominator — a figure
        biased toward zero **under**-triggers, and the failure is keeping an
        ambiguously written check and reporting counts over it.
        """
        return None if not self.multi_voter else self.split / self.multi_voter

    def reconciles(self, rows: int) -> bool:
        """Whether the counts account for **every** trajectory in the matrix.

        The property that makes this layer checkable, and worth more than any
        number it computes. Nothing can be silently dropped: every row lands in
        exactly one of the four bands or one of the two non-observation causes.

        Under v34 the denominator is the judged set **minus what was manually
        excluded**, which is why :attr:`excluded` is a term here from the first
        implementation rather than a later addition (§0 v34).
        """
        return self.rows == rows


def check_counts(matrix: VerdictMatrix) -> CheckCounts:
    """Count one matrix.

    Args:
        matrix: One (model, check, requirement class).

    Returns:
        The counts, reconciling against ``len(matrix.rows)``.
    """
    tally = {
        "performed": 0,
        "partial": 0,
        "not_performed": 0,
        "split": 0,
        "unobserved_reachable": 0,
        "excluded": 0,
        "multi_voter": 0,
    }
    band_field = {
        Outcome.PERFORMED: "performed",
        Outcome.PARTIAL: "partial",
        Outcome.NOT_PERFORMED: "not_performed",
    }

    for row in matrix.rows:
        if row.excluded_reason is not None:
            tally["excluded"] += 1
            continue
        if len(row.votes) >= 2:
            # the disagreement rate's own denominator, counted after the
            # excluded rows are gone: an excluded row yields no observation, so
            # it is no more a row on which two readings could differ than it is
            # a row we got a reading on (§0 v42)
            tally["multi_voter"] += 1
        resolution, _ = row.resolution()
        if resolution is Resolution.RESOLVED:
            outcome = next(iter(row.votes.values()))
            tally[band_field[outcome]] += 1
        elif resolution is Resolution.SPLIT:
            tally["split"] += 1
        else:
            # `Resolution.UNREACHABLE` is not reachable from here: every row
            # passes ``reachable=True`` (`MatrixRow.resolution`), so an empty
            # vote set resolves as `UNOBSERVED_REACHABLE` and nothing else can
            # arrive (§0 v37).
            tally["unobserved_reachable"] += 1

    return CheckCounts(
        model=matrix.model,
        check_id=matrix.check_id,
        requirement=matrix.requirement,
        instances=matrix.instances,
        judge_calls=len(matrix.columns),
        **tally,
    )


# -- per-cause completeness exclusions (item 4) -----------------------------


class CompletenessExclusions(BaseModel):
    """Trajectories excluded **before** judging, counted per cause.

    **Exclusion counts are results, not hygiene** (§5). Context exhaustion and
    truncation are downstream of the model's own retrieval and reasoning, so the
    exclusion correlates with the behaviour under test: a model that retrieves
    profligately has more of its trajectories discarded. The same holds for the
    turn cap and the token limit, because those are our numbers, and a model that
    cannot finish inside our turn cap is telling us something. Treating them as
    noise biases the rate in favour of the model that is worse at scoping its
    work.

    **Every exclusion is reported per cause**, never as one bucket and never
    absorbed into a rate (§5). :attr:`by_cause` is the reporting key, and the
    string is
    :attr:`~genomics_harness.scoring.ExcludedTrajectory.reason` verbatim — a
    cause visible in the trace is reported against the trace, so a truncation
    excludes as ``completeness:truncated`` and never as ``stop_cause:output_cap``.

    **Not the review-based exclusion, and the two never merge** (§5). This one
    happens before any judge runs and its trajectories reach no scoring log at
    all — which is why it is computable only from the generation log, and why
    v34's scoring-logs-only input could not produce it. The review exclusion
    happens *after* judging, keeps its row in the matrix, and is counted as
    :attr:`CheckCounts.excluded` under its own name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectories: int = 0
    """Every trajectory in the generation logs, judged or not. The denominator."""

    judgeable: int = 0
    """What reached the judges: axis 1 ``CLEAN``, axis 2 ``MODEL_STOPPED``, the
    carrier complete rather than provisional, and the two records agreeing."""

    by_cause: dict[str, int] = Field(default_factory=dict)
    """Cause to count. Never summed into one bucket for reporting."""

    @property
    def excluded(self) -> int:
        return sum(self.by_cause.values())

    def rate_by_cause(self) -> dict[str, float]:
        """Per-cause exclusion rate over :attr:`trajectories`.

        §5 asks for per-model exclusion **rates** per cause alongside the
        per-check counts, and this is the one place a rate is the reported form
        rather than a division a reader is left to do — because it is what makes
        exclusion drift between two models visible. Empty on an empty
        denominator: nothing was run is not the same claim as nothing was
        excluded.
        """
        if not self.trajectories:
            return {}
        return {
            cause: count / self.trajectories
            for cause, count in sorted(self.by_cause.items())
        }

    def lines(self) -> list[str]:
        """One printable line per cause, plus the totals."""
        out = [
            f"trajectories {self.trajectories} · judged {self.judgeable} · "
            f"excluded {self.excluded}"
        ]
        out.extend(
            f"  {cause}: {count}" for cause, count in sorted(self.by_cause.items())
        )
        return out


def completeness_exclusions(
    evaluations: Sequence[EvaluationLogs],
) -> CompletenessExclusions:
    """Count the pre-judging exclusions per cause, over the generation logs.

    Recomputed here rather than carried: `select_judgeable` is the one function
    that makes the decision, it is deterministic over a stored log, and running
    it again is how the report's number and the driver's number are the same
    number rather than two that agree by convention.

    Args:
        evaluations: The paired evaluations a report runs over.

    Returns:
        The pooled counts. A report is per model (§5), so pooling across the
        evaluations in one report is a per-model number by construction.

    Raises:
        ValueError: A generation log was read header-only.
    """
    trajectories = 0
    judgeable = 0
    by_cause: dict[str, int] = {}
    for evaluation in evaluations:
        selection: Selection = select_judgeable(evaluation.generation.log_or_raise())
        trajectories += len(selection.judgeable) + len(selection.excluded)
        judgeable += len(selection.judgeable)
        for cause, count in selection.causes.items():
            by_cause[cause] = by_cause.get(cause, 0) + count
    return CompletenessExclusions(
        trajectories=trajectories, judgeable=judgeable, by_cause=by_cause
    )


# -- the two report-level surfaces (item 5) ---------------------------------


class ReviewItem(BaseModel):
    """One split cell in a report, addressed so a human can resolve it.

    The report-level twin of the per-evaluation `ReviewItem` retired at v40, and
    a distinct type because the address is different: a report is keyed by
    trajectory ``uuid`` across evaluations, where the retired per-evaluation form
    was keyed by ``(sample_id, epoch)`` within one. ``(sample_id, epoch)`` is not
    a key across evaluations at all — Inspect numbers epochs from 1 within each
    ``eval()`` call (§0 v34).

    **The paragraph that stood here described a live per-evaluation version over
    a `Panel`.** Both went at v40 with the per-evaluation layer, so it named two
    things that no longer exist and referred a reader to an open question that
    was closed; the sentence also carried a doubled word. v41 residue, corrected
    (§9, §0 v48).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uuid: str
    instance: str
    evaluation: str
    check_id: str
    requirement: Requirement
    """Which of the two matrices the split was in. A check split as required is
    a different observation from the same check split as enriching."""

    verdicts: dict[str, Outcome]
    """Voting judge call to the band it voted, sorted by column id."""


def review_queue(
    matrices: Mapping[tuple[str, str, Requirement], VerdictMatrix],
) -> list[ReviewItem]:
    """Every split cell across a report's matrices, as a list a human works through.

    §5 routes a split verdict to manual review, and **a split cell goes to review
    while the trajectory stays**: review is per cell and never voids the
    trajectory or the evaluation around it.

    **Re-exported at package level**, as ``genomics_harness.review_queue``. The
    name was held by the per-evaluation function until v40; retiring that layer
    freed it, which §14.34 records as the collision the retirement resolves
    rather than manages. Until this round the docstring still said the name was
    taken and sent a reader to ``report.review_queue`` instead — a wrong
    instruction rather than a stale one, and v41 residue (§9, §0 v48).
    :attr:`Report.review` is the same queue, already computed.

    Args:
        matrices: From :func:`build_matrices`.

    Returns:
        The queue, ordered by check, requirement class and trajectory, so two
        runs over one report produce the same list.
    """
    queue: list[ReviewItem] = []
    for _, check_id, requirement in sorted(
        matrices, key=lambda key: (key[1], key[2].value, key[0])
    ):
        matrix = matrices[
            next(key for key in matrices if key[1:] == (check_id, requirement))
        ]
        for row in matrix.rows:
            if row.excluded_reason is not None:
                continue
            resolution, _ = row.resolution()
            if resolution is not Resolution.SPLIT:
                continue
            queue.append(
                ReviewItem(
                    uuid=row.uuid,
                    instance=row.instance,
                    evaluation=row.evaluation,
                    check_id=check_id,
                    requirement=requirement,
                    verdicts=dict(sorted(row.votes.items())),
                )
            )
    return queue


class JudgeConfigurationError(ValueError):
    """One judge name carries two different `JudgeConfig`s across a report.

    **Narrower than a combination axis, and a refusal rather than an
    accommodation** (§0 v42). Judges are *not* a combination axis: an evaluation
    judged by three and one judged by four still combine, and the matrix gains a
    column present on some rows and absent on others, which the stored artifact
    already expects. What is refused is one thing only — the same scorer *name*
    standing for two different configurations inside one report.

    Per-judge numbers key on the scorer name alone, and within one report a name
    means one configuration: `judge_sonnet_xhigh` is Sonnet 5 at xhigh, so pooling
    that name across the evaluations a report combines is correct. Keying on
    ``(JudgeConfig digest, scorer name)`` instead would split the collision into
    two silently-correct rows, and a key that tolerates an error hides it —
    rejected for that reason and recorded so it is not re-proposed.
    """


def assert_one_configuration_per_judge(
    evaluations: Sequence[EvaluationLogs],
) -> dict[str, JudgeConfig]:
    """Refuse a report whose judge names do not each mean one configuration.

    The guard on :func:`lone_dissents`' key. `judge_opus_a` and `judge_opus_b`
    stay two rows at identical configuration — §5 rules them two independent
    draws rather than one instrument sampled twice — so this never merges two
    names; it only refuses one name standing for two things.

    A judge whose scoring log carries no matching `JudgeConfig` in its sidecar is
    skipped: an absent configuration is not a second configuration.

    Args:
        evaluations: The paired evaluations a report would run over.

    Returns:
        Judge name to the one configuration it carries, for the names that carry
        one at all.

    Raises:
        JudgeConfigurationError: A judge name carries two different
            configurations. The message names the name and both configurations.
    """
    seen: dict[str, JudgeConfig] = {}
    where: dict[str, str] = {}
    for evaluation in evaluations:
        for log in evaluation.scoring:
            config = next(
                (j for j in log.provenance.judges if j.name == log.judge), None
            )
            if config is None:
                continue
            first = seen.get(log.judge)
            if first is None:
                seen[log.judge] = config
                where[log.judge] = log.location
                continue
            if first.digest != config.digest:
                raise JudgeConfigurationError(
                    f"judge {log.judge!r} carries two different configurations "
                    f"across the evaluations this report combines: "
                    f"{first.model_dump(mode='json')} in {where[log.judge]}, and "
                    f"{config.model_dump(mode='json')} in {log.location}. "
                    "Per-judge numbers key on the scorer name alone, so within "
                    "one report a name must mean one configuration; keying on "
                    "the configuration digest beside the name would split this "
                    "into two silently-correct rows"
                )
    return seen


class LoneDissent(BaseModel):
    """One judge's lone dissents on one check, over the rows it could have had them on.

    **Lone-dissent attribution is a report-level number, per judge and per
    check** (§0 v42). Three or more voters lets a lone dissent be attributed: one
    judge dissenting against two or more that agree among themselves is
    attributable to that judge. A three-way disagreement is attributable to
    nobody and counts only as a split.

    **What the number is about, stated here because its obvious reading is
    unsafe.** It measures the **instrument** and not the model, and agreement
    measures consistency rather than correctness — a check every judge scores
    identically and wrongly is invisible to it — so a judge that dissents often
    is not thereby unreliable.

    **It must never feed a discard rule.** §14.32's vote-discarding mechanism
    stays a *declared* list and is never a threshold over this count. The
    concrete hazard is §11's: what settles the house-shaped-assumption threat is
    a judge of a different pedigree, and that judge is precisely the one a
    lone-dissent count marks as the outlier. A harness discarding on this number
    would select for lineage agreement — automating the validity threat §11
    exists to detect.

    **Dissents are counted flat, not by band pair.** A `partial` dissenting from
    a unanimous `performed` and a `not_performed` dissenting from the same are
    different observations, and §5 forbids averaging the bands, so the difference
    has no defensible single number. :func:`review_queue` carries every voting
    judge and its band per split row, which is where that detail lives.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    judge: str
    """The scorer name alone — `judge_sonnet_xhigh`, not the column id.

    A judge name is fixed to a model and an effort, so pooling a name across the
    evaluations a report combines is correct;
    :func:`assert_one_configuration_per_judge` is what makes that safe.
    `judge_opus_a` and `judge_opus_b` stay two rows at identical configuration,
    because §5 rules them two independent draws rather than one instrument
    sampled twice.
    """

    check_id: str
    """**Per check, and not per requirement class** — §5 names two key components
    for this number where every other number a report computes carries three.
    Pooling the two matrices double-counts nothing: a trajectory carries one
    requirement class per check, so its row exists in exactly one of them."""

    dissents: int = 0
    """Rows where this judge was the lone dissenter."""

    attributable: int = 0
    """**The denominator: the rows on which this judge could have been a lone
    dissenter**, which is the disagreement rate's rule applied a second time.

    A row qualifies when this judge voted, at least two other judges voted, and
    those others agreed among themselves. The three ineligible shapes are
    therefore **the judge omitting the check**, **fewer than three judges
    voting**, and **the other judges splitting among themselves** — on the last,
    no dissent could be lone. An excluded row qualifies on none of them: review
    found the trajectory invalid, so it yields no observation of any kind.

    **The retired layer's whole-panel denominator is neither necessary nor
    sufficient** — it demands every judge where three suffice, and it admits rows
    on which the others disagreed — and it is not to be rebuilt (§0 v42). This is
    the first denominator in a report that depends on the *other* judges'
    behaviour.
    """


def lone_dissents(
    matrices: Mapping[tuple[str, str, Requirement], VerdictMatrix],
) -> list[LoneDissent]:
    """Attribute lone dissents, per judge and per check.

    See :class:`LoneDissent` for what the number is about and the one thing it
    must never feed.

    Args:
        matrices: From :func:`build_matrices`.

    Returns:
        One entry per (judge, check) with a non-empty denominator, ordered by
        judge then check so two runs over one report produce the same list.

    Raises:
        ValueError: A row carries a cell under a column the matrix does not
            list. The scorer name is the key, and it is read from the column
            rather than re-derived from the column id.
    """
    dissents: dict[tuple[str, str], int] = {}
    attributable: dict[tuple[str, str], int] = {}

    for (_, check_id, _), matrix in matrices.items():
        judge_of = {column.id: column.judge for column in matrix.columns}
        for row in matrix.rows:
            if row.excluded_reason is not None:
                continue
            votes = row.votes
            if len(votes) < 3:
                # fewer than three voters: no row here can carry a dissent that
                # is lone, whichever judge is asked about (§0 v42)
                continue
            named: dict[str, Outcome] = {}
            for column_id, outcome in votes.items():
                judge = judge_of.get(column_id)
                if judge is None:
                    raise ValueError(
                        f"matrix {matrix.check_id}/{matrix.requirement.value}: "
                        f"trajectory {row.uuid} carries a cell under column "
                        f"{column_id!r}, which the matrix does not list. The "
                        "per-judge key is the scorer name and it is read from "
                        "the column, never re-derived from the column id"
                    )
                named[judge] = outcome
            for judge, outcome in named.items():
                others = [band for other, band in named.items() if other != judge]
                if len(set(others)) != 1:
                    # the other judges split among themselves, so no dissent
                    # here is attributable to anybody
                    continue
                key = (judge, check_id)
                attributable[key] = attributable.get(key, 0) + 1
                if outcome != others[0]:
                    dissents[key] = dissents.get(key, 0) + 1

    return [
        LoneDissent(
            judge=judge,
            check_id=check_id,
            dissents=dissents.get((judge, check_id), 0),
            attributable=count,
        )
        for (judge, check_id), count in sorted(attributable.items())
    ]


class ChainViolation(BaseModel):
    """One trajectory on which a refinement outscored the check it refines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uuid: str
    instance: str
    evaluation: str
    chain: list[str]
    """The whole chain, base first."""

    base: str
    refinement: str
    """The offending link: ``refinement`` refines ``base``."""

    base_band: float
    refinement_band: float


class ChainConsistency(BaseModel):
    """One refinement chain's coherence over a report's judged set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chain: list[str]
    """Base first, built from the catalogue's own ``refines`` links and never
    from a hardcoded id triple."""

    consistent: int = 0
    violations: int = 0

    no_claim: int = 0
    """Trajectories the chain says nothing about: some link has no row, did not
    resolve, or was excluded by review. Counted as **neither** consistent nor
    violating — a chain read across a split or an omission would report a judging
    gap as a model incoherence."""

    offending: list[ChainViolation] = Field(default_factory=list)


def chain_consistency(
    matrices: Mapping[tuple[str, str, Requirement], VerdictMatrix],
    catalogue: Catalogue,
) -> list[ChainConsistency]:
    """Flag refinements credited above the check they refine, across a report.

    Surfacing a refined step implies the step it refines — the judges are told as
    much (`judge_prompt._NESTING_INSTRUCTION`) — so a refinement resolved to a
    strictly higher band than its base is incoherent, and it is incoherent in the
    *resolved* reading rather than in any one judge's reply.

    **The chains come from the catalogue's own ``refines`` links**, as the
    per-evaluation version's do. A chain is read **across** the requirement
    classes, not within one: a trajectory holds each of its checks in exactly one
    matrix, and B6 enriching refining B5 required is still one trajectory's one
    chain. Splitting the read by requirement class would silently drop every
    chain whose links disagree on weight, which two instances over one substrate
    make the common case rather than the exotic one.

    Args:
        matrices: From :func:`build_matrices`.
        catalogue: The checks, whose ``refines`` links are the chains.

    Returns:
        One report per chain, in catalogue chain order. Empty when the catalogue
        has no refinements.
    """
    chains = refinement_chains(catalogue, catalogue.ids)
    if not chains:
        return []

    # uuid -> check id -> (band, instance, evaluation). A row that did not
    # resolve, or that review excluded, contributes no entry at all, which is
    # what makes "some link says nothing" the same test as "some link is absent".
    resolved: dict[str, dict[str, float]] = {}
    where: dict[str, tuple[str, str]] = {}
    trajectories: list[str] = []
    for matrix in matrices.values():
        for row in matrix.rows:
            if row.uuid not in where:
                where[row.uuid] = (row.instance, row.evaluation)
                trajectories.append(row.uuid)
            if row.excluded_reason is not None:
                continue
            resolution, band = row.resolution()
            if resolution is Resolution.RESOLVED and band is not None:
                resolved.setdefault(row.uuid, {})[matrix.check_id] = band

    reports: list[ChainConsistency] = []
    for chain in chains:
        consistent = 0
        no_claim = 0
        violations: list[ChainViolation] = []

        for uuid in trajectories:
            bands = resolved.get(uuid, {})
            if any(check_id not in bands for check_id in chain):
                no_claim += 1
                continue

            instance, evaluation = where[uuid]
            offending = [
                ChainViolation(
                    uuid=uuid,
                    instance=instance,
                    evaluation=evaluation,
                    chain=list(chain),
                    base=base,
                    refinement=refinement,
                    base_band=bands[base],
                    refinement_band=bands[refinement],
                )
                for base, refinement in zip(chain, chain[1:])
                if bands[refinement] > bands[base]
            ]
            if offending:
                violations.extend(offending)
            else:
                consistent += 1

        reports.append(
            ChainConsistency(
                chain=list(chain),
                consistent=consistent,
                violations=len(violations),
                no_claim=no_claim,
                offending=violations,
            )
        )
    return reports


# -- judge spend, at report level ------------------------------------------


class JudgeSpend(BaseModel):
    """One judge's own token spend, pooled over the evaluations a report reads.

    **The positive half of the §5 spend rule.** The control forbidding the three
    stale fields has been in place since v15: on a scoring log
    ``EvalSample.model_usage``, ``EvalStats.model_usage`` and
    ``ScoreEvent.model_usage`` are untouched by `score()` and therefore carry the
    **generation's** numbers — a real number that is not the judge's, which §7
    calls worse than a missing one. What was absent is a reader that returns the
    judge's number at the unit a report is about. This is that reader's output.

    **Recorded and never reported as a rate.** Spend is a quantity §5 measures
    and never caps, and it is pooled per judge rather than divided by anything:
    the denominator that would make it a rate is :attr:`scored`, which is already
    on the row.

    :attr:`recorded` and :attr:`scored` are two counts on purpose, and the gap
    between them is the honest one. A judge that recorded no usage is not a judge
    that spent nothing — `mockllm` driven by a callable records none at all
    (`_providers/mockllm.py:88-95`) — so a total summed over fewer trajectories
    than were scored is a total over a subset, and the row says which.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    judge: str
    """The judge's score key, pooled across every call this report combines."""

    calls: int = 0
    """``score()`` calls — one per evaluation this judge ran over."""

    scored: int = 0
    """Trajectories this judge produced a `Score` for."""

    recorded: int = 0
    """Of those, the ones carrying usage. ``scored - recorded`` recorded none."""

    usage: ModelUsage = Field(default_factory=ModelUsage)
    """The sum over :attr:`recorded` trajectories. The **whole** `ModelUsage`,
    because ``input_tokens`` is what §5 reports as the judge-input cost in the
    tokenizer that bills, and §8's cost model needs the cache columns."""


def judge_spend_totals(evaluations: Sequence[EvaluationLogs]) -> list[JudgeSpend]:
    """Each judge's own spend across the evaluations a report combines.

    Read through :func:`~genomics_harness.scoring.judge_usage` and from nowhere
    else, which is the whole discipline: that function reads one documented
    `Score.metadata` key, and the three aggregate fields that look like carriers
    hold the generation's usage on a scoring log (§5, §6.6, §0 v14).

    Args:
        evaluations: Paired evaluations, from :func:`load_evaluation`.

    Returns:
        One row per judge, in first-seen order. Pooled by judge **name**, which
        is the unit §5 reports spend at — :class:`JudgeCall` keeps the
        per-evaluation columns apart because two calls are two independent draws,
        and that distinction is about verdicts rather than about cost.

    Raises:
        ValueError: A scoring log was read header-only.
    """
    rows: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        for log in evaluation.scoring:
            row = rows.setdefault(
                log.judge,
                {"calls": 0, "scored": 0, "recorded": 0, "usage": ModelUsage()},
            )
            row["calls"] += 1
            for sample in log.samples:
                judged = (sample.scores or {}).get(log.judge)
                if judged is None:
                    continue
                row["scored"] += 1
                usage = judge_usage(judged)
                if usage is not None:
                    row["recorded"] += 1
                    row["usage"] = row["usage"] + usage
    return [
        JudgeSpend(
            judge=judge,
            calls=row["calls"],
            scored=row["scored"],
            recorded=row["recorded"],
            usage=row["usage"],
        )
        for judge, row in rows.items()
    ]


class Report(BaseModel):
    """The analysis over a group of combinable evaluations, for one model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    evaluations: list[str]
    """``EvalSpec.eval_id`` per evaluation, which distinguishes two evaluations
    whose sidecars are byte-identical — the repeat case this design supports,
    three evaluations at one epoch standing in for one at three."""

    judge_calls: list[JudgeCall]
    catalogue_release: str
    checks: list[str]
    """The intersection the report ran over."""

    dropped_checks: list[str]
    """Checks some catalogue lacked, named rather than silently absent."""

    matrices: dict[str, VerdictMatrix]
    """Keyed ``"{check_id}:{requirement}"`` — the two matrices per check never
    merge."""

    counts: dict[str, CheckCounts]
    """Same keys as :attr:`matrices`."""

    exclusions: list[Exclusion] = Field(default_factory=list)
    """Trajectories review found invalid, **after** judging. Their rows stay in
    the matrices and are counted as :attr:`CheckCounts.excluded`."""

    completeness: CompletenessExclusions = Field(default_factory=CompletenessExclusions)
    """Trajectories excluded **before** judging, per cause. Beside
    :attr:`exclusions` and never merged into it: one happens before judging and
    one after, and §5 keeps them apart."""

    review: list[ReviewItem] = Field(default_factory=list)
    """Every split cell, as a queue a human works through."""

    consistency: list[ChainConsistency] = Field(default_factory=list)
    """One entry per refinement chain the catalogue declares."""

    dissent: list[LoneDissent] = Field(default_factory=list)
    """Lone dissents, per judge and per check, with the rows each judge could
    have had one on. It measures the instrument and not the model, and **it must
    never feed a discard rule** (:class:`LoneDissent`, §11, §14.32)."""

    spend: list[JudgeSpend] = Field(default_factory=list)
    """Each judge's own token spend, from the one `Score.metadata` key that
    carries it. A reported quantity and never a capped one (§5); it enters no
    rate and no verdict, and it is on the report because the unit §5 measures
    judge cost at is the evaluation group, not the single log."""

    def reconciles(self) -> bool:
        """Whether every check's counts account for every row of its matrix."""
        return all(
            self.counts[key].reconciles(len(matrix.rows))
            for key, matrix in self.matrices.items()
        )

    def rows(self) -> list[str]:
        """The report, one printable line per (check, requirement class).

        **Both quantities, in the form §5 shows** (§0 v37): the displayed
        trajectory count is every row the matrix holds, and *observed* — resolved
        plus split — is what the rates divide by. Printing one field for both
        reported one of them wrong, which is the defect this line was built from.

        The reconciliation is checkable in the line itself: the bands and the
        split sum to *observed*, and *observed* plus the non-observations sums to
        *trajectories*.

        **Every non-observation prints under its own name**, so the line closes
        arithmetically whatever the input. §5's worked example carries one
        non-observation term because its example has no reviewed exclusion; a
        report that has one must not bucket the two, since §5 rules exclusions are
        reported per cause and never as one bucket.

        **The disagreement term is the exception and prints last** (§0 v42). Its
        denominator is :attr:`CheckCounts.multi_voter` and not *observed*, it
        sums with nothing on this line, and it is here because the multi-voter
        count appears nowhere else — displaying *observed* alone would leave the
        narrower rate unrecoverable from the wider one.
        """
        lines: list[str] = []
        for key in sorted(self.counts):
            counts = self.counts[key]
            lines.append(
                f"{counts.check_id} ({counts.requirement.value}) — "
                f"{counts.rows} trajectories · "
                f"{counts.observed} observed · "
                f"{counts.performed} performed · "
                f"{counts.partial} partial · "
                f"{counts.not_performed} not_performed · "
                f"{counts.split} split · "
                f"{counts.unobserved_reachable} unobserved · "
                f"{counts.excluded} excluded · "
                f"disagreement {counts.split}/{counts.multi_voter}"
            )
        return lines


def _matrix_key(check_id: str, requirement: Requirement) -> str:
    return f"{check_id}:{requirement.value}"


def build_report(
    evaluations: Sequence[EvaluationLogs], *, exclusions: ExclusionList | None = None
) -> Report:
    """The whole layer: pair, refuse if the evaluations do not combine, count.

    **The signature is not backwards compatible with v34's** and that is
    deliberate. It took scoring log paths; §5's input is, per evaluation, a
    generation log **and** the scoring logs taken from it, and if either is
    missing the report refuses and produces nothing. A widening that left the old
    call working would have kept every read on the generation side going through
    ``score()``'s copy while looking correct (§7).

    Args:
        evaluations: Paired evaluations, from :func:`load_evaluation`.
        exclusions: Trajectories review found invalid, from
            :func:`read_exclusions`.

    Returns:
        The report — the matrices, the counts, the per-cause pre-judging
        exclusions, the review queue and the refinement-chain reports.

    Raises:
        CombinationError: The evaluations do not combine. Names the log and axis.
        ExclusionError: Propagated from the exclusion list.
        PairingError: Propagated from :func:`load_evaluation`.
        ValueError: Nothing to report, a scoring log supplied twice, or a stored
            score whose shape could not be confirmed.
    """
    comparison = assert_combinable(evaluations)
    assert_one_configuration_per_judge(evaluations)

    matrices = build_matrices(
        evaluations, exclusions=exclusions, checks=comparison.shared
    )
    counts = {
        _matrix_key(check_id, requirement): check_counts(matrix)
        for (_, check_id, requirement), matrix in matrices.items()
    }
    keyed = {
        _matrix_key(check_id, requirement): matrix
        for (_, check_id, requirement), matrix in matrices.items()
    }

    ids: list[str] = []
    calls: list[JudgeCall] = []
    for evaluation in evaluations:
        if evaluation.evaluation not in ids:
            ids.append(evaluation.evaluation)
        for log in evaluation.scoring:
            judge_config = next(
                (j for j in log.provenance.judges if j.name == log.judge), None
            )
            calls.append(
                JudgeCall(
                    evaluation=evaluation.evaluation,
                    judge=log.judge,
                    model=None if judge_config is None else judge_config.model,
                    reasoning_effort=(
                        None if judge_config is None else judge_config.reasoning_effort
                    ),
                    location=log.location,
                )
            )

    catalogue = evaluations[0].scoring[0].catalogue
    return Report(
        model=evaluations[0].model,
        evaluations=ids,
        judge_calls=calls,
        catalogue_release=catalogue.release,
        checks=comparison.shared,
        dropped_checks=comparison.dropped,
        matrices=keyed,
        counts=counts,
        exclusions=list((exclusions or ExclusionList()).exclusions),
        completeness=completeness_exclusions(evaluations),
        review=review_queue(matrices),
        consistency=chain_consistency(matrices, catalogue),
        dissent=lone_dissents(matrices),
        spend=judge_spend_totals(evaluations),
    )


# -- the survey -------------------------------------------------------------


class SurveyEvaluation(BaseModel):
    """One paired evaluation as the survey sees it, from headers and sidecars alone.

    Every axis field here is read from the **generation** log's own provenance
    record; the catalogue and the judges are read from the scoring side. That is
    the same split :class:`EvaluationLogs` makes, one read down (§0 v35).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation: str
    model: str
    generation_location: str
    """The generation log this evaluation's axes were read from."""

    locations: list[str]
    """Its scoring logs."""

    judges: list[str]
    catalogue_release: str
    catalogue_sha256: str
    instances_name: str
    instances_sha256: str
    renderer_version: str
    framework_version: str
    solver_sha256: str | None = None
    dataset_snapshot: str = ""
    """Recorded and not enforced, until OneKGPd exposes an identifier (§2)."""

    unknown_instances: bool = False
    """The instances digest matches no set resolvable in this tree.

    §14.31's check, and this is the only place it can run: it has to run at
    analysis time, against the instances and the catalogue a log names.

    **It says nothing about combinability, and it must not be read as saying so.**
    What combines is decided from what the logs themselves record (§5's axis
    table, :func:`assert_combinable`), and instances are the one axis that may
    vary — the whole point of the operation. This flag is about *this checkout*:
    a log carrying a digest nothing here resolves cannot have the instances
    behind its trajectories shown, so an operator reading the survey cannot see
    what was run. The remedy §14.31 names — re-judge it or remove it — is that
    provenance decision and not a report refusal.

    **The empty ``[sets]`` table is why the distinction is load-bearing.** A
    release checkout ships with no instances at all (§16), so
    :func:`_tree_reference` resolves nothing and *every* evaluation raises this
    flag while every one of them still combines and still reports. A survey that
    let the flag decide what groups would tell an operator on a release checkout
    that none of their evaluations can be reported.
    """

    unknown_catalogue: bool = False
    """This log's check **definitions** match no catalogue shipped in this tree.

    Compared on ``(id, text)`` over the log's checks rather than on the
    `Catalogue` digest, and the reason is a two-carrier one. The delivered
    `Catalogue` a judge is handed is the pinned file's in-scope subset carrying a
    generated ``notes`` string (§7, §0 v25), and that string is inside the
    digest — so reproducing the digest here would mean reproducing that sentence
    in a second place, which is the drift this whole section exists to close.
    ``text`` alone is the definition (§0 v34), so comparing definitions asks the
    question §14.31 actually asks: do this log's verdicts rest on checks that
    still exist here?
    """


class SurveyGroup(BaseModel):
    """A maximal set of evaluations that combine with each other."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    evaluations: list[str]
    checks: list[str]
    """The intersection a report over this group would run over."""

    dropped_checks: list[str] = Field(default_factory=list)


class Survey(BaseModel):
    """Which evaluations combine with which, and why the rest do not.

    **A second mode surveys a log directory and reports which evaluations combine
    with which** (§0 v34). It reads headers only. This is the operator's
    instrument for deciding what to re-judge and what to discard.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluations: list[SurveyEvaluation] = Field(default_factory=list)
    groups: list[SurveyGroup] = Field(default_factory=list)
    incombinable: list[str] = Field(default_factory=list)
    """One line per pair that does not combine, naming the axis."""

    unpaired: list[str] = Field(default_factory=list)
    """Logs that read cleanly but have no other half — **its own named
    condition**, never a drop (§5).

    A generation log with no scoring logs beside it has not been judged; scoring
    logs with no generation log beside them cannot supply a requirement class, a
    solver digest or an exclusion cause, so neither half can enter a report on
    its own. The survey is the operator's instrument for deciding what to
    re-judge and what to discard, and an input it silently omitted would be the
    one thing it exists to surface."""

    unreadable: list[str] = Field(default_factory=list)
    """Logs the survey could not read as either half, with the reason. Never
    raises: a survey over a directory is exactly where mixed contents are
    expected, and a survey that raises on a stray file tells an operator
    nothing."""

    def lines(self) -> list[str]:
        """The survey as text.

        **The `combine` lines and the `unmatched` lines answer different
        questions and are not in tension** (§0 v43 reported them as two
        contradictory lines about one set of logs). `combine` is read from what
        the logs record, over §5's combination axes, where instances are the one
        axis that may vary. `unmatched` is read against *this checkout* and says
        the digests resolve to nothing here — a provenance-resolution failure,
        not a report refusal. Every evaluation in a release checkout raises the
        second and none of them is thereby uncombinable
        (:attr:`SurveyEvaluation.unknown_instances`).
        """
        out: list[str] = []
        for group in self.groups:
            out.append(
                f"combine ({group.model}): {len(group.evaluations)} evaluations "
                f"{group.evaluations} over {len(group.checks)} checks"
                + (f", dropping {group.dropped_checks}" if group.dropped_checks else "")
            )
        out.extend(self.incombinable)
        for evaluation in self.evaluations:
            if evaluation.unknown_instances or evaluation.unknown_catalogue:
                unmatched = [
                    name
                    for name, flag in (
                        ("instances", evaluation.unknown_instances),
                        ("catalogue", evaluation.unknown_catalogue),
                    )
                    if flag
                ]
                out.append(
                    f"unmatched ({evaluation.evaluation}): {unmatched} match "
                    "nothing in this tree, so what these verdicts rest on cannot "
                    "be resolved from this checkout — re-judge it or remove it. "
                    "It does not decide combinability: any `combine` line above "
                    "is read from what the logs themselves record"
                )
        out.extend(f"unpaired: {line}" for line in self.unpaired)
        out.extend(f"unreadable: {line}" for line in self.unreadable)
        return out


def _tree_reference() -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    """What this tree can still resolve: instance-set digests, and check definitions.

    Both lookups are best-effort, and that is deliberate: a tree with an empty
    ``[sets]`` table is a shipped release shape (§16) and a survey must still run
    there. What it cannot then do is match anything, which is reported as
    unmatched rather than raised — the survey's job is to tell an operator what
    it found, and a survey that raises on a release checkout tells them nothing.

    Returns:
        Instance-set digests, and ``(check id, text)`` for every check the pinned
        catalogue file carries.
    """
    from .catalogue_file import load_catalogue
    from .composition import available_sets, resolve_set

    instances: set[str] = set()
    try:
        for name in available_sets():
            try:
                instances.add(resolve_set(name).digest)
            except Exception:  # noqa: BLE001 - an unresolvable set is not this tool's error
                continue
    except Exception:  # noqa: BLE001 - a missing or unreadable config is reported, not raised
        pass

    definitions: set[tuple[str, str]] = set()
    try:
        for check in load_catalogue().checks:
            definitions.add((check.id, check.text))
    except Exception:  # noqa: BLE001 - an unloadable catalogue matches nothing
        pass
    return frozenset(instances), frozenset(definitions)


def survey(log_dir: Path | str) -> Survey:
    """Survey a log directory: which evaluations combine, and why the rest do not.

    Reads headers only, plus each scoring log's sidecar. Never opens a sample.

    **The survey pairs too** (§0 v35). A directory holds both halves, and a
    report's input is a pair — so a log with no other half is reported under
    :attr:`Survey.unpaired`, its own named condition, rather than dropped. A
    scoring log's header is told from a generation log's by whether any scorer
    carries a ``catalogue`` option, never by position and never by name (§0 v36),
    and the two are joined on ``EvalSpec.eval_id``, which ``score()`` preserves.

    Args:
        log_dir: The directory of stored logs, generation and scoring together.

    Returns:
        The survey.
    """
    known_instances, known_definitions = _tree_reference()

    # **Which half a log is, is decided by its header and by nothing else.**
    # Loading can fail for reasons that have nothing to do with which half it is
    # — a scoring log written before the judge axis was carried has no sidecar —
    # and classifying on load success would report such a log as "not a scoring
    # log", which is both false and the opposite of the remedy.
    scoring: list[ScoringLog] = []
    generation: list[GenerationLog] = []
    unreadable: list[str] = []
    for info in list_eval_logs(str(log_dir)):
        location = _local_path(info.name)
        try:
            header = read_eval_log(location, header_only=True)
        except Exception as error:  # noqa: BLE001 - a mixed directory is the normal case
            unreadable.append(
                f"{location}: could not be read as an Inspect log ({error})"
            )
            continue
        judged = bool(_carrying_catalogue(header))
        try:
            if judged:
                scoring.extend(load_scoring_logs([location], header_only=True))
            else:
                generation.append(load_generation_log(location, header_only=True))
        except Exception as error:  # noqa: BLE001 - reported, never raised
            half = "scoring" if judged else "generation"
            unreadable.append(
                f"{location}: a {half} log this survey could not load ({error})"
            )

    scoring_by_evaluation: dict[str, list[ScoringLog]] = {}
    for log in scoring:
        scoring_by_evaluation.setdefault(log.evaluation, []).append(log)
    generation_by_evaluation: dict[str, GenerationLog] = {}
    unpaired: list[str] = []
    for one in generation:
        if one.evaluation in generation_by_evaluation:
            unreadable.append(
                f"{one.location}: a second generation log for evaluation "
                f"{one.evaluation!r}, which "
                f"{generation_by_evaluation[one.evaluation].location} already "
                "names. `eval_id` is unique per eval() call, so one of these is a "
                "copy"
            )
            continue
        generation_by_evaluation[one.evaluation] = one

    # the pairing, and both ways it can fail — each named, never dropped (§5)
    paired: dict[str, EvaluationLogs] = {}
    for evaluation_id, logs in scoring_by_evaluation.items():
        matched = generation_by_evaluation.get(evaluation_id)
        if matched is None:
            unpaired.append(
                f"{evaluation_id}: {len(logs)} scoring log(s) "
                f"{[log.location for log in logs]} with no generation log in this "
                "directory. The requirement class, the solver digest and the "
                "exclusion causes are read from the generation side, so these "
                "cannot enter a report on their own"
            )
            continue
        try:
            paired[evaluation_id] = EvaluationLogs(generation=matched, scoring=logs)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            unpaired.append(f"{evaluation_id}: {error}")
    for evaluation_id, lone in generation_by_evaluation.items():
        if evaluation_id not in scoring_by_evaluation:
            unpaired.append(
                f"{evaluation_id}: generation log {lone.location} with no scoring "
                "log in this directory. It has not been judged, so it carries no "
                "verdicts to report"
            )

    evaluations: list[SurveyEvaluation] = []
    for evaluation_id, pair in paired.items():
        first = pair.scoring[0]
        provenance = pair.generation.provenance
        solver = provenance.solver
        evaluations.append(
            SurveyEvaluation(
                evaluation=evaluation_id,
                model=pair.model,
                generation_location=pair.generation.location,
                locations=[log.location for log in pair.scoring],
                judges=pair.judges,
                catalogue_release=first.catalogue.release,
                catalogue_sha256=first.catalogue.digest,
                instances_name=provenance.instances_name,
                instances_sha256=provenance.instances_sha256,
                renderer_version=provenance.renderer_version,
                framework_version=provenance.framework_version,
                solver_sha256=None if solver is None else solver.digest,
                dataset_snapshot=provenance.dataset_snapshot.release,
                unknown_instances=provenance.instances_sha256 not in known_instances,
                unknown_catalogue=not all(
                    (check.id, check.text) in known_definitions
                    for check in first.catalogue.checks
                ),
            )
        )

    # group by transitive combinability, and record why each rejected pair failed
    groups: list[list[str]] = []
    incombinable: list[str] = []
    for evaluation_id in sorted(paired):
        placed = False
        for group in groups:
            candidate = [paired[e] for e in [*group, evaluation_id]]
            try:
                assert_combinable(candidate)
            except (CombinationError, ValueError) as error:
                incombinable.append(f"{evaluation_id} does not join {group}: {error}")
                continue
            group.append(evaluation_id)
            placed = True
            break
        if not placed:
            groups.append([evaluation_id])

    survey_groups: list[SurveyGroup] = []
    for group in groups:
        members = [paired[e] for e in group]
        comparison = catalogue_comparison(
            [log.catalogue for pair in members for log in pair.scoring]
        )
        survey_groups.append(
            SurveyGroup(
                model=members[0].model,
                evaluations=group,
                checks=comparison.shared,
                dropped_checks=comparison.dropped,
            )
        )

    return Survey(
        evaluations=evaluations,
        groups=survey_groups,
        incombinable=incombinable,
        unpaired=unpaired,
        unreadable=unreadable,
    )
