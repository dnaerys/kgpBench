"""Provenance — the seven inputs that can change a verdict, and how each reaches a log.

Four separate hazards found during design turned out to be the same failure: a
number produced by machinery that changed underneath it, with nothing recording
the change. The rule that generalises them is that every input which can affect
a verdict carries a content digest into the log, and the analysis layer refuses
to compare two results whose digests differ on an axis it is not controlling for.

| Input | Carried as |
|---|---|
| Instance definitions | `Task(version=)` + `Task(metadata={"instances_sha256"})` |
| Check catalogue | scorer factory argument + ``catalogue_release`` + digest |
| Judge configuration | model string and effort, per judge, per scoring run |
| Solver configuration | :class:`SolverConfig` digest, in task metadata |
| Trajectory rendering format | :data:`~genomics_harness.version.RENDERER_VERSION` |
| Framework version | installed `inspect_ai` distribution version |
| Dataset snapshot | OneKGPd pinned identifier, in task metadata |
| Trajectory completeness | :mod:`genomics_harness.trajectory` |

Placeholders are used where the producing component does not exist yet. The
fields exist now because retrofitting one makes everything generated before the
retrofit unusable for cross-round comparison.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .catalogue import Catalogue
from .digest import canonical_json, sha256_digest, short_digest
from .instances import InstanceSet
from .version import HARNESS_VERSION, RENDERER_VERSION

__all__ = [
    "COMPARISON_AXES",
    "ComparabilityError",
    "DatasetSnapshot",
    "JudgeConfig",
    "PROVENANCE_KEY",
    "PROVENANCE_SIDECAR_SUFFIX",
    "ProvenanceMismatch",
    "ReasoningEffort",
    "RunProvenance",
    "SolverConfig",
    "assert_comparable",
    "compare_provenance",
    "framework_version",
    "provenance_from_metadata",
    "provenance_sidecar_path",
    "read_run_provenance",
    "task_metadata",
    "task_version",
    "write_run_provenance",
]

PROVENANCE_KEY = "provenance"
"""Key under which :meth:`RunProvenance.to_task_metadata` nests itself."""

PROVENANCE_SIDECAR_SUFFIX = ".provenance.json"
"""Appended to a scoring log's own path to name its provenance sidecar.

A **scoring** run has nowhere else to put one. Generation goes through `eval()`
and carries :meth:`RunProvenance.to_task_metadata` in ``Task(metadata=)``;
scoring goes through `score()`, which accepts no task metadata at all, so the
judge axis — the one axis a scoring run is the only witness to — reached no
carrier the analysis layer could read. The sidecar is that carrier. See
:func:`provenance_sidecar_path`.
"""

UNKNOWN_SNAPSHOT = "onekgpd-unpinned"
"""Placeholder release identifier — see :class:`DatasetSnapshot`."""


def framework_version() -> str:
    """Resolve the Inspect release actually in use.

    Reads the installed distribution version of `inspect_ai`. Under the release
    model the harness runs against a pinned PyPI wheel, so the version *is* the
    framework identity: it is a fact the environment already holds, it is correct
    in both comparison directions, and it does not depend on where the venv sits.

    This is a read of a **declared dependency**, not an observation of a
    repository. What the resolver installed is what ``harness/pyproject.toml``
    pins; a disagreement between the two is a broken environment rather than a
    provenance question, and it is caught by a test rather than represented as a
    second field here.

    Three earlier behaviours are gone:

    - It ran ``git rev-parse HEAD`` in the directory supplying the package. That
      was sound only while `inspect_ai` was an in-tree checkout: once the wheel
      is installed under a venv inside a git work tree, git's upward repository
      discovery walks past site-packages and resolves the *enclosing*
      repository's HEAD, recording the harness's own commit under a field named
      for the framework — with no error anywhere.
    - An ``INSPECT_FORK_SHA`` environment override was consulted first, which
      meant the regression guard against that git value covered the default path
      only: setting the variable to a SHA reintroduced exactly what the guard
      exists to catch.
    - A ``PackageNotFoundError`` fallback returned ``"unknown"``. Two runs in a
      broken environment would both record ``"unknown"`` and compare as *equal*
      on the framework axis, which is the one thing this axis exists to prevent.
      Raising is correct: every caller that constructs a
      :class:`RunProvenance` does so at task-build time, before any model call,
      so the failure is loud and costs nothing.

    Returns:
        The installed `inspect_ai` version, e.g. ``"0.3.252"``.

    Raises:
        importlib.metadata.PackageNotFoundError: `inspect_ai` has no
            distribution metadata in this environment.
    """
    return version("inspect_ai")


class DatasetSnapshot(BaseModel):
    """Identity of the cohort and annotation release ground truth was built against.

    The MCP tool surface does not expose a release identifier — ``getDatasetInfo``
    returns cohort shape only (verified against the live server, see
    `design/foundations.md` §4). So ``release`` has to be supplied by hand from
    the OneKGPd page, and cannot be validated automatically.

    ``fingerprint`` is what makes a silent substrate swap detectable anyway: the
    counts ``getDatasetInfo`` does return, recorded per run. They will not catch
    an annotation-only change — an updated ClinVar or AlphaMissense release
    leaves every count identical — so they are a floor, not a guarantee.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    release: str = UNKNOWN_SNAPSHOT
    """Human-supplied pinned identifier from the OneKGPd page."""

    annotation_sources: list[str] = Field(default_factory=list)
    """e.g. ``["gnomAD 4.1", "AlphaMissense", "ClinVar", "VEP"]``, each versioned."""

    fingerprint: dict[str, Any] = Field(default_factory=dict)
    """Verbatim ``getDatasetInfo`` response for the run."""

    mcp_url: str | None = None

    @property
    def digest(self) -> str:
        return sha256_digest(self.model_dump(mode="json"))

    def is_pinned(self) -> bool:
        return self.release != UNKNOWN_SNAPSHOT


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""Mirrors `GenerateConfig.reasoning_effort` (`model/_generate_config.py:291`),
which ships no exported alias. A `Literal` of strings, so it stays a JSON scalar
where task arguments must be.

Defined above its first use rather than beside :class:`SolverConfig`: pydantic
resolves a field annotation when the model class is built, and both
:class:`JudgeConfig` and :class:`SolverConfig` are typed over it.
"""


class JudgeConfig(BaseModel):
    """One judge's configuration, per scoring run.

    Effort is part of the identity: the same model at a different reasoning
    effort is a different judge, and comparing across it without saying so is
    the confound this record exists to prevent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    """Stable scorer name, e.g. ``judge_sonnet_xhigh``. Score keys derive from the
    `@scorer` factory name, not its arguments (`integration-map.md` §9.6), so
    this must match the registered name to be traceable."""

    model: str
    """Provider-qualified, e.g. ``anthropic/claude-sonnet-5``."""

    reasoning_effort: ReasoningEffort | None = None
    """The effort this judge runs at, typed over the permitted levels.

    A free ``str`` until v15, on the reading that this record is provenance
    rather than an argument. It is both: `judge_roles` builds the request from
    it, and **forward compatibility was never available** — the framework's own
    `GenerateConfig.reasoning_effort` is the same `Literal`
    (`model/_generate_config.py:291`), so a level this mirror lacks is one no
    provider can be sent either, whatever we type the field as. What the free
    ``str`` bought was a failure that landed on whoever launched the scoring pass
    instead of on whoever wrote the configuration — the earlier-failure argument
    that decided :attr:`SolverConfig.reasoning_effort` in §0 v9.

    Typing it is **digest-neutral**: a `Literal` of ``str`` dumps as the same
    plain string, so no existing configuration's :attr:`digest` moves. Pinned
    against the pre-change digests in ``test_provenance.py``.
    """

    reasoning_tokens: int | None = None
    temperature: float | None = None
    prompt_version: str | None = None
    """Version of the judge's own prompt, separate from the catalogue rubric."""

    @property
    def digest(self) -> str:
        return sha256_digest(self.model_dump(mode="json"))


class SolverConfig(BaseModel):
    """The turn loop's policy, as one digested provenance axis (§7).

    **The solver writes nothing into the model's conversation.** The continuation
    turn was the last exception and went in v7 (§0 v7, §11), so the solver's whole
    influence is now configuration: ``max_output_tokens`` decides how much the
    model may generate, ``turn_cap`` decides how long a trajectory may run, and
    ``reasoning_effort`` decides whether there is a readable reasoning channel at
    all. Change any of them and trajectories change with nothing else changing —
    the same failure shape as an unrecorded rendering revision.

    **No field here is a placeholder.** ``continuation_budget``, ``floor``,
    ``headroom``, ``chars_per_token`` and ``context_window`` were, pending
    §14.19; that question closed in v7 by the policy being withdrawn, and the
    fields went with it (§0 v7). What remains is §7's row exactly: turn cap,
    ``reasoning_effort``, **M_max**, ``token_limit``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_cap: int = 40
    """Maximum model turns before the solver stops itself.

    The solver owns termination — `react()` is never in the path (§6.3). A
    runaway guard, and the tasks in §8 are single-question investigations over a
    tool surface of 14 tools. It is on this axis because a trajectory the cap
    stops is excluded, and a model that cannot finish inside it is telling us
    something (§5)."""

    reasoning_effort: ReasoningEffort | None
    """What the generation runs at, **required with no default** (§6.6, §7, §0 v9).

    ``xhigh`` for both models under test (§5), and every caller names it: a
    default here is one model's effort silently carried onto another, on the one
    axis where a wrong value produces no error at all. Omitting it raises a
    `ValidationError` rather than producing a config that digests as though a
    choice had been made.

    **Unset is not "the model's default effort" — it is a silent double
    falsification.** With `reasoning_effort` unset `is_using_thinking` is false
    (`_providers/anthropic.py:1114-1118`), Inspect sends no ``thinking`` field,
    the model reasons anyway, and `display` defaults to ``"omitted"`` on Claude
    4.7+. Observed live on the same prompt (§0 v8): one thinking block whose
    signature runs 6,272 characters, ``.summary`` **0 characters**, and
    `ModelUsage.reasoning_tokens` **1**. The judge reads nothing and the cost
    record says the model barely thought. Nothing errors and nothing warns.

    It is a field here, rather than only a task argument, because the digest is
    the only thing that records what a corpus was generated at — and a convention
    cannot guard a failure that produces no error (§7, §0 v6). ``None`` is still
    accepted, explicitly, for the `mockllm` runs that have no reasoning channel
    to empty — a value a caller chooses, which is the opposite of a default."""

    max_output_tokens: int | None = None
    """**M_max** — the model's maximum output, or ``None`` to read it from
    `model/_model_data/*.yml` at run time.

    ``None`` is the intended setting: §6.3 says M_max is read and never assumed.
    A value pins it, which is what a model absent from the info DB needs —
    resolution **raises** rather than defaulting, because a defaulted number is
    the §7 failure shape.

    It is a ceiling rather than a budget. Measurement says thinking does not
    expand to fill it: the same task consumed ~8,200 thinking tokens whether
    offered 10,000 or 30,000, and finished on ``end_turn`` (§0 v7)."""

    token_limit: int | None = None
    """`Task(token_limit=)` — a runaway guardrail set well above expected use,
    never a budget (§5). Recorded here because it can end a trajectory, and a
    trajectory that ends differently is a different trajectory."""

    solver_name: str = "research_agent"
    """The registered `@solver` factory this configuration drives."""

    @property
    def digest(self) -> str:
        return sha256_digest(self.model_dump(mode="json"))

    @property
    def short(self) -> str:
        return short_digest(self.model_dump(mode="json"))


class RunProvenance(BaseModel):
    """Every provenance input for one run, in one object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instances_name: str
    instances_sha256: str
    instances_version: str
    """``instances_sha256[:16]`` — the value passed to ``Task(version=)``."""

    catalogue_release: str | None = None
    catalogue_sha256: str | None = None
    """``None`` on a generation run: generation carries no scorer options at all,
    which is what makes trajectory logs rubric-free."""

    judges: list[JudgeConfig] = Field(default_factory=list)
    solver: SolverConfig | None = None
    """``None`` on a scoring run: the scoring pass has no turn loop."""

    renderer_version: str = RENDERER_VERSION
    framework_version: str = Field(default_factory=framework_version)
    dataset_snapshot: DatasetSnapshot = Field(default_factory=DatasetSnapshot)
    harness_version: str = HARNESS_VERSION

    @classmethod
    def build(
        cls,
        instances: InstanceSet,
        *,
        catalogue: Catalogue | None = None,
        judges: Sequence[JudgeConfig] = (),
        solver: SolverConfig | None = None,
        dataset_snapshot: DatasetSnapshot | None = None,
        renderer_version: str = RENDERER_VERSION,
    ) -> RunProvenance:
        """Assemble provenance from the objects a run already has."""
        snapshot = dataset_snapshot or DatasetSnapshot(
            release=instances.dataset_snapshot or UNKNOWN_SNAPSHOT
        )
        return cls(
            instances_name=instances.name,
            instances_sha256=instances.digest,
            instances_version=instances.version,
            catalogue_release=None if catalogue is None else catalogue.release,
            catalogue_sha256=None if catalogue is None else catalogue.digest,
            judges=list(judges),
            solver=solver,
            renderer_version=renderer_version,
            dataset_snapshot=snapshot,
        )

    def to_task_metadata(self) -> dict[str, Any]:
        """The `Task(metadata=)` payload.

        ``instances_sha256`` is lifted to the top level as well as nested: it is
        the one field named directly by the design documents and by
        ``Task(version=)``, and a top-level key survives a summary-level read.
        ``solver_sha256`` is lifted for the same reason — §7 names it as the
        carrier for the solver axis, and it is what the analysis layer compares
        before reporting a cap-policy change as a model result.

        **``reasoning_effort`` is lifted beside the solver digest, and beside is
        the whole of it** (§7). Effort is a per-model choice: two models at two
        efforts differ on the solver digest, so the layer meets a digest
        difference it cannot decompose — on the one axis where the headline
        result *is* the per-model comparison. A digest is not decomposable by
        construction, so the fix is a second carrier the layer can hold
        explicitly, not a narrower digest.

        Three things this deliberately does not do. It does **not** add a field
        to :class:`RunProvenance`, so no record already written moves its digest
        — the same discipline §7 records for the model axis at v38. It does
        **not** touch :attr:`SolverConfig.digest`, which still covers effort:
        the value is carried twice on purpose, once where it is compared and
        once where it can be read. And it does not fire anything: §7 records the
        condition as not yet reached, both models running at ``xhigh``, so this
        is the carrier put in place ahead of the comparison that needs it (§11).

        The key is absent, not ``None``, when no ``SolverConfig`` was supplied —
        a task with no solver record must not read as one that ran at no effort,
        which is the value §7 calls the one that produces no error at all.
        """
        metadata: dict[str, Any] = {
            "instances_sha256": self.instances_sha256,
            PROVENANCE_KEY: self.model_dump(mode="json"),
        }
        if self.solver is not None:
            metadata["solver_sha256"] = self.solver.digest
            metadata["reasoning_effort"] = self.solver.reasoning_effort
        return metadata

    @property
    def digest(self) -> str:
        """Digest over the whole provenance record."""
        return sha256_digest(self.model_dump(mode="json"))

    def unpinned(self) -> list[str]:
        """Axes still carrying a placeholder.

        **A test-only introspection helper. It has no production caller by
        decision, not by oversight** (§7, §0 v24). §7 left open whether to wire
        this into the task-build path as a warning; it is ruled here as the
        helper it already is, and must not be wired in.

        The reason is the one §7 used to delete ``framework_commit_pinned``: a
        permanently-on warning is not a warning. Its two clauses are not alike.
        The renderer clause is **live** — a renderer version carrying the
        ``-unimplemented`` suffix is a real state that can start and stop being
        true. The snapshot clause is **permanently true**, and stays so until
        OneKGPd's tool surface returns a dataset identifier, which is a server
        change decided in §2 with its content undecided (§2, `foundations.md`
        §4.4). Wiring this into a run path would therefore print on every run
        until that change lands — reproducing exactly the defect that deletion
        removed.

        What it is for: asking a `RunProvenance` in hand which axes are still
        placeholders. That is a question tests ask and a run does not, because a
        run cannot act on the answer.

        Returns:
            Names of fields that have not been given a real value yet, in a fixed
            order. Empty means every axis is pinned — which, per the clause above,
            no run produces today.
        """
        missing: list[str] = []
        if self.renderer_version.endswith("-unimplemented"):
            missing.append("renderer_version")
        if not self.dataset_snapshot.is_pinned():
            missing.append("dataset_snapshot.release")
        return missing


class ProvenanceMismatch(BaseModel):
    """One axis on which two runs differ."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: str
    left: Any = None
    right: Any = None


def compare_provenance(
    left: RunProvenance, right: RunProvenance
) -> list[ProvenanceMismatch]:
    """Every axis on which two runs differ.

    The analysis layer calls this before any cross-round or cross-model
    comparison and refuses to proceed on an axis it did not declare it was
    varying. "The B1 rate rose from 0.4 to 0.7" is a statement about a model only
    when instances, catalogue, judges and renderer are all pinned or explicitly
    varied.

    Args:
        left: One run's provenance.
        right: The other's.

    Returns:
        The differing axes, empty when the two are comparable on every axis.
    """
    mismatches: list[ProvenanceMismatch] = []

    def differ(axis: str, a: Any, b: Any) -> None:
        if a != b:
            mismatches.append(ProvenanceMismatch(axis=axis, left=a, right=b))

    differ("instances", left.instances_sha256, right.instances_sha256)
    differ("catalogue", left.catalogue_sha256, right.catalogue_sha256)
    differ(
        "solver",
        None if left.solver is None else left.solver.digest,
        None if right.solver is None else right.solver.digest,
    )
    differ("renderer", left.renderer_version, right.renderer_version)
    differ("framework_version", left.framework_version, right.framework_version)
    differ(
        "dataset_snapshot",
        left.dataset_snapshot.digest,
        right.dataset_snapshot.digest,
    )
    differ(
        "judges",
        [j.digest for j in left.judges],
        [j.digest for j in right.judges],
    )
    return mismatches


def task_version(instances: InstanceSet) -> str:
    """The value for ``Task(version=)``.

    `eval_set` task identity does not include the dataset, so this is the only
    thing that makes a rotated instance set look like new work.
    """
    return instances.version


def task_metadata(
    instances: InstanceSet,
    *,
    catalogue: Catalogue | None = None,
    judges: Sequence[JudgeConfig] = (),
    solver: SolverConfig | None = None,
    dataset_snapshot: DatasetSnapshot | None = None,
) -> dict[str, Any]:
    """The `Task(metadata=)` payload for a run over ``instances``."""
    return RunProvenance.build(
        instances,
        catalogue=catalogue,
        judges=judges,
        solver=solver,
        dataset_snapshot=dataset_snapshot,
    ).to_task_metadata()


def provenance_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> RunProvenance | None:
    """Recover a :class:`RunProvenance` from a log's task metadata.

    :meth:`RunProvenance.to_task_metadata` nests the whole record under
    :data:`PROVENANCE_KEY`, and `eval()` copies ``Task(metadata=)`` verbatim into
    ``EvalSpec.metadata``, so this is the inverse of the generation-side write.

    Args:
        metadata: ``log.eval.metadata``, or ``None``.

    Returns:
        The record, or ``None`` when the log carries none — a log produced
        outside :func:`~genomics_harness.dataset.task_kwargs`.

    Raises:
        pydantic.ValidationError: The key is present but is not a provenance
            record. Refusing beats reading a partial one: every axis this record
            carries exists to make a comparison **fail**, and a field silently
            defaulted is a field two runs will agree on.
    """
    payload = (metadata or {}).get(PROVENANCE_KEY)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"task metadata {PROVENANCE_KEY!r} is {type(payload).__name__}, not a "
            "provenance record"
        )
    return RunProvenance.model_validate(dict(payload))


def provenance_sidecar_path(log_location: Path | str) -> Path:
    """Where a log's provenance sidecar lives.

    Beside the log and named from it, so the pair travels together and a
    directory listing shows which scoring logs have a provenance record and
    which do not.
    """
    return Path(f"{log_location}{PROVENANCE_SIDECAR_SUFFIX}")


def write_run_provenance(provenance: RunProvenance, path: Path | str) -> Path:
    """Write a provenance record as its own file.

    Serialised through :func:`~genomics_harness.digest.canonical_json`, the same
    function the digests are computed over, so the bytes on disk are a
    deterministic function of the record and two identical runs produce
    byte-identical sidecars.

    Args:
        provenance: The record to write.
        path: The destination, created along with any missing parents.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        canonical_json(provenance.model_dump(mode="json")), encoding="utf-8"
    )
    return destination


def read_run_provenance(path: Path | str) -> RunProvenance:
    """Read a provenance record written by :func:`write_run_provenance`.

    Args:
        path: The sidecar, or the log it sits beside — a path that does not end
            in :data:`PROVENANCE_SIDECAR_SUFFIX` is resolved through
            :func:`provenance_sidecar_path`.

    Returns:
        The record.

    Raises:
        FileNotFoundError: There is no sidecar. Loud on purpose: a missing
            record is the state §5.1 of `cleanups-august-2026.md` measured, and
            an analysis layer that treated it as "no differences" would permit
            exactly the comparison the record exists to refuse.
    """
    location = Path(path)
    if not location.name.endswith(PROVENANCE_SIDECAR_SUFFIX):
        location = provenance_sidecar_path(location)
    if not location.exists():
        raise FileNotFoundError(
            f"no provenance sidecar at {location}. A scoring log with no "
            "sidecar carries no provenance axes, so its verdicts cannot enter "
            "a report or be compared across runs. Re-judge the trajectory "
            "through genomics_harness.scoring.run_judges, which writes the "
            "sidecar beside each scoring log it produces, or remove the log."
        )
    return RunProvenance.model_validate(
        json.loads(location.read_text(encoding="utf-8"))
    )


def catalogue_provenance(catalogue: Catalogue) -> dict[str, str]:
    """The catalogue identity a scorer records alongside its verdicts."""
    return {
        "catalogue_release": catalogue.release,
        "catalogue_sha256": catalogue.digest,
        "catalogue_short": short_digest(catalogue.model_dump(mode="json")),
    }


# -- the cross-axis refusal -------------------------------------------------
#
# Moved here from the retired per-evaluation analysis layer at v40 (§14.34,
# §0 v40). It never was an aggregation: it reads two `RunProvenance` records
# and refuses, so it belongs beside `compare_provenance`, which it wraps.
#
# **Combining is not comparing** (§5, §0 v34). `assert_comparable` refuses two
# results differing on the instances axis, which is the axis a *combination*
# varies by design; `report.assert_combinable` is that other operation, with
# its own axis tuple and its own refusal. Neither reuses the other's escape
# hatch, and `test_report.py` pins the two tuples apart.


COMPARISON_AXES: tuple[str, ...] = (
    "instances",
    "catalogue",
    "solver",
    "renderer",
    "framework_version",
    "dataset_snapshot",
    "judges",
)
"""Every axis :func:`~genomics_harness.provenance.compare_provenance` reports.

Named here so :func:`assert_comparable` can reject a misspelled declaration
rather than silently refusing on an axis the caller thought they had permitted.
Pinned against `compare_provenance`'s actual output by a test, because a literal
list of another function's behaviour goes stale in exactly one direction: a new
axis appears, nothing here mentions it, and a comparison across it is refused
with a message naming an axis the caller cannot declare.
"""


class ComparabilityError(ValueError):
    """Two results differ on an axis the caller did not declare as varied."""


def assert_comparable(
    left: RunProvenance,
    right: RunProvenance,
    *,
    varying: Iterable[str] = (),
) -> list[ProvenanceMismatch]:
    """Refuse a comparison across an axis nobody declared.

    "The B1 rate rose from 0.4 to 0.7" is a statement about a model only when
    instances, catalogue, solver, renderer and **judges** are pinned or
    explicitly varied. The judges axis is the one a scoring run is the only
    witness to, and it reaches this function through the provenance sidecar
    :func:`~genomics_harness.scoring.run_judges` writes — there is no field of a
    scoring log that carries it.

    Refusing is a **raise**, not a warning: a warning on a comparison that has
    already been made is a note attached to a number someone will quote.

    Args:
        left: One run's provenance.
        right: The other's.
        varying: Axes the caller is deliberately varying, from
            :data:`COMPARISON_AXES`.

    Returns:
        The mismatches on the declared axes — what actually varied, so a report
        can say so.

    Raises:
        ComparabilityError: The two differ on an undeclared axis.
        KeyError: ``varying`` names something that is not an axis.
    """
    declared = set(varying)
    unknown = sorted(declared - set(COMPARISON_AXES))
    if unknown:
        raise KeyError(
            f"{unknown} are not comparison axes; known: {list(COMPARISON_AXES)}"
        )

    mismatches = compare_provenance(left, right)
    undeclared = [m for m in mismatches if m.axis not in declared]
    if undeclared:
        axes = sorted(m.axis for m in undeclared)
        raise ComparabilityError(
            f"the two runs differ on {axes}, which was not declared as varied "
            f"(declared: {sorted(declared)}). A difference in verdicts across "
            "an uncontrolled axis is not a result about the model"
        )
    return mismatches
