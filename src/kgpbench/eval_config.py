# Copyright 2026 Dnaerys Pty Ltd
# SPDX-License-Identifier: Apache-2.0

"""The eval-level configuration, and the settings it may not carry.

`inspect eval` offers an option for four of the settings
:class:`~kgpbench.provenance.SolverConfig` owns, and both may be given on one
command line. That is the state this module refuses. Two further options name
the epoch count and its reducer, which the **task** owns rather than the solver
configuration; they are refused here too, on the terms the last section below
sets out.

**Why presence and not disagreement, for those four.** Each of them also arrives through an
``INSPECT_EVAL_*`` environment variable and through a ``--generate-config``
file, on the identical channel and indistinguishable from a typed option. Under
an equality rule a stale exported variable holding today's value passes
silently and stops agreeing the moment the task argument changes; under a
presence rule there is one read and nothing compared, and the invisible case is
caught with the visible one. So a value that *agrees* with the task's is refused
too, and so is one whose option is currently inert.

**Why the refusals say what carries the value and never that the operator typed
it.** The same variables mean a refusal can name an option nobody passed. The
message names the carriers and leaves the operator to find which of them is
set.

**Three sites, because none of them is visible at the others.**

* :func:`assert_no_generate_config_collision` runs **inside the ``@task``
  body**, where `active_generate_config()` returns the eval-level
  `GenerateConfig` alone — the model is initialised before the task function is
  called (`_eval/eval.py:788-794`, `:1900`), so the task's own config does not
  yet exist to be merged with it. Presence is exact: the baseline read is empty
  rather than defaulted, and none of the options carries a click ``default=``.
  A refusal there writes no log directory at all.
* :func:`assert_no_limit_collision` runs at **solve time**, where the two sample
  limits read reconciled — `_eval/run.py:252-256` and `:279-285` overwrite the
  `Task` attribute with the eval-level value when one was given. They are not
  visible in the task body. A refusal there costs one errored sample per sample
  and epoch, with no model event.
* :func:`assert_no_epoch_collision` runs at **solve time** as well, and for the
  same reason one line further on: the epoch reconciliation is
  `_eval/run.py:252-256`, three lines above the token limit's.

**A `TaskStart` hook is not a fourth site.** `_emit_to_all` catches every
exception a hook raises and logs a warning (`hooks/_hooks.py:1054-1060`); the
one type it re-raises is `LimitExceededError`, so refusing there means writing a
limit's vocabulary into the completeness axis for a configuration fault.

**The epoch options are here too, and are compared rather than counted.**
``repeats`` is a task argument rather than a `SolverConfig` field, so the four
above cannot cover it, and the eval-level count is *never* readable on its own:
the `EvalConfig` that would carry it is built after the task function has
already run (`_eval/eval.py:788`, `:889`), and the reconciliation then assigns
in both directions (`_eval/run.py:252-256`), leaving no carrier that records
which side supplied the number. What the reconciliation does leave is
`Task.epochs` holding the **effective** count, on the very object the builder
returned — so :class:`EpochClaim` keeps the declared count beside a reference to
that task, and :func:`assert_no_epoch_collision` compares the two. Inequality is
the trigger and presence is not: an eval-level count equal to ``repeats`` is two
claimants agreeing, and an absent one is copied across from the task, so both
read equal and neither fires.

``--effort`` and ``--reasoning-tokens`` are outside it too, and deliberately:
they name `GenerateConfig` fields this harness does not carry, and the scope is
the fields `SolverConfig` owns rather than every option that touches thinking
(`specification.md` §7, §14.45).

**`active_generate_config` has no public route.** `inspect_ai.model` does not
re-export it and no other module reaches the context variable it reads. It is on
§7's inventory of the private-framework internals this package imports, which is
where that list is kept and counted, and it is the only reason
`inspect_ai.model._generate_config` is on it. It is on the re-audit surface a
version bump checks (§7).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, NamedTuple

from inspect_ai import Epochs, Task
from inspect_ai._util.registry import registry_unqualified_name
from inspect_ai.model._generate_config import active_generate_config
from inspect_ai.util import sample_limits

from .provenance import SolverConfig

__all__ = [
    "EPOCH_OPTIONS",
    "GENERATE_CONFIG_OPTIONS",
    "LIMIT_OPTIONS",
    "SOLVER_OWNED_OPTIONS",
    "EpochClaim",
    "EvalConfigCollision",
    "assert_no_epoch_collision",
    "assert_no_generate_config_collision",
    "assert_no_limit_collision",
]


class EvalOption(NamedTuple):
    """One eval-level option naming a setting `SolverConfig` owns."""

    solver_field: str
    """The :class:`~kgpbench.provenance.SolverConfig` field it names."""

    carrier: str
    """Where the value is read at run time — a `GenerateConfig` field name for
    :data:`GENERATE_CONFIG_OPTIONS`, a `SampleLimits` attribute name for
    :data:`LIMIT_OPTIONS`."""

    flag: str
    """The `inspect eval` option."""

    envvar: str
    """Its environment variable — the same channel, and the reason a refusal
    never says the operator typed anything."""

    remedy: str
    """What to do instead, in the operator's own vocabulary."""


GENERATE_CONFIG_OPTIONS: Final[tuple[EvalOption, ...]] = (
    EvalOption(
        solver_field="reasoning_effort",
        carrier="reasoning_effort",
        flag="--reasoning-effort",
        envvar="INSPECT_EVAL_REASONING_EFFORT",
        remedy="the effort comes from -T reasoning_effort=",
    ),
    EvalOption(
        solver_field="max_output_tokens",
        carrier="max_tokens",
        flag="--max-tokens",
        envvar="INSPECT_EVAL_MAX_TOKENS",
        remedy="M_max comes from -T max_output_tokens=",
    ),
)
"""The two visible inside the ``@task`` body, read off the eval-level
`GenerateConfig`."""

LIMIT_OPTIONS: Final[tuple[EvalOption, ...]] = (
    EvalOption(
        solver_field="token_limit",
        carrier="token",
        flag="--token-limit",
        envvar="INSPECT_EVAL_TOKEN_LIMIT",
        remedy="the token bound comes from the solver configuration's token_limit",
    ),
    EvalOption(
        solver_field="turn_cap",
        carrier="turn",
        flag="--turn-limit",
        envvar="INSPECT_EVAL_TURN_LIMIT",
        remedy="the turn bound comes from the solver configuration's turn_cap",
    ),
)
"""The two visible at solve time, read off the reconciled sample limits."""

SOLVER_OWNED_OPTIONS: Final[tuple[EvalOption, ...]] = (
    GENERATE_CONFIG_OPTIONS + LIMIT_OPTIONS
)
"""All four, in one tuple, so a test can hold the scope against §14.45's list."""


class EpochOption(NamedTuple):
    """One eval-level option naming a setting the **task** owns."""

    flag: str
    """The `inspect eval` option."""

    envvar: str
    """Its environment variable — the same channel, and the reason a refusal
    never says the operator typed anything."""

    remedy: str
    """What to do instead, in the operator's own vocabulary."""


EPOCH_COUNT_OPTIONS: Final[tuple[EpochOption, ...]] = (
    EpochOption(
        flag="--epochs",
        envvar="INSPECT_EVAL_EPOCHS",
        remedy="the repeat count comes from -T repeats=",
    ),
)
"""What sets an eval-level epoch count. Refused on **inequality** with the
task's ``repeats``, because an equal count is two claimants agreeing and an
absent one is copied across from the task."""

EPOCH_REDUCER_OPTIONS: Final[tuple[EpochOption, ...]] = (
    EpochOption(
        flag="--epochs-reducer",
        envvar="INSPECT_EVAL_EPOCHS_REDUCER",
        remedy=(
            "there is nowhere to move it to, because every repeat is reported "
            "on its own and none is collapsed"
        ),
    ),
)
"""What sets an eval-level epoch reducer. Refused on **presence**: a reducer
over per-epoch scores is the one thing this harness cannot have, which is what
``Epochs(repeats, [])`` exists to suppress (§5, §6.2 invariant 1). It wins the
reconciliation outright, so this is a real collapse held off by the shape of
one score value rather than by the framework — see
:func:`assert_no_epoch_collision`.

``--no-epochs-reducer`` is not here, and its absence is measured rather than
chosen — same place."""

EPOCH_OPTIONS: Final[tuple[EpochOption, ...]] = (
    EPOCH_COUNT_OPTIONS + EPOCH_REDUCER_OPTIONS
)
"""Both, in one tuple, so a test can hold the scope."""


class EpochClaim:
    """What a generation task declared about epochs, beside the object the
    framework rewrites.

    **Why a reference to the task is the mechanism.** The eval-level epoch count
    is not readable on its own at any point inside a run: the `EvalConfig` that
    would carry it is constructed *after* the task function has already returned
    (`_eval/eval.py:788` resolves tasks, `:889` builds the config), and the
    reconciliation that follows assigns in both directions
    (`_eval/run.py:252-256`) — the task's count copied into the eval config when
    no eval-level count was given, the eval-level count written over
    ``Task.epochs`` when one was. Neither carrier then records which side
    supplied the number.

    What that reconciliation *does* leave is the **effective** count on
    ``Task.epochs``, on the very object :func:`~kgpbench.tasks
    .build_generation_task` returned — measured: the framework resolves a task
    without copying it, by the registered route and the direct one alike. So the
    declared count has to be kept somewhere else, and this is that somewhere.
    The pair is the whole mechanism: neither half is a refusal on its own.

    ``Task.epochs_reducer`` is rewritten by the same three lines and is read the
    same way.

    Args:
        declared: The `Epochs` the task was built with. Held whole rather than
            decomposed, so the comparison is against what the builder actually
            declared rather than against a second copy of it.
    """

    __slots__ = ("_task", "declared")

    def __init__(self, declared: Epochs) -> None:
        self.declared = declared
        self._task: Task | None = None

    def bind(self, task: Task) -> None:
        """Bind the task built around this claim. Called once, by the builder."""
        self._task = task

    @property
    def bound(self) -> bool:
        """Whether a task has been bound. Unbound, nothing can be compared."""
        return self._task is not None

    @property
    def effective_epochs(self) -> int | None:
        """``Task.epochs`` as the framework left it, or ``None`` if unbound."""
        return None if self._task is None else self._task.epochs

    @property
    def effective_reducer(self) -> list[str]:
        """The reducers in force, by unqualified registry name."""
        if self._task is None or self._task.epochs_reducer is None:
            return []
        reducers = self._task.epochs_reducer
        if not isinstance(reducers, list):
            reducers = [reducers]
        return [registry_unqualified_name(reducer) for reducer in reducers]


class EvalConfigCollision(RuntimeError):
    """The eval-level configuration names a setting one of this harness's owns.

    Raised before any model call, from all three sites above. One exception type
    across them because they are one operator-facing statement: the harness will
    not run a configuration that travels by two mechanisms at once.

    The trigger is **presence** for the four settings `SolverConfig` owns and for
    the epoch reducer, and **inequality** for the epoch count — because that is
    the one of the six whose eval-level value is never its own at any readable
    point, so an equal count and no count at all are the same read
    (:func:`assert_no_epoch_collision`).

    `RuntimeError` rather than `ValueError` because no argument to any function
    here is wrong — the fault is in a channel the caller did not pass through —
    and because :func:`~kgpbench.tasks.generation` already reserves `ValueError`
    for the values it was handed.
    """


def _refusal(
    carried: list[str],
    options: Sequence[EvalOption] | Sequence[EpochOption],
    files: list[str],
    *,
    owner: str = "the solver configuration",
) -> str:
    """One message for every site: condition, remedy, then the carriers.

    Rationale is in this module's docstring rather than in the string, per §16.
    The carriers are part of the condition and not of the rationale: an operator
    who does not know that an environment variable and a config file set the
    same field cannot act on the refusal at all. Neither of §16's two exceptions
    applies — there is no quantity to guide, and the failure this prevents is
    not a silent one.

    ``owner`` is the one thing that varies across the three sites, because the
    epoch settings are the task's rather than the solver configuration's. The
    sentence shape does not vary: the harness will not run a configuration that
    travels by two mechanisms at once, whichever of its own layers owns it.
    """
    settings = _join(carried)
    remedies = "; ".join(option.remedy for option in options)
    carriers = _join(
        [part for option in options for part in (option.flag, option.envvar)] + files
    )
    subject = "these settings" if len(options) > 1 else "this setting"
    clear = "Clear them" if len(options) > 1 else "Clear it"
    return (
        f"the eval-level configuration carries {settings}, and {owner} owns "
        f"{subject}. {clear} there: {remedies}. "
        f"{carriers} all set the eval-level value"
    )


def _join(parts: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — the last join is a word."""
    if len(parts) < 2:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def assert_no_generate_config_collision() -> None:
    """Refuse an eval-level ``reasoning_effort`` or ``max_tokens``.

    Call from inside a ``@task`` body, where the read is the eval-level
    `GenerateConfig` alone and a refusal writes nothing to disk.

    Raises:
        EvalConfigCollision: Either field is set at eval level, whatever its
            value and whether or not it agrees with the task's.
    """
    config = active_generate_config()
    present = [
        option
        for option in GENERATE_CONFIG_OPTIONS
        if getattr(config, option.carrier, None) is not None
    ]
    if not present:
        return
    carried = [
        f"{option.carrier}={getattr(config, option.carrier)!r}" for option in present
    ]
    raise EvalConfigCollision(
        _refusal(carried, present, ["--generate-config", "--run-config"])
    )


def assert_no_limit_collision(config: SolverConfig) -> None:
    """Refuse an eval-level token limit or turn limit.

    Call from inside a solver, where both limits read reconciled and the
    harness's own values are what ``config`` supplies.

    **The two are detected differently, because the harness sets one and not the
    other.** ``token_limit`` reaches `Task(token_limit=)`, so the reconciled
    value is the operator's exactly when it differs from ``config``'s — which is
    presence while ``SolverConfig.token_limit`` stays ``None``, and inequality
    if a round ever pins it. The harness sets the framework's **turn** limit
    nowhere — ``turn_cap`` bounds the solver's own loop instead — so any turn
    limit in force is the operator's and presence is exact with nothing to
    compare.

    Args:
        config: The solver configuration this run was built with.

    Raises:
        EvalConfigCollision: Either limit is set at eval level.
    """
    limits = sample_limits()
    effective: dict[str, int | float | None] = {
        "token": limits.token.limit,
        "turn": limits.turn.limit,
    }
    ours: dict[str, int | None] = {"token": config.token_limit, "turn": None}
    present = [
        option
        for option in LIMIT_OPTIONS
        if effective[option.carrier] != ours[option.carrier]
    ]
    if not present:
        return
    carried = [
        f"a {option.carrier} limit of {effective[option.carrier]!r}"
        for option in present
    ]
    raise EvalConfigCollision(_refusal(carried, present, ["--run-config"]))


def assert_no_epoch_collision(claim: EpochClaim | None) -> None:
    """Refuse an eval-level epoch count that differs, and a reducer of any kind.

    Call from inside a solver, where the reconciliation has already happened and
    ``claim`` holds both halves of the comparison.

    **The count is refused on inequality and the reducer on presence, and the
    difference is not an inconsistency.** An eval-level count *equal* to the
    task's is two claimants naming the same number, and the run is exactly what
    the operator asked for either way; where no count is given the framework
    copies the task's across, so that reads equal too and falls out of the same
    comparison with no branch of its own. A reducer has no such reading: this
    harness declares ``Epochs(repeats, [])`` precisely to suppress reduction,
    per-repeat rows are reported individually and never collapsed (§5, §6.2
    invariant 1), so any reducer in force is one the harness must not honour.

    **And the eval-level reducer is not inert, which is measured rather than
    assumed.** ``log.reductions`` is empty on a generation run under
    ``--epochs-reducer mean``, which reads as this task's declaration surviving.
    It does not survive: the eval-level reducer replaces ``Task.epochs_reducer``
    outright (`_eval/run.py:258-261`) and reaches
    ``EvalSpec.config.epochs_reducer``, so the header records a reduction policy
    this harness forbids. What produces no reduction is that `render_capture`'s
    score value is a **string** with nothing to collapse; against a numeric score
    the same configuration reduces, measured. So the collapse §5 forbids is one
    score-value shape away rather than unreachable — and would be refused here
    either way, because inertness is a property of a release and the rule is
    about what an operator has stated an intent to do.

    **``--no-epochs-reducer`` is not refused, because it cannot be seen.** It
    reaches `eval()` as ``Epochs(n, [])`` (`_cli/eval.py:1901-1908`), and
    ``[]`` is what this task declares, so the reconciliation writes the value
    that was already there: measured, the bound task,
    ``EvalSpec.config.epochs_reducer`` and the written log all read the same
    **value** under it as under no option at all. Only object identity differs,
    and that is the rule deliberately not written — were `Task.__init__` ever to
    copy the list it is handed, an identity rule would refuse every ordinary
    run, and a rule that can fail *that* way is worse than a gap it is honest
    about. ``--no-epochs-reducer`` also cannot arrive without ``--epochs``,
    which the count comparison does see.

    Args:
        claim: What the task declared, bound to the task. ``None`` — or a claim
            nothing bound — is a solver built outside
            :func:`~kgpbench.tasks.build_generation_task`, which is every route
            with no `Task` to compare against; there is nothing to check and
            nothing is checked.

    Raises:
        EvalConfigCollision: The epoch count in force is not the task's, or a
            reducer is in force at all.
    """
    if claim is None or not claim.bound:
        return

    carried: list[str] = []
    options: list[EpochOption] = []

    effective = claim.effective_epochs
    if effective != claim.declared.epochs:
        carried.append(
            f"an epoch count of {effective!r} where this task declares "
            f"{claim.declared.epochs!r}"
        )
        options.extend(EPOCH_COUNT_OPTIONS)

    declared = {
        registry_unqualified_name(reducer) for reducer in (claim.declared.reducer or [])
    }
    undeclared = [
        reducer for reducer in claim.effective_reducer if reducer not in declared
    ]
    if undeclared:
        word = "reducers" if len(undeclared) > 1 else "reducer"
        carried.append(f"the epoch {word} {_join([repr(r) for r in undeclared])}")
        options.extend(EPOCH_REDUCER_OPTIONS)

    if not options:
        return
    raise EvalConfigCollision(
        _refusal(carried, options, ["--run-config"], owner="this task")
    )
