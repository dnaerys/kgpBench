"""Named instance sets, declared in a config file and resolved at task construction.

The harness must be publishable, and runnable, with **no instance module present
at all** (specification §16). A composition module that imported its instances
could not ship without them, so nothing here names one: the sets live in a
committed TOML file, as lists of module **paths**, and the only thing that
crosses into the registered `@task` is a set *name* (§4, §0 v31).

```toml
[sets]
s1 = ["your_package.some_instance"]
s2 = ["your_package.some_instance", "your_package.another_instance"]
```

Each named module supplies one symbol, :data:`INSTANCE_SYMBOL` — one contract,
applied uniformly, so a module is either a member or it is not and there is no
per-module wiring to keep honest.

**What raises and what warns.** Selection is explicit: a missing config, a
missing or unknown set name, or a module in the *selected* set that will not
import or lacks the symbol, all raise :class:`CompositionError`. A forgotten
selector must fail loudly rather than silently re-pay for every instance in some
default set. Modules in sets that were **not** selected are checked with
:func:`importlib.util.find_spec` — existence without execution — and a failure
warns, which catches the common staleness (a deleted or renamed module) across
the whole config on any run. That limit is chosen: a module that exists but is
broken warns only when it is selected, because importing every set's modules to
close the gap would execute code the operator did not ask to run.

**The print is a requirement, not a convenience** (§4). Resolution happens at
task construction, before any model call, and prints the resolved config path
and the resolved members to the console — never into the log, so the published
record stays neutral. A `mockllm` run therefore exposes a wrong or stale config
at zero cost, which is the first step of every round (§6.5).

**Order does not matter and the name is not digested.** A set is its members, so
members are sorted by id before the :class:`~genomics_harness.instances.InstanceSet`
is built and the digest is order-independent — two orderings of one set give one
``Task(version=)``. The digest covers the resolved members and neither the config
path nor the set name, so an edited config that changes membership changes the
digest and an unchanged one reproduces it.

**The costs are accepted rather than mitigated** (§0 v31): dynamic import is
invisible to `mypy`, the config is not type-checked, and a bad config is a
runtime error. What the module can do is validate at run time, which is what the
`isinstance` check below is: everything a static type would have promised is
asserted against the object that actually arrived.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import tomllib
import warnings
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from .instances import Instance, InstanceSet

__all__ = [
    "INSTANCE_SYMBOL",
    "SET_CONFIG_ENV",
    "SET_CONFIG_PATH",
    "CompositionError",
    "available_sets",
    "config_path",
    "load_set_config",
    "resolve_set",
]

SET_CONFIG_PATH: Final = Path(__file__).resolve().parent / "data" / "instance-sets.toml"
"""The default config, shipped inside the package as `data/instance-sets.toml`.

Inside the package and not beside it, for the reason the catalogue is
(`catalogue_file.py`): a path resolved from ``__file__`` holds for an installed
wheel and not only for this tree, and neither file has a working directory to be
relative to.
"""

SET_CONFIG_ENV: Final = "KGPBENCH_INSTANCE_SETS"
"""Environment variable overriding :data:`SET_CONFIG_PATH`.

An override rather than a task argument. A path in ``task_args`` reaches
``header.json`` and would record a value that looks like provenance and is not:
the same path yields different members after an edit, and §7 records observations
rather than claims. What is recorded instead is the digest over the members the
path resolved to.
"""

INSTANCE_SYMBOL: Final = "INSTANCE"
"""The one symbol a member module must define, bound to an `Instance`.

Uniform on purpose. A per-module symbol name in the config would be a second
thing to get wrong, and a module that does not define this one is not a member
however much else it defines.
"""

_SETS_TABLE: Final = "sets"


class CompositionError(Exception):
    """A set could not be resolved.

    One error type for every failure — missing config, unknown name, a module
    that will not import, a module without :data:`INSTANCE_SYMBOL` — because the
    caller's response to all of them is the same: fix the config or the module,
    and nothing has been paid for yet.
    """


def config_path(path: Path | None = None) -> Path:
    """The config this process reads.

    Args:
        path: An explicit path, which wins over everything. For tests and for a
            caller that already holds one.

    Returns:
        ``path`` if given, else the :data:`SET_CONFIG_ENV` override if set, else
        :data:`SET_CONFIG_PATH`.
    """
    if path is not None:
        return Path(path)
    override = os.environ.get(SET_CONFIG_ENV)
    if override:
        return Path(override)
    return SET_CONFIG_PATH


def load_set_config(path: Path | None = None) -> dict[str, list[str]]:
    """Read and validate the config's one table.

    Args:
        path: Overrides :func:`config_path`.

    Returns:
        Set name to list of module paths. An empty ``[sets]`` table gives an
        empty mapping, which is the harness-only release (§16) and not an error
        until something asks it to resolve a name.

    Raises:
        CompositionError: The file is missing, is not readable as TOML, has no
            ``[sets]`` table, or a set is not a list of strings.
    """
    resolved = config_path(path)
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise CompositionError(
            f"no instance-set config at {resolved}: {error}. Set "
            f"{SET_CONFIG_ENV} to a config file, or ship one at "
            f"{SET_CONFIG_PATH}"
        ) from error
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise CompositionError(
            f"{resolved} is not readable as TOML: {error}"
        ) from error

    table = parsed.get(_SETS_TABLE)
    if table is None:
        raise CompositionError(
            f"{resolved} has no [{_SETS_TABLE}] table; the schema is one table "
            f"mapping a set name to a list of instance module paths"
        )
    if not isinstance(table, dict):
        raise CompositionError(
            f"{resolved}: [{_SETS_TABLE}] is {type(table).__name__}, not a table"
        )

    sets: dict[str, list[str]] = {}
    for name, members in table.items():
        if not isinstance(members, list) or not all(
            isinstance(member, str) for member in members
        ):
            raise CompositionError(
                f"{resolved}: set {name!r} must be a list of module paths, "
                f"not {members!r}"
            )
        sets[name] = list(members)
    return sets


def available_sets(path: Path | None = None) -> list[str]:
    """The set names the config declares, sorted.

    Args:
        path: Overrides :func:`config_path`.

    Returns:
        Every declared name. Empty is legitimate (§16).

    Raises:
        CompositionError: As :func:`load_set_config`.
    """
    return sorted(load_set_config(path))


def resolve_set(
    name: str | None,
    *,
    path: Path | None = None,
    announce: bool = True,
) -> InstanceSet:
    """Resolve one named set into an `InstanceSet`, printing what it resolved to.

    Args:
        name: The set name. ``None`` raises: selection is explicit and there is
            no default set, because a wrong or forgotten selector that quietly
            ran something would re-pay for every instance in it.
        path: Overrides :func:`config_path`.
        announce: Print the resolved config path and members to the console.
            On by default — the print is what makes a `mockllm` run expose a bad
            config at zero cost (§4). It goes to stdout, never into the log.

    Returns:
        The set, members sorted by id, named for the config's set name.

    Raises:
        CompositionError: No name, an unknown name, a missing config, or a
            selected module that will not import, lacks
            :data:`INSTANCE_SYMBOL`, or binds it to something that is not an
            `Instance`.
    """
    resolved_path = config_path(path)
    sets = load_set_config(resolved_path)

    if name is None:
        raise CompositionError(
            f"no instance set selected; pass one of {sorted(sets)} "
            f"(from {resolved_path})"
        )
    if name not in sets:
        raise CompositionError(
            f"unknown instance set {name!r}; available: {sorted(sets)} "
            f"(from {resolved_path})"
        )

    _warn_on_unselected(sets, selected=name, path=resolved_path)

    members = [
        (module_path, _import_member(module_path, set_name=name))
        for module_path in sets[name]
    ]
    members.sort(key=lambda member: member[1].id)
    instances = [instance for _, instance in members]
    try:
        built = InstanceSet(
            name=name,
            dataset_snapshot=_shared_snapshot(instances),
            instances=instances,
        )
    except ValidationError as error:
        raise CompositionError(
            f"set {name!r} in {resolved_path} does not compose: {error}"
        ) from error

    if announce:
        _announce(built, members=members, path=resolved_path)
    return built


def _import_member(module_path: str, *, set_name: str) -> Instance:
    """Import one member module and take its `INSTANCE`.

    The type is checked against the object rather than declared, because
    `importlib` returns a module `mypy` cannot see into — validate at run time
    rather than contorting the types (§0 v31).
    """
    try:
        module = importlib.import_module(module_path)
    except Exception as error:  # any import failure is the same failure
        raise CompositionError(
            f"set {set_name!r} names {module_path!r}, which will not import: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not hasattr(module, INSTANCE_SYMBOL):
        raise CompositionError(
            f"set {set_name!r} names {module_path!r}, which defines no "
            f"{INSTANCE_SYMBOL}; every member module binds that one symbol to "
            f"its Instance"
        )
    candidate = getattr(module, INSTANCE_SYMBOL)
    if not isinstance(candidate, Instance):
        raise CompositionError(
            f"{module_path}.{INSTANCE_SYMBOL} is {type(candidate).__name__}, "
            f"not an Instance"
        )
    return candidate


def _warn_on_unselected(
    sets: dict[str, list[str]], *, selected: str, path: Path
) -> None:
    """Check every unselected set's modules for existence, without executing them.

    `find_spec` answers whether a module could be imported; it does not import
    it. So a renamed or deleted module is caught across the whole config on any
    run, and a module that exists but raises on import is not — that one warns
    only when it is selected, and closing the gap would mean running code the
    operator did not ask for (§4).
    """
    selected_members = set(sets[selected])
    for name, members in sorted(sets.items()):
        if name == selected:
            continue
        for module_path in members:
            if module_path in selected_members:
                continue
            try:
                detail = (
                    None
                    if importlib.util.find_spec(module_path) is not None
                    else "no module of that name"
                )
            except Exception as error:  # a parent package that will not import
                detail = f"{type(error).__name__}: {error}"
            if detail is not None:
                warnings.warn(
                    f"instance set {name!r} in {path} names {module_path!r}, "
                    f"which cannot be found ({detail}). It was not selected, so "
                    f"this run proceeds",
                    stacklevel=3,
                )


def _shared_snapshot(instances: list[Instance]) -> str | None:
    """The members' dataset snapshot when they agree on one, else ``None``.

    Derived rather than declared, because there is nowhere left to declare it:
    the config names modules and the set-level value used to be written beside
    the members in an instance module. Deriving it keeps
    :class:`~genomics_harness.provenance.RunProvenance` recording the snapshot
    the members were actually built against, which is what the
    ``dataset_snapshot`` comparison axis reads (§7). Members that disagree have
    no shared snapshot, and each instance still carries its own.
    """
    snapshots = {instance.dataset_snapshot for instance in instances}
    if len(snapshots) == 1:
        return snapshots.pop()
    return None


def _announce(
    instances: InstanceSet, *, members: list[tuple[str, Instance]], path: Path
) -> None:
    """Print the resolution to the console. Never to the log.

    The member **ids** are what §4 asks for; the module path each one resolved
    from is printed beside it because an id is a content digest nobody can
    eyeball, and the print exists to expose a wrong or stale config at a glance.
    """
    print(f"genomics_harness: instance set {instances.name!r} from {path}")
    for module_path, instance in members:
        print(f"genomics_harness:   {module_path} -> {instance.id}")
    if not members:
        print("genomics_harness:   (no members)")
