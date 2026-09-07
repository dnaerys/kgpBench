"""The composition layer — named sets in a config file, resolved at task construction.

Three things are being established.

**The harness runs with no instance present.** That is the point of the mechanism
(specification §16): the package must import, register its task and refuse
cleanly with an empty ``[sets]`` table, because the harness publishes while the
instance set is still being built. Nothing under ``src/genomics_harness`` may
import an instance module, and one test reads the source to say so.

**Selection is explicit and failures are loud.** A missing config, no set name,
an unknown name, or a selected module that will not import or carries no
``INSTANCE`` all raise, naming the sets that do exist. Only the *unselected*
sets warn, and only for a module that cannot be found at all — existence
checked with `find_spec`, which does not execute it.

**The digest is over the members.** Two orderings of one set give one
``Task(version=)``, and an unchanged config reproduces the digest.

**Nothing here pins the shipped config's contents.** `data/instance-sets.toml` is
operator-maintained — sets are added, rotated and retired as ordinary operation,
and §16's release shape ships it empty or with demo entries — so a test asserting
its membership or its digests would fail on normal use and on the shape the
harness is meant to publish in. The digest pins are over the synthetic fixture
instances in `harness_fixtures`, which is where an algorithm change is caught;
of the shipped file this module asserts only that it **resolves**, with no value
read off it. Instance ids are pinned, because an id is a digest over its
instance's own content and moves only when that content does — never under
operation on the config.

No network, no keys: `mockllm/model` where a run is needed at all.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput

from genomics_harness import (
    INSTANCE_SYMBOL,
    SET_CONFIG_ENV,
    SET_CONFIG_PATH,
    CompositionError,
    Instance,
    InstanceSet,
    SolverConfig,
    available_sets,
    build_generation_task,
    config_path,
    genomics_generation,
    load_set_config,
    resolve_set,
)

MCP_URL = "https://db.dnaerys.org/mcp"

# The fixture sets, pinned. Their content is fixed in `harness_fixtures`, so
# these values move only if the digest *algorithm* moves — which is the risk
# worth a pin, since the digest is `Task(version=)` and `eval_set` treats a
# moved version as new work (§6.2 invariant 3).
ALPHA_ID = "00000000000000aa"
BETA_ID = "00000000000000bb"
ONE_DIGEST = "9568dd9d0f6f1312c1e0a2ed5af9134127d12fb678a7172bbf7985eceef6d363"
BOTH_DIGEST = "0991c9ed72be7aaa95b0df401b93d1075ca151a1111fa4884c80c449c8d947ed"

PACKAGE = "genomics_harness"


@pytest.fixture
def shipped_package() -> tuple[Path, Path]:
    """This package's source directory and its `pyproject.toml`.

    Resolved from the imported package rather than from the working directory or
    from a repository root, so it holds wherever the suite runs and in **either**
    layout the source lives in — beside its siblings in the development tree, or
    alone in a checkout whose root is the package (§16). Nothing above the
    package root is reached: `pyproject.toml` sits two levels up from
    ``src/genomics_harness``, which is the package's own root and not its parent's.
    """
    import genomics_harness

    package = Path(genomics_harness.__file__).resolve().parent
    pyproject = package.parents[1] / "pyproject.toml"
    assert pyproject.is_file(), (
        f"no pyproject.toml at {pyproject}; an absent file is one of the "
        "failures these tests exist to catch, not a reason to skip"
    )
    return package, pyproject


# -- the shipped config: it resolves, and that is all that is asserted -------


def test_every_shipped_set_resolves() -> None:
    """The operator's file, checked for consistency and not for content.

    Every declared name resolves, every module imports, every ``INSTANCE`` is an
    `Instance`. No membership and no digest is read: those are the operator's to
    change, and §16 ships this file empty. An empty table passes here by having
    nothing to resolve, which is the correct answer for a harness-only release.
    """
    for name in available_sets():
        resolved = resolve_set(name, announce=False)
        assert resolved.name == name
        assert len(resolved) >= 1, f"set {name!r} declares no members"
        assert resolved.ids() == sorted(resolved.ids())
        for instance in resolved.instances:
            assert isinstance(instance, Instance)
            assert instance.has_opaque_id(), (
                f"{instance.id!r} in set {name!r} is not opaque, and "
                "EvalDataset.sample_ids publishes it (§12)"
            )


def _written_ids(resolved: InstanceSet) -> list[str]:
    """Members of ``resolved`` whose ``id`` is not the digest of their content.

    The collective identity predicate, in one place so that the assertion over
    the shipped sets and the mutation control over a fixture set run the *same*
    rule (§4, §0 v44, §0 v46).
    """
    return [
        instance.id
        for instance in resolved.instances
        if instance.id != instance.opaque_id
    ]


def test_every_instance_the_shipped_config_resolves_has_a_derived_id() -> None:
    """§4's collective assertion: ``id == opaque_id`` over the resolved sets.

    **Collective rather than per-instance, and that is the whole design.** §4
    rules the identity at *construction* and by no validator — an instance built
    through `Instance.with_derived_id` cannot carry a written id, and one written
    the other way is caught by nothing. Per-instance pins cover only the
    instances somebody remembered to pin; one assertion over what the config
    resolves covers every instance added later, which is the class the ruling
    leaves open.

    **It is not a validator by another route.** It reads the resolved shipped
    sets and nothing else, so `Instance.from_sample_metadata` still loads any log
    this harness has written, including one whose instance has since been edited
    — the reconstruction path §5's report layer and every scoring run depend on
    (§4).

    **It pins no value from the operator's file.** Which sets exist and which
    modules they name are read, never asserted; §15's rule is about pinning
    contents, and nothing here records a membership, a digest or a count.
    """
    names = available_sets()
    checked = 0
    for name in names:
        resolved = resolve_set(name, announce=False)
        offenders = _written_ids(resolved)
        assert offenders == [], (
            f"set {name!r} resolves instances whose id is not a digest of their "
            f"own content: {offenders}. An id written by hand passes every other "
            "gate until someone adds that instance's own pin (§4, §0 v44)"
        )
        checked += len(resolved.instances)

    # an empty `[sets]` table is the harness-only release shape and passes by
    # having nothing to resolve (§16); a non-empty one that checked nothing
    # would be this assertion passing vacuously, which is a different thing
    assert not names or checked >= 1


def test_the_collective_identity_assertion_fires_on_a_written_id(
    fixture_modules,
) -> None:
    """The mutation control: the rule above, run over an instance that breaks it.

    `FIXTURE_ALPHA` carries the written id ``00000000000000aa``, which is not the
    digest of its content — the synthetic-construction case §4 keeps available on
    purpose. So the fixture carries the difference by construction, and the
    predicate the shipped assertion runs is shown to fail on exactly the edit it
    exists to catch: an instance module writing its own id.
    """
    config = fixture_modules.config({"written": [fixture_modules.alpha]})
    resolved = resolve_set("written", path=config, announce=False)

    assert resolved.instances[0].id != resolved.instances[0].opaque_id
    assert _written_ids(resolved) == [ALPHA_ID]


def test_the_shipped_config_parses_as_one_table_of_module_path_lists() -> None:
    """The schema, not the contents. Empty is legitimate."""
    sets = load_set_config()
    assert all(isinstance(name, str) for name in sets)
    assert all(
        isinstance(member, str) for members in sets.values() for member in members
    )


def test_the_config_is_declared_as_package_data(shipped_package) -> None:
    """A wheel ships what `package-data` declares, and nothing else.

    The config is read at run time from inside the package, so a build that
    omits it produces an install that imports fine and raises on the first
    resolution — the same failure the catalogue's declaration is pinned against
    (`test_catalogue_file.py`). Asserted as "some declared glob still matches
    where the file actually is", so moving the file, renaming the directory or
    editing the declaration all move this test with them.
    """
    package, pyproject = shipped_package
    with pyproject.open("rb") as handle:
        globs = tomllib.load(handle)["tool"]["setuptools"]["package-data"][PACKAGE]
    relative = SET_CONFIG_PATH.relative_to(package)
    assert any(relative.match(glob) for glob in globs), (
        f"no glob in {globs} matches {relative}, so a built wheel would not "
        "carry the instance-set config"
    )


# -- the digest, over fixtures ----------------------------------------------


def test_a_resolved_set_digests_to_its_pinned_value(fixture_modules) -> None:
    """The algorithm pin. Fixture content is fixed, so only the algorithm moves it."""
    config = fixture_modules.config(
        {
            "one": [fixture_modules.alpha],
            "both": [fixture_modules.alpha, fixture_modules.beta],
        }
    )
    one = resolve_set("one", path=config, announce=False)
    both = resolve_set("both", path=config, announce=False)

    assert one.ids() == [ALPHA_ID]
    assert one.digest == ONE_DIGEST
    assert one.version == ONE_DIGEST[:16]
    assert both.ids() == [ALPHA_ID, BETA_ID]
    assert both.digest == BOTH_DIGEST
    assert both.version == BOTH_DIGEST[:16]


def test_composition_adds_nothing_to_the_digest_payload(fixture_modules) -> None:
    """A resolved set and a hand-built one over the same members are one digest.

    Which is the same claim as "the digest is over the resolved members": if
    composition contributed anything of its own — the config path, the set name,
    the order it read them in — these two would differ.
    """
    config = fixture_modules.config(
        {"both": [fixture_modules.alpha, fixture_modules.beta]}
    )
    resolved = resolve_set("both", path=config, announce=False)
    alpha, beta = fixture_modules.instances
    by_hand = InstanceSet(name="another-name", instances=[beta, alpha])

    assert resolved.digest == by_hand.digest == BOTH_DIGEST


def test_two_orderings_of_one_set_give_one_digest(fixture_modules) -> None:
    """A set is its members, not their order (§4).

    Otherwise a reordered config would move `Task(version=)` and report a re-run
    as new work — the `eval_set` hazard in the other direction.
    """
    config = fixture_modules.config(
        {
            "forward": [fixture_modules.alpha, fixture_modules.beta],
            "reverse": [fixture_modules.beta, fixture_modules.alpha],
        }
    )
    forward = resolve_set("forward", path=config, announce=False)
    reverse = resolve_set("reverse", path=config, announce=False)

    assert forward.ids() == reverse.ids() == [ALPHA_ID, BETA_ID]
    assert forward.digest == reverse.digest


def test_an_unchanged_config_reproduces_the_digest(fixture_modules) -> None:
    """Resolution is a read; two resolutions of one file are one claim."""
    config = fixture_modules.config(
        {"both": [fixture_modules.alpha, fixture_modules.beta]}
    )
    once = resolve_set("both", path=config, announce=False)
    again = resolve_set("both", path=config, announce=False)

    assert once.digest == again.digest == BOTH_DIGEST


def test_the_digest_does_not_cover_the_set_name(fixture_modules) -> None:
    """Same members under another name are the same work (§4).

    The name comes from the config, so digesting it would make a rename look
    like a membership change while the members it names are byte-identical.
    """
    config = fixture_modules.config(
        {"one": [fixture_modules.alpha], "renamed": [fixture_modules.alpha]}
    )
    assert (
        resolve_set("one", path=config, announce=False).digest
        == resolve_set("renamed", path=config, announce=False).digest
    )


def test_the_digest_moves_when_a_members_content_moves(fixture_modules) -> None:
    """A set is its members' **content**, not their ids (§4, §0 v43).

    Held deliberately apart from the id: each edit below leaves every member id
    where it was, so what moves the digest is the content in the payload and not
    the sort key. Under v44 a real content edit moves both — this asserts the
    half that is about the payload, which is the half §4's bullet states and the
    half that was written the other way round from v31 until v43.

    `test_invariant_2_task_version` asserts the same property through
    `Task(version=)`, which is this digest's first 16 hex; this pins it on the
    digest §4 names.
    """
    alpha, beta = fixture_modules.instances
    baseline = InstanceSet(name="s", instances=[alpha, beta])
    edits = {
        "a prompt": alpha.model_copy(update={"prompt": "a different question"}),
        "a snapshot": alpha.model_copy(update={"dataset_snapshot": "other"}),
        "the notes": alpha.model_copy(update={"notes": "a note"}),
        "an attribute": alpha.model_copy(update={"attributes": {"gene": "GENE2"}}),
    }
    digests = {}
    for what, edited in edits.items():
        assert edited.id == alpha.id, f"{what} moved the id; this test isolates content"
        digests[what] = InstanceSet(name="s", instances=[edited, beta]).digest
    for what, digest in digests.items():
        assert digest != baseline.digest, f"editing {what} left the set digest"
    assert len(set(digests.values())) == len(digests)


def test_the_digest_does_not_cover_the_set_level_snapshot(fixture_modules) -> None:
    """It is derived from the members and travels on its own axis (§4, §7).

    Covering it would digest the same fact twice, and a set-level value that
    disagreed with its members raises rather than reaching the payload. Asserted
    on the digest here; `test_invariant_2_task_version` asserts it on
    `Task(version=)`.
    """
    alpha, beta = fixture_modules.instances
    declared = InstanceSet(
        name="s", instances=[alpha, beta], dataset_snapshot="fixture-snapshot"
    )
    absent = InstanceSet(name="s", instances=[alpha, beta])
    other = InstanceSet(
        name="s", instances=[alpha, beta], dataset_snapshot="onekgpd-something-else"
    )
    assert declared.digest == absent.digest == other.digest == BOTH_DIGEST


def test_the_set_carries_the_snapshot_its_members_agree_on(fixture_modules) -> None:
    """Derived from the members, since there is nowhere left to declare it.

    `RunProvenance` falls back to the set-level snapshot, which is a comparison
    axis (§7), so losing it in the move would have quietly unpinned the
    substrate the ground truth was built against.
    """
    config = fixture_modules.config(
        {"both": [fixture_modules.alpha, fixture_modules.beta]}
    )
    assert (
        resolve_set("both", path=config, announce=False).dataset_snapshot
        == "fixture-snapshot"
    )


# -- the dataset ------------------------------------------------------------


def test_a_singleton_and_a_two_member_set_give_one_sample_each_way(
    fixture_modules,
) -> None:
    """The set is the dataset, and nothing between them adds or drops a row."""
    config = fixture_modules.config(
        {
            "one": [fixture_modules.alpha],
            "both": [fixture_modules.alpha, fixture_modules.beta],
        }
    )
    for name, expected in (("one", 1), ("both", 2)):
        built = build_generation_task(
            resolve_set(name, path=config, announce=False), mcp_url=MCP_URL
        )
        assert len(built.dataset) == expected


def test_the_dataset_is_emitted_in_sorted_member_order(fixture_modules) -> None:
    config = fixture_modules.config(
        {"reverse": [fixture_modules.beta, fixture_modules.alpha]}
    )
    resolved = resolve_set("reverse", path=config, announce=False)
    built = build_generation_task(resolved, mcp_url=MCP_URL)

    assert [str(sample.id) for sample in built.dataset] == [ALPHA_ID, BETA_ID]


def test_a_mock_run_over_a_resolved_set_produces_one_sample_per_member(
    fixture_modules, tmp_path: Path
) -> None:
    """The `mockllm` pattern (§6.5): no network, no key, a full transcript."""
    config = fixture_modules.config(
        {"both": [fixture_modules.alpha, fixture_modules.beta]}
    )
    resolved = resolve_set("both", path=config, announce=False)
    log = inspect_eval(
        build_generation_task(
            resolved,
            mcp_url=MCP_URL,
            repeats=1,
            solver_config=SolverConfig(
                max_output_tokens=16_000, turn_cap=1, reasoning_effort=None
            ),
        ),
        model="mockllm/model",
        model_args={
            "custom_outputs": [ModelOutput.from_content("mockllm/model", "done")] * 8
        },
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]

    assert log.eval.task_version == BOTH_DIGEST[:16]
    assert sorted(str(sample.id) for sample in (log.samples or [])) == [
        ALPHA_ID,
        BETA_ID,
    ]


# -- the empty config: the harness with no instance at all ------------------


def test_an_empty_sets_table_imports_registers_and_refuses_cleanly(
    fixture_modules,
) -> None:
    """§16, and the whole reason composition goes through a config.

    The harness publishes while the instance set is still being built, so it has
    to import, register its task, and fail loudly rather than silently run empty
    when nothing is declared.
    """
    import importlib

    from inspect_ai._util.registry import registry_info

    empty = fixture_modules.config({}, name="empty.toml")
    assert load_set_config(empty) == {}
    assert available_sets(empty) == []

    module = importlib.import_module("genomics_harness")
    assert registry_info(module.genomics_generation).name

    with pytest.raises(CompositionError, match="unknown instance set"):
        resolve_set("anything", path=empty)
    with pytest.raises(CompositionError, match="no instance set selected"):
        resolve_set(None, path=empty)


def _import_roots(source: str) -> list[str]:
    """The top-level name each import in ``source`` reaches for.

    Parsed, not string-matched, so every spelling Python allows is seen the same
    way. A relative import reaches for this package and is reported as such.
    """
    roots: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots.append(PACKAGE if node.level else (node.module or "").split(".")[0])
    return roots


def _declared_dependencies(pyproject: Path) -> set[str]:
    """The distributions `harness/pyproject.toml` declares, as import roots."""
    with pyproject.open("rb") as handle:
        declared = tomllib.load(handle)["project"]["dependencies"]
    return {
        re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0].strip().replace("-", "_")
        for spec in declared
    }


def foreign_imports(package: Path, *, allowed: set[str]) -> list[str]:
    """Imports in ``package`` reaching outside the stdlib, itself and ``allowed``.

    The predicate, in one place so the assertion over the real package and the
    positive control below run the *same* rule.
    """
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        for root in _import_roots(path.read_text(encoding="utf-8")):
            if root in allowed or root in sys.stdlib_module_names:
                continue
            offenders.append(f"{path.name}: {root}")
    return offenders


def test_no_module_in_the_package_imports_an_instance_module(shipped_package) -> None:
    """The compile-time reference the mechanism exists to remove (§16, §4).

    An import of an instance module anywhere under `src/genomics_harness` would
    make the harness unshippable without the instances, whatever the config then
    said. A source check, because that is where the property lives.

    **The rule is stated as a whitelist and not as a needle, and that is the
    repair the split forced** (§0 v58). Until September 2026 this searched import
    lines for a literal naming the instance module that then lived beside this
    package. Once the instances moved to a package of their own that literal
    could appear in no file, so the check would have scanned the same twenty-two
    modules forever and matched, by construction, nothing — a checker that cannot
    fire, which this project does not count as evidence. Substituting the new
    package's name would only move the same problem one rename along, and would
    put an instance package's name in published source besides (§16).

    So the assertion is inverted: every import must reach the standard library,
    this package itself, or a distribution `pyproject.toml` declares. An instance
    package fails it under any name, and so does any other accidental coupling —
    which is strictly more than the needle caught. The allowed set is *read from
    the packaging metadata* rather than listed here, so adding a dependency moves
    this test with it and adding an undeclared import does not.
    """
    package, pyproject = shipped_package
    allowed = _declared_dependencies(pyproject) | {PACKAGE}
    assert {"inspect_ai", "pydantic"} <= allowed, (
        f"the declared dependencies read as {sorted(allowed)}, which does not "
        "look like this package's; the whitelist is not being read correctly"
    )
    assert foreign_imports(package, allowed=allowed) == []


def test_the_foreign_import_rule_fires_on_an_instance_package(tmp_path: Path) -> None:
    """The positive control. A rule that has never matched anything is not evidence.

    A hand-written module importing an instance package is put through the same
    predicate, under the same allowed set the real assertion computes. It catches
    the import whatever the package is called, which is the property the literal
    needle could not have after the move.
    """
    control = tmp_path / "package"
    control.mkdir()
    (control / "clean.py").write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pydantic import BaseModel\n"
        "from .instances import Instance\n",
        encoding="utf-8",
    )
    allowed = {PACKAGE, "pydantic"}
    assert foreign_imports(control, allowed=allowed) == []

    (control / "offender.py").write_text(
        "from some_instance_package.a_gene import INSTANCE\n", encoding="utf-8"
    )
    assert foreign_imports(control, allowed=allowed) == [
        "offender.py: some_instance_package"
    ]


# -- selection is explicit --------------------------------------------------


def test_selecting_nothing_raises_and_lists_the_available_sets(
    fixture_modules,
) -> None:
    config = fixture_modules.config(
        {
            "one": [fixture_modules.alpha],
            "both": [fixture_modules.alpha, fixture_modules.beta],
        }
    )
    with pytest.raises(CompositionError) as raised:
        resolve_set(None, path=config)

    assert "no instance set selected" in str(raised.value)
    assert "'both'" in str(raised.value) and "'one'" in str(raised.value)


def test_an_unknown_name_raises_and_lists_the_available_sets(fixture_modules) -> None:
    config = fixture_modules.config({"one": [fixture_modules.alpha]})
    with pytest.raises(CompositionError) as raised:
        resolve_set("not-declared", path=config)

    assert "unknown instance set 'not-declared'" in str(raised.value)
    assert "'one'" in str(raised.value)


def test_a_missing_config_raises_and_names_the_override(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere.toml"
    with pytest.raises(CompositionError) as raised:
        resolve_set("one", path=missing)

    assert str(missing) in str(raised.value)
    assert SET_CONFIG_ENV in str(raised.value)


def test_a_config_without_a_sets_table_raises(tmp_path: Path) -> None:
    path = tmp_path / "instance-sets.toml"
    path.write_text('title = "not a set config"\n', encoding="utf-8")
    with pytest.raises(CompositionError, match=r"no \[sets\] table"):
        load_set_config(path)


def test_a_set_that_is_not_a_list_of_module_paths_raises(tmp_path: Path) -> None:
    path = tmp_path / "instance-sets.toml"
    path.write_text('[sets]\none = "not-a-list"\n', encoding="utf-8")
    with pytest.raises(CompositionError, match="list of module paths"):
        load_set_config(path)


def test_a_selected_module_that_will_not_import_raises(fixture_modules) -> None:
    config = fixture_modules.config({"one": ["kgpbench_fixture_absent"]})
    with pytest.raises(CompositionError) as raised:
        resolve_set("one", path=config)

    assert "will not import" in str(raised.value)
    assert "kgpbench_fixture_absent" in str(raised.value)


def test_a_selected_module_without_the_agreed_symbol_raises(fixture_modules) -> None:
    """One contract, applied uniformly: a module without `INSTANCE` is not a member."""
    (fixture_modules.directory / "kgpbench_fixture_no_symbol.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    config = fixture_modules.config({"one": ["kgpbench_fixture_no_symbol"]})
    with pytest.raises(CompositionError) as raised:
        resolve_set("one", path=config)

    assert f"defines no {INSTANCE_SYMBOL}" in str(raised.value)


def test_a_symbol_that_is_not_an_instance_raises(fixture_modules) -> None:
    """Validated at run time, because `mypy` cannot see through `import_module`."""
    (fixture_modules.directory / "kgpbench_fixture_wrong_type.py").write_text(
        f'{INSTANCE_SYMBOL} = "not an Instance"\n', encoding="utf-8"
    )
    config = fixture_modules.config({"one": ["kgpbench_fixture_wrong_type"]})
    with pytest.raises(CompositionError, match="not an Instance"):
        resolve_set("one", path=config)


def test_two_members_resolving_to_one_instance_raises(fixture_modules) -> None:
    """`InstanceSet` refuses duplicate ids; the error says which set (§4)."""
    config = fixture_modules.config(
        {"doubled": [fixture_modules.alpha, fixture_modules.alpha]}
    )
    with pytest.raises(CompositionError, match="does not compose"):
        resolve_set("doubled", path=config)


# -- unselected sets warn, and are never executed ---------------------------


def test_an_unselected_set_naming_a_missing_module_warns_and_the_run_proceeds(
    fixture_modules,
) -> None:
    """The common staleness — a deleted or renamed module — caught on any run (§4)."""
    config = fixture_modules.config(
        {"one": [fixture_modules.alpha], "stale": ["kgpbench_fixture_deleted"]}
    )
    with pytest.warns(UserWarning, match="kgpbench_fixture_deleted"):
        resolved = resolve_set("one", path=config, announce=False)

    assert resolved.ids() == [ALPHA_ID]


def test_an_unselected_module_is_checked_without_being_imported(
    fixture_modules, monkeypatch
) -> None:
    """`find_spec` answers whether it could be imported; it does not import it.

    Its limit is chosen and not overlooked: a module that exists but raises on
    import warns only when it is selected, because importing every set's modules
    to close that would execute code the operator did not ask to run (§4).
    """
    import importlib

    imported: list[str] = []
    real = importlib.import_module

    def recording(name: str, package: str | None = None):
        imported.append(name)
        return real(name, package)

    config = fixture_modules.config(
        {"one": [fixture_modules.alpha], "other": [fixture_modules.beta]}
    )
    monkeypatch.setattr(importlib, "import_module", recording)
    resolve_set("one", path=config, announce=False)

    assert imported == [fixture_modules.alpha]


# -- the config path, and the startup print --------------------------------


def test_the_env_var_overrides_the_default_path(fixture_modules, monkeypatch) -> None:
    """An override, never a task argument — a path in `task_args` looks like
    provenance and is not, since the same path yields different members after an
    edit (§4, §7).

    The ambient variable is cleared first: an operator who exports it — which is
    exactly how a private tree points at its own instances — would otherwise fail
    this test on the default-path half, and the environment is not the subject.
    """
    monkeypatch.delenv(SET_CONFIG_ENV, raising=False)
    assert config_path() == SET_CONFIG_PATH

    override = fixture_modules.config({"only": [fixture_modules.alpha]})
    monkeypatch.setenv(SET_CONFIG_ENV, str(override))

    assert config_path() == override
    assert available_sets() == ["only"]
    assert resolve_set("only", announce=False).ids() == [ALPHA_ID]


def test_resolution_prints_the_config_path_and_the_members(
    fixture_modules, capsys
) -> None:
    """The print is a requirement, not a convenience (§4).

    It is what makes a `mockllm` run expose a wrong or stale config at zero
    cost, and it goes to the console — never into the log, so the published
    record stays neutral.
    """
    config = fixture_modules.config(
        {"both": [fixture_modules.alpha, fixture_modules.beta]}
    )
    resolve_set("both", path=config)
    printed = capsys.readouterr().out

    assert str(config) in printed
    assert "'both'" in printed
    for module_path in (fixture_modules.alpha, fixture_modules.beta):
        assert module_path in printed
    for member in (ALPHA_ID, BETA_ID):
        assert member in printed


# -- the registered task ----------------------------------------------------


def test_the_registered_task_takes_the_set_name_and_no_member_and_no_path(
    fixture_modules, monkeypatch
) -> None:
    """`task_args` reaches `header.json` whole, defaults included (§13.3).

    So the set name is the only composition string that publishes, the config
    path is not an argument at all, and nothing structured is either.
    """
    import json

    from inspect_ai._eval.task.constants import TASK_ALL_PARAMS_ATTR

    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    built = genomics_generation(mcp_url=MCP_URL, set="one", reasoning_effort=None)
    captured = getattr(built, TASK_ALL_PARAMS_ATTR)

    assert set(captured) == {
        "mcp_url",
        "set",
        "repeats",
        "instance_set",
        "reasoning_effort",
    }
    assert captured["set"] == "one"
    for name, value in captured.items():
        assert isinstance(value, (str, int, float, bool, type(None))), name
    json.dumps(captured)


def test_the_registered_task_takes_its_version_from_the_set_digest(
    fixture_modules, monkeypatch
) -> None:
    monkeypatch.setenv(
        SET_CONFIG_ENV,
        str(
            fixture_modules.config(
                {"both": [fixture_modules.alpha, fixture_modules.beta]}
            )
        ),
    )
    built = genomics_generation(mcp_url=MCP_URL, set="both", reasoning_effort=None)

    assert built.version == BOTH_DIGEST[:16]
    assert (built.metadata or {})["instances_sha256"] == BOTH_DIGEST


def test_the_registered_task_refuses_an_unknown_set_before_any_model_call(
    fixture_modules, monkeypatch
) -> None:
    """Resolution is at task construction, so a bad selector costs nothing."""
    monkeypatch.setenv(
        SET_CONFIG_ENV, str(fixture_modules.config({"one": [fixture_modules.alpha]}))
    )
    with pytest.raises(CompositionError, match="unknown instance set"):
        genomics_generation(mcp_url=MCP_URL, set="not-declared", reasoning_effort=None)
    with pytest.raises(CompositionError, match="no instance set selected"):
        genomics_generation(mcp_url=MCP_URL, reasoning_effort=None)


def test_the_fixture_modules_are_not_the_shipped_ones(fixture_modules) -> None:
    """A guard on this module's own discipline, cheap to state.

    If a fixture module path ever became one the shipped config names, the digest
    pins above would silently start pinning operator-maintained content — the
    failure this file was rewritten to remove.

    **Stated as a disjointness, and it reads no value out of the config** (§15).
    Until the split it compared against two literal module paths held in this
    file; those named instances, and no surface publishing with the harness may
    (§16). What it asks now is that the two sets do not meet, which is the actual
    property — it holds over whatever the operator's file happens to name, it
    holds vacuously and correctly over the empty table a release ships, and it
    pins no membership.
    """
    declared = {module for members in load_set_config().values() for module in members}
    fixtures = {fixture_modules.alpha, fixture_modules.beta}
    assert fixtures.isdisjoint(declared), (
        f"fixture modules {sorted(fixtures & declared)} are named by the shipped "
        "config; the digest pins in this file would then be pinning "
        "operator-maintained content"
    )
    assert sys.modules[fixture_modules.alpha].INSTANCE.id == ALPHA_ID
