"""The pinned catalogue file — its model, its loader, and its digest.

Every assertion about content runs against `genomics_harness/data/catalogue-c1.json`
itself. That is the point of the round: until now the 22 checks existed as prose
and as hand-built fixtures, so `chain_consistency` and `refinement_chains` had
only ever read a fixture's `refines` links — of which two in this tree disagreed
on the direction (specification §0 v20). A fixture cannot settle what the
catalogue says.

Mutations are built by loading the real file's raw JSON, changing one thing, and
writing it to `tmp_path`. The real file is never written to.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import genomics_harness
from genomics_harness import (
    CATALOGUE_PATH,
    DIGEST_PLACEHOLDER_SUFFIX,
    CatalogueDigestError,
    CatalogueFile,
    Check,
    catalogue_digest,
    load_catalogue,
)
from genomics_harness.digest import sha256_digest
from genomics_harness.judge_prompt import refinement_chains

CATALOGUE = load_catalogue()

# §4's fifteen, in the file's declaration order.
IN_SCOPE = (
    "A1",
    "A2",
    "A3",
    "A5",
    "B1",
    "B3",
    "B5",
    "B6",
    "C5",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
)
NOT_IMPLEMENTED = ("A4", "B2", "B4", "C1", "C2", "C3", "C4")


def raw() -> dict[str, Any]:
    """The real file's JSON, parsed and unvalidated, as a fresh mutable copy."""
    payload = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def written(tmp_path: Path, payload: dict[str, Any], *, digest: bool = True) -> Path:
    """Write ``payload`` to a file, recomputing its digest unless told not to."""
    path = tmp_path / "catalogue.json"
    if digest:
        without = {k: v for k, v in payload.items() if k != "content_sha256"}
        payload = {**payload, "content_sha256": sha256_digest(without)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# -- the real file ---------------------------------------------------------


def test_the_pinned_catalogue_loads_and_verifies() -> None:
    assert CATALOGUE.catalogue_release == "c1"
    assert len(CATALOGUE) == 22
    assert CATALOGUE.content_sha256 == CATALOGUE.digest


def test_the_recorded_digest_is_no_longer_the_placeholder() -> None:
    """The field shipped as ``sha256-unimplemented`` (§0 v13's convention).

    A digest that is still the placeholder records nothing, and a loader that
    accepts one has an axis in name only.
    """
    assert not CATALOGUE.content_sha256.endswith(DIGEST_PLACEHOLDER_SUFFIX)
    assert len(CATALOGUE.content_sha256) == 64
    assert set(CATALOGUE.content_sha256) <= set("0123456789abcdef")


def test_fifteen_checks_are_in_scope_and_seven_are_not() -> None:
    assert tuple(CATALOGUE.in_scope_ids) == IN_SCOPE
    assert tuple(c.id for c in CATALOGUE if not c.in_scope) == NOT_IMPLEMENTED


def test_contamination_class_is_null_on_exactly_the_seven_not_in_scope() -> None:
    """The true state, not a gap: §12 Principle 2 classifies only the fifteen.

    Enforced in the model in the forward direction only — an in-scope check must
    carry a class. This pins the converse for the delivered file without making it
    a rule a later catalogue has to obey.
    """
    unclassified = tuple(c.id for c in CATALOGUE if c.contamination_class is None)
    assert unclassified == NOT_IMPLEMENTED


def test_every_check_id_carries_its_own_group_prefix() -> None:
    assert all(check.id.startswith(check.group) for check in CATALOGUE)
    assert sorted({check.group for check in CATALOGUE}) == ["A", "B", "C", "D"]


def test_d4_carries_no_talos_anchor_and_does_not_restate_it_in_text() -> None:
    """A null anchor is the check having none, not the field being unfilled.

    The field is the only carrier. D4's `text` used to explain the null in prose,
    which §4 forbids — a field is never also written into `text` — and that
    sentence was deleted at v26. What the null means is stated in the spec, not
    in the check it is null on.
    """
    d4 = CATALOGUE.get("D4")
    assert d4.talos_anchor is None
    assert "Talos" not in d4.text


def test_the_read_surface_is_by_id() -> None:
    assert "B6" in CATALOGUE
    assert "B7" not in CATALOGUE
    assert CATALOGUE.get("B6").title == "Statistical power"
    with pytest.raises(KeyError):
        CATALOGUE.get("B7")
    assert CATALOGUE.ids == [check.id for check in CATALOGUE]


# -- the nest --------------------------------------------------------------


def test_the_nest_direction_is_on_the_file() -> None:
    """B6 is the base; B5 refines B6; B1 refines B5 (§4).

    The links themselves, not what a chain builder makes of them — six tests once
    passed over an inverted mapping because each asserted its consequences (§15).
    """
    assert CATALOGUE.get("B6").refines is None
    assert CATALOGUE.get("B5").refines == "B6"
    assert CATALOGUE.get("B1").refines == "B5"


def test_only_the_nest_carries_a_refinement_link() -> None:
    assert [c.id for c in CATALOGUE if c.refines is not None] == ["B1", "B5"]


def test_refinement_chains_over_the_real_file() -> None:
    """The headline: one chain, base first.

    ``refinement_chains`` is annotated for ``Catalogue`` and needs only ``in``,
    ``get`` and ``ids``, so it runs over the file model unchanged — which is what
    puts the delivered function over the delivered data for the first time.
    """
    chains = refinement_chains(CATALOGUE, CATALOGUE.ids)
    assert chains == [["B6", "B5", "B1"]]


def test_refinement_chains_over_the_fifteen_in_scope() -> None:
    chains = refinement_chains(CATALOGUE, CATALOGUE.in_scope_ids)
    assert chains == [["B6", "B5", "B1"]]


# -- the digest ------------------------------------------------------------


def test_the_digest_is_canonical_json_of_everything_but_itself() -> None:
    """Computed independently from the raw file rather than from the model.

    This is also the round-trip assertion: the model reproduces the file's JSON
    exactly, so ``model_dump`` losing a field would show up here as a different
    digest rather than as nothing at all.
    """
    payload = raw()
    del payload["content_sha256"]
    assert CATALOGUE.digest == sha256_digest(payload)
    assert CATALOGUE.model_dump(mode="json", exclude={"content_sha256"}) == payload


def test_the_digest_ignores_layout(tmp_path: Path) -> None:
    """It is taken over parsed content, so reindenting the file changes nothing."""
    payload = raw()
    reordered = {k: payload[k] for k in reversed(list(payload))}
    path = tmp_path / "reindented.json"
    path.write_text(json.dumps(reordered, indent=8), encoding="utf-8")
    assert catalogue_digest(path) == CATALOGUE.digest
    assert load_catalogue(path).content_sha256 == CATALOGUE.content_sha256


def test_the_digest_moves_on_one_character_of_check_text(tmp_path: Path) -> None:
    payload = raw()
    payload["checks"][0]["text"] += "."
    assert catalogue_digest(written(tmp_path, payload)) != CATALOGUE.digest


def test_the_digest_moves_on_the_release_label(tmp_path: Path) -> None:
    """Relabelling is a change: one label must not name two contents.

    ``"c2"`` is a control value, not a release that exists — but it is spelled
    in the form §4 permits, because a test that reaches for ``vN`` to make its
    point teaches the spelling the project forbids.
    """
    payload = raw()
    payload["catalogue_release"] = "c2"
    assert catalogue_digest(written(tmp_path, payload)) != CATALOGUE.digest


def test_the_digest_moves_on_the_notes(tmp_path: Path) -> None:
    payload = raw()
    payload["notes"] = "something else"
    assert catalogue_digest(written(tmp_path, payload)) != CATALOGUE.digest


def test_a_file_edited_without_recomputing_its_digest_fails_loudly(
    tmp_path: Path,
) -> None:
    """The point of the field. A warning here is a digest nobody reads."""
    payload = raw()
    payload["checks"][0]["text"] += "."
    path = written(tmp_path, payload, digest=False)
    with pytest.raises(CatalogueDigestError) as caught:
        load_catalogue(path)
    assert payload["content_sha256"] in str(caught.value)


def test_the_placeholder_is_reported_as_never_computed(tmp_path: Path) -> None:
    """Distinguished from a mismatch: the two have different fixes."""
    payload = raw()
    payload["content_sha256"] = "sha256-unimplemented"
    with pytest.raises(CatalogueDigestError) as caught:
        load_catalogue(written(tmp_path, payload, digest=False))
    assert "never" in str(caught.value)


def test_catalogue_digest_reads_a_file_whose_recorded_digest_is_wrong(
    tmp_path: Path,
) -> None:
    """The recompute path, and it returns a string rather than a catalogue."""
    payload = raw()
    payload["content_sha256"] = "sha256-unimplemented"
    path = written(tmp_path, payload, digest=False)
    assert catalogue_digest(path) == CATALOGUE.digest
    with pytest.raises(CatalogueDigestError):
        load_catalogue(path)


def test_a_digest_error_is_a_value_error() -> None:
    assert issubclass(CatalogueDigestError, ValueError)


# -- validation ------------------------------------------------------------


def test_an_unknown_field_on_a_check_is_an_error(tmp_path: Path) -> None:
    payload = raw()
    payload["checks"][0]["required"] = True
    with pytest.raises(ValidationError, match="required"):
        load_catalogue(written(tmp_path, payload))


def test_an_unknown_field_on_the_file_is_an_error(tmp_path: Path) -> None:
    payload = raw()
    payload["renderer_version"] = "r2"
    with pytest.raises(ValidationError, match="renderer_version"):
        load_catalogue(written(tmp_path, payload))


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "group",
        "title",
        "text",
        "ground_truth_class",
        "contamination_class",
        "construction",
        "genotypes",
        "talos_anchor",
        "in_scope",
        "refines",
    ],
)
def test_a_missing_field_on_a_check_is_an_error(tmp_path: Path, field: str) -> None:
    """All eleven are required, the three nullable ones included.

    A key absent is an error and not a default: the field set is fixed, and a
    silently defaulted `text` is an empty judge-prompt block.
    """
    payload = raw()
    del payload["checks"][0][field]
    with pytest.raises(ValidationError, match=field):
        load_catalogue(written(tmp_path, payload))


@pytest.mark.parametrize("field", ["catalogue_release", "content_sha256", "notes"])
def test_a_missing_field_on_the_file_is_an_error(tmp_path: Path, field: str) -> None:
    payload = raw()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        load_catalogue(written(tmp_path, payload, digest=field != "content_sha256"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ground_truth_class", "computed"),
        ("ground_truth_class", None),
        ("contamination_class", "behavior_scoring"),
        ("construction", "emergent_or_dedicated"),
        ("construction", None),
    ],
)
def test_a_value_outside_a_closed_set_is_an_error(
    tmp_path: Path, field: str, value: str | None
) -> None:
    """An error and not a warning. A misspelt class silently drops a check out of
    every group it belongs to."""
    payload = raw()
    payload["checks"][0][field] = value
    with pytest.raises(ValidationError, match=field):
        load_catalogue(written(tmp_path, payload))


def test_a_blank_operative_definition_is_an_error(tmp_path: Path) -> None:
    payload = raw()
    payload["checks"][0]["text"] = "   "
    with pytest.raises(ValidationError, match="blank"):
        load_catalogue(written(tmp_path, payload))


def test_a_duplicate_check_id_is_an_error(tmp_path: Path) -> None:
    payload = raw()
    payload["checks"].append(dict(payload["checks"][0]))
    with pytest.raises(ValidationError, match="duplicate check id 'A1'"):
        load_catalogue(written(tmp_path, payload))


def test_a_check_cannot_refine_itself(tmp_path: Path) -> None:
    payload = raw()
    payload["checks"][0]["refines"] = payload["checks"][0]["id"]
    with pytest.raises(ValidationError, match="cannot refine itself"):
        load_catalogue(written(tmp_path, payload))


def test_an_in_scope_check_must_carry_a_contamination_class(tmp_path: Path) -> None:
    payload = raw()
    for check in payload["checks"]:
        if check["id"] == "A1":
            check["contamination_class"] = None
    with pytest.raises(ValidationError, match="in scope but carries no"):
        load_catalogue(written(tmp_path, payload))


def test_an_out_of_scope_check_may_be_classified_early(tmp_path: Path) -> None:
    """Only the forward direction is a rule. Classifying A4 ahead of its
    implementation is not an error."""
    payload = raw()
    for check in payload["checks"]:
        if check["id"] == "A4":
            check["contamination_class"] = "answer_scoring"
    loaded = load_catalogue(written(tmp_path, payload))
    assert loaded.get("A4").contamination_class is not None
    assert not loaded.get("A4").in_scope


# -- subsets, and the dangling target --------------------------------------


def test_a_subset_whose_refines_target_is_absent_stays_loadable(
    tmp_path: Path,
) -> None:
    """Decided (§4): a chain terminates at an absent target, and a catalogue
    carrying one is valid. This is where the file model and the delivered
    ``Catalogue`` part company — the latter rejects a dangling `refines`."""
    payload = raw()
    payload["checks"] = [c for c in payload["checks"] if c["id"] in {"B1", "B5"}]
    loaded = load_catalogue(written(tmp_path, payload))
    assert loaded.ids == ["B1", "B5"]
    assert loaded.get("B5").refines == "B6"
    assert "B6" not in loaded


def test_a_chain_terminates_at_an_absent_target(tmp_path: Path) -> None:
    payload = raw()
    payload["checks"] = [c for c in payload["checks"] if c["id"] in {"B1", "B5"}]
    loaded = load_catalogue(written(tmp_path, payload))
    assert refinement_chains(loaded, loaded.ids) == [["B5", "B1"]]


def test_a_subset_holding_only_a_leaf_produces_no_chain(tmp_path: Path) -> None:
    payload = raw()
    payload["checks"] = [c for c in payload["checks"] if c["id"] == "B1"]
    loaded = load_catalogue(written(tmp_path, payload))
    assert refinement_chains(loaded, loaded.ids) == []


# -- locating the file -----------------------------------------------------


def package_dir() -> Path:
    """The installed package's own directory, whatever it is."""
    return Path(genomics_harness.__file__).resolve().parent


def test_the_default_path_is_the_pinned_file() -> None:
    """And it is *inside* the package, which is what makes an install carry it.

    Containment is the property, not the directory's name: naming it here would
    restate the code, and the name is pinned where it has to hold — against the
    packaging declaration, in the test below.
    """
    assert CATALOGUE_PATH.name == "catalogue-c1.json"
    assert CATALOGUE_PATH.is_relative_to(package_dir())
    assert CATALOGUE_PATH.is_file()


def test_the_filename_and_the_release_field_carry_the_same_release() -> None:
    """One string in two places, and neither is the authority on its own (§0 v21).

    A file renamed without its field, or relabelled without its filename, gives
    a reader two answers to "which release produced this verdict" — and the
    filename is the one a person reads first while the field is the one a log
    records.
    """
    assert CATALOGUE_PATH.stem == f"catalogue-{CATALOGUE.catalogue_release}"
    assert not CATALOGUE.catalogue_release.startswith("v")


def test_the_digest_does_not_cover_the_filename(tmp_path: Path) -> None:
    """The rename moved no digest, and this is why (§0 v21).

    `digest` is taken over parsed content and the model carries no path field,
    so the same bytes under any name digest identically. Obvious until a release
    depends on it: re-scoring a deferred log resolves the release by digest, and
    a digest that moved on a rename would orphan every verdict already written
    against it.
    """
    elsewhere = tmp_path / "some-other-name.json"
    elsewhere.write_text(CATALOGUE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_catalogue(elsewhere).digest == CATALOGUE.digest
    assert load_catalogue(elsewhere).content_sha256 == CATALOGUE.content_sha256


def test_the_catalogue_is_declared_as_package_data() -> None:
    """A wheel ships what `package-data` declares, and nothing else.

    The catalogue is read at run time from inside the package, so a build that
    omits it produces an install that imports fine and raises `FileNotFoundError`
    on the first `load_catalogue()`. Three edits break it silently — moving the
    file, renaming its directory, or editing the declaration — and this asserts
    the fact that all three change: that some declared glob still matches where
    the catalogue actually is.

    Verified once against a real wheel (`design/small-fixes-august-2026.md` §4);
    this is what re-runs. Building a wheel here would need the network and the
    build backend, so the declaration is what gets checked.

    **A missing `pyproject.toml` fails rather than skips** (§0 v21). It used to
    skip, on the reading that an absent file meant an installed tree rather than
    an editable one — but a build that dropped the declaration, or a tree in
    which the file moved, is exactly what this exists to catch, and a skip
    reports that as "not applicable". A test whose failure mode is silence is
    the shape §9 keeps finding. `requires-python` is 3.11, so `tomllib` is
    stdlib and there is nothing left to make the read conditional on.
    """
    pyproject = package_dir().parents[1] / "pyproject.toml"
    assert pyproject.is_file(), (
        f"no pyproject.toml at {pyproject}. This test reads the packaging "
        "declaration to check that a built wheel would carry the catalogue, so "
        "an absent file is one of the failures it exists to catch, not a reason "
        "to skip. Run the suite against the editable tree."
    )

    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)
    globs = config["tool"]["setuptools"]["package-data"]["genomics_harness"]

    relative = CATALOGUE_PATH.relative_to(package_dir())
    assert any(relative.match(glob) for glob in globs), (
        f"no glob in {globs} matches {relative}, so a built wheel would not "
        "carry the catalogue"
    )


def test_a_missing_file_names_the_path_it_looked_at(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_catalogue(missing)


def test_a_file_that_is_not_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "catalogue.json"
    path.write_text("checks: []", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_catalogue(path)


def test_a_string_path_is_accepted(tmp_path: Path) -> None:
    assert load_catalogue(str(CATALOGUE_PATH)).digest == CATALOGUE.digest


# -- the model is frozen and forbids extras --------------------------------


def test_the_models_are_frozen() -> None:
    """A verdict cites a catalogue digest; a catalogue mutated after loading
    would make that citation false in memory."""
    with pytest.raises(ValidationError):
        CATALOGUE.get("B6").text = "something else"
    with pytest.raises(ValidationError):
        CATALOGUE.catalogue_release = "c2"


def test_the_field_set_is_the_files(tmp_path: Path) -> None:
    """Eleven per check, four per file — §4's list, and nothing bolted on.

    `Check` **is** the file's check model since the collapse of August 2026, so
    this asserts one field set and not a correspondence between two.
    """
    assert set(Check.model_fields) == {
        "id",
        "group",
        "title",
        "text",
        "ground_truth_class",
        "contamination_class",
        "construction",
        "genotypes",
        "talos_anchor",
        "in_scope",
        "refines",
    }
    assert set(CatalogueFile.model_fields) == {
        "catalogue_release",
        "content_sha256",
        "notes",
        "checks",
    }
