"""The step from the pinned file to the object a judge factory takes.

Since the model collapse of August 2026 nothing is synthesised on the way
across: a `Check` **is** the file's check. What these assert is that the step is
a filter and a relabel and nothing else — that the delivered ids are the file's
own in-scope ids, that every check crosses byte-identical, and that the
wire-form route round-trips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgpbench import (
    Catalogue,
    CatalogueUnresolved,
    catalogue_arg,
    delivered_catalogue,
    load_catalogue,
    load_wire_catalogue,
    resolve_catalogue,
)


def test_the_delivered_catalogue_is_the_in_scope_set() -> None:
    """The rule, not the side effect it used to be.

    `Check.contamination_class` is nullable now, so nothing raises on a check
    out of scope and nothing enforces the subset by accident. Asserted against
    the file's own ``in_scope_ids`` rather than a literal count, so adding a
    check to scope moves both sides together and a filter bug moves only one.
    """
    pinned = load_catalogue()

    delivered = delivered_catalogue()

    assert delivered.ids == pinned.in_scope_ids
    assert all(check.in_scope for check in delivered)
    assert len(delivered) < len(pinned)


def test_every_check_crosses_unchanged() -> None:
    """One check model, so equality is the assertion — not a field-by-field map."""
    pinned = load_catalogue()
    delivered = delivered_catalogue()

    for check in delivered:
        assert check == pinned.get(check.id)


def test_the_delivered_catalogue_is_the_files_release() -> None:
    """Same label, different digest: a filter and a container, nothing invented."""
    pinned = load_catalogue()
    delivered = delivered_catalogue()

    assert delivered.release == pinned.catalogue_release
    assert delivered.digest != pinned.content_sha256
    assert pinned.catalogue_release in (delivered.notes or "")
    assert pinned.content_sha256 in (delivered.notes or "")


def test_the_nest_survives_the_restriction_to_the_in_scope_set() -> None:
    """B1 -> B5 -> B6 stays whole, which is why the subset loads at all.

    `Catalogue` rejects a `refines` pointing outside itself, so this is asserted
    by construction as well — but asserting the links directly says *which*
    property held rather than only that nothing raised.
    """
    delivered = delivered_catalogue()

    assert delivered.get("B1").refines == "B5"
    assert delivered.get("B5").refines == "B6"
    assert delivered.get("B6").refines is None


def test_resolve_by_release_label_is_the_pinned_in_scope_set() -> None:
    pinned = load_catalogue()

    resolved = resolve_catalogue(pinned.catalogue_release)

    assert resolved.catalogue.digest == delivered_catalogue().digest
    assert pinned.catalogue_release in resolved.source
    assert pinned.content_sha256[:16] in resolved.source


def test_resolve_a_wire_form_synthesises_nothing(tmp_path: Path) -> None:
    """The second route round-trips a catalogue somebody else ruled on."""
    original = delivered_catalogue()
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(catalogue_arg(original)), encoding="utf-8")

    resolved = resolve_catalogue(str(path))

    assert resolved.catalogue.digest == original.digest
    assert str(path) in resolved.source


def test_a_missing_or_unparseable_catalogue_refuses(tmp_path: Path) -> None:
    """Refusing beats defaulting: what a judge is given decides its verdicts."""
    with pytest.raises(CatalogueUnresolved, match="no catalogue file"):
        load_wire_catalogue(tmp_path / "absent.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(CatalogueUnresolved, match="not JSON"):
        load_wire_catalogue(broken)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps({"release": "x", "checks": [{"id": "Z"}]}), encoding="utf-8"
    )
    with pytest.raises(CatalogueUnresolved, match="does not load as a Catalogue"):
        load_wire_catalogue(wrong)


def test_a_spec_that_names_neither_refuses(tmp_path: Path) -> None:
    """A release label that does not exist is a path that does not exist."""
    with pytest.raises(CatalogueUnresolved, match="no catalogue file"):
        resolve_catalogue("c9")


def test_the_delivered_catalogue_survives_the_wire_round_trip() -> None:
    """It is passed to a `@scorer` factory, so it has to be JSON on both sides.

    `_eval/score.py` splats a stored header back into the factory as plain JSON
    types, so a catalogue that only works as a model breaks in the deferred path
    and nowhere else (§6.3).
    """
    original = delivered_catalogue()

    round_tripped = Catalogue.of(json.loads(json.dumps(catalogue_arg(original))))

    assert round_tripped.digest == original.digest
