"""The pinned catalogue file — its model, and the loader that verifies its digest.

`genomics_harness/data/catalogue-c1.json` is the authority for check semantics
(specification §4). This module is the only thing that reads it.

**The file's check model is :class:`~genomics_harness.checks.Check` itself.**
There is one check model, not two: the file loads straight into the delivered
type, so there is no mapping to keep honest and no field set that can drift out
of correspondence. The file's field names won that collapse — it is the
deliverable people read, and renaming its fields would edit the artifact and
move its digest for nothing. The three closed sets are the delivered enums
(`GroundTruthClass`, `ContaminationClass`, `CheckKind`), read and not reshaped:
the file's vocabulary is the harness's vocabulary, and a second spelling of
`computable` is the drift the pinned file exists to close.

What this module still owns is the **file container** — the release label, the
digest, and the rule that a `refines` may dangle. The delivered
:class:`~genomics_harness.catalogue.Catalogue` is a different container over the
same checks, with a different rule (it rejects a dangling `refines`), and the
two stay separate.

**What the digest covers.** Every field except `content_sha256` itself, through
:func:`~genomics_harness.digest.canonical_json` — the path every other digested
axis uses (§7). It is taken over the *parsed* content, so reindenting the file or
reordering its keys leaves the digest alone and changing a single character of a
check's `text` does not. :meth:`CatalogueFile.verify` raises on a mismatch rather
than warning: a file edited without its digest recomputed is exactly the failure
§7 puts every digest in the log to catch, and a warning is a digest nobody reads.

**A dangling `refines` is valid, and a chain terminates at an absent target.**
A subset of this catalogue must stay loadable, so `refines` is not resolved
against the checks present. That is where this container and the delivered
:class:`~genomics_harness.catalogue.Catalogue` deliberately differ — the latter
rejects a `refines` pointing nowhere. `refinement_chains` already drops a link
whose target is out of the set it was handed, so the chain terminates without a
special case.

**The read surface is deliberately the one
:func:`~genomics_harness.judge_prompt.refinement_chains` needs** — `in`, `get`
and `ids`. That function is annotated for `Catalogue` but requires only those
three, so it runs over a `CatalogueFile` unchanged, and the B6 → B5 → B1 nest
comes off the real file rather than off a fixture.

**Locating the file.** It lives inside the package, as declared package data, so
the default path is `__file__`'s own directory and resolves identically for an
editable tree and an installed wheel. The directory is `data/` and not `checks/`:
a `checks/` directory beside `checks.py` is shadowed by the module today and
would shadow *it* the day someone adds an `__init__.py`, which is a failure with
no error until there is a confusing one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .checks import Check
from .digest import sha256_digest

__all__ = [
    "CATALOGUE_PATH",
    "DIGEST_PLACEHOLDER_SUFFIX",
    "CatalogueDigestError",
    "CatalogueFile",
    "catalogue_digest",
    "load_catalogue",
]

CATALOGUE_PATH: Final = Path(__file__).resolve().parent / "data" / "catalogue-c1.json"
"""The pinned catalogue, shipped inside the package as `data/catalogue-c1.json`.

**The filename carries the release string and so does
:attr:`CatalogueFile.catalogue_release`** — `c1`, `c2`, …, never `vN`, which
throughout this project means a document revision (§4, §0 v21). Releases
accumulate on disk rather than overwriting, because re-scoring a deferred
generation log needs the release that produced its verdicts.

**One release exists, and this constant is how it is found.** There is no
release-selection machinery and none is wanted until a `c2` is actually
created: a chooser over a set of one is a mechanism whose only exercise is a
test written for it. The path stays overridable — every reader takes
``path=None`` and falls back here — which is what a second release will use on
the day it exists.

The rename did not move the digest: it is taken over parsed content and the
model carries no path field (§0 v21). A test asserts that directly, because
"the digest does not cover the filename" is the kind of claim that is obvious
until a release depends on it.
"""

DIGEST_PLACEHOLDER_SUFFIX: Final = "-unimplemented"
"""§0 v13's convention for an axis recorded before it exists.

`content_sha256` shipped as ``"sha256-unimplemented"``. A value still carrying
the suffix is reported as never computed rather than as a mismatch, because the
two have different fixes: one is a missing step, the other is an edited file.
"""


class CatalogueDigestError(ValueError):
    """The recorded ``content_sha256`` is not the digest of the file's content."""


class CatalogueFile(BaseModel):
    """The catalogue as the pinned data file carries it.

    ``catalogue_release`` is the human-assigned label; ``content_sha256`` is
    derived from everything else and is what decides whether two verdicts were
    produced against the same rubric. They are separate on purpose — a version
    label can be applied to an edited catalogue by mistake, a digest cannot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalogue_release: str
    """The release label — ``"c1"``, ``"c2"``, … Not authoritative on its own.

    Never ``vN``: throughout this project that spelling means a document
    revision (§4). The label names the release and the digest distinguishes
    contents within it, so during development the label sits still while the
    digest moves freely, and at freeze the label is what a verdict cites.
    """

    content_sha256: str
    """The recorded digest, checked by :meth:`verify`. Outside :attr:`digest`."""

    notes: str | None
    """Required, and may be ``None``. Covered by the digest: a note is a
    statement about what the checks mean, and revising one is a revision of the
    catalogue."""

    checks: list[Check]

    @field_validator("catalogue_release", "content_sha256")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _ids_are_unique(self) -> CatalogueFile:
        seen: set[str] = set()
        for check in self.checks:
            if check.id in seen:
                raise ValueError(f"duplicate check id {check.id!r}")
            seen.add(check.id)
        return self

    # -- identity ---------------------------------------------------------

    @property
    def digest(self) -> str:
        """SHA-256 over the canonical serialisation of everything but the digest.

        ``content_sha256`` is excluded because it cannot cover itself. Everything
        else is in, ``catalogue_release`` and ``notes`` included: relabelling a
        catalogue is a change to it, and a report citing a version label must not
        be able to cite two different contents under it.
        """
        return sha256_digest(self.model_dump(mode="json", exclude={"content_sha256"}))

    def verify(self) -> None:
        """Raise unless :attr:`content_sha256` is the digest of the content.

        Raises:
            CatalogueDigestError: The recorded digest is the placeholder, or does
                not match.
        """
        computed = self.digest
        if self.content_sha256 == computed:
            return
        if self.content_sha256.endswith(DIGEST_PLACEHOLDER_SUFFIX):
            raise CatalogueDigestError(
                f"catalogue {self.catalogue_release!r} still carries the "
                f"{self.content_sha256!r} placeholder, so its digest has never "
                f"been computed. Its content digests to {computed}."
            )
        raise CatalogueDigestError(
            f"catalogue {self.catalogue_release!r} records content_sha256 "
            f"{self.content_sha256!r}, but its content digests to {computed}. "
            "The file was edited without recomputing the digest, or the digest "
            "was carried over from another catalogue."
        )

    # -- access -----------------------------------------------------------

    @property
    def ids(self) -> list[str]:
        """Check ids in declaration order.

        Declaration order, not sorted order: it is the key order of every verdict
        dict built from this catalogue, and a stable order makes two logs
        diffable by eye. The digest does not depend on it being sorted.
        """
        return [check.id for check in self.checks]

    @property
    def in_scope_ids(self) -> list[str]:
        """Ids of the checks in scope, in declaration order."""
        return [check.id for check in self.checks if check.in_scope]

    def get(self, check_id: str) -> Check:
        """Return the check with this id.

        Raises:
            KeyError: No such check.
        """
        for check in self.checks:
            if check.id == check_id:
                return check
        raise KeyError(f"no check {check_id!r} in catalogue {self.catalogue_release!r}")

    def __contains__(self, check_id: object) -> bool:
        return any(check.id == check_id for check in self.checks)

    def __iter__(self) -> Iterator[Check]:  # type: ignore[override]
        return iter(self.checks)

    def __len__(self) -> int:
        return len(self.checks)


def _parse(path: Path) -> CatalogueFile:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no catalogue file at {path}. The catalogue ships as package data "
            "inside `genomics_harness/data/`; an install that dropped it, or a "
            "path passed explicitly, is what gets you here."
        ) from None
    return CatalogueFile.model_validate(json.loads(text))


def load_catalogue(path: Path | str | None = None) -> CatalogueFile:
    """Load the pinned catalogue and verify its recorded digest.

    Args:
        path: The file to read. Defaults to :data:`CATALOGUE_PATH`.

    Returns:
        The validated catalogue, whose ``content_sha256`` is the digest of its
        own content.

    Raises:
        FileNotFoundError: No file at ``path``.
        json.JSONDecodeError: The file is not JSON.
        pydantic.ValidationError: An unknown field, a missing field, or a value
            outside one of the three closed sets.
        CatalogueDigestError: The recorded digest does not match the content.
    """
    catalogue = _parse(CATALOGUE_PATH if path is None else Path(path))
    catalogue.verify()
    return catalogue


def catalogue_digest(path: Path | str | None = None) -> str:
    """The digest a catalogue file's content *should* record, without checking it.

    The one door around :func:`load_catalogue`'s verification, and it is a door
    to a string rather than to a catalogue: recomputing the digest after a
    deliberate edit is a real need, obtaining an unverified catalogue object is
    not.

    Args:
        path: The file to read. Defaults to :data:`CATALOGUE_PATH`.
    """
    return _parse(CATALOGUE_PATH if path is None else Path(path)).digest
