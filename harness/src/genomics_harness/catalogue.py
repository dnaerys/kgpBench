"""The check catalogue — a versioned collection of checks with a content digest.

The catalogue is passed as a `@scorer` factory argument, which is what makes a
scoring log record *which* catalogue produced its verdicts. That route imposes a
serialisation constraint, and the constraint is sharper than "JSON-serialisable":

* on the way out, `registry_tag` binds the factory's arguments
  (`_util/registry.py:149-181`) and they land in `EvalScorer.options:
  dict[str, Any]` (`log/_log.py`), serialised by pydantic;
* on the way back, `resolve_scorers` splats them into the factory
  (`_eval/score.py:531-577`, ``**(score.options or {})``) — as **plain JSON
  types**, because that is what came off the disk.

So a factory that accepts a `Catalogue` model gets a `Catalogue` in-run and a
`dict` on re-score, and any code that touched a model attribute breaks only in
the deferred path. The rule this module enforces instead:

    the wire type is a plain ``dict``; :meth:`Catalogue.of` converts.

:func:`catalogue_arg` produces the dict to pass, :meth:`Catalogue.of` accepts
either form. ``tests/test_catalogue_roundtrip.py`` verifies both directions
against a real log rather than asserting them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .checks import Check
from .digest import sha256_digest, short_digest

__all__ = ["Catalogue", "catalogue_arg"]


class Catalogue(BaseModel):
    """A versioned, digested collection of checks.

    ``release`` is the delivered catalogue's release label — the pinned file's
    own ``catalogue_release`` (§4) — and ``digest`` is derived from the content
    and is what actually decides whether two verdicts are comparable. They are
    separate on purpose: a release label can be applied to an edited catalogue
    by mistake, a digest cannot.

    Constructing an inconsistent catalogue — a duplicate id, a `refines` pointing
    nowhere — raises `pydantic.ValidationError`, which is a `ValueError`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    release: str
    """The release label — ``"c1"``, ``"c2"``, … Not authoritative on its own.

    Named for what it holds. Catalogue releases are ``cN`` (§4): ``vN`` is a
    document revision and ``rN`` is the renderer, so a field called ``version``
    holding ``"c1"`` was a name that did not resolve (§7).
    """

    checks: list[Check] = Field(default_factory=list)

    notes: str | None = None

    @field_validator("release")
    @classmethod
    def _release_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("catalogue release must not be blank")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Catalogue:
        seen: set[str] = set()
        for check in self.checks:
            if check.id in seen:
                raise ValueError(f"duplicate check id {check.id!r}")
            seen.add(check.id)
        for check in self.checks:
            if check.refines is not None and check.refines not in seen:
                raise ValueError(
                    f"check {check.id!r} refines {check.refines!r}, which is not "
                    "in the catalogue"
                )
        return self

    # -- identity ---------------------------------------------------------

    @property
    def digest(self) -> str:
        """SHA-256 over the canonical serialisation of the whole catalogue.

        Includes ``release`` and ``notes``: relabelling a catalogue is a change
        to the catalogue, and a report that cites a release label must not be
        able to cite two different contents under it.
        """
        return sha256_digest(self.model_dump(mode="json"))

    @property
    def short(self) -> str:
        """First 16 hex characters of :attr:`digest`."""
        return short_digest(self.model_dump(mode="json"))

    # -- access -----------------------------------------------------------

    @property
    def ids(self) -> list[str]:
        """Check ids in declaration order.

        Declaration order, not sorted order: this is the key order of every
        verdict dict the harness produces, and a stable order makes two logs
        diffable by eye. The digest does not depend on it being sorted.
        """
        return [check.id for check in self.checks]

    def __iter__(self) -> Iterator[Check]:  # type: ignore[override]
        return iter(self.checks)

    def __len__(self) -> int:
        return len(self.checks)

    def __contains__(self, check_id: object) -> bool:
        return any(check.id == check_id for check in self.checks)

    def get(self, check_id: str) -> Check:
        """Return the check with this id.

        Raises:
            KeyError: No such check.
        """
        for check in self.checks:
            if check.id == check_id:
                return check
        raise KeyError(f"no check {check_id!r} in catalogue {self.release!r}")

    def in_scope(self) -> Catalogue:
        """A catalogue holding only the checks marked :attr:`Check.in_scope`.

        **The digest distinguishes the filtered catalogue; the label does not.**
        Filtering yields a different object with a different :attr:`digest`, and
        that digest is what provenance compares — so the release label carries
        through unchanged, because the subset is the same release rather than a
        second one (§0 v25 item 12, §0 v27 item 5). A suffixed label would be a
        second name for what is not a second release, which is the shape v25
        retired when it deleted ``c1-derived-rubric``.

        **This is the rule the delivered catalogue follows, and this method is
        not yet the thing that performs it** (§4, §14.40). A judge's ``catalogue``
        argument is the pinned file's in-scope subset labelled with the file's own
        release, unsuffixed; the step that produces it starts from a
        :class:`~genomics_harness.catalogue_file.CatalogueFile`, and this method
        takes a :class:`Catalogue`, so reaching it from :func:`load_catalogue`
        means constructing a catalogue over all 22 checks first. Closing that gap
        is a build item, not a property of this method.

        Named for the field it filters on. ``implemented`` was the pre-v25
        spelling of :attr:`Check.in_scope` and outlived it here by one round.
        """
        return Catalogue(
            release=self.release,
            checks=[c for c in self.checks if c.in_scope],
            notes=self.notes,
        )

    # -- the scorer-argument wire form ------------------------------------

    @classmethod
    def of(cls, value: Catalogue | Mapping[str, Any]) -> Catalogue:
        """Accept either a :class:`Catalogue` or its wire form.

        A `@scorer` factory calls this on its ``catalogue`` argument, which is a
        `Catalogue` when constructed in-process and a plain `dict` when
        `_eval/score.py` rebuilds the scorer from a stored log header.
        """
        if isinstance(value, Catalogue):
            return value
        if isinstance(value, Mapping):
            return cls.model_validate(dict(value))
        raise TypeError(
            f"catalogue must be a Catalogue or a mapping, got {type(value).__name__}"
        )

    def to_arg(self) -> dict[str, Any]:
        """The wire form: plain JSON types, suitable as a scorer factory argument."""
        return self.model_dump(mode="json")

    @classmethod
    def from_checks(cls, release: str, checks: Iterable[Check]) -> Catalogue:
        return cls(release=release, checks=list(checks))


def catalogue_arg(catalogue: Catalogue) -> dict[str, Any]:
    """The value to pass as a `@scorer` factory's ``catalogue=`` argument.

    Passing the model itself works in-run and degrades to a `dict` on re-score.
    Passing this instead makes both paths identical.
    """
    return catalogue.to_arg()
