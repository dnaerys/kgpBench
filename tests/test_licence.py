"""The licence file — that it is there, and that it is the canonical text.

The harness is Apache License 2.0. What that needs from the distribution is a
licence file at the distribution root, a declaration in packaging metadata, and
both of those present in the built wheel and sdist. This module covers the
**file**; the built artifacts are covered where a builder exists, which is
`release/publish.py` (there is no `setuptools`, `build` or `pip` in this
package's own venv, so a test here cannot build a wheel to look inside).

**What the pins are for.** Licence-identification tooling matches against the
canonical form, so this file's value is in being byte-identical to the text
Apache publishes rather than in saying roughly the same thing. Three edits
would break that while leaving the file looking entirely correct, and each has a
pin naming it:

* reflowing the text — the copy in SPDX's ``license-list-data`` is left-aligned
  with the indentation stripped, which is a plausible thing to paste in and is
  **not** the canonical form. Caught by the digest and the length.
* deleting the appendix — the ``APPENDIX: How to apply the Apache License to
  your work`` boilerplate reads like instructions aimed at us rather than part
  of the licence, and it is part of the canonical text. Caught by its heading.
* filling in the appendix's placeholder — ``Copyright [yyyy] [name of copyright
  owner]`` is a template and stays a template. Substituting the real holder is
  the most likely well-meant edit of the three. Caught by the literal.

The digest is pinned as a literal and not computed from the file, because the
file is what is under test (specification §15). It was taken from
``https://www.apache.org/licenses/LICENSE-2.0.txt`` and corroborated offline
against seven packages in this venv that ship a byte-identical copy.

**A missing file fails rather than skips.** Same ruling as the packaging
assertions in `test_catalogue_file` and `test_composition` (§0 v21): a tree in
which the licence moved or was dropped is exactly what this exists to catch, and
a skip reports that as "not applicable". The wheel ships no tests, so the
layouts this runs in are the development tree, the published root and an
unpacked sdist — the licence is at the package root in all three. It can also
run against a non-editable install with the tests taken from a clone, and there
the path is genuinely absent; the message says so rather than the test
vanishing.

**And no `NOTICE` file, deliberately.** Apache §4(d) makes a NOTICE, once
present, an obligation on every downstream redistributor to carry it — and the
harness is built to be forked and brought instances to (§16), so the one thing
it must not do is attach a perpetual string to every fork. Nothing needs a
NOTICE for the licence to be correct, so the guard is that one never appears.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import kgpbench

CANONICAL_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
"""sha256 of the canonical Apache License 2.0 text, as Apache publishes it."""

CANONICAL_BYTES = 11358
"""Its length. Pinned beside the digest because it names the failure readably.

A digest mismatch says only *different*; a length of 10,280 says *the SPDX
reflow*, which is the substitution worth recognising on sight.
"""

COPYRIGHT_LINE = "Copyright 2026 Dnaerys Pty Ltd"
"""The notice as the README carries it, and from there it publishes.

It reaches the wheel's own ``METADATA`` through ``readme = "README.md"``, so the
README is both where a reader finds it and how it gets into the distribution.

**The string is no longer unique in the tree, and this is not the assertion that
counts it.** Every module under ``src/kgpbench`` now opens with a two-line SPDX
attribution marker carrying this same line, so a fork taking only ``src/`` carries
a notice away with it — Apache §4(c) obliges a redistributor to retain the
attribution notices present in the Source form, and until those markers there were
none to retain. That is a second carrier for a second purpose, not a duplicate of
this one: the subject below is the README, which is the only route by which the
notice reaches packaging metadata. Counting the source occurrences here would bind
this assertion to the module count and fail it on every module added.
"""


def package_root() -> Path:
    """The directory holding ``pyproject.toml``, ``src`` and ``tests``.

    Resolved upward from the imported package, two levels above
    ``src/kgpbench``, and no further: the same source is developed in a tree
    where this directory sits beside its siblings and published as a snapshot
    whose root *is* this directory (§16), so anything reaching above it is
    correct in one layout and wrong in the other.
    """
    return Path(kgpbench.__file__).resolve().parents[2]


def licence_text() -> str:
    """The licence file's contents.

    Raises:
        AssertionError: There is no licence file at the package root.
    """
    path = package_root() / "LICENSE"
    assert path.is_file(), (
        f"no LICENSE at {path}. The harness is Apache License 2.0 and the file "
        "at the distribution root is how a distribution states that, so an "
        "absent file is the failure this module exists to catch rather than a "
        "reason to skip. If the package was installed non-editably and these "
        "tests came from a clone, the licence is in the installed "
        "`*.dist-info/licenses/`; run the suite against the tree instead."
    )
    return path.read_text(encoding="utf-8")


def test_the_licence_is_the_canonical_apache_2_text() -> None:
    """Byte-identical to what Apache publishes — digest and length both pinned."""
    raw = (package_root() / "LICENSE").read_bytes()
    assert licence_text()  # the readable refusal, before the opaque one

    assert len(raw) == CANONICAL_BYTES, (
        f"LICENSE is {len(raw)} bytes, not {CANONICAL_BYTES}. The canonical text "
        "is indented and hard-wrapped; 10280 bytes is SPDX's reflowed copy, "
        "which licence-identification tooling does not match"
    )
    assert hashlib.sha256(raw).hexdigest() == CANONICAL_SHA256, (
        "LICENSE is not the canonical Apache License 2.0 text. Re-fetch it from "
        "https://www.apache.org/licenses/LICENSE-2.0.txt rather than editing it"
    )


def test_the_appendix_boilerplate_is_present_and_unfilled() -> None:
    """The appendix ships, and its placeholder stays a placeholder.

    The digest above already covers both. These assert the two specific edits
    separately so that a failure names which one happened instead of reporting
    only that some byte moved.
    """
    text = licence_text()

    assert "APPENDIX: How to apply the Apache License to your work." in text, (
        "the appendix boilerplate is missing. It reads as instructions to the "
        "licensor rather than as licence text, and it is part of the canonical "
        "text all the same"
    )
    assert "Copyright [yyyy] [name of copyright owner]" in text, (
        "the appendix's placeholder has been filled in. It is a template for a "
        "downstream licensor to copy, not this project's own notice — ours is "
        "the one line in README.md"
    )
    assert COPYRIGHT_LINE not in text, (
        f"{COPYRIGHT_LINE!r} is in LICENSE. The canonical text carries no "
        "holder, and adding one makes the file non-canonical"
    )


def test_the_copyright_line_is_in_the_readme_exactly_once() -> None:
    """One line, in the file that becomes the project page and the METADATA."""
    readme = package_root() / "README.md"
    assert readme.is_file(), f"no README.md at {readme}"
    text = readme.read_text(encoding="utf-8")

    assert text.count(COPYRIGHT_LINE) == 1, (
        f"README.md carries {COPYRIGHT_LINE!r} {text.count(COPYRIGHT_LINE)} "
        "times; it is one line, once, in this file"
    )
    carrier = [line for line in text.splitlines() if COPYRIGHT_LINE in line]
    assert len(carrier) == 1 and "LICENSE" in carrier[0], (
        f"the copyright line does not point at the licence file: {carrier!r}. "
        "A reader who finds the notice should reach the grant from it"
    )


def test_no_notice_file_publishes() -> None:
    """Apache §4(d): a NOTICE binds every downstream redistributor. There is none.

    **The distribution root only, and that is the whole point of the scope.**
    setuptools' default ``license_files`` is a set of globs rooted here and not
    recursive — ``LICEN[CS]E*``, ``COPYING*``, ``NOTICE*``, ``AUTHORS*`` — so a
    file dropped *here* is carried into the wheel and the sdist with nothing
    declaring it, and a file anywhere else is not. This asserts over the one
    location where the hazard is automatic.

    It was written as ``root.rglob("NOTICE*")`` and that was wrong in a way only
    running it found: the subject became *whatever is under the directory* (§15),
    which picks up build residue, and under a non-editable install resolves into
    ``site-packages`` and reported seven other distributions' NOTICE files. The
    corpus-wide statement — every file that actually publishes — belongs to the
    release guard, which computes the publishing set; this cannot, and should not
    guess at it.

    ``sorted(...)`` materialises the glob: a bare generator is always truthy, so
    ``assert not root.glob(...)`` cannot fail (§15).
    """
    root = package_root()
    found = sorted(path.name for path in root.glob("NOTICE*") if path.is_file())
    assert found == [], (
        f"NOTICE file(s) at {root}: {found}. Apache §4(d) makes a NOTICE, once "
        "shipped, an obligation on everyone who redistributes this — and the "
        "harness exists to be forked and brought instances to (§16). At the "
        "distribution root it also ships without being declared, through "
        "setuptools' default license-files glob. The licence needs no NOTICE"
    )
