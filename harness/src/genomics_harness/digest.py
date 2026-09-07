"""Canonical serialisation and content digests.

Every provenance digest in the harness is produced here, so that two structurally
identical objects always produce the same hex string and two different ones never
do. The failure this guards against is asymmetric and quiet: an unstable digest
produces spurious `eval_set` re-runs and false provenance mismatches, while a
digest that collides across genuinely different inputs makes a cross-round
comparison silently wrong.

Canonicalisation rules, in the order applied:

1. **Unicode.** Every string — keys and values alike — is normalised to NFC
   before anything else, so `"caf\\u00e9"` and `"cafe\\u0301"` digest identically.
   Two keys of one mapping that normalise to the same string are a collision and
   raise.
2. **Key order.** Mapping keys are sorted by codepoint *after* normalisation.
   Insertion order therefore cannot reach the output.
3. **Numbers.** `bool` is tested before `int` (`isinstance(True, int)` is true).
   Integers render exactly. Floats render with `repr()`, which is the shortest
   string that round-trips, and `-0.0` is folded to `0.0`. Non-finite floats
   raise: JSON has no representation for them, so any round trip through a log
   would change them.
4. **Types.** Only `None`, `bool`, `int`, `float`, `str`, sequences and mappings
   are accepted. Sets are rejected because they have no defined order; anything
   else is rejected because its serialisation would be an implicit decision made
   somewhere else.

`int` and `float` stay distinct: `1` and `1.0` produce different digests. That
is deliberate — a ground-truth count and a ground-truth ratio are different
claims, and pydantic preserves the distinction in `model_dump(mode="json")`.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final

__all__ = [
    "CanonicalisationError",
    "canonical_bytes",
    "canonical_json",
    "sha256_digest",
    "short_digest",
    "SHORT_DIGEST_LENGTH",
]

SHORT_DIGEST_LENGTH: Final = 16
"""Characters of hex kept by :func:`short_digest`.

Sixteen hex characters is 64 bits. It is what reaches ``Task(version=)``, which
is a human-facing identifier that also has to be collision-free across the few
hundred instance sets this harness will ever produce.
"""


class CanonicalisationError(TypeError):
    """A value cannot be canonically serialised."""


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _render_float(value: float) -> str:
    if not math.isfinite(value):
        raise CanonicalisationError(
            f"non-finite float {value!r} cannot be canonically serialised: "
            "JSON has no representation for it, so it would not survive a log "
            "round trip. Use None for an absent value."
        )
    if value == 0.0:
        # collapse -0.0, whose repr differs but whose value does not
        return "0.0"
    return repr(value)


def _escape(text: str) -> str:
    out = ['"']
    for ch in text:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or 0xD800 <= ord(ch) <= 0xDFFF:
            # control characters, and lone surrogates which cannot be encoded
            # as UTF-8 at all
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _write(value: Any, out: list[str], path: str) -> None:
    if value is None:
        out.append("null")
        return

    # bool before int: isinstance(True, int) is True
    if isinstance(value, bool):
        out.append("true" if value else "false")
        return

    if isinstance(value, int):
        out.append(str(value))
        return

    if isinstance(value, float):
        try:
            out.append(_render_float(value))
        except CanonicalisationError as ex:
            raise CanonicalisationError(f"at {path}: {ex}") from None
        return

    if isinstance(value, str):
        out.append(_escape(_normalise(value)))
        return

    if isinstance(value, Mapping):
        normalised: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise CanonicalisationError(
                    f"at {path}: mapping key {raw_key!r} is {type(raw_key).__name__}, "
                    "only str keys can be canonically ordered"
                )
            key = _normalise(raw_key)
            if key in normalised:
                raise CanonicalisationError(
                    f"at {path}: keys {raw_key!r} and another key both normalise "
                    f"to {key!r}; the mapping is ambiguous under NFC"
                )
            normalised[key] = item
        out.append("{")
        for index, key in enumerate(sorted(normalised)):
            if index:
                out.append(",")
            out.append(_escape(key))
            out.append(":")
            _write(normalised[key], out, f"{path}.{key}")
        out.append("}")
        return

    if isinstance(value, (set, frozenset)):
        raise CanonicalisationError(
            f"at {path}: sets have no defined order and cannot be canonically "
            "serialised; convert to a sorted list at the point where the order "
            "is decided"
        )

    if isinstance(value, Sequence):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write(item, out, f"{path}[{index}]")
        out.append("]")
        return

    raise CanonicalisationError(
        f"at {path}: {type(value).__name__} is not canonically serialisable. "
        "Convert it explicitly — an implicit conversion here would be a "
        "provenance decision made in the wrong place."
    )


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to its canonical JSON text.

    Args:
        value: A JSON-shaped structure. Pydantic models must already have been
            through ``model_dump(mode="json")``.

    Returns:
        Canonical JSON: NFC-normalised, key-sorted, no insignificant whitespace.

    Raises:
        CanonicalisationError: The value contains something whose canonical form
            is not defined.
    """
    out: list[str] = []
    _write(value, out, "$")
    return "".join(out)


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON of ``value`` encoded as UTF-8."""
    return canonical_json(value).encode("utf-8")


def sha256_digest(value: Any) -> str:
    """SHA-256 of the canonical serialisation of ``value``, as lowercase hex."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def short_digest(value: Any) -> str:
    """First :data:`SHORT_DIGEST_LENGTH` hex characters of :func:`sha256_digest`."""
    return sha256_digest(value)[:SHORT_DIGEST_LENGTH]
