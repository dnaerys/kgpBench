"""Canonical serialisation and digest stability.

The digest feeds `Task(version=)`. An unstable one produces spurious `eval_set`
re-runs and false provenance mismatches; a colliding one makes a cross-round
comparison silently wrong. Both are quiet, so they are tested rather than
assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from genomics_harness import CanonicalisationError, canonical_json, sha256_digest
from genomics_harness.digest import SHORT_DIGEST_LENGTH, short_digest

# -- ordering -------------------------------------------------------------


def test_insertion_order_does_not_reach_the_digest():
    forward = {"alpha": 1, "beta": 2, "gamma": 3}
    backward = {"gamma": 3, "beta": 2, "alpha": 1}
    assert list(forward) != list(backward)  # the insertion orders really differ
    assert sha256_digest(forward) == sha256_digest(backward)


def test_nested_insertion_order_does_not_reach_the_digest():
    left = {"outer": [{"a": 1, "b": {"x": 1, "y": 2}}]}
    right = {"outer": [{"b": {"y": 2, "x": 1}, "a": 1}]}
    assert sha256_digest(left) == sha256_digest(right)


def test_list_order_does_reach_the_digest():
    """Sequences are ordered; reordering instances is a change."""
    assert sha256_digest([1, 2]) != sha256_digest([2, 1])


def test_canonical_json_is_sorted_and_compact():
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == (
        '{"a":[1,{"c":3,"d":2}],"b":1}'
    )


# -- process restarts -----------------------------------------------------

_SUBPROCESS = textwrap.dedent(
    """
    import json, sys
    from genomics_harness import sha256_digest
    payload = json.loads(sys.argv[1])
    print(sha256_digest(payload))
    """
)


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345", "random"])
def test_digest_is_stable_across_process_restarts(hash_seed):
    """A digest that depends on PYTHONHASHSEED would be stable within a process
    and different between runs — the worst possible failure shape."""
    payload = {
        "z": "last",
        "a": "first",
        "nested": {"q": [1, 2.5, None, True], "p": "café"},
        "ints": list(range(20)),
    }
    argument = json.dumps(payload)
    outputs = set()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS, argument],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1
    assert outputs.pop() == sha256_digest(payload)


# -- unicode --------------------------------------------------------------


# Written as escapes, not as literal accented characters: an editor or a git
# filter that normalises this file must not be able to make these tests pass by
# quietly making the two spellings identical in the source.
COMPOSED = "caf\u00e9"  # e-acute as one codepoint
DECOMPOSED = "cafe\u0301"  # e followed by U+0301 combining acute


def test_the_two_spellings_really_do_differ():
    assert COMPOSED != DECOMPOSED
    assert (len(COMPOSED), len(DECOMPOSED)) == (4, 5)


def test_nfc_normalisation_of_values():
    assert sha256_digest({"gene": COMPOSED}) == sha256_digest({"gene": DECOMPOSED})


def test_nfc_normalisation_of_keys():
    assert sha256_digest({COMPOSED: 1}) == sha256_digest({DECOMPOSED: 1})


def test_keys_colliding_under_nfc_raise():
    with pytest.raises(CanonicalisationError, match="normalise"):
        sha256_digest({COMPOSED: 1, DECOMPOSED: 2})


def test_non_ascii_is_emitted_verbatim_not_escaped():
    greek = "\u03b1\u03b2\u03b3"
    assert canonical_json({"k": greek}) == '{"k":"' + greek + '"}'
    assert canonical_json({"k": greek}).encode("utf-8").decode("utf-8")


# -- numbers --------------------------------------------------------------


def test_float_formatting_is_shortest_round_trip():
    assert canonical_json(0.1 + 0.2) == "0.30000000000000004"
    assert canonical_json(1e300) == "1e+300"


def test_negative_zero_folds_to_zero():
    assert sha256_digest(-0.0) == sha256_digest(0.0)


def test_int_and_float_stay_distinct():
    """A ground-truth count and a ground-truth ratio are different claims."""
    assert sha256_digest(1) != sha256_digest(1.0)
    assert canonical_json(1) == "1"
    assert canonical_json(1.0) == "1.0"


def test_bool_is_not_an_int():
    assert canonical_json(True) == "true"
    assert canonical_json(1) == "1"
    assert sha256_digest({"x": True}) != sha256_digest({"x": 1})


def test_bool_keys_in_nested_positions():
    assert canonical_json([True, False, 0, 1]) == "[true,false,0,1]"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_raise(value):
    """They have no JSON representation, so they would not survive a log round
    trip — and a digest over a value that changes on the way to disk is a lie."""
    with pytest.raises(CanonicalisationError, match="non-finite"):
        sha256_digest({"x": value})


def test_large_ints_are_exact():
    big = 2**80 + 1
    assert canonical_json(big) == str(big)


# -- rejected types -------------------------------------------------------


def test_sets_are_rejected():
    with pytest.raises(CanonicalisationError, match="no defined order"):
        sha256_digest({"x": {1, 2, 3}})


def test_unknown_types_are_rejected():
    class Thing:
        pass

    with pytest.raises(CanonicalisationError, match="not canonically serialisable"):
        sha256_digest({"x": Thing()})


def test_non_string_keys_are_rejected():
    with pytest.raises(CanonicalisationError, match="only str keys"):
        sha256_digest({1: "a"})


def test_error_message_names_the_path():
    with pytest.raises(CanonicalisationError, match=r"\$\.outer\[1\]\.inner"):
        sha256_digest({"outer": [0, {"inner": float("nan")}]})


# -- escaping -------------------------------------------------------------


def test_control_characters_and_quotes_escape():
    assert canonical_json({'a"b': "c\nd\te\\f"}) == '{"a\\"b":"c\\nd\\te\\\\f"}'


def test_lone_surrogates_escape_rather_than_crash():
    """A surrogate cannot be UTF-8 encoded; escaping keeps the digest defined."""
    assert sha256_digest("\ud800")  # does not raise
    assert "\\ud800" in canonical_json("\ud800")


# -- shape ----------------------------------------------------------------


def test_short_digest_is_a_prefix_of_the_full_one():
    payload = {"a": 1}
    assert len(short_digest(payload)) == SHORT_DIGEST_LENGTH
    assert sha256_digest(payload).startswith(short_digest(payload))


def test_digest_changes_when_anything_changes():
    base = {"id": "x", "prompt": "p", "attributes": {"ac": 1}}
    assert sha256_digest(base) != sha256_digest({**base, "prompt": "p "})
    assert sha256_digest(base) != sha256_digest({**base, "attributes": {"ac": 2}})
    assert sha256_digest(base) != sha256_digest({**base, "extra": None})
