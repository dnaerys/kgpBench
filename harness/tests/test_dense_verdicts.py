"""The dense verdict builder — the single function that makes the key-set invariant hold.

The rule it enforces has a direction: build the dict from the catalogue, never
from what the judge mentioned. Every test here is about that direction.
"""

from __future__ import annotations

import math

import pytest

from genomics_harness import (
    Catalogue,
    Outcome,
    Verdict,
    VerdictSet,
    dense_verdicts,
    outcome_of_score_value,
    score_value_of_outcome,
)


def test_missing_keys_become_unreachable(catalogue):
    dense, discarded = dense_verdicts(catalogue, {"A1": Outcome.PERFORMED})
    assert list(dense) == catalogue.ids
    assert dense["A1"].outcome is Outcome.PERFORMED
    assert all(
        dense[c].outcome is Outcome.NOT_REACHABLE for c in catalogue.ids if c != "A1"
    )
    assert discarded == {}


def test_the_judge_cannot_widen_the_key_set(catalogue):
    """A judge inventing a check must not add a column that later voids results."""
    dense, discarded = dense_verdicts(
        catalogue, {"A1": Outcome.PERFORMED, "Z9": Outcome.PERFORMED}
    )
    assert "Z9" not in dense
    assert list(dense) == catalogue.ids
    assert "Z9" in discarded
    assert "not in catalogue" in discarded["Z9"]


def test_unreachable_checks_are_forced_regardless_of_what_the_judge_said(catalogue):
    dense, discarded = dense_verdicts(
        catalogue,
        {"A1": Outcome.PERFORMED, "D1": Outcome.PERFORMED},
        reachable=["A1"],
    )
    assert dense["A1"].outcome is Outcome.PERFORMED
    assert dense["D1"].outcome is Outcome.NOT_REACHABLE
    assert "does not make" in discarded["D1"]


def test_key_order_is_catalogue_order(catalogue):
    produced = {c: Outcome.PERFORMED for c in reversed(catalogue.ids)}
    dense, _ = dense_verdicts(catalogue, produced)
    assert list(dense) == catalogue.ids


def test_accepts_verdicts_outcomes_and_names(catalogue):
    dense, _ = dense_verdicts(
        catalogue,
        {
            "A1": Verdict(check_id="A1", outcome=Outcome.PARTIAL, rationale="thin"),
            "B1": Outcome.PERFORMED,
            "B5": "not_performed",
        },
    )
    assert dense["A1"].outcome is Outcome.PARTIAL
    assert dense["A1"].rationale == "thin"
    assert dense["B1"].outcome is Outcome.PERFORMED
    assert dense["B5"].outcome is Outcome.NOT_PERFORMED


def test_a_verdict_carrying_the_wrong_check_id_is_corrected(catalogue):
    dense, _ = dense_verdicts(
        catalogue, {"A1": Verdict(check_id="B1", outcome=Outcome.PERFORMED)}
    )
    assert dense["A1"].check_id == "A1"


def test_unknown_verdict_type_raises(catalogue):
    with pytest.raises(TypeError, match="expected Verdict"):
        dense_verdicts(catalogue, {"A1": 1.0})  # type: ignore[dict-item]


def test_empty_judge_output_still_produces_every_key(catalogue):
    dense, _ = dense_verdicts(catalogue, {})
    assert list(dense) == catalogue.ids
    assert {v.outcome for v in dense.values()} == {Outcome.NOT_REACHABLE}


def test_two_trajectories_with_different_reachability_agree_on_keys(catalogue):
    left, _ = dense_verdicts(catalogue, {"A1": Outcome.PERFORMED}, reachable=["A1"])
    right, _ = dense_verdicts(catalogue, {"D1": Outcome.PARTIAL}, reachable=["D1"])
    assert list(left) == list(right) == catalogue.ids


# -- the float encoding ---------------------------------------------------


def test_unreachable_encodes_as_nan():
    assert math.isnan(score_value_of_outcome(Outcome.NOT_REACHABLE))


@pytest.mark.parametrize(
    ("outcome", "encoded"),
    [
        (Outcome.PERFORMED, 1.0),
        (Outcome.PARTIAL, 0.5),
        (Outcome.NOT_PERFORMED, 0.0),
    ],
)
def test_outcome_encoding_round_trips(outcome, encoded):
    assert score_value_of_outcome(outcome) == encoded
    assert outcome_of_score_value(encoded) is outcome


def test_none_decodes_to_unreachable():
    """`float("nan")` reads back off disk as `None`; both mean the same thing."""
    assert outcome_of_score_value(None) is Outcome.NOT_REACHABLE
    assert outcome_of_score_value(float("nan")) is Outcome.NOT_REACHABLE


@pytest.mark.parametrize("value", [0.25, 2.0, -1.0, "C", True, [], {}])
def test_unrecognised_encodings_raise_rather_than_coerce(value):
    with pytest.raises(ValueError):
        outcome_of_score_value(value)


# -- VerdictSet -----------------------------------------------------------


def _verdict_set(catalogue: Catalogue, **kwargs) -> VerdictSet:
    dense, discarded = dense_verdicts(catalogue, kwargs)
    return VerdictSet(
        judge="judge_test",
        catalogue_release=catalogue.release,
        catalogue_digest=catalogue.digest,
        renderer_version="r0-test",
        verdicts=dense,
        discarded=discarded,
    )


def test_verdict_set_score_value_is_dense_and_nan_for_unreachable(catalogue):
    verdict_set = _verdict_set(catalogue, A1=Outcome.PERFORMED)
    value = verdict_set.to_score_value()
    assert list(value) == catalogue.ids
    assert value["A1"] == 1.0
    assert all(math.isnan(value[c]) for c in catalogue.ids if c != "A1")


def test_validate_dense_accepts_a_built_set(catalogue):
    _verdict_set(catalogue, A1=Outcome.PERFORMED).validate_dense(catalogue)


def test_validate_dense_rejects_a_hand_built_ragged_set(catalogue):
    verdict_set = VerdictSet(
        judge="judge_test",
        catalogue_release=catalogue.release,
        catalogue_digest=catalogue.digest,
        renderer_version="r0-test",
        verdicts={"A1": Verdict(check_id="A1", outcome=Outcome.PERFORMED)},
    )
    with pytest.raises(ValueError, match="not dense"):
        verdict_set.validate_dense(catalogue)
