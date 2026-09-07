"""Invariant 1 — identical key sets across every sample, and a NaN key in `results`.

Guards against a ragged dict-valued score voiding `log.results` after every
sample has run and been paid for. The framework's behaviour is asymmetric: a key
that appears only in later samples is dropped silently, a key present in sample 1
and absent later raises inside results construction
(`_eval/task/results.py:489`, `:528`).

Both halves are exercised: the harness path holds the invariant, and the
negative control shows what happens when it does not — because an invariant test
whose subject cannot fail proves nothing.
"""

from __future__ import annotations

import math

from inspect_ai import eval as inspect_eval
from mock_harness import (
    MOCK_CATALOGUE,
    clean_model,
    mock_task,
    ragged_judge,
)

from genomics_harness import read_log


def _run(instances, tmp_path, scorers=None, epochs=3):
    logs = inspect_eval(
        mock_task(instances, scorers=scorers, epochs=epochs),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "logs"),
    )
    return logs[0]


def test_every_sample_carries_the_full_catalogue_key_set(instances, tmp_path):
    log = _run(instances, tmp_path)
    assert log.status == "success"

    samples = log.samples or []
    assert len(samples) == len(instances) * 3  # two instances, three epochs

    key_sets = set()
    for sample in samples:
        for score in (sample.scores or {}).values():
            assert isinstance(score.value, dict)
            key_sets.add(tuple(score.value))

    assert len(key_sets) == 1
    assert list(key_sets.pop()) == MOCK_CATALOGUE.ids


def test_a_nan_key_survives_to_results_as_unscored(instances, tmp_path):
    """Each instance makes a different subset reachable, so every check is NaN
    on some samples and scored on others — and none of them may go missing."""
    log = _run(instances, tmp_path)

    assert log.results is not None
    by_name = {score.name: score for score in log.results.scores}
    assert set(by_name) == set(MOCK_CATALOGUE.ids)

    # instance one makes A1 and B1 reachable, instance two makes D1 reachable;
    # three epochs each
    assert (by_name["A1"].scored_samples, by_name["A1"].unscored_samples) == (3, 3)
    assert (by_name["D1"].scored_samples, by_name["D1"].unscored_samples) == (3, 3)

    # unreachable counted as unscored, not as zero
    assert by_name["A1"].metrics["mean"].value == 1.0


def test_nan_reaches_disk_as_null_and_reads_back_as_none(instances, tmp_path):
    """The round trip the whole analysis layer is designed around."""
    log = _run(instances, tmp_path)
    stored = read_log(log.location)

    unreachable = 0
    for sample in stored.samples or []:
        for score in (sample.scores or {}).values():
            assert set(score.value) == set(MOCK_CATALOGUE.ids)
            for value in score.value.values():
                if value is None:
                    unreachable += 1
                else:
                    assert not math.isnan(value)
    # instance one reaches A1+B1 so D1 is unreachable (1 per sample);
    # instance two reaches D1 so A1+B1 are (2 per sample); three epochs each
    assert unreachable == 3 * 1 + 3 * 2


def test_the_judge_cannot_add_a_key_by_mentioning_one(instances, tmp_path):
    """`dense_judge` deliberately produces a verdict for a check outside the
    catalogue; it must be discarded rather than widening the key set."""
    log = _run(instances, tmp_path)
    for sample in log.samples or []:
        for score in (sample.scores or {}).values():
            assert "ZZ_not_in_catalogue" not in score.value
            assert "ZZ_not_in_catalogue" in (score.metadata or {})["discarded"]


def test_negative_control_ragged_keys_void_results(instances, tmp_path):
    """What the invariant is protecting against, demonstrated rather than asserted
    from the documents.

    The per-sample scores still land on disk; it is the aggregate that is lost.
    """
    log = inspect_eval(
        mock_task(instances, scorers=[ragged_judge()], epochs=2, name="ragged"),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "ragged-logs"),
    )[0]

    assert log.status == "error"
    assert log.results is None
    assert log.error is not None
    assert "isn't present in the score value dictionary" in str(log.error.message)


def test_negative_control_scores_survive_on_disk_even_when_results_are_void(
    instances, tmp_path
):
    log = inspect_eval(
        mock_task(instances, scorers=[ragged_judge()], epochs=2, name="ragged2"),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "ragged2-logs"),
    )[0]
    stored = read_log(log.location)
    scored = [s for s in (stored.samples or []) if s.scores]
    assert scored, "per-sample scores are lost as well as the aggregate"
