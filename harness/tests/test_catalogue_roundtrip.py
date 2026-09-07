"""The catalogue as a `@scorer` factory argument — verified, not assumed.

The catalogue is passed as a factory argument because that is what makes a
scoring log record which catalogue produced its verdicts, and makes a re-score
reproduce the catalogue that was actually used rather than today's. The brief
calls the constraint "must be JSON-serialisable". The constraint is sharper than
that, and the difference only shows up in the deferred path — which is the path
the harness uses.

On the way out, `registry_tag` binds the factory's explicitly-passed arguments
(`_util/registry.py:149-181`, note `apply_defaults=False`), `as_scorer_spec`
copies them via `registry_params`, and they land in `EvalScorer.options`. On the
way back, `resolve_scorers` (`_eval/score.py:531-577`) splats them into the
factory — as whatever pydantic produced on disk, which is plain JSON types.

So a factory that accepts a `Catalogue` receives a `Catalogue` in-run and a
`dict` on re-score. These tests establish that, and establish that the wire form
this harness uses round-trips identically in both paths.
"""

from __future__ import annotations

import json

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai import score
from inspect_ai._eval.score import resolve_scorers
from mock_harness import MOCK_CATALOGUE, catalogue_probe, clean_model, mock_task

from genomics_harness import Catalogue, read_log


def _run(instances, tmp_path, scorers, name):
    return inspect_eval(
        mock_task(instances, scorers=scorers, name=name),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / name),
    )[0]


# -- the wire form --------------------------------------------------------


def test_the_wire_form_is_plain_json_types():
    arg = MOCK_CATALOGUE.to_arg()
    assert isinstance(arg, dict)
    assert json.loads(json.dumps(arg)) == arg


def test_catalogue_of_accepts_both_forms_and_preserves_the_digest():
    from_model = Catalogue.of(MOCK_CATALOGUE)
    from_wire = Catalogue.of(MOCK_CATALOGUE.to_arg())
    assert from_model == from_wire == MOCK_CATALOGUE
    assert from_wire.digest == MOCK_CATALOGUE.digest


def test_catalogue_of_rejects_anything_else():
    with pytest.raises(TypeError, match="Catalogue or a mapping"):
        Catalogue.of(["A1", "B1"])  # type: ignore[arg-type]


# -- the header capture ---------------------------------------------------


def test_the_catalogue_lands_in_the_log_header(instances, tmp_path):
    log = _run(
        instances, tmp_path, [catalogue_probe(MOCK_CATALOGUE.to_arg())], "header"
    )
    stored = read_log(log.location)

    assert stored.eval.scorers is not None
    options = stored.eval.scorers[0].options
    assert options is not None
    recovered = Catalogue.of(options["catalogue"])
    assert recovered == MOCK_CATALOGUE
    assert recovered.digest == MOCK_CATALOGUE.digest

    # the check text travels too — this is the reproducibility mechanism, and
    # also what publishes the rubric with the scoring log
    assert recovered.checks[0].text == "the trajectory performed A1"


def test_the_header_survives_a_read_and_rebuilds_the_scorer(instances, tmp_path):
    log = _run(
        instances, tmp_path, [catalogue_probe(MOCK_CATALOGUE.to_arg())], "rebuild"
    )
    stored = read_log(log.location)
    rebuilt = resolve_scorers(stored)
    assert len(rebuilt) == 1


def test_the_same_catalogue_is_seen_inline_and_on_rescore(instances, tmp_path):
    """The property that matters: one digest, both paths."""
    log = _run(instances, tmp_path, [catalogue_probe(MOCK_CATALOGUE.to_arg())], "both")
    stored = read_log(log.location)
    rescored = score(stored, resolve_scorers(stored), action="append", display="none")

    observations = [
        (name, s.metadata)
        for sample in rescored.samples or []
        for name, s in (sample.scores or {}).items()
    ]
    assert observations

    digests = {m["resolved_digest"] for _, m in observations}
    assert digests == {MOCK_CATALOGUE.digest}

    received = {m["received_type"] for _, m in observations}
    assert received == {"dict"}, (
        f"the wire form must be the same type in both paths; got {received}"
    )


def test_a_catalogue_model_passed_directly_degrades_to_a_dict_on_rescore(
    instances, tmp_path
):
    """Why the wire form exists, demonstrated rather than asserted.

    Passing the model works in-run — and comes back as a `dict` on re-score,
    because that is what was serialised. Code that reached for a model attribute
    would break here and nowhere else.
    """
    log = _run(instances, tmp_path, [catalogue_probe(MOCK_CATALOGUE)], "model-arg")
    inline = [
        s.metadata["received_type"]
        for sample in log.samples or []
        for s in (sample.scores or {}).values()
    ]
    assert set(inline) == {"Catalogue"}

    stored = read_log(log.location)
    rescored = score(stored, resolve_scorers(stored), action="append", display="none")
    deferred = [
        s.metadata["received_type"]
        for sample in rescored.samples or []
        for name, s in (sample.scores or {}).items()
        if name.endswith("1")
    ]
    assert set(deferred) == {"dict"}


def test_an_evolved_catalogue_does_not_leak_into_a_rescore(instances, tmp_path):
    """The failure a module-level constant would produce.

    The stored header pins the catalogue, so re-scoring an old log reproduces
    the catalogue that was actually used even after the module constant moves on.
    """
    log = _run(
        instances, tmp_path, [catalogue_probe(MOCK_CATALOGUE.to_arg())], "evolved"
    )
    stored = read_log(log.location)

    evolved = MOCK_CATALOGUE.model_copy(update={"release": "v-mock-2"})
    assert evolved.digest != MOCK_CATALOGUE.digest

    rescored = score(stored, resolve_scorers(stored), action="append", display="none")
    digests = {
        s.metadata["resolved_digest"]
        for sample in rescored.samples or []
        for s in (sample.scores or {}).values()
    }
    assert digests == {MOCK_CATALOGUE.digest}
    assert evolved.digest not in digests


def test_generation_only_runs_still_publish_scorer_options(instances, tmp_path):
    """`TaskLogger` is constructed with the scorer specs regardless of `score=`
    (`_eval/run.py:333-362`), so a `score=False` run is only rubric-free if the
    task carries no scorers at all."""
    with_scorers = inspect_eval(
        mock_task(
            instances, scorers=[catalogue_probe(MOCK_CATALOGUE.to_arg())], name="nos1"
        ),
        model=clean_model(),
        display="none",
        score=False,
        log_dir=str(tmp_path / "nos1"),
    )[0]
    stored = read_log(with_scorers.location)
    assert stored.eval.scorers is not None
    assert "catalogue" in (stored.eval.scorers[0].options or {})

    without = inspect_eval(
        mock_task(instances, scorers=[], name="nos2"),
        model=clean_model(),
        display="none",
        score=False,
        log_dir=str(tmp_path / "nos2"),
    )[0]
    stored_without = read_log(without.location)
    assert not stored_without.eval.scorers


def test_only_explicitly_passed_arguments_are_recorded(instances, tmp_path):
    """`registry_tag` binds without `apply_defaults`, so a catalogue supplied as
    a parameter default would be invisible in the header."""
    log = _run(
        instances, tmp_path, [catalogue_probe(MOCK_CATALOGUE.to_arg())], "explicit"
    )
    stored = read_log(log.location)
    options = stored.eval.scorers[0].options or {}
    assert set(options) == {"catalogue"}
