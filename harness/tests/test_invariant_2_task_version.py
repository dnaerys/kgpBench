"""Invariant 2 — `Task.version` moves when an instance changes and only then.

Guards against a silent `eval_set` no-op on instance rotation. `task_identifier()`
(`_eval/evalset.py`) hashes the task file, name, task args, model, generate
config, model roles, the resolved plan, and `AdditionalHashFields` — which
includes `version` and does **not** include the dataset. Rotate instances without
moving `version` and `eval_set` believes the work is done, so last round's
numbers are reported as this round's.
"""

from __future__ import annotations

from inspect_ai import eval as inspect_eval
from mock_harness import clean_model, mock_task

from genomics_harness import (
    Instance,
    InstanceSet,
    Requirement,
    task_metadata,
    task_version,
)


def _edited(instances: InstanceSet, prompt: str) -> InstanceSet:
    head, *rest = instances.instances
    return instances.model_copy(
        update={"instances": [head.model_copy(update={"prompt": prompt}), *rest]}
    )


# -- the digest itself ----------------------------------------------------


def test_version_changes_when_a_prompt_changes(instances):
    assert task_version(instances) != task_version(_edited(instances, "a new question"))


def test_version_changes_when_ground_truth_changes(instances):
    head, *rest = instances.instances
    changed = head.model_copy(
        update={
            "ground_truth": [
                gt.model_copy(update={"expected": 43}) for gt in head.ground_truth
            ]
        }
    )
    assert task_version(instances) != task_version(
        instances.model_copy(update={"instances": [changed, *rest]})
    )


def test_version_changes_when_the_check_relation_changes(instances):
    """Flipping a check from required to enriching changes what the run means."""
    head, *rest = instances.instances
    changed = head.model_copy(
        update={
            "checks": [
                tc.model_copy(update={"requirement": Requirement.ENRICHING})
                for tc in head.checks
            ]
        }
    )
    assert task_version(instances) != task_version(
        instances.model_copy(update={"instances": [changed, *rest]})
    )


def test_version_ignores_the_set_name_and_the_set_level_snapshot(instances):
    """The digest is over the resolved members and nothing else (§4, §0 v31).

    The name comes from the composition config, so digesting it would make a
    rename look like a membership change; the set-level snapshot is derived from
    the members and travels on its own comparison axis (§7). Both left the
    payload in the same edit that added the sort.
    """
    renamed = instances.model_copy(update={"name": "another-name"})
    resnapped = instances.model_copy(update={"dataset_snapshot": "onekgpd-other"})
    assert task_version(instances) == task_version(renamed) == task_version(resnapped)


def test_version_changes_when_an_instance_is_added_or_removed(instances):
    extra = Instance(id="ffffffffffffffff", prompt="third question")
    grown = instances.model_copy(update={"instances": [*instances.instances, extra]})
    shrunk = instances.model_copy(update={"instances": instances.instances[:1]})
    assert (
        len({task_version(instances), task_version(grown), task_version(shrunk)}) == 3
    )


def test_version_is_unchanged_when_instances_are_reordered(instances):
    """**Inverted at v31**, when composition moved into a config file (§4).

    It used to assert the opposite, on the argument that sample order decides
    positional ids and log order. Neither survives: every instance here carries
    an explicit id, so nothing is positional, and a set is now *declared* as a
    list of module paths in a file people edit — so a reordered config would
    move `Task(version=)` and make `eval_set` re-run a round it had already
    paid for, which is invariant 2's own hazard pointing the other way. A set is
    its members; the members are digested sorted by id, and the dataset is
    emitted in that order.
    """
    reordered = instances.model_copy(
        update={"instances": list(reversed(instances.instances))}
    )
    assert task_version(instances) == task_version(reordered)


def test_version_is_unchanged_when_nothing_is(instances):
    rebuilt = InstanceSet.model_validate(instances.model_dump(mode="json"))
    assert task_version(rebuilt) == task_version(instances)


def test_version_is_unchanged_by_field_insertion_order(instances):
    """A round trip through JSON with keys reversed must not move the digest."""
    payload = instances.model_dump(mode="json")
    reversed_keys = {k: payload[k] for k in reversed(list(payload))}
    reversed_keys["instances"] = [
        {k: inst[k] for k in reversed(list(inst))} for inst in payload["instances"]
    ]
    assert task_version(InstanceSet.model_validate(reversed_keys)) == task_version(
        instances
    )


def test_version_length_is_the_short_digest(instances):
    assert len(task_version(instances)) == 16
    assert instances.digest.startswith(task_version(instances))


# -- the wiring -----------------------------------------------------------


def test_version_reaches_the_log_and_moves_with_the_dataset(instances, tmp_path):
    before = inspect_eval(
        mock_task(instances),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "before"),
    )[0]
    after = inspect_eval(
        mock_task(_edited(instances, "a new question")),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "after"),
    )[0]

    assert before.eval.task_version == task_version(instances)
    assert before.eval.task_version != after.eval.task_version


def test_task_identifier_moves_with_the_dataset(instances, tmp_path):
    """The identity `eval_set` actually consults.

    Reaching into `_eval.evalset` is deliberate: the public surface does not
    expose the function whose behaviour the invariant is about, and asserting on
    `Task.version` alone would leave the actual hazard untested.
    """
    from inspect_ai._eval.evalset import task_identifier

    before = inspect_eval(
        mock_task(instances),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "before"),
    )[0]
    after = inspect_eval(
        mock_task(_edited(instances, "a new question")),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "after"),
    )[0]
    same = inspect_eval(
        mock_task(instances),
        model=clean_model(),
        display="none",
        log_dir=str(tmp_path / "same"),
    )[0]

    assert task_identifier(before, None) != task_identifier(after, None)
    assert task_identifier(before, None) == task_identifier(same, None)


def test_instances_sha256_reaches_task_metadata(instances):
    metadata = task_metadata(instances)
    assert metadata["instances_sha256"] == instances.digest
    assert metadata["provenance"]["instances_version"] == task_version(instances)
