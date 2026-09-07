"""The four objects: Check, Catalogue, Instance, VerdictSet."""

from __future__ import annotations

import pytest
from harness_fixtures import make_check, make_task_check, pinned_catalogue
from inspect_ai.scorer import Target
from pydantic import ValidationError

from genomics_harness import (
    Catalogue,
    Check,
    CheckGroundTruth,
    CheckKind,
    ContaminationClass,
    GroundTruthClass,
    Instance,
    InstanceSet,
    Requirement,
    SelectionCriterion,
    SelectionRule,
    TaskCheck,
    build_dataset,
    build_sample,
)

# -- Check ----------------------------------------------------------------


def test_a_check_carries_no_requirement_marker():
    """Required-or-enriching is per task, so it must not be settable per check."""
    assert "requirement" not in Check.model_fields
    with pytest.raises(ValueError):
        Check(
            id="A1",
            group="A",
            title="t",
            text="s",
            ground_truth_class=GroundTruthClass.COMPUTABLE,
            contamination_class=ContaminationClass.BEHAVIOUR_SCORING,
            construction=CheckKind.EMERGENT,
            genotypes=False,
            talos_anchor=None,
            in_scope=True,
            refines=None,
            requirement="required",  # type: ignore[call-arg]
        )


def test_the_three_classifications_are_closed_sets():
    assert {c.value for c in GroundTruthClass} == {
        "computable",
        "hybrid",
        "calibrated",
    }
    assert {c.value for c in ContaminationClass} == {
        "behaviour_scoring",
        "answer_scoring",
        "mixed",
    }
    assert {c.value for c in CheckKind} == {"dedicated", "emergent"}


def test_nesting_is_expressible():
    """B6 is the base, B5 refines B6, B1 refines B5 (§4), and judging the three
    as independent is a known way to get the wrong answer.

    Read off the pinned catalogue rather than transcribed: this test asserted the
    inversion, passing, from a hand-built catalogue of its own (§0 v20).
    """
    catalogue = pinned_catalogue(["B6", "B5", "B1"])
    assert catalogue.get("B6").refines is None
    assert catalogue.get("B5").refines == "B6"
    assert catalogue.get("B1").refines == "B5"


def test_a_check_cannot_refine_itself():
    with pytest.raises(ValueError, match="cannot refine itself"):
        make_check("B1", refines="B1")


def test_checks_are_frozen():
    with pytest.raises(ValueError):
        make_check("A1").id = "A2"  # type: ignore[misc]


# -- Catalogue ------------------------------------------------------------


def test_duplicate_check_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate check id"):
        Catalogue(release="v", checks=[make_check("A1"), make_check("A1")])


def test_a_dangling_refines_is_rejected():
    with pytest.raises(ValueError, match="not in the catalogue"):
        Catalogue(release="v", checks=[make_check("B5", refines="B1")])


def test_ids_are_in_declaration_order_not_sorted_order():
    catalogue = Catalogue(
        release="v", checks=[make_check("D1"), make_check("A1"), make_check("B1")]
    )
    assert catalogue.ids == ["D1", "A1", "B1"]


def test_the_digest_covers_the_check_text(catalogue):
    """`text` is the operative definition, so editing it is a new catalogue."""
    edited = catalogue.model_copy(
        update={
            "checks": [
                catalogue.checks[0].model_copy(update={"text": "a looser definition"}),
                *catalogue.checks[1:],
            ]
        }
    )
    assert edited.digest != catalogue.digest


def test_the_digest_covers_the_release_label(catalogue):
    """A report citing 'c1' must not be able to cite two different contents."""
    assert (
        catalogue.model_copy(update={"release": "v-other"}).digest != catalogue.digest
    )


def test_the_digest_is_insensitive_to_field_insertion_order(catalogue):
    payload = catalogue.model_dump(mode="json")
    reversed_payload = {k: payload[k] for k in reversed(list(payload))}
    assert Catalogue.model_validate(reversed_payload).digest == catalogue.digest


def test_filtering_to_the_in_scope_checks_is_a_different_catalogue():
    """The digest distinguishes the subset; the label carries through unchanged.

    Nothing else in either suite asserts that filtering one `Catalogue` yields
    another with a different digest — the bridge test that looks like this one
    compares against the *file's* recorded content digest, a different digest
    over a different object (§0 v27 item 5). The label assertion is the other
    half: the subset is the same release, not a second one, which is the rule
    `delivered_catalogue()` already followed.
    """
    catalogue = Catalogue(
        release="v",
        checks=[make_check("A1"), make_check("A4", in_scope=False)],
    )
    subset = catalogue.in_scope()
    assert subset.ids == ["A1"]
    assert subset.digest != catalogue.digest
    assert subset.release == "v"


def test_lookup_and_membership(catalogue):
    assert "A1" in catalogue
    assert "ZZ" not in catalogue
    assert catalogue.get("A1").id == "A1"
    with pytest.raises(KeyError):
        catalogue.get("ZZ")


def test_blank_release_is_rejected():
    with pytest.raises(ValueError, match="must not be blank"):
        Catalogue(release="  ", checks=[])


# -- Instance -------------------------------------------------------------


def test_ground_truth_for_an_unreachable_check_is_rejected():
    with pytest.raises(ValueError, match="does not make reachable"):
        Instance(
            id="aaaaaaaaaaaaaaaa",
            prompt="q",
            checks=[make_task_check("A1", Requirement.REQUIRED)],
            ground_truth=[CheckGroundTruth(check_id="D1", expected=1)],
        )


def test_a_check_listed_twice_is_rejected():
    with pytest.raises(ValueError, match="lists checks twice"):
        Instance(
            id="aaaaaaaaaaaaaaaa",
            prompt="q",
            checks=[
                make_task_check("A1", Requirement.REQUIRED),
                make_task_check("A1", Requirement.ENRICHING),
            ],
        )


def test_a_task_check_without_its_three_bands_does_not_construct():
    """Required with no default, and the fact asserted rather than a consequence.

    §4: an instance that omits the grading criteria must fail *at construction*
    rather than silently leave the judge with the scaffold alone — the same
    pattern as `SolverConfig.reasoning_effort`, and for the same reason. A
    default would produce a well-formed prompt with no criteria in it, which is
    a failure with no error anywhere.

    Asserted on the fields themselves (§15, "assert the fact, not its
    consequences"): a test that only checked a built prompt for band lines would
    pass over a default that silently supplied empty ones.
    """
    for omitted in ("performed", "partial", "not_performed"):
        kwargs = {
            "check_id": "A1",
            "requirement": Requirement.REQUIRED,
            "performed": "p",
            "partial": "q",
            "not_performed": "r",
        }
        del kwargs[omitted]
        with pytest.raises(ValidationError, match=omitted):
            TaskCheck(**kwargs)

    assert all(
        TaskCheck.model_fields[name].is_required()
        for name in ("performed", "partial", "not_performed")
    )


def test_a_ground_truth_without_an_expected_outcome_does_not_construct():
    """`expected` is required: every check has one, and it need not be a number.

    A check whose correct answer is a bounded refusal to conclude *expects that
    refusal*, and saying so is what a judge needs most. Calling such an
    expectation absent conflates "the outcome is not a number" with "there is no
    outcome", and writes a gap into the prompt for exactly the checks whose
    expectation is hardest to state (§4, §0 v25 §8).
    """
    with pytest.raises(ValidationError, match="expected"):
        CheckGroundTruth(check_id="A1")

    assert CheckGroundTruth.model_fields["expected"].is_required()
    # a non-numeric expectation is an expectation, and constructs
    assert (
        CheckGroundTruth(
            check_id="A1", expected="a bounded refusal to conclude"
        ).tolerance
        == ()
    )


def test_a_null_expected_outcome_is_rejected_as_well_as_an_omitted_one():
    """Requiredness catches omission and not nullity, so a validator catches nullity.

    ``expected`` is typed ``Any``, so *omitting* it fails at construction while
    passing ``None`` explicitly does not — and the prompt builder no longer has a
    branch for the null case, which makes it a latent gap in a judge's prompt
    rather than a caught error (§0 v27 §6). Asserted on the field itself in both
    construction paths, because a test that built a prompt and looked for a
    missing line would assert the consequence rather than the rule (§15).
    """
    with pytest.raises(ValidationError, match="expected must not be None"):
        CheckGroundTruth(check_id="A1", expected=None)

    # the deferred path revalidates the stored wire form, and it refuses too
    with pytest.raises(ValidationError, match="expected must not be None"):
        CheckGroundTruth.model_validate({"check_id": "A1", "expected": None})


def test_the_same_check_carries_different_weight_in_different_instances():
    """The reason the marker is on the relation and not on the check."""
    required = Instance(
        id="aaaaaaaaaaaaaaaa",
        prompt="q",
        checks=[make_task_check("B5", Requirement.REQUIRED)],
    )
    enriching = Instance(
        id="bbbbbbbbbbbbbbbb",
        prompt="q",
        checks=[make_task_check("B5", Requirement.ENRICHING)],
    )
    assert required.required_check_ids == ["B5"]
    assert enriching.required_check_ids == []
    assert enriching.reachable_check_ids == ["B5"]


def test_an_instance_is_validated_against_a_catalogue(catalogue):
    instance = Instance(
        id="aaaaaaaaaaaaaaaa",
        prompt="q",
        checks=[make_task_check("ZZ", Requirement.REQUIRED)],
    )
    problems = instance.validate_against(catalogue)
    assert len(problems) == 1
    assert "not in catalogue" in problems[0]


def test_instance_set_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate instance ids"):
        InstanceSet(
            name="x",
            instances=[
                Instance(id="aaaaaaaaaaaaaaaa", prompt="q"),
                Instance(id="aaaaaaaaaaaaaaaa", prompt="r"),
            ],
        )


def _content(**overrides):
    """Instance fields with no id, for `with_derived_id`."""
    fields = {
        "prompt": "Investigate GENE1 at chr7:100000 and report what you find.",
        "checks": [make_task_check("A1", Requirement.REQUIRED)],
        "ground_truth": [CheckGroundTruth(check_id="A1", expected=42)],
        "attributes": {"gene": "GENE1", "chromosome": "7"},
        "dataset_snapshot": "onekgpd-test-snapshot",
        "notes": "a note",
    }
    fields.update(overrides)
    return fields


def test_a_derived_id_is_stable_and_reveals_nothing():
    """The three properties `opaque_id` carried before v44, on what replaced it.

    Determinism, discrimination and non-disclosure were asserted against the
    parts-tuple helper (`opaque_id(*parts)`), which this ruling retires. The
    rule they test is live and now lives on the content derivation, so the
    assertions move with it rather than being dropped (§4, §0 v44).
    """
    left = Instance.with_derived_id(**_content())
    right = Instance.with_derived_id(**_content())
    other = Instance.with_derived_id(**_content(prompt="a different question"))

    assert left.id == right.id != other.id
    assert len(left.id) == 16
    # the fixture's own identifying strings, so the assertion is about *this*
    # derivation rather than about ids in general. The symbol and the coordinate
    # are synthetic: no surface publishing with the harness carries a real one
    # (§16), and the instances that assert against their real strings keep them
    # in their own package, where the needles are real and the tree is private.
    for leak in ("GENE1", "gene1", "100000", "chr7"):
        assert leak not in left.id


def test_a_derived_id_is_the_digest_of_the_content_it_was_built_from():
    """The fixed point: what `with_derived_id` sets is what the content derives."""
    instance = Instance.with_derived_id(**_content())
    assert instance.id == instance.opaque_id
    assert instance.opaque_id == instance.content_sha256[:16]
    assert instance.has_opaque_id()


def test_the_id_is_excluded_from_the_payload_it_is_derived_from():
    """A value cannot be derived from a payload containing it (§4).

    The edit this catches is dropping ``exclude={"id"}`` from
    `Instance.content_sha256`: without it these three instances — one content
    under three ids — would derive three different values.
    """
    fields = _content()
    ids = ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "id-not-yet-derived"]
    derived = {Instance(id=i, **fields).opaque_id for i in ids}
    assert len(derived) == 1
    assert derived == {Instance.with_derived_id(**fields).id}


def test_with_derived_id_refuses_an_id_rather_than_ignoring_one():
    """Refused, not overwritten: a caller passing one has the rule backwards."""
    with pytest.raises(TypeError, match="derives the id"):
        Instance.with_derived_id(id="aaaaaaaaaaaaaaaa", **_content())


def test_every_content_field_moves_the_derived_id():
    """No field an instance carries is outside the derivation (§4, §0 v44).

    One edit per field, each against the same baseline, so a field silently
    dropped from the payload is caught by name rather than by a single edit
    that happens to touch several.
    """
    baseline = Instance.with_derived_id(**_content())
    edits = {
        "prompt": _content(prompt="another question entirely"),
        "checks": _content(
            checks=[make_task_check("A1", Requirement.ENRICHING)],
        ),
        "ground_truth": _content(
            ground_truth=[CheckGroundTruth(check_id="A1", expected=43)]
        ),
        "attributes": _content(attributes={"gene": "GENE2", "chromosome": "1"}),
        "dataset_snapshot": _content(dataset_snapshot="onekgpd-other"),
        "notes": _content(notes="a different note"),
    }
    moved = {
        field: Instance.with_derived_id(**fields).id for field, fields in edits.items()
    }
    for field, value in moved.items():
        assert value != baseline.id, f"editing {field} did not move the id"
    assert len(set(moved.values())) == len(moved)


def test_non_opaque_ids_are_reported_but_not_refused(instances):
    """`EvalDataset.sample_ids` is recorded unconditionally, so a descriptive id
    leaks the instance from a header-only read — but a debugging run with
    readable ids is legitimate."""
    assert instances.non_opaque_ids() == []
    leaky = instances.model_copy(
        update={
            "instances": [
                instances.instances[0].model_copy(update={"id": "GENE1_chr7_100000"})
            ]
        }
    )
    assert leaky.non_opaque_ids() == ["GENE1_chr7_100000"]


# -- selection criteria ---------------------------------------------------


def test_criteria_are_executable_rather_than_prose(instances):
    rule = SelectionRule(
        name="high-count autosomal",
        all_of=[
            SelectionCriterion(attribute="cohort_ac", op="gte", value=100),
            SelectionCriterion(attribute="chromosome", op="in", value=["1", "2"]),
        ],
    )
    selected = rule.select(instances.instances)
    assert [i.attributes["gene"] for i in selected] == ["GENE1"]


def test_a_criterion_on_a_missing_attribute_does_not_match():
    instance = Instance(id="aaaaaaaaaaaaaaaa", prompt="q", attributes={"gene": "X"})
    assert not SelectionCriterion(attribute="ac", op="gte", value=1).evaluate(
        instance.attributes
    )
    assert SelectionCriterion(attribute="ac", op="absent").evaluate(instance.attributes)
    assert SelectionCriterion(attribute="gene", op="exists").evaluate(
        instance.attributes
    )


def test_ordered_comparison_on_a_non_number_raises_rather_than_coercing():
    criterion = SelectionCriterion(attribute="gene", op="gt", value=1)
    with pytest.raises(TypeError, match="needs a number"):
        criterion.evaluate({"gene": "GENE1"})


def test_unknown_operators_are_rejected():
    with pytest.raises(ValueError, match="unknown operator"):
        SelectionCriterion(attribute="a", op="matches", value="x")


def test_any_of_is_a_disjunction(instances):
    rule = SelectionRule(
        name="either gene",
        any_of=[
            SelectionCriterion(attribute="gene", op="eq", value="GENE1"),
            SelectionCriterion(attribute="gene", op="eq", value="GENE2"),
        ],
    )
    assert len(rule.select(instances.instances)) == 2


# -- the Inspect boundary -------------------------------------------------


def test_ground_truth_goes_to_metadata_and_target_stays_empty(instances):
    sample = build_sample(instances.instances[0])
    assert sample.target == ""
    assert Target(sample.target).text == ""
    recovered = Instance.from_sample_metadata(sample.metadata)
    assert recovered == instances.instances[0]
    assert recovered.truth_for("A1").expected == 42


def test_the_dataset_preserves_our_ids(instances):
    dataset = build_dataset(instances)
    assert [s.id for s in dataset] == instances.ids()
    assert dataset.name == instances.name


def test_metadata_is_namespaced_so_solver_keys_cannot_collide(instances):
    sample = build_sample(instances.instances[0])
    assert set(sample.metadata) == {"instance"}


def test_recovering_an_instance_from_foreign_metadata_raises():
    with pytest.raises(KeyError, match="carries no 'instance' key"):
        Instance.from_sample_metadata({"something": "else"})
