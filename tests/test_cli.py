"""The published operator surface — `kgpbench judge`, `report` and `survey`.

Every test here runs at $0 on `mockllm/model`. The `judge` subcommand resolves a
judge's identity from :func:`~kgpbench.scoring.judge_configs`, which
names Anthropic models, so the tests that run it patch that one seam and let
everything downstream — :func:`~kgpbench.scoring.judge_roles`,
:func:`~kgpbench.scoring.run_judges`, `score()`, the sidecar write — run
unpatched. Patching the *source* of the identity rather than the roles built
from it is what keeps the run honest: the declaration and the request still come
from one place, which is the property the command exists to have.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import rich
from inspect_ai._display.core.rich import rich_initialise
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.log import EvalLog
from mock_judging import (
    CATALOGUE,
    JudgeRecorder,
    generation_log,
    judge_model,
    verdict_reply,
)

from kgpbench import (
    JudgeConfig,
    catalogue_arg,
    provenance_sidecar_path,
    read_run_provenance,
)
from kgpbench.cli import main
from kgpbench.judge_prompt import JUDGE_PROMPT_VERSION
from kgpbench.report import build_report, load_evaluation
from kgpbench.scoring import _resolve_judge_configs

VERDICTS = {"B6": "performed", "B5": "performed", "B1": "not_performed"}

MOCK_CONFIG = JudgeConfig(
    name="judge_opus_a",
    model="mockllm/model",
    reasoning_effort="xhigh",
    prompt_version=JUDGE_PROMPT_VERSION,
)
"""What the command reads the judge's identity from, pointed at the fixture.

The model is what the run will actually wire, because ``judge_roles`` builds the
role *from this object* — so the sidecar this produces is true of the run rather
than of the reference panel it stands in for.
"""


@pytest.fixture
def clean_log(tmp_path: Path) -> EvalLog:
    return generation_log(tmp_path)


@pytest.fixture
def catalogue_file(tmp_path: Path) -> Path:
    """The fixture catalogue as a wire form, which `--catalogue` accepts."""
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(catalogue_arg(CATALOGUE)), encoding="utf-8")
    return path


def _mock_identity(monkeypatch: pytest.MonkeyPatch, recorder: JudgeRecorder) -> None:
    """Point the command's one identity seam at the fixture."""

    def configs(names: object = None) -> tuple[JudgeConfig, ...]:
        return (MOCK_CONFIG,)

    monkeypatch.setattr("kgpbench.cli.judge_configs", configs)
    monkeypatch.setattr(
        "kgpbench.cli.judge_roles",
        lambda given: {
            config.name: judge_model(recorder, verdict_reply(VERDICTS))
            for config in given
        },
    )


def _judge(clean_log: EvalLog, catalogue_file: Path, scoring: Path, *extra: str) -> int:
    assert clean_log.location
    return main(
        [
            "judge",
            "--generation",
            clean_log.location,
            "--catalogue",
            str(catalogue_file),
            "--judge",
            "judge_opus_a",
            "--scoring-dir",
            str(scoring),
            *extra,
        ]
    )


# -- judge -----------------------------------------------------------------


def test_judge_writes_one_scoring_log_and_its_sidecar(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One judge per invocation, and the sidecar travels with the log (§14.40)."""
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"

    assert _judge(clean_log, catalogue_file, scoring) == 0

    log = scoring / "judge_opus_a.eval"
    assert log.exists()
    assert provenance_sidecar_path(log).exists()
    assert not (scoring / "judge_opus_b.eval").exists()


def test_the_sidecar_records_the_model_that_ran(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command's roles are built from its configs, so the two cannot disagree.

    This is the property the §7 evidence failure is the absence of: a record
    naming a model beside a transcript that never called it. Asserted against
    what the fixture wired rather than against `MOCK_CONFIG`, so a command that
    started resolving the two from separate sources would fail here.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"

    _judge(clean_log, catalogue_file, scoring)

    provenance = read_run_provenance(scoring / "judge_opus_a.eval")
    assert [judge.name for judge in provenance.judges] == ["judge_opus_a"]
    assert provenance.judges[0].model == "mockllm/model"
    assert provenance.judges[0].reasoning_effort == "xhigh"
    assert provenance.judges[0].prompt_version == JUDGE_PROMPT_VERSION


def test_judge_refuses_an_unresolvable_catalogue(
    clean_log: EvalLog, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`CatalogueUnresolved` is an operator error, so it exits 1 with its message.

    Reached before anything is wired, so it costs nothing: the identity is read
    after the catalogue resolves.
    """
    assert clean_log.location
    status = main(
        [
            "judge",
            "--generation",
            clean_log.location,
            "--catalogue",
            str(tmp_path / "absent.json"),
            "--judge",
            "judge_opus_a",
            "--scoring-dir",
            str(tmp_path / "scoring"),
        ]
    )

    assert status == 1
    assert "CatalogueUnresolved" in capsys.readouterr().out


def test_judge_refuses_an_existing_scoring_log(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A scoring log is one judge's verdicts over one corpus, and this command
    offers no way to replace it.

    Both halves. The refusal fires on a second run into the same directory, and
    the remedy it names — a different ``--scoring-dir`` — is the one this command
    can actually perform, so the second half runs it and requires 0.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()

    assert _judge(clean_log, catalogue_file, scoring) == 1
    printed = capsys.readouterr().out
    assert "FileExistsError" in printed
    assert "--scoring-dir" in printed

    assert _judge(clean_log, catalogue_file, tmp_path / "second-round") == 0


def test_judge_has_no_overwrite_flag(
    clean_log: EvalLog, catalogue_file: Path, tmp_path: Path
) -> None:
    """The positive control on a removal (§14.40).

    ``--overwrite`` was this command's own addition and is not forced by
    :func:`~kgpbench.scoring.run_judges`'s signature the way
    ``--generation`` and ``--scoring-dir`` are. It moves an irreversible step
    against a paid artifact onto the published surface at one flag, so it is off
    it; ``run_judges(overwrite=)`` stays reachable from Python. This test is what
    fails if the flag comes back — argparse exits 2 on an unknown option, which
    is a different code from the refusals `main` translates.
    """
    assert clean_log.location
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "judge",
                "--generation",
                clean_log.location,
                "--catalogue",
                str(catalogue_file),
                "--judge",
                "judge_opus_a",
                "--scoring-dir",
                str(tmp_path / "scoring"),
                "--overwrite",
            ]
        )
    assert raised.value.code == 2


def test_judge_requires_the_arguments_that_decide_the_verdicts() -> None:
    """`--judge` and `--catalogue` are required, and neither has a default.

    Argparse exits 2 rather than raising, which is why this asserts `SystemExit`
    and not a refusal: the point is that the surface has no default panel and no
    default catalogue to fall back to (§14.40, §14.42).
    """
    for missing in ("--judge", "--catalogue"):
        argv = [
            "judge",
            "--generation",
            "g.eval",
            "--catalogue",
            "c1",
            "--judge",
            "judge_opus_a",
            "--scoring-dir",
            "s",
        ]
        index = argv.index(missing)
        del argv[index : index + 2]
        with pytest.raises(SystemExit) as raised:
            main(argv)
        assert raised.value.code == 2


def test_judge_refuses_a_name_no_factory_ships() -> None:
    """The shell route closes to the shipped names; the Python route does not.

    `run_judges(judges=[...])` takes any `@scorer` whose factory argument is
    named ``catalogue``, which is how §11's non-Anthropic validity check reaches
    this package. A command line cannot hand over a constructed `Scorer`.
    """
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "judge",
                "--generation",
                "g.eval",
                "--catalogue",
                "c1",
                "--judge",
                "judge_of_my_own",
                "--scoring-dir",
                "s",
            ]
        )
    assert raised.value.code == 2


# -- report ----------------------------------------------------------------


def _report(clean_log: EvalLog, scoring: Sequence[Path], *extra: str) -> int:
    assert clean_log.location
    argv = ["report", "--generation", clean_log.location]
    for log in scoring:
        argv += ["--scoring", str(log)]
    return main([*argv, *extra])


def test_report_prints_one_row_per_check_and_requirement(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()

    assert _report(clean_log, [scoring / "judge_opus_a.eval"]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines
    assert all("trajectories" in line and "disagreement" in line for line in lines)


def test_the_json_dump_carries_no_rate_fields(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The rider §14.40 exists to enforce, and the edit it forbids.

    `disagreement_rate`, `performed_rate` and `rows` are computed properties over
    fields the dump already carries, so their absence drops no information — and
    it keeps §5's prohibition on a rate feeding an automated consumer structural
    rather than remembered. A consumer wanting to threshold must divide.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()
    # the object the dump is of, so the recovered value is checked against the
    # property rather than against a second copy of the same arithmetic
    assert clean_log.location
    report = build_report(
        [load_evaluation(clean_log.location, [scoring / "judge_opus_a.eval"])]
    )
    key = next(iter(sorted(report.counts)))

    assert _report(clean_log, [scoring / "judge_opus_a.eval"], "--json") == 0

    dumped = json.loads(capsys.readouterr().out)
    assert "rows" not in dumped
    counts = next(iter(dumped["counts"].values()))
    for absent in ("disagreement_rate", "performed_rate", "observed", "rows"):
        assert absent not in counts

    # every term both rates and both denominators are built from is present, so
    # a consumer can recover any of them — by writing the division itself
    assert {
        "performed",
        "partial",
        "not_performed",
        "split",
        "multi_voter",
        "unobserved_reachable",
        "excluded",
    } <= set(counts)
    observed = (
        counts["performed"]
        + counts["partial"]
        + counts["not_performed"]
        + counts["split"]
    )
    assert observed == report.counts[key].observed


def test_report_applies_a_declared_exclusion_list(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§5's exclusion list, expressible from the published command (§14.40).

    The ground is §7: a published report whose exclusions were applied through
    Python is one a stranger cannot reproduce from the published logs — the
    numbers reconcile and the reader cannot see which trajectories were dropped
    or why. So both halves are asserted here. The printed ``excluded`` term
    moves, and the JSON carries the entry with its reason, which is what makes
    the count checkable.

    The uuid is taken from a check the same logs report as **performed**, so the
    fixture can carry the difference: excluding a row a judge had already
    omitted would move ``excluded`` while every band stayed where it was, and the
    assertion would pass without the row having left a denominator (§15).
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()

    assert clean_log.location
    logs = [scoring / "judge_opus_a.eval"]
    before = build_report([load_evaluation(clean_log.location, logs)])
    key = next(key for key, counts in sorted(before.counts.items()) if counts.performed)
    victim = before.matrices[key].rows[0].uuid
    assert before.counts[key].excluded == 0
    assert before.counts[key].observed == 1

    listing = tmp_path / "exclusions.json"
    listing.write_text(
        json.dumps([{"uuid": victim, "reason": "tool output truncated on review"}]),
        encoding="utf-8",
    )

    assert _report(clean_log, logs, "--exclusions", str(listing), "--json") == 0

    dumped = json.loads(capsys.readouterr().out)
    assert dumped["exclusions"] == [
        {"uuid": victim, "reason": "tool output truncated on review"}
    ]
    counts = dumped["counts"][key]
    assert counts["excluded"] == 1
    observed = (
        counts["performed"]
        + counts["partial"]
        + counts["not_performed"]
        + counts["split"]
    )
    assert observed == 0, "the excluded row still counted as an observation"
    assert counts["performed"] < before.counts[key].performed


def test_report_refuses_a_bad_exclusion_list(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other half: a supplied file either applies or refuses.

    Never degrades to an empty list — silently reading no exclusions produces a
    report that looks complete and counts trajectories review rejected (§5, §9).
    Both shapes the flag can meet are exercised: a file that is not there, and
    one whose entry carries no reason.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()
    logs = [scoring / "judge_opus_a.eval"]

    assert _report(clean_log, logs, "--exclusions", str(tmp_path / "absent.json")) == 1
    assert "ExclusionError" in capsys.readouterr().out

    reasonless = tmp_path / "reasonless.json"
    reasonless.write_text(
        json.dumps([{"uuid": "some-uuid", "reason": "   "}]), encoding="utf-8"
    )

    assert _report(clean_log, logs, "--exclusions", str(reasonless)) == 1
    printed = capsys.readouterr().out
    assert "ExclusionError" in printed and "no reason" in printed


def test_report_without_the_flag_excludes_nothing(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flag is optional, and its absence is no exclusions rather than an error.

    A report with nothing excluded is the ordinary case. This is what fails if
    the flag ever acquires a default path.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()

    assert _report(clean_log, [scoring / "judge_opus_a.eval"], "--json") == 0

    dumped = json.loads(capsys.readouterr().out)
    assert dumped["exclusions"] == []
    assert all(counts["excluded"] == 0 for counts in dumped["counts"].values())


def test_report_refuses_a_generation_log_with_no_scoring_logs(
    clean_log: EvalLog, capsys: pytest.CaptureFixture[str]
) -> None:
    """§5's input is both halves; a generation log alone carries no verdicts."""
    assert clean_log.location
    with pytest.raises(SystemExit) as raised:
        main(["report", "--generation", clean_log.location])
    assert raised.value.code == 2


def test_report_refuses_a_scoring_log_with_no_sidecar(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The sidecar is derived from the log's own path and is never supplied.

    A log whose sidecar is gone carries no provenance axes, so it cannot enter a
    report — `read_run_provenance` raises rather than comparing equal.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()
    provenance_sidecar_path(scoring / "judge_opus_a.eval").unlink()

    assert _report(clean_log, [scoring / "judge_opus_a.eval"]) == 1
    assert "no provenance sidecar" in capsys.readouterr().out


def test_report_refuses_the_same_scoring_log_supplied_twice(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The message was already right; it arrived as a stack trace.

    Copying a round directory and then naming both copies is the ordinary way
    into this, and counting one trajectory as two observations is what it costs.
    `main` reserves a traceback for a bug in this package, so the assertion is
    on the exit status and the printed line rather than on the exception type.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    original = scoring / "judge_opus_a.eval"
    copied = tmp_path / "copied"
    copied.mkdir()
    duplicate = copied / original.name
    duplicate.write_bytes(original.read_bytes())
    provenance_sidecar_path(duplicate).write_bytes(
        provenance_sidecar_path(original).read_bytes()
    )
    capsys.readouterr()

    assert _report(clean_log, [original, duplicate]) == 1

    printed = capsys.readouterr().out
    assert printed.startswith("PairingError: "), printed
    assert "are the same judge" in printed, printed
    assert "count one trajectory as two observations" in printed, printed


def test_report_refuses_a_generation_log_named_as_a_scoring_log(
    clean_log: EvalLog, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mirror of a scoring log named as the generation log, and it refused
    differently until the two took one type.

    Both are one mistake in two directions — the halves swapped — and an
    operator meets whichever they typed. This one reached the shell as a stack
    trace while the other printed a sentence and exited 1.
    """
    assert clean_log.location
    capsys.readouterr()

    assert _report(clean_log, [Path(clean_log.location)]) == 1

    printed = capsys.readouterr().out
    assert printed.startswith("PairingError: "), printed
    assert "no scorer carries" in printed, printed
    assert "A generation log is not a scoring log" in printed, printed


def test_a_well_formed_report_still_exits_zero(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other half of the two refusals above.

    A boundary that refused everything would pass both of them, so the ordinary
    invocation has to keep working: one generation log, one scoring log taken
    from it, exit 0 and rows printed rather than a refusal line.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()

    assert _report(clean_log, [scoring / "judge_opus_a.eval"]) == 0

    printed = capsys.readouterr().out
    assert "PairingError" not in printed, printed
    assert printed.strip(), "a clean report printed nothing"


# -- survey ----------------------------------------------------------------


def test_survey_reports_what_is_in_a_directory(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Headers only, and before any set has been chosen."""
    _mock_identity(monkeypatch, JudgeRecorder())
    _judge(clean_log, catalogue_file, tmp_path / "scoring")
    capsys.readouterr()

    assert main(["survey", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "report (" in printed, printed
    assert "combine" not in printed, printed


# -- the refusal boundary --------------------------------------------------


def test_an_unexpected_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main` translates named operator errors and nothing else.

    The edit this catches is widening the `except` to `Exception`, which would
    turn a bug in this package into an exit code an operator reads as their own
    mistake. `TypeError` stands for any such bug.
    """

    def boom(log_dir: object) -> None:
        raise TypeError("a defect in this package, not an operator error")

    monkeypatch.setattr("kgpbench.cli.survey", boom)

    with pytest.raises(TypeError):
        main(["survey", "."])


MARKED_UP_KEY_ERROR = (
    "ERROR: Unable to initialise Anthropic client\n\n"
    "No [bold][blue]ANTHROPIC_API_KEY[/blue][/bold] defined in the environment."
)
"""The framework's own message for the likeliest first-run failure, verbatim.

`environment_prerequisite_error` (`model/_providers/util/util.py`) wraps every
environment-variable name in ``[bold][blue]`` because inspect prints through
`rich`. Written here as a **literal** rather than obtained from that private
function: the assertion below is that no bracket tag survives, and an assertion
about an absence must be able to fire — sourcing the fixture from upstream would
make it pass vacuously the day upstream stopped using markup (§15).
"""


def test_a_missing_api_key_is_a_refusal_and_not_a_traceback(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other side of the same boundary, and the likeliest first-run failure.

    `judge_roles` calls `get_model`, which initialises a provider client — so an
    absent ``ANTHROPIC_API_KEY`` raises before `run_judges` is entered, at zero
    model calls and before the scoring directory exists. The framework's own
    message names the variable, and what the operator needs is that message
    rather than a stack trace through it. Raised here rather than relied on from
    the environment, so the test says the same thing on a machine that has a key.

    **What must be absent is the point** (F2). The message is composed with rich
    markup, so asserting only that ``ANTHROPIC_API_KEY`` appears is true of the
    marked-up string too — which is how a version printing
    ``No [bold][blue]ANTHROPIC_API_KEY[/blue][/bold] defined in the
    environment.`` at an operator survived a round. The assertion is therefore
    that the variable is named **and** that no bracket tag reaches the operator.
    """

    def unwired(given: object) -> dict[str, object]:
        raise PrerequisiteError(MARKED_UP_KEY_ERROR)

    monkeypatch.setattr("kgpbench.cli.judge_configs", lambda names=None: (MOCK_CONFIG,))
    monkeypatch.setattr("kgpbench.cli.judge_roles", unwired)
    scoring = tmp_path / "scoring"

    assert _judge(clean_log, catalogue_file, scoring) == 1

    printed = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in printed
    for tag in ("[bold]", "[/bold]", "[blue]", "[/blue]"):
        assert tag not in printed, (
            f"the framework's rich markup reached the operator as {tag!r}: this "
            "message is printed through builtin `print` again"
        )
    assert not scoring.exists()


def test_a_framework_message_survives_the_display_the_harness_runs_under(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The measured trap in the obvious spelling of F2, pinned so it stays shut.

    `rich.print` writes to rich's **global** console, and
    `inspect_ai._display.core.rich.rich_initialise` calls
    ``rich.reconfigure(quiet=True)`` for ``display="none"`` — which is what
    `run_judges` passes on every invocation. After that the global console emits
    nothing, so ``from rich import print`` would have swapped *the operator sees
    `[bold][blue]`* for *the operator sees nothing*: strictly worse than the
    defect it repairs.

    The reconfiguration is applied here explicitly rather than left to whatever
    a fixture happened to run, so this control does not depend on test ordering.
    """
    # `reconfigure` replaces rich's global console; register the current one so
    # that a session-wide side effect does not leak out of this test
    monkeypatch.setattr(rich, "_console", None, raising=False)
    rich_initialise(display="none")
    assert rich.get_console().quiet, "the reconfiguration under test did not happen"

    def unwired(given: object) -> dict[str, object]:
        raise PrerequisiteError(MARKED_UP_KEY_ERROR)

    monkeypatch.setattr("kgpbench.cli.judge_configs", lambda names=None: (MOCK_CONFIG,))
    monkeypatch.setattr("kgpbench.cli.judge_roles", unwired)

    assert _judge(clean_log, catalogue_file, tmp_path / "scoring") == 1

    printed = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in printed, (
        "the framework's message was swallowed: it is going through rich's "
        "global console, which inspect has reconfigured to quiet"
    )
    assert "[bold]" not in printed


def test_the_harness_own_refusals_do_not_go_through_rich(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The fence, and the half that keeps the pins honest (F2).

    Only messages the harness did not write are rendered. `rich` wraps to
    terminal width, and this project pins message strings — one printed through
    `rich` would acquire line breaks that differ between one terminal, another
    and a CI runner. So a refusal this package wrote must reach the operator
    with its own line breaks and nothing else, however wide or narrow the
    terminal the test runs under.
    """
    missing = tmp_path / "nope"

    assert main(["survey", str(missing)]) == 1

    printed = capsys.readouterr().out
    assert printed.startswith("FileNotFoundError: "), printed
    # one refusal, one line: `rich` at this width would have wrapped it
    assert len(printed.rstrip("\n").splitlines()) == 1, printed
    assert str(missing) in printed


def test_a_command_that_succeeds_exits_zero(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other half: the translation is not returning 1 for everything."""
    _mock_identity(monkeypatch, JudgeRecorder())

    assert _judge(clean_log, catalogue_file, tmp_path / "scoring") == 0
    assert "CatalogueUnresolved" not in capsys.readouterr().out


# -- the two published strings, which must say what the command does -------


def test_the_absent_sidecar_refusal_names_the_command_a_stranger_can_run(
    tmp_path: Path,
) -> None:
    """§14.38: the message names one route, and after `kgpbench judge` exists
    that route is the one a stranger can perform.

    Pinned as a literal rather than against the source that writes it, because
    the whole point of the string is what it tells someone who is not us (§15).
    Naming *one* route is part of the pin: the item records that describing it
    as naming two was itself a defect, so the Python function this string used to
    name must be gone as well as the command being present.

    **What this does not assert is that the message names no private command.**
    Writing that assertion means writing the private name in a file that
    publishes, which is the rule itself violated to test it — measured, and
    refused by the release guard. That rule holds over every publishing file at
    once and is enforced there, not here.
    """
    with pytest.raises(FileNotFoundError) as raised:
        read_run_provenance(tmp_path / "absent.eval")

    message = str(raised.value)
    assert "`kgpbench judge`" in message
    assert "run_judges" not in message


def test_the_rung_three_refusal_says_the_two_remedies_differ() -> None:
    """§14.42's held clause: `configs=` alone depends on the log, and the message
    must say so — and must say what `kgpbench judge` actually does.

    The command wires ``model_roles``; the message names that as the route that
    always runs and states the condition the other one carries. Both halves are
    pinned, because a message naming two interchangeable-looking remedies is the
    defect this replaced.
    """
    with pytest.raises(KeyError) as raised:
        _resolve_judge_configs(["judge_opus_a"], None, None)

    message = str(raised.value)
    assert "model_roles=" in message and "configs=" in message
    assert "`kgpbench judge`" in message
    assert "generation log's own header" in message

    # and the paragraph of reasoning that stood beside them is gone: a published
    # refusal names the condition and the remedy, and the *why* moved to
    # `_resolve_judge_configs`'s docstring (§14.40). Pinned by absence, so the
    # old string coming back fails this test rather than passing it.
    for rationale in (
        "resolves its model from",
        "nothing to read it off",
    ):
        assert rationale not in message, (
            f"design rationale is back in the refusal: {rationale!r}. It belongs "
            "in the docstring, beside the code"
        )
