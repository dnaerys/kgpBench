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
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._display.core.rich import rich_initialise
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from mock_harness import clean_model
from mock_judging import (
    CATALOGUE,
    INSTANCES,
    JudgeRecorder,
    generation_log,
    generation_outputs,
    generation_task,
    judge_model,
    verdict_reply,
)

from kgpbench import (
    JudgeConfig,
    catalogue_arg,
    provenance_sidecar_path,
    read_run_provenance,
)
from kgpbench.cli import _parser, main
from kgpbench.judge import JUDGE_MOCKLLM
from kgpbench.judge_prompt import JUDGE_PROMPT_VERSION
from kgpbench.report import build_report, load_evaluation, read_exclusions
from kgpbench.scoring import JudgeProviderUnavailable, _resolve_judge_configs

VERDICTS = {"B6": "performed", "B5": "performed", "B1": "not_performed"}

MOCK_CONFIG = JudgeConfig(
    name="judge_opus_5",
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


def _mock_identity(
    monkeypatch: pytest.MonkeyPatch,
    recorder: JudgeRecorder,
    verdicts: dict[str, str] = VERDICTS,
) -> None:
    """Point the command's one identity seam at the fixture."""

    def configs(names: object = None) -> tuple[JudgeConfig, ...]:
        return (MOCK_CONFIG,)

    monkeypatch.setattr("kgpbench.cli.judge_configs", configs)
    monkeypatch.setattr(
        "kgpbench.cli.judge_roles",
        lambda given: {
            config.name: judge_model(recorder, verdict_reply(verdicts))
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
            "judge_opus_5",
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

    log = scoring / "judge_opus_5.eval"
    assert log.exists()
    assert provenance_sidecar_path(log).exists()
    assert not (scoring / "judge_opus_5_b.eval").exists()


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

    provenance = read_run_provenance(scoring / "judge_opus_5.eval")
    assert [judge.name for judge in provenance.judges] == ["judge_opus_5"]
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
            "judge_opus_5",
            "--scoring-dir",
            str(tmp_path / "scoring"),
        ]
    )

    assert status == 1
    assert "CatalogueUnresolved" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("absent.json", None),
        ("list.json", b"[1, 2]"),
        ("latin.json", b"\xff\xfe not text"),
        ("a-directory", b""),
    ],
)
def test_judge_names_the_shipped_release_when_the_catalogue_does_not_load(
    name: str,
    content: bytes | None,
    clean_log: EvalLog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1, the reader's refusal, then the one move that needs no file.

    Three of these printed a traceback until September 2026 — `TypeError`,
    `UnicodeDecodeError`, `IsADirectoryError` — none of which `main` translates.
    Nothing is judged and nothing is written. The negative half is every test
    here that judges under a wire form that loads and prints no such line.
    """
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    named = tmp_path / name
    if name == "a-directory":
        named.mkdir()
    elif content is not None:
        named.write_bytes(content)
    scoring = tmp_path / "scoring"

    assert clean_log.location
    status = main(
        [
            "judge",
            "--generation",
            clean_log.location,
            "--catalogue",
            str(named),
            "--judge",
            "judge_opus_5",
            "--scoring-dir",
            str(scoring),
        ]
    )

    printed = capsys.readouterr().out
    assert status == 1
    assert printed.startswith("CatalogueUnresolved: "), printed
    assert str(named) in printed.splitlines()[0], printed
    assert printed.rstrip().endswith("To use the shipped catalogue, pass `c1`")
    assert not recorder.prompts
    assert not scoring.exists()


def test_a_catalogue_that_loads_carries_no_release_remedy(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The negative half, stated: a wire form that loads is judged under, and
    the remedy for one that does not is printed nowhere."""
    _mock_identity(monkeypatch, JudgeRecorder())

    assert _judge(clean_log, catalogue_file, tmp_path / "scoring") == 0

    printed = capsys.readouterr().out
    assert f"catalogue: read from {catalogue_file}" in printed
    assert "To use the shipped catalogue" not in printed


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
                "judge_opus_5",
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
            "judge_opus_5",
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


def _judge_named(generation: str | Path, scoring: Path, catalogue_file: Path) -> int:
    """`kgpbench judge` with ``--generation`` spelled as the operator typed it."""
    return main(
        [
            "judge",
            "--generation",
            str(generation),
            "--catalogue",
            str(catalogue_file),
            "--judge",
            "judge_opus_5",
            "--scoring-dir",
            str(scoring),
        ]
    )


def test_judge_refuses_a_scoring_log_as_the_generation_log(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The §6.6 hazard, refused before any model call and before any write.

    A judge's own model events are spliced into every sample it scores, so a
    scoring log re-derives completeness from the judge's stop reason, and a
    truncated trajectory can come back judgeable. The command judged one until
    September 2026.

    **One condition, both commands.** The printed refusal is the one `kgpbench
    report` prints for the same log named as ``--generation``, byte for byte,
    because both go through `assert_generation_log`. The negative half is every
    test above that judges a generation log and exits 0.
    """
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    first = tmp_path / "scoring"
    assert _judge(clean_log, catalogue_file, first) == 0
    scoring_log = first / "judge_opus_5.eval"
    calls = len(recorder.prompts)
    assert calls, "the fixture judge was never called on the generation log"
    capsys.readouterr()

    second = tmp_path / "rejudged"
    assert _judge_named(scoring_log, second, catalogue_file) == 1

    printed = capsys.readouterr().out
    refusal = printed.splitlines()[-1]
    assert refusal.startswith("PairingError: "), printed
    assert "so this is a scoring log and not a generation log" in refusal, printed
    assert len(recorder.prompts) == calls, "a judge ran over the scoring log"
    assert not second.exists(), "the scoring directory was created"

    main(["report", "--generation", str(scoring_log), "--scoring", str(scoring_log)])
    assert capsys.readouterr().out.splitlines()[-1] == refusal


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("garbage.eval", b"not an eval log at all"),
        ("empty.eval", b""),
        ("config.json", b'{"a": 1}'),
        ("notes.txt", b"a note"),
        ("a-directory.eval", None),
    ],
)
def test_judge_refuses_a_file_that_is_not_a_log(
    name: str,
    content: bytes | None,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1 with the condition and the remedy, never a traceback.

    Each shape reaches the framework's reader and fails there differently —
    ``EOCD not found``, a validation error, ``No recorder for location``,
    `IsADirectoryError` — and none of those types is on `main`'s boundary. The
    translation is at the read, into the type the boundary already carries.
    """
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    named = tmp_path / name
    if content is None:
        named.mkdir()
    else:
        named.write_bytes(content)
    scoring = tmp_path / "scoring"

    assert _judge_named(named, scoring, catalogue_file) == 1

    refusal = capsys.readouterr().out.splitlines()[-1]
    assert refusal.startswith(f"PairingError: {named}: "), refusal
    assert "could not be read as an Inspect log" in refusal, refusal
    assert "Name the .eval file `inspect eval` wrote" in refusal, refusal
    assert not recorder.prompts
    assert not scoring.exists()


def test_judge_refuses_a_missing_generation_file(
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A path naming nothing: exit 1, the condition and the remedy.

    It was a refusal already, as the framework's bare ``[Errno 2]``, which names
    the condition and no move.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    missing = tmp_path / "generation" / "absent.eval"
    scoring = tmp_path / "scoring"

    assert _judge_named(missing, scoring, catalogue_file) == 1

    refusal = capsys.readouterr().out.splitlines()[-1]
    assert refusal.startswith(f"FileNotFoundError: {missing}: no such file"), refusal
    assert "Name the .eval file `inspect eval` wrote" in refusal, refusal
    assert not scoring.exists()


def test_judge_refuses_the_generation_logs_own_directory(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1 with a remedy, and nothing written — it was a stack trace.

    Both spellings of the one directory. The negative half is the ordinary
    invocation with a scoring directory of its own, asserted last in the same
    test so that a refusal firing on every directory cannot pass it.
    """
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    assert clean_log.location
    own = Path(clean_log.location).parent
    before = sorted(path.name for path in own.iterdir())

    for spelling in (own, own / "."):
        assert _judge(clean_log, catalogue_file, spelling) == 1
        refusal = capsys.readouterr().out.splitlines()[-1]
        assert refusal.startswith("FileExistsError: scoring directory "), refusal
        assert "is the generation log's own directory" in refusal, refusal
        assert "(`--scoring-dir`)" in refusal, refusal

    assert sorted(path.name for path in own.iterdir()) == before
    assert not recorder.prompts

    scoring = tmp_path / "scoring"
    assert _judge(clean_log, catalogue_file, scoring) == 0
    assert (scoring / "judge_opus_5.eval").exists()


def test_judge_refuses_an_inspect_log_this_harness_did_not_generate(
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real Inspect log with no provenance record, refused before any write.

    The refusal already existed, in `scoring_provenance`, as a bare `ValueError`
    raised after the scoring directory had been created. It is now the one
    `kgpbench report` makes on the same log, through the same function.
    """
    foreign = inspect_eval(
        Task(dataset=[Sample(input="q", target="x")]),
        model=clean_model(),
        log_dir=str(tmp_path / "foreign"),
        display="none",
    )[0]
    assert foreign.location
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    scoring = tmp_path / "scoring"

    assert _judge_named(foreign.location, scoring, catalogue_file) == 1

    refusal = capsys.readouterr().out.splitlines()[-1]
    assert refusal.startswith("PairingError: "), refusal
    assert "carries no provenance record" in refusal, refusal
    assert not recorder.prompts
    assert not scoring.exists()


def test_every_command_names_the_operator_route_for_a_log_with_no_record(
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One refusal, three commands, and its remedy is the move an operator makes.

    The record is written by the generation task, and an operator reaches that
    task as `inspect eval kgpbench/generation`. The remedy named `task_kwargs`
    until September 2026, a Python helper no command reaches. `judge` and
    `report` print the same last line and `survey` prints it under ``refused``,
    because all three read the log through one loader. The negative half is
    :func:`test_a_generation_log_carrying_its_record_is_judged_and_reported`.
    """
    foreign = inspect_eval(
        Task(dataset=[Sample(input="q", target="x")]),
        model=clean_model(),
        log_dir=str(tmp_path / "foreign"),
        display="none",
    )[0]
    location = str(foreign.location)
    _mock_identity(monkeypatch, JudgeRecorder())
    capsys.readouterr()

    assert _judge_named(location, tmp_path / "scoring", catalogue_file) == 1
    judged = capsys.readouterr().out.splitlines()[-1]
    assert main(["report", "--generation", location, "--scoring", location]) == 1
    reported = capsys.readouterr().out.splitlines()[-1]
    assert main(["survey", str(tmp_path / "foreign")]) == 0
    surveyed = capsys.readouterr().out.splitlines()

    remedy = (
        "Generate the log with `inspect eval kgpbench/generation`, which writes "
        "the record"
    )
    assert judged.startswith("PairingError: "), judged
    assert judged.endswith(remedy), judged
    assert "task_kwargs" not in judged, judged
    assert reported == judged
    message = judged.removeprefix("PairingError: ")
    assert f"refused ({foreign.eval.model}): {message}" in surveyed, surveyed


def test_a_generation_log_carrying_its_record_is_judged_and_reported(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The negative half: a log the generation task wrote passes all three.

    A refusal firing on every log, or on the record's mere presence, fails here
    on the first command it reaches.
    """
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    scoring = tmp_path / "scoring"
    capsys.readouterr()

    assert _judge(clean_log, catalogue_file, scoring) == 0
    assert recorder.prompts, "the judge never ran"
    assert _report(clean_log, [scoring / "judge_opus_5.eval"]) == 0
    assert main(["survey", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "provenance record" not in printed, printed
    assert "refused " not in printed, printed
    assert "report (" in printed, printed


def _named_generation_log(tmp_path: Path, name: str, **eval_kwargs: object) -> str:
    """A generation log from the route `inspect eval` takes: the model **named**.

    `EvalSpec.model_generate_config` records the eval-level configuration only
    when the framework built the model from a name (`mock_judging`'s
    `generation_outputs`), so both halves of the eval-level control are built
    here, and differ in ``eval_kwargs`` alone. The shipped task refuses an
    eval-level effort before it builds, so the shipped solver is run through
    the fixture task, which is how a log from somewhere else is reconstructed.
    """
    logs = inspect_eval(
        generation_task(f"cli-{name}", INSTANCES),
        model="mockllm/model",
        model_args={"custom_outputs": generation_outputs("an answer")},
        log_dir=str(tmp_path / name),
        display="none",
        **eval_kwargs,
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].location
    return str(logs[0].location)


@pytest.mark.parametrize(
    ("eval_kwargs", "named"),
    [
        ({"reasoning_effort": "low"}, "reasoning_effort"),
        ({"max_tokens": 999}, "max_tokens"),
    ],
)
def test_judge_refuses_a_generation_log_whose_run_set_an_eval_level_value(
    eval_kwargs: dict[str, object],
    named: str,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The log `kgpbench report` refuses, refused before any judge spends on it.

    Every verdict judged over it would pair with a generation log no report
    takes. `judge` judged it until September 2026. **One condition, both
    commands**: the last line printed is the one `report` prints for the same
    log, because both read it through the same check.
    """
    location = _named_generation_log(tmp_path, "overridden", **eval_kwargs)
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    scoring = tmp_path / "scoring"

    assert _judge_named(location, scoring, catalogue_file) == 1

    refusal = capsys.readouterr().out.splitlines()[-1]
    assert refusal.startswith(f"EvalConfigRecorded: {location}: "), refusal
    assert named in refusal, refusal
    assert not recorder.prompts, "a judge ran over the log"
    assert not scoring.exists(), "the scoring directory was created"

    main(["report", "--generation", location, "--scoring", location])
    assert capsys.readouterr().out.splitlines()[-1] == refusal


def test_judge_still_judges_a_generation_log_with_nothing_set_at_eval_level(
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative half, on the route where the field is written at all.

    The same builder as the refusal above with nothing set at eval level, so a
    check that fired on every log read this way, or on the field's mere
    existence, fails here.
    """
    location = _named_generation_log(tmp_path, "clean")
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    scoring = tmp_path / "scoring"

    assert _judge_named(location, scoring, catalogue_file) == 0

    assert (scoring / "judge_opus_5.eval").exists()
    assert recorder.prompts, "the judge never ran"


def test_the_judging_summary_names_every_exclusion_cause(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The total and the cause of every excluded trajectory, on one line.

    The fixture's ``TRUNC.`` trajectory stops at ``max_tokens`` and is excluded
    as ``completeness:truncated``; the other two are judged.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    capsys.readouterr()

    assert _judge(clean_log, catalogue_file, tmp_path / "scoring") == 0

    lines = capsys.readouterr().out.splitlines()
    assert "judged 2 trajectories, excluded 1: 1 completeness:truncated" in lines, lines


def test_a_judging_summary_with_nothing_excluded_is_the_bare_totals(
    tmp_path: Path,
    catalogue_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other half: no cause appended when there is none to name."""
    kept = INSTANCES.model_copy(update={"instances": INSTANCES.instances[:2]})
    log = generation_log(tmp_path, name="clean-only", instances=kept)
    _mock_identity(monkeypatch, JudgeRecorder())
    capsys.readouterr()

    assert _judge(log, catalogue_file, tmp_path / "scoring") == 0

    lines = capsys.readouterr().out.splitlines()
    assert "judged 2 trajectories, excluded 0" in lines, lines


def _truncated_only(tmp_path: Path) -> EvalLog:
    """A generation log whose every trajectory is excluded before judging."""
    only = INSTANCES.model_copy(update={"instances": [INSTANCES.instances[2]]})
    return generation_log(tmp_path, name="all-excluded", instances=only)


def test_judging_an_evaluation_whose_every_trajectory_was_excluded_still_writes(
    tmp_path: Path,
    catalogue_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Not a refusal: the pair is written and the causes are printed.

    A pass that judged nothing is still a record of what was excluded and why,
    and the report reads its exclusions through it.
    """
    log = _truncated_only(tmp_path)
    recorder = JudgeRecorder()
    _mock_identity(monkeypatch, recorder)
    scoring = tmp_path / "scoring"
    capsys.readouterr()

    assert _judge(log, catalogue_file, scoring) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "judged 0 trajectories, excluded 1: 1 completeness:truncated" in lines
    assert (scoring / "judge_opus_5.eval").exists()
    assert provenance_sidecar_path(scoring / "judge_opus_5.eval").exists()
    assert not recorder.prompts


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

    assert _report(clean_log, [scoring / "judge_opus_5.eval"]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    # the check lines, then the pre-judging exclusions block, which is not one
    tail = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("excluded before judging")
    )
    checks = lines[:tail]
    assert checks
    assert all("trajectories" in line and "disagreement" in line for line in checks)
    assert not any("disagreement" in line for line in lines[tail:]), lines


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
        [load_evaluation(clean_log.location, [scoring / "judge_opus_5.eval"])]
    )
    key = next(iter(sorted(report.counts)))

    assert _report(clean_log, [scoring / "judge_opus_5.eval"], "--json") == 0

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
    logs = [scoring / "judge_opus_5.eval"]
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
    logs = [scoring / "judge_opus_5.eval"]

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

    assert _report(clean_log, [scoring / "judge_opus_5.eval"], "--json") == 0

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
    provenance_sidecar_path(scoring / "judge_opus_5.eval").unlink()

    assert _report(clean_log, [scoring / "judge_opus_5.eval"]) == 1
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
    original = scoring / "judge_opus_5.eval"
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


def test_report_refuses_a_path_that_holds_no_log_on_either_side(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing, or not a log, as either half: exit 1 with the remedy for that half.

    The read is shared with `kgpbench judge`, so the report reaches the same
    translation; a file that is not a log was a stack trace here as well.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    scored = scoring / "judge_opus_5.eval"
    garbage = tmp_path / "garbage.eval"
    garbage.write_bytes(b"not an eval log at all")
    missing = tmp_path / "absent.eval"
    assert clean_log.location
    capsys.readouterr()

    cases = [
        (garbage, scored, "PairingError", "Name the .eval file `inspect eval`"),
        (missing, scored, "FileNotFoundError", "Name the .eval file `inspect eval`"),
        (clean_log.location, garbage, "PairingError", "`kgpbench judge` wrote"),
        (clean_log.location, missing, "FileNotFoundError", "`kgpbench judge` wrote"),
    ]
    for generation, scoring_log, kind, remedy in cases:
        status = main(
            ["report", "--generation", str(generation), "--scoring", str(scoring_log)]
        )
        refusal = capsys.readouterr().out.splitlines()[-1]
        assert status == 1, refusal
        assert refusal.startswith(f"{kind}: "), refusal
        assert remedy in refusal, refusal


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

    assert _report(clean_log, [scoring / "judge_opus_5.eval"]) == 0

    printed = capsys.readouterr().out
    assert "PairingError" not in printed, printed
    assert printed.strip(), "a clean report printed nothing"


def test_report_prints_the_pre_judging_exclusions_after_the_check_lines(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The check lines, then the exclusions by cause, and the report reconciles.

    The negative half of the new block: a report with exclusions of both kinds —
    one trajectory excluded before judging, one after, by review — still
    reconciles, and the review exclusion is still counted on its check line and
    never in the block.
    """
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(clean_log, catalogue_file, scoring)
    capsys.readouterr()
    assert clean_log.location
    logs = [scoring / "judge_opus_5.eval"]
    report = build_report([load_evaluation(clean_log.location, logs)])
    key = next(key for key, counts in sorted(report.counts.items()) if counts.performed)
    victim = report.matrices[key].rows[0].uuid
    listing = tmp_path / "exclusions.json"
    listing.write_text(
        json.dumps([{"uuid": victim, "reason": "reviewed"}]), encoding="utf-8"
    )

    assert _report(clean_log, logs, "--exclusions", str(listing)) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[-2:] == [
        "excluded before judging — trajectories 3 · judged 2 · excluded 1",
        "  completeness:truncated: 1",
    ], lines
    checks = lines[:-2]
    assert checks and all("disagreement" in line for line in checks), lines
    assert any("· 1 excluded ·" in line for line in checks), checks

    reviewed = build_report(
        [load_evaluation(clean_log.location, logs)],
        exclusions=read_exclusions(listing),
    )
    assert reviewed.reconciles()
    assert reviewed.counts[key].excluded == 1
    assert reviewed.completeness.by_cause == {"completeness:truncated": 1}


def test_report_prints_the_exclusions_block_when_nothing_was_excluded(
    tmp_path: Path,
    catalogue_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On every report: a zero is printed as a zero, with no cause lines."""
    kept = INSTANCES.model_copy(update={"instances": INSTANCES.instances[:2]})
    log = generation_log(tmp_path, name="clean-only", instances=kept)
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(log, catalogue_file, scoring)
    capsys.readouterr()

    assert _report(log, [scoring / "judge_opus_5.eval"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == (
        "excluded before judging — trajectories 2 · judged 2 · excluded 0"
    ), lines
    assert all("disagreement" in line for line in lines[:-1]), lines


def test_a_report_whose_every_trajectory_was_excluded_prints_its_exclusions(
    tmp_path: Path,
    catalogue_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It printed nothing and exited 0; the causes were in ``--json`` alone."""
    log = _truncated_only(tmp_path)
    _mock_identity(monkeypatch, JudgeRecorder())
    scoring = tmp_path / "scoring"
    _judge(log, catalogue_file, scoring)
    capsys.readouterr()

    assert _report(log, [scoring / "judge_opus_5.eval"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "excluded before judging — trajectories 1 · judged 0 · excluded 1",
        "  completeness:truncated: 1",
    ]


def test_a_not_reachable_vote_on_a_reachable_check_reports_as_an_omission(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The report builds, counts it ``unobserved``, and ``--json`` says omitted.

    It stopped the report with a traceback: the vote left the check neither
    answered nor in ``missing_verdicts``, and `judge_reading` refuses a record
    that contradicts itself.
    """
    _mock_identity(monkeypatch, JudgeRecorder(), {**VERDICTS, "B6": "not_reachable"})
    scoring = tmp_path / "scoring"
    assert _judge(clean_log, catalogue_file, scoring) == 0
    capsys.readouterr()
    logs = [scoring / "judge_opus_5.eval"]

    assert _report(clean_log, logs) == 0
    lines = capsys.readouterr().out.splitlines()
    (b6,) = [line for line in lines if line.startswith("B6 (required)")]
    assert "0 observed" in b6 and "1 unobserved" in b6, b6

    assert _report(clean_log, logs, "--json") == 0
    dumped = json.loads(capsys.readouterr().out)
    (row,) = dumped["matrices"]["B6:required"]["rows"]
    (cell,) = row["cells"].values()
    assert cell["verdict"] is None and cell["abstention"] == "omitted", cell


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


def test_a_missing_provider_library_reaches_the_operator_as_a_refusal(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The condition a stranger meets first, on the boundary rather than through it.

    `inspect_ai` declares no provider client libraries, so a fresh install of
    this package can reach `mockllm/model` and no paid model at all — which
    makes *the provider is not installed* the likeliest first failure of
    `kgpbench judge`, ahead of the missing key it used to be ranked behind. It
    is an operator condition with a remedy, so it belongs on the refusal side
    of `main`'s boundary and not in a traceback.

    What is asserted here is the boundary: exit 1, the message printed, and
    nothing written. What the message *says* is asserted where it is composed
    (`test_scoring_driver`), because it is composed there.
    """

    def unavailable(given: object) -> dict[str, object]:
        raise JudgeProviderUnavailable(
            "judge 'judge_opus_5' runs on 'anthropic/claude-opus-5', and this "
            "environment cannot reach that provider"
        )

    monkeypatch.setattr("kgpbench.cli.judge_configs", lambda names=None: (MOCK_CONFIG,))
    monkeypatch.setattr("kgpbench.cli.judge_roles", unavailable)
    scoring = tmp_path / "scoring"

    assert _judge(clean_log, catalogue_file, scoring) == 1

    printed = capsys.readouterr().out
    assert printed.startswith("JudgeProviderUnavailable: "), printed
    assert "judge_opus_5" in printed
    assert not scoring.exists(), "a scoring directory survived the refusal"


def test_the_rehearsal_caveat_is_printed_beside_the_judge_line(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--help` is read once; the run output is read every time and afterwards.

    An operator who chose `judge_mockllm` six weeks ago and is now looking at a
    scoring log and its transcript has no reason to open `--help` again, and the
    thing they most need to know about that log is that its verdicts are not
    verdicts. So the caveat is printed beside the judge line on every run and
    not only where the flag is chosen.
    """
    recorder = JudgeRecorder()

    def configs(names: object = None) -> tuple[JudgeConfig, ...]:
        return (
            JudgeConfig(
                name=JUDGE_MOCKLLM,
                model="mockllm/model",
                prompt_version=JUDGE_PROMPT_VERSION,
            ),
        )

    monkeypatch.setattr("kgpbench.cli.judge_configs", configs)
    monkeypatch.setattr(
        "kgpbench.cli.judge_roles",
        lambda given: {
            config.name: judge_model(recorder, verdict_reply(VERDICTS))
            for config in given
        },
    )
    assert clean_log.location
    status = main(
        [
            "judge",
            "--generation",
            clean_log.location,
            "--catalogue",
            str(catalogue_file),
            "--judge",
            JUDGE_MOCKLLM,
            "--scoring-dir",
            str(tmp_path / "scoring"),
        ]
    )

    assert status == 0
    printed = capsys.readouterr().out
    assert "REHEARSAL" in printed
    assert "means nothing" in printed


def test_a_paid_judge_gets_no_rehearsal_caveat(
    clean_log: EvalLog,
    catalogue_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The negative half. A caveat printed on every run says nothing about any.

    `MOCK_CONFIG` is named ``judge_opus_5`` and wired to a fixture model, which
    is exactly the shape that would make an unconditional print look correct:
    the run is free, and the line must still be absent, because what decides it
    is the **judge** and not what happens to be behind the role.
    """
    _mock_identity(monkeypatch, JudgeRecorder())

    assert _judge(clean_log, catalogue_file, tmp_path / "scoring") == 0
    assert "REHEARSAL" not in capsys.readouterr().out


def test_the_rehearsal_judge_is_named_where_the_operator_will_see_it() -> None:
    """`--judge`'s own help says the mock's verdicts are worthless (§14.40).

    The caveat has to live on the surface an operator meets while choosing, not
    only in a README they may never open. Asserted on the parser rather than on
    the docstring, because the parser is what `--help` prints.
    """
    actions = {
        action.dest: action
        for action in _parser()._actions  # noqa: SLF001 - argparse has no public reader
    }
    # the subparser holds `--judge`, so reach it the way argparse stores it
    subparsers = actions["command"]
    judge_parser = subparsers.choices["judge"]
    (judge_action,) = [
        action for action in judge_parser._actions if action.dest == "judge"
    ]

    assert JUDGE_MOCKLLM in (judge_action.choices or ())
    assert judge_action.help is not None
    assert JUDGE_MOCKLLM in judge_action.help
    assert "MEANINGLESS" in judge_action.help
    assert "costs nothing" in judge_action.help


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
        _resolve_judge_configs(["judge_opus_5"], None, None)

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
