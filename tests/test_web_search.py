"""Web search granted per instance — the declaration, the tools, and the refusal.

Three things are established, in the order §4 and §6.3 rule them:

1. **The declaration is instance content and moves nothing it should not.**
   ``Instance.web_search`` is left out of the serialised instance while it is
   ``None``, so every instance written before the field keeps its id and every
   set its digest, and a stored payload without the key loads as no search. A
   key outside Inspect's six internal providers is refused, the external three
   among them.
2. **A sample that declares search gets the vendor's search beside the OneKGPd
   tools, and a sample that does not gets exactly what it got before.** All six
   internal providers are named, every time, with the instance's options merged
   in, so no route reaches a bare ``web_search()``.
3. **A model with no search of its own is refused before any model call**, at
   both sites: the registered task's body, which writes no log at all, and the
   solver, which errors every sample of the run with no model event.

`mockllm/model` throughout, and it is the model the refusal refuses — so the
tests that observe the tools a search sample receives disable the refusal by
monkeypatch and say so. The live tests reach the OneKGPd server for its tool
list and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from harness_fixtures import (
    DATASET_ID,
    FIXTURE_ALPHA,
    write_instance_module,
    write_set_config,
)
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.event import ModelEvent
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.tool import ToolDef
from pydantic import ValidationError

from kgpbench import (
    INTERNAL_SEARCH_PROVIDERS,
    SEARCH_MODEL_APIS,
    SET_CONFIG_ENV,
    Instance,
    InstanceSet,
    SolverConfig,
    WebSearchUnavailable,
    assert_web_search_available,
    build_dataset,
    build_generation_task,
    generation,
    read_log,
    research_agent,
    resolve_set,
    web_search_tool,
)

MCP_URL = "https://db.dnaerys.org/mcp"

MOCK_CONFIG = SolverConfig(max_output_tokens=16_000, turn_cap=4, reasoning_effort=None)

SEARCHING = Instance.with_derived_id(
    prompt="Fixture search. What does the public record say about the cohort?",
    dataset_snapshot=DATASET_ID,
    web_search={"anthropic": {"max_uses": 3, "blocked_domains": ["example.com"]}},
)
"""A search instance, with options on one provider, built the way modules build."""

SEARCHING_DEFAULTS = Instance.with_derived_id(
    prompt="Fixture search, every vendor's defaults.",
    dataset_snapshot=DATASET_ID,
    web_search={},
)


# -- 1. the declaration ------------------------------------------------------


def test_an_instance_that_declares_no_search_serialises_without_the_key() -> None:
    """The rule for a field added after a release, on every serialisation route."""
    assert FIXTURE_ALPHA.web_search is None
    assert "web_search" not in FIXTURE_ALPHA.model_dump(mode="json")
    assert "web_search" not in FIXTURE_ALPHA.model_dump()
    assert '"web_search"' not in FIXTURE_ALPHA.model_dump_json()
    assert "web_search" not in FIXTURE_ALPHA.sample_metadata()["instance"]


def test_a_stored_instance_without_the_key_loads_as_no_search_and_keeps_its_id() -> (
    None
):
    """What a release reads back out of a log its predecessor wrote.

    The payload is the stored wire form — no ``web_search`` key at all — and it
    loads as no search, and its content still derives the id it was stored
    under. Asserted on the published logs themselves in the round's report; here
    on a payload of the same shape, since no test reads a fixed log (§15).
    """
    from kgpbench_demo.wnt10a import INSTANCE

    # the instance as it was published, before it declared OneKGPd's dataset id:
    # the earlier dataset string, and the id that content derived
    published = INSTANCE.model_copy(
        update={
            "dataset_snapshot": "onekgpd-variants-138044723-samples-3202",
            "id": "0584547da36abd7d",
        }
    )
    stored = published.sample_metadata()
    assert "web_search" not in stored["instance"]
    loaded = Instance.from_sample_metadata(stored)
    assert loaded.web_search is None
    # the id that instance was published under, derived again from its content
    assert loaded.opaque_id == loaded.id == "0584547da36abd7d"


def test_an_explicit_null_reads_as_the_missing_key_does() -> None:
    """``null`` and absence are one meaning, so neither moves an id."""
    payload = {**FIXTURE_ALPHA.model_dump(mode="json"), "web_search": None}
    loaded = Instance.model_validate(payload)
    assert loaded.web_search is None
    assert loaded.opaque_id == FIXTURE_ALPHA.opaque_id


def test_a_declaration_is_content_and_moves_the_id() -> None:
    """``{}`` is a declaration, not an absence: search with every vendor's defaults."""
    base = {"prompt": SEARCHING_DEFAULTS.prompt, "dataset_snapshot": DATASET_ID}
    without = Instance.with_derived_id(**base)
    with_defaults = Instance.with_derived_id(**base, web_search={})
    with_options = Instance.with_derived_id(
        **base, web_search={"anthropic": {"max_uses": 3}}
    )
    assert len({without.id, with_defaults.id, with_options.id}) == 3
    assert with_defaults.model_dump(mode="json")["web_search"] == {}
    assert with_defaults == SEARCHING_DEFAULTS


def test_the_declaration_reaches_the_set_digest_and_task_version() -> None:
    plain = InstanceSet(name="s", instances=[FIXTURE_ALPHA])
    searching = InstanceSet(name="s", instances=[FIXTURE_ALPHA, SEARCHING])
    assert plain.version != searching.version
    assert not plain.declares_web_search
    assert searching.declares_web_search


@pytest.mark.parametrize("provider", INTERNAL_SEARCH_PROVIDERS)
def test_every_internal_provider_is_accepted(provider: str) -> None:
    instance = Instance.with_derived_id(prompt="p", web_search={provider: {}})
    assert instance.web_search == {provider: {}}


@pytest.mark.parametrize("provider", ["tavily", "google", "exa", "bing", "Anthropic"])
def test_a_key_outside_the_six_internal_providers_is_refused(provider: str) -> None:
    """The external three by construction, and any name Inspect would not accept."""
    with pytest.raises(ValidationError) as refused:
        Instance.with_derived_id(prompt="p", web_search={provider: {}})
    message = str(refused.value)
    assert repr(provider) in message
    for name in INTERNAL_SEARCH_PROVIDERS:
        assert name in message
    # the condition and the remedy, not the reason
    assert "§" not in message
    assert "generated answer" not in message


@pytest.mark.parametrize("value", [True, False, None, "on", 3])
def test_a_provider_value_must_be_an_option_mapping(value: object) -> None:
    """``False`` would switch a vendor's search off; none of these is an option set."""
    with pytest.raises(ValidationError):
        Instance.with_derived_id(prompt="p", web_search={"anthropic": value})


def test_the_internal_providers_are_inspect_s_internal_providers() -> None:
    """The six, taken from the framework rather than restated, less the external three."""
    from typing import get_args

    from inspect_ai.tool._tools._web_search._web_search import (
        EXTERNAL_PROVIDERS,
        WebSearchProvider,
    )

    internal = set(get_args(WebSearchProvider)) - set(EXTERNAL_PROVIDERS)
    assert set(INTERNAL_SEARCH_PROVIDERS) == internal
    assert len(INTERNAL_SEARCH_PROVIDERS) == 6


# -- 2. the tool ------------------------------------------------------------


def _options(declaration: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return dict(ToolDef(web_search_tool(declaration)).options or {})


def test_all_six_providers_are_named_with_the_instance_options_merged_in() -> None:
    options = _options({"anthropic": {"max_uses": 3}, "openai": {"x": 1}})
    assert options.pop("__internal_tool_type__") == "web_search"
    assert set(options) == set(INTERNAL_SEARCH_PROVIDERS)
    assert options["anthropic"] == {"max_uses": 3}
    assert options["openai"] == {"x": 1}
    for name in {"gemini", "grok", "mistral", "perplexity"}:
        assert options[name] == {}


def test_an_empty_declaration_names_all_six_with_their_defaults() -> None:
    options = _options({})
    options.pop("__internal_tool_type__")
    assert options == {name: {} for name in INTERNAL_SEARCH_PROVIDERS}


def test_the_google_environment_cannot_turn_search_external(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The silent move a bare ``web_search()`` makes, and that naming the six closes.

    The control is in the test: with both variables set, the bare call resolves
    to Google's external search and nothing internal.
    """
    from inspect_ai.tool import web_search

    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "placeholder")
    monkeypatch.setenv("GOOGLE_CSE_ID", "placeholder")
    bare = dict(ToolDef(web_search()).options or {})
    bare.pop("__internal_tool_type__")
    assert set(bare) == {"google"}, (
        "the control: the hazard is live in this environment"
    )

    options = _options({})
    options.pop("__internal_tool_type__")
    assert set(options) == set(INTERNAL_SEARCH_PROVIDERS)


def test_the_declaration_is_not_mutated_by_building_the_tool() -> None:
    declaration = {"anthropic": {"max_uses": 3}}
    _options(declaration)
    assert declaration == {"anthropic": {"max_uses": 3}}


# -- 3. the refusal, as a function ------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-5",
        "anthropic/bedrock/anthropic.claude-sonnet-5",
        "openai/gpt-5",
        "google/gemini-2.5-pro",
        "grok/grok-4",
        "mistral/mistral-large-latest",
        "perplexity/sonar",
    ],
)
def test_a_model_api_with_internal_search_passes(model: str) -> None:
    assert_web_search_available(model)


@pytest.mark.parametrize(
    "model",
    [
        "mockllm/model",
        "bedrock/anthropic.claude-sonnet-5",
        "openai-api/x/y",
        "none/none",
    ],
)
def test_a_model_api_without_internal_search_is_refused(model: str) -> None:
    with pytest.raises(WebSearchUnavailable):
        assert_web_search_available(model)


def test_the_refusal_names_the_condition_and_the_remedy_and_nothing_else() -> None:
    """A published string (§16): condition, remedy — and the dropped clauses absent.

    The absences are the pin. The reasons — external providers put another
    model between the agent and the web, a client tool raises out of
    `execute_tools` — are the docstring's, and none of them is here.
    """
    with pytest.raises(WebSearchUnavailable) as refused:
        assert_web_search_available("mockllm/model")
    message = str(refused.value)
    assert message.startswith("mockllm/model: ")
    assert "declares web search" in message
    assert "mockllm model API has no web search of its own" in message
    for api in SEARCH_MODEL_APIS:
        assert api in message
    assert "select a set whose instances declare no web search" in message
    for absent in (
        "§",
        "because",
        "Tavily",
        "Exa",
        "external",
        "client tool",
        "ValueError",
        "execute_tools",
        "sibling",
        "not_performed",
    ):
        assert absent not in message, absent


# -- 4. the refusal, on both routes -----------------------------------------


@pytest.fixture
def search_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Module-backed sets: one plain, one search, one mixed, under the env config."""
    directory = tmp_path / "search_modules"
    directory.mkdir()
    plain = write_instance_module(directory, "kgpbench_fixture_plain", FIXTURE_ALPHA)
    searching = write_instance_module(directory, "kgpbench_fixture_search", SEARCHING)
    monkeypatch.syspath_prepend(str(directory))
    config = write_set_config(
        directory,
        {
            "plain": [plain],
            "search": [searching],
            "mixed": [plain, searching],
        },
    )
    monkeypatch.setenv(SET_CONFIG_ENV, str(config))
    return {"plain": plain, "search": searching}


def _counting_model() -> tuple[Any, list[int]]:
    calls: list[int] = []

    def outputs(input, tools, tool_choice, config):  # type: ignore[no-untyped-def]
        calls.append(1)
        return ModelOutput.from_content(
            model="mockllm/model", content="an answer", stop_reason="stop"
        )

    return get_model("mockllm/model", custom_outputs=outputs, memoize=False), calls


@pytest.mark.parametrize("set_name", ["search", "mixed"])
def test_the_registered_route_refuses_in_the_task_body_and_writes_no_log(
    search_sets: dict[str, str], tmp_path: Path, set_name: str
) -> None:
    """``inspect eval kgpbench/generation`` on `mockllm`: refused before any model call.

    The task body runs after the framework made the model active, so the read is
    there and the refusal leaves nothing on disk — no log, and so no model event
    in one. Under the mutation that removes the refusal the run still reaches no
    provider: the model is the counting mock.
    """
    model, calls = _counting_model()
    log_dir = tmp_path / "never-written"
    with pytest.raises(WebSearchUnavailable, match="declares web search"):
        inspect_eval(
            "kgpbench/generation",
            task_args={
                "mcp_url": MCP_URL,
                "set": set_name,
                "reasoning_effort": "xhigh",
                "max_output_tokens": 16_000,
            },
            model=model,
            display="none",
            log_dir=str(log_dir),
        )
    assert calls == []
    assert not log_dir.exists()


def test_the_registered_function_passed_to_eval_refuses_the_same_way(
    search_sets: dict[str, str], tmp_path: Path
) -> None:
    """``eval(generation, …)``: the framework calls the function, so the model is seen."""
    model, calls = _counting_model()
    log_dir = tmp_path / "never-written"
    with pytest.raises(WebSearchUnavailable):
        inspect_eval(
            generation,
            task_args={
                "mcp_url": MCP_URL,
                "set": "search",
                "reasoning_effort": "xhigh",
                "max_output_tokens": 16_000,
            },
            model=model,
            display="none",
            log_dir=str(log_dir),
        )
    assert calls == []
    assert not log_dir.exists()


def test_called_before_eval_the_task_body_leaves_the_refusal_to_the_solver(
    search_sets: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No active model and no ``INSPECT_EVAL_MODEL``: the search set still builds.

    The task body has no model to read, so it cannot refuse; the solver does,
    before its first model call (below). The plain set beside it builds exactly
    as before.
    """
    monkeypatch.delenv("INSPECT_EVAL_MODEL", raising=False)
    plain = generation(MCP_URL, reasoning_effort="xhigh", set="plain")
    searching = generation(MCP_URL, reasoning_effort="xhigh", set="search")
    assert plain.name == "kgpbench-generation-plain"
    assert searching.name == "kgpbench-generation-search"


def _instances(*members: Instance) -> InstanceSet:
    return InstanceSet(name="fixture", instances=list(members))


@pytest.mark.parametrize(
    "members", [(SEARCHING,), (FIXTURE_ALPHA, SEARCHING)], ids=["search", "mixed"]
)
def test_the_builder_route_refuses_in_the_solver_before_any_model_call(
    members: tuple[Instance, ...], tmp_path: Path
) -> None:
    """A `Task` built before `eval()` has no model to read, so the solver refuses.

    Every sample of the run errors — in the mixed set the one that declares no
    search too, because the refusal reads the set — and no sample carries a
    model event. No MCP session opens: the refusal precedes the connection.
    """
    model, calls = _counting_model()
    built = build_generation_task(
        _instances(*members), mcp_url=MCP_URL, solver_config=MOCK_CONFIG
    )
    log = inspect_eval(
        built, model=model, display="none", log_dir=str(tmp_path / "builder")
    )[0]

    assert log.status == "error"
    assert calls == []
    stored = read_log(log.location)
    samples = stored.samples or []
    assert len(samples) == len(members)
    for sample in samples:
        assert sample.error is not None
        assert "WebSearchUnavailable" in (sample.error.traceback or "")
        assert "declares web search" in sample.error.message
        assert not [event for event in sample.events if isinstance(event, ModelEvent)]


def test_a_solver_built_alone_refuses_on_the_sample_s_own_declaration(
    tmp_path: Path,
) -> None:
    """No set to read: a search sample is refused, a plain one beside it runs."""
    model, calls = _counting_model()
    task = Task(
        name="alone",
        dataset=build_dataset(_instances(FIXTURE_ALPHA, SEARCHING)),
        solver=research_agent([], MOCK_CONFIG),
    )
    log = inspect_eval(task, model=model, display="none", log_dir=str(tmp_path / "a"))[
        0
    ]
    by_id = {sample.id: sample for sample in read_log(log.location).samples or []}
    assert by_id[SEARCHING.id].error is not None
    assert by_id[FIXTURE_ALPHA.id].error is None
    assert calls == [1]


@pytest.mark.live_mcp
@pytest.mark.parametrize("route", ["registered", "builder"])
def test_the_refusal_is_silent_for_a_set_without_search(
    search_sets: dict[str, str], tmp_path: Path, route: str
) -> None:
    """The negative half on both routes: a plain set runs on `mockllm` to success."""
    model, calls = _counting_model()
    if route == "registered":
        task: Any = "kgpbench/generation"
        task_args = {
            "mcp_url": MCP_URL,
            "set": "plain",
            "reasoning_effort": "xhigh",
            "max_output_tokens": 16_000,
        }
    else:
        task = build_generation_task(
            resolve_set("plain"), mcp_url=MCP_URL, solver_config=MOCK_CONFIG
        )
        task_args = {}
    log = inspect_eval(
        task,
        task_args=task_args,
        model=model,
        display="none",
        log_dir=str(tmp_path / route),
    )[0]
    assert log.status == "success", log.error
    assert calls == [1]


# -- 5. the tools per sample --------------------------------------------------


def _recording_model() -> tuple[Any, dict[str, list[Any]]]:
    """A mock that records, per prompt, the tools it was offered."""
    offered: dict[str, list[Any]] = {}

    def outputs(input, tools, tool_choice, config):  # type: ignore[no-untyped-def]
        offered[input[0].text] = list(tools)
        return ModelOutput.from_content(
            model="mockllm/model", content="an answer", stop_reason="stop"
        )

    return get_model("mockllm/model", custom_outputs=outputs, memoize=False), offered


async def _local_tool(symbol: str) -> str:
    """Look up a symbol.

    Args:
        symbol: The symbol.
    """
    return symbol


LOCAL_TOOL = ToolDef(_local_tool, name="local_tool").as_tool()


@pytest.fixture
def refusal_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let `mockllm` through, so the tools a search sample receives can be observed.

    Without this, every test below would observe the refusal instead — which is
    the behaviour above. Patched where the solver looks it up.
    """
    monkeypatch.setattr("kgpbench.solver.assert_web_search_available", lambda _: None)


def test_a_mixed_set_offers_search_to_the_search_sample_only(
    refusal_disabled: None, tmp_path: Path
) -> None:
    model, offered = _recording_model()
    task = Task(
        name="mixed",
        dataset=MemoryDataset(
            build_dataset(_instances(FIXTURE_ALPHA, SEARCHING)).samples
        ),
        solver=research_agent([LOCAL_TOOL], MOCK_CONFIG, set_declares_web_search=True),
    )
    log = inspect_eval(task, model=model, display="none", log_dir=str(tmp_path / "m"))[
        0
    ]
    assert log.status == "success", log.error

    plain = [tool.name for tool in offered[FIXTURE_ALPHA.prompt]]
    searching = offered[SEARCHING.prompt]
    assert plain == ["local_tool"]
    assert [tool.name for tool in searching] == ["local_tool", "web_search"]

    options = dict(searching[-1].options or {})
    assert options.pop("__internal_tool_type__") == "web_search"
    assert set(options) == set(INTERNAL_SEARCH_PROVIDERS)
    assert options["anthropic"] == SEARCHING.web_search["anthropic"]  # type: ignore[index]


def test_a_sample_with_no_instance_key_declares_nothing(
    refusal_disabled: None, tmp_path: Path
) -> None:
    """The solver driven directly, as most tests drive it: no metadata, no search."""
    from inspect_ai.dataset import Sample

    model, offered = _recording_model()
    task = Task(
        name="bare",
        dataset=MemoryDataset([Sample(input="investigate", id="i1")]),
        solver=research_agent([LOCAL_TOOL], MOCK_CONFIG),
    )
    inspect_eval(task, model=model, display="none", log_dir=str(tmp_path / "b"))
    assert [tool.name for tool in offered["investigate"]] == ["local_tool"]


@pytest.mark.live_mcp
def test_the_builder_offers_the_onekgpd_tools_unchanged_and_search_after_them(
    refusal_disabled: None, tmp_path: Path
) -> None:
    """Against the live tool list: the plain sample's tools are what a plain set gets."""
    model, offered = _recording_model()
    built = build_generation_task(
        _instances(FIXTURE_ALPHA, SEARCHING), mcp_url=MCP_URL, solver_config=MOCK_CONFIG
    )
    log = inspect_eval(
        built, model=model, display="none", log_dir=str(tmp_path / "mixed")
    )[0]
    assert log.status == "success", log.error

    baseline_model, baseline = _recording_model()
    plain_only = build_generation_task(
        _instances(FIXTURE_ALPHA), mcp_url=MCP_URL, solver_config=MOCK_CONFIG
    )
    inspect_eval(
        plain_only,
        model=baseline_model,
        display="none",
        log_dir=str(tmp_path / "plain"),
    )

    today = [tool.name for tool in baseline[FIXTURE_ALPHA.prompt]]
    assert len(today) > 1 and "web_search" not in today
    assert [tool.name for tool in offered[FIXTURE_ALPHA.prompt]] == today
    assert [tool.name for tool in offered[SEARCHING.prompt]] == [*today, "web_search"]
