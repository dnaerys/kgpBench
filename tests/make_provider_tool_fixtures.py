"""Write `provider_tool_fixtures.py`: provider-run shapes from the Anthropic SDK's types.

Not a test module, and not collected. The test environment carries no
`anthropic` SDK, so the shapes no paid run has produced — a page fetch, code
execution, a fetched PDF, an error, consecutive cited text pieces — are generated here, in an
environment that has it, and committed as literals. From this directory:

    <venv-with-anthropic>/bin/python -I make_provider_tool_fixtures.py provider_tool_fixtures.py
    ruff format provider_tool_fixtures.py

Every block below is written in its wire form and validated strictly against
the SDK's own model for its type, so a field the SDK does not define is an error
here rather than a guess. The message is then built the way the client builds a
response — `construct_type` into the non-beta `anthropic.types.Message`, which
is what `client.messages.create` returns to Inspect — and handed to Inspect's
`model_output_from_message` (`_providers/anthropic.py:3236`), so the
`ContentToolUse` / `ContentText` items are whatever the real path makes of them. No request is sent and no key is read. The output is a Python
module of plain literals, written for `harness/tests`, which cannot import the SDK.
"""

from __future__ import annotations

import asyncio
import base64
import pprint
import sys
from importlib.metadata import version

from anthropic._models import construct_type_unchecked
from anthropic.types import (
    BashCodeExecutionToolResultBlock,
    CodeExecutionToolResultBlock,
    Message,
    ServerToolUseBlock,
    TextBlock,
    WebFetchToolResultBlock,
    WebSearchToolResultBlock,
)
from inspect_ai.model._providers.anthropic import model_output_from_message

DIRECT = {"type": "direct"}
# ids in the provider's shape. Distinctive so a test can assert none reaches text.
CODE_ID = "srvtoolu_01FIXTURECODEEXECUTION0001"
FETCH_ID = "srvtoolu_01FIXTUREWEBFETCH000000001"
FETCH_ERROR_ID = "srvtoolu_01FIXTUREWEBFETCHERROR0001"
FETCH_PDF_ID = "srvtoolu_01FIXTUREWEBFETCHPDF00000001"
ENCRYPTED_CODE_ID = "srvtoolu_01FIXTUREENCRYPTEDCODE0001"
SEARCH_ERROR_ID = "srvtoolu_01FIXTURESEARCHERROR000001"
SEARCH_ID = "srvtoolu_01FIXTURESEARCH000000000001"

ENCRYPTED_PAGE = "E" * 1234
ENCRYPTED_INDEX = "I" * 200
ENCRYPTED_STDOUT = "S" * 321
# a fetched PDF's bytes, base64 as the wire carries them; small, but the shape of
# a real one, header and all
PDF_DATA = base64.b64encode(b"%PDF-1.7\n" + b"rs0000001 fixture page. " * 40).decode()

# each wire block is validated strictly against the SDK's own model for its type
STRICT = {
    "text": TextBlock,
    "server_tool_use": ServerToolUseBlock,
    "web_search_tool_result": WebSearchToolResultBlock,
    "web_fetch_tool_result": WebFetchToolResultBlock,
    "code_execution_tool_result": CodeExecutionToolResultBlock,
    "bash_code_execution_tool_result": BashCodeExecutionToolResultBlock,
}


def text(
    value: str, citations: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {"type": "text", "text": value, "citations": citations}


def use(
    id: str, name: str, input: dict[str, object], caller: dict[str, object] = DIRECT
) -> dict[str, object]:
    return {
        "type": "server_tool_use",
        "id": id,
        "name": name,
        "input": input,
        "caller": caller,
    }


def shape(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    """The wire blocks, as `client.messages.create` hands them to Inspect."""
    for block in blocks:
        STRICT[str(block["type"])].model_validate(block)
    message = construct_type_unchecked(
        value={
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": blocks,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        type_=Message,
    )
    output, _ = asyncio.run(
        model_output_from_message(None, "claude-sonnet-5", message, tools=[])
    )
    assert not output.message.tool_calls, output.message.tool_calls
    content = output.message.content
    assert not isinstance(content, str)
    return [item.model_dump(mode="json", exclude_none=True) for item in content]


BY_CODE = {"type": "code_execution_20260120", "tool_id": CODE_ID}

# 1. a page fetch issued by code, and the code that issued it: bash in the
# container, then a fetch whose `caller` names the code call
FETCH_AND_CODE = shape(
    [
        text("I will filter the results with code."),
        use(
            CODE_ID,
            "bash_code_execution",
            {"command": "python3 -c \"print('rs0000001 count: 7')\""},
        ),
        {
            "type": "bash_code_execution_tool_result",
            "tool_use_id": CODE_ID,
            "content": {
                "type": "bash_code_execution_result",
                "stdout": "rs0000001 count: 7\n",
                "stderr": "",
                "return_code": 0,
                "content": [],
            },
        },
        use(
            FETCH_ID,
            "web_fetch",
            {"url": "https://example.org/variant/rs0000001"},
            BY_CODE,
        ),
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": FETCH_ID,
            "caller": BY_CODE,
            "content": {
                "type": "web_fetch_result",
                "url": "https://example.org/variant/rs0000001",
                "retrieved_at": "2026-10-09T08:00:00Z",
                "content": {
                    "type": "document",
                    "title": "rs0000001 — a fixture page",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "rs0000001 is a fixture variant.\n\nSeven carriers.",
                    },
                },
            },
        },
        text("The page reports seven carriers."),
    ]
)

# 2. code execution whose stdout the provider encrypted — the SDK's
# `encrypted_code_execution_result`, "for PFC + web_search results"
ENCRYPTED_CODE = shape(
    [
        use(ENCRYPTED_CODE_ID, "code_execution", {"code": "print(len(results))"}),
        {
            "type": "code_execution_tool_result",
            "tool_use_id": ENCRYPTED_CODE_ID,
            "content": {
                "type": "encrypted_code_execution_result",
                "encrypted_stdout": ENCRYPTED_STDOUT,
                "stderr": "",
                "return_code": 0,
                "content": [],
            },
        },
        text("Done."),
    ]
)

# 3. a page fetch that returned a PDF: the document's source is the SDK's
# `Base64PDFSource`, its bytes base64 under a media type
FETCHED_PDF = shape(
    [
        use(FETCH_PDF_ID, "web_fetch", {"url": "https://example.org/variant.pdf"}),
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": FETCH_PDF_ID,
            "caller": DIRECT,
            "content": {
                "type": "web_fetch_result",
                "url": "https://example.org/variant.pdf",
                "retrieved_at": "2026-10-09T08:00:00Z",
                "content": {
                    "type": "document",
                    "title": "rs0000001 — a fixture PDF",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": PDF_DATA,
                    },
                },
            },
        },
        text("The PDF reports seven carriers."),
    ]
)

# 4. a search that errored, and a fetch that errored
ERRORS = shape(
    [
        use(SEARCH_ERROR_ID, "web_search", {"query": "rs0000001 carriers"}),
        {
            "type": "web_search_tool_result",
            "tool_use_id": SEARCH_ERROR_ID,
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": "max_uses_exceeded",
            },
        },
        use(FETCH_ERROR_ID, "web_fetch", {"url": "https://example.org/missing"}),
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": FETCH_ERROR_ID,
            "caller": DIRECT,
            "content": {
                "type": "web_fetch_tool_result_error",
                "error_code": "url_not_accessible",
            },
        },
        text("Neither search nor fetch returned."),
    ]
)

CITATION = {
    "type": "web_search_result_location",
    "url": "https://example.org/variant/rs0000001",
    "title": "rs0000001 — a fixture page",
    "encrypted_index": ENCRYPTED_INDEX,
}

# 5. consecutive text pieces with citations, after one successful search
CITED = shape(
    [
        use(SEARCH_ID, "web_search", {"query": "rs0000001"}),
        {
            "type": "web_search_tool_result",
            "tool_use_id": SEARCH_ID,
            "content": [
                {
                    "type": "web_search_result",
                    "title": "rs0000001 — a fixture page",
                    "url": "https://example.org/variant/rs0000001",
                    "encrypted_content": ENCRYPTED_PAGE,
                    "page_age": "3 days ago",
                }
            ],
        },
        text("The variant is reported in "),
        text("seven carriers", [{**CITATION, "cited_text": "Seven carriers."}]),
        text(", and "),
        text(
            "no homozygote",
            [{**CITATION, "title": None, "cited_text": "No homozygote."}],
        ),
        text("."),
    ]
)

HEADER = f'''"""Provider-run tool shapes no paid run has produced, built from the SDK's types.

Generated, then ``ruff format``-ted — do not edit by hand. ``make_provider_tool_fixtures.py``
validates each wire block against the SDK's own model (``anthropic=={version("anthropic")}``),
builds the response as the client does and passes it through Inspect's own Anthropic
parser, ``model_output_from_message`` (``inspect_ai=={version("inspect_ai")}``,
``_providers/anthropic.py:3236``). Each
list is one assistant message's ``content`` as ``model_dump(mode="json",
exclude_none=True)`` wrote it, which is also how a log stores it. No request was
sent. The test environment carries no SDK, which is why these are literals.
"""

# ruff: noqa: E501

from __future__ import annotations

from typing import Any
'''

with open(sys.argv[1], "w", encoding="utf-8") as out:
    out.write(HEADER)
    for name, value in [
        ("CODE_ID", CODE_ID),
        ("FETCH_ID", FETCH_ID),
        ("FETCH_ERROR_ID", FETCH_ERROR_ID),
        ("FETCH_PDF_ID", FETCH_PDF_ID),
        ("ENCRYPTED_CODE_ID", ENCRYPTED_CODE_ID),
        ("SEARCH_ERROR_ID", SEARCH_ERROR_ID),
        ("SEARCH_ID", SEARCH_ID),
    ]:
        out.write(f"\n{name} = {value!r}\n")
    for name, value in [
        ("FETCH_AND_CODE", FETCH_AND_CODE),
        ("ENCRYPTED_CODE", ENCRYPTED_CODE),
        ("FETCHED_PDF", FETCHED_PDF),
        ("ERRORS", ERRORS),
        ("CITED", CITED),
    ]:
        out.write(
            f"\n{name}: list[dict[str, Any]] = {pprint.pformat(value, width=88, sort_dicts=False)}\n"
        )
print("written", sys.argv[1])
