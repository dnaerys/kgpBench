"""Provider-run tool shapes no paid run has produced, built from the SDK's types.

Generated, then ``ruff format``-ted — do not edit by hand. ``make_provider_tool_fixtures.py``
validates each wire block against the SDK's own model (``anthropic==1.12.1``),
builds the response as the client does and passes it through Inspect's own Anthropic
parser, ``model_output_from_message`` (``inspect_ai==0.3.252``,
``_providers/anthropic.py:3236``). Each
list is one assistant message's ``content`` as ``model_dump(mode="json",
exclude_none=True)`` wrote it, which is also how a log stores it. No request was
sent. The test environment carries no SDK, which is why these are literals.
"""

# ruff: noqa: E501

from __future__ import annotations

from typing import Any

CODE_ID = "srvtoolu_01FIXTURECODEEXECUTION0001"

FETCH_ID = "srvtoolu_01FIXTUREWEBFETCH000000001"

FETCH_ERROR_ID = "srvtoolu_01FIXTUREWEBFETCHERROR0001"

FETCH_PDF_ID = "srvtoolu_01FIXTUREWEBFETCHPDF00000001"

ENCRYPTED_CODE_ID = "srvtoolu_01FIXTUREENCRYPTEDCODE0001"

SEARCH_ERROR_ID = "srvtoolu_01FIXTURESEARCHERROR000001"

SEARCH_ID = "srvtoolu_01FIXTURESEARCH000000000001"

FETCH_AND_CODE: list[dict[str, Any]] = [
    {"type": "text", "text": "I will filter the results with code."},
    {
        "type": "tool_use",
        "tool_type": "code_execution",
        "id": "srvtoolu_01FIXTURECODEEXECUTION0001",
        "name": "bash_code_execution",
        "arguments": '{\n  "command": "python3 -c \\"print(\'rs0000001 count: 7\')\\""\n}',
        "result": "{\n"
        '  "content": {\n'
        '    "content": [],\n'
        '    "return_code": 0,\n'
        '    "stderr": "",\n'
        '    "stdout": "rs0000001 count: 7\\n",\n'
        '    "type": "bash_code_execution_result"\n'
        "  },\n"
        '  "tool_use_id": "srvtoolu_01FIXTURECODEEXECUTION0001",\n'
        '  "type": "bash_code_execution_tool_result"\n'
        "}",
    },
    {
        "type": "tool_use",
        "tool_type": "web_search",
        "id": "srvtoolu_01FIXTUREWEBFETCH000000001",
        "name": "web_fetch",
        "arguments": '{\n  "url": "https://example.org/variant/rs0000001"\n}',
        "result": "{\n"
        '  "caller": {\n'
        '    "tool_id": "srvtoolu_01FIXTURECODEEXECUTION0001",\n'
        '    "type": "code_execution_20260120"\n'
        "  },\n"
        '  "content": {\n'
        '    "content": {\n'
        '      "source": {\n'
        '        "data": "rs0000001 is a fixture variant.\\n\\nSeven carriers.",\n'
        '        "media_type": "text/plain",\n'
        '        "type": "text"\n'
        "      },\n"
        '      "title": "rs0000001 — a fixture page",\n'
        '      "type": "document"\n'
        "    },\n"
        '    "retrieved_at": "2026-10-09T08:00:00Z",\n'
        '    "type": "web_fetch_result",\n'
        '    "url": "https://example.org/variant/rs0000001"\n'
        "  },\n"
        '  "tool_use_id": "srvtoolu_01FIXTUREWEBFETCH000000001",\n'
        '  "type": "web_fetch_tool_result"\n'
        "}",
    },
    {"type": "text", "text": "The page reports seven carriers."},
]

ENCRYPTED_CODE: list[dict[str, Any]] = [
    {
        "type": "tool_use",
        "tool_type": "code_execution",
        "id": "srvtoolu_01FIXTUREENCRYPTEDCODE0001",
        "name": "code_execution",
        "arguments": '{\n  "code": "print(len(results))"\n}',
        "result": "{\n"
        '  "content": {\n'
        '    "content": [],\n'
        '    "encrypted_stdout": '
        '"SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS",\n'
        '    "return_code": 0,\n'
        '    "stderr": "",\n'
        '    "type": "encrypted_code_execution_result"\n'
        "  },\n"
        '  "tool_use_id": "srvtoolu_01FIXTUREENCRYPTEDCODE0001",\n'
        '  "type": "code_execution_tool_result"\n'
        "}",
    },
    {"type": "text", "text": "Done."},
]

FETCHED_PDF: list[dict[str, Any]] = [
    {
        "type": "tool_use",
        "tool_type": "web_search",
        "id": "srvtoolu_01FIXTUREWEBFETCHPDF00000001",
        "name": "web_fetch",
        "arguments": '{\n  "url": "https://example.org/variant.pdf"\n}',
        "result": "{\n"
        '  "caller": {\n'
        '    "type": "direct"\n'
        "  },\n"
        '  "content": {\n'
        '    "content": {\n'
        '      "source": {\n'
        '        "data": '
        '"JVBERi0xLjcKcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4gcnMwMDAwMDAxIGZpeHR1cmUgcGFnZS4g",\n'
        '        "media_type": "application/pdf",\n'
        '        "type": "base64"\n'
        "      },\n"
        '      "title": "rs0000001 — a fixture PDF",\n'
        '      "type": "document"\n'
        "    },\n"
        '    "retrieved_at": "2026-10-09T08:00:00Z",\n'
        '    "type": "web_fetch_result",\n'
        '    "url": "https://example.org/variant.pdf"\n'
        "  },\n"
        '  "tool_use_id": "srvtoolu_01FIXTUREWEBFETCHPDF00000001",\n'
        '  "type": "web_fetch_tool_result"\n'
        "}",
    },
    {"type": "text", "text": "The PDF reports seven carriers."},
]

ERRORS: list[dict[str, Any]] = [
    {
        "type": "tool_use",
        "tool_type": "web_search",
        "id": "srvtoolu_01FIXTURESEARCHERROR000001",
        "name": "web_search",
        "arguments": '{\n  "query": "rs0000001 carriers"\n}',
        "result": '{"error_code":"max_uses_exceeded","type":"web_search_tool_result_error"}',
        "error": "max_uses_exceeded",
    },
    {
        "type": "tool_use",
        "tool_type": "web_search",
        "id": "srvtoolu_01FIXTUREWEBFETCHERROR0001",
        "name": "web_fetch",
        "arguments": '{\n  "url": "https://example.org/missing"\n}',
        "result": "{\n"
        '  "caller": {\n'
        '    "type": "direct"\n'
        "  },\n"
        '  "content": {\n'
        '    "error_code": "url_not_accessible",\n'
        '    "type": "web_fetch_tool_result_error"\n'
        "  },\n"
        '  "tool_use_id": "srvtoolu_01FIXTUREWEBFETCHERROR0001",\n'
        '  "type": "web_fetch_tool_result"\n'
        "}",
    },
    {"type": "text", "text": "Neither search nor fetch returned."},
]

CITED: list[dict[str, Any]] = [
    {
        "type": "tool_use",
        "tool_type": "web_search",
        "id": "srvtoolu_01FIXTURESEARCH000000000001",
        "name": "web_search",
        "arguments": '{\n  "query": "rs0000001"\n}',
        "result": '[{"encrypted_content":"EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE","page_age":"3 '
        'days ago","title":"rs0000001 — a fixture '
        'page","type":"web_search_result","url":"https://example.org/variant/rs0000001"}]',
    },
    {"type": "text", "text": "The variant is reported in "},
    {
        "type": "text",
        "text": "seven carriers",
        "citations": [
            {
                "cited_text": "Seven carriers.",
                "title": "rs0000001 — a fixture page",
                "internal": {
                    "encrypted_index": "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
                },
                "type": "url",
                "url": "https://example.org/variant/rs0000001",
            }
        ],
    },
    {"type": "text", "text": ", and "},
    {
        "type": "text",
        "text": "no homozygote",
        "citations": [
            {
                "cited_text": "No homozygote.",
                "internal": {
                    "encrypted_index": "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
                },
                "type": "url",
                "url": "https://example.org/variant/rs0000001",
            }
        ],
    },
    {"type": "text", "text": "."},
]
