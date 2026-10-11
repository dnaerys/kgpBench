"""OneKGPd's dataset id, answered locally for every test that may not leave the machine.

`build_generation_task` reads ``datasetId`` from the server before it builds
anything (`kgpbench.dataset_id`), so every test that builds a generation task,
including the many that never run it, would otherwise reach the network. For a
test without the `live_mcp` marker, the fixture here replaces the one function
that makes the call, and nothing around it, with an answer carrying
:data:`DATASET_ID`. A marked test reads the real server.

**The fixture instances declare the live id, not a synthetic one**, so the same
fixtures serve both kinds of test: an unmarked test reads this answer, a marked
one reads the server, and both see the id the instances declare. When the server
moves to a new id, the marked tests refuse until the fixtures declare it, which
is the refusal working.

**Imported into a conftest, where being autouse makes it cover every test in that
tree**, as `network_guard` is. A test whose subject is the read itself sets the
function to what it needs after this fixture has run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from harness_fixtures import DATASET_ID
from network_guard import LIVE_MCP_MARKER

import kgpbench.dataset_id

__all__ = ["DATASET_INFO", "answer_the_dataset_id_locally"]

DATASET_INFO = json.dumps(
    {
        "variantsTotal": 138044723,
        "samplesTotal": 3202,
        "samplesMaleCount": 1598,
        "samplesFemaleCount": 1604,
        "datasetId": DATASET_ID,
    }
)
"""``getDatasetInfo``'s text, as the live server returned it on 2026-10-10."""


@pytest.fixture(autouse=True)
def answer_the_dataset_id_locally(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Answer ``getDatasetInfo`` locally, unless the test may reach the server."""
    if request.node.get_closest_marker(LIVE_MCP_MARKER) is None:
        monkeypatch.setattr(
            kgpbench.dataset_id,
            "_get_dataset_info",
            lambda mcp_url: (False, DATASET_INFO),
        )
    yield
