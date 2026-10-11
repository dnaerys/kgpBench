# Copyright 2026 Dnaerys Pty Ltd
# SPDX-License-Identifier: Apache-2.0

"""Version identifiers the harness owns.

Two of these are provenance inputs in their own right and must not be conflated.
The framework version says what trajectory objects are *available*; the renderer
version says how the harness *turns them into the text a judge reads*. A
framework bump changes the former, a rendering revision changes the latter, and
a rendering revision changes verdicts with the model, the instance, the
catalogue and the judge all held fixed.

The framework axis is
:func:`~kgpbench.provenance.framework_version` — the installed
`inspect_ai` release, read from distribution metadata. It is not a commit and
there is no fork: `inspect_ai` is an ordinary pinned dependency.
"""

from __future__ import annotations

__all__ = ["HARNESS_VERSION", "RENDERER_VERSION"]

HARNESS_VERSION = "0.1.1"

RENDERER_VERSION = "r3"
"""Identifier for the trajectory rendering format.

It is here from the first commit because retrofitting a provenance field makes
everything generated before the retrofit unusable for cross-round comparison.

``r0`` → ``r1`` at the v7/v8 cleanup, while the rendering was still a stub: the
completeness section lost its ``continuations`` field with the continuation
policy, and the reasoning section stopped labelling every block
``redacted=True`` — which is what `.redacted` reads on an ordinary block (§0 v8).

``r1`` → ``r2`` when the level-5 renderer landed. Three changes, any one of which
would move every digest: the ``completeness`` section is withdrawn from what
judges read (§6.4, closed at v12), the document is chronological and fenced
rather than one block per channel, and every fenced body goes through
`neutralize_structural_delimiters` (§13.4). No corpus exists, so nothing is
invalidated.

``r2`` → ``r3`` renders the tool calls a model's provider ran during
generation, and changes nothing else (§6.4, §0 v121). An assistant message
carrying a `ContentToolUse` or a citation renders in content order: each
provider-run call as an arguments, a result and an optional error block in the
``tool_events`` channel, the answer's text pieces joined as written rather than
with the newline ``ChatMessageBase.text`` inserts, and its citations in one
cited-sources block after the answer, with each field the vendor made opaque
replaced by a marker. Every other message renders through the ``r2`` code, so a
trajectory carrying neither renders to its ``r2`` bytes — measured over every
trajectory in the four published generation logs — and its digest does not
move. No published trajectory carries either.

Before anything was published under ``r3``, Anthropic's list gained two entries
and the version did not move (§0 v123): ``encrypted_stdout``, the code
container's encrypted output, renders as the withheld marker, and a fetched
document whose source is base64 bytes renders its data as
``[not rendered: <media type>, base64, N characters]``. Each moves only the
digest of a trajectory that carries it, and none has been recorded.

Bump it for **any** change to what a judge reads — block order, labels,
tool-result delimiters, how reasoning is labelled. Not for changes to the
harness that do not alter the rendered text.
"""
