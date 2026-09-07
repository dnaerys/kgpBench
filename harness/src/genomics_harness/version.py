"""Version identifiers the harness owns.

Two of these are provenance inputs in their own right and must not be conflated.
The framework version says what trajectory objects are *available*; the renderer
version says how the harness *turns them into the text a judge reads*. A
framework bump changes the former, a rendering revision changes the latter, and
a rendering revision changes verdicts with the model, the instance, the
catalogue and the judge all held fixed.

The framework axis is
:func:`~genomics_harness.provenance.framework_version` — the installed
`inspect_ai` release, read from distribution metadata. It is not a commit and
there is no fork: `inspect_ai` is an ordinary pinned dependency.
"""

from __future__ import annotations

__all__ = ["HARNESS_VERSION", "RENDERER_VERSION"]

HARNESS_VERSION = "0.1.0"

RENDERER_VERSION = "r2"
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

Bump it for **any** change to what a judge reads — block order, labels,
tool-result delimiters, how reasoning is labelled. Not for changes to the
harness that do not alter the rendered text.
"""
