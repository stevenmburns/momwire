"""Exactly one workflow may ask GitHub to generate the release body (#662).

Two of them did, and the releases came out with the changelog twice —
v0.37.0 through v0.40.0, every one.  The theory in the comment that licensed
it was that ``generate_release_notes`` is a CREATE-time setting, so the two
jobs racing to publish a tag both had to ask or the loser would leave the
release bodyless.  It is not.  ``softprops/action-gh-release`` passes the
release's EXISTING body back when it updates, and GitHub pre-pends that body
to the notes it generates, so whichever job ran second appended a second copy.

The fix is an asymmetry — ``wheels.yml`` owns the body, ``eznec-dropin.yml``
owns the zip — and an asymmetry is exactly the kind of thing that gets
"tidied" back into symmetry by someone reading the two jobs side by side.
Hence this file.

It greps rather than parsing YAML: ``pyyaml`` is not a declared test dep, and
the claim is about a literal line either being present or not.  A workflow
that spelt the input some other way would slip past, which is a real limit —
but the bug being gated is a copy-paste of this exact line into a second job.
"""

from __future__ import annotations

import pathlib

WORKFLOWS = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows"

# The one job allowed to ask, and the one that must not.  Named rather than
# counted so the failure says WHICH file grew the second copy.
OWNER = "wheels.yml"
ATTACHES_ONLY = "eznec-dropin.yml"


def _asking():
    return {
        p.name
        for p in sorted(WORKFLOWS.glob("*.yml"))
        if "generate_release_notes: true" in p.read_text()
    }


def test_exactly_one_workflow_generates_the_release_body():
    """The whole of #662, as one set comparison."""
    assert _asking() == {OWNER}, (
        "the release changelog is appended once per asker, so a second "
        f"workflow asking ships it twice; expected only {OWNER}"
    )


def test_the_drop_in_still_attaches_to_the_release():
    """The fix is "don't ask for notes", not "don't publish the zip".

    Deleting the release job outright would also satisfy the test above, and
    would silently stop shipping the Windows drop-in that
    ``releases/latest/download/momwire-eznec-windows.zip`` — the link the site
    hands EZNEC users — resolves to.
    """
    text = (WORKFLOWS / ATTACHES_ONLY).read_text()
    assert "softprops/action-gh-release" in text
    assert "files: momwire-eznec-windows.zip" in text
