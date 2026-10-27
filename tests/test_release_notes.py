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
    ``releases/latest/download/momwire-windows.zip`` — the link the site
    hands EZNEC, SimNEC and 4nec2 users — resolves to (``momwire-eznec-
    windows.zip`` until momwire#1583).
    """
    text = (WORKFLOWS / ATTACHES_ONLY).read_text()
    assert "softprops/action-gh-release" in text
    assert "files: momwire-windows.zip" in text
    assert "files: momwire-windows-win7.zip" in text


def test_the_linux_executable_attaches_to_the_release():
    """momwire#1512's Linux artifact rides the same release, as an attach
    only: built and gated on every run, attached on a pushed tag."""
    text = (WORKFLOWS / "standalone.yml").read_text()
    assert "softprops/action-gh-release" in text
    assert "files: momwire-nec5-linux-x86_64.tar.gz" in text
    assert 'tags: ["v*"]' in text


def test_the_windows_and_macos_executables_attach_to_the_release():
    """The preview assets (0.79.0): the Windows zip and the macOS tarball
    ride the same release as attaches only, beside the Linux tarball and
    under the names README.md gives them."""
    text = (WORKFLOWS / "standalone.yml").read_text()
    assert "files: momwire-nec5-windows-x64.zip" in text
    assert "files: momwire-nec5-macos-arm64.tar.gz" in text
    readme = (WORKFLOWS.parent.parent / "README.md").read_text(encoding="utf-8")
    for name in ("momwire-nec5-windows-x64.zip", "momwire-nec5-macos-arm64.tar.gz"):
        assert name in readme


def test_the_nec42_sinusoidal_exe_ships_in_the_linux_tarball_as_a_preview():
    """0.83.0 (momwire#1607): the NEC-4.2 slot's sinusoidal no-Python build
    rides the Linux tarball, gated before a tag attaches it, and every note
    that lists it says it is a preview and what it still refuses."""
    root = WORKFLOWS.parent.parent
    script = (root / "standalone" / "build_release_linux.sh").read_text()
    assert "momwire-nec42-sinusoidal-linux sin42" in script
    assert (root / "standalone" / "release" / "momwire-nec42-sinusoidal-linux").exists()
    workflow = (WORKFLOWS / "standalone.yml").read_text()
    assert "needs: [linux-artifact, linux, linux-nec42]" in workflow
    readme_txt = (root / "standalone" / "release" / "README.txt").read_text()
    assert "PREVIEW: momwire-nec42-sinusoidal-linux" in readme_txt
    for refusal in ("B6",):
        assert refusal in readme_txt, refusal
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "NEC-4.2 sinusoidal engine on Linux, without Python (preview)" in readme
