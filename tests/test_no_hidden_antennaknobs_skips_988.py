"""#988: a test guarded by `importorskip("antennaknobs")` runs in NO CI lane.

No momwire workflow installs antennaknobs (it appears in `.github/workflows/`
only inside comments), and antennaknobs' CI does not run momwire's tests. So
that guard does not mean "skipped on some machines" — it means the test is
never executed by any gate, anywhere.

That is not hypothetical. When this tripwire was written, five files carried
the guard and THREE of their cases were RED: #971's step-function gates, which
#979 had fixed and therefore invalidated, merged and stayed red through a
nine-lane dispatch because no lane could see them.

The lesson, for whoever trips this: "the dispatched lanes were green" is a
statement about the LANES, not about the tests. Count the skips in the lane
that matters.

A test may stay on the guard only if it is a DRIFT CHECK — comparing a banked
fixture against the live antennaknobs catalog — because the protection then
lives in the fixture-driven test and this one only reports that the bank has
aged. Anything that gates momwire's own behaviour must run without
antennaknobs installed.
"""

import pathlib
import re

TESTS = pathlib.Path(__file__).parent
SELF = pathlib.Path(__file__).name
# Matches the bare package AND submodule paths: `antennaknobs.nec_import`
# guards `test_deck_nec2_corpus.py` and a pattern anchored on a closing quote
# right after the package name misses it entirely.
GUARD = re.compile(r'importorskip\(\s*["\']antennaknobs[.\'"]')

# Allowlist: file -> why it may still skip. Every entry is a promise that the
# behaviour it covers is gated somewhere that CI can see.
ALLOWED = {
    # The drift check for the bank the verdict gate uses. The protection is
    # `test_g984_the_catalog_verdicts_are_what_the_threshold_promises`, which
    # reads tests/fixtures/catalog_geometries.json and needs nothing installed.
    "test_somm_aca_stagnation_973.py": "drift check only; the gate is fixture-driven",
    # Still to bank (#988 items 1-3). Counted here so the debt is visible and
    # cannot grow silently rather than being discovered again by accident.
    "test_aca_tol_default_971.py": "#988 follow-up: geometries not yet banked",
    "test_arrayblock_no_repeats_972.py": "#988 follow-up: geometries not yet banked",
    "test_fragmentation_fallback_972.py": "#988 follow-up: geometries not yet banked",
    # These two ARE the comparison against antennaknobs' importer -- the
    # reference is the other codebase, so there is no fixture that could stand
    # in for it. Found by this tripwire, not by the hand grep that opened #988,
    # which missed submodule guards like importorskip("antennaknobs.nec_import").
    "test_deck_nec2_corpus.py": "the antennaknobs importer IS the reference",
    "test_deck_nec2_xnec2c_corpus.py": "the antennaknobs importer IS the reference",
}


def test_g988_no_new_file_hides_behind_importorskip_antennaknobs():
    offenders = sorted(
        p.name
        for p in TESTS.glob("test_*.py")
        # `p.name != SELF`: this file quotes the pattern it looks for, so it
        # matches itself -- the same self-match an AST gate hit in #936.
        if p.name != SELF
        and GUARD.search(p.read_text(errors="replace"))
        and p.name not in ALLOWED
    )
    assert not offenders, (
        "these files guard tests behind importorskip('antennaknobs'), which "
        "means no CI lane runs them (#988): "
        + ", ".join(offenders)
        + ". Bank the geometry as a fixture under tests/fixtures/ and drive "
        "the test from that, or add the file to ALLOWED with the reason its "
        "behaviour is gated elsewhere."
    )


def test_g988_the_allowlist_does_not_name_files_that_are_gone():
    """An allowlist entry for a deleted or fixed file is a licence nobody is
    using -- and would silently re-permit the shape if the name came back."""
    stale = sorted(n for n in ALLOWED if not (TESTS / n).exists())
    assert not stale, f"ALLOWED names files that no longer exist: {stale}"

    fixed = sorted(
        n
        for n in ALLOWED
        if (TESTS / n).exists()
        and not GUARD.search((TESTS / n).read_text(errors="replace"))
    )
    assert not fixed, f"these no longer use the guard and should leave ALLOWED: {fixed}"
