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
    # These four MOVE to antennaknobs' tests/ (antennaknobs#1299), where both
    # packages are installed and CI runs against the recorded pointer. Until
    # that lands they stay here and stay guarded; the entry is a pointer to the
    # decision, not a promise to bank them.
    #
    # WHOLE-CATALOG CENSUSES. These walk `antennaknobs.designs` with pkgutil
    # and assert a property over EVERY design ("exactly one deck trips the
    # fragmentation predicate", "repeats are not the common case"). The
    # catalog IS the subject, so there is no fixture that could stand in --
    # banking one deck, or thirteen, would change what they measure. They are
    # NOT the same category as a test that merely needed a geometry, and the
    # #988 fix does not apply to them.
    "test_arrayblock_no_repeats_972.py": "whole-catalog census; moves to AK, antennaknobs#1299",
    "test_fragmentation_fallback_972.py": "whole-catalog census; moves to AK, antennaknobs#1299",
    "test_deck_nec2_corpus.py": "the AK importer IS the reference; moves to AK, antennaknobs#1299",
    "test_deck_nec2_xnec2c_corpus.py": "the AK importer IS the reference; moves to AK, antennaknobs#1299",
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
