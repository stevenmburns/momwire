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

import ast
import pathlib

TESTS = pathlib.Path(__file__).parent


# Matches the bare package AND submodule paths: `antennaknobs.nec_import`
# guards `test_deck_nec2_corpus.py` and a pattern anchored on a closing quote
# right after the package name misses it entirely.
def _guards_on_antennaknobs(path: pathlib.Path) -> bool:
    """Does this file CALL `importorskip` on antennaknobs?

    Parsed, not grepped, and that is a correction rather than a refinement.
    The regex this replaced matched the pattern wherever it appeared --
    including in prose ABOUT the pattern. It went red on
    `test_arrayblock_no_repeats_972.py` and `test_fragmentation_fallback_972.py`
    when antennaknobs#1299 removed their guarded tests and left a comment
    explaining where the tests went: the comment quoted the guard, so the
    tripwire read the explanation of a fix as the defect.

    That is the fifth time this repo has hit the shape -- the #936 AST gate,
    this module's own `SELF` exclusion, and two tripwires written for
    momwire#999 all matched their own writing about the thing they check. A
    source tripwire that greps cannot tell code from a description of code;
    one that parses does not have to be told.

    The `SELF` exclusion is gone with it: this module quotes the pattern
    constantly and never calls it, so parsing exempts it for the right reason
    instead of by name.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:  # pragma: no cover - a test file that will not parse
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "importorskip" or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value == "antennaknobs" or arg.value.startswith("antennaknobs."):
                return True
    return False


# Allowlist: file -> why it may still skip. Every entry is a promise that the
# behaviour it covers is gated somewhere that CI can see.
ALLOWED = {
    # The drift check for the bank the verdict gate uses. The protection is
    # `test_g984_the_catalog_verdicts_are_what_the_threshold_promises`, which
    # reads tests/fixtures/catalog_geometries.json and needs nothing installed.
    "test_somm_aca_stagnation_973.py": "drift check only; the gate is fixture-driven",
}


def test_g988_no_new_file_hides_behind_importorskip_antennaknobs():
    offenders = sorted(
        p.name
        for p in TESTS.glob("test_*.py")
        if _guards_on_antennaknobs(p) and p.name not in ALLOWED
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
        if (TESTS / n).exists() and not _guards_on_antennaknobs(TESTS / n)
    )
    assert not fixed, f"these no longer use the guard and should leave ALLOWED: {fixed}"
