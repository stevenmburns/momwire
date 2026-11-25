"""Line 2's order of sacrifice when it overflows 79 columns (momwire#2158).

The basis name says which engine answered, so it is the last field to give
way. The pragma note's names go first, then the note, then the rename note,
and only then the basis. The overlay marker and the feature stamp are never
cut. ``compose_stamp`` (Python) and ``nec5_engine_stamp`` (the C++ writer's
``engine_stamp``, bound narrowly for this) take the same steps, so the
executables and the Python route print the same line 2.
"""

from __future__ import annotations

import pytest

from momwire import _accel
from momwire.dropin.nec5._printout import compose_stamp

V, VAR = "0.97.1", "avx2"
NOTE = "(pragma ignored: zzz)"
RENAME = "(renamed from razor-2p)"


def feat(n: int) -> str:
    """A feature stamp of exactly ``n`` columns (the never-cut part)."""
    return "+" + "f" * (n - 1)


# Columns: line 2 is " momwire 0.97.1 <basis> avx2" + tail, 79 at most. The
# fixed part is 21 columns plus the basis, so bs2 leaves a tail of 55 columns
# and razor2p one of 52; each tail part costs its length plus a space.
# (basis, fixed, pragma note, rename note, expected line 2)
CASES = {
    # fits: byte for byte the uncut line
    "fits": ("bs2", "", NOTE, "", " momwire 0.97.1 bs2 avx2 (pragma ignored: zzz)"),
    "fits-all": (
        "bs2",
        "+x",
        NOTE,
        "(renamed from sg)",
        " momwire 0.97.1 bs2 avx2 +x (pragma ignored: zzz) (renamed from sg)",
    ),
    "nothing": ("bs2", "", "", "", " momwire 0.97.1 bs2 avx2"),
    # 1 + 35 + 1 + 21 = 58 > 55: the names go, the note stays (1 + 35 + 17 = 53)
    "names-first": (
        "bs2",
        feat(35),
        NOTE,
        "",
        " momwire 0.97.1 bs2 avx2 " + feat(35) + " (pragma ignored)",
    ),
    # 1 + 40 + 17 = 58 > 55: the note goes
    "note-next": ("bs2", feat(40), NOTE, "", " momwire 0.97.1 bs2 avx2 " + feat(40)),
    # razor2p: 1 + 20 + 17 + 24 = 62 > 52, but 1 + 20 + 24 = 45: the rename
    # note outlives the pragma note, which is cut first
    "pragma-note-before-rename": (
        "razor2p",
        feat(20),
        NOTE,
        RENAME,
        " momwire 0.97.1 razor2p avx2 " + feat(20) + " " + RENAME,
    ),
    # 1 + 30 + 24 = 55 > 52: the rename note goes, the basis stays
    "rename-before-basis": (
        "razor2p",
        feat(30),
        NOTE,
        RENAME,
        " momwire 0.97.1 razor2p avx2 " + feat(30),
    ),
    # the fixed part alone leaves less than the basis needs: only then the
    # basis is cut, and every note is already gone
    "basis-last": (
        "razor2p",
        feat(52),
        NOTE,
        RENAME,
        " momwire 0.97.1 razor avx2 " + feat(52),
    ),
    "basis-cut-to-one-column": (
        "razor2p",
        feat(56),
        NOTE,
        RENAME,
        " momwire 0.97.1 r avx2 " + feat(56),
    ),
    "long-ignored-name": (
        "razor2p",
        "",
        "(pragma ignored: " + "x" * 120 + ")",
        "",
        " momwire 0.97.1 razor2p avx2 (pragma ignored)",
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_the_order_of_sacrifice(case):
    basis, fixed, pragmas, rename, expected = CASES[case]
    line = compose_stamp(basis, V, VAR, fixed, pragmas, rename)
    assert line == expected
    assert len(line) <= 79


@pytest.mark.parametrize("case", CASES)
def test_the_fixed_part_is_never_cut_while_it_fits(case):
    basis, fixed, pragmas, rename, _ = CASES[case]
    line = compose_stamp(basis, V, VAR, fixed, pragmas, rename)
    if len(f" momwire {V}  {VAR} {fixed}") + 1 <= 79:
        assert fixed in line


def test_a_line_that_fits_is_byte_identical_to_the_uncut_line():
    for basis in ("bs2", "razor2p", "sinusoidal"):
        for fixed in ("", "(stale: 0.96.0)", feat(40)):
            for pragmas in ("", NOTE):
                for rename in ("", RENAME):
                    parts = " ".join(p for p in (fixed, pragmas, rename) if p)
                    uncut = f" momwire {V} {basis} {VAR}" + (
                        f" {parts}" if parts else ""
                    )
                    if len(uncut) <= 79:
                        assert (
                            compose_stamp(basis, V, VAR, fixed, pragmas, rename)
                            == uncut
                        )


@pytest.fixture
def cxx():
    acc = _accel.acc
    if acc is None or not hasattr(acc, "nec5_engine_stamp"):
        pytest.skip("the compiled writer is not loaded")
    return acc.nec5_engine_stamp


@pytest.mark.parametrize("case", CASES)
def test_the_c_plus_plus_writer_cuts_in_the_same_order(cxx, case):
    basis, fixed, pragmas, rename, expected = CASES[case]
    assert cxx(basis, V, VAR, fixed, pragmas, rename) == expected


def test_the_c_plus_plus_writer_equals_python_over_a_sweep(cxx):
    for pad in range(0, 70, 3):
        fixed = "+" + "f" * pad if pad else ""
        for pragmas in ("", NOTE, "(pragma ignored: " + "y" * pad + ")"):
            for rename in ("", RENAME):
                for basis in ("bs2", "razor2p", "x" * 90, ""):
                    args = (basis, V, VAR, fixed, pragmas, rename)
                    assert cxx(*args) == compose_stamp(*args), args
