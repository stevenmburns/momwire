"""The far series' degree bound — momwire#999, step 1.

`_bspline_static_far_inline.h` is the C++ twin of `_bspline_static_far.py`,
and its own docstring says it is written "operation for operation" against
it. From momwire#883 until #999 that was false in a way nothing could see:
#883 re-ran the deriver with `MAX_D = 3`, the numpy lane became generic in p
(it expands the binomial with `math.comb` and guards the domain with an
explicit raise), and the C++ lane kept

    static const double binom[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 2, 1}};

with a comment reading "C(p, j) for p <= 2, which is what the generated
family covers" — true when written, false the moment #883 landed.
`binom[p][j]` at p = 3 reads a row past the array. No bound check sits
anywhere on the path: `J_static_dispatch` reaches the series through its
far-ratio EARLY RETURN, before its own `switch`, so neither the switch's
`default:` nor the table entry points' `max_d > 2` guard covers it.

Why it never went red: `_BSPLINE_ACCEL_MAX_D = 2`, three modules away in
`_bspline_kernels.py`, stops degree 3 before the accelerator sees it. **The
gate that made the defect unreachable is the gate that raising the degree
axis removes** — so this had to be fixed before, not with, the rest of #999.

What this module pins:

  * **the degree-3 arms exist and are right.** Seven of the sixteen (p, q)
    only became reachable here; before #999 they returned whatever sat past
    the array. Measured on the pre-#999 build with this gate in place:
    p=0 q=3 gave -1.92e-81 where numpy gives 7.699e-08, and p=3 q=3 gave
    -2.96e-159 against 2.442e-12 — every degree-3 arm wrong to a relative
    1.0, several with the wrong SIGN. The tolerance below is not doing
    delicate work; any bar under 1 catches this. It is set at the house
    1e-14 so that a future loosening is visible as a change.

  * **degrees 1 and 2 are unperturbed.** The same measurement found 0 of the
    9 low-degree arms disturbed, which is the bar #999's later steps have to
    keep clearing: they renumber a dispatch that d=1 and d=2 already use.

  * **the exported bound cannot drift from the generated family.** This is
    the tripwire that would have caught #883 the day it landed.

  * **the fixed-size table does not come back.** A spelling with no array in
    it cannot be indexed out of; a source check keeps it that way.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from momwire._bspline_static_far import FAR_RATIO, MAX_P, J_static_far

acc = pytest.importorskip("momwire._accelerators")

pytestmark = pytest.mark.skipif(
    not hasattr(acc, "bspline_j_static_far"),
    reason="far-series leaf binding not built",
)

# A pair well inside the series' regime: `far_ratio` = 0.05 against a switch
# at 0.5. The series is only an answer where the caller has checked that, and
# a gate that straddled the switch would be testing the dispatch instead.
ALPHA, BETA = 0.0, 0.05
A_LEFT, B_RIGHT = 1.0, 1.05
A_WIRE = 5e-4


def _far_ratio(alpha, beta, A, B, a):
    xi0 = 0.5 * (alpha + beta) - 0.5 * (A + B)
    return 0.5 * ((beta - alpha) + (B - A)) / np.sqrt(xi0**2 + a**2)


def test_the_probe_pair_is_actually_far():
    """Otherwise every value below is outside the series' domain and the
    agreement measures nothing. The first thing to check is the instrument."""
    assert _far_ratio(ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE) <= FAR_RATIO


@pytest.mark.parametrize("p", range(MAX_P + 1))
@pytest.mark.parametrize("q", range(MAX_P + 1))
def test_the_two_lanes_agree_at_every_degree(p, q):
    cxx = acc.bspline_j_static_far(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
    npy = float(J_static_far(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE))
    assert np.isfinite(cxx)
    assert abs(cxx - npy) <= 1e-14 * abs(npy), f"(p, q) = ({p}, {q})"


def test_the_degree_three_arms_are_the_ones_that_were_broken():
    """Named separately from the sweep above so the count is legible: seven
    of the sixteen (p, q) pairs touch p = 3 or q = 3, and all seven were
    reading past the binomial table before #999."""
    arms = [(p, q) for p in range(4) for q in range(4) if p == 3 or q == 3]
    assert len(arms) == 7
    for p, q in arms:
        cxx = acc.bspline_j_static_far(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
        npy = float(J_static_far(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE))
        # Positive: every surviving term of the centred-moment expansion is
        # positive (see `_centred_moments`), so a negative value here is the
        # out-of-bounds read's signature, not a small error.
        assert cxx > 0.0, f"(p, q) = ({p}, {q}) came back {cxx!r}"
        assert abs(cxx - npy) <= 1e-14 * abs(npy)


def test_the_domain_guard_matches_the_numpy_twin():
    """Same inputs, both lanes refuse. The numpy twin's guard is a policy
    bound keeping the two spellings' DOMAINS equal — the series itself is
    generic in p — so a C++ lane that quietly answered where numpy raises
    would be the same class of divergence as the table was."""
    for p, q in ((MAX_P + 1, 0), (0, MAX_P + 1), (-1, 0), (0, -1)):
        with pytest.raises(Exception):
            acc.bspline_j_static_far(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)
        with pytest.raises(ValueError):
            J_static_far(p, q, ALPHA, BETA, A_LEFT, B_RIGHT, A_WIRE)


def test_the_exported_bound_equals_the_generated_family():
    """The tripwire #883 did not have.

    `BSPLINE_FAR_MAX_P` is a hand-written C++ mirror of a generated Python
    constant, which is exactly the shape that drifted. Exporting it costs one
    line and turns the next bump from an out-of-bounds read into a red lane.
    """
    assert acc.BSPLINE_FAR_MAX_P == MAX_P


def test_the_fixed_size_binomial_table_does_not_come_back():
    """Source check, deliberately narrow: a `binom[...]` array literal in the
    far header is the defect, and the stepped recurrence that replaced it has
    no array to index. Kept as a tripwire because the table is the obvious
    thing to reach for when someone next wants a binomial here.

    COMMENTS ARE STRIPPED FIRST, and that is not a detail. The header's own
    explanation of what went wrong quotes the indexing expression, so a naive
    substring search over the whole file matches the prose describing the
    defect and reds on the fix. Written without the strip, this test failed
    on its first run for exactly that reason — the third time this repo has
    hit the shape (the #936 AST gate, the #988 `importorskip` tripwire, and
    ruff's own `# noqa` self-quoting trap in CLAUDE.md). A source tripwire
    has to be told the difference between code and writing about code.
    """
    import momwire

    header = pathlib.Path(momwire.__file__).parent / "_bspline_static_far_inline.h"
    code = re.sub(r"//[^\n]*", "", header.read_text())
    assert "binom[" not in code, "the fixed-size binomial table is back"
    assert "c_pj" in code, "the stepped binomial recurrence is gone"
