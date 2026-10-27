"""The C++ Sinusoidal-Galerkin mixed-potential fill and solve is
`SinusoidalGalerkinSolver`'s, to the bit.

`src/momwire/_sinusoidal_galerkin_fill_core.cpp` copies
`compute_port_solution` on the default `fill="mixed-potential"` route in free
space (momwire#1718 S2): `_assemble_Z_mp`'s observer windows (the tiered pair
moments from the B-spline core's sinusoidal kernels, one call per radius run;
the near pairs, by the parallel reduction's kernel and numpy's own product
rules; the windowed assembler), `_drive_columns`, `_solve_in_place` and
`_port_currents`. `scripts/core_split/sinusoidal_galerkin_fill_compare.py`
runs the REAL solver at every operating point the NEC-4.2 build records, and
these tests hold the C++ dump (G, the coefficients, Y) equal to it over:
- the NEC-4.2 fixtures;
- hand-written decks: bends and junctions (the graded rule on a touching
  pair, by end), a near pair that neither touches nor runs parallel (the
  32-point product rule), a batch past numpy's temporary-elision size (whose
  extended-kernel factor Python names, so no interpreter elides it) and one whose rule outgrows the einsum iterator's buffer, a radius step
  (one moment call per radius run, the extended kernel's source tube), the
  reduced kernel (EK -1), two sources, an off-centre feed request, an FR
  sweep and one segment;
- the declines (a ground, wire loading) and the constructor's refusal of the
  extended kernel at a stepped-radius junction.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="the reader copies CPython 3.12's float rules"
)


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_sinusoidal_galerkin_fill_compare",
        ROOT / "scripts" / "core_split" / "sinusoidal_galerkin_fill_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "sinusoidal_galerkin_fill_dump"):
        pytest.skip("no compiled accelerator with the Sinusoidal-Galerkin fill")
    try:
        nc.blas_addresses()
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    return _accel.acc


@pytest.fixture(autouse=True)
def _one_blas_thread():
    """One BLAS thread on both sides: OpenBLAS's LU is deterministic at a
    given width, not across widths."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


def _decks(root):
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]


@pytest.mark.slow  # every NEC-4.2 fixture, solved by both copies
def test_the_nec42_fixtures_have_the_same_fill(acc):
    counts, mismatches = nc.compare(_decks(FIXTURES / "eznec_nec42"))
    assert not mismatches, mismatches[:2]
    assert counts["fills"] >= 80, counts


FR = "FR 0 1 0 0 299.7925\n"


def deck(geometry: str, body: str, ge: str = "GE 0") -> str:
    return "CM fill\nCE\n" + geometry + ge + "\n" + body + "EN\n"


def dipole(n: int, half: float = 0.25, r: float = 0.001) -> str:
    return f"GW 1 {n} 0 0 {-half} 0 0 {half} {r}\n"


SQUARE = (
    "GW 1 4 0 0 0 .2 0 0 .001\nGW 2 4 .2 0 0 .2 .2 0 .001\n"
    "GW 3 4 .2 .2 0 0 .2 0 .001\nGW 4 4 0 .2 0 0 0 0 .001\n"
)
# a zig-zag: every segment touches its neighbour at a bend, and the pairs two
# apart are near without touching or running parallel
ZIGZAG = "".join(
    f"GW {w + 1} 3 {0.1 * w} 0 {0.05 * (w % 2)} {0.1 * (w + 1)} 0 "
    f"{0.05 * ((w + 1) % 2)} .001\n"
    for w in range(8)
)
# a fan of 20 spokes from one hub: 20 touching batches of one rule each, past
# numpy's temporary-elision size on the extended kernel's factor (which
# Python names, so the product's order is the interpreter's choice nowhere)
FAN = "".join(
    f"GW {w + 1} 4 0 0 0 {0.3 * (1 + 0.01 * w):.4f} {0.05 * w:.4f} .1 .0005\n"
    for w in range(20)
)

CASES = [
    ("dipole", deck(dipole(11), FR + "EX 0 1 6 0 1\nXQ\n"), "fills"),
    (
        "dipole_reduced_kernel",
        deck(dipole(11), "EK -1\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    ("square_loop", deck(SQUARE, FR + "EX 0 1 2 0 1\nXQ\n"), "fills"),
    ("zigzag", deck(ZIGZAG, FR + "EX 0 1 2 0 1\nXQ\n"), "fills"),
    ("hub_fan", deck(FAN, FR + "EX 0 1 1 0 1\nXQ\n"), "fills"),
    (
        "t_junction",
        deck(
            "GW 1 10 0 0 -.25 0 0 .25 .001\nGW 2 5 0 0 0 .3 0 0 .001\n",
            FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        # thin wires on a fat one's line: the graded rule's eps is tiny, so
        # its panels outgrow the einsum iterator's buffer (q_i q_j > 8192)
        "thin_bend_wide_rule",
        deck(
            "GW 1 6 0 0 0 .3 0 0 1e-7\nGW 2 6 .3 0 0 .3 .3 0 1e-7\n",
            FR + "EX 0 1 3 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "radius_step_detached_ek",
        deck(
            "GW 1 15 0 0 -.25 0 0 .25 .001\nGW 2 15 .05 0 -.25 .05 0 .25 .003\n",
            FR + "EX 0 1 8 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "radius_step_junction_reduced",
        deck(
            "GW 1 5 0 0 -.25 0 0 0 .001\nGW 2 5 0 0 0 0 0 .25 .002\n",
            "EK -1\n" + FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "two_sources",
        deck(
            dipole(11) + "GW 2 11 .2 0 -.25 .2 0 .25 .001\n",
            FR + "EX 0 1 6 0 1\nEX 0 2 6 0 0 1\nXQ\n",
        ),
        "fills",
    ),
    ("sweep", deck(dipole(11), "FR 0 4 0 0 250 25\nEX 0 1 6 0 1\nXQ\n"), "fills"),
    ("one_segment", deck(dipole(1, 0.05), FR + "EX 0 1 1 0 1\nXQ\n"), "fills"),
    (
        "over_pec_declines",
        deck(
            dipole(11).replace("0 0 -0.25 0 0 0.25", "0 0 .1 0 0 .6"),
            "GN 1\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "declined",
    ),
    (
        "ld5_declines",
        deck(dipole(11), "LD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "declined",
    ),
]


@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_has_the_same_fill(name, text, expect, acc):
    cxx = nc.cxx_fill(text, acc)
    assert nc.classify(cxx) == expect, cxx[:500]
    if expect == "declined":
        return
    diff = nc.dc.first_difference(nc.py_fill(text), cxx)
    assert diff is None, diff


def test_the_stepped_radius_junction_refuses_as_the_constructor_does(acc):
    """The extended kernel at a junction whose members differ in radius:
    `SinusoidalGalerkinSolver.__init__`'s NotImplementedError, word for
    word, in both copies."""
    text = deck(
        "GW 1 5 0 0 -.25 0 0 0 .001\nGW 2 5 0 0 0 0 0 .25 .002\n",
        FR + "EX 0 1 5 0 1\nXQ\n",
    )
    cxx = nc.cxx_fill(text, acc)
    assert " ! NotImplementedError " in cxx, cxx[:300]
    assert nc.dc.first_difference(nc.py_fill(text), cxx) is None


def test_the_rules_are_reached(acc):
    """Not vacuous: the near-pair batches these decks were written for are
    the ones Python runs (the product rule, a touching batch above numpy's
    elision size, a rule wider than the einsum buffer, the parallel kernel),
    and a radius step takes one moment call per run."""
    from momwire import _sinusoidal_mp as mp

    seen = {"widths": set(), "elided": 0, "wide": 0, "parallel": 0, "tiered": 0}
    batch, par, tiered = (
        mp._pair_moments_two_rules_batch,
        mp.parallel_pair_moments,
        mp.pair_moments_tiered,
    )

    def spy_batch(l_i, r_i, l_j, r_j, a2, k, ti, wi, tj, wj, ek=None):
        n = l_i.shape[0]
        seen["widths"].add(ti.size)
        seen["elided"] += ek is not None and n * ti.size * tj.size >= 16384
        seen["wide"] += ti.size * tj.size > 8192
        return batch(l_i, r_i, l_j, r_j, a2, k, ti, wi, tj, wj, ek=ek)

    def spy_par(*a, **kw):
        seen["parallel"] += 1
        return par(*a, **kw)

    def spy_tiered(*a, **kw):
        seen["tiered"] += 1
        return tiered(*a, **kw)

    mp._pair_moments_two_rules_batch, mp.parallel_pair_moments = spy_batch, spy_par
    mp.pair_moments_tiered = spy_tiered
    try:
        for name, text, expect in CASES:
            if name in ("zigzag", "hub_fan", "thin_bend_wide_rule"):
                nc.py_fill(text)
        assert mp.NEAR_N_QP in seen["widths"], seen
        assert seen["elided"] and seen["wide"] and seen["parallel"], seen
        seen["tiered"] = 0
        nc.py_fill(dict((c[0], c[1]) for c in CASES)["radius_step_detached_ek"])
        assert seen["tiered"] == 2, seen
    finally:
        mp._pair_moments_two_rules_batch, mp.parallel_pair_moments = batch, par
        mp.pair_moments_tiered = tiered


@pytest.mark.slow  # N = 1000: two observer windows
def test_a_deck_of_two_windows_has_the_same_fill(acc):
    """N past WINDOW_BYTES / (144 N) rows: the fill's second observer window
    accumulates into rows the first one already wrote."""
    text = deck(
        "GW 1 500 0 0 -5 0 0 5 .001\nGW 2 500 .5 0 -5 .5 0 5 .001\n",
        "FR 0 1 0 0 14\nEX 0 1 250 0 1\nXQ\n",
    )
    cxx = nc.cxx_fill(text, acc)
    assert nc.classify(cxx) == "fills"
    assert nc.dc.first_difference(nc.py_fill(text), cxx) is None


def test_a_wire_listed_twice_refuses_in_the_galerkin_solvers_name(acc):
    """The front's duplicated-wire refusals (momwire#1042 / #1333) carry
    `type(self).__name__`, so the Galerkin front says SinusoidalGalerkinSolver
    where the point-matched one says SinusoidalSolver."""
    path = FIXTURES / "razor_printout_small" / "dup_counts_differ_raises.nec"
    text = path.read_bytes().decode("latin-1")
    cxx = nc.cxx_fill(text, acc)
    assert " ! ValueError " + "SinusoidalGalerkinSolver: ".encode().hex() in cxx, cxx[
        :300
    ]
    assert nc.dc.first_difference(nc.py_fill(text), cxx) is None
