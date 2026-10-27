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
- a perfect ground (GN 1, S5a): the image block, a vertical and an inverted
  vee above it, a horizontal wire parallel to its image, a detached radius
  step (the image's source tube);
- wire loading (S6): LD 5 on every wire and on one, LD 2 / LD 3 per metre,
  LD 5 on a bent wire over GN 1 and on short segments (the overlap
  integrals' series branch);
- a Sommerfeld ground above the plane (S5b): the image at C2 and the
  remainder (`remainder_Q_above`: the fused kernel, the wing rows' fold,
  the grazing pairs' graded correction) on a vertical, a low horizontal, a
  T junction (a basis of more than three wings) and a loaded vee on the
  reduced kernel;
- a wire end on the plane (S7), on the direct-field fill: a monopole on GN 1,
  GN 2 and GN 3, dense (N < 60) and banded, with the extended kernel and
  without; an L and a T standing on the plane (momwire#299's end bracket, on
  the free and the image block), a slanted contact (its image is not
  coaxial with it), a loaded monopole, and the coincident-segment refusal;
- the constructor's refusal of the extended kernel at a stepped-radius
  junction.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import warnings

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
def _exact_below_grids(monkeypatch):
    """Every below/below grid the exact fill (momwire#1471's anchors off on
    both copies): whether a grid is blended reads the frequencies the
    PROCESS has seen, and the C++ source keeps its own list."""
    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")


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


GN2 = "GN 2 0 0 0 13 .005\n"
GN3 = "GN 3 0 0 0 5 .02\n"
MONO = "GW 1 10 0 0 0 0 0 .25 .001\n"
MONO70 = "GW 1 70 0 0 0 0 0 .25 .001\n"
# the NEC-4.2 slot reads EK as a no-op; its kernel falls back to the reduced
# one at a junction whose members' radii differ (the dialect default's rule)
STEP = "GW 1 5 0 0 0 0 0 .1 .001\nGW 2 5 0 0 .1 0 0 .25 .002\n"
STEP70 = "GW 1 35 0 0 0 0 0 .1 .001\nGW 2 35 0 0 .1 0 0 .25 .002\n"
# a vertical, then a horizontal arm: a bend whose node the end bracket reduces
ELL = "GW 1 8 0 0 0 0 0 .2 .001\nGW 2 8 0 0 .2 .2 0 .2 .001\n"
ELL_BANDED = "GW 1 40 0 0 0 0 0 .2 .001\nGW 2 40 0 0 .2 .2 0 .2 .001\n"
TEE = (
    "GW 1 8 0 0 0 0 0 .2 .001\nGW 2 6 0 0 .2 .15 0 .2 .001\n"
    "GW 3 6 0 0 .2 -.15 0 .2 .001\n"
)

CONTACT_CASES = [
    ("monopole_on_pec", deck(MONO, "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1")),
    ("monopole_on_gn2", deck(MONO, GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1")),
    ("monopole_on_gn3", deck(MONO, GN3 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1")),
    (
        "monopole_banded_on_pec",
        deck(MONO70, "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    (
        "monopole_banded_on_gn2",
        deck(MONO70, GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    ("ell_on_pec", deck(ELL, "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1")),
    ("ell_on_gn2", deck(ELL, GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1")),
    (
        "ell_banded_on_pec",
        deck(ELL_BANDED, "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    (
        "ell_banded_on_gn2",
        deck(ELL_BANDED, GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    ("tee_on_gn2", deck(TEE, GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1")),
    (
        "slanted_on_gn2",
        deck(
            "GW 1 10 0 0 0 .1 0 .25 .001\n", GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"
        ),
    ),
    (
        "monopole_loaded_on_pec",
        deck(MONO, "GN 1\nLD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    (
        "two_monopoles_on_gn2",
        deck(
            MONO + "GW 2 10 .1 0 0 .1 0 .25 .001\n",
            GN2 + FR + "EX 0 1 1 0 1\nEX 0 2 1 0 0 1\nXQ\n",
            ge="GE 1",
        ),
    ),
]

# The NEC-4.2 slot's reduced kernel: the fill dump drives the solver with the
# kernel the deck names (the dialect's extended one, refused at a radius-step
# junction), so these reach the direct fill on the reduced kernel through the
# run half's fallback, and `test_sinusoidal_galerkin_nec4_printout.py` gates
# them by printout.
CONTACT_STEP_CASES = [
    (
        "monopole_on_pec_reduced",
        deck(STEP, "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    (
        "monopole_on_gn2_reduced",
        deck(STEP, GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    (
        "monopole_banded_on_pec_reduced",
        deck(STEP70, "GN 1\n" + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
    (
        "monopole_banded_on_gn3_reduced",
        deck(STEP70, GN3 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
    ),
]

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
        "vertical_over_pec",
        deck(
            dipole(11).replace("0 0 -0.25 0 0 0.25", "0 0 .1 0 0 .6"),
            "GN 1\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        # every segment parallel to its own image, near it
        "horizontal_low_over_pec",
        deck(
            "GW 1 21 -.25 0 .02 .25 0 .02 .001\n",
            "GN 1\n" + FR + "EX 0 1 11 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "inverted_vee_over_pec_reduced",
        deck(
            "GW 1 8 -.3 0 .1 0 0 .3 .001\nGW 2 8 0 0 .3 .3 0 .1 .001\n",
            "GN 1\nEK -1\n" + FR + "EX 0 1 8 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "radius_step_detached_over_pec",
        deck(
            "GW 1 15 0 0 .05 0 0 .55 .001\nGW 2 15 .05 0 .05 .05 0 .55 .003\n",
            "GN 1\n" + FR + "EX 0 1 8 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "vertical_over_gn2",
        deck(
            dipole(11).replace("0 0 -0.25 0 0 0.25", "0 0 .1 0 0 .6"),
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        # low over the plane: the remainder's grazing pairs, on graded panels
        "horizontal_low_over_gn3",
        deck(
            "GW 1 21 -.25 0 .01 .25 0 .01 .001\n",
            "GN 3 0 0 0 5 .02\n" + FR + "EX 0 1 11 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        # a junction basis of more than three entries: its extra wing rows
        "t_junction_over_gn2",
        deck(
            "GW 1 10 -.25 0 .2 .25 0 .2 .001\nGW 2 5 0 0 .2 0 .3 .2 .001\n"
            "GW 3 5 0 0 .2 0 -.3 .2 .001\n",
            "GN 2 0 0 0 13 .005\n" + FR + "EX 0 1 5 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "inverted_vee_over_gn2_reduced_loaded",
        deck(
            "GW 1 8 -.3 0 .05 0 0 .3 .001\nGW 2 8 0 0 .3 .3 0 .05 .001\n",
            "GN 2 0 0 0 13 .005\nEK -1\nLD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 8 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "ld5_copper",
        deck(dipole(11), "LD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "ld5_one_wire_of_two",
        deck(
            dipole(21, 0.5) + "GW 2 11 .3 0 -.25 .3 0 .25 .001\n",
            "LD 5 2 0 0 3.5e7\n" + FR + "EX 0 1 11 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "ld2_series_per_metre",
        deck(dipole(11), "LD 2 1 0 0 10 1e-7 0\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "ld3_parallel_per_metre",
        deck(dipole(11), "LD 3 1 0 0 1e4 1e-6 0\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "ld5_bent_over_pec",
        deck(
            "GW 1 8 -.3 0 .1 0 0 .3 .001\nGW 2 8 0 0 .3 .3 0 .1 .001\n",
            "GN 1\nLD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 8 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        # a short segment (k h ~ 1e-3): the overlap's series branch
        "ld5_low_frequency",
        deck(dipole(11), "LD 5 0 0 0 5.8e7\nFR 0 1 0 0 1\nEX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
]


CASES += [(name, text, "fills") for name, text in CONTACT_CASES]

# A wholly-buried deck (momwire#1718 S11a): the mixed-potential fill in the
# lower medium, at k_m and eta_m, with the below family's remainder. Each
# refusal is the reference's, in its words; a deck on both sides of the
# interface declines (the transmitted blocks; a crossing junction is S11b).
BURIED = "GW 1 11 -.25 0 -.1 .25 0 -.1 .001\n"
BURIED_CASES = [
    ("buried_dipole_gn2", deck(BURIED, GN2 + FR + "EX 0 1 6 0 1\nXQ\n"), "fills"),
    ("buried_dipole_gn3", deck(BURIED, GN3 + FR + "EX 0 1 6 0 1\nXQ\n"), "fills"),
    (
        "buried_dipole_gn2_ld5",
        deck(BURIED, GN2 + "LD 5 0 0 0 5.8e7\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "buried_dipole_gn2_ld0",
        deck(BURIED, GN2 + "LD 0 1 3 3 20 1e-8 0\n" + FR + "EX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    (
        "buried_bent_pair_gn2",
        deck(
            "GW 1 8 -.3 0 -.05 0 0 -.15 .001\nGW 2 8 0 0 -.15 .3 0 -.05 .001\n",
            GN2 + FR + "EX 0 1 8 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        # just under the plane: the remainder's pairs at a raised order
        "buried_shallow_gn2",
        deck("GW 1 15 -.3 0 -.004 .3 0 -.004 .001\n", GN2 + FR + "EX 0 1 8 0 1\nXQ\n"),
        "fills",
    ),
    (
        "buried_two_radii_gn2",
        deck(
            BURIED + "GW 2 11 -.25 .1 -.1 .25 .1 -.1 .002\n",
            GN2 + FR + "EX 0 1 6 0 1\nXQ\n",
        ),
        "fills",
    ),
    (
        "buried_sweep_gn2",
        deck(BURIED, GN2 + "FR 0 3 0 0 280 10\nEX 0 1 6 0 1\nXQ\n"),
        "fills",
    ),
    # refusals, the reference's words
    ("buried_over_pec", deck(BURIED, "GN 1\n" + FR + "EX 0 1 6 0 1\nXQ\n"), "fills"),
    (
        "buried_beside_contact",
        deck(MONO + BURIED, GN2 + FR + "EX 0 1 1 0 1\nXQ\n", ge="GE 1"),
        "fills",
    ),
    # wires on both sides of the interface decline: detached, and a card
    # wire through the plane, which the slot splits into a crossing junction
    (
        "buried_split_at_the_plane",
        deck("GW 1 11 0 0 -.2 0 0 .2 .001\n", GN2 + FR + "EX 0 1 6 0 1\nXQ\n"),
        "declined",
    ),
    (
        "buried_beside_elevated",
        deck(
            BURIED + "GW 2 11 -.25 0 .2 .25 0 .2 .001\n",
            GN2 + FR + "EX 0 2 6 0 1\nXQ\n",
        ),
        "declined",
    ),
]
CASES += BURIED_CASES


@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_has_the_same_fill(name, text, expect, acc):
    cxx = nc.cxx_fill(text, acc)
    assert nc.classify(cxx) == expect, cxx[:500]
    if expect == "declined":
        return
    diff = nc.dc.first_difference(nc.py_fill(text), cxx)
    assert diff is None, diff


@pytest.fixture(scope="module")
def acc_fill_only():
    """The accelerator alone: the fill (G) takes no BLAS, so these run where
    numpy bundles no OpenBLAS (macOS's Accelerate build), the platform whose
    libm and contracted plain C the full comparison above cannot reach."""
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "sinusoidal_galerkin_fill_dump"):
        pytest.skip("no compiled accelerator with the Sinusoidal-Galerkin fill")
    return _accel.acc


@pytest.mark.parametrize("name, text, expect", CASES, ids=[c[0] for c in CASES])
def test_a_hand_written_deck_has_the_same_fill_on_every_platform(
    name, text, expect, acc_fill_only
):
    """G alone, both copies, on every lane (test-macos included). A
    Sommerfeld ground's remainder takes numpy's products, so it declines
    by name here and is held by the full comparison above."""
    cxx = nc.cxx_fill(text, acc_fill_only, g_only=True)
    kind = nc.classify(cxx)
    if expect == "declined":
        assert kind == "declined", cxx[:500]
        return
    if kind == "declined":
        why = bytes.fromhex(cxx.split(" ? declined ", 1)[1].split()[0]).decode()
        assert "no numpy products" in why, why
        return
    assert kind == expect, cxx[:500]
    diff = nc.dc.first_difference(nc.py_fill(text, g_only=True), cxx)
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


def test_the_buried_remainder_raises_its_pairs(acc):
    """Not vacuous: the shallow buried wire's remainder re-integrates pairs
    at a raised order (`remainder_raised_pairs`), and the C++ fill is
    Python's there too."""
    from momwire import _sinusoidal_mp as mp

    seen = []
    raised = mp.remainder_raised_pairs

    def spy(*a, **kw):
        out = raised(*a, **kw)
        seen.append(out[0].size)
        return out

    mp.remainder_raised_pairs = spy
    try:
        text = dict((c[0], c[1]) for c in CASES)["buried_shallow_gn2"]
        py = nc.py_fill(text)
    finally:
        mp.remainder_raised_pairs = raised
    assert seen and max(seen) > 0, seen
    assert nc.dc.first_difference(py, nc.cxx_fill(text, acc)) is None


def test_the_remainder_branches_are_reached(acc):
    """Not vacuous: the low horizontal wire takes the graded correction and
    the T junction folds a basis's extra wing rows."""
    from momwire import _remainder_graded
    from momwire import _sinusoidal_mp as mp

    seen = {"graded": 0, "extra": 0}
    graded, wings = _remainder_graded.pair_moments, mp.basis_wings

    def spy_graded(*a, **kw):
        seen["graded"] += 1
        return graded(*a, **kw)

    def spy_wings(starts, jbasis, coef, n_basis, width=mp.N_SHAPES):
        out = wings(starts, jbasis, coef, n_basis, width)
        seen["extra"] += out[0].size > n_basis
        return out

    _remainder_graded.pair_moments, mp.basis_wings = spy_graded, spy_wings
    try:
        cases = dict((c[0], c[1]) for c in CASES)
        nc.py_fill(cases["horizontal_low_over_gn3"])
        nc.py_fill(cases["t_junction_over_gn2"])
    finally:
        _remainder_graded.pair_moments, mp.basis_wings = graded, wings
    assert seen["graded"] and seen["extra"], seen


def test_the_direct_fill_branches_are_reached():
    """Not vacuous: the contact decks take the direct-field fill, dense and
    banded, over a folded PEC image and a composed Sommerfeld one, with the
    end bracket's product (dense and banded) and the contact charge's EK
    correction, on the extended kernel's labels and without them."""
    from momwire import sinusoidal_galerkin as sgm

    S = sgm.SinusoidalGalerkinSolver
    names = (
        "_assemble_G_banded",
        "_scatter_coef_product",
        "_tested_sommerfeld_remainder",
        "_bracket_coef_product",
        "_bracket_coef_mats",
        "_contact_charge_ek_delta",
        "_far_fill_accel",
    )
    seen = dict.fromkeys(names, 0)
    seen.update(fold=0, compose=0, ek_far=0, reduced_far=0)
    saved = {n: getattr(S, n) for n in names}
    fold = S._fold_ground_block

    def wrap(name):
        def inner(self, *a, **kw):
            seen[name] += 1
            if name == "_far_fill_accel":
                seen["ek_far" if kw.get("ek") is not None else "reduced_far"] += 1
            return saved[name](self, *a, **kw)

        return inner

    def spy_fold(self, geom, k, ctx, contribs, fg, *a, **kw):
        seen[fg.mode] += 1
        return fold(self, geom, k, ctx, contribs, fg, *a, **kw)

    for n in names:
        setattr(S, n, wrap(n))
    S._fold_ground_block = spy_fold
    try:
        for _name, text in CONTACT_CASES:
            nc.py_fill(text)
        from momwire.eznec import _nec4

        for _name, text in CONTACT_STEP_CASES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _nec4.render(text, basis="sinusoidal-galerkin")
    finally:
        for n in names:
            setattr(S, n, saved[n])
        S._fold_ground_block = fold
    assert all(seen.values()), seen
