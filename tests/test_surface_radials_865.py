"""Severns 2009 Table 1, the SURFACE rows — radials on the ground, served.

Severns, R. (N6LF), QEX Mar/Apr 2009 part 3, Table 1: feed-point impedance of
a 33.5 ft vertical against the number of 33 ft No. 18 insulated radials LYING
ON GRASS at 7.2 MHz. momwire refuses a wire in the plane (momwire#865) and
always will — a conductor ON the interface is not a physical configuration.
What a real surface radial is, is the ELEVATED family at the conductor's own
centre height, and this gate is that claim measured against the one modern
series that states its soil.

## The height, and the 2 % the record got wrong

The anchor is h = **1.02 mm = 2a exactly**, sitting ON the validity floor.
`scratch/buried-flow/unit5-surface-radials.md` proposed 1.0 mm and called it
"h/a = 2", which is true for the NOMINAL No. 18 conductor radius of 0.5 mm —
but the deck carries the shared crossing-serve radius a = 0.51 mm, so 1.0 mm
is h/a = 1.96 and the floor refuses it. A 2 % inconsistency in the record, not
a disagreement about the physics: moving 1.00 -> 1.02 mm costs ~3 ohm at N = 4
and under 0.2 ohm at N >= 16, well inside anything gated here.

## THE BAR IS OURS, and the assumptions are stated rather than summed

As with the elevated anchor (#866), Severns states no uncertainty for Zi. Here
three physical quantities are also unstated, and one of them dominates
everything — so the anchor FIXES them by specification and the docstring names
their leverage, instead of folding them into a bar that would swallow the gate:

    term                                   N = 8      N = 16
    grass, +1 mm of stand-off             40.92      9.62     <-- dominant
    soil, sigma 0.015 vs 0.020             5.51      1.59
    mesh, n_rad 10 -> 20                   2.93        --

**One millimetre of grass is worth 41 ohm at N = 8.** That is not a modelling
error; it is the class. A wire on a lossy dielectric is a slow-wave line, the
radial goes electrically long, and a sparse screen detunes — which is why
`_surface_height` warns unconditionally inside h/a < 20, and why a row bar
built to cover a real installation's grass would be ~49 ohm on an 85 ohm
quantity and gate nothing.

So the deck is specified exactly (h = 2a, sigma 0.020, n_rad 10) and

    ROW_BAR = soil 5.5 + mesh 2.9 + a QUARTER millimetre of grass 10.2
            = 18.0 ohm

Each term measured on this deck. The worst row sits at 9.98, so there is 1.8x
of headroom and the bar is not fitted to the residual below.

## The class residual, named rather than absorbed

At h = 2a on momwire main 80217bd, n_rad 10, with nothing tuned:

    N     momwire              measured        R err    X err
    4     127.02 - 20.89j     137.0 + 14.9j    -9.98   -35.79
    8      80.84 +  8.55j      85.5 +  8.0j    -4.66    +0.55
    16     51.78 + 13.66j      56.1 +  6.2j    -4.32    +7.46
    32     40.49 +  8.52j      42.9 +  2.1j    -2.41    +6.42
    64     37.26 +  4.74j      39.7 -  1.2j    -2.44    +5.94

**The residual has a consistent SIGN, and that is the finding.** For every
N >= 8 momwire reads R LOW by 2.4 to 4.7 ohm and X HIGH by 0.6 to 7.5. Quoting
it as "|dZ| of 4.7 to 8.6" hides that split and makes it look like scatter.

Candidate causes, none picked: (1) the insulation dielectric is not modelled
at all and h only stands in for its thickness — a series dielectric is the
kind of thing that lowers R and raises X together, which makes this the
leading candidate; (2) grass, which raises the effective h non-uniformly along
a radial in a way one number cannot express; (3) the soil under Table 1, which
the author says he does not know (momwire#838: "ground conditions at the site
had changed"). Not absorbed into the bar silently — the bar covers it, this
paragraph names it, and closing it is a coating model, not a tuning.

## What is gated, and what is not

Gated: R falls monotonically with N, which is the measurement's headline shape
and the one thing every candidate cause above leaves alone; and each row's R
within `ROW_BAR`, with X gated too for N >= 8.

**NOT gated: X at N = 4.** Measured +14.9, model -20.9. The four-radial deck
sits on the resonance the module docstring describes, where dR/dh is ~50 ohm
per millimetre and the reactance crosses through zero; pinning its sign would
pin our stand-off convention rather than his measurement. R at N = 4 IS gated,
because the resonance moves the reactance far more than the resistance.

## Lanes, and a cost correction worth reading before moving them

Every gate here is `slow` AND `crossgate`, so none of it runs in the PR lane.

That is a change from this unit's plan, and the reason is a measurement I got
wrong first time. Standalone, these solves are 5.7 s (N = 4), 11 s (8) and
35 s (16) — the numbers the lane split was chosen from. IN THE SUITE they are
about five times slower, because `conftest` pins `OMP_NUM_THREADS=1` on every
xdist worker (deliberately: unpinned, the accelerator's pools contend and the
whole run measures 74 % slower than serial). Measured here: 282 s scattered,
378 s grouped onto one worker. The module IS in `_FIXTURE_GROUP_FILES` so its
cache is paid once rather than per worker, but grouping cannot buy back
threads.

So the honest figures for this class are the in-suite ones: ~380 s for
N = 4/8/16, and roughly 620 s and 2250 s for N = 32 and 64 at the same mesh.
The first belongs on the push lane; the last two are not gateable here at all
and are banked instead.
"""

import numpy as np
import pytest

from momwire._surface_height import SURFACE_HEIGHT_CLASS
from momwire.bspline import BSplineSolver

FT = 0.3048
F_HZ = 7.2e6
WAVELENGTH = 299792458.0 / F_HZ
MAST = 33.5 * FT
RADIAL = 33.0 * FT
A_18AWG = 0.51e-3  # the shared crossing-serve radius the deck carries
SIGMA, EPS_R = 0.020, 30.0

# h = 2a EXACTLY: the validity floor, and the anchor sits on it. See docstring.
H_ANCHOR = SURFACE_HEIGHT_CLASS.floor_h_over_a * A_18AWG

MEASURED = {
    4: 137.0 + 14.9j,
    8: 85.5 + 8.0j,
    16: 56.1 + 6.2j,
    32: 42.9 + 2.1j,
    64: 39.7 - 1.2j,
}

ROW_BAR = 18.0  # ohm; built additively in the docstring, not fitted


def _deck(n, h=H_ANCHOR, *, sigma=SIGMA, n_rad=10, degree=2):
    ang = 2.0 * np.pi * np.arange(n) / n
    wires = [
        np.array([(RADIAL * np.cos(t), RADIAL * np.sin(t), h), (0.0, 0.0, h)])
        for t in ang
    ]
    n_per_edge = [[n_rad] for _ in ang]
    mast = len(wires)
    wires.append(np.array([(0.0, 0.0, z) for z in (MAST + h, 0.5 + h, 0.05 + h, h)]))
    n_per_edge.append([19, 2, 3])
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        junctions=[[(i, "end") for i in range(n)] + [(mast, "end")]],
        feeds=[(mast, MAST - 0.05, 1 + 0j)],
        wavelength=WAVELENGTH,
        wire_radius=A_18AWG,
        ground_z=0.0,
        ground_eps=(EPS_R, sigma),
        ground_model="sommerfeld",
        degree=degree,
        # n_qp_pair omitted: this deck is wholly above the interface, so
        # momwire#863's per-deck default resolves to the free-space order.
    )


_CACHE: dict = {}


def _z(n, **kw):
    key = (n, tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = _deck(n, **kw).compute_impedance()[0]
    return _CACHE[key]


# ---------------------------------------------------------------------------
# N = 4/8/16 — the gated rows (push lane; see the docstring on cost)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.crossgate
@pytest.mark.parametrize("n", [4, 8, 16])
def test_the_surface_row_sits_within_our_stated_envelope(n):
    """R for every row; X too once the screen is off the resonance."""
    z = _z(n)
    assert abs(z.real - MEASURED[n].real) <= ROW_BAR, (n, z, MEASURED[n])
    if n >= 8:
        assert abs(z.imag - MEASURED[n].imag) <= ROW_BAR, (n, z, MEASURED[n])


@pytest.mark.slow
@pytest.mark.crossgate
def test_the_resistance_falls_as_radials_are_added():
    """The measurement's headline shape, 137 -> 85.5 -> 56.1 ohm, and the one
    thing every candidate cause of the residual leaves alone."""
    r = [_z(n).real for n in (4, 8, 16)]
    assert r[0] > r[1] > r[2], r


@pytest.mark.slow
@pytest.mark.crossgate
def test_the_residual_keeps_its_sign():
    """The finding, pinned so it cannot quietly become scatter: momwire reads
    R LOW and X HIGH on every row off the resonance. If a future change makes
    the residual two-sided, that is a different physics story and this gate
    should be read again rather than rebanked."""
    for n in (8, 16):
        z = _z(n)
        assert z.real < MEASURED[n].real, (n, z)
        assert z.imag > MEASURED[n].imag, (n, z)


@pytest.mark.slow
@pytest.mark.crossgate
def test_degree_1_is_the_second_reading():
    d1, d2 = _z(8, degree=1), _z(8)
    assert abs(d1 - d2) < ROW_BAR, (d1, d2)
    # ...and they must DIFFER, so an ignored `degree` cannot pass as agreement.
    assert abs(d1 - d2) > 0.01, ("degree kwarg ignored?", d1, d2)


@pytest.mark.slow
@pytest.mark.crossgate
def test_the_anchor_sits_exactly_on_the_validity_floor():
    """h = 2a, not the record's 1.0 mm — see the docstring's 2 % note. Pinned
    because it is the one number a reader is most likely to 'correct' back."""
    assert H_ANCHOR == pytest.approx(2.0 * A_18AWG)
    assert H_ANCHOR / A_18AWG == pytest.approx(SURFACE_HEIGHT_CLASS.floor_h_over_a)
    # A hair below it must refuse, which is what makes "on the floor" mean
    # something rather than being a round number that happens to work.
    with pytest.raises(ValueError, match="validity floor"):
        _deck(4, h=H_ANCHOR * 0.99).compute_impedance()


@pytest.mark.slow
@pytest.mark.crossgate
def test_the_grass_term_is_why_this_is_indicative_and_not_predictive():
    """One millimetre of stand-off is worth more than the whole envelope.

    Vacuity guard as much as physics: if `h` stopped reaching the geometry the
    rows above would still pass, since they only ever solve at one height.
    """
    moved = abs(_z(8, h=H_ANCHOR + 0.001) - _z(8))
    assert moved > ROW_BAR, ("h does not reach the deck?", moved)


# N = 32 and 64 are NOT gated, and the reason is cost rather than doubt.
# Their measured values are banked in the docstring (40.49 + 8.52j and
# 37.26 + 4.74j, both keeping the class's sign), and they carry the same
# monotone fall in R. In the SUITE they cost about 620 s and 2250 s -- see the
# lane note in the docstring -- so gating them would put the better part of an
# hour on the push lane for two rows that say what N = 16 already says.


# ---------------------------------------------------------------------------
# The serve's own contract: floor, advisory, and the sentence that did NOT move
# ---------------------------------------------------------------------------
#
# Deliberately a TINY deck (2 wires, 3 segments) rather than the anchor. These
# gate the refusal and the advisory, which are geometry-time decisions and do
# not care about the mesh — so they cost about a second and stay in the PR
# lane, where the rows above cannot.


def _tiny(h, a=A_18AWG):
    return BSplineSolver(
        wires=[
            np.array([(2.0, 0.0, h), (0.0, 0.0, h)]),
            np.array([(0.0, 0.0, h), (0.0, 0.0, 2.0 + h)]),
        ],
        n_per_edge_per_wire=[[2], [2]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(1, 0.05, 1 + 0j)],
        wavelength=WAVELENGTH,
        wire_radius=a,
        ground_z=0.0,
        ground_eps=(EPS_R, SIGMA),
        ground_model="sommerfeld",
    )


def test_a_wire_in_the_plane_still_refuses_with_its_own_sentence():
    """momwire#865 unit 5 changed the SERVE, not this. z = 0 is still not a
    physical configuration and still says so in the words it always used."""
    with pytest.raises(ValueError, match="lying in the ground plane"):
        _tiny(0.0).compute_impedance()


@pytest.mark.parametrize("frac", [0.5, 0.98])
def test_below_the_validity_floor_refuses_by_name(frac):
    h = SURFACE_HEIGHT_CLASS.floor_h_over_a * A_18AWG * frac
    with pytest.raises(ValueError, match="validity floor"):
        _tiny(h).compute_impedance()


def test_the_floor_refusal_names_the_height_and_the_way_out():
    """A refusal that only says no costs the user a round trip."""
    with pytest.raises(ValueError) as e:
        _tiny(0.5 * SURFACE_HEIGHT_CLASS.floor_h_over_a * A_18AWG).compute_impedance()
    msg = str(e.value)
    assert "h/a" in msg and "mm" in msg
    assert f"{SURFACE_HEIGHT_CLASS.default_h_mm:.1f}" in msg  # the recommendation
    assert SURFACE_HEIGHT_CLASS.issue in msg


def test_the_advisory_fires_inside_the_band_and_is_silent_above_it():
    """Unconditional within the class, absent outside it — and the band edge
    is `advisory_h_over_a`, whose value is read off a measured slope table in
    `_surface_height` rather than chosen."""
    from momwire._surface_height import SurfaceRadialHeight

    edge = SURFACE_HEIGHT_CLASS.advisory_h_over_a * A_18AWG
    for h, expected in ((2.0 * A_18AWG, True), (0.9 * edge, True), (1.1 * edge, False)):
        with pytest.warns(SurfaceRadialHeight) if expected else _no_advisory():
            _tiny(h).compute_impedance()


class _no_advisory:
    """`pytest.warns` has no "and not this one" form that also allows others."""

    def __enter__(self):
        from momwire._surface_height import SurfaceRadialHeight

        self._cm = __import__("warnings").catch_warnings(record=True)
        self._rec = self._cm.__enter__()
        __import__("warnings").simplefilter("always")
        self._cls = SurfaceRadialHeight
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        hits = [w for w in self._rec if issubclass(w.category, self._cls)]
        assert not hits, f"advisory fired above the band: {hits[0].message}"
        return False


def test_the_advisory_quotes_the_committed_row_not_a_literal():
    """Pinned to `SURFACE_HEIGHT_CLASS` so a re-sweep updates the sentence
    without editing prose — the `_razor_class` contract."""
    from momwire._surface_height import surface_height_message

    text = surface_height_message(2.0 * A_18AWG, A_18AWG, 4)
    assert text.startswith("a conductor lies within")  # the filter's hook
    assert f"{SURFACE_HEIGHT_CLASS.slope_sparse_ohm_per_mm:.0f} ohm per" in text
    assert f"N >= {SURFACE_HEIGHT_CLASS.dense_n}" in text
    assert "h/a = 2.0" in text
    # The sparse deck gets the "indicative rather than predictive" clause; a
    # dense one does not.
    assert "indicative rather than predictive" in text
    dense = surface_height_message(2.0 * A_18AWG, A_18AWG, 32)
    assert "indicative rather than predictive" not in dense
