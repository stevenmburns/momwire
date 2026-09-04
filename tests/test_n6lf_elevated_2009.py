"""N6LF 2009 Table 1 — the first MODERN measured anchor on the elevated class.

Severns, R. (N6LF), "Experimental Determination of Ground System Performance
for HF Verticals, Part 3: Comparisons Between Ground Surface and Elevated
Radials", QEX Mar/Apr 2009, pp. 29-32. Table 1's elevated rows are a measured
feed-point impedance against radial height on a geometry momwire serves today
-- nothing lies in the ground plane -- which is what makes them reachable when
the rest of his series is not (see #865).

## THE BAR HERE IS OURS, NOT HIS. That is the opposite of BLE (#838).

BLE's envelope is built from terms BLE itself states. **Severns states no
uncertainty for Zi at all.** Part 1 of the series gives +-0.05 dB
*repeatability* for |S21|, which is the transmission measurement, and
describes an OSL calibration at the feed point before and after every run --
but no impedance accuracy figure anywhere in seven parts. Part 3 says the
sequence "was repeated three times on different days" and "the results did not
change significantly", which is qualitative and is recorded here as qualitative
only. So every number in the envelope below is a MODELLING spread we measured,
not a measurement uncertainty he published, and this gate must never be cited
as "agreement inside his stated error".

## The three quantities the paper does not state

**1. The soil under Table 1 is unknown, by the author's own statement.** Part 3
gives soil constants only in *segment two*'s figure captions (0.015 S/m /
eps_r 30), on a different deck (35 ft radials, 34 ft vertical), and says of the
Table 1 run:

    "This was done as a check because segment one of this experiment had been
     done earlier and ground conditions at the site had changed."

Part 2 gives 0.015 / 30 for the site and Part 4 gives 0.020 / 30 ("N6LF
soil"). Those disagree, and the author says the Table 1 soil matched neither.
**The true value is not bracketed by the two published ones.** Both are swept
here for that reason, not because one is believed.

**2. The mast diameter is never stated.** Part 3 says only "a 33.5 foot
tubular aluminum vertical antenna". Part 2 describes a telescoping vertical at
the same site "averaging 1 inch in diameter", which is suggestive and is
deliberately NOT adopted. 1 in and 2 in are swept as a nuisance parameter and
the gate holds for both.

**3. The radial insulation is unquantified, and it is the LARGEST unstated
term.** The radials are "no. 18 AWG insulated wire"; no wall thickness or
permittivity is given. Measured on this deck, a jacket moves the answer far
more than the other two gaps combined, and almost entirely in REACTANCE:

    jacket                       shift at h=6      at h=48
    0.15 mm wall, eps_r 2.3        1.76 ohm        1.60 ohm
    0.25 mm wall, eps_r 2.3        2.72            2.48
    0.25 mm wall, eps_r 3.0        3.22            2.93
    0.39 mm wall, eps_r 3.0        4.60            4.18
    0.79 mm wall, eps_r 4.0        8.64            7.82

and it does not move consistently toward the measurement: at h = 48 a thin
jacket improves agreement (bare X = -11.13 against -9.7 measured; 0.15 mm PE
gives -9.53), while at h = 6 it makes it worse (bare +8.73, jacketed +10.48,
against +6.4 measured). So the residual is not explained by insulation alone.
The decks below are modelled BARE and the jacket is carried in the envelope.

## The deck, and what is read rather than assumed

  7.2 MHz (lambda = 41.64 m); mast 33.5 ft = 10.211 m; four radials 33 ft =
  10.058 m of no. 18 AWG (radius 0.512 mm); the radials AND the base of the
  vertical elevated together to 6 / 12 / 48 in -- Part 3: "elevating the
  radials and the base of the vertical to 6 inches, 12 inches and finally
  48 inches". Antenna insulated from ground with a common-mode choke, so the
  feed is the base gap and no feedline is modelled.

Auto quadrature: `n_qp_pair` is deliberately OMITTED so momwire's own default
applies. These decks have no wire below the interface, so that resolves to the
free-space order (#863) -- the gate would be measuring our own override
otherwise.

## Table 1, transcribed twice

Dual-read as the scan rule requires: once from the PDF text layer, once from a
visual read of the page rendered at 200 dpi. The two agree exactly. The column
header is "Radial Height (Inches)"; there is no depth column anywhere in the
series.

    N radials   radial height (in)   Zi (ohm)
       64             0              39.7 - j1.2     <- surface rows, NOT
       32             0              42.9 + j2.1        reachable (#865)
       16             0              56.1 + j6.2
        8             0              85.5 + j8.0
        4             0              137  + j14.9
        4             6              43   + j6.4     <- gated below
        4            12              40.6 + j0.08
        4            48              34.8 - j9.7

## Banked corner table (degree 2, auto quadrature, bare wire)

    h(in)  dia   sigma      R        X      |Z - Zmeas|
      6    1 in  0.015    40.54    +8.58       3.28
      6    1 in  0.020    40.94    +8.73       3.11
      6    2 in  0.015    41.23   +10.10       4.10
      6    2 in  0.020    41.64   +10.24       4.08
     12    1 in  0.015    39.15    +0.88       1.65
     12    1 in  0.020    39.51    +1.06       1.47
     12    2 in  0.015    39.79    +2.39       2.45
     12    2 in  0.020    40.16    +2.57       2.53
     48    1 in  0.015    34.61   -11.31       1.62
     48    1 in  0.020    34.92   -11.13       1.43
     48    2 in  0.015    35.08    -9.77       0.29
     48    2 in  0.020    35.40    -9.59       0.61

## The envelope, term by term

    mast diameter, 1 in vs 2 in            1.7 ohm
    soil, 0.015 vs 0.020                   0.4
    basis, degree 1 vs degree 2            0.5
    mesh, x1 vs x2                         0.2
    insulation, thin-jacket allowance      3.2   (0.25 mm wall, eps_r 3.0)
    ------------------------------------------
    ROW_BAR                                6.0 ohm

Additive, every term measured on this deck, and NOT tuned to pass: the worst
corner sits at 4.10 ohm, so there is 1.9 ohm of headroom. The bar is loose in
absolute terms (15% of a 40 ohm quantity) and it is loose because the
UNSTATED insulation dominates it. **The shape gates below carry the real
content; the row bar only catches gross error.**

## What is gated, and what is not

Gated: R falls monotonically with height at every corner; X is positive at
6 in and negative at 48 in at every corner; each row within ROW_BAR of the
measurement at every corner.

NOT gated: the sign of X at 12 in. It sits at +0.08 measured -- indistinguishable
from zero -- and the model puts it between +0.88 and +2.57 across the corners.
The crossing is real and is gated by its endpoints; pinning where it falls
would be pinning our own reactance, not his measurement.

## For #865, and easy to lose

Part 4 observes that surface radials "really behave more like elevated radials
even though they may be lying right in the dirt", and its radials are insulated
wire in grass. So the wire-in-the-plane serve (#865) is probably not the buried
fill taken to a z = 0 limit: the conductor sits a jacket's thickness off the
boundary, and the reactance sensitivity measured above is the size of that
question.
"""

import numpy as np
import pytest

from momwire import BSplineSolver

# crossgate rows are ALSO slow, per pyproject's marker convention: the
# default lane excludes `slow`, not `crossgate`, so a crossgate-only file
# runs (guardrail-exempt) on every PR. This file's rows are certification
# solves and belong on the push-to-main lanes.
pytestmark = pytest.mark.slow

FT = 0.3048
IN = 0.0254
WAVELENGTH = 299.792458e6 / 7.2e6
A_18AWG = 0.512e-3  # no. 18 AWG conductor radius

# Table 1's elevated rows, read off the paper (dual-transcribed, see docstring).
MEASURED = {6: 43 + 6.4j, 12: 40.6 + 0.08j, 48: 34.8 - 9.7j}

# The two nuisance parameters the paper leaves unstated, swept rather than chosen.
MAST_RADII = {"1 in": 0.5 * IN, "2 in": 1.0 * IN}
SOILS = (0.015, 0.020)
DEFAULT_CORNER = (0.5 * IN, 0.020)

ROW_BAR = 6.0  # ohm; built additively in the docstring, not tuned to pass


def _deck(height_in, sigma, a_mast, *, degree=2, mesh=1):
    """Four radials to a hub with the mast on it, all lifted together.

    The base rises WITH the radials -- Part 3 elevates "the radials and the
    base of the vertical" as one -- so the junction is at the radial height,
    not at ground.
    """
    z0 = height_in * IN
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    length = 33.0 * FT
    wires = [
        np.array([(length * dx, length * dy, z0), (0.0, 0.0, z0)]) for dx, dy in dirs
    ]
    n_per_edge = [[12 * mesh] for _ in dirs]
    mast = len(wires)
    wires.append(np.array([(0.0, 0.0, z0), (0.0, 0.0, z0 + 33.5 * FT)]))
    n_per_edge.append([24 * mesh])
    return dict(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        junctions=[[(i, "end") for i in range(len(dirs))] + [(mast, "start")]],
        feeds=[(mast, 0.02, 1 + 0j)],
        wavelength=WAVELENGTH,
        # Per-wire radii: the mast diameter is the swept nuisance, the radials
        # are no. 18 throughout. A single scalar here would silently ignore the
        # sweep -- it did, in the probe that sized these terms.
        wire_radius=[A_18AWG] * len(dirs) + [a_mast],
        ground_z=0.0,
        ground_eps=(30.0, sigma),
        ground_model="sommerfeld",
        degree=degree,
        # n_qp_pair deliberately omitted -- see the docstring.
    )


_CACHE: dict = {}


def _z(height_in, sigma, a_mast, *, degree=2, mesh=1):
    key = (height_in, sigma, a_mast, degree, mesh)
    if key not in _CACHE:
        _CACHE[key] = BSplineSolver(
            **_deck(height_in, sigma, a_mast, degree=degree, mesh=mesh)
        ).compute_impedance()[0]
    return _CACHE[key]


def _corners():
    for dia, a in MAST_RADII.items():
        for sigma in SOILS:
            yield dia, a, sigma


@pytest.mark.crossgate
def test_the_resistance_falls_monotonically_with_radial_height():
    """The measurement's primary shape: 43 -> 40.6 -> 34.8 ohm as the screen
    and base rise. Held at every corner of the two unstated parameters."""
    for dia, a, sigma in _corners():
        r = [_z(h, sigma, a).real for h in (6, 12, 48)]
        assert r[0] > r[1] > r[2], (dia, sigma, r)


@pytest.mark.crossgate
def test_the_reactance_crosses_zero_between_the_lowest_and_highest_row():
    """Gated by its ENDPOINTS, not by where it crosses.

    Measured X runs +6.4 -> +0.08 -> -9.7. The 12 in row is indistinguishable
    from zero and the model puts it between +0.88 and +2.57 across corners, so
    pinning its sign would pin our reactance rather than his measurement.
    """
    for dia, a, sigma in _corners():
        assert _z(6, sigma, a).imag > 0, (dia, sigma, _z(6, sigma, a))
        assert _z(48, sigma, a).imag < 0, (dia, sigma, _z(48, sigma, a))


@pytest.mark.crossgate
@pytest.mark.parametrize("height_in", [6, 12, 48])
def test_each_row_sits_within_our_stated_envelope(height_in):
    """ROW_BAR is OURS, not his -- he states no uncertainty for Zi. See the
    docstring's term-by-term build; the worst corner is 4.10 against a 6.0 bar,
    so this is not tuned to pass."""
    for dia, a, sigma in _corners():
        z = _z(height_in, sigma, a)
        residual = abs(z - MEASURED[height_in])
        assert residual <= ROW_BAR, (height_in, dia, sigma, z, residual)


@pytest.mark.crossgate
def test_degree_1_is_the_second_reading_on_the_default_corner():
    """The same trunk at degree 1, which is what replaced the razor/NEC-5 pair
    underground (momwire#813/#814). A basis disagreement much larger than this
    would mean the shape above belongs to one basis rather than to the physics."""
    a, sigma = DEFAULT_CORNER
    for h in (6, 12, 48):
        d1 = _z(h, sigma, a, degree=1)
        d2 = _z(h, sigma, a)
        assert abs(d1 - d2) < 1.0, (h, d1, d2)
        # ...and they must actually DIFFER. Solved rather than compared against
        # a banked constant, on BLE's precedent, so an ignored `degree` kwarg
        # collapses the pair to the bit instead of passing as perfect agreement.
        assert abs(d1 - d2) > 0.05, ("degree kwarg ignored?", h, d1, d2)
    # ...and degree 1 must carry the same shape, not merely the same numbers.
    r1 = [_z(h, sigma, a, degree=1).real for h in (6, 12, 48)]
    assert r1[0] > r1[1] > r1[2], r1
    assert _z(6, sigma, a, degree=1).imag > 0
    assert _z(48, sigma, a, degree=1).imag < 0


@pytest.mark.crossgate
def test_the_auto_answer_is_inside_its_own_mesh_envelope():
    """One refinement rung, so the shipped answer is not a coarse-mesh
    coincidence. Doubling every edge moves it far less than the row bar."""
    a, sigma = DEFAULT_CORNER
    for h in (6, 48):
        coarse = _z(h, sigma, a)
        fine = _z(h, sigma, a, mesh=2)
        assert abs(coarse - fine) < 0.5, (h, coarse, fine)
        # Same vacuity guard: an ignored `n_per_edge_per_wire` would make the
        # two rungs bit-identical and this gate meaningless.
        assert abs(coarse - fine) > 0.01, ("mesh kwarg ignored?", h, coarse, fine)


@pytest.mark.crossgate
def test_the_nuisance_sweep_actually_sweeps():
    """Vacuity guard for every "at every corner" claim above.

    `wire_radius` takes a scalar OR a per-wire sequence. As a scalar the mast
    diameter is silently ignored, all four corners become the same deck, and
    the corner assertions become one corner counted four times -- passing
    exactly as well while testing a quarter as much.

    This is not hypothetical. The probe that sized the envelope's terms had
    precisely that bug and reported the diameter spread as 0.00 ohm, which
    inverted the conclusion: the correct value is 1.6-1.7 ohm and the diameter
    is the LARGEST of the three stated-parameter gaps, not a negligible one.
    """
    z_thin = _z(48, 0.020, MAST_RADII["1 in"])
    z_fat = _z(48, 0.020, MAST_RADII["2 in"])
    assert abs(z_thin - z_fat) > 0.5, ("mast diameter does nothing", z_thin, z_fat)

    a, _ = DEFAULT_CORNER
    assert abs(_z(48, 0.015, a) - _z(48, 0.020, a)) > 0.1, "soil does nothing"
