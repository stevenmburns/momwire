"""The nec2 portal's near field over the Sommerfeld ground (momwire#550).

``GN 2`` + ``NE`` used to refuse at the parser: "the near field of a
Sommerfeld half-space is not an image".  The evaluator that answers it —
``direct + C₂·image + remainder``, the remainder from
:mod:`momwire._field_point` — existed since momwire#545, but only the NEC-5
seam composed it.  momwire#1336 moved the composition into
:mod:`momwire._near_readout`, and the portal now asks it the same question.

The gate is the NEC-5 seam itself: the same physical dipole, written once in
each dialect, solved by each door, must print the same near field.  The two
decks cannot be the same MESH — nec2 feeds a segment centre, NEC-5 a node —
so the comparison is made per ampere of feed current (each door's own
``ANTENNA INPUT PARAMETERS`` current), which takes the feed-gap modelling out
and leaves the current distribution and the composition.  Measured on
Skylake 2026-10-04 against a table-scale bar:

    nec2 / NEC-5 segments   worst |E| / scale   worst live phase
          21 / 20              2.12e-3             0.20 deg
          41 / 40              1.27e-3             0.12 deg
          81 / 80              7.94e-4             0.08 deg

converging with the mesh, so what is left is discretisation, not the
composition.  The same deck over ``GN 1`` reads 4.6 % of scale away, which is
what says the finite ground is really being composed rather than aliased to
the PEC image.  The frequencies are chosen so both doors solve at one
wavelength: NEC-5's ``c`` is 299.8 MHz·m (``eznec._serve.SPEED_OF_LIGHT_MHZ_M``)
and the portal's is the SI one.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.deck import DeckError, parse
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.portal import _portal

# The table-scale bar and the phase bar, each ~2.4x the 21/20 measurement.
WORST_MAGNITUDE = 5.0e-3
WORST_PHASE_DEG = 0.5

F5 = 14.0
F2 = F5 * 299.792458 / _serve.SPEED_OF_LIGHT_MHZ_M
NE2 = "NE 0 5 1 4 -4. 0. 1. 2. 0. 2."
NE5 = "NE 0,5,1,4,-4.,0.,1.,2.,0.,2."
WIRE = "GW 1 {n} -5. 0. 6. 5. 0. 6. .001"


def _nec2(n=21, ground="GN 2 0 0 0 13. .005", wire=WIRE, ge="GE 1", ne=NE2):
    return (
        "CE elevated dipole over a Sommerfeld ground\n"
        f"{wire.format(n=n)}\n{ge}\n{ground}\n"
        f"EX 0 1 {(n + 1) // 2} 0 1. 0.\nFR 0 1 0 0 {F2!r} 0\n{ne}\n"
    )


def _nec5(n=20):
    return (
        "CM elevated dipole\nCE\n"
        f"GW 1,{n},-5.,0.,6.,5.,0.,6.,.001\nGE 1,-1\nFR 0,1,0,0,{F5!r}\n"
        "GN 0,0,0,0,13.,.005\n"
        f"EX 4,1,{n // 2},0,1.,0.\nPQ 0\n{NE5}\nEN\n"
    )


def _portal_table(text):
    """``(feed current, rows)`` from the portal's printout; rows are
    ``x y z |Ex| ph |Ey| ph |Ez| ph``."""
    out, _err = _portal.render_deck(text)
    printed = "\n".join(out)
    assert "ERROR" not in printed, printed
    current, rows = None, []
    lines = iter(out)
    for line in lines:
        if "ANTENNA INPUT PARAMETERS" in line:
            for row in lines:
                tokens = row.split()
                if len(tokens) == 11 and tokens[0].isdigit():
                    current = complex(float(tokens[4]), float(tokens[5]))
                    break
        if "NEAR ELECTRIC FIELDS" in line:
            for row in lines:
                if "Compute Time" in row:
                    break
                tokens = row.split()
                if len(tokens) != 9:
                    continue
                try:
                    rows.append([float(t) for t in tokens])
                except ValueError:
                    continue
    return current, np.array(rows)


def _portal_refusal(text):
    out, _err = _portal.render_deck(text)
    printed = "\n".join(out)
    assert "NEAR ELECTRIC FIELDS" not in printed
    return printed


def test_gn2_near_field_is_the_nec5_seams_for_the_same_dipole():
    """The capability: an nec2 deck with ``GN 2`` and ``NE`` prints a near
    field, and it is the NEC-5 seam's answer for the same structure."""
    i2, rows2 = _portal_table(_nec2())
    data = _serve.serve(parse_nec5(_nec5()))
    i5 = data.sources[0].current
    rows5 = np.array(
        [
            list(r.point)
            + [x for pair in zip(r.magnitudes, r.phases_deg) for x in pair]
            for r in data.near_fields[0].rows
        ]
    )
    assert rows2.shape == rows5.shape == (20, 9)
    np.testing.assert_allclose(rows2[:, :3], rows5[:, :3])

    mag2 = rows2[:, [3, 5, 7]] / abs(i2)
    mag5 = rows5[:, [3, 5, 7]] / abs(i5)
    scale = mag5.max()
    assert np.abs(mag2 - mag5).max() / scale < WORST_MAGNITUDE

    live = mag5 > 1e-4 * scale
    phase2 = rows2[:, [4, 6, 8]] - np.degrees(np.angle(i2))
    phase5 = rows5[:, [4, 6, 8]] - np.degrees(np.angle(i5))
    wrapped = (phase2 - phase5 + 180.0) % 360.0 - 180.0
    assert np.abs(wrapped[live]).max() < WORST_PHASE_DEG

    # ...and the finite ground is really composed: the PEC image alone (the
    # same deck over GN 1) reads several percent of scale away.
    _i1, rows1 = _portal_table(_nec2(ground="GN 1"))
    assert (
        np.abs(rows1[:, [3, 5, 7]] - rows2[:, [3, 5, 7]]).max()
        / rows2[:, [3, 5, 7]].max()
        > 0.02
    )


def test_gn0_near_field_refuses_by_name_at_the_parser():
    """``GN 0`` is the reflection-coefficient ground: a far-field
    construction with no near-field measurement behind it."""
    with pytest.raises(DeckError) as exc:
        parse(_nec2(ground="GN 0 0 0 0 13. .005"))
    assert str(exc.value) == (
        "NE over the GN 0 reflection-coefficient ground is not supported by "
        "this engine (a reflection coefficient is a far-field construction); "
        "GN 2, the Sommerfeld ground, serves the near field"
    )


def test_a_grid_point_in_the_soil_refuses_by_name():
    printed = _portal_refusal(_nec2(ne="NE 0 1 1 2 1. 0. -1. 0. 0. 2."))
    assert "which is below the finite ground" in printed


def test_a_grid_point_on_a_ground_contact_refuses_by_name():
    """A monopole standing on the ground: the field AT its base is singular
    over a finite ground, so that one cell refuses rather than printing a
    number set by the sampling constant."""
    text = (
        "CE monopole on a Sommerfeld ground\n"
        "GW 1 10 0. 0. 0. 0. 0. 5. .001\nGE 1\nGN 2 0 0 0 13. .005\n"
        "EX 0 1 1 0 1. 0.\nFR 0 1 0 0 14. 0\nNE 0 1 1 1 0. 0. 0. 0. 0. 0.\n"
    )
    printed = _portal_refusal(text)
    assert "where wire 1 stands on the finite ground" in printed


def test_a_buried_deck_near_field_refuses_by_name():
    text = _nec2(wire="GW 1 {n} -5. 0. -.5 5. 0. -.5 .001", ge="GE -1")
    printed = _portal_refusal(text)
    assert "a buried deck's near field is not supported" in printed
