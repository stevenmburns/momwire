"""momwire#1580: a short wire nothing connects to is an ordinary wire.

The phantom-wire rule (momwire#1139) took EZNEC's parked circuit node out of
the solve: a wire small against the wavelength and far from everything else.
Geometry alone also caught an ordinary wire of that shape -- a 2-segment
probe 236 m from a vertical, the way a launch-angle or transfer study places
one -- and dropped it SILENTLY: no current, an infinite impedance, and the
coupling it was there to measure gone, where NEC-5 solves it.

The rule now asks the deck first: only a wire a TL or NT card connects to can
be the parked node (EZNEC parks one only to hang a circuit on it). Any wire
the rule still drops is named in an advisory.

The fixture is synthetic, of the same shape as the deck that reported it. Its
reference numbers are NEC-5's (x13), run as a black box on this deck.

Gates, the first three failing with the production change reverted:

1. the probe is not a phantom;
2. served, its driving point and the vertical's coupled current are NEC-5's
   (razor-2p, the served default, to 4 digits);
3. the B-spline engine serves it too, within its basis difference;
4. what must NOT move: a parked wire an NT connects to is still the phantom,
   and is now named in an advisory.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve, _shell

HERE = Path(__file__).parent
PROBE = HERE / "fixtures" / "eznec_phantom_1580" / "probe_transfer.nec"
WA7ARK = HERE / "fixtures" / "eznec_phantom_1139" / "ground_rod_efhw_wa7ark.nec"

# NEC-5 (x13), ANTENNA INPUT PARAMETERS, run 1 (1 V on the probe):
# the probe's impedance, and the current the vertical's 1e-10 V source sees.
NEC5_PROBE_Z = complex(4.6871e-01, -4.4060e03)
NEC5_VERTICAL_I = complex(-3.9444e-07, 4.6769e-08)
# Run 2 (1 V on the vertical): the vertical's impedance.
NEC5_VERTICAL_Z = complex(71.957, 0.064)


def deck(path):
    return parse_nec5(path.read_bytes().decode("latin-1"))


def wavelength(d):
    return _serve.SPEED_OF_LIGHT_MHZ_M / d.frequency_mhz


def input_rows(printout):
    """``[(tag, current, impedance), ...]`` from every ANTENNA INPUT
    PARAMETERS table, in printout order."""
    lines = printout.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" not in line:
            continue
        for row in lines[i + 4 : i + 6]:
            f = row.split()
            rows.append(
                (
                    int(f[0]),
                    complex(float(f[5]), float(f[6])),
                    complex(float(f[7]), float(f[8])),
                )
            )
    return rows


def served(basis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _shell.render(PROBE.read_text(), basis=basis)


def test_an_unconnected_short_far_wire_is_not_a_phantom():
    d = deck(PROBE)
    # It IS the shape the geometric rule reads as parked ...
    assert _serve._phantom_tags_geometric(d, wavelength(d)) == {1}
    # ... and nothing connects to it, so it is an ordinary wire.
    assert _serve._phantom_tags(d, wavelength(d)) == frozenset()


def test_the_probe_solves_as_nec5_does():
    rows = input_rows(served("razor-2p"))
    assert [r[0] for r in rows] == [1, 2, 1, 2]
    (_, _, probe_z), (_, vertical_i, _), _, (_, _, vertical_z) = rows
    assert probe_z == pytest.approx(NEC5_PROBE_Z, rel=1e-4)
    assert vertical_i == pytest.approx(NEC5_VERTICAL_I, rel=1e-4)
    assert vertical_z == pytest.approx(NEC5_VERTICAL_Z, rel=1e-4)


def test_the_bspline_engine_solves_the_probe_too():
    # A 2-segment wire is where the two bases differ most (~8 % in the
    # probe's reactance); the gate is that it carries current at all, and
    # couples to the vertical with NEC-5's magnitude.
    rows = input_rows(served("bspline"))
    (_, _, probe_z), (_, vertical_i, _), _, _ = rows
    assert probe_z == pytest.approx(NEC5_PROBE_Z, rel=0.15)
    assert abs(vertical_i) == pytest.approx(abs(NEC5_VERTICAL_I), rel=0.15)


def test_a_parked_wire_an_nt_connects_to_is_still_the_phantom_and_is_named():
    d = deck(WA7ARK)
    assert _serve._phantom_tags(d, wavelength(d)) == {7}
    with pytest.warns(UserWarning, match=r"GW tag\(s\) 7 as EZNEC's parked"):
        _serve.build_mesh(
            d,
            _serve.structure_of(d),
            crossing=bool(_serve._crossing_nodes(d)),
        )
