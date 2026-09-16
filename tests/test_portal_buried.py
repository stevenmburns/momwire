"""The NEC-2 portal's buried serve matrix — what answers, and what refuses.

``momwire.eznec`` has had this gate since momwire#553
(``test_eznec_buried_refusal.py``); this dialect had none, and the two seams
had drifted apart on one row.

**What serves.**  A wire strictly below a ``GN 2`` interface is filled through
the buried Sommerfeld families, and its impedance, currents, charges and
RADIATION PATTERN all print.  The first three are momwire#553's product; the
pattern is momwire#570's.

**What refuses, and why each sentence is different.**  ``GN 0`` and ``GN 1``
carry no lower medium to put a wire in; a wire with points on both sides of
the interface is refused here (momwire's native API serves the split
crossing spelling since momwire#524 phase 2; this portal has not adopted
it); a deck that
stands a wire END in the plane AND buries another hits the contact/buried
combination momwire#553 U5 measured itself out of; ``NE``/``NH`` over any
finite ground is momwire#524 phase 3.

**The row this file was opened for.**  ``RP`` on a buried deck was SERVED
here while the NEC-5 seam refused it — and served was the wrong word for
what it did.  ``_far_moments`` summed the upper medium's wavenumber with a
Fresnel-reflected image, and nothing in that sum crossed the interface, so a
buried element contributed as though it radiated in air from a point
underground.  It did not blow up; it printed a plausible pattern.  Measured
on the deck below: -20.55 dB at theta 60, -25.72 dB at 75, only the grazing
sample floored, and a POWER BUDGET claiming 100 % efficiency for a wire
dissipating in soil at eps_r 13 / sigma 0.005 S/m.  A number a reader cannot
tell from a real one is worse than a refusal, so for a while the table
refused instead.

momwire#570 is the answer that refusal was holding the place for: the shared
readout now carries a below-interface element to the far zone through the
TRANSMITTED Fresnel factors, and the table prints again — off those two
numbers, not on them, which is what the tripwire below checks.  The energy
the soil takes is visible in the AVERAGE POWER GAIN rather than in the power
budget, which follows NEC's own books and is still input minus conductor
loss.

Every refusal here is REPORTED, not fatal: rc 0, the two-line error frame, and
the ``NX`` echo still emitted.  The daemon that skips the sentinel leaves
SimNEC blocked in ``readLine()`` forever, so the last test asks for it by name.
"""

from __future__ import annotations

import io

import pytest

from momwire.portal import _portal as nec_portal

_GN2 = "GN 2 0 0 0 13. .005"  # Sommerfeld: the only ground with a lower medium
_GN0 = "GN 0 0 0 0 13. .005"  # reflection-coefficient: no lower medium at all

# A 5 m wire 0.5 m under the interface, centre-fed at 14 MHz.  Buried deep
# enough that no end is within the contact tolerance of the plane, so the
# contact/buried refusal is not what answers.
_BURIED_WIRE = "GW 1 9 0. -2.5 -0.5 0. 2.5 -0.5 0.001"
_ABOVE_WIRE = "GW 1 9 0. -2.5 5. 0. 2.5 5. 0.001"
_DRIVE = "EX 0 1 5 0 1.\nFR 0 1 0 0 14.0 1"
# XNDA 1001: the trailing A digit asks for the AVERAGE POWER GAIN line, which
# is where a buried deck's ground loss becomes a number.
_RP = "RP 0 3 1 1001 60. 0. 15. 0."
# What the old image path printed for _BURIED_WIRE, kept as the thing the
# served table must NOT be.
_IMAGED_TOTAL_DB = {60.0: -20.55, 75.0: -25.72}


def _deck(wire: str, ground: str, report: str) -> str:
    return f"CE buried\n{wire}\nGE -1\n{ground}\n{_DRIVE}\n{report}\nNX\n"


@pytest.fixture
def scoped_engine():
    with nec_portal.engine_scope():
        yield


def _run(deck: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rc = nec_portal.main([], stdin=io.StringIO(deck), stdout=out, stderr=err)
    return rc, out.getvalue(), err.getvalue()


def _errors(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("ERROR:")]


def _pattern_rows(text: str) -> dict[float, tuple[str, str, str]]:
    """``{theta: (VERTC, HORIZ, TOTAL)}`` off the RADIATION PATTERNS table,
    as PRINTED — the strings, so a floored column is legible as the token the
    oracle writes rather than as a float somebody re-parsed."""
    rows: dict[float, tuple[str, str, str]] = {}
    body = text.split("RADIATION PATTERNS")[1]
    for line in body.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].replace("-", "").replace(".", "").isdigit():
            rows[float(parts[0])] = (parts[2], parts[3], parts[4])
    return rows


def _average_power_gain(text: str) -> float:
    (line,) = [ln for ln in text.splitlines() if "AVERAGE POWER GAIN" in ln]
    return float(line.split("AVERAGE POWER GAIN:")[1].split()[0])


@pytest.mark.integration
def test_a_buried_deck_still_serves_its_impedance(scoped_engine):
    """momwire#553's product, and the thing the RP refusal must not cost.

    The refusal below is keyed on the REPORT card, so a buried deck that asks
    for nothing but its input parameters has to be unaffected by it.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN2, "XQ"))
    assert rc == 0 and err == ""
    assert _errors(out) == []
    assert "ANTENNA INPUT PARAMETERS" in out


@pytest.mark.integration
def test_rp_on_a_buried_deck_serves_the_transmitted_pattern(
    scoped_engine, record_property
):
    """The row this file exists for (momwire#570).

    Four claims, and the middle two are the ones a "table printed" assertion
    would miss:

    * the table is THERE, with no error beside it;
    * only the grazing sample floors. A floored row at 60 or 75 degrees would
      mean the transmitted factors had collapsed everywhere rather than at
      the horizon, which is exactly the shape a sign error takes;
    * the served rows are NOT the numbers the old image path printed. That is
      the tripwire against the refusal being lifted without the physics
      landing under it — a regression that would otherwise pass every
      assertion above;
    * the AVERAGE POWER GAIN is well under 1. The POWER BUDGET still reads
      100 % efficiency, by NEC's own definition (input minus conductor loss),
      so this line is where the soil's absorption becomes a number.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN2, _RP))
    assert rc == 0 and err == ""
    assert _errors(out) == []
    assert "RADIATION PATTERNS" in out

    rows = _pattern_rows(out)
    assert set(rows) == {60.0, 75.0, 90.0}
    for theta in (60.0, 75.0):
        _vertc, horiz, total = rows[theta]
        assert "-999.99" not in (horiz, total), (
            f"theta {theta:g} floored — the transmitted factors vanish at the "
            "horizon, and only there"
        )
        record_property(f"buried_total_db_theta_{theta:g}", total)
        moved = abs(float(total) - _IMAGED_TOTAL_DB[theta])
        assert moved > 0.5, (
            f"theta {theta:g} prints {total} dB, within {moved:.2f} dB of the "
            f"{_IMAGED_TOTAL_DB[theta]} dB the OLD image path printed — the "
            "table is back but the physics under it may not be"
        )
    # The one row NEC floors, here for the same reason it does above ground.
    assert rows[90.0][2] == "-999.99"

    average = _average_power_gain(out)
    record_property("buried_average_power_gain", f"{average:.4E}")
    assert 0.0 < average < 1.0, (
        f"a buried deck radiating into lossy soil averaged {average:.4E} — "
        "above 1 would mean the pattern carries more power than went in"
    )
    # ...and the budget is deliberately NOT where that loss shows.
    assert "EFFICIENCY" in out


@pytest.mark.integration
def test_rp_over_a_sommerfeld_ground_serves_when_nothing_is_buried(
    scoped_engine, record_property
):
    """The control, and the scale the buried number above is read against.

    The same wire, the same ground, the same card, lifted into the air: an
    above-ground far field over a Sommerfeld half-space is the
    Fresnel-reflected image's own common case.  Its AVERAGE POWER GAIN is
    recorded beside the buried deck's so the two are one comparison rather
    than two loose numbers — the gap between them IS the soil's absorption.
    """
    rc, out, err = _run(_deck(_ABOVE_WIRE, _GN2, _RP))
    assert rc == 0 and err == ""
    assert _errors(out) == []
    assert "RADIATION PATTERNS" in out
    average = _average_power_gain(out)
    record_property("above_average_power_gain", f"{average:.4E}")
    assert 0.0 < average < 1.0


@pytest.mark.integration
def test_a_buried_deck_over_a_ground_with_no_lower_medium_refuses_first(scoped_engine):
    """``GN 0`` has nowhere to put the wire, and that refusal is untouched by
    momwire#570: the transmitted factors need a lower medium to transmit INTO,
    so serving the pattern over a Sommerfeld ground does not serve it here.
    The deck never reaches a fill, and the reader gets the geometry sentence
    rather than anything about the card.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN0, _RP))
    assert rc == 0 and err == ""
    (message,) = _errors(out)
    assert "no lower medium" in message
    assert "RADIATION PATTERNS" not in out


@pytest.mark.integration
def test_the_refusal_is_reported_and_the_nx_sentinel_still_lands(scoped_engine):
    """The daemon contract (grammar doc §2, §10.1): a refused deck is a
    REPORT, and the sentinel is emitted whether the deck ran or not.  An
    engine that dies on the refusal leaves SimNEC blocked in ``readLine()``
    with no timeout, which is the failure mode momwire#564's roster gate
    documents from the other direction.

    Carried on the ``GN 0`` deck since momwire#570: the ``GN 2`` one this used
    to ride on serves now, and a contract about how a refusal is DELIVERED
    needs a deck that still produces one.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN0, _RP))
    assert rc == 0
    assert err == ""
    assert "ERROR-NEC2C" in out, (
        "the two-line error frame is how SimNEC reads a refusal"
    )
    assert "NX" in out
