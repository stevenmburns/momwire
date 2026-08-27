"""The NEC-2 portal's buried serve matrix — what answers, and what refuses.

``momwire.eznec`` has had this gate since momwire#553
(``test_eznec_buried_refusal.py``); this dialect had none, and the two seams
had drifted apart on one row.

**What serves.**  A wire strictly below a ``GN 2`` interface is filled through
the buried Sommerfeld families and its impedance, currents and charges print.
That is momwire#553's product and the first test here is its tripwire.

**What refuses, and why each sentence is different.**  ``GN 0`` and ``GN 1``
carry no lower medium to put a wire in; a wire with points on both sides of
the interface is refused here (momwire's native API serves the split
crossing spelling since momwire#524 phase 2; this portal has not adopted
it); a deck that
stands a wire END in the plane AND buries another hits the contact/buried
combination momwire#553 U5 measured itself out of; ``NE``/``NH`` over any
finite ground is momwire#524 phase 3.

**The row this file was opened for.**  ``RP`` on a buried deck was SERVED
here while the NEC-5 seam refused it — and served is the wrong word for what
it did.  ``_far_moments`` sums the upper medium's wavenumber with a
Fresnel-reflected image and nothing in that sum crosses the interface, so a
buried element contributes as though it radiated in air from a point
underground.  It does not blow up; it prints a plausible pattern.  Measured
before the refusal landed, on the deck below: -20.55 dB at theta 60, -25.72 dB
at 75, only the grazing sample floored to -999.99, and a POWER BUDGET claiming
100 % efficiency for a wire dissipating in soil at eps_r 13 / sigma 0.005 S/m.
A number a reader cannot tell from a real one is worse than a refusal, which
is momwire#570's own conclusion — "until then RP on a deck with buried wires
refuses naming this issue".

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
_RP = "RP 0 3 1 1000 60. 0. 15. 0."


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


def test_a_buried_deck_still_serves_its_impedance(scoped_engine):
    """momwire#553's product, and the thing the RP refusal must not cost.

    The refusal below is keyed on the REPORT card, so a buried deck that asks
    for nothing but its input parameters has to be unaffected by it.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN2, "XQ"))
    assert rc == 0 and err == ""
    assert _errors(out) == []
    assert "ANTENNA INPUT PARAMETERS" in out


def test_rp_on_a_buried_deck_refuses_and_names_the_issue(scoped_engine):
    """The row this file exists for (momwire#570).

    The sentence has to carry the MECHANISM and not just the verdict: a
    reader told only "not served" goes looking for a flag to turn it on,
    where one told the sum never crosses the interface knows there is
    nothing to turn on until the transmitted asymptotics land.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN2, _RP))
    assert rc == 0 and err == ""

    (message,) = _errors(out)
    assert "RP" in message
    assert "below the ground plane" in message
    assert "FAR-ZONE asymptotics" in message
    assert "momwire#570" in message
    # And it says what DOES serve, so the refusal is a fork in the road
    # rather than a dead end.
    assert "IMPEDANCE, its CURRENTS and its CHARGES are all served" in message
    # The sentence is `_medium_spec`'s, so the NEC-5 seam prints the same one
    # — a single owner rather than two copies to keep equal.
    from momwire import _medium_spec

    assert message.endswith(_medium_spec.buried_far_field_refusal())

    # No table, and in particular not a floored one: -999.99 rows were the
    # symptom, so a gate that only banned the number would have passed the
    # -20.55 dB rows that were the actual problem.
    assert "RADIATION PATTERNS" not in out
    assert "-999.99" not in out


def test_rp_over_a_sommerfeld_ground_serves_when_nothing_is_buried(scoped_engine):
    """The control, and the reason the refusal is keyed on BURIED rather than
    on the ground kind.

    A finite ground is not the obstruction — an above-ground far field over a
    Sommerfeld half-space is exactly what the Fresnel-reflected image is for,
    and it is the common case.  ``NE``/``NH`` refuse on the ground kind
    (a near field is not an image); ``RP`` must not copy that.
    """
    rc, out, err = _run(_deck(_ABOVE_WIRE, _GN2, _RP))
    assert rc == 0 and err == ""
    assert _errors(out) == []
    assert "RADIATION PATTERNS" in out


def test_a_buried_deck_over_a_ground_with_no_lower_medium_refuses_first(scoped_engine):
    """``GN 0`` has nowhere to put the wire, and that refusal outranks the
    report card's — the deck never reaches a fill, so there is no far field to
    decline.  Two refusals, and the reader gets the one that came first.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN0, _RP))
    assert rc == 0 and err == ""
    (message,) = _errors(out)
    assert "no lower medium" in message
    assert "momwire#570" not in message


def test_the_refusal_is_reported_and_the_nx_sentinel_still_lands(scoped_engine):
    """The daemon contract (grammar doc §2, §10.1): a refused deck is a
    REPORT, and the sentinel is emitted whether the deck ran or not.  An
    engine that dies on the refusal leaves SimNEC blocked in ``readLine()``
    with no timeout, which is the failure mode momwire#564's roster gate
    documents from the other direction.
    """
    rc, out, err = _run(_deck(_BURIED_WIRE, _GN2, _RP))
    assert rc == 0
    assert err == ""
    assert "ERROR-NEC2C" in out, (
        "the two-line error frame is how SimNEC reads a refusal"
    )
    assert "NX" in out
