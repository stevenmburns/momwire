"""One end-to-end portal solve per rostered basis (momwire#564, scope item 3).

The NEC-2 dialect's roster-totality gate, and the sibling of
``test_eznec_basis_choice.py`` — which asks the same question of the NEC-5
seam and had no counterpart here.

**The gap this closes.**  ``test_deck_build_solver`` only CONSTRUCTS each
entry in ``momwire.deck.BASES``, and no portal test parametrized over that
mapping, so nothing anywhere solved a deck through every rostered name.  A
basis could therefore be on the roster and answer nothing, and stay invisible
until a user picked it in a SimNEC portal dialog.

That is not hypothetical: momwire#559 rostered ``"pulse"`` and the review
caught it before merge.  ``HarringtonSolver`` implements no
``compute_port_solution``, so ``_y_and_port_coeffs`` raises ``AttributeError``
on EVERY deck — and the raise is not a refusal.  It escapes ``main`` entirely
rather than becoming a ``NEC ERROR`` line and a return code, so the daemon
dies mid-frame and SimNEC blocks in ``readLine()`` waiting for an ``NX``
sentinel that will never come.  Measured again while this gate was written:
still the failure shape, still uncaught.  A test that calls ``main`` is
therefore enough to gate it — an escaping exception fails the test by
existing, with no assertion needed to notice.

**Why the parametrization is derived rather than listed.**  ``sorted(BASES)``
means every future family pays for its own roster entry at the moment it
claims one, which is the whole point: "on the roster" and "answers a deck"
cannot drift apart again by anyone forgetting to extend a list here.  The
portal's ``_BASES`` is a total comprehension over the same mapping, so a name
added without a banner-suffix decision raises ``KeyError`` at import and takes
this module's collection down with it — also a failure, and a louder one.

**Why the bar is a band and not digits.**  Eight roster entries over six
formulations, and they are meant to disagree in the third digit — that is what
having a roster is FOR.  Measured across the eight on this deck: R 62.7-65.1
ohms, X -72.2 to -55.6 ohms, broadside gain 2.03-2.11 dBi.  The bounds below
are wide enough that a legitimate seventh formulation clears them without
anyone re-baselining, and narrow enough that the ways an end-to-end solve goes
wrong while still printing a table — a mis-mapped port, an unexcited RHS, a
silent collapse to a different basis's answer — land nowhere near them.  Value
agreement between bases is not this file's question; it is
``test_portal_differential.py``'s.
"""

from __future__ import annotations

import io

import pytest

from momwire.deck._solver import BASES
from momwire.portal import _portal as nec_portal
from momwire.portal._portal import BANNER_VERSION

# A bare centre-fed dipole in free space: 10 m of wire at 14 MHz, 11 segments
# so that segment 6 is the middle one, read out broadside at 1 km.
#
# Deliberately the SIMPLEST deck that exercises the whole chain — fill, port
# solve, currents, far field, printout — because the failure this gates is
# universal (#559's roster entry crashed on every deck, not on a hard one).
# Nothing optional is in it on purpose: an `LD` card, a ground or a network
# would each be a capability a future rostered basis might legitimately refuse,
# and a refusal is not the thing being measured here.  This deck is the floor
# every roster entry must clear to deserve the name.
_ROSTER_DECK = (
    "CE roster gate (momwire#564)\n"
    "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
    "GE 0\n"
    "EX 0 1 6 0 1.\n"
    "FR 0 1 0 0 14.0 1\n"
    "RP 0 1 1 1000 90. 0. 0. 0.\n"
    "NX\n"
)


@pytest.fixture
def scoped_engine():
    """`main` sets the invocation's basis and never puts it back — the daemon's
    session contract (momwire#587).  A suite that runs it many times over has
    to bound that itself, or one test's `--basis` decides what the next one
    solves with.
    """
    with nec_portal.engine_scope():
        yield


def _run(basis: str, deck: str = _ROSTER_DECK) -> tuple[int, str, str]:
    """One portal invocation, in process, exactly as SimNEC drives it."""
    out, err = io.StringIO(), io.StringIO()
    rc = nec_portal.main(
        ["--basis", basis], stdin=io.StringIO(deck), stdout=out, stderr=err
    )
    return rc, out.getvalue(), err.getvalue()


def _aip_impedance(text: str) -> complex:
    """The impedance the one ANTENNA INPUT PARAMETERS row printed."""
    rows, inside = [], False
    for line in text.splitlines():
        if line.strip(" -") == "ANTENNA INPUT PARAMETERS":
            inside = True
            continue
        if not inside:
            continue
        tokens = line.split()
        if len(tokens) == 11 and tokens[0].isdigit() and tokens[1].isdigit():
            rows.append(tokens)
        elif rows and not line.strip():
            inside = False
    assert len(rows) == 1, f"expected one AIP row, got {len(rows)}"
    return complex(float(rows[0][6]), float(rows[0][7]))


def _pattern_gains(text: str) -> list[float]:
    """The TOTAL gain column of every RADIATION PATTERNS row."""
    gains, armed = [], False
    for line in text.splitlines():
        tokens = line.split()
        if tokens[:2] == ["DEGREES", "DEGREES"]:
            armed = True
        elif armed and len(tokens) in (11, 12):
            gains.append(float(tokens[4]))
        elif armed:
            armed = False
    return gains


@pytest.mark.parametrize("basis", sorted(BASES))
def test_every_rostered_basis_answers_a_deck_end_to_end(basis, scoped_engine):
    """The gate: each name in `deck.BASES`, driven through `main` as SimNEC
    drives it, must answer this deck with a physical dipole.
    """
    rc, out, err = _run(basis)

    # 1. The invocation completed.  #559's roster entry never got this far.
    assert rc == 0, f"{basis}: rc={rc}\n{err}"
    assert err == "", f"{basis}: wrote to stderr: {err!r}"

    # 2. It answered rather than refused.  A refusal is a legitimate response
    #    to a deck a basis cannot serve — but not to THIS deck, which asks for
    #    nothing but a driven wire in free space.
    for marker in ("NEC ERROR", "ERROR-NEC2C"):
        assert marker not in out, f"{basis}: took the error path ({marker})"

    # 3. The banner says which physics answered, exactly.  Equality rather
    #    than containment: `bspline`'s suffix is empty, so `suffix in out`
    #    would pass for it no matter what ran.  The printout is bracketed by
    #    the banner box, so there is more than one such line and EVERY one of
    #    them has to agree; how many there are is the fixture tests' business.
    suffix = nec_portal._BASES[basis][2]
    stamped = [ln for ln in out.splitlines() if ln.startswith("VERSION:")]
    assert stamped, f"{basis}: printout carries no VERSION banner"
    assert set(stamped) == {f"VERSION:{BANNER_VERSION}{suffix}"}, (basis, stamped)

    # 4. The port solve produced a dipole's impedance, not a table of noise.
    z = _aip_impedance(out)
    assert 30.0 < z.real < 150.0, f"{basis}: R = {z.real}"
    assert -150.0 < z.imag < 50.0, f"{basis}: X = {z.imag}"

    # 5. And the currents behind it radiate like one — the half of the chain
    #    an impedance alone cannot see.  2.15 dBi is the ideal half-wave
    #    figure; every roster entry lands within a tenth of a dB of it.
    gains = _pattern_gains(out)
    assert len(gains) == 1, f"{basis}: expected one pattern row, got {len(gains)}"
    assert 1.5 < gains[0] < 2.6, f"{basis}: broadside gain = {gains[0]} dBi"


def test_the_banner_suffixes_tell_the_roster_entries_apart():
    """A suffix is how a session transcript records which physics answered, so
    two entries sharing one would make the banner unable to say — and the
    per-basis gate above would still pass for both, since each would find the
    line it expected.  The empty suffix is the default basis's, and there is
    exactly one of those.
    """
    suffixes = [nec_portal._BASES[name][2] for name in sorted(BASES)]
    assert len(set(suffixes)) == len(suffixes), suffixes
    assert suffixes.count("") == 1
