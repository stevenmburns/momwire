"""Below-plane and in-plane wires refuse BY NAME through the seam.

momwire#525 (the naming), momwire#524 (the buried-wire capability itself).
Before this, both cases leaked the catch-all ``INTERNAL ERROR IN MOMWIRE
ENGINE - ValueError: ...`` frame — the solver's construction guard doing the
work with an internal traceback line reaching the printout. Safe, but not
the refusal grammar. The licensed engine SERVES the buried deck (probe
2026-08-21: monopole + radial 15 cm down over GN 0 → 92.13 − j70.14 Ω), so
the sentence must read as a capability statement, not a deck error.

Free space is exempt on both counts: z < 0 is legal geometry with no ground,
and the same wires must serve under GN -1.
"""

from momwire.eznec._shell import render


def deck(radial_z, ge, ground):
    return (
        "CM buried probe\n"
        "CE\n"
        "GW 1,15,0.,0.,10.,0.,0.,0.,.001\n"
        f"GW 2,10,0.,0.,{radial_z},5.,0.,{radial_z},.001\n"
        f"GE {ge}\n"
        "FR 0,1,0,0,7.\n"
        f"{ground}\n"
        "EX 4,1,7,0,1.,0.\n"
        "PQ 0\n"
        "XQ 0\n"
        "EN\n"
    )


def reason(text):
    out = render(text)
    assert " ***** NEC ERROR - " in out
    return out.split(" ***** NEC ERROR - ")[1]


def test_buried_wire_refuses_by_name_under_every_served_ground():
    for ground in ("GN 0,0,0,0,13.,.005", "GD 0,0,0,0,13.,.005", "GN 1"):
        r = reason(deck(-0.15, "1,-1", ground))
        assert "wire 2" in r, ground
        assert "below the ground plane" in r, ground
        assert "buried wires are not served" in r, ground
        assert "INTERNAL ERROR" not in r, ground


def test_in_plane_wire_refuses_by_name_not_internal_error():
    r = reason(deck(0.0, "1,-1", "GN 0,0,0,0,13.,.005"))
    assert "wire 2" in r
    assert "in the ground plane" in r
    assert "INTERNAL ERROR" not in r


def test_free_space_serves_the_same_wires():
    out = render(deck(-0.15, "0,-1", "GN -1"))
    assert "NEC ERROR" not in out


def test_ground_contact_still_serves():
    contact = (
        "CM contact control\n"
        "CE\n"
        "GW 1,15,0.,0.,10.,0.,0.,0.,.001\n"
        "GE 1,-1\n"
        "FR 0,1,0,0,7.\n"
        "GN 0,0,0,0,13.,.005\n"
        "EX 4,1,7,0,1.,0.\n"
        "PQ 0\n"
        "XQ 0\n"
        "EN\n"
    )
    out = render(contact)
    assert "NEC ERROR" not in out


def test_a_slight_standoff_above_the_plane_serves():
    """1 cm above ground is a legal elevated radial, not a refusal."""
    out = render(deck(0.01, "1,-1", "GN 0,0,0,0,13.,.005"))
    assert "NEC ERROR" not in out
