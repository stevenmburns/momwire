"""razor over the NEC-5 corpus — the pairing with nothing to be ambiguous
about (stevenmburns/momwire#623, #672, #673).

A dialect's sources land where that dialect's basis has a degree of freedom,
and the two NEC dialects do not agree about where that is:

* **NEC-2** writes ``EX 0 tag seg`` and means the segment's CENTRE.
  ``deck/_nec2_geometry.py``'s ``resolve`` says so outright.
* **NEC-5** writes an endpoint as ``(tag, n)`` where ``n`` is a NODE
  (``deck/_nec5.py``), so its sources land on knots.

`RazorSolver` carries its sites on knots. So a NEC-5 deck is its natural
corpus and a NEC-2 deck is not: a segment centre is exactly halfway between
two knots, which is a tie by construction, and #623's tap counts 58 of them
coming from the hand-written NEC-2 decks in `test_deck_build_solver_razor.py`.

Measured here over the captures instead:

    razor serves           77 of 80, exactly the set `bspline` serves
    feed/load snaps        202
    ambiguous ones         0
    smallest margin seen   3.3e-2 of the arc

Nine decades clear of `_feed_snap.AMBIGUITY_TOL_FRAC`, and above even the
1.25e-3 that #623 measured as the smallest REAL separation anywhere in the
tree. Nothing here is close to a tie.

That the point-matched families serve 0 of 80 is the same fact from the other
side: a NEC-5 deck connects at nodes, node ports need a basis whose unknowns
live there, and #177's KCL identity is why a point-matched solver cannot have
them. The dialect does not merely prefer a knot basis — it requires one.
"""

from __future__ import annotations

import warnings

import pytest

from momwire import _feed_snap
from test_eznec_basis_choice import ACCEPTS, SERVED
from test_eznec_serve import MANIFEST, deck_text, render


def _served_by(basis: str) -> frozenset[str]:
    return frozenset(c for c in ACCEPTS if ACCEPTS[c].get(basis) is SERVED)


def test_razor_hosts_exactly_what_the_default_basis_hosts():
    """Coverage, as a set rather than a count: a basis that lost one deck and
    gained another would read the same by count and is not the same claim."""
    assert _served_by("razor") == _served_by("bspline")
    assert _served_by("razor-nec5") == _served_by("bspline")
    assert len(_served_by("razor")) == 77


def test_the_point_matched_families_host_none_of_it():
    """The control, and the reason the pairing is structural rather than a
    preference: NEC-5 connects at nodes, and a point-matched basis has no
    unknown there to drive (#177)."""
    assert _served_by("sinusoidal") == frozenset()
    assert _served_by("pulse") == frozenset()


@pytest.mark.slow
def test_no_feed_or_load_on_the_nec5_corpus_is_ambiguous_under_razor(monkeypatch):
    """**The gate this module exists for.**

    Every snap razor makes serving the whole corpus, and how close the
    nearest one comes to a tie. A NEC-2 deck on this basis ties by
    construction; a NEC-5 deck does not come near it. If that stops being
    true, either the dialect's source convention moved or the mesh did, and
    both are worth failing on.
    """
    monkeypatch.setenv(_feed_snap._TAP, "1")
    monkeypatch.setattr(_feed_snap, "_TALLY", [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _feed_snap.AmbiguousSite)
        for entry in MANIFEST["captures"]:
            render(deck_text(entry["id"]), basis="razor")

    snaps = list(_feed_snap._TALLY)
    assert snaps, "the corpus stopped reaching a snap at all"
    fracs = [s["frac"] for s in snaps if s["frac"] is not None]
    ties = [f for f in fracs if f <= _feed_snap.AMBIGUITY_TOL_FRAC]
    assert not ties, f"{len(ties)} of {len(snaps)} snaps are ambiguous"
    # Not merely clear of the bar — clear of the smallest separation #623
    # found anywhere, so this is a statement about the pairing and not about
    # where the bar happens to sit.
    assert min(fracs) > 1.0e-3, min(fracs)
