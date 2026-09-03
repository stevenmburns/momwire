"""momwire#814 prep — everything that makes the eventual flip one line.

Unit 3 turns razor's `buried` capability cell True. Three things have to move
together for that to be honest: the wholly-below family (momwire#812), the
crossing family (momwire#813), and the declared row consumers read. While they
were three independent `False`s the flip was three edits that could land
half-done, and one half-done state is worse than either end of it — a deck
SERVED by the fill while the row still declares a refusal is refused by every
consumer and answered by the solver at the same time.

They are derived from one name now, `_SERVE_BURIED`. This file holds the
derivation and the promises the flip must keep, so that flipping that one line
is the whole flip.

**Nothing here flips it.** The gates that need the fill running monkeypatch
the family flags by name, exactly as momwire#812's and #813's own gates do.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _medium_spec  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402

# The cells that are TRUE refusals and outlive the flip. A PEC or refl-coef
# ground has no lower medium whatever razor can fill; a mid-span crossing is
# still momwire's guess where the model must speak; contact+buried is
# momwire#567's measured scope decision, and it binds both trunks.
_SURVIVES_THE_FLIP = (
    "buried+pec",
    "buried+refl-coef",
    "buried+crossing",
    "buried+contact",
)

# The two the flip retires. Both say "not served YET" and nothing else.
_RETIRED_BY_THE_FLIP = ("buried", "buried+crossing_junction")


# ---------------------------------------------------------------------------
# (1) the derivation — one constant, three consequences
# ---------------------------------------------------------------------------


def test_the_two_family_flags_are_the_one_constant():
    assert _razor._SERVE_BELOW_PLANE is _razor._SERVE_BURIED
    assert _razor._SERVE_CROSSING is _razor._SERVE_BURIED


def test_the_declared_cell_is_the_one_constant():
    """The half-done state this exists to prevent: the fill serving a deck the
    row still refuses. Consumers read the row."""
    assert RazorSolver.capabilities.buried is _razor._SERVE_BURIED


def test_the_flip_retires_exactly_two_cells():
    refusals = RazorSolver.capabilities.refusals
    for cell in _SURVIVES_THE_FLIP:
        assert refusals.get(cell), f"{cell} must be declared on either side of the flip"
    for cell in _RETIRED_BY_THE_FLIP:
        assert (cell in refusals) is not _razor._SERVE_BURIED, (
            f"{cell} is declared exactly while the buried cell is False"
        )


def test_the_surviving_cells_are_the_shared_sentences():
    """They survive because they are not razor's gap: three come from
    `_medium_spec` and bind both trunks, so a flip that deleted them would be
    claiming razor serves what momwire refuses everywhere."""
    refusals = RazorSolver.capabilities.refusals
    assert refusals["buried+pec"] == _medium_spec.BURIED_PEC_REFUSAL
    assert refusals["buried+refl-coef"] == _medium_spec.BURIED_REFL_REFUSAL
    assert refusals["buried+crossing"] == _medium_spec.CROSSING_REFUSAL
    assert refusals["buried+contact"] == _medium_spec.CONTACT_WITH_BURIED_REFUSAL


# ---------------------------------------------------------------------------
# (2) the promise the row makes about what razor RAISES
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_razor._SERVE_BURIED, reason="these are the pre-flip sentences")
@pytest.mark.parametrize(
    "cell, build",
    [
        ("buried+crossing_junction", crossing_deck(1)),
        ("buried+crossing_junction", hub_deck()),
    ],
)
def test_a_buried_deck_raises_the_sentence_its_cell_declares(cell, build):
    """antennaknobs' catalog gate holds razor to this from the other side of
    the seam; this is the same promise, gated where the sentence lives."""
    declared = RazorSolver.capabilities.refusals[cell]
    with pytest.raises(ValueError) as exc:
        RazorSolver(**build, n_qp_path=8)
    assert str(exc.value).endswith(declared)


# ---------------------------------------------------------------------------
# (3) the flipped arm, exercised today: razor's labels ARE bspline's
# ---------------------------------------------------------------------------


def _crossing_members(solver):
    """The crossing junction's members as a SET.

    Compared as a set and not by junction index on purpose: bspline reads the
    junction list it was DECLARED, razor detects its own, so the two orderings
    are not required to agree and gating on the index would pin a coincidence.
    """
    if isinstance(solver, RazorSolver):
        groups = solver._find_junctions()
        return {
            frozenset(map(tuple, groups[j]["ends"]))
            for j in solver._crossing_junctions()
        }
    return {
        frozenset(map(tuple, solver.junctions[j])) for j in solver._crossing_junctions()
    }


@pytest.mark.parametrize(
    "name, build", [("hub", hub_deck()), ("crossing", crossing_deck(1))]
)
def test_razors_labels_are_bsplines_on_the_flip_decks(name, build, monkeypatch):
    """The flipped arm of antennaknobs' catalog gate, run TODAY.

    momwire#814's definition of done has razor answering where it refuses now.
    The answer is unit 3's; the LABELS are not, and they are what the flip
    exposes first — a deck razor labels differently from bspline is a deck the
    flip would send to the wrong fill. So the labels are gated here, before
    anything is flipped, on the two decks the flip is for.
    """
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", True)

    b = BSplineSolver(**build)
    r = RazorSolver(**build, n_qp_path=8)

    assert r._wire_media() == b._wire_media()
    assert _medium_spec.BELOW in r._wire_media()
    assert len(r._crossing_junctions()) == len(b._crossing_junctions()) == 1
    assert _crossing_members(r) == _crossing_members(b)


def test_the_labels_do_not_need_the_flip_to_agree(monkeypatch):
    """...and they agree with the flags OFF too, on the wholly-below deck the
    crossing flag has nothing to do with: labelling is `_medium_spec`'s, and
    momwire#848 made both trunks share the one geometric test it keys on. If
    this ever diverges, the flip is not the thing that broke it."""
    build = dict(hub_deck())
    b = BSplineSolver(**build)
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    r = RazorSolver(**build, n_qp_path=8)
    assert r._grounded_junction_ends() == b._grounded_junction_ends()
