"""momwire#814 prep, restaged by momwire#1149 — two constants, two cells.

razor's buried fill has two halves: the wholly-below family (momwire#812),
which since momwire#1149 U1 also carries the DETACHED deck, and the crossing
family (momwire#813). They used to move together under one name,
`_SERVE_BURIED`, so the flip could not land half-done. The 2026-09-22
re-measurement split the verdict — everything but the crossing node is
twin-grade and reciprocity-decaying, the node is not — so the serve is
staged as SinusoidalGalerkinSolver's was (momwire#980 D1/D2 before D3):

* `_SERVE_BELOW_PLANE` (True since U0) owns the `buried` cell;
* `_SERVE_CROSSING` (True since U2) owns `buried+crossing_junction`.

The half-done state the single name prevented is still the thing to prevent,
now per pair: a deck SERVED by the fill while the row still declares a
refusal is refused by every consumer and answered by the solver at the same
time. So each constant is held equal to exactly one declared cell here, and
the crossing flag may never be on with the below flag off (the crossing
assembly fills its below half through the below family).

The gates that need the crossing fill running monkeypatch the family flag by
name, exactly as momwire#813's own gates do.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _below_interface  # noqa: E402
from momwire import _medium_spec  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402

# The cells that are TRUE refusals and outlive the whole arc. A PEC or
# refl-coef ground has no lower medium whatever razor can fill; a mid-span
# crossing is still momwire's guess where the model must speak;
# contact+buried is momwire#567's measured scope decision, and it binds both
# trunks; EK below the plane is refused tree-wide.
_SURVIVES_THE_ARC = (
    "buried+pec",
    "buried+refl-coef",
    "buried+crossing",
    "buried+contact",
    "buried+extended_kernel",
)

# Which constant owns which "not served YET" cell.
_OWNED = {
    "buried": "_SERVE_BELOW_PLANE",
    "buried+crossing_junction": "_SERVE_CROSSING",
}


# ---------------------------------------------------------------------------
# (1) two constants, each owning one cell
# ---------------------------------------------------------------------------


def test_the_single_constant_is_gone_and_the_two_are_their_own():
    """`_SERVE_BURIED` derived both flags; a leftover copy would be a third
    name a flip could forget. The state is pinned too, so moving either
    flag is a deliberate edit to this line as well (momwire#1149 U2 moved the
    second)."""
    assert not hasattr(_razor, "_SERVE_BURIED")
    assert (_razor._SERVE_BELOW_PLANE, _razor._SERVE_CROSSING) == (True, True)


def test_the_crossing_flag_never_runs_ahead_of_the_below_flag():
    """`_assemble_Z_crossing` fills its below half through the below family,
    so a crossing serve without it is not a state the code can be in."""
    assert _razor._SERVE_BELOW_PLANE or not _razor._SERVE_CROSSING


def test_the_declared_buried_cell_is_the_below_constant():
    """The half-done state this exists to prevent: the fill serving a deck the
    row still refuses. Consumers read the row."""
    assert RazorSolver.capabilities.buried is _razor._SERVE_BELOW_PLANE


@pytest.mark.parametrize("cell", sorted(_OWNED))
def test_each_not_yet_cell_is_declared_exactly_while_its_constant_is_off(cell):
    served = getattr(_razor, _OWNED[cell])
    assert (cell in RazorSolver.capabilities.refusals) is not served, (
        f"{cell} must be declared exactly while {_OWNED[cell]} is False"
    )


def test_the_surviving_cells_are_declared_on_either_side_of_the_flip():
    refusals = RazorSolver.capabilities.refusals
    for cell in _SURVIVES_THE_ARC:
        assert refusals.get(cell), f"{cell} must be declared on either side"


def test_the_surviving_cells_are_the_shared_sentences():
    """They survive because they are not razor's gap: they come from
    `_medium_spec` / `_below_interface` and bind both trunks, so a flip that
    deleted them would be claiming razor serves what momwire refuses
    everywhere."""
    refusals = RazorSolver.capabilities.refusals
    assert refusals["buried+pec"] == _medium_spec.BURIED_PEC_REFUSAL
    assert refusals["buried+refl-coef"] == _medium_spec.BURIED_REFL_REFUSAL
    assert refusals["buried+crossing"] == _medium_spec.CROSSING_REFUSAL
    assert refusals["buried+contact"] == _medium_spec.CONTACT_WITH_BURIED_REFUSAL
    assert (
        refusals["buried+extended_kernel"]
        is _below_interface.BURIED_EXTENDED_KERNEL_REFUSAL
    )
    assert (
        refusals["buried+extended_kernel"]
        is (BSplineSolver.capabilities.refusals["buried+extended_kernel"])
    )


def test_the_units_own_scope_cells_are_declared():
    """What momwire#1149 served around, declared rather than raised bare.
    Loading on a crossing deck was U0's cell until U3 served bare-metal
    loading, mixed radii on a detached deck U1's until U2b served them, and
    a jacket on a buried wire of a crossing deck U3's until momwire#1154
    served it; all three cells are gone."""
    caps = RazorSolver.capabilities
    assert caps.refusal("crossing_junction", "insulation") is None
    assert "crossing_junction+insulation" not in caps.refusals
    assert caps.refusal("wire_loading", "crossing_junction") is None
    assert "wire_loading+crossing_junction" not in caps.refusals
    assert caps.refusal("per_wire_radius", "detached") is None
    assert "per_wire_radius+detached" not in caps.refusals


# ---------------------------------------------------------------------------
# (2) the flip decks, and the sentence the constant still owns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [crossing_deck(1), hub_deck()])
def test_the_flip_decks_are_served_now(build):
    """The two decks whose sentence this section pinned before U2 construct
    on the crossing route and answer. What they answer is
    `tests/test_razor_crossing_node_1149.py`'s to gate."""
    kw = {k: v for k, v in build.items() if k != "junctions"}
    s = RazorSolver(**kw, n_qp_path=8)
    assert s._crossing
    z = complex(s.compute_impedance()[0])
    assert z.real > 0


def test_switched_off_the_crossing_deck_raises_the_named_sentence(monkeypatch):
    """The constant still owns its sentence: a build with the crossing serve
    off refuses by name (the row is built at import, so it is the raise that
    is checked here, not the row)."""
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", False)
    kw = {k: v for k, v in crossing_deck(1).items() if k != "junctions"}
    with pytest.raises(ValueError) as exc:
        RazorSolver(**kw, n_qp_path=8)
    assert str(exc.value).endswith(_razor._CROSSING_NOT_SERVED_REFUSAL)


# ---------------------------------------------------------------------------
# (3) the crossing arm, exercised today: razor's labels ARE bspline's
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
