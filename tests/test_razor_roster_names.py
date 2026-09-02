"""Razor's ROSTER names, and the refusals that are not about feeds.

Split out of `test_deck_build_solver_razor.py` when momwire#673 closed it.
That module's premise was `basis="razor"` meeting a NEC-2 deck, which #673
refuses at the seam: the `nec2` dialect addresses segment CENTRES and
`RazorSolver.capabilities.centre_feeds` is False, so `build_solver` now
declines by name rather than letting `_snap_to_knot` move the gap half a
cell. Fifteen of that module's tests were assertions ABOUT the mismatch and
went with it.

These six did not. They are about which names exist and what they bind --
`razor-2p` as the current spelling, `razor-nec5` as the deprecated alias,
plain `razor` retired at momwire#753 -- plus two refusals reached through
`node_gaps` rather than `feeds`, which the #673 gate does not touch. Three
of those claims were pinned NOWHERE else in the suite, so deleting the file
wholesale would have dropped them; they are moved verbatim instead.
"""

from __future__ import annotations

import pathlib

import pytest

from momwire.deck import BASES, build_solver
from momwire.deck.model import DeckModel, DeckWire
from momwire.razor import RazorSolver


def test_razor_2p_and_razor_nec5_are_in_the_roster():
    solver_class, kwargs = BASES["razor-2p"]
    assert solver_class is RazorSolver
    assert dict(kwargs) == {"nec5_quadrature": True}
    solver_class, kwargs = BASES["razor-nec5"]
    assert solver_class is RazorSolver
    assert dict(kwargs) == {"nec5_quadrature": True}


def test_plain_razor_retired_from_the_roster_753():
    """momwire#753: the Gauss-Legendre testing-path lane is the SAME class as
    `razor-2p`, differing only in where the testing path is sampled — not a
    second roster entry. It is reached by constructing `RazorSolver` directly,
    whose `nec5_quadrature` kwarg defaults to `False` (the GL lane)."""
    assert "razor" not in BASES
    import inspect

    default = (
        inspect.signature(RazorSolver.__init__).parameters["nec5_quadrature"].default
    )
    assert default is False


def test_razor_2p_is_the_current_spelling_and_razor_nec5_an_alias():
    """`razor-2p` names the RULE; `razor-nec5` is the deprecated spelling.

    They must stay indistinguishable — same class, same kwargs — because the
    old name shipped in the named entry points and in antennaknobs' CLI
    roster, and an install that still says `razor-nec5` must keep getting
    exactly what it got before.

    The rename is about the claim the name makes, not about any host
    mis-reading it: EZNEC takes engine class and path as explicit fields and
    infers nothing from the name, and SimNEC sniffs nec2c before nec5 so
    `momwire-nec2c-razor-nec5` always classified correctly.
    """
    two_p, kw_2p = BASES["razor-2p"]
    alias, kw_alias = BASES["razor-nec5"]
    assert two_p is alias is RazorSolver
    assert dict(kw_2p) == dict(kw_alias) == {"nec5_quadrature": True}
    # And both remain distinct from the Gauss-Legendre lane, which retired
    # from the roster at momwire#753 and is reached by construction instead
    # (see test_plain_razor_retired_from_the_roster_753).


def test_every_razor_lane_has_an_entry_point_and_a_banner_suffix():
    """A roster name a filename-only host cannot select is not really served.

    `_BANNER_SUFFIXES` already fails at import if a roster name is missing
    (which is how `razor-2p` announced itself when it was added), so this
    covers the other half: the console script, and the client's own copy of
    the roster that `test_the_clients_basis_roster_is_the_engines_own` binds.
    """
    import tomllib

    from momwire.portal._portal import _BANNER_SUFFIXES, basis_from_program_name
    from momwire_nec2c_client import BASIS_NAMES

    root = pathlib.Path(__file__).resolve().parent.parent
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    for name in ("razor-2p", "razor-nec5"):
        assert name in _BANNER_SUFFIXES, name
        assert name in BASIS_NAMES, name
        script = f"momwire-nec2c-{name}"
        assert script in scripts, f"{script} missing from [project.scripts]"
        assert basis_from_program_name(script, "nec2c-") == name
    # Plain `razor` retired from the roster at momwire#753 — none of the
    # three should still know its name.
    assert "razor" not in _BANNER_SUFFIXES
    assert "razor" not in BASIS_NAMES
    assert "momwire-nec2c-razor" not in scripts
    # The suffixes must be distinct, or a printout cannot say which ran.
    subset = {n: _BANNER_SUFFIXES[n] for n in ("razor-2p", "razor-nec5")}
    assert len(set(subset.values())) == 2, subset


def test_a_node_gap_on_a_free_end_refuses_by_naming_the_geometry():
    """momwire#603 U4 turned this from a FAMILY refusal into a SITE one.

    Razor served no node gap at all when this test was written, so a lone
    wire's start refused with "node gaps are not supported". It serves them
    now, and this deck is still wrong — the named end is nobody's junction,
    so there is no through-current path for a series EMF to sit in. The
    refusal says that instead, which is the thing the caller can act on.
    """
    # The nec2 dialect never emits `node_gaps`; build the model directly, as
    # a NEC-5 dialect's edge-source translation would.
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(4,),
            ),
        ),
        node_gaps=((0, 0, 1 + 0j),),
    )
    with pytest.raises(ValueError, match="not a member of any junction group"):
        build_solver(model, basis="razor-2p", frequency_mhz=30.0)


def test_the_refusals_are_not_bare_key_or_type_errors():
    """The issue's own wording: a `KeyError`/`TypeError` here would mean the
    translation dropped the card silently instead of naming razor's reason."""
    model = DeckModel(
        wires=(
            DeckWire(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                radius=1e-3,
                edge_elements=(4,),
            ),
        ),
        node_gaps=((0, 0, 1 + 0j),),
    )
    try:
        build_solver(model, basis="razor-2p", frequency_mhz=30.0)
    except (NotImplementedError, ValueError):
        pass
    else:
        pytest.fail("expected a refusal")
