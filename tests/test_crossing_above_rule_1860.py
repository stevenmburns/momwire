"""A crossing deck's above/above remainder at the above-ground rule
(momwire#1860 B1, the opt-in feature `crossing_above_rule`).

nec5-bspline's and nec42-bspline's crossing route filled the above x above
Sommerfeld remainder as a field-form block at `n_qp_buried_field` (q = 6), the
order the cross block's 1/R^3 integrand needs, where the same deck without its
buried rod fills those pairs at `n_qp_sommerfeld` plus #1189's raised pairs.
On, the crossing route takes that rule (`_below_interface.
_crossing_above_remainder`), with the pairs of a segment touching the plane
kept at the cross block's order (`BSplineSolver.
_crossing_above_remainder_rule`), on a grid that costs no more than the
shipped route's (`_below_interface.crossing_above_grid`): where it would
cost more, B1 declines that fill.

What is held:

- off by default, and off is the shipped fill: the lazy above grid and the
  q = 6 block, and the B1 route never entered;
- on, the B1 route serves the crossing deck's above block and the shipped
  route does not, and a deck with no crossing is untouched;
- on, efhw_lo (graded pairs, a census that does not take the whole grid)
  and the hub (16 above segments) decline and print the off printout;
- on a lazy view, B1 reads exactly what it reads on the whole grid;
- on, a row-restricted fill's rows are the full fill's rows, bit for bit;
- on, the printed input impedance is within `ZIN_RTOL` of off for
  nec5-bspline and nec42-bspline on the EFHW-like inverted L (141 and 590
  segments), a buried-hub vertical and a rod Beverage (synthetic decks).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from momwire import _below_interface, _features
from momwire.bspline import BSplineSolver

from test_crossing_serve_524 import crossing_deck
from test_crossing_sparse_1109 import _whole_wire_segments

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

ENV = "MOMWIRE_X_CROSSING_ABOVE_RULE"

# The stated tolerance: |Zin(on) - Zin(off)| / |Zin(off)|. Measured on
# Skylake (the PR's table): printed Zin unchanged on both EFHW meshes and the
# hub, 1.5e-6 on the Beverage (0.001 ohm); this leaves the printout's last
# digit and a factor of a few.
ZIN_RTOL = 1e-5


def _efhw(n_vert, n_rod, n_top, n_cp):
    return f"""CM EFHW-like inverted L, buried 3 m rod, 4 m counterpoise at 1 cm
CE
GW 1 {n_vert} 0 0 0 0 0 10 0.001
GW 2 {n_rod} 0 0 0 0 0 -3 0.008
GW 3 {n_top} 0 0 10 30 0 10 0.001
GW 4 {n_cp} -0.02 0 0.01 -4 0 0.01 0.001
GE -1
GN 2 0 0 0 13 0.005
FR 0 1 0 0 3.6 0
EX 0 1 1 0 1 0
XQ
EN
"""


def _hub():
    radials = "\n".join(
        f"GW {2 + i} 10 0.0 0.0 -0.15 {5.0 * np.cos(i * np.pi / 4):.9f} "
        f"{5.0 * np.sin(i * np.pi / 4):.9f} -0.15 0.001"
        for i in range(8)
    )
    return f"""CM vertical from a buried hub, 8 radials
CE
GW 1 2 0.0 0.0 -0.15 0.0 0.0 0.0 0.001
GW 10 16 0.0 0.0 0.0 0.0 0.0 10.0 0.001
{radials}
GE -1
GN 2 0 0 0 13 0.005
EX 0 10 2 0 1.0 0.0
FR 0 1 0 0 7.0 0
XQ
EN
"""


BEVERAGE = """CM rod Beverage (momwire#1187 geometry): 245.73 m at 2.5 m, 1.8 m rods
CE
GW 1 7 0 0 -1.8 0 0 0 0.000814
GW 2 1 0 0 0 0 0 0.25 0.000814
GW 3 1 0 0 0.25 0 0 2.5 0.000814
GW 4 126 0 0 2.5 245.73 0 2.5 0.000814
GW 5 1 245.73 0 2.5 245.73 0 0.25 0.000814
GW 6 1 245.73 0 0.25 245.73 0 0 0.000814
GW 7 7 245.73 0 0 245.73 0 -1.8 0.000814
GE -1
LD 4 6 1 1 500 0
GN 2 0 0 0 13 0.005
FR 0 1 0 0 1.83 0
EX 0 2 1 0 1 0
XQ
EN
"""

# The ACA screen's vertical runs through the plane: the case that set the
# crossing node's band at two segments (`_crossing_above_remainder_rule`).
ACA = (
    Path(__file__).parent
    / "fixtures"
    / "bspline_printout_crossing"
    / "aca_vertical_screen.nec"
).read_text()

DECKS = {
    "efhw_lo": _efhw(30, 9, 90, 12),
    "efhw_hi": _efhw(120, 38, 380, 52),
    "hub": _hub(),
    "beverage": BEVERAGE,
    "aca": ACA,
}


_FLOAT = re.compile(r"[-+]?\d\.\d+E[-+]\d+")


def _zin(text):
    """The first source's input impedance, from either dialect's table (the
    NEC-4.2 one runs its columns together)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "INPUT PARAMETERS" in line:
            for r in lines[i + 1 : i + 8]:
                f = _FLOAT.findall(r)
                if len(f) >= 6:
                    return complex(float(f[4]), float(f[5]))
    raise AssertionError("no input impedance in the printout")


def _render(deck, dialect):
    from momwire.dropin.nec5 import _shell

    # Level 1, as antennaknobs runs it: the shell's render without its
    # drop-in context.
    return _shell.render.__wrapped__(deck, basis="bspline", dialect=dialect)


@pytest.fixture
def routes(monkeypatch):
    """Count which route fills the above block."""
    seen = {"b1": 0, "lazy": 0}
    real_b1 = _below_interface._crossing_above_remainder
    real_lazy = _below_interface.lazy_above_grid

    def b1(*a, **kw):
        seen["b1"] += 1
        return real_b1(*a, **kw)

    def lazy(*a, **kw):
        seen["lazy"] += 1
        return real_lazy(*a, **kw)

    monkeypatch.setattr(_below_interface, "_crossing_above_remainder", b1)
    monkeypatch.setattr(_below_interface, "lazy_above_grid", lazy)
    return seen


def test_registry_entry():
    f = _features.FEATURES["crossing_above_rule"]
    assert (f.reach, f.kind, f.env, f.kwarg, f.issue) == (
        1,
        "python",
        ENV,
        "crossing_above_rule",
        1860,
    )
    assert "tests/test_crossing_above_rule_1860.py" in f.paths


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not _features.enabled("crossing_above_rule")
    assert not BSplineSolver(**crossing_deck())._crossing_above_rule
    monkeypatch.setenv(ENV, "1")
    assert BSplineSolver(**crossing_deck())._crossing_above_rule
    # The keyword wins over the environment.
    assert not BSplineSolver(
        **crossing_deck(), crossing_above_rule=False
    )._crossing_above_rule


def test_off_is_the_shipped_route_and_on_is_b1(monkeypatch, tmp_path, routes):
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    # The hub has 16 above segments: let B1 serve it (`_B1_MIN_ABOVE_SEGS`).
    monkeypatch.setattr(_below_interface, "_B1_MIN_ABOVE_SEGS", 0)
    deck = DECKS["hub"]
    monkeypatch.delenv(ENV, raising=False)
    off = _render(deck, "nec5")
    assert routes == {"b1": 0, "lazy": 1}
    monkeypatch.setenv(ENV, "1")
    on = _render(deck, "nec5")
    assert routes == {"b1": 1, "lazy": 1}
    assert "+crossing_above_rule" in on.splitlines()[1]
    assert "+crossing_above_rule" not in off.splitlines()[1]
    # Explicitly off through the keyword is the default fill, bit for bit.
    s_off = BSplineSolver(**crossing_deck(), crossing_above_rule=False)
    s_def = BSplineSolver(**crossing_deck())
    monkeypatch.delenv(ENV, raising=False)
    s_def._crossing_above_rule = False
    (z_a, i_a), (z_b, i_b) = s_off.compute_impedance(), s_def.compute_impedance()
    assert z_a == z_b and np.array_equal(i_a, i_b)


def test_on_declines_where_it_would_fill_more_grid(monkeypatch, tmp_path, routes):
    """efhw_lo: its counterpoise pairs go on #1201's graded panels, so B1's
    reads cannot be censused, and the shipped census does not take the whole
    grid. B1 declines, and the printout is the off one but for line 2."""
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    deck = DECKS["efhw_lo"]
    monkeypatch.delenv(ENV, raising=False)
    off = _render(deck, "nec5")
    monkeypatch.setenv(ENV, "1")
    on = _render(deck, "nec5")
    assert routes["b1"] == 0
    assert on.splitlines()[2:] == off.splitlines()[2:]


def test_on_declines_a_small_above_set(monkeypatch, tmp_path, routes):
    """The hub's 16 above segments are under `_B1_MIN_ABOVE_SEGS`: B1 declines
    and the printout is the off one but for line 2."""
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    deck = DECKS["hub"]
    monkeypatch.delenv(ENV, raising=False)
    off = _render(deck, "nec5")
    monkeypatch.setenv(ENV, "1")
    on = _render(deck, "nec5")
    assert routes["b1"] == 0
    assert on.splitlines()[2:] == off.splitlines()[2:]


def test_the_lazy_view_holds_every_node_b1_reads(monkeypatch, tmp_path, routes):
    """B1 on the view filled at its own census reads what it reads on the
    whole grid, bit for bit: an unfilled node reads NaN, so a census that
    missed one would show."""
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(_below_interface, "_B1_MIN_ABOVE_SEGS", 0)
    deck = DECKS["hub"]
    seen = []
    real = _below_interface.crossing_above_grid

    def spy(somm_grid_fn, *a, **kw):
        out = real(somm_grid_fn, *a, **kw)
        seen.append(getattr(out, "_lazy_src", None) is not None)
        return out

    monkeypatch.setattr(_below_interface, "crossing_above_grid", spy)
    lazy = _render(deck, "nec5")
    assert seen == [True]

    def whole(somm_grid_fn, eps_t, r1_max, *a, **kw):
        return somm_grid_fn(eps_t, r1_max)

    monkeypatch.setattr(_below_interface, "crossing_above_grid", whole)
    assert _render(deck, "nec5") == lazy
    assert routes["b1"] == 2


def test_a_deck_with_no_crossing_is_untouched(monkeypatch, tmp_path, routes):
    # A buried dipole and an above-ground one, no wire through the plane:
    # the transmitted route, which B1 does not change.
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    deck = """CM a buried dipole under an elevated one
CE
GW 1 11 -2 0 -0.3 2 0 -0.3 0.001
GW 2 11 -2 0 1.0 2 0 1.0 0.001
GE -1
GN 2 0 0 0 13 0.005
EX 0 2 6 0 1 0
FR 0 1 0 0 30 0
XQ
EN
"""
    monkeypatch.delenv(ENV, raising=False)
    off = _render(deck, "nec5")
    monkeypatch.setenv(ENV, "1")
    on = _render(deck, "nec5")
    assert routes["b1"] == 0
    assert on.splitlines()[2:] == off.splitlines()[2:]


@pytest.mark.filterwarnings("ignore::momwire._crossing_fill.CoarseCrossingNode")
def test_restricted_rows_are_the_full_fills_rows(monkeypatch, tmp_path, routes):
    """The shipped contract (`test_crossing_sparse_1109`'s p2b_4), on: the
    requested rows to the fill's 1e-12, every other row exactly zero."""
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_below_interface, "_B1_MIN_ABOVE_SEGS", 0)
    s = BSplineSolver(**crossing_deck(1), crossing_above_rule=True)
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    full = s._compute_Z_operator_buried(geom, supp_seg, polys)
    above = _whole_wire_segments(s, geom, [1])  # the ABOVE wire
    part = s._compute_Z_operator_buried(geom, supp_seg, polys, rows=above)
    assert routes["b1"] == 2
    R = _below_interface._crossing_basis_rows(supp_seg, polys, above)
    other = np.setdiff1d(np.arange(supp_seg.shape[0]), R)
    assert np.all(part[other] == 0.0)
    rel = np.max(np.abs(part[R] - full[R])) / np.max(np.abs(full[R]))
    assert rel <= 1e-12, rel


@pytest.mark.parametrize("dialect", ["nec5", "nec4"])
@pytest.mark.parametrize(
    "name",
    [
        "efhw_lo",
        "hub",
        pytest.param("beverage", marks=pytest.mark.slow),
        pytest.param("efhw_hi", marks=pytest.mark.slow),
        pytest.param("aca", marks=pytest.mark.slow),
    ],
)
def test_on_zin_agrees_with_off(monkeypatch, tmp_path, name, dialect):
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    if name == "aca" and dialect == "nec4":
        pytest.skip("a NEC-5 deck: the NEC-4.2 slot does not solve it")
    deck = DECKS[name]
    monkeypatch.delenv(ENV, raising=False)
    z_off = _zin(_render(deck, dialect))
    monkeypatch.setenv(ENV, "1")
    z_on = _zin(_render(deck, dialect))
    assert abs(z_on - z_off) <= ZIN_RTOL * abs(z_off), (name, dialect, z_off, z_on)
