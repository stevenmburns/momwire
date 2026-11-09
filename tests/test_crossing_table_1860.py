"""The crossing table (momwire#1860 L5, the opt-in feature `crossing_table`).

A fill's crossing rows (source and observer on opposite sides of the
interface) inside a fixed wavelength-scaled grid are read from ONE table per
frequency instead of the near-interface column twin: the approach of the
NEC-4 theory manual sec. 5.1.3, at a finer grid (lambda0/100) and a higher
order (tricubic), with the singular part taken out in closed form
(`_crossing_table_inline.h`) and the remainder stored times R.

What is held:

- off, nothing changes: the feature is off by default, no scope is set, and
  a crossing solve never reaches the table;
- the C++ build is the column twin at the nodes plus the closed form, and
  the C++ read is its numpy walk, each to rounding;
- the closed form takes out what is singular at the corner: R (f - S)
  shrinks with R there for all six kernels;
- a call takes the table only on its own work inside the extents against
  the build's (the census), and the decks where taking it did not pay keep
  the twin;
- on, the printed input impedance of four buried decks stays within the
  accuracy class measured for it (the tolerances below), and the
  input admittance seen at every basis function moves <= 2e-3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momwire import _features, _ground_refl, _lapack
from momwire import _near_interface as ni
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

from test_crossing_serve_524 import crossing_deck

pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        ni._nia is None or not hasattr(ni._nia, "crossing_table_1860"),
        reason="the near-interface accelerator is not built",
    ),
]

FIX = Path(__file__).parent / "fixtures"


def _efhw(n_vert, n_rod, n_top, n_cp, label):
    """The synthetic EFHW-like inverted L of momwire#1860 (not a user's
    model): a 3 m rod under a 10 m vertical, a 30 m top, and a 4 m
    counterpoise at 1 cm, over average ground at 3.6 MHz. Kept as text here,
    not under tests/fixtures, which the C++ driver tests sweep whole."""
    return f"""CM EFHW-like inverted L, buried 3 m rod, 4 m counterpoise at 1 cm, GN 2 average ground
CM {label}
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


# The rod Beverage of momwire#1187 (tests/beverage_deck_1187.py's geometry)
# with a 500 ohm termination, as the momwire#1860 bench runs it.
BEVERAGE = """CM Rod Beverage (momwire#1187 geometry): 245.73 m at 2.5 m, 1.8 m rods, average ground
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
DECKS = {
    "efhw_lo": _efhw(30, 9, 90, 12, "efhw_lo 141 segs"),
    "efhw_hi": _efhw(120, 38, 380, 52, "efhw_hi 590 segs"),
    "d3_hub": None,  # tests/fixtures/near_buried_1653/d3_hub.nec5.nec
    "d4_slant": None,  # tests/fixtures/near_buried_1653/d4_slant.nec5.nec
    "beverage": BEVERAGE,
}


def _deck_text(name):
    if name == "d3_hub":
        return (FIX / "near_buried_1653" / "d3_hub.nec5.nec").read_text()
    if name.startswith("d4_slant"):
        # The fixture's NE/NH cards are dropped (the accuracy study ran it
        # without them) and an XQ is added; "d4_slant@0.97" scales its 7 MHz.
        text = (FIX / "near_buried_1653" / "d4_slant.nec5.nec").read_text()
        lines = [x for x in text.splitlines() if not x.startswith(("NE", "NH"))]
        scale = float(name.partition("@")[2] or 1.0)
        fr = f"FR 0 1 0 0 {7.0 * scale:.6f} 0"
        out = [fr if x.startswith("FR") else x for x in lines if x != "EN"]
        return "\n".join([*out, "XQ", "EN"]) + "\n"
    return DECKS[name]


C0 = 299792458.0
F = 3.6e6
K_P = 2.0 * np.pi * F / C0
SOIL = _ground_refl.eps_tilde((13.0, 0.005), 2.0 * np.pi * F, 8.8541878128e-12)
K_M = ni.k_medium(complex(SOIL), K_P)


def _twin(rho, z, zp):
    """The column twin at members (one column per distinct rho, in order)."""
    rho = np.asarray(rho, float)
    order = np.argsort(rho, kind="stable")
    r = rho[order]
    u, cnt = np.unique(r, return_counts=True)
    off = np.zeros(u.size + 1, dtype=np.intp)
    off[1:] = np.cumsum(cnt)
    v = np.asarray(
        ni._nia.near_interface_six_columns(
            K_P,
            K_M,
            u,
            off,
            np.ascontiguousarray(np.asarray(z, float)[order]),
            np.ascontiguousarray(np.asarray(zp, float)[order]),
            ni._LAM_MULT,
            ni._COLUMN_P,
            ni._DETOUR,
            1,
            ni._GX,
            ni._GW,
            grid=True,
        )
    )
    out = np.empty_like(v)
    out[order] = v
    return out


def _small_grid():
    h, _n = ni.crossing_table_grid(K_P)
    return h, np.array([5, 6, 4], dtype=np.intp)


def _build(h, n):
    return np.asarray(
        ni._nia.crossing_table_build(
            K_P, K_M, h, n, ni._LAM_MULT, ni._COLUMN_P, ni._DETOUR, 1, ni._GX, ni._GW
        )
    )


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_CROSSING_TABLE", raising=False)
    assert not _features.enabled("crossing_table")
    assert not BSplineSolver(**crossing_deck())._crossing_table
    assert not RazorSolver(**crossing_deck())._crossing_table
    assert ni._CROSSING_TABLE.get() is None


def test_the_grid_is_fixed_in_wavelengths():
    h, n = ni.crossing_table_grid(K_P)
    lam0 = 2.0 * np.pi / K_P
    assert list(n) == [51, 26, 13]
    assert np.allclose(h / lam0, [0.01, 0.01, 0.005])


def test_the_build_is_the_twin_plus_the_closed_form():
    h, n = _small_grid()
    vals = _build(h, n)
    i, j, k = np.meshgrid(*(np.arange(m) for m in n), indexing="ij")
    rho, z, zp = i.ravel() * h[0], j.ravel() * h[1], -k.ravel() * h[2]
    live = np.hypot(rho, z - zp) > 0
    f = _twin(rho[live], z[live], zp[live])
    R = np.hypot(rho[live], z[live] - zp[live])
    ref = (f - ni.crossing_singular(rho[live], z[live], zp[live], K_P, K_M)) * R[
        :, None
    ]
    got = vals.reshape(-1, 6)[live]
    scale = np.abs(f * R[:, None]).max(axis=1)
    assert np.max(np.abs(got - ref).max(axis=1) / scale) < 1e-12
    assert np.all(vals.reshape(-1, 6)[~live] == 0)


def test_the_read_is_its_numpy_walk():
    h, n = _small_grid()
    vals = _build(h, n)
    rng = np.random.default_rng(1860)
    m = 500
    lim = h * (n - 1)
    rho = rng.uniform(0.0, lim[0], m)
    z = rng.uniform(0.0, lim[1], m)
    zp = -rng.uniform(0.0, lim[2], m)
    rho[:3] = [1e-3, 0.0, lim[0]]  # the corner's column, a node, the edge
    z[:3] = [0.0, 0.5 * h[1], lim[1]]
    zp[:3] = [0.0, -h[2], -lim[2]]
    got = np.asarray(ni._nia.crossing_table_lookup(K_P, K_M, h, n, vals, rho, z, zp, 1))
    ref = ni.crossing_table_lookup_numpy(K_P, K_M, h, n, vals, rho, z, zp)
    assert np.max(np.abs(got - ref).max(axis=1) / np.abs(ref).max(axis=1)) < 1e-13
    with pytest.raises(ValueError):
        ni._nia.crossing_table_lookup(
            K_P, K_M, h, n, vals, np.array([0.0]), np.array([0.0]), np.array([0.0]), 1
        )


def test_the_closed_form_takes_out_the_corner():
    # R (f - S) goes to zero linearly at the corner for all six kernels: the
    # leading and next orders are both out (without the next order, dzpV's
    # stays O(1) on its own scale).
    rems = []
    for R in (1e-2, 1e-3):
        rho, z, zp = np.array([R / 2]), np.array([R / 2]), np.array([-R / 3])
        Rr = np.hypot(rho, z - zp)
        f = _twin(rho, z, zp)
        rems.append(
            (
                np.abs(f - ni.crossing_singular(rho, z, zp, K_P, K_M)) * Rr[:, None]
            ).ravel()
        )
    assert np.all(rems[1] < 0.2 * rems[0])


def test_a_small_fill_declines_the_table(monkeypatch):
    built = []
    real = ni._nia.crossing_table_build
    monkeypatch.setattr(
        ni._nia,
        "crossing_table_build",
        lambda *a, **k: built.append(1) or real(*a, **k),
    )
    with ni.crossing_table_scope(True):
        rho = np.array([0.5])
        off = np.array([0, 3], dtype=np.intp)
        z = np.array([0.1, 0.2, 0.3])
        zp = np.array([-0.1, -0.2, -0.3])
        got = ni._twin_columns(K_P, K_M, rho, off, z, zp, ni._LAM_MULT)
        assert not ni._CROSSING_TABLE.get().taken
    assert not built
    assert np.array_equal(got, _twin(np.repeat(rho, 3), z, zp))


def _printout(deck, basis, on, monkeypatch):
    from momwire.dropin.nec5 import _shell

    if on:
        monkeypatch.setenv("MOMWIRE_X_CROSSING_TABLE", "1")
    else:
        monkeypatch.delenv("MOMWIRE_X_CROSSING_TABLE", raising=False)
    Z = []
    real = _lapack.lu_factor

    def keep(a, *x, **k):
        Z.append(np.array(a, copy=True))
        return real(a, *x, **k)

    monkeypatch.setattr(_lapack, "lu_factor", keep)
    ni._CROSSING_TABLE_STATS["table_rows"] = 0
    # Every fill takes the table here (no build weight), so that its accuracy
    # is what is measured; whether a fill SHOULD is the next test's.
    monkeypatch.setattr(ni, "_CROSSING_TABLE_SETUP", 0.0)
    monkeypatch.setattr(ni, "_CROSSING_TABLE_NODE", 0.0)
    # Level 1, as antennaknobs runs it: the drop-in context is level 4, where
    # a reach-1 native feature refuses.
    text = _shell.render.__wrapped__(_deck_text(deck), basis=basis, dialect="nec5")
    monkeypatch.setattr(_lapack, "lu_factor", real)
    monkeypatch.undo()
    return text, Z[0], ni._CROSSING_TABLE_STATS["table_rows"]


def _zin(text):
    lines = text[text.index("ANTENNA INPUT PARAMETERS") :].splitlines()
    for line in lines:
        f = line.split()
        if len(f) >= 11 and f[0].isdigit() and f[1].isdigit():
            return complex(float(f[7]), float(f[8]))
    raise AssertionError("no input impedance in the printout")


# Printed Zin (5 significant figures) against the exact route, |dZin| / |Zin|.
# Measured on Skylake (momwire#1860 L5, both engines; Z from this module's
# route), then x3 for margin and a printed last digit:
#   efhw_lo 1.7e-5, efhw_hi 1.9e-5, d3_hub 9.4e-4, Beverage 8.4e-5
# d4_slant (momwire#1860 accuracy study, v0.89.0, razor-2p only) 4.8-7.3e-4
# at f0 x 0.97 / 1.00 / 1.03 (re-measured here: 4.9, 6.3, 7.2e-4; ~0.035 ohm),
# then x4: 3e-3. Its 6,037 table rows are what the census takes.
# (diag(Z^-1) at most 1.05e-3).
CASES = [
    ("efhw_lo", 1e-4),
    pytest.param("efhw_hi", 1e-4, marks=pytest.mark.slow),
    pytest.param("d3_hub", 3e-3, marks=pytest.mark.slow),
    pytest.param("beverage", 3e-4, marks=pytest.mark.slow),
]


@pytest.mark.skipif(
    not _features.certified("crossing_table"),
    reason="crossing_table is not certified on this platform",
)
@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
@pytest.mark.parametrize(("deck", "tol"), CASES)
def test_on_the_printed_zin_stays_in_its_class(monkeypatch, deck, tol, basis):
    off, z_off, _ = _printout(deck, basis, False, monkeypatch)
    on, z_on, served = _printout(deck, basis, True, monkeypatch)
    assert served > 0  # the table ran
    a, b = _zin(off), _zin(on)
    assert abs(b - a) <= tol * abs(a), (a, b)
    ya = np.diag(np.linalg.inv(z_off))
    yb = np.diag(np.linalg.inv(z_on))
    assert np.max(np.abs(yb - ya) / np.abs(ya)) <= 2e-3


@pytest.mark.skipif(
    not _features.certified("crossing_table"),
    reason="crossing_table is not certified on this platform",
)
@pytest.mark.parametrize("scale", ["0.97", "1.00", "1.03"])
def test_d4_slant_takes_the_table_and_stays_in_its_class(monkeypatch, scale):
    # The census takes the table here (6,037 rows at the deck's own
    # frequency) under nec5-razor-2p; no other test covers this deck.
    deck = f"d4_slant@{scale}"
    off, _, _ = _printout(deck, "razor-2p", False, monkeypatch)
    on, _, served = _printout(deck, "razor-2p", True, monkeypatch)
    assert served > 0
    a, b = _zin(off), _zin(on)
    assert abs(b - a) <= 3e-3 * abs(a), (a, b)


@pytest.mark.skipif(
    not _features.certified("crossing_table"),
    reason="crossing_table is not certified on this platform",
)
@pytest.mark.parametrize(
    ("deck", "taken"),
    [
        # Measured (Skylake, T1, factored_columns + height_sheet_recal on):
        # efhw_hi's main crossing call alone outweighs the build, and its
        # fill drops 1179 -> ~935 ms; no single call of the other three does,
        # and taking the table there measured slower (efhw_lo +9 ms,
        # aca_vertical_screen under nec5-bspline +42 ms) or no faster.
        pytest.param("efhw_hi", True, marks=pytest.mark.slow),
        ("efhw_lo", False),
        ("d3_hub", False),
        ("beverage", False),
        ("d4_slant", True),
    ],
)
def test_the_census_takes_the_table_where_it_pays(monkeypatch, deck, taken):
    from momwire.dropin.nec5 import _shell

    monkeypatch.setenv("MOMWIRE_X_CROSSING_TABLE", "1")
    ni._CROSSING_TABLE_STATS["table_rows"] = 0
    _shell.render.__wrapped__(_deck_text(deck), basis="razor-2p", dialect="nec5")
    assert (ni._CROSSING_TABLE_STATS["table_rows"] > 0) is taken
