"""Razor's exact-kernel correction cache (momwire#1454).

`_razor_exact_plan` keeps the correction's geometry-only tables across calls
and, per wavenumber, each group's C++ point-moment source with its pair
cache. These hold that a hit is EXACT (Z to the bit against the uncached
route), that the cache really serves (the counters move), and that every
input the result depends on -- geometry, radius, frequency, ground, basis
flags, the kernel's rules, the route -- misses rather than meeting a stale
entry. The byte gate over the fixture decks lives in the PR (#1454).
"""

import threading
import warnings

import numpy as np
import pytest

from momwire import RazorSolver, _exact_kernel as EK, _razor_exact_plan as P

needs_accel = pytest.mark.skipif(
    not EK._HAVE_POINT_ACCEL, reason="accelerator not built"
)

L = 0.47


def _dipole(length=L):
    return [np.array([(0.0, 0.0, -length / 2), (0.0, 0.0, length / 2)])]


BENT = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])]


def _solver(wires=None, npe=None, a=0.0125, wavelength=1.0, **kw):
    kw.setdefault("nec5_quadrature", True)
    kw.setdefault("extended_kernel", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RazorSolver(
            wires=_dipole() if wires is None else wires,
            n_per_edge_per_wire=[[37]] if npe is None else npe,
            wavelength=wavelength,
            wire_radius=a,
            exact_kernel=True,
            **kw,
        )


def _z(**kw):
    """The whole impedance matrix, so a single moved entry shows."""
    s = _solver(**kw)
    return np.asarray(s._assemble_Z(s._build_geometry(), s.k))


def _fresh(**kw):
    """Z through the uncached route (the plan rebuilt, the sources new)."""
    saved = P.ENABLED
    P.ENABLED = False
    try:
        return _z(**kw)
    finally:
        P.ENABLED = saved


@pytest.fixture(autouse=True)
def _clean_cache():
    P.clear()
    yield
    P.clear()


def _same_bits(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


@needs_accel
@pytest.mark.parametrize("nec5", [True, False])
def test_a_repeat_call_hits_and_adds_the_same_bits(nec5):
    kw = dict(nec5_quadrature=nec5, wires=BENT, npe=[[20, 12]])
    if not nec5:
        kw["n_qp_path"] = 8
    want = _fresh(**kw)
    first = _z(**kw)
    assert P.STATS == {"plan_hit": 0, "plan_miss": 1, "rows_hit": 0, "rows_miss": 1}
    again = _z(**kw)
    assert P.STATS["plan_hit"] == 1 and P.STATS["rows_hit"] == 1
    assert _same_bits(first, want) and _same_bits(again, want)


@needs_accel
def test_a_geometry_change_misses():
    _z()
    _z(wires=_dipole(0.48))
    assert P.STATS["plan_hit"] == 0 and P.STATS["plan_miss"] == 2
    assert _same_bits(_z(wires=_dipole(0.48)), _fresh(wires=_dipole(0.48)))


@needs_accel
def test_a_mesh_change_misses():
    _z()
    got = _z(npe=[[39]])
    assert P.STATS["plan_hit"] == 0 and P.STATS["plan_miss"] == 2
    assert _same_bits(got, _fresh(npe=[[39]]))


@needs_accel
def test_a_radius_change_misses():
    _z()
    got = _z(a=0.012)
    assert P.STATS["plan_hit"] == 0 and P.STATS["plan_miss"] == 2
    assert _same_bits(got, _fresh(a=0.012))


@needs_accel
def test_a_frequency_change_reuses_the_geometry_but_not_the_moments():
    _z()
    got = _z(wavelength=1.1)
    assert P.STATS["plan_hit"] == 1, "the geometry-only half should be shared"
    assert P.STATS["rows_hit"] == 0 and P.STATS["rows_miss"] == 2
    assert _same_bits(got, _fresh(wavelength=1.1))
    # ...and the first frequency's moments are still there to be found.
    assert _same_bits(_z(), _fresh())
    assert P.STATS["rows_hit"] == 1


@needs_accel
@pytest.mark.parametrize(
    "ground",
    [
        dict(ground_z=-0.5),
        dict(ground_z=-0.6),
        dict(ground_z=-0.5, ground_eps=(13.0, 0.005), ground_model="refl-coef"),
    ],
)
def test_a_ground_change_misses(ground):
    _z()
    got = _z(**ground)
    assert P.STATS["plan_hit"] == 0 and P.STATS["plan_miss"] == 2
    assert P.STATS["rows_hit"] == 0
    assert _same_bits(got, _fresh(**ground))


@needs_accel
@pytest.mark.parametrize(
    "change",
    [
        dict(nec5_quadrature=False, n_qp_path=8),
        dict(extended_kernel=False),
    ],
)
def test_a_basis_change_misses(change):
    _z()
    got = _z(**change)
    assert P.STATS["plan_hit"] == 0 and P.STATS["plan_miss"] == 2
    assert _same_bits(got, _fresh(**change))


@needs_accel
def test_the_kernel_rules_key_the_moments(monkeypatch):
    base = _z()
    monkeypatch.setattr(EK, "_N_GL", 12)
    moved = _z()
    assert P.STATS["plan_hit"] == 1 and P.STATS["rows_hit"] == 0
    assert _same_bits(moved, _fresh())
    assert not _same_bits(moved, base)


def test_the_numpy_route_is_never_kept(monkeypatch):
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    _z()
    _z()
    assert P.STATS["rows_hit"] == 0


def test_disabled_stores_nothing():
    saved = P.ENABLED
    P.ENABLED = False
    try:
        _z()
        _z()
    finally:
        P.ENABLED = saved
    assert P.STATS == {"plan_hit": 0, "plan_miss": 0, "rows_hit": 0, "rows_miss": 0}
    assert not P._PLANS


@needs_accel
def test_the_cache_is_bounded():
    for i in range(P._PLAN_ENTRIES + 2):
        _z(wires=_dipole(0.40 + 0.01 * i))
    assert len(P._PLANS) == P._PLAN_ENTRIES
    for i in range(P._ROWS_ENTRIES + 2):
        _z(wavelength=1.0 + 0.05 * i)
    plan = next(reversed(P._PLANS.values()))
    assert len(plan._rows) == P._ROWS_ENTRIES


@needs_accel
def test_concurrent_calls_add_the_same_bits():
    want = _fresh()
    _z()
    out = [None] * 4

    def run(i):
        out[i] = _z()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(_same_bits(z, want) for z in out)


@needs_accel
def test_the_seam_sequence_matches_fresh_printouts():
    """Through the EZNEC seam, as the resident server serves it: one deck,
    the same deck at a second frequency, a second geometry, then the first
    again -- every printout byte-identical to the uncached route's, and the
    repeats hit."""
    from momwire.eznec import _shell

    deck = (
        "CM fat-wire dipole (momwire#1454)\nCE\n"
        "GW 1,12,0.,-.25,0.,0.,.25,0.,{a}\nGE 0,-1\n"
        "FR 0,1,0,0,{f}\nGN -1\nEX 4,1,6,0,1.414214,0.\nPQ 0\n"
        "RP 0,1,37,1000,90.,0.,0.,10.,0.\nEN\n"
    )
    seq = [
        dict(a="0.02", f="299.7925"),
        dict(a="0.02", f="280.0"),
        dict(a="0.021", f="299.7925"),
        dict(a="0.02", f="299.7925"),
    ]

    def render(cfg):
        return _shell.render(deck.format(**cfg), basis="razor-2p")

    def strip(text):
        return "\n".join(x for x in text.splitlines() if "TIME" not in x)

    P.ENABLED = False
    try:
        want = [strip(render(c)) for c in seq]
    finally:
        P.ENABLED = True
    P.clear()
    got = [strip(render(c)) for c in seq]
    assert got == want
    assert P.STATS["plan_hit"] >= 2  # the second frequency and the repeat
    assert P.STATS["rows_hit"] >= 1  # the repeat
