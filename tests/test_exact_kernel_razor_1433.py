"""The exact ring kernel on razor (momwire#1433).

Razor's rows are path-tested, so the correction is the kernel layer's 1-D
point moments assembled by the fill's own wing algebra
(`RazorSolver._add_exact_kernel_correction`; derivation in
`scratch/1433/DERIVATION.md`). These hold: the primitive against adaptive
quadrature, the C++ route against numpy, the mechanism (hand the correction
the fill's own kernel and Z must not move), the option's semantics (off and
unengaged "auto" are bit-identical and never enter the correction), the
refusals, and the advisories.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from momwire import RazorSolver, _exact_kernel as EK, _wire_spec
from momwire import razor as RZ

L = 0.47
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]
BENT = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])]
JUNCTION = [
    np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.2)]),
    np.array([(0.0, 0.0, 0.2), (0.0, 0.0, 0.3), (0.1, 0.0, 0.4)]),
]
_DECKS = {
    "dipole": dict(wires=DIPOLE, n_per_edge_per_wire=[[37]]),
    "bent": dict(wires=BENT, n_per_edge_per_wire=[[20, 12]]),
    "junction": dict(wires=JUNCTION, n_per_edge_per_wire=[[19], [9, 13]]),
    "pec monopole": dict(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25)])],
        n_per_edge_per_wire=[[30]],
        ground_z=0.0,
        feeds=[(0, 0.0, 1.0)],
    ),
}


def _solver(exact=False, a=0.00425, **kw):
    kw = {"wires": DIPOLE, "n_per_edge_per_wire": [[37]], **kw}
    kw.setdefault("nec5_quadrature", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RazorSolver(wavelength=1.0, wire_radius=a, exact_kernel=exact, **kw)


def _z(s):
    return np.asarray(s.compute_impedance()[0])


def _reduced(z, a, k):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    return np.exp(-1j * k * R) / (4 * np.pi * R)


def _extended(z, a, k):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    kr = k * R
    c1 = 1 + 1j * kr
    c2 = 3 * c1 - kr * kr
    fac = 1 + 0.25 * a**4 / R**4 * c2 - 0.5 * a * a / R**2 * c1
    return np.exp(-1j * kr) / (4 * np.pi * R) * fac


# -- the primitive ------------------------------------------------------------


@pytest.mark.parametrize(
    ("c", "h"),
    [
        (-0.001, 0.002),  # observer at the centre (razor-2p's self term)
        (-0.0015, 0.002),  # inside, off centre
        (0.0, 0.002),  # at the start
        (-0.002, 0.002),  # at the end
        (0.0005, 0.001),  # adjacent, within 2a
        (-0.03, 0.006),  # far, behind
        (-0.0001, 0.02),  # long segment, observer near its start
    ],
)
def test_point_moments_match_adaptive_quadrature(c, h):
    a, k = 0.00425, 2 * np.pi

    def ref(q):
        pts = [-c] if 0 < -c < h else None

        def f(v, part):
            return getattr(EK.ring_kernel(np.array([c + v]), a, k)[0] * v**q, part)

        kw = dict(points=pts, limit=400, epsabs=0, epsrel=1e-13)
        return (
            quad(f, 0, h, args=("real",), **kw)[0]
            + 1j * quad(f, 0, h, args=("imag",), **kw)[0]
        )

    got = EK.point_moments(c, h, a, k, 3)
    want = np.array([ref(q) for q in range(3)])
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-12


@pytest.mark.skipif(not EK._HAVE_POINT_ACCEL, reason="accelerator not built")
@pytest.mark.parametrize("nq", [2, 4])
def test_point_rows_accel_matches_numpy(nq):
    """Forward and backward segments, graded lengths, observers inside, at
    ends and beyond: the C++ route (mirror-keyed dedup) agrees with numpy to
    1e-12 of the block's largest entry -- the kernel's own 3e-12 relative
    accuracy, of which these routes share the nodes."""
    rng = np.random.default_rng(1)
    a, k = 0.008, 2 * np.pi / 21.4
    h = np.concatenate([np.full(30, 0.005), rng.uniform(0.002, 0.03, 20)])
    sgn = np.where(rng.random(50) < 0.3, -1.0, 1.0)
    start = np.concatenate([[0.0], np.cumsum(h)[:-1]])
    x0 = np.where(sgn > 0, start, start + h)
    xo = np.concatenate([start + 0.5 * h, start, rng.uniform(-0.05, 0.6, 20)])
    A = EK.PointRowsAccel(x0, sgn, h, a, k, nq).rows(xo)
    B = EK.PointRows(x0, sgn, h, a, k, nq).rows(xo)
    assert np.max(np.abs(A - B)) <= 1e-12 * np.max(np.abs(B))


@pytest.mark.skipif(not EK._HAVE_POINT_ACCEL, reason="accelerator not built")
def test_point_rows_dedup_is_linear_on_a_uniform_run():
    n = 400
    x0 = np.arange(n) * 0.006
    R = EK.PointRowsAccel(x0, np.ones(n), np.full(n, 0.006), 0.008, 0.3, 2)
    R.rows(x0 + 0.003)
    assert R.evaluated <= n


# -- the mechanism ------------------------------------------------------------


@pytest.mark.parametrize("nec5", [True, False])
@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("deck", sorted(_DECKS))
def test_swapped_kernel_reproduces_the_fill(monkeypatch, deck, ek, nec5):
    """Hand the correction the fill's OWN kernel: Z must not move beyond the
    fill's quadrature accuracy (measured <= 5.4e-9, the EK bend)."""
    kern = _extended if ek else _reduced
    common = dict(_DECKS[deck], extended_kernel=ek, nec5_quadrature=nec5)
    if not nec5:
        common["n_qp_path"] = 8
    base = _z(_solver(**common))
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(EK, "ring_kernel", kern)
    monkeypatch.setattr(
        EK,
        "ring_kernel_split",
        lambda z, a, k: (np.zeros(np.shape(z), complex), kern(z, a, k)),
    )
    calls = []
    orig = RazorSolver._add_exact_kernel_correction

    def spy(self, *args):
        calls.append(1)
        return orig(self, *args)

    monkeypatch.setattr(RazorSolver, "_add_exact_kernel_correction", spy)
    swapped = _z(_solver(exact=True, **common))
    assert calls, "the correction did not run"
    assert np.max(np.abs(swapped - base)) <= 1e-8 * np.max(np.abs(base))


def test_swapped_kernel_detects_a_wrong_kernel(monkeypatch):
    base = _z(_solver())
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(EK, "ring_kernel", lambda z, a, k: _reduced(z, a, k) * 1.000001)
    monkeypatch.setattr(
        EK,
        "ring_kernel_split",
        lambda z, a, k: (np.zeros(np.shape(z), complex), _reduced(z, a, k) * 1.000001),
    )
    moved = _z(_solver(exact=True))
    assert abs(moved - base) > 1e-8 * abs(base)


@pytest.mark.parametrize("deck", ["dipole", "bent", "junction"])
@pytest.mark.parametrize("nec5", [True, False])
def test_accel_matches_numpy(monkeypatch, deck, nec5):
    """The C++ point moments and assembly against the numpy route, at the
    1e-12 the kernel holds (measured <= 1.4e-13)."""
    if not (EK._HAVE_POINT_ACCEL and RZ._HAVE_EXACT_POINT_ASSEMBLE_ACCEL):
        pytest.skip("accelerator not built")
    common = dict(_DECKS[deck], extended_kernel=True, nec5_quadrature=nec5)
    if not nec5:
        common["n_qp_path"] = 8
    fast = _z(_solver(exact=True, **common))
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(RZ, "_HAVE_EXACT_POINT_ASSEMBLE_ACCEL", False)
    slow = _z(_solver(exact=True, **common))
    assert np.max(np.abs(fast - slow)) <= 1e-12 * np.max(np.abs(slow))


# -- the option -----------------------------------------------------------------


@pytest.mark.parametrize("ek", [False, True])
def test_default_off_is_untouched(monkeypatch, ek):
    def boom(*_a, **_k):
        raise AssertionError("exact-kernel code ran with exact_kernel=False")

    monkeypatch.setattr(RazorSolver, "_add_exact_kernel_correction", boom)
    monkeypatch.setattr(EK, "point_rows", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s0 = RazorSolver(
            wires=DIPOLE,
            n_per_edge_per_wire=[[19]],
            wavelength=1.0,
            wire_radius=0.00425,
            extended_kernel=ek,
            nec5_quadrature=True,
        )
    z0 = _z(s0)
    z1 = _z(_solver(exact=False, n_per_edge_per_wire=[[19]], extended_kernel=ek))
    assert np.array_equal(z0, z1)


def test_capability_is_declared():
    assert "auto" in RazorSolver.EXACT_KERNEL_CHOICES
    assert RazorSolver.EXACT_KERNEL_AUTO_H_OVER_A == 3.0
    assert RazorSolver.EXACT_KERNEL_FIXED_GAP_FEED is False


@pytest.mark.parametrize(
    ("a", "engaged"),
    [
        (0.0005, False),  # thin: h/a 49
        (0.00425, False),  # n=19: h/a 5.8
        (L / 19 / 3.03, False),
        (L / 19 / 2.97, True),
        (0.0125, True),
    ],
)
def test_auto_threshold_and_bits(monkeypatch, a, engaged):
    """ "auto" resolves to the h/a < 3 gate and reproduces whichever of True
    or False it resolved to, bit for bit; unengaged it never enters the
    correction."""
    kw = dict(n_per_edge_per_wire=[[19]], a=a, extended_kernel=True)
    s = _solver(exact="auto", **kw)
    assert s.exact_kernel_requested == "auto"
    assert s.exact_kernel is engaged
    if not engaged:

        def boom(*_a, **_k):
            raise AssertionError("correction ran under an unengaged auto")

        monkeypatch.setattr(RazorSolver, "_add_exact_kernel_correction", boom)
    z_auto = _z(s)
    z_ref = _z(_solver(exact=engaged, **kw))
    assert np.array_equal(z_auto, z_ref)


BURIED = dict(
    wires=[np.array([(-L / 2, 0.0, -0.3), (L / 2, 0.0, -0.3)])],
    n_per_edge_per_wire=[[19]],
    ground_z=0.0,
    ground_eps=(13.0, 0.005),
    ground_model="sommerfeld",
    a=0.0125,
)


def test_buried_refused_under_true_and_silent_under_auto():
    s = _solver(exact=True, **BURIED)
    with pytest.raises(NotImplementedError, match="buried wires"):
        s.compute_impedance()
    assert _solver(exact="auto", **BURIED).exact_kernel is False


def test_restricted_rows_refused_under_true():
    s = _solver(exact=True, a=0.0125, **_DECKS["junction"])
    geom = s._build_geometry()
    with pytest.raises(NotImplementedError, match="dense fill only"):
        s._compute_Z_operator(geom, rows=np.arange(19))


def test_swept_matches_single_solves():
    s = _solver(exact=True, a=0.0125)
    ks = np.array([5.9, 6.28])
    swept = np.asarray(s.compute_impedance_swept(ks))
    single = []
    for k in ks:
        si = _solver(exact=True, a=0.0125)
        si.k = k
        single.append(np.asarray(si.compute_impedance()[0]).ravel()[0])
    assert np.allclose(np.ravel(swept), single, rtol=1e-12, atol=0)


# -- the advisories ------------------------------------------------------------


def _warnings_of(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        RazorSolver(wavelength=1.0, **kw)
    return w


FAT = dict(wires=DIPOLE, n_per_edge_per_wire=[[401]], wire_radius=0.01)


def test_short_segments_silent_under_the_exact_kernel():
    def short(w):
        return [x for x in w if x.category is _wire_spec.ShortSegments]

    assert short(_warnings_of(**FAT))
    assert not short(_warnings_of(exact_kernel="auto", **FAT))


def test_gap_floor_states_the_knot_drift():
    def gap(w):
        return [str(x.message) for x in w if x.category is _wire_spec.GapMeshFloor]

    (ran,) = gap(_warnings_of(exact_kernel="auto", extended_kernel=True, **FAT))
    assert "zero-width knot gaps" in ran and "bspline's end-port feed" in ran
    (could,) = gap(_warnings_of(extended_kernel=True, **FAT))
    assert "removes the collapse" in could


def test_point_gap_warns_under_true_only():
    def rt(w):
        return [str(x.message) for x in w if x.category is RuntimeWarning]

    assert any(
        "no fixed-width gap" in m for m in rt(_warnings_of(exact_kernel=True, **FAT))
    )
    assert not rt(_warnings_of(exact_kernel="auto", **FAT))


# -- the roster: razor-2p binds "auto" through the NEC-5 seam ------------------

_SEAM_DECK = (
    "CM synthetic fat-wire dipole (momwire#1433)\nCE\n"
    "GW 1,12,0.,-.25,0.,0.,.25,0.,{a}\nGE 0,-1\n"
    "FR 0,1,0,0,299.7925\nGN -1\nEX 4,1,6,0,1.414214,0.\nPQ 0\n"
    "RP 0,1,37,1000,90.,0.,0.,10.,0.\nEN\n"
)


def _seam_counted(monkeypatch):
    seen = []
    orig = RazorSolver._add_exact_kernel_correction

    def counted(self, *args):
        seen.append(1)
        return orig(self, *args)

    monkeypatch.setattr(RazorSolver, "_add_exact_kernel_correction", counted)
    return seen


@pytest.mark.parametrize("basis", ["razor-2p", "razor-nec5"])
def test_the_seam_runs_the_exact_kernel_on_a_fat_wire(monkeypatch, basis):
    from momwire.eznec import _shell

    seen = _seam_counted(monkeypatch)
    _shell.render(_SEAM_DECK.format(a="0.02"), basis=basis)  # h/a 2.08
    assert seen == [1]


@pytest.mark.parametrize("a", ["0.0005", "0.013"])  # h/a 83 and 3.21
def test_the_seam_prints_the_same_bytes_above_the_gate(monkeypatch, a):
    from types import MappingProxyType

    from momwire.deck import _solver
    from momwire.eznec import _shell

    seen = _seam_counted(monkeypatch)
    ours = _shell.render(_SEAM_DECK.format(a=a), basis="razor-2p")
    assert seen == []
    monkeypatch.setattr(
        _solver,
        "BASES",
        MappingProxyType(
            {
                **_solver.BASES,
                "razor-2p": (RazorSolver, MappingProxyType({"nec5_quadrature": True})),
            }
        ),
    )
    assert _shell.render(_SEAM_DECK.format(a=a), basis="razor-2p") == ours
