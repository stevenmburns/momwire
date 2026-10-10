"""`exact_kernel="auto"` (momwire#1408 follow-up).

"auto" resolves at construction to True iff the route serves the exact ring
kernel and some segment is shorter than `EXACT_KERNEL_AUTO_H_OVER_A` (3)
radii. It must never raise where True would refuse, it must reproduce True's
or False's Z bit for bit (whichever it resolved to), and the solver must say
which ran (`exact_kernel`) next to what was asked (`exact_kernel_requested`).
"""

import math
import warnings

import numpy as np
import pytest

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver

L = 0.47
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]
# A bent fat wire: a straight run that turns 45 degrees half way.
BENT = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])]


def _build(cls=BSplineSolver, wires=DIPOLE, n=19, a=0.00425, **kw):
    npe = kw.pop("npe", [[n]] * len(wires))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(
            wires=wires,
            n_per_edge_per_wire=npe,
            wavelength=1.0,
            wire_radius=a,
            **kw,
        )


def _z(s):
    return np.asarray(s.compute_impedance()[0])


def _h_over_a_min(n, a):
    return (L / n) / a


def test_capability_is_declared():
    assert "auto" in BSplineSolver.EXACT_KERNEL_CHOICES
    assert True in BSplineSolver.EXACT_KERNEL_CHOICES
    assert BSplineSolver.EXACT_KERNEL_AUTO_H_OVER_A == 3.0


def test_other_strings_refuse():
    with pytest.raises(ValueError, match="'auto'"):
        _build(exact_kernel="on")


@pytest.mark.parametrize(
    ("a", "engaged"),
    [
        (0.0005, False),  # thin: h/a 49
        (0.00425, False),  # d/lambda 0.0085 at n=19: h/a 5.8
        # either side of the threshold at n=19: h/a = 3.03 and 2.97
        (L / 19 / 3.03, False),
        (L / 19 / 2.97, True),
        (0.0125, True),  # h/a 1.98
    ],
)
def test_threshold(a, engaged):
    s = _build(a=a, exact_kernel="auto")
    assert s.exact_kernel_requested == "auto"
    assert s.exact_kernel is engaged


def test_one_short_segment_is_enough():
    """A graded mesh whose only short segments sit at one end engages."""
    a = 0.004
    wires = [
        np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, -L / 2 + 0.02), (0.0, 0.0, L / 2)])
    ]
    s = _build(wires=wires, a=a, npe=[[2, 17]], exact_kernel="auto")
    # 0.01 m / 0.004 m = 2.5 radii on the first edge; 0.45/17/0.004 = 6.6 after
    assert s.exact_kernel is True
    s = _build(wires=wires, a=a, npe=[[1, 17]], exact_kernel="auto")
    assert s.exact_kernel is False  # 5 radii and 6.6 radii


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize(("a", "n"), [(0.00425, 19), (0.00425, 75)])
def test_auto_reproduces_what_it_resolved_to(a, n, ek):
    auto = _build(n=n, a=a, exact_kernel="auto", extended_kernel=ek)
    same = _build(n=n, a=a, exact_kernel=auto.exact_kernel, extended_kernel=ek)
    assert np.array_equal(_z(auto), _z(same))
    if auto.exact_kernel:
        off = _build(n=n, a=a, exact_kernel=False, extended_kernel=ek)
        assert not np.array_equal(_z(auto), _z(off))


def test_auto_engages_on_a_bent_fat_wire():
    """Bends are accepted (Steve 10-09): the coaxial runs on either side of a
    bend take the exact kernel, the cross-bend pairs keep EK (momwire#1413)."""
    s = _build(wires=BENT, npe=[[20, 12]], a=0.006, exact_kernel="auto")
    assert s.exact_kernel is True
    ref = _build(wires=BENT, npe=[[20, 12]], a=0.006, exact_kernel=True)
    assert np.array_equal(_z(s), _z(ref))


def test_point_gap_warns_only_when_asked_for():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        BSplineSolver(
            wires=DIPOLE,
            n_per_edge_per_wire=[[75]],
            wavelength=1.0,
            wire_radius=0.00425,
            exact_kernel="auto",
        )
    assert not [w for w in rec if "no limiting reactance" in str(w.message)]
    with pytest.warns(RuntimeWarning, match="no limiting reactance"):
        BSplineSolver(
            wires=DIPOLE,
            n_per_edge_per_wire=[[75]],
            wavelength=1.0,
            wire_radius=0.00425,
            exact_kernel=True,
        )


# --------------------------------------------------------------------------
# "auto" never raises where True would refuse
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [HMatrixSolver, ArrayBlockSolver])
def test_subclasses_resolve_off(cls):
    s = _build(cls=cls, n=75, exact_kernel="auto")
    assert s.exact_kernel is False
    with pytest.raises(NotImplementedError, match="1408"):
        _build(cls=cls, n=75, exact_kernel=True)


def test_enrichment_resolves_off():
    s = _build(n=75, exact_kernel="auto", use_singular_enrichment=True)
    assert s.exact_kernel is False
    with pytest.raises(NotImplementedError, match="1408"):
        _build(n=75, exact_kernel=True, use_singular_enrichment=True)


def _screen(**kw):
    """A buried 4-radial screen with fat wires (the rise is 2.5 radii per
    segment), the deck momwire#1029's sector route serves."""
    depth, radial, mast = 0.15, 6.334, 10.556
    wires = []
    for i in range(4):
        th = 2 * math.pi * i / 4
        c, s = round(math.cos(th), 15), round(math.sin(th), 15)
        wires.append(np.array([(0.0, 0.0, -depth), (radial * c, radial * s, -depth)]))
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    wires.append(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, mast)]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=[[6]] * 4 + [[3], [8]],
            wavelength=42.2,
            wire_radius=0.02,
            feeds=[(5, 0.5 * mast / 8, 1 + 0j)],
            ground_z=0.0,
            ground_eps=(13.0, 0.005),
            ground_model="sommerfeld",
            **kw,
        )


def test_buried_resolves_off():
    s = _screen(exact_kernel="auto")
    assert s.exact_kernel is False
    true = _screen(exact_kernel=True)
    assert true._exact_kernel_route_refusal(None, False) is not None


def test_rotational_symmetry_resolves_off():
    s = _screen(exact_kernel="auto", rotational_symmetry=True)
    assert s.exact_kernel is False
    with pytest.raises(NotImplementedError, match="1408"):
        _screen(exact_kernel=True, rotational_symmetry=True)


def test_an_unserved_fill_route_skips_under_auto(monkeypatch):
    """A fill route found unserved only at fill time (restricted rows, a
    complex k) skips the correction under "auto" and records that it did,
    where True refuses it."""
    off = _z(_build(n=75, exact_kernel=False))
    s = _build(n=75, exact_kernel="auto")
    assert s.exact_kernel is True
    monkeypatch.setattr(
        BSplineSolver, "_exact_kernel_route_refusal", lambda *_a: "unserved"
    )
    assert np.array_equal(_z(s), off)
    assert s.exact_kernel is False
    t = _build(n=75, exact_kernel=True)
    with pytest.raises(NotImplementedError, match="unserved"):
        _z(t)
