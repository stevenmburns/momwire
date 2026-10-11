"""The exact kernel's shared plumbing (momwire#1433 PR A).

`_exact_kernel_seam` is bs2's former option/refusal/grouping code, lifted so
other families take `exact_kernel` with the same semantics. bs2's own
behaviour is gated bit for bit by the existing exact-kernel tests and by
`scratch/1433/bitid_bs2.py`; these pin the seam's own surface.
"""

import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, HMatrixSolver, _exact_kernel_seam, _wire_spec

L = 0.47
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]


def test_parse_option():
    assert _exact_kernel_seam.parse_option(False) == (False, False)
    assert _exact_kernel_seam.parse_option(True) == (True, True)
    assert _exact_kernel_seam.parse_option("auto") == ("auto", False)
    with pytest.raises(ValueError, match="'auto'"):
        _exact_kernel_seam.parse_option("on")


def test_bs2_takes_the_seam_constants():
    assert BSplineSolver.EXACT_KERNEL_CHOICES is _exact_kernel_seam.EXACT_KERNEL_CHOICES
    assert BSplineSolver.EXACT_KERNEL_AUTO_H_OVER_A == 3.0
    assert BSplineSolver.EXACT_KERNEL_FIXED_GAP_FEED is True
    assert isinstance(BSplineSolver(wires=DIPOLE), _exact_kernel_seam.ExactKernelOption)


def test_unserved_subclass_refuses_by_class_name():
    with pytest.raises(NotImplementedError, match="not served by HMatrixSolver"):
        HMatrixSolver(wires=DIPOLE, exact_kernel=True)


def test_coaxial_groups_partition_the_segments():
    """A T: the stem's two collinear halves are one group, the arm another."""
    s = BSplineSolver(
        wires=[
            np.array([(0.0, 0.0, -0.2), (0.0, 0.0, 0.0)]),
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.2)]),
            np.array([(0.0, 0.0, 0.0), (0.15, 0.0, 0.0)]),
        ],
        n_per_edge_per_wire=[[5], [5], [4]],
        wire_radius=0.005,
    )
    geom = s._build_geometry()
    _labels, groups = _exact_kernel_seam.coaxial_groups(
        geom["seg_l"], geom["seg_r"], geom["tangents"], s._seg_radius(geom)
    )
    sizes = sorted(S.size for _g, S in groups)
    assert sizes == [4, 10]
    assert np.array_equal(
        np.sort(np.concatenate([S for _g, S in groups])), np.arange(14)
    )
    axis, x0, sgn, h = _exact_kernel_seam.group_frame(
        groups[0][1], geom["seg_l"], geom["seg_r"], geom["tangents"]
    )
    assert np.all(np.abs(sgn) == 1.0) and np.all(h > 0)
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-12


def _gap_text(exact_kernel, fixed_gap_feed):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _wire_spec.advise_gap_mesh_floor(
            "Fam",
            [DIPOLE[0]],
            [[401]],
            [0.01],
            True,
            [("source 0", [(0, None)])],
            gap_model="knot",
            exact_kernel=exact_kernel,
            fixed_gap_feed=fixed_gap_feed,
        )
    (msg,) = [str(x.message) for x in w if x.category is _wire_spec.GapMeshFloor]
    return msg


def test_gap_floor_text_follows_the_family_feed():
    # bs2 (a fixed-gap feed exists): the end-port feed is named as the fix.
    assert "A gap of fixed width is the end-port feed" in _gap_text(True, True)
    assert "exact_kernel=True with an end-port feed" in _gap_text(False, True)
    # A family whose feeds are knots is told where a fixed gap is instead.
    ran = _gap_text(True, False)
    assert "zero-width knot gaps" in ran and "bspline's end-port feed" in ran
    could = _gap_text(False, False)
    assert "removes the collapse" in could and "still drifts" in could


def test_point_gap_warning_text():
    assert _exact_kernel_seam.point_gap_warning(True).endswith(
        "A fixed geometric gap is two facing wire ends driven through "
        "junction_ports (momwire#1408)."
    )
    assert "no fixed-width gap" in _exact_kernel_seam.point_gap_warning(False)
