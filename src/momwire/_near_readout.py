"""The near-field readout: one composition, every seam (momwire#1336).

The element sum is :func:`momwire._far_readout._element_fields` (the
mixed-potential form, momwire#719 U1) and the ground REMAINDER at a point is
:mod:`momwire._field_point`'s (momwire#545).  What lives here is the third
piece every seam used to spell for itself — the COMPOSITION of those two over
the solve's ground — plus the two questions every seam asks before composing:
where the grid's points are, and whether any of them sits where no composition
converges.

Three copies of the PEC-image term lived in the tree before this module
(``eznec._serve._near_field``, ``eznec._nec4._near_field``,
``portal._portal._near_field_lines``), and the finite-ground half of it lived
in only one of them.  That is why a finite-ground near field was a NEC-5-slot
capability and a ``GN 2`` deck on the nec2 portal refused one (momwire#550):
the evaluator never cared which dialect asked, the composition did.

The composition, one line with three spellings
----------------------------------------------
Which spelling a solve gets is :func:`near_ground`'s, read off the solver's
own ground decision (:func:`momwire._ground_spec.ground_config`, the one owner
of ``C₂`` in this tree):

  ``free``     the elements alone
  ``pec``      the elements plus their geometric mirror
  ``compose``  ``direct + C₂·image + remainder``

The IMAGE is the geometric mirror with the horizontal moments flipped and the
continuity charge NEGATED, which is one statement twice over (reversing a
horizontal current reverses dI/ds, and mirroring a vertical one reverses the
arc direction).  Over a finite ground it is scaled by ``C₂ = (ε̃−1)/(ε̃+1)``
and the smooth remainder is added, NEC's own decomposition of the half-space
Green's function (theory manual eqs 136-147) evaluated at a point.  The
association is :class:`momwire._field_ground.FieldGround`'s ``"compose"``
contract and not a style: the coefficient goes on the LEFT of the image block
and ``coef·img + rem`` is associated before the outer sum, because that
contract is about float64 evaluation order.

A reflection-coefficient solve (nec2's ``GN 0``) has no near-field spelling
here.  Its currents are a PEC image weighted by an angle-dependent Fresnel
factor the solve reads at each pair; a near field built from it has no
measurement behind it yet (momwire#550's scope note), so :func:`near_ground`
reports it as ``refl-coef`` and the seam refuses it by name.

What stays with each seam is everything a host reads: the card's grid
convention beyond the rectangular walk, the dust floor its printout prints
under, the row layout and the refusal sentences.  The arithmetic does not.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import _field_point, _ground_refl, _ground_spec
from ._far_readout import _element_fields, _image_moments

__all__ = [
    "NEAR_FIELD_SUBDIV",
    "NearGround",
    "near_field_at",
    "near_ground",
    "near_point_refusal",
    "rectangular_grid",
]

# How finely the solved current is resampled before it is summed at an
# observation point.  The far field never needs it — every mesh element is
# already electrically small and only the radiation-zone limit survives — but
# a point a metre from a metre-long element resolves the variation along it.
# MEASURED to be converged on the NEC-5 seam's finite-ground captures: 0115
# reads 1.33 % worst-cell magnitude at ``subdiv = 1``, 1.92 % at 4, 1.9479 %
# at 8 and 1.9572 % at 32, and 0109 reads 10.34 / 5.02 / 5.4494 / 5.5834 %.
# Everything past 8 moves the answer by less than a tenth of a percent, so
# what is left at 8 is the formulation difference and not the sampling.
# Every seam's near field has used 8 since it first printed one; this is the
# one copy of it.
NEAR_FIELD_SUBDIV = 8


class NearGround(NamedTuple):
    """The ground a near field is COMPOSED over, in this module's terms.

    ``kind`` is ``"free"``, ``"pec"``, ``"compose"`` or ``"refl-coef"`` — the
    last one the solve's, never served (module docstring).  ``eps_tilde`` and
    ``coefficient`` are set for ``"compose"`` alone.
    """

    kind: str
    ground_z: float = 0.0
    eps_tilde: complex | None = None
    coefficient: complex | None = None


def near_ground(solver, *, medium: tuple[float, float] | None = None) -> NearGround:
    """``solver``'s ground, as the near-field composition needs it.

    Read off :func:`~momwire._ground_spec.ground_config` — what the FILL did
    — so a near field cannot be composed over a different half-space from
    the one its currents were solved in.

    ``medium`` is the one exception, and it is a dialect's: NEC-5's bare
    ``GD`` (the MININEC-type ground) solves its currents over a PERFECT image
    and evaluates its near field IN the medium (``eznec._serve``, "One near
    field, four grounds and one point").  Its solver never saw the medium, so
    ``ground_config`` hands back the PEC row; given ``medium = (εr, σ)`` the
    ε̃ is folded here through :func:`~momwire._ground_refl.eps_tilde` — the
    function ``ground_config`` itself calls, given the same pair — and ``C₂``
    is written in the one expression ``ground_config`` writes it in.  Two
    roads, one arithmetic.  A solve that already composes (a Sommerfeld fill)
    ignores ``medium``: the fill's own ε̃ wins.
    """
    config = _ground_spec.ground_config(solver, solver.omega)
    if config is None:
        return NearGround("free")
    ground_z = solver.ground_z
    if config.mode == "compose":
        assert config.eps_tilde is not None
        return NearGround(
            "compose", ground_z, config.eps_tilde, complex(config.image_coefficient)
        )
    if medium is not None:
        eps_t = _ground_refl.eps_tilde(medium, solver.omega, solver.eps)
        return NearGround("compose", ground_z, eps_t, (eps_t - 1.0) / (eps_t + 1.0))
    if config.eps_tilde is None:
        return NearGround("pec", ground_z)
    return NearGround("refl-coef", ground_z)


def rectangular_grid(counts, origin, step) -> np.ndarray:
    """An ``NE``/``NH`` card's rectangular grid, in METRES, X fastest, then
    Y, then Z — the nesting nec2c prints (``dipole_ne_nearfield.out``) and
    the one the NEC-5 seam measured on its own oracle (``NX = 2, NY = 3,
    NZ = 2``).  One walk for every seam, so the points a table prints and the
    points a refusal inspects are one list."""
    n_x, n_y, n_z = counts
    start = np.asarray(origin, dtype=float)
    step = np.asarray(step, dtype=float)
    return np.array(
        [
            start + np.array([ix, iy, iz]) * step
            for iz in range(n_z)
            for iy in range(n_y)
            for ix in range(n_x)
        ]
    )


def near_field_at(points, elements, k, radius, magnetic, ground: NearGround, omega):
    """E (or H) at ``points``: ``(N, 3)`` complex, peak, in the units the
    solve's drive set — nothing is scaled.

    ``elements`` is ``solver.element_currents(coeffs, subdiv=...)``'s
    ``(mid, moment, nodes, delta)``, ``radius`` the thin-wire regularisation
    :func:`~momwire._far_readout._element_fields` reads, ``omega`` the radian
    frequency the remainder's surfaces are normalised at.  A ``refl-coef``
    ground is a ``ValueError``: the seam refuses it by name before asking.
    """
    field = _element_fields(points, elements, k, radius, magnetic)
    if ground.kind == "free":
        return field
    if ground.kind not in ("pec", "compose"):
        raise ValueError(f"no near-field composition over a {ground.kind} ground")
    mid, moment, nodes, delta = elements
    mid_img, moment_img = _image_moments(mid, moment, ground.ground_z)
    nodes_img = nodes.copy()
    nodes_img[:, 2] = 2.0 * ground.ground_z - nodes[:, 2]
    image = _element_fields(
        points, (mid_img, moment_img, nodes_img, -delta), k, radius, magnetic
    )
    if ground.kind == "pec":
        return field + image
    evaluate = (
        _field_point.reflected_h_field_at
        if magnetic
        else _field_point.reflected_field_at
    )
    remainder = evaluate(
        points, mid, moment, ground.eps_tilde, ground.ground_z, k, omega
    )
    np.multiply(ground.coefficient, image, out=image)
    return field + (image + remainder)


def near_point_refusal(points, polylines, ground_z):
    """The first observation point no finite-ground composition answers, or
    ``None``.

    ``("below", point, None)`` for a point under the interface — the field in
    the soil is the TRANSMITTED one, which this composition is not
    (momwire#1257) — checked first because it is the grid's shape rather than
    one unlucky cell.  ``("contact", point, wire)`` for a point ON a wire end
    standing in the plane: the image cancels the base node's continuity
    charge only by ``1 − C₂ = 2/(1 + ε̃)``, so a finite residual charge sits
    exactly at the observer and the composed field there diverges as the
    current is resampled (36.6 → 1230 V/m over subdiv 1 → 64, gated in
    ``tests/test_field_point.py``).

    Asked over a FINITE ground only; over a perfect one ``C₂ = 1`` and the
    contact charge cancels exactly.  The contact tolerance is the solver's
    own, per wire — :func:`~momwire._ground_spec.ground_touch_tol` — so "this
    end stands on the ground" means here exactly what it means to the basis
    that grows an image off it.  ``wire`` is an index into ``polylines``.
    """
    for point in points:
        if float(point[2]) < ground_z:
            return ("below", point, None)
    contacts = []
    for index, polyline in enumerate(polylines):
        ends = np.array([polyline[0], polyline[-1]], dtype=float)
        tol = _ground_spec.ground_touch_tol(ends)
        for end in ends:
            if abs(float(end[2]) - ground_z) <= tol:
                contacts.append((index, end, tol))
    for point in points:
        for index, end, tol in contacts:
            if float(np.linalg.norm(point - end)) <= tol:
                return ("contact", point, index)
    return None
