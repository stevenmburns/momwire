"""`_couplings.COUPLINGS` is measured, not annotated (antennaknobs#1006 G2-4).

A table of refused combinations is worth having only if it stays true. Prose
in a table drifts silently from the code that raises; a table that is checked
by CONSTRUCTING each refused cell cannot. So every flat entry below is
exercised by building the combination it forbids and requiring the refusal,
and the conditional entry is built twice — with the condition and without —
so the condition is tested rather than believed.

That direction matters more than it looks. The failure this guards is not "a
coupling was described wrongly", it is "a coupling stopped being true and the
table went on saying it" — the panel would then grey out a cell momwire serves
perfectly well, and nobody would find out from a green suite.

Two different checks on the prose, because one claim was too strong. Some
refusals raise the constant verbatim; others compose it — `HMatrixSolver`
raises `f"{who}: {_BURIED_FAST_OPERATOR_REFUSAL}"`, so the table's reason is
the BODY of that message, not the whole of it. So:

  * the raised message must CONTAIN the table's reason — that is what proves
    the coupling still refuses, and it holds for wrapped and bare alike;
  * the table's reason must BE the module constant, by identity — that is the
    anti-retype check, and it is what makes drift impossible rather than
    merely unlikely.

Equality against the raised string would have been wrong for two of the six
entries, and asserting it was the first thing this file got wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    HMatrixSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)
from momwire._capabilities import AXIS_VALUES
from momwire._couplings import COUPLINGS

WL = 42.83  # ~7 MHz
SOIL_A = (13.0, 0.005)


def _mono(top=11.0, bottom=1.0):
    return np.array([(0.0, 0.0, top), (0.0, 0.0, bottom)])


def _radial(length=5.0, depth=0.15):
    return np.array([(0.0, 0.0, -depth), (length, 0.0, -depth)], dtype=float)


def _buried_kw():
    return dict(
        wires=[_mono(), _radial()],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def _find(axis_a, value_a, axis_b, value_b):
    for c in COUPLINGS:
        if (c.axis_a, c.value_a, c.axis_b, c.value_b) == (
            axis_a,
            value_a,
            axis_b,
            value_b,
        ):
            return c
    raise AssertionError(f"no coupling {axis_a}={value_a} x {axis_b}={value_b}")


# --------------------------------------------------------------------------
# The table's own shape
# --------------------------------------------------------------------------


def test_axis_sides_name_real_axes_and_kwarg_sides_are_marked():
    """`axis_a` is always a compositional axis. `axis_b` is one only when
    `b_is_axis`, which is how a consumer skips what it cannot render without
    keeping a second list of exceptions."""
    for c in COUPLINGS:
        assert c.axis_a in AXIS_VALUES, c
        assert c.value_a in AXIS_VALUES[c.axis_a], c
        if c.b_is_axis:
            assert c.axis_b in AXIS_VALUES or c.axis_b in (
                "ground_model",
                "wire_position",
            ), c
        else:
            assert c.axis_b not in AXIS_VALUES, (
                f"{c.axis_b} IS an axis — drop b_is_axis=False"
            )


# --------------------------------------------------------------------------
# Each refusal, by construction
# --------------------------------------------------------------------------


def test_point_matching_really_refuses_the_point_gap():
    c = _find("testing", "point-matching", "feed_model", "point-gap")
    with pytest.raises(NotImplementedError) as exc:
        SinusoidalSolver(
            wires=[np.array([(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)])],
            n_per_edge_per_wire=[[11]],
            wavelength=WL,
            wire_radius=1e-3,
            feed_model="point",
        )
    assert c.reason in str(exc.value)


@pytest.mark.parametrize(
    "cls,value", [(HMatrixSolver, "aca"), (ArrayBlockSolver, "element-block")]
)
def test_the_accelerated_assemblies_really_refuse_buried(cls, value):
    c = _find("solve_strategy", value, "wire_position", "buried")
    s = cls(**_buried_kw())
    with pytest.raises(NotImplementedError) as exc:
        s.build_hmatrix()
    # Composed, not verbatim: the raise prefixes the caller's name.
    assert c.reason in str(exc.value)


def test_the_extended_kernel_really_refuses_buried():
    c = _find("kernel", "extended", "wire_position", "buried")
    with pytest.raises((NotImplementedError, ValueError)) as exc:
        BSplineSolver(**_buried_kw(), extended_kernel=True).compute_impedance()
    assert c.reason in str(exc.value)


def test_the_extended_kernel_really_requires_the_near_correction():
    c = _find("kernel", "extended", "near_correction", "False")
    with pytest.raises(NotImplementedError) as exc:
        SinusoidalGalerkinSolver(
            wires=[np.array([(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)])],
            n_per_edge_per_wire=[[11]],
            wavelength=WL,
            wire_radius=1e-3,
            extended_kernel=True,
            near_correction=False,
        )
    assert c.reason in str(exc.value)


# --------------------------------------------------------------------------
# The conditional one, both ways — the condition is TESTED, not annotated
# --------------------------------------------------------------------------


def _junction_deck(radii):
    """Two wires meeting at the origin, joined, with the given per-wire radii."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]),
            np.array([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]),
        ],
        n_per_edge_per_wire=[[9], [9]],
        junctions=[[(0, "start"), (1, "start")]],
        wavelength=WL,
        wire_radius=radii,
        extended_kernel=True,
    )


def test_the_stepped_radius_junction_refusal_needs_the_step():
    """WITH a radius step it raises; WITHOUT one the same deck constructs.

    This is the whole reason `condition` exists as a field. Stated flat, this
    coupling would read as "the extended kernel refuses junctions" — false,
    and it would send a user to the wrong workaround. Uniform-radius junctions
    are the overwhelmingly common case and are untouched.
    """
    c = _find("kernel", "extended", "junction_ports", "True")
    assert c.condition, "the conditional entry lost its condition"

    with pytest.raises((NotImplementedError, ValueError)) as exc:
        SinusoidalGalerkinSolver(**_junction_deck([1e-3, 2e-3]))
    assert c.reason in str(exc.value)

    # ...and the same geometry with ONE radius is served.
    SinusoidalGalerkinSolver(**_junction_deck(1e-3))


def test_every_flat_entry_is_covered_by_a_construction_above():
    """A new row must arrive with a construction, not just a sentence.

    Without this, adding an entry to COUPLINGS is free and unverified — which
    is exactly the annotated-table failure the module exists to avoid.
    """
    covered = {
        ("testing", "point-matching", "feed_model", "point-gap"),
        ("solve_strategy", "aca", "wire_position", "buried"),
        ("solve_strategy", "element-block", "wire_position", "buried"),
        ("kernel", "extended", "wire_position", "buried"),
        ("kernel", "extended", "near_correction", "False"),
        ("kernel", "extended", "junction_ports", "True"),
    }
    declared = {(c.axis_a, c.value_a, c.axis_b, c.value_b) for c in COUPLINGS}
    assert declared == covered, (
        "COUPLINGS changed without its construction gate: "
        f"uncovered={sorted(declared - covered)} stale={sorted(covered - declared)}"
    )


def test_every_reason_IS_the_module_constant_and_not_a_copy():
    """Identity, not equality. A table that retyped the prose would read as
    authoritative and drift the first time a refusal was reworded; holding the
    same object makes that impossible rather than unlikely."""
    from momwire.bspline import _BURIED_EXTENDED_KERNEL_REFUSAL
    from momwire.hmatrix import HMatrixSolver as _H
    from momwire.sinusoidal import _POINT_FEED_MODEL_REFUSAL
    from momwire.sinusoidal_galerkin import (
        _EK_NEAR_CORRECTION_REFUSAL,
        _EK_STEPPED_RADIUS_JUNCTION_REFUSAL,
    )

    want = {
        ("testing", "feed_model"): _POINT_FEED_MODEL_REFUSAL,
        ("solve_strategy", "wire_position"): _H.capabilities.refusals["buried"],
        ("kernel", "wire_position"): _BURIED_EXTENDED_KERNEL_REFUSAL,
        ("kernel", "near_correction"): _EK_NEAR_CORRECTION_REFUSAL,
        ("kernel", "junction_ports"): _EK_STEPPED_RADIUS_JUNCTION_REFUSAL,
    }
    for c in COUPLINGS:
        assert c.reason is want[(c.axis_a, c.axis_b)], (
            f"{c.axis_a}={c.value_a} x {c.axis_b}: reason is a COPY, not the "
            "module constant"
        )
