"""`PulseSolver`'s declared capability row, cross-checked against reality.

The same definition-of-done `tests/test_capabilities.py` holds every other
solver to, in the shape of its § 2 (RazorSolver) section: every declared
False/absent cell with a refusal must both RAISE on a tiny deck and have
`capabilities.refusal(cell)` say why; every True cell's kwarg must be
ACCEPTED. Kept in its own module only because momwire#416's probe was
built alongside an in-flight razor unit and stayed off shared files — it
belongs merged into `tests/test_capabilities.py` § 2.5 at review time.
"""

import numpy as np
import pytest

from momwire import PulseSolver
from momwire.harrington import HarringtonSolver

WAVELENGTH = 10.0


def _wire(z=0.0, n=2):
    return [np.array([(0.0, 0.0, z), (0.0, 1.0, z)])], [[n]]


def test_pulse_serves_free_space_pec_refl_coef_and_sommerfeld():
    """The row momwire#416 predicted, updated by momwire#430: a solver that
    wrote no ground code of its own declares THREE grounds, because
    `PotentialGround` served all three through one surface — the third,
    Sommerfeld's, only once #398 unit 5 shipped `Remainder.field_windows`.
    Declaration and constructor are checked together — a served declaration
    with a refusing constructor is exactly the drift this file exists to
    catch.
    """
    c = PulseSolver.capabilities
    assert c.grounds == frozenset({"pec", "refl-coef", "sommerfeld"})
    for ground_name in ("pec", "refl-coef", "sommerfeld"):
        assert c.refusal(ground_name) is None
        assert ground_name not in c.refusals

    wires, npe = _wire(z=1.0)
    for ground in (
        dict(ground_z=0.0),
        dict(ground_z=0.0, ground_eps=10 - 1j),
        dict(ground_z=0.0, ground_eps=10 - 1j, ground_model="sommerfeld"),
    ):
        z, _ = PulseSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **ground
        ).compute_impedance()
        assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_pulse_ground_contact_refuses():
    """The fill would run; nothing in this probe measures it. Not a
    capability cell (no solver declares "ground contact" as an axis), so
    this pins the constructor's own refusal — the honest boundary of what
    the probe shipped."""
    wires, npe = _wire(z=0.0)
    with pytest.raises(NotImplementedError, match="touches the ground plane"):
        PulseSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, ground_z=0.0
        )
    with pytest.raises(ValueError, match="dips below the ground plane"):
        PulseSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, ground_z=1.0
        )


@pytest.mark.parametrize(
    "cell,kwarg",
    [
        ("junction_ports", {"junction_ports": [0]}),
        ("node_gaps", {"node_gaps": [(0, "end", 1.0 + 0j)]}),
        ("extended_kernel", {"extended_kernel": True}),
    ],
)
def test_pulse_out_of_scope_kwargs_refuse(cell, kwarg):
    wires, npe = _wire()
    assert PulseSolver.capabilities.refusal(cell)
    with pytest.raises(NotImplementedError):
        PulseSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kwarg
        )


def test_pulse_sommerfeld_without_ground_eps_is_a_valueerror_not_a_refusal():
    """`ground_model="sommerfeld"` is no longer `_OUT_OF_SCOPE` (momwire#430)
    — it is a served capability with its own constructor contract, so a bare
    `ground_model="sommerfeld"` (no `ground_eps`, no `ground_z`) fails the
    same way `BSplineSolver` fails it: a `ValueError` about `ground_eps`,
    not a `NotImplementedError` about being out of scope.
    """
    wires, npe = _wire()
    with pytest.raises(ValueError, match="requires ground_eps"):
        PulseSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wavelength=WAVELENGTH,
            ground_z=0.0,
            ground_model="sommerfeld",
        )


@pytest.mark.parametrize("kwarg", [{"degree": 2}, {"junctions": [[(0, "end")]]}])
def test_pulse_refuses_the_kwargs_that_have_no_meaning_here(kwarg):
    """`degree` and `junctions` are not capability cells — the first is a
    B-spline knob and the second is a spec this basis does not need at all
    (coincident endpoint charges superpose by arithmetic). Both are refused
    with a reason rather than silently ignored."""
    wires, npe = _wire()
    with pytest.raises(NotImplementedError):
        PulseSolver(
            wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kwarg
        )


def test_pulse_per_wire_radius_refuses():
    wires, npe = _wire()
    assert PulseSolver.capabilities.refusal("per_wire_radius")
    with pytest.raises(NotImplementedError):
        PulseSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wire_radius=[0.001, 0.0015],
            wavelength=WAVELENGTH,
        )


@pytest.mark.parametrize(
    "cell,kwarg",
    [
        ("wire_loading", {"wire_conductivity": 1e7}),
        ("insulation", {"insulation_radius": 1e-3}),
    ],
)
def test_a_wire_load_is_refused_by_name_and_not_as_a_caller_typo(cell, kwarg):
    """momwire#564 scope item 2: these three kwargs used to fall through.

    `capabilities.wire_loading` has said False since momwire#396, but no
    sentence went with it, so `build_solver` spelling an `LD 5` / `LD 6` as
    `wire_conductivity=` / `insulation_radius=` reached the caller-typo
    `TypeError` at the bottom of `__init__`.  That is not a refusal: the
    portal's frame catches `ValueError` / `NotImplementedError` and writes a
    `NEC ERROR` line, and a `TypeError` goes straight through it, killing the
    daemon while the host waits on a sentinel that never arrives.

    So the exception TYPE is the assertion here, not an implementation
    detail — and the sentence has to name the class the caller actually
    constructed, which for a `HarringtonSolver` is not `PulseSolver`.
    """
    wires, npe = _wire()
    for cls in (PulseSolver, HarringtonSolver):
        with pytest.raises(NotImplementedError) as exc:
            cls(wires=wires, n_per_edge_per_wire=npe, wavelength=WAVELENGTH, **kwarg)
        assert cls.__name__ in str(exc.value)
        assert "does not serve wire loading" in str(exc.value)
    # The declared cell and the raised sentence are the same prose, per class.
    assert PulseSolver.capabilities.refusal("wire_loading")
    assert "PulseSolver" in PulseSolver.capabilities.refusal("wire_loading")
    assert "HarringtonSolver" in HarringtonSolver.capabilities.refusal("wire_loading")


def test_a_genuine_typo_is_still_a_typeerror():
    """The other half: `_OUT_OF_SCOPE` must not become a catch-all.

    A name nothing in the tree spells is a caller mistake, and turning it into
    a polite refusal would hide it.  The class name is still the constructed
    one, which is the whole of the naming fix.
    """
    wires, npe = _wire()
    for cls in (PulseSolver, HarringtonSolver):
        with pytest.raises(TypeError) as exc:
            cls(
                wires=wires,
                n_per_edge_per_wire=npe,
                wavelength=WAVELENGTH,
                use_singular_enrichment=True,
            )
        assert str(exc.value).startswith(f"{cls.__name__} got unexpected keyword")


def test_pulse_served_row_is_three_grounds_and_nothing_else():
    c = PulseSolver.capabilities
    assert c.grounds == frozenset({"pec", "refl-coef", "sommerfeld"})
    assert not any(
        [
            c.wire_loading,
            c.extended_kernel,
            c.junction_ports,
            c.node_gaps,
            c.per_wire_radius,
            c.singular_enrichment,
        ]
    )


def test_pulse_is_exported_from_the_top_level():
    import momwire

    assert momwire.PulseSolver is PulseSolver
    assert "PulseSolver" in momwire.__all__
