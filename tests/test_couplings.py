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

A GATE WHOSE PASSING CONDITION IS THAT THE AUTHOR ALSO EDITED THE CHECKLIST
MEASURES THE AUTHOR, NOT THE CODE. That is the general rule, and this file
learned it the expensive way (momwire#888). The coverage gate below used to
compare COUPLINGS against a literal set written a few lines above it, so
adding a row AND adding its name to that set turned it green with no
construction anywhere — which is exactly what happened while the four #888
rows were being written, and it passed. Its name said "covered by a
construction"; what it actually checked was that someone had typed the row
twice.

So the checklist is now derived from evidence the tests themselves leave: the
`_find_for(...)` CALL SITES in this module, read out of its own source. A row
counts as covered only if some test looks it up in order to build it, and
`_find_for` separately asserts the class it was handed is one the row's
`applies_to` names — so a call site is bound to a construction AND to the
data, and neither can be satisfied by editing a list.

Static parsing rather than a set recorded at runtime, which was the tempting
repair and the wrong one: this suite runs under `--dist loadgroup`, so a
module-global "what did we actually build" is split across xdist workers and
under-reports — a gate quietly measuring less than it claims, which is the
same family of failure one level up.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from momwire import (
    ArrayBlockSolver,
    BSplineSolver,
    HMatrixSolver,
    RazorSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
)
from momwire._capabilities import AXIS_VALUES
from momwire._capabilities import _combo_key
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


def _find_for(cls, axis_a, value_a, axis_b, value_b):
    """The row, plus the check that binds the gate to the data.

    A coupling is PER-CLASS, and the table's `applies_to` is the only place
    that fact lives. This asserts the class the test actually constructs is
    one the row names — so the data cannot drift from what the gate knows,
    which is exactly how the field came to be missing in the first place.
    """
    c = _find(axis_a, value_a, axis_b, value_b)
    assert cls.__name__ in c.applies_to, (
        f"{cls.__name__} raises this coupling but the row names "
        f"{c.applies_to} — applies_to is wrong or the gate is on the wrong class"
    )
    return c


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
    """Each side names a compositional axis exactly when its flag says so.

    `axis_a` was ALWAYS an axis for the first six rows, and momwire#888 added
    two where neither side is (`per_wire_radius`, `wire_loading`) — hence
    `a_is_axis`. Both directions are asserted: a flag claiming "axis" must
    name a real one, and a flag claiming "keyword" must NOT, so the marker
    cannot be set carelessly in either direction.
    """
    DERIVED = ("ground_model", "wire_position")
    for c in COUPLINGS:
        if c.a_is_axis:
            assert c.axis_a in AXIS_VALUES or c.axis_a in DERIVED, c
            if c.axis_a in AXIS_VALUES:
                assert c.value_a in AXIS_VALUES[c.axis_a], c
        else:
            assert c.axis_a not in AXIS_VALUES, (
                f"{c.axis_a} IS an axis — drop a_is_axis=False"
            )
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
    c = _find_for(
        SinusoidalSolver, "testing", "point-matching", "feed_model", "point-gap"
    )
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
    c = _find_for(cls, "solve_strategy", value, "wire_position", "buried")
    s = cls(**_buried_kw())
    with pytest.raises(NotImplementedError) as exc:
        s.build_hmatrix()
    # Composed, not verbatim: the raise prefixes the caller's name.
    assert c.reason in str(exc.value)


def test_the_extended_kernel_really_refuses_buried():
    c = _find_for(BSplineSolver, "kernel", "extended", "wire_position", "buried")
    with pytest.raises((NotImplementedError, ValueError)) as exc:
        BSplineSolver(**_buried_kw(), extended_kernel=True).compute_impedance()
    assert c.reason in str(exc.value)


def test_the_extended_kernel_really_requires_the_near_correction():
    c = _find_for(
        SinusoidalGalerkinSolver, "kernel", "extended", "near_correction", "False"
    )
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
    c = _find_for(
        SinusoidalGalerkinSolver, "kernel", "extended", "junction_ports", "True"
    )
    assert c.condition, "the conditional entry lost its condition"

    with pytest.raises((NotImplementedError, ValueError)) as exc:
        SinusoidalGalerkinSolver(**_junction_deck([1e-3, 2e-3]))
    assert c.reason in str(exc.value)

    # ...and the same geometry with ONE radius is served.
    SinusoidalGalerkinSolver(**_junction_deck(1e-3))


# ---- momwire#888: the singular-enrichment rows and ground contact ---------


def _enrich_kw(**over):
    """A one-wire deck with singular enrichment armed."""
    return dict(
        wires=[_mono()],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 6.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=1e-3,
        use_singular_enrichment=True,
        **over,
    )


def test_the_extended_kernel_really_refuses_singular_enrichment():
    """The row antennaknobs was missing, and the one that cost something.

    With no served row to reference, that frontend hand-wrote its own
    sentence citing momwire#271 where this refusal cites momwire#249 follow-up
    C. The table cannot stop a downstream copy from existing; it can make
    deleting the copy possible, which needs this row to be true.
    """
    c = _find_for(BSplineSolver, "kernel", "extended", "singular_enrichment", "True")
    with pytest.raises(NotImplementedError) as exc:
        BSplineSolver(**_enrich_kw(extended_kernel=True)).compute_impedance()
    assert c.reason in str(exc.value)


def test_mixed_per_wire_radii_really_refuse_singular_enrichment():
    c = _find_for(
        BSplineSolver, "per_wire_radius", "True", "singular_enrichment", "True"
    )
    kw = _enrich_kw()
    kw["wires"] = [_mono(), _mono(9.0, 2.0)]
    kw["n_per_edge_per_wire"] = [[9], [9]]
    # The step is what the refusal is about: one radius everywhere is served.
    kw["wire_radius"] = [1e-3, 2e-3]
    with pytest.raises(NotImplementedError) as exc:
        BSplineSolver(**kw).compute_impedance()
    assert c.reason in str(exc.value)


@pytest.mark.parametrize(
    "loading",
    [
        pytest.param(dict(wire_conductivity=5.8e7), id="conductivity"),
        pytest.param(
            dict(insulation_radius=2e-3, insulation_eps_r=2.3), id="insulation"
        ),
    ],
)
def test_distributed_wire_loading_really_refuses_singular_enrichment(loading):
    """Both spellings of "distributed loading", because the row names the
    CONCEPT and either kwarg reaches it — a gate on only one would go green if
    the other stopped refusing."""
    c = _find_for(BSplineSolver, "wire_loading", "True", "singular_enrichment", "True")
    with pytest.raises(NotImplementedError) as exc:
        BSplineSolver(**_enrich_kw(**loading)).compute_impedance()
    assert c.reason in str(exc.value)


@pytest.mark.parametrize(
    "cls",
    [
        BSplineSolver,
        HMatrixSolver,
        ArrayBlockSolver,
        SinusoidalSolver,
        SinusoidalGalerkinSolver,
        RazorSolver,
    ],
)
def test_ground_contact_under_refl_coef_is_refused_by_all_six(cls):
    """Every class the row names, constructed — the `applies_to` was MEASURED
    and the measurement is what this preserves.

    Declared in four modules, reaching six classes through inheritance. The
    seventh class, `HarringtonSolver`, is deliberately NOT here: see the
    companion test below.
    """
    c = _find_for(cls, "wire_position", "contact", "ground_model", "refl-coef")
    with pytest.raises(NotImplementedError) as exc:
        cls(
            wires=[_mono(bottom=0.0)],
            n_per_edge_per_wire=[[11]],
            feeds=[(0, 5.0, 1 + 0j)],
            wavelength=WL,
            wire_radius=1e-3,
            ground_z=0.0,
            ground_eps=SOIL_A,
            ground_model="refl-coef",
        ).compute_impedance()
    # Composed: the raise prefixes which wire end is in the plane.
    assert c.reason in str(exc.value)


def test_harrington_is_absent_from_the_contact_row_and_that_is_correct():
    """The row is NOT universal, though all seven classes refuse the pair.

    `HarringtonSolver` refuses `contact` OUTRIGHT, under every ground model —
    a single-cell refusal, not a coupling. Listing it would tell a `pulse`
    user the PAIRING is the problem and imply contact over Sommerfeld works,
    which is false. Asserted by measuring both, because "all seven refuse it"
    is true and is the wrong reason to add the seventh.
    """
    from momwire.harrington import HarringtonSolver

    c = _find("wire_position", "contact", "ground_model", "refl-coef")
    assert "HarringtonSolver" not in c.applies_to

    caps = HarringtonSolver.capabilities
    # Refuses contact on its own, and under a ground model the six SERVE.
    assert caps.refusal("contact") is not None
    assert caps.refusal("contact", "sommerfeld") is not None
    # ...and the six do serve exactly those, which is what makes their
    # refusal of the PAIR a coupling rather than a missing cell.
    assert BSplineSolver.capabilities.refusal("contact") is None
    assert BSplineSolver.capabilities.refusal("contact", "sommerfeld") is None


def test_every_flat_entry_is_covered_by_a_construction_above():
    """A new row must arrive with a construction, not just a sentence.

    THE CHECKLIST IS READ OUT OF THIS FILE'S OWN SOURCE, not maintained by
    hand. That is a deliberate change (momwire#888): the previous version
    compared COUPLINGS against a literal set written just above, so adding a
    row AND adding its name to that set turned the test green with no
    construction anywhere — which is exactly what happened while writing the
    four #888 rows, and it went green. A gate whose passing condition is "the
    author also edited the checklist" measures the author, not the code.

    So the covered set is now derived by parsing every `_find_for(...)` CALL
    SITE in this module: a row is covered iff some test in this file actually
    looks it up in order to construct it. `_find_for` additionally asserts the
    class it was handed is one the row's `applies_to` names, so a call site is
    not merely a mention — it is bound to a construction and to the data.

    Static rather than runtime because the suite runs under
    `--dist loadgroup`: a session-global "what did we build" set is split
    across xdist workers and would under-report, which is a gate that quietly
    measures less than it claims.
    """
    import ast

    tree = ast.parse(pathlib.Path(__file__).read_text())
    patterns = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_find_for"
        ):
            # A parametrized cell (`_find_for(cls, "solve_strategy", value,
            # ...)`) is not a literal, so it reads as a WILDCARD here. That is
            # sound: `_find` raises when no row matches, so the runtime call
            # still proves the specific value exists — the wildcard only
            # widens which declared row this call site can vouch for, and it
            # cannot vouch for a row that no test looks up.
            pat = tuple(
                a.value
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
                else None
                for a in node.args[1:5]
            )
            assert len(pat) == 4, f"_find_for called with {len(pat)} cells"
            patterns.append(pat)

    declared = {(c.axis_a, c.value_a, c.axis_b, c.value_b) for c in COUPLINGS}
    covered = {
        row
        for row in declared
        for pat in patterns
        if all(p is None or p == v for p, v in zip(pat, row))
    }
    unused = [
        pat
        for pat in patterns
        if not any(
            all(p is None or p == v for p, v in zip(pat, row)) for row in declared
        )
    ]
    assert declared == covered, (
        "COUPLINGS changed without its construction gate: "
        f"uncovered={sorted(declared - covered)}"
    )
    assert not unused, f"construction for a row that no longer exists: {unused}"


def test_every_reason_IS_the_module_constant_and_not_a_copy():
    """Identity, not equality. A table that retyped the prose would read as
    authoritative and drift the first time a refusal was reworded; holding the
    same object makes that impossible rather than unlikely."""
    from momwire._ground_spec import CONTACT_UNDER_REFL_COEF_REFUSAL
    from momwire.bspline import (
        _BURIED_EXTENDED_KERNEL_REFUSAL,
        _ENRICHMENT_EXTENDED_KERNEL_REFUSAL,
        _ENRICHMENT_PER_WIRE_RADIUS_REFUSAL,
        _ENRICHMENT_WIRE_LOADING_REFUSAL,
    )
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
        # momwire#888
        ("kernel", "singular_enrichment"): _ENRICHMENT_EXTENDED_KERNEL_REFUSAL,
        (
            "per_wire_radius",
            "singular_enrichment",
        ): _ENRICHMENT_PER_WIRE_RADIUS_REFUSAL,
        ("wire_loading", "singular_enrichment"): _ENRICHMENT_WIRE_LOADING_REFUSAL,
        ("wire_position", "ground_model"): CONTACT_UNDER_REFL_COEF_REFUSAL,
    }
    for c in COUPLINGS:
        assert c.reason is want[(c.axis_a, c.axis_b)], (
            f"{c.axis_a}={c.value_a} x {c.axis_b}: reason is a COPY, not the "
            "module constant"
        )


def test_every_row_names_a_real_solver_class_it_applies_to():
    """`applies_to` must be populated and must name importable classes.

    An empty tuple would make the AK-side filter show a coupling to nobody;
    a typo would make it show to nobody while looking populated. Both are the
    silent-omission direction, so both fail here.
    """
    import momwire

    for c in COUPLINGS:
        assert c.applies_to, f"{c.axis_a}={c.value_a}: applies_to is empty"
        for name in c.applies_to:
            assert isinstance(getattr(momwire, name, None), type), (
                f"{c.axis_a}={c.value_a}: applies_to names {name!r}, "
                "which is not a momwire solver class"
            )


def test_no_row_UNDER_attributes_a_refusal_the_class_actually_raises():
    """`applies_to` guards mis-attribution in BOTH directions.

    A ONE-DIRECTIONAL GATE PASSES ON THE DIRECTION IN FRONT OF THE AUTHOR.
    That is the general lesson and it is worth stating here because it cost
    something three separate times in one day: this row set, the
    `exposed`/`accepted` split downstream, and the coverage checklist above.
    Each time the gate written was the one whose failure the author had just
    imagined, and the mirror was left unguarded until a consumer tripped it.

    Only the over-attribution half was being checked — "a backend is told
    only about couplings that apply to IT" — and the missing half cost
    something: the three singular-enrichment rows named `BSplineSolver`
    alone, while `HMatrixSolver` and `ArrayBlockSolver` inherit that
    `refusals` dict and raise identically. A downstream panel that greys a
    control from the served rows therefore stopped greying it on the two
    accelerators, and nothing here said so.

    So: for every row, every class whose capabilities REFUSE that combination
    must be named. Asked of the capability rather than by construction
    because it is a question about the declared surface, and the construction
    gates above already prove the declarations are true.
    """
    import momwire

    classes = {
        name: getattr(momwire, name)
        for name in (
            "BSplineSolver",
            "HMatrixSolver",
            "ArrayBlockSolver",
            "SinusoidalSolver",
            "SinusoidalGalerkinSolver",
            "RazorSolver",
        )
    }
    # The refusal KEY each row is about. Kept beside the row rather than
    # derived from `axis_a`/`axis_b`, because the table's axis names are the
    # panel's vocabulary and the dict's keys are momwire's.
    keys = {
        ("kernel", "singular_enrichment"): ("extended_kernel", "singular_enrichment"),
        ("per_wire_radius", "singular_enrichment"): (
            "per_wire_radius",
            "singular_enrichment",
        ),
        ("wire_loading", "singular_enrichment"): (
            "wire_loading",
            "singular_enrichment",
        ),
        ("wire_position", "ground_model"): ("contact", "refl-coef"),
    }
    checked = 0
    for c in COUPLINGS:
        cells = keys.get((c.axis_a, c.axis_b))
        if cells is None:
            continue
        for name, cls in classes.items():
            # THE COMBO KEY, not `refusal(a, b)`. That method falls back to
            # single-cell refusals, so `SinusoidalSolver` "refuses"
            # extended_kernel+singular_enrichment only because it does not
            # serve enrichment AT ALL — which is not a coupling and must not
            # be attributed as one. Same distinction that keeps
            # `HarringtonSolver` off the contact row.
            refuses = _combo_key(cells) in cls.capabilities.refusals
            # `HarringtonSolver` is excluded from the contact row on purpose
            # (single-cell refusal, not a coupling) and is not in `classes`.
            if refuses and name not in c.applies_to:
                raise AssertionError(
                    f"{c.axis_a}={c.value_a} x {c.axis_b}: {name} refuses this "
                    f"and the row does not name it — applies_to={c.applies_to}"
                )
            checked += 1
    assert checked >= 24, checked


def test_the_bspline_family_shares_one_applies_to_tuple():
    """One inherited `refusals` dict, one tuple. Three literals would be three
    things to drift, and the drift is invisible until a consumer greys a
    control on one class and not its subclass."""
    rows = [c for c in COUPLINGS if c.axis_b == "singular_enrichment"]
    assert len(rows) == 3
    first = rows[0].applies_to
    assert all(r.applies_to is first for r in rows)
    assert first == ("BSplineSolver", "HMatrixSolver", "ArrayBlockSolver")
