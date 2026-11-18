"""Which solver honours which feature (momwire#1980).

What is held here:

- every feature names the solver families that honour it
  (`Feature.solvers`), and each family is a solver class's
  `FEATURE_SOLVER`; the roster's basis names map to their class's family
  (`_features.basis_solver`);
- each constructor reads, through `_features.bind`, exactly the features its
  family honours: the kwargs it passes resolve, a feature it does not honour
  is neither resolved nor read from the environment, and a part follows its
  whole (momwire#1982);
- line 2 of a printout never names a feature the running solver ignores
  (`_features.stamp`'s `solver`): MOMWIRE_X_KCL_NULLSPACE=1 on a razor-2p
  deck leaves line 2 bare, as the executable refuses it outright, while
  MOMWIRE_X_PAIR_REUSE=1 there still names the whole its part runs for;
- the executables stamp the same way: a carried feature a basis ignores is
  refused (`EXE_FEATURE_BASES`) or accepted unstamped (`EXE_IGNORED`), never
  both and never accepted and stamped;
- the near-interface levers are set in one place (`_near_interface.
  under_levers`) and are off for the families that do not honour them.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
from pathlib import Path

import pytest

from momwire import _features as F
from momwire import _near_interface
from momwire.bspline import BSplineSolver
from momwire.deck import BASES, build_solver, parse
from momwire.dropin.nec5 import _printout
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "momwire"
EXE_SOURCE = ROOT / "standalone" / "momwire_razor2p.cpp"

CLASSES = {
    "bspline": BSplineSolver,
    "razor": RazorSolver,
    "sinusoidal": SinusoidalSolver,
    "sinusoidal_galerkin": SinusoidalGalerkinSolver,
}

DIPOLE = (
    "CM d\nCE\nGW 1 21 0 0 -2.5 0 0 2.5 0.001\nGE 0\n"
    "EX 0 1 11 0 1 0\nFR 0 1 0 0 28.0\nXQ\nEN\n"
)


@pytest.fixture
def no_features(monkeypatch):
    """Every feature variable (and alias) unset."""
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _construct(kind, **kw):
    """A small free-space dipole on family `kind`'s solver."""
    if kind == "razor":
        wire = [[[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]]]
        return RazorSolver(
            wires=wire,
            nsegs=21,
            wire_radius=1e-3,
            wavelength=20.0,
            nec5_quadrature=True,
            **kw,
        )
    if kind == "bspline":
        wire = [[[0.0, -2.5, 10.0], [0.0, 2.5, 10.0]]]
        return BSplineSolver(
            wires=wire, n_per_edge_per_wire=[[21]], wire_radius=1e-3, **kw
        )
    basis = {"sinusoidal": "sinusoidal", "sinusoidal_galerkin": "sinusoidal-galerkin"}
    assert not kw
    return build_solver(parse(DIPOLE, dialect="nec2"), basis=basis[kind]).solver


def _constructor_features(kind):
    """The features with a kwarg that family `kind` honours."""
    return {
        n for n, f in F.FEATURES.items() if f.kwarg is not None and kind in f.solvers
    }


# --- the registry ---------------------------------------------------------------


def test_every_feature_names_known_families():
    for f in F.FEATURES.values():
        assert f.solvers, f.name
        assert len(set(f.solvers)) == len(f.solvers), f.name
        assert set(f.solvers) <= set(F.SOLVERS), f.name


def test_each_family_is_a_solver_class():
    assert {k: c.FEATURE_SOLVER for k, c in CLASSES.items()} == {k: k for k in CLASSES}
    assert set(CLASSES) == set(F.SOLVERS)


def test_the_roster_maps_to_its_classes_families():
    for name, (cls, _kw) in BASES.items():
        assert F.basis_solver(name) == getattr(cls, "FEATURE_SOLVER", ""), name
    # The executables' spellings are known too.
    for name in F._EXE_BASES:
        assert F.basis_solver(name) is not None, name
    assert F.basis_solver("no-such-basis") is None


# --- the constructors -----------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(CLASSES))
def test_each_constructor_reads_exactly_its_families_features(
    no_features, kind, monkeypatch
):
    read = []
    real = F.enabled

    def spy(name, kwarg_value=None, **kw):
        read.append(name)
        return real(name, kwarg_value, **kw)

    monkeypatch.setattr(F, "enabled", spy)
    solver = _construct(kind)
    expected = _constructor_features(kind)
    assert set(read) & {n for n, f in F.FEATURES.items() if f.kwarg} == expected
    assert {n for n, on in solver.features_bound.items() if on} <= expected
    assert expected <= set(solver.features_bound)


@pytest.mark.parametrize("kind", sorted(CLASSES))
def test_a_feature_a_family_ignores_is_not_read_from_the_environment(no_features, kind):
    # Every variable on: each family resolves on only what it honours.
    for f in F.FEATURES.values():
        if f.kind == "python" and f.kwarg is not None:
            no_features.setenv(f.env, "1")
    solver = _construct(kind)
    on = {n for n, v in solver.features_bound.items() if v}
    # On: the python ones the environment asked for, and the promoted ones
    # (on by default where certified), each only where the family honours it.
    # A native part follows its whole: razor_pair_reuse, on pair_reuse's
    # variable (momwire#1982).
    honoured = {
        n
        for n in _constructor_features(kind)
        if F.FEATURES[n].kind == "python"
        or F.default_on(n)
        or F.FEATURES[n].part_of == "pair_reuse"
    }
    assert on == honoured


@pytest.mark.parametrize("kind", ["bspline", "razor", "sinusoidal"])
def test_a_constructor_takes_the_kwarg_of_exactly_what_its_family_honours(kind):
    # The Galerkin class takes its base's keywords through **kwargs, so its
    # narrower set is held by what its fill reads, below.
    taken = set(inspect.signature(CLASSES[kind].__init__).parameters)
    kwargs = {f.kwarg: n for n, f in F.FEATURES.items() if f.kwarg is not None}
    assert {n for k, n in kwargs.items() if k in taken} == _constructor_features(kind)


_SG_DECKS = {
    "free": DIPOLE,
    "pec": DIPOLE.replace("GE 0\n", "GE 1\n").replace(
        "0 0 -2.5 0 0 2.5", "0 -2.5 8 0 2.5 8"
    ),
    "buried": (
        "CM b\nCE\nGW 1 21 0 -2.5 -1 0 2.5 -1 0.001\nGE -1\n"
        "GN 2 0 0 0 13 0.005\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.0\nXQ\nEN\n"
    ),
}


@pytest.mark.parametrize("deck", sorted(_SG_DECKS))
def test_the_galerkin_fill_never_reads_what_its_family_does_not_honour(
    no_features, monkeypatch, deck
):
    # SinusoidalGalerkinSolver inherits the attributes of pair_reuse and
    # point_cplx_staged from the constructor it shares: reading either in a
    # solve would mean the family honours it after all.
    read = []

    def spy(attr):
        store = f"_spied{attr}"

        def get(self):
            read.append(attr)
            return getattr(self, store)

        def put(self, value):
            setattr(self, store, value)

        return property(get, put)

    for attr in ("_pair_reuse_on", "_pair_reuse_verify", "_point_cplx_staged"):
        monkeypatch.setattr(SinusoidalGalerkinSolver, attr, spy(attr), raising=False)
    built = build_solver(
        parse(_SG_DECKS[deck], dialect="nec2"), basis="sinusoidal-galerkin"
    )
    built.solver.compute_impedance()
    assert read == []


def test_bind_refuses_a_constructor_that_skips_an_honoured_feature():
    class _S:
        pass

    with pytest.raises(TypeError, match="kcl_nullspace"):
        F.bind(
            _S(),
            "bspline",
            **{n: None for n in _constructor_features("bspline") - {"kcl_nullspace"}},
        )
    with pytest.raises(TypeError, match="no feature named"):
        F.bind(_S(), "razor", not_a_feature=None)
    with pytest.raises(ValueError):
        F.bind(_S(), "pulse")


def test_bind_hands_a_part_its_wholes_value(no_features):
    class _S:
        pass

    names = _constructor_features("razor")
    kw = dict.fromkeys(names)
    assert F.bind(_S(), "razor", **{**kw, "pair_reuse": True})["razor_pair_reuse"]
    assert not F.bind(_S(), "razor", **{**kw, "pair_reuse": False})["razor_pair_reuse"]
    # The part's own keyword wins over the whole's.
    own_off = {**kw, "pair_reuse": True, "razor_pair_reuse": False}
    assert not F.bind(_S(), "razor", **own_off)["razor_pair_reuse"]
    # bspline's part takes the same road.
    kw = dict.fromkeys(_constructor_features("bspline"))
    assert F.bind(_S(), "bspline", **{**kw, "pair_reuse": True})["offset_reuse"]


def test_a_part_withheld_outside_its_reach_is_kept_and_not_stamped(
    no_features, monkeypatch
):
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    monkeypatch.setattr(F, "certified", lambda *a, **k: False)
    solver = _construct("razor")
    assert solver.features_withheld == ("razor_pair_reuse",)
    assert solver._pair_reuse == 0
    assert solver.pair_reuse_declined == "not certified on this platform"
    assert F.stamp("python", solver="razor") == ""


def test_the_env_only_features_are_read_where_their_family_runs():
    # A feature with no kwarg is read where it runs, not by a constructor;
    # the modules reading it are its families' own.
    homes = {"razor": {"razor.py", "_razor_exact_plan.py", "_exact_kernel_series.py"}}
    for name, f in F.FEATURES.items():
        if f.kwarg is not None:
            continue
        readers = {
            p.name
            for p in SRC.rglob("*.py")
            if p.name != "_features.py"
            and re.search(rf'["\']{name}["\']', p.read_text(encoding="utf-8"))
        }
        allowed = set().union(*(homes[s] for s in f.solvers))
        assert readers and readers <= allowed, (name, readers)


# --- line 2 -------------------------------------------------------------------------


def _bare_stamp(basis):
    version = _printout._momwire_version()
    variant = _printout._accelerator_variant()
    return f" momwire {version} {basis} {variant}"


def test_line_2_names_a_feature_only_where_the_solver_honours_it(no_features):
    for name, f in F.FEATURES.items():
        for var in (f.env, *f.aliases):
            no_features.delenv(var, raising=False)
        no_features.setenv(f.env, "1")
        for basis in BASES:
            kind = F.basis_solver(basis)
            tail = F.stamp("python", solver=kind)
            named = f"+{name}" in tail
            assert named == (
                F.honours(name, kind) and not F.default_on(name, "python")
            ), (name, basis, tail)
        no_features.delenv(f.env)


def test_kcl_nullspace_on_razor_2p_leaves_line_2_bare(no_features):
    from momwire.dropin.nec5 import _shell

    no_features.setenv("MOMWIRE_X_KCL_NULLSPACE", "1")
    with F.context_scope("dropin"):
        assert _printout.engine_stamp("razor-2p") == _bare_stamp("razor-2p")
        assert _printout.engine_stamp("bspline") == (
            _bare_stamp("bspline") + " +kcl_nullspace uncertified on this platform"
        )
    deck = (
        ROOT
        / "tests"
        / "fixtures"
        / "eznec"
        / "decks"
        / "0036_dipole-in-free-space.nec"
    ).read_text("latin-1")
    line2 = _shell.render(deck, basis="razor-2p").split("\n")[1]
    assert line2 == _bare_stamp("razor-2p")


@pytest.mark.parametrize("basis", ["razor-2p", "bspline", "sinusoidal"])
def test_the_whole_pair_reuse_switch_stays_named_where_it_runs(no_features, basis):
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    with F.context_scope("dropin"):
        assert _printout.engine_stamp(basis) == (
            _bare_stamp(basis) + " +pair_reuse uncertified on this platform"
        )


@pytest.mark.parametrize(
    ("basis", "part"),
    [("razor-2p", "RAZOR_PAIR_REUSE"), ("bspline", "OFFSET_REUSE")],
)
def test_the_whole_is_not_named_where_its_part_is_set_off(no_features, basis, part):
    # The whole acts on these two families only through the part.
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    no_features.setenv(f"MOMWIRE_X_{part}", "0")
    with F.context_scope("dropin"):
        assert _printout.engine_stamp(basis) == _bare_stamp(basis)


@pytest.mark.parametrize(
    ("name", "honoured", "ignored"),
    [
        ("offset_reuse", "bspline", "razor-2p"),
        ("razor_pair_reuse", "razor-2p", "bspline"),
        ("crossing_anchors", "razor-2p", "bspline"),
        ("crossing_above_rule", "bspline", "razor-2p"),
        ("point_cplx_staged", "sinusoidal", "sinusoidal-galerkin"),
    ],
)
def test_each_feature_is_named_on_its_family_alone(
    no_features, name, honoured, ignored
):
    no_features.setenv(F.FEATURES[name].env, "1")
    with F.context_scope("python"):
        assert f"+{name}" in F.stamp(solver=F.basis_solver(honoured))
        assert F.stamp(solver=F.basis_solver(ignored)) == ""


def test_an_unknown_basis_names_every_feature_on(no_features):
    no_features.setenv("MOMWIRE_X_KCL_NULLSPACE", "1")
    with F.context_scope("dropin"):
        assert "+kcl_nullspace" in _printout.engine_stamp("some-basis")


# --- the executables -------------------------------------------------------------


def _exe_bases():
    """Every BASIS an executable is built with (standalone/CMakeLists.txt),
    with the source's default."""
    cmake = (ROOT / "standalone" / "CMakeLists.txt").read_text(encoding="utf-8")
    names = set(re.findall(r'MW_EXE_BASIS_NAME="([a-z0-9-]+)"', cmake))
    src = EXE_SOURCE.read_text(encoding="utf-8")
    names |= set(re.findall(r'#define MW_EXE_BASIS_NAME "([a-z0-9-]+)"', src))
    return names


def test_the_executables_stamp_only_what_their_formulation_honours():
    src = EXE_SOURCE.read_text(encoding="utf-8")
    table = re.search(
        r"const char \*const EXE_IGNORED\[\]\[2\] = \{(.*?)\n\};", src, re.S
    )
    assert table is not None
    ignored = set(re.findall(r'\{"([a-z0-9-]+)", "([a-z0-9_]+)"\}', table[1]))
    carried = re.search(r"const ExeFeature EXE_FEATURES\[\] = \{(.*?)\n\};", src, re.S)
    names = re.findall(r'\{"([a-z0-9_]+)", \d+, (?:true|false)\}', carried[1])
    only_on = re.search(
        r"const char \*const EXE_FEATURE_BASES\[\]\[2\] = \{(.*?)\n\};", src, re.S
    )
    assert only_on is not None
    only_on = dict(re.findall(r'\{"([a-z0-9_]+)", "([a-z0-9-]+)"\}', only_on[1]))
    bases = _exe_bases()
    assert {"razor-2p", "bspline", "sinusoidal"} <= bases
    # A feature an executable carries but cannot reach (the nec42-sinusoidal
    # one, past the levers' reach) is refused, never stamped, so it needs no
    # row. Nor does one `EXE_FEATURE_BASES` keeps to another basis: nec5-bspline
    # refuses razor_pair_reuse and crossing_anchors, and a refused request must
    # stay refused rather than become accepted and unstamped. Every other
    # carried feature its formulation ignores needs a row.
    # Each BASIS's shallowest level: "bspline" names both bspline
    # executables (nec5's at level 6 on Windows, nec42's at 9).
    shallowest = {
        "razor-2p": F._level("exe-nec5-bspline-razor-2p-windows"),
        "bspline": F._level("exe-nec5-bspline-razor-2p-windows"),
        "sinusoidal": F._level("exe-nec42-sinusoidal"),
    }
    assert set(shallowest) == bases
    expected = set()
    for basis in bases:
        kind = F.basis_solver(basis)
        level = shallowest[basis]
        for name in names:
            if F.FEATURES[name].reach < level:
                continue
            if only_on.get(name, basis) != basis:
                assert (basis, name) not in ignored, "refused and ignored"
                continue
            if not F.honours(name, kind):
                expected.add((basis, name))
    assert ignored == expected
    # What keeps nec5-bspline's refusal: the two razor-2p-only rows stay.
    assert only_on == {
        "razor_pair_reuse": "razor-2p",
        "crossing_anchors": "razor-2p",
    }


# --- the near-interface levers --------------------------------------------------


def test_the_levers_are_set_in_one_place():
    for method in (
        BSplineSolver._compute_Z_operator_buried,
        RazorSolver._assemble_Z_from_prepared,
        SinusoidalSolver._assemble_Z,
        SinusoidalGalerkinSolver._assemble_Z,
    ):
        assert method.__wrapped__, method
    for path in ("bspline.py", "razor.py", "sinusoidal.py", "sinusoidal_galerkin.py"):
        text = (SRC / path).read_text(encoding="utf-8")
        assert "_near_interface.factored_columns_scope(" not in text, path
        assert "_near_interface.crossing_table_scope(" not in text, path


@pytest.mark.parametrize("kind", sorted(CLASSES))
def test_the_levers_follow_the_family(no_features, kind):
    solver = _construct(kind)
    for name in _near_interface.LEVERS:
        setattr(solver, f"_{name}", True)
    levers = _near_interface.solver_levers(solver)
    assert levers == {n: F.honours(n, kind) for n in _near_interface.LEVERS}
    if kind.startswith("sinusoidal"):
        # Explicitly off: these families honour none (Steve's call to change).
        assert not any(levers.values())


EXE = os.environ.get("MOMWIRE_RAZOR2P_STANDALONE_EXE")


@pytest.mark.skipif(
    not EXE, reason="MOMWIRE_RAZOR2P_STANDALONE_EXE does not name a built executable"
)
def test_the_razor_exe_runs_crossing_above_rule_unstamped(tmp_path):
    # nec5-razor-2p's executable carries crossing_above_rule and, as Python's
    # razor-2p, ignores it: it runs, and line 2 stays bare, as Python's does.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(F.ENV_PREFIX)
        and k not in {a for f in F.FEATURES.values() for a in f.aliases}
    }
    deck = (
        ROOT
        / "tests"
        / "fixtures"
        / "eznec"
        / "decks"
        / "0036_dipole-in-free-space.nec"
    )
    out = tmp_path / "out.txt"
    r = subprocess.run(
        [EXE, str(deck), str(out)], env=env, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    bare = out.read_bytes().decode("latin-1").splitlines()[1]
    assert "+" not in bare
    env["MOMWIRE_X_CROSSING_ABOVE_RULE"] = "1"
    out.unlink()
    r = subprocess.run(
        [EXE, str(deck), str(out)], env=env, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    assert out.read_bytes().decode("latin-1").splitlines()[1] == bare
