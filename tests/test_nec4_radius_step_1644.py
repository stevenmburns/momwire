"""The NEC-4 dialect's radius-step junction on the point-matched sinusoidal
basis (momwire#1644).

On the NEC-4 dialect (the NEC-4.2 slot, momwire-nec42-sinusoidal)
`SinusoidalSolver(junction_rule="nec42")` takes NEC-4's condition on charge
at a junction -- a small potential-continuity solve per node (NEC-4 theory
manual, UCRL-MA-109338 Pt. II, Section 3.4) -- in place of NEC-2's log rule,
with NEC-4's thin-wire kernel and end caps beside it. The NEC-2 dialect, the
class default and every other family stay bit for bit what they were.

THE TOLERANCE. Two codes that model a junction the same way still differ
where they evaluate the same integrals differently, and that part is visible
with no junction in the deck at all. So each gate deck is compared with its
uniform-radius twin (the same deck at one radius, same mesh, same feed), and
for each part of Z (R and X separately)

    |Z_momwire - Z_nec42| <= |F| + 0.02 |S|

* F, the FLOOR: the same model's residual against NEC-4.2 on the uniform
  twin, the NEC-4 kernel and caps forced on (`_forced_nec4`). It carries no
  junction physics, and it is radius-independent here: +0.04 ohm X at 20 and
  40 segments per 3 m half, -0.29 at 80 and -0.85 at 160, R under 0.005.
* S, the STEP: NEC-4.2's own step response, Z_nec42(step) - Z_nec42(twin),
  15 to 48 ohm of X on the ladder. 2 % of it is what the junction model may
  leave unexplained. Measured with the feed away from the step: at most
  1.5 % (0.10 to 0.57 ohm of X after the floor), where NEC-2's log rule
  misses 45 to 75 % of it.

A feed on the segment BESIDE the step is reported separately and not held to
this: the residual there grows with the mesh (-0.2 ohm at 1:2 x1 to -4.1 ohm
at 1:10 x8) and is not explained -- the first of the two open cases, guarded
only against regression. The second is an equal-radius T junction (+3.0 to
+10.6 ohm, growing with the mesh), which NEC-4.2 solves (a symmetric Y it
does not need to: it solves to equal factors).
"""

from __future__ import annotations

import gzip
import math
import pathlib
import re
import warnings

import numpy as np
import pytest

from momwire import _nec4_junction as nj
from momwire import _sinusoidal_nec4 as sn
from momwire.deck import build_solver, parse
from momwire.eznec._nec4 import solve as nec4_solve
from momwire.sinusoidal import (
    _EULER_GAMMA,
    RadiusStepJunction,
    SinusoidalSolver,
)
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

try:
    from momwire._accel import acc as _acc
except ImportError:  # pragma: no cover - the extension is optional
    _acc = None

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "nec42_radius_step_1644"
K14 = 2 * math.pi * 14e6 / 299792458.0
_NUM = re.compile(r"[-+]?\d\.\d+E[-+]\d\d")
_HAVE_ACC = _acc is not None and hasattr(_acc, "nec4_junction_factors")


def _nec42(name):
    """NEC-4.2's input impedance for a fixture deck, off its printout."""
    with gzip.open(FIX / "nec42" / f"{name}.out.gz", "rt", errors="replace") as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" in line:
            v = _NUM.findall(lines[i + 4])
            return complex(float(v[4]), float(v[5]))
    raise AssertionError(f"{name}: no ANTENNA INPUT PARAMETERS")


def _momwire(name):
    """The NEC-4.2 slot's sinusoidal answer for a fixture deck."""
    text = (FIX / "decks" / f"{name}.nec").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nec4_solve(text, basis="sinusoidal").runs[0].sources[0].impedance


def _forced_nec4(name, monkeypatch):
    """The NEC-4 kernel and caps on a deck the junction rule does not reach
    (a uniform twin): the floor F of the tolerance."""
    factors = SinusoidalSolver._nec4_end_factors

    def forced_factors(self, geom, k, a):
        got = factors(self, geom, k, a)
        if got is not None:
            return got
        ac = 1.0 / (np.log(2.0 / (k * a)) - _EULER_GAMMA)
        base = np.broadcast_to(np.asarray(ac, dtype=float), (geom["n_segs"],))
        return base.copy(), base.copy()

    with monkeypatch.context() as m:
        m.setattr(
            SinusoidalSolver, "_nec4_live", lambda s, g: s.junction_rule == "nec42"
        )
        m.setattr(SinusoidalSolver, "_nec4_end_factors", forced_factors)
        return _momwire(name)


def _within(name, twin, monkeypatch, share=0.02):
    z = _momwire(name)
    ref = _nec42(name)
    floor = _forced_nec4(twin, monkeypatch) - _nec42(twin)
    step = ref - _nec42(twin)
    for part in ("real", "imag"):
        miss = abs(getattr(z - ref, part))
        bound = abs(getattr(floor, part)) + share * abs(getattr(step, part))
        assert miss <= bound, (
            f"{name} {part}: |dZ| {miss:.4f} > {bound:.4f} "
            f"(floor {getattr(floor, part):+.4f}, step {getattr(step, part):+.3f})"
        )


# --- the junction solve ----------------------------------------------------


def _quad(f, lo, hi, z):
    from scipy.integrate import quad

    pts = [z] if lo < z < hi else None
    return quad(f, lo, hi, points=pts, limit=400, epsabs=1e-14, epsrel=1e-12)[0]


@pytest.mark.parametrize(
    "b, z, d",
    [(1e-3, 0.075, 0.15), (1e-3, -0.075, 0.15), (0.02, 0.009, 0.019), (0.1, 0.05, 0.1)],
)
def test_the_solve_integrates_its_elements(b, z, d):
    """Appendix A's reductions against adaptive quadrature of the kernel's
    real part, cos(kR)/R."""
    k = K14

    def g(s):
        r = math.hypot(b, s - z)
        return math.cos(k * r) / r

    tri = _quad(lambda s: (d - s) * g(s), 0.0, d, z)
    assert nj._triangle(k, b, z, d) == pytest.approx(tri, rel=1e-10)
    # the semi-infinite constant: the finite part plus -(pi/2) Y0(k b)
    from scipy.special import y0

    fin = _quad(
        lambda u: math.cos(k * math.hypot(b, u)) / math.hypot(b, u), 0, abs(z), 0
    )
    semi = -0.5 * math.pi * y0(k * b) + math.copysign(fin, z)
    assert nj._semi_infinite(k, b, z) == pytest.approx(semi, rel=1e-10)
    assert nj._bessel_y0(k * b) == pytest.approx(y0(k * b), rel=1e-13)


def test_symmetric_nodes_solve_to_equal_factors():
    y = [
        (math.cos(a), 0.0, math.sin(a))
        for a in (math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6)
    ]
    f = nj.junction_charge_factors(K14, y, [1e-3] * 3, [0.2] * 3, 0.1)
    assert f == pytest.approx([0.1] * 3, rel=1e-12)
    two = nj.junction_charge_factors(
        K14, [(0, 0, -1.0), (0, 0, 1.0)], [1e-3] * 2, [0.1] * 2, 0.1
    )
    assert two == pytest.approx([0.1, 0.1], rel=1e-12)


def test_a_step_puts_less_charge_on_the_thin_side_as_the_mesh_refines():
    """The behaviour (3-41) misses: the thin side's share falls with Delta/a."""
    ratios = []
    for n in (20, 40, 80, 160):
        d = 3.0 / n
        thin, thick = nj.junction_densities(
            K14, [(0, 0, -1.0), (0, 0, 1.0)], [1e-3, 5e-3], [d, d]
        )
        ratios.append(thin / thick)
    assert ratios == sorted(ratios, reverse=True)
    log_rule = (math.log(2 / (K14 * 5e-3)) - _EULER_GAMMA) / (
        math.log(2 / (K14 * 1e-3)) - _EULER_GAMMA
    )
    assert ratios[0] < log_rule


@pytest.mark.skipif(not _HAVE_ACC, reason="the C++ extension is not loaded")
def test_the_cxx_junction_solve_is_pythons_to_the_bit():
    rng = np.random.default_rng(1644)
    for _ in range(200):
        m = int(rng.integers(2, 5))
        dirs = []
        for _ in range(m):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            dirs.append(tuple(float(x) for x in v))
        radii = [float(x) for x in rng.uniform(2e-4, 2e-2, m)]
        lengths = [float(x) for x in rng.uniform(0.01, 0.4, m)]
        first = float(rng.uniform(0.05, 0.2))
        try:
            py = nj.junction_charge_factors(K14, dirs, radii, lengths, first)
        except ValueError as exc:
            with pytest.raises(ValueError, match="too far"):
                _acc.nec4_junction_factors(
                    K14, [list(d) for d in dirs], radii, lengths, first
                )
            assert "too far" in str(exc)
            continue
        cx = _acc.nec4_junction_factors(
            K14, [list(d) for d in dirs], radii, lengths, first
        )
        assert [x.hex() for x in cx] == [x.hex() for x in py]


@pytest.mark.skipif(not _HAVE_ACC, reason="the C++ extension is not loaded")
def test_the_cxx_kernel_and_caps_match_their_numpy_references():
    from momwire._quadrature import leggauss

    rng = np.random.default_rng(16440)
    pts = np.cumsum(rng.normal(size=(61, 3)) * 0.1, axis=0)
    c = 0.5 * (pts[1:] + pts[:-1])
    t = pts[1:] - pts[:-1]
    h = np.linalg.norm(t, axis=1)
    t /= h[:, None]
    a = rng.uniform(1e-3, 2e-2, 60)
    gx, gw = (np.asarray(v) for v in leggauss(8))
    eta = 376.73
    cx = _acc.sinusoidal_field_tensor_nec4(c, t, c, t, h, a, K14, eta, gx, gw, 0)
    ref = sn.nec4_field_components(K14, eta, c, t, c, t, 0.5 * h, a, gx, gw)
    for x, y in zip(cx, ref, strict=True):
        assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(y))
    for _ in range(4):
        pos = rng.normal(size=3) * 0.1
        nrm = rng.normal(size=3)
        nrm /= np.linalg.norm(nrm)
        c0 = complex(*rng.normal(size=2))
        obs = np.vstack([c, pos + np.outer(rng.uniform(-0.5, 0.5, 5), nrm)])
        ot = np.vstack([t, np.tile(nrm, (5, 1))])
        x = _acc.sinusoidal_cap_field(obs, ot, pos, nrm, 0.01, K14, eta, c0)
        y = sn.cap_field_numpy(K14, eta, obs, ot, pos, nrm, 0.01, c0)
        assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(y))


def test_the_per_end_basis_is_nec2s_at_equal_factors_and_no_caps():
    s = SinusoidalSolver(
        wavelength=2 * math.pi / K14,
        wires=[
            np.array([(0, 0, -3.0), (0, 0, 0.0)]),
            np.array([(0, 0, 0.0), (0, 0, 3.0)]),
        ],
        n_per_edge_per_wire=[[20], [20]],
        wire_radius=[1e-3, 5e-3],
        feeds=[(1, 0.075, 1 + 0j)],
    )
    g = s._build_geometry()
    ref = s._basis_coefs(g, s.k)
    a = s._seg_radius(g)
    ac = 1.0 / (np.log(2.0 / (s.k * a)) - _EULER_GAMMA)
    got = sn.basis_coefs_per_end(g, s.k, ac.copy(), ac.copy(), np.zeros_like(ac))
    for key in ("A", "B", "C", "AC"):
        assert np.allclose(got[key], ref[key], rtol=1e-13, atol=1e-15)
    for key in ("starts", "jbasis", "sigma"):
        assert np.array_equal(got[key], ref[key])


# --- what the option touches, and what it does not --------------------------

_STEP_DECK = """CM momwire#1644 stepped dipole
CE
GW 1 20 0 0 -3 0 0 0 {a1}
GW 2 20 0 0 0 0 0 3 {a2}
GE 0
EX 0 2 10 0 1.0 0
FR 0 1 0 0 14.
XQ
EN
"""


def _solver(**kw):
    return SinusoidalSolver(
        wavelength=2 * math.pi / K14,
        wires=[
            np.array([(0, 0, -3.0), (0, 0, 0.0)]),
            np.array([(0, 0, 0.0), (0, 0, 3.0)]),
        ],
        n_per_edge_per_wire=[[20], [20]],
        feeds=[(1, 1.425, 1 + 0j)],
        **kw,
    )


def test_the_default_is_nec2_and_spelling_it_changes_no_bit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RadiusStepJunction)
        a = _solver(wire_radius=[1e-3, 5e-3]).compute_impedance()[0]
        b = _solver(wire_radius=[1e-3, 5e-3], junction_rule="nec2").compute_impedance()[
            0
        ]
    assert _solver(wire_radius=1e-3).junction_rule == "nec2"
    assert a == b


def test_the_option_is_bit_identical_where_it_solves_no_node():
    """A two-wire junction at one radius is uniform by the rule, so the NEC-4
    option builds NEC-2's basis and kernel, bit for bit."""
    a = _solver(wire_radius=1e-3).compute_impedance()[0]
    b = _solver(wire_radius=1e-3, junction_rule="nec42").compute_impedance()[0]
    assert a == b
    ek = _solver(wire_radius=1e-3, extended_kernel=True).compute_impedance()[0]
    ek4 = _solver(
        wire_radius=1e-3, extended_kernel=True, junction_rule="nec42"
    ).compute_impedance()[0]
    assert ek == ek4


def test_the_option_moves_a_step():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RadiusStepJunction)
        a = _solver(wire_radius=[1e-3, 5e-3]).compute_impedance()[0]
    b = _solver(wire_radius=[1e-3, 5e-3], junction_rule="nec42").compute_impedance()[0]
    assert abs(a - b) > 10.0


def test_the_dialects_choose_the_rule():
    text = _STEP_DECK.format(a1=0.001, a2=0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nec2 = build_solver(parse(text, dialect="nec2"), basis="sinusoidal").solver
        nec4 = build_solver(parse(text, dialect="nec4"), basis="sinusoidal").solver
        sg = build_solver(
            parse(text, dialect="nec4"), basis="sinusoidal-galerkin"
        ).solver
    assert nec2.junction_rule == "nec2"
    assert nec4.junction_rule == "nec42"
    assert sg.junction_rule == "nec2"


@pytest.fixture(autouse=True)
def _no_rule_override(monkeypatch):
    """Every test starts from the dialect defaults; the override tests set it."""
    monkeypatch.delenv("MOMWIRE_JUNCTION_RULE", raising=False)


def _deck_z(text, dialect, basis="sinusoidal"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = build_solver(parse(text, dialect=dialect), basis=basis)
    y = b.solver.compute_y_matrix()
    return b.solver, complex(1.0 / y[0, 0])


@pytest.mark.parametrize("dialect", ["nec2", "nec4"])
@pytest.mark.parametrize("rule", ["nec2", "nec42"])
def test_either_rule_on_either_dialect(dialect, rule, monkeypatch):
    """The dialect picks the DEFAULT; `MOMWIRE_JUNCTION_RULE` overrides it on
    both, and the override is exactly the solver option spelled out."""
    text = _STEP_DECK.format(a1=0.001, a2=0.005)
    monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", rule)
    solver, z = _deck_z(text, dialect)
    assert solver.junction_rule == rule
    # the NEC-4 dialect's default extended kernel rides along on the NEC-2
    # rule (the NEC-4.2 rule has one kernel of its own)
    direct = _solver(
        wire_radius=[1e-3, 5e-3], junction_rule=rule, extended_kernel=dialect == "nec4"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zd = complex(1.0 / direct.compute_y_matrix()[0, 0])
    assert z == pytest.approx(zd, rel=1e-9)
    monkeypatch.delenv("MOMWIRE_JUNCTION_RULE")
    default = "nec42" if dialect == "nec4" else "nec2"
    solver, z0 = _deck_z(text, dialect)
    assert solver.junction_rule == default
    assert (z0 == z) == (rule == default)


def test_nec2_through_the_override_is_the_nec2_dialect_to_the_bit(monkeypatch):
    """On a nec2 deck the override's "nec2" builds the very solve the dialect
    always did: the kwarg is not even passed."""
    text = _STEP_DECK.format(a1=0.001, a2=0.005)
    _, z0 = _deck_z(text, "nec2")
    monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", "nec2")
    _, z1 = _deck_z(text, "nec2")
    assert z0 == z1


def test_an_unknown_override_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", "nec4")
    with pytest.raises(ValueError, match="MOMWIRE_JUNCTION_RULE='nec4' names no"):
        _deck_z(_STEP_DECK.format(a1=0.001, a2=0.005), "nec4")


@pytest.mark.parametrize("rule", [None, "nec2", "nec42"])
def test_sin_galerkin_takes_no_rule(rule, monkeypatch):
    """The Galerkin subclass never sees the option, set or not: same Z."""
    text = _STEP_DECK.format(a1=0.001, a2=0.005)
    _, z0 = _deck_z(text, "nec4", basis="sinusoidal-galerkin")
    if rule is not None:
        monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", rule)
    solver, z = _deck_z(text, "nec4", basis="sinusoidal-galerkin")
    assert solver.junction_rule == "nec2" and z == z0


def test_the_class_flag_says_who_takes_the_rule():
    """`takes_junction_rule` is what `build_solver` reads; only the
    point-matched class sets it."""
    assert SinusoidalSolver.takes_junction_rule is True
    assert SinusoidalGalerkinSolver.takes_junction_rule is False


def test_the_warm_server_keys_carry_the_override(monkeypatch):
    import momwire_eznec_client as ez
    import momwire_nec2c_client as n2

    k_ez, k_n2 = ez.config_key(None, 900.0, "nec4"), n2.config_key([], 900.0)
    monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", "nec2")
    assert ez.config_key(None, 900.0, "nec4") != k_ez
    assert n2.config_key([], 900.0) != k_n2
    monkeypatch.setenv("MOMWIRE_JUNCTION_RULE", "nec42")
    assert ez.config_key(None, 900.0, "nec4") not in (k_ez,)


def test_a_typo_and_the_galerkin_lane_are_refused():
    with pytest.raises(ValueError, match="junction_rule must be one of"):
        _solver(wire_radius=1e-3, junction_rule="nec5")
    with pytest.raises(NotImplementedError, match="point-matched lane"):
        SinusoidalGalerkinSolver(
            wavelength=2 * math.pi / K14,
            wires=[np.array([(0, 0, -3.0), (0, 0, 3.0)])],
            n_per_edge_per_wire=[[21]],
            wire_radius=1e-3,
            feeds=[(0, 3.0, 1 + 0j)],
            junction_rule="nec42",
        )


def _advice(**kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solver(wire_radius=[1e-3, 5e-3], **kw)
    return [
        str(w.message) for w in caught if issubclass(w.category, RadiusStepJunction)
    ]


def test_the_nec2_advisory_is_retired_where_the_nec4_rule_serves():
    assert len(_advice()) == 1
    assert _advice(junction_rule="nec42") == []
    assert _advice(junction_rule="nec42", ground_z=-5.0) == []
    assert (
        _advice(
            junction_rule="nec42",
            ground_z=-5.0,
            ground_eps=(13.0, 0.005),
            ground_model="sommerfeld",
        )
        == []
    )


def test_the_advisory_stays_where_the_nec4_rule_does_not_reach():
    (msg,) = _advice(junction_rule="nec42", ground_z=-5.0, ground_eps=(13.0, 0.005))
    assert "does not reach here" in msg and "follows NEC-2" in msg


# --- the gates against licensed NEC-4.2 ------------------------------------

_RATIOS = (("0.5-1", "uni1"), ("1-5", "uni5"), ("1-10", "uni10"))
_MESH = (20, 40, 80, 160)


@pytest.mark.parametrize("n", _MESH)
@pytest.mark.parametrize("step, twin", _RATIOS, ids=["1:2", "1:5", "1:10"])
def test_the_ladder_tracks_nec42_with_the_feed_away_from_the_step(
    step, twin, n, monkeypatch
):
    _within(f"L_step{step}_away_n{n}", f"L_{twin}_away_n{n}", monkeypatch)


@pytest.mark.parametrize("n", _MESH)
@pytest.mark.parametrize("step, twin", _RATIOS, ids=["1:2", "1:5", "1:10"])
def test_a_feed_beside_the_step_is_open_but_far_better_than_the_log_rule(step, twin, n):
    """Open case 1 (not explained, growing with the mesh): guarded against
    regression only, at 5 ohm X and 0.15 ohm R (the log rule is off by 19
    to 196 ohm of X on these decks)."""
    name = f"L_step{step}_step_n{n}"
    d = _momwire(name) - _nec42(name)
    assert abs(d.imag) <= 5.0 and abs(d.real) <= 0.15


@pytest.mark.parametrize("mult", (1, 2, 4, 8))
def test_a_tapered_yagi_element_tracks_nec42(mult, monkeypatch):
    _within(f"Y_elem_x{mult}", f"Yu_elem_x{mult}", monkeypatch)


@pytest.mark.parametrize("mult", (1, 2, 4, 8))
def test_a_tapered_three_element_yagi_tracks_nec42(mult, monkeypatch):
    _within(f"Y3_x{mult}", f"Yu3_x{mult}", monkeypatch)


@pytest.mark.parametrize(
    "name", ["J_Y_n10", "J_Y_n20", "J_Y_n40", "J_L_n21", "J_L_n41", "J_L_n81"]
)
def test_the_y_and_the_stepped_l_track_nec42(name):
    """E4b / E4c: within half an ohm of X and 0.05 of R."""
    d = _momwire(name) - _nec42(name)
    assert abs(d.imag) <= 0.5 and abs(d.real) <= 0.05


@pytest.mark.parametrize("n", (10, 20, 40, 80, 160))
def test_an_equal_radius_t_is_open_but_nearer_than_the_log_rule(n):
    """Open case 2: NEC-4.2 solves an equal-radius T, and this lane's solve
    lands +3.0 to +10.6 ohm of X from it, growing with the mesh (the log
    rule lands -15 to -19). Guarded against regression only."""
    d = _momwire(f"J_T_n{n}") - _nec42(f"J_T_n{n}")
    assert 2.5 <= d.imag <= 11.0 and abs(d.real) <= 0.03


@pytest.mark.parametrize(
    "name",
    [
        "G_hdip_pec_n20",
        "G_hdip_pec_n40",
        "G_hdip_somm_n20",
        "G_hdip_somm_n40",
        "G_vert_pec_n20",
        "G_vert_pec_n40",
    ],
)
def test_steps_over_ground_track_nec42(name):
    """A stepped horizontal dipole over the perfect and the Sommerfeld
    ground, and a grounded stepped vertical over the perfect one: within
    0.6 ohm of X and 0.05 of R (measured at most 0.47 and 0.035)."""
    d = _momwire(name) - _nec42(name)
    assert abs(d.imag) <= 0.6 and abs(d.real) <= 0.05


@pytest.mark.parametrize(
    "deck",
    ["dup_phantom_served_h689.nec", "dup_no_ports_raises_h784.nec"],
)
def test_a_singular_node_keeps_the_log_rule(deck):
    """Coincident members (a wire left in place twice) give the junction
    solve no unique split; that node keeps the log rule and the deck prints
    as it did rather than raising."""
    from momwire.eznec import _nec4

    text = (
        pathlib.Path(__file__).parent / "fixtures" / "razor_printout_small" / deck
    ).read_text()
    out = _nec4.render(text, basis="sinusoidal")
    assert "ZeroDivisionError" not in out
