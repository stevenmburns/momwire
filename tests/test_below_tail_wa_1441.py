"""momwire#1441: the below/below tail's weighted-averages stop.

`_tail_below` walks the λ tail panel by panel between J₀ zeros. A grazing node
(θ → 0, h = |z| + |z′| → 0) decays only as e^{−λh}, so the walk used to run to
λ ≈ ln(1/rtol)/h: ~6.4/tan θ panels, 7,000 at 0.05°, 66x a steep node's work
and two thirds of a buried table's fill time on Dan's soil. Mosig–Michalski
weighted averages (Sidi's W-algorithm on the lattice-aligned partial sums,
remainder estimate ω_n = (−1)^n ξ_n^p e^{−ξ_n h}) stop it in ~12–20 panels.

What is gated here, and why each:

* **Against an extended-precision reference.** The values pinned below are
  mpmath at 40 digits (scratch/wa-tail/mpref.py: the same contour, the tail
  walked to λh = 16 at 40 digits and finished by Wynn in mp, two walk lengths
  agreeing to ≤ 1e-20 of scale). This is the only honest arbiter at grazing:
  the double-precision long walk the study refereed against is itself wrong
  there (its D₂ is a difference of two O(1/λ) terms that cancel to O(λ⁻³),
  and it evaluates D₂ out to λ ~ 1e5 |k_m|).
* **Nothing that goes quiet by itself moves.** The quiet test runs first on
  every panel, so a steep node stops exactly where it did — bit for bit, on
  both machines.
* **C++ vs numpy**, the repo's model: the same rule on both, at tolerance.
* **The transmitted family**, which shares `_tail_below`, keeps its bytes.
* **The tail rule is in every key** that can hold a below/below table, so a
  table filled under one rule is never served under the other.
"""

import numpy as np
import pytest

from momwire import _sommerfeld_below as below
from momwire import _sommerfeld_transmitted as trans

EPS0 = 8.8541878128e-12
C0 = 299792458.0


def _soil(name):
    f, er, sg = {
        "dan": (6.0078e6, 13.0, None),
        "salt1.8": (1.8e6, 81.0, 5.0),
        "fresh1.8": (1.8e6, 80.0, 0.001),
        "salt28": (28.0e6, 81.0, 5.0),
        "fresh28": (28.0e6, 80.0, 0.001),
    }[name]
    et = (
        complex(13.0, -14.96)
        if sg is None
        else complex(er, -sg / (2 * np.pi * f * EPS0))
    )
    return f, et


def _node(soil, r1_wl, th_deg):
    f, et = _soil(soil)
    kp = 2 * np.pi * f / C0
    km = below.k_medium(et, kp)
    lam_m = 2 * np.pi / abs(km)
    th = np.radians(th_deg)
    return et, kp, km, r1_wl * lam_m * np.cos(th), r1_wl * lam_m * np.sin(th)


def _six(soil, r1, th, numpy=False, wa=True, health=None):
    et, kp, _, rho, h = _node(soil, r1, th)
    was = below._FORCE_NUMPY, below._TAIL_WA
    below._FORCE_NUMPY, below._TAIL_WA = numpy, wa
    try:
        return below._six_integrals_below(et, kp, rho, h, health=health)
    finally:
        below._FORCE_NUMPY, below._TAIL_WA = was


def _vec(v, ref):
    return float(np.max(np.abs(v - ref)) / np.max(np.abs(ref)))


needs_accel = pytest.mark.skipif(
    not below._HAVE_BELOW_FILLS_ACCEL, reason="no below/below accelerator"
)

# ---------------------------------------------------------------------------
# The extended-precision reference
# ---------------------------------------------------------------------------
#
# REF[(soil, R1/lam_m, theta_deg)] = the six integrals, mpmath at 40 digits.
# Copied from scratch/wa-tail/spot.jsonl (a subset of the 41 nodes there).
REF = {
    ("dan", 0.02, 0.03): [
        complex(-0.186698496190108, -0.28392858184607567),
        complex(-4.159372815227127, 0.43340610705335),
        complex(4.023399754647152, -0.48448911557464325),
        complex(3.8313909996106195, -0.7526797405858796),
        complex(-0.37463090235084917, 2.4955330746583693),
        complex(-3.7988558315726135, 0.8925408391933324),
    ],
    ("salt1.8", 0.02, 0.03): [
        complex(-3.918459697751531, -3.4659023086991234),
        complex(-71.0477158949812, -3.636092624979381),
        complex(66.94403002019604, -0.4193189262271253),
        complex(63.08193080919203, -3.7221796976955797),
        complex(-0.15205026645033604, 0.1674863497530235),
        complex(-59.163400597377205, 7.186658969593163),
    ],
    ("fresh28", 0.02, 0.05809882393801885): [
        complex(-0.29723040514054494, -3.4088856974998487),
        complex(-40.39994651673485, -2.0881632969872186),
        complex(40.45834630934977, -0.14613664808108706),
        complex(40.01803459933588, -3.4331015200684014),
        complex(0.022048582140901267, 0.3243182524888337),
        complex(-40.225561563771336, 6.811834700461634),
    ],
    ("dan", 2.0, 0.05): [
        complex(0.004894300067734113, 0.00028024519370350006),
        complex(0.07078802980627415, -0.008663093960056758),
        complex(-0.01881207645223782, 0.0005634603744629215),
        complex(-0.0006413739155359137, 0.0014137512504538816),
        complex(-0.1733871227568769, -0.16571554869633065),
        complex(-0.001580217402732787, 0.0007152827044538355),
    ],
    ("salt1.8", 2.0, 0.1): [
        complex(0.00010172163390679126, 0.01664966696775406),
        complex(-1.3181230806766928, 0.08749723149092838),
        complex(0.07395818584667647, -0.07501957715042715),
        complex(-0.00013194687807129054, -0.008384857380943554),
        complex(0.0013776887520182356, 0.01854732798427495),
        complex(2.8430054925480712e-05, -0.008291205649514365),
    ],
    ("dan", 1.0, 0.05): [
        complex(0.024222648011037295, 0.013440002839587722),
        complex(0.017637447297597135, 0.15179850653835722),
        complex(-0.01565434554488659, -0.047462163061018046),
        complex(-0.009274433742241722, 0.000600664653650395),
        complex(0.33035710484966574, -0.4244620384458107),
        complex(-0.020145797076363218, -0.008345077667303863),
    ],
    ("fresh28", 1.0, 1.0): [
        complex(1.5997852063715794, -0.5506946338228726),
        complex(-0.9600211108865414, 1.4751236364323552),
        complex(-0.3769876869138259, -0.24212182589369688),
        complex(-0.004153448052693403, 0.26906948767751354),
        complex(-0.022721923269893057, -0.04350323662039808),
        complex(-1.607779205940511, 0.2999667131948322),
    ],
    ("salt28", 1.0, 0.05): [
        complex(0.017944141113187977, 0.699371786216488),
        complex(-10.507287294917289, 0.9815419285581777),
        complex(1.2407470385743455, -1.195964865517666),
        complex(-0.023254450394972877, -0.2870924302335322),
        complex(0.0014999604495374514, 0.009472406070247607),
        complex(0.004922159421703119, -0.4155430225203502),
    ],
    ("dan", 1.0, 45.0): [
        complex(0.002706683217579276, -0.005203960065823474),
        complex(-0.007466203630397611, -0.03520518581853077),
        complex(0.0009020812876831768, 0.013733849489998166),
        complex(0.0027377310452864827, 0.001629692414251512),
        complex(-0.08893555525230669, 0.08580779861111659),
        complex(-0.004350709622533204, 0.0021247780824061326),
    ],
}

# Vec-norm error (max |got - ref| / max |ref|). MEASURED on Rocketlake over all
# 48 nodes of scratch/wa-tail/spot*.jsonl, both machines: worst 3.4e-12, and
# componentwise worst 4.5e-11, both at salt28, R1 = 0.02 lam_m, 0.042 deg.
# Main's engine at the R1 = 0.02 corner reads up to 4.4e-9 vec-norm (5.9e-8
# componentwise) on the same references.
REF_TOL = 3e-11


@pytest.mark.parametrize("key", sorted(REF))
@pytest.mark.parametrize("use_numpy", [False, True])
def test_wa_meets_the_extended_precision_reference(key, use_numpy):
    if not use_numpy and not below._HAVE_BELOW_FILLS_ACCEL:
        pytest.skip("no below/below accelerator")
    got = _six(*key, numpy=use_numpy)
    ref = np.array(REF[key])
    assert _vec(got, ref) < REF_TOL, (key, _vec(got, ref))


# ---------------------------------------------------------------------------
# What WA changes and what it does not
# ---------------------------------------------------------------------------

GRAZING = [
    ("dan", 1.0, 0.05),
    ("dan", 0.2, 0.1),
    ("salt1.8", 2.0, 0.05),
    ("fresh28", 1.0, 0.5),
    ("fresh1.8", 0.02, 0.03),
]
STEEP = [("dan", 1.0, 45.0), ("dan", 2.0, 80.0), ("salt28", 0.5, 60.0)]


@pytest.mark.parametrize("node", GRAZING)
def test_a_grazing_tail_stops_in_tens_of_panels(node):
    """The point of the change: ~7,000 panels at 0.05 deg before, ~12-20
    after (measured max 2x margin below)."""
    h = below.Health()
    _six(*node, numpy=True, health=h)
    assert h.max_tail_panels <= 40, h.as_dict()
    assert h.nonconvergent == 0 and h.accelerated == 0, h.as_dict()


@pytest.mark.parametrize("node", STEEP)
@pytest.mark.parametrize("use_numpy", [True, False])
def test_a_tail_that_goes_quiet_is_unmoved_bit_for_bit(node, use_numpy):
    if not use_numpy and not below._HAVE_BELOW_FILLS_ACCEL:
        pytest.skip("no below/below accelerator")
    a = _six(*node, numpy=use_numpy, wa=False)
    b = _six(*node, numpy=use_numpy, wa=True)
    assert np.array_equal(a, b)


@needs_accel
@pytest.mark.parametrize("node", GRAZING + STEEP)
def test_cpp_and_numpy_take_the_same_stop(node):
    hc, hn = below.Health(), below.Health()
    c = _six(*node, numpy=False, health=hc)
    n = _six(*node, numpy=True, health=hn)
    # MEASURED worst 1.8e-13 (salt1.8 R1 = 0.02 lam_m, 0.03 deg); the rest
    # read 6e-16 .. 7e-15.
    assert _vec(c, n) < 1e-11, (node, _vec(c, n))
    assert hc.max_tail_panels == hn.max_tail_panels


@pytest.mark.parametrize("th", [0.0125, 0.005, 0.0])
def test_under_the_floor_the_tail_converges_without_the_cap(th):
    """The floor exists because the plain walk exhausts its budget below it
    (`test_grazing_band_838`'s cap gates, which now run the plain walk to reach
    it). Under WA the same node converges in tens of panels, down to h = 0."""
    h = below.Health()
    _six("dan", 1.0, th, numpy=True, health=h)
    assert h.nonconvergent == 0 and h.max_tail_panels <= 40, h.as_dict()


def test_the_transmitted_family_keeps_its_bytes(monkeypatch):
    """`_sommerfeld_transmitted` reuses `_tail_below` and DEPENDS on the Wynn
    fallback for its grazing rows; it must not see the WA rule."""
    et, kp, km, rho, h = _node("dan", 1.0, 0.3)
    args = (et, kp, rho, 0.4 * h, -0.6 * h)
    monkeypatch.setattr(below, "_FORCE_NUMPY", True)
    monkeypatch.setattr(trans, "_FORCE_NUMPY", True, raising=False)
    monkeypatch.setattr(below, "_TAIL_WA", False)
    a = trans._six_integrals_transmitted(*args)
    monkeypatch.setattr(below, "_TAIL_WA", True)
    b = trans._six_integrals_transmitted(*args)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Keys and the binary
# ---------------------------------------------------------------------------


def test_the_tail_rule_is_in_every_below_key(monkeypatch):
    monkeypatch.setattr(below, "_TAIL_WA", True)
    wa = below._tail_rule()
    monkeypatch.setattr(below, "_TAIL_WA", False)
    plain = below._tail_rule()
    assert wa != plain
    plan = below._anchor_plan(complex(13.0, -8.66))
    assert plan.key(0)[-1] == plain
    monkeypatch.setattr(below, "_TAIL_WA", True)
    assert plan.key(0)[-1] == wa


def test_a_grid_filled_under_one_rule_is_not_served_under_the_other(monkeypatch):
    from momwire import _sommerfeld as sm

    monkeypatch.setattr(sm, "_GRID_CACHE", {})
    monkeypatch.setattr(below, "_GRID_CACHE", sm._GRID_CACHE)
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "0")
    monkeypatch.setenv(below.ENV_ANCHORS, "off")
    made = []
    monkeypatch.setattr(
        below, "SommerfeldGridBelow", lambda *a, **k: made.append(object()) or made[-1]
    )
    f = 6.0078e6
    _, et = _soil("dan")
    kp = 2 * np.pi * f / C0
    monkeypatch.setattr(below, "_TAIL_WA", True)
    wa_rule = below._tail_rule()
    g1 = below.get_grid_below(et, kp, 5.0, 2 * np.pi * f)
    assert below.get_grid_below(et, kp, 5.0, 2 * np.pi * f) is g1
    monkeypatch.setattr(below, "_TAIL_WA", False)
    g2 = below.get_grid_below(et, kp, 5.0, 2 * np.pi * f)
    assert g2 is not g1 and len(made) == 2
    assert {k[-1] for k in sm._GRID_CACHE} == {wa_rule, ("tail", "wynn")}


@needs_accel
def test_the_binary_declares_the_rule_and_refuses_a_bad_one():
    acc = below._acc
    assert acc.below_tail_wa_1441
    et, kp, km, rho, h = _node("dan", 1.0, 0.05)
    with pytest.raises(ValueError, match="wa_pow"):
        acc.below_six_integrals_batch(
            kp,
            km,
            np.array([rho]),
            np.array([h]),
            1e-11,
            below._ADAPT_DEPTH,
            below._DETOUR,
            below._GX,
            below._GW,
            False,
            1e-9,
            below._ADAPT_DEPTH_COARSE,
            below._DETOUR_COARSE,
            below._GXC,
            below._GWC,
            below._MAX_TAIL_PANELS,
            wa_pow=np.zeros(5),
        )
