"""The far field of a source BELOW the interface (momwire#570).

``_far_moments`` used to be an above-ground-only readout that did not say so:
it summed ``exp(+jk r̂·r_n)`` in the UPPER medium's wavenumber and added a
Fresnel-weighted image, and handed a buried element the same treatment. That
prints a plausible pattern rather than failing, which is why both seams
refused the whole table instead. What replaced the refusal is
``transmitted_factors`` / ``transmitted_moments``: the stationary-phase value
of the below-to-above Sommerfeld family, written in the Fresnel spelling.

Four gates, and each one is a different kind of evidence:

* **G1, the ε̃ = 1 collapse** — algebra. With no medium there, the transmitted
  moment must BE the free-space moment, so the new path is pinned to the old
  one at the only parameter value where they have to agree.
* **G2, the oracle** — momwire's own numerical transmitted integrals, at two
  ranges, extrapolated in 1/R. This is the gate that would catch a wrong
  factor, and it is run with two adversarial spellings beside it so a reader
  can see the bar has teeth rather than being told so.
* **G3, bit identity** — an above-ground deck's pattern must not move by one
  ULP. Pinned twice: against literals computed before the change, and against
  a transmitted helper monkeypatched to raise.
* **G4, the two limits** — the horizon, where every factor vanishes, and the
  grounds with no lower medium, which raise instead of imaging.

The conventions are the readout's own: ``e^{+jωt}``, air above ``z =
ground_z`` at a real ``k_p``, soil below at ``k_m`` from
``_sommerfeld_below.k_medium``, and ``E = -jηk/(4π)·e^{-jkR}/R·(M_θ θ̂ +
M_φ φ̂)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from momwire import _far_readout
from momwire._far_readout import (
    Ground,
    _eps_complex,
    _far_moments,
    transmitted_factors,
    transmitted_moments,
)
from momwire._sommerfeld_below import k_medium
from momwire._sommerfeld_transmitted import _six_integrals_transmitted

C0 = 299_792_458.0
MU0 = 4.0e-7 * math.pi
SOIL_A = (13.0, 0.005)


def medium(freq_hz, eps_r, sigma):
    """``(eps_t, k_p, k_m, omega)`` — the readout's own spelling of a soil."""
    omega = 2.0 * math.pi * freq_hz
    eps_t = _eps_complex(eps_r, sigma, freq_hz)
    k_p = omega / C0
    return eps_t, k_p, k_medium(eps_t, k_p), omega


def free_space_moments(mid, moment, k, theta, phi):
    """The DIRECT far-field moment projected on ``θ̂`` and ``φ̂`` — written
    out here rather than borrowed, so G1 compares the transmitted path with
    the textbook rather than with another branch of the same function."""
    sin_t, cos_t = np.sin(theta)[:, None], np.cos(theta)[:, None]
    cos_p, sin_p = np.cos(phi)[None, :], np.sin(phi)[None, :]
    shape = (theta.size, phi.size)
    rhat = np.stack(
        [sin_t * cos_p, sin_t * sin_p, np.broadcast_to(cos_t, shape)], axis=-1
    )
    th = np.stack(
        [cos_t * cos_p, cos_t * sin_p, -np.broadcast_to(sin_t, shape)], axis=-1
    )
    ph = np.stack(
        [
            -np.broadcast_to(sin_p, shape),
            np.broadcast_to(cos_p, shape),
            np.zeros(shape),
        ],
        axis=-1,
    )
    carrier = np.exp(1j * k * np.einsum("ijc,nc->ijn", rhat, mid))
    total = np.einsum("ijn,nc->ijc", carrier, moment)
    return np.sum(total * th, axis=-1), np.sum(total * ph, axis=-1)


# ----------------------------------------------------------------------
# G1 — the ε̃ = 1 collapse
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ground_z", [0.0, 3.25], ids=["plane-at-0", "plane-lifted"])
def test_g1_at_eps_one_the_transmitted_moment_is_the_free_space_moment(
    ground_z, record_property
):
    """No medium, no interface: the transmitted moment must equal the direct
    one to round-off on a 20-element set of arbitrary complex moments.

    ``ground_z`` is swept because the two legs are referenced to different
    things — the lateral one to the origin, the depth one to the PLANE — and
    the factor that reconciles them is 1 at ``ground_z = 0``, which is the
    only value either NEC dialect passes. A lifted plane is the only way to
    see it, and without it the collapse would hold only up to a constant
    phase.
    """
    _eps, k_p, k_m, _w = medium(7e6, 1.0, 0.0)
    assert k_m == k_p, "eps_tilde = 1 must leave the lower medium's k untouched"

    rng = np.random.default_rng(570)
    mid = rng.uniform(-3.0, 3.0, (20, 3))
    mid[:, 2] = ground_z - rng.uniform(0.05, 2.0, 20)
    moment = rng.normal(size=(20, 3)) + 1j * rng.normal(size=(20, 3))
    theta = np.radians(np.linspace(0.0, 89.0, 90))
    phi = np.radians(np.linspace(0.0, 360.0, 73))

    got = transmitted_moments(mid, moment, k_p, k_m, theta, phi, ground_z)
    want = free_space_moments(mid, moment, k_p, theta, phi)
    scale = max(np.max(np.abs(want[0])), np.max(np.abs(want[1])))
    rel = float(
        max(np.max(np.abs(got[0] - want[0])), np.max(np.abs(got[1] - want[1]))) / scale
    )
    record_property(f"g1_moment_rel_at_ground_z_{ground_z:g}", f"{rel:.3e}")
    assert rel < 1e-12, f"transmitted moment is {rel:.3e} from the free-space one"


def test_g1_at_eps_one_the_factors_are_one_cos_and_minus_sin(record_property):
    """The same statement one level down, where a misreading is legible:
    ``T_e`` is the φ̂ component of a unit horizontal moment, ``T_h`` its θ̂
    component (``cosθ``) and ``T_v`` a unit vertical moment's (``-sinθ``)."""
    _eps, k_p, k_m, _w = medium(7e6, 1.0, 0.0)
    theta = np.radians(np.linspace(0.0, 89.0, 90))
    t_e, t_h, t_v = transmitted_factors(theta, k_p, k_m)
    worst = float(
        max(
            np.max(np.abs(t_e - 1.0)),
            np.max(np.abs(t_h - np.cos(theta))),
            np.max(np.abs(t_v + np.sin(theta))),
        )
    )
    record_property("g1_factor_abs", f"{worst:.3e}")
    assert worst < 1e-12


# ----------------------------------------------------------------------
# G2 — the oracle: momwire's own numerical transmitted integrals
# ----------------------------------------------------------------------

_G2_FREQ = 7e6
_G2_THETAS_DEG = (10.0, 30.0, 50.0)
# The two elementary sources the six integrals are written for: a horizontal
# one read off-meridian (so the φ̂ component is loaded too) and a vertical one.
_G2_SOURCES = (("HED", 0.15, 0.6), ("VED", 0.6, 0.0))


def _numerical_far(eps_t, k_p, omega, radius, theta, phi, depth, kind):
    """``R·e^{+jk_pR}·(E_θ, E_φ)`` from the six numerical λ-integrals.

    The assembly is the transmitted family's own: ``E_ρ^V = C₁·I0``,
    ``E_z^V = C₁·I1``, ``E_ρ^H = C₁cosφ(I2+I5)``, ``E_φ^H = -C₁sinφ(I3+I5)``,
    ``E_z^H = -C₁cosφ·I4`` with ``C₁ = -jωμ₀/4π``; the spherical components
    then follow from ``θ̂`` and ``r̂`` in the ρ-z plane.
    """
    rho, z = radius * math.sin(theta), radius * math.cos(theta)
    i0, i1, i2, i3, i4, i5 = _six_integrals_transmitted(
        eps_t, k_p, rho, z, -depth, rtol=1e-11
    )
    c1 = -1j * omega * MU0 / (4.0 * math.pi)
    if kind == "VED":
        e_rho, e_phi, e_z = c1 * i0, 0j, c1 * i1
    else:
        e_rho = c1 * math.cos(phi) * (i2 + i5)
        e_phi = -c1 * math.sin(phi) * (i3 + i5)
        e_z = -c1 * math.cos(phi) * i4
    e_theta = e_rho * math.cos(theta) - e_z * math.sin(theta)
    return np.array([e_theta, e_phi]) / (np.exp(-1j * k_p * radius) / radius)


def _closed_far(k_p, k_m, omega, theta, phi, depth, kind):
    """The same quantity from the closed form under test."""
    mid = np.array([[0.0, 0.0, -depth]])
    moment = (
        np.array([[1.0, 0.0, 0.0]]) if kind == "HED" else np.array([[0.0, 0.0, 1.0]])
    )
    m_theta, m_phi = transmitted_moments(
        mid, moment, k_p, k_m, np.array([theta]), np.array([phi]), 0.0
    )
    c1 = -1j * omega * MU0 / (4.0 * math.pi)
    return np.array([c1 * m_theta[0, 0], c1 * m_phi[0, 0]])


def test_g2_the_closed_form_is_the_limit_of_momwire_s_own_integrals(record_property):
    """The transmitted field at 40 and 80 free-space wavelengths, extrapolated
    in 1/R, against the closed form.

    **Why Richardson and not a bar at one range.** The approach to the far
    zone is a clean 1/R here — measured halving ratios 2.000 on every rung —
    with a constant in front that GROWS toward grazing, so a single-range
    absolute bar tests the range chosen at least as much as the formula:
    the raw residual at 80 λ₀ is 2.0e-3 at θ = 10° and 6.1e-3 at 70° for the
    same correct closed form. ``2·E(80λ₀) - E(40λ₀)`` removes that leading
    term and leaves a bar that is about the physics.
    """
    eps_t, k_p, k_m, omega = medium(_G2_FREQ, *SOIL_A)
    lam0 = 2.0 * math.pi / k_p
    worst = 0.0
    for kind, depth, phi in _G2_SOURCES:
        for deg in _G2_THETAS_DEG:
            theta = math.radians(deg)
            near = _numerical_far(eps_t, k_p, omega, 40 * lam0, theta, phi, depth, kind)
            far = _numerical_far(eps_t, k_p, omega, 80 * lam0, theta, phi, depth, kind)
            richardson = 2.0 * far - near
            want = _closed_far(k_p, k_m, omega, theta, phi, depth, kind)
            rel = float(np.linalg.norm(richardson - want) / np.linalg.norm(want))
            record_property(f"g2_rel_{kind}_{deg:.0f}", f"{rel:.3e}")
            worst = max(worst, rel)
    record_property("g2_worst_rel", f"{worst:.3e}")
    assert worst < 3e-5, f"closed form is {worst:.3e} from the extrapolated integral"


@pytest.mark.parametrize("spelling", ["depth-leg sign", "T_v sign"])
def test_g2_the_bar_has_teeth(spelling, record_property):
    """The same comparison against two DELIBERATELY wrong spellings.

    A gate that only ever sees the right answer cannot distinguish a tight bar
    from a vacuous one. ``exp(+j k_mz d)`` is the depth leg with the soil
    turned into a gain medium — a sign no reader can check by inspection — and
    a negated ``T_v`` is the polarisation error that survives every
    magnitude-only check.

    Both wrong answers are built from the RIGHT one in closed form rather than
    by patching the module, which is what makes them exact: the source is one
    element at one depth, so flipping the depth leg multiplies its moment by
    ``exp(+2j k_mz d)`` and nothing else, and a VED's whole moment is ``T_v``,
    so negating that factor negates the field. ``T_v`` multiplies ``m_z``
    alone, so it is probed on the VERTICAL source only — scoring it on the
    horizontal one would compare the right answer with itself and pass.
    """
    eps_t, k_p, k_m, omega = medium(_G2_FREQ, *SOIL_A)
    lam0 = 2.0 * math.pi / k_p
    kinds = ("HED", "VED") if spelling == "depth-leg sign" else ("VED",)
    closest = math.inf
    for kind, depth, phi in _G2_SOURCES:
        if kind not in kinds:
            continue
        for deg in _G2_THETAS_DEG:
            theta = math.radians(deg)
            near = _numerical_far(eps_t, k_p, omega, 40 * lam0, theta, phi, depth, kind)
            far = _numerical_far(eps_t, k_p, omega, 80 * lam0, theta, phi, depth, kind)
            richardson = 2.0 * far - near
            right = _closed_far(k_p, k_m, omega, theta, phi, depth, kind)
            if spelling == "depth-leg sign":
                k_mz = complex(_far_readout._k_mz(np.array([theta]), k_p, k_m)[0])
                wrong = right * np.exp(2j * k_mz * depth)
            else:
                wrong = -right
            closest = min(
                closest,
                float(np.linalg.norm(richardson - wrong) / np.linalg.norm(wrong)),
            )
    record_property(f"g2_adversarial_{spelling.replace(' ', '_')}", f"{closest:.3e}")
    assert closest > 1e-2, (
        f"the {spelling} error comes within {closest:.3e} of the oracle, so the "
        "3e-5 bar above is not what is holding the formula up"
    )


# ----------------------------------------------------------------------
# G3 — an above-ground pattern moves by nothing
# ----------------------------------------------------------------------

# Computed on the commit BEFORE the transmitted path existed, at full repr
# precision so `==` is the right comparison. Four grounds, because the change
# rearranged the function's head and each branch has to be shown untouched.
_G3_MID = np.array(
    [
        [0.0, 0.0, 3.0],
        [1.5, -0.75, 0.25],
        [-2.0, 2.0, 11.5],
        [0.3, 0.4, 0.5],
    ]
)
_G3_MOMENT = np.array(
    [
        [0.5 + 0.25j, -0.125 - 0.75j, 1.0 + 0.0j],
        [-1.0 + 0.0j, 0.25 + 0.5j, -0.375 + 0.125j],
        [0.0 + 1.0j, 0.75 - 0.25j, 0.125 + 0.625j],
        [0.625 - 0.5j, -0.25 + 0.0j, 0.875 + 0.375j],
    ]
)
_G3_THETA = np.radians(np.array([13.0, 61.0, 90.0]))
_G3_PHI = np.radians(np.array([0.0, 118.0]))
_G3_K = 2.0 * np.pi / 42.8
_G3_GROUNDS = {
    "free": Ground("free"),
    "pec": Ground("pec"),
    "refl": Ground("refl", 13.0, 0.005),
    "sommerfeld": Ground("sommerfeld", 13.0, 0.005),
}
_G3_PIN = {
    ("free", "theta"): [
        [
            (-1.1235253681580926 - 0.34424205655495876j),
            (0.5611641731831213 + 0.3011172907306126j),
        ],
        [
            (-1.2524663835841507 - 0.868788028181473j),
            (-0.37308327431121063 - 1.0209148920032198j),
        ],
        [
            (-1.7649229140835958 - 1.015308341908841j),
            (-1.393883428104243 - 1.224014242695155j),
        ],
    ],
    ("free", "phi"): [
        [
            (0.39960899195747857 + 0.5171451412638356j),
            (0.7055192285666871 - 0.09847417264429165j),
        ],
        [
            (0.6954377416130626 - 0.048914101199708564j),
            (0.42084698098767404 - 0.5167225188828779j),
        ],
        [
            (0.4055437671279228 - 0.6748293933967909j),
            (-0.15599356177425622 - 0.6656686596105943j),
        ],
    ],
    ("pec", "theta"): [
        [
            (-2.6930035466267688 + 0.352373075212265j),
            (1.160156442188728 + 0.8855798142614788j),
        ],
        [
            (-3.5777985000386217 - 1.1998323417401342j),
            (-2.052259320781922 - 1.2487648989017606j),
        ],
        [
            (-3.5298458281671916 - 2.030616683817682j),
            (-2.787766856208486 - 2.44802848539031j),
        ],
    ],
    ("pec", "phi"): [
        [
            (1.1841341981610867 + 1.3360431493160267j),
            (1.4307705667245392 - 0.8904027906695221j),
        ],
        [
            (0.9305949571484712 + 0.9011047848099848j),
            (1.1529033112319844 - 0.2715830395465021j),
        ],
        [
            (3.3306690738754696e-16 + 1.1102230246251565e-16j),
            (1.9605374942586781e-16 - 1.0424362768233018e-16j),
        ],
    ],
    ("refl", "theta"): [
        [
            (-2.024337244192792 + 0.28301770457112524j),
            (1.0079005370041179 + 0.595463750596719j),
        ],
        [
            (-2.159574468951825 - 0.607087195846127j),
            (-1.026281105175534 - 0.8278265112727713j),
        ],
        [
            (-8.600139172102991e-16 - 4.124456793128058e-16j),
            (-8.406856879770307e-16 - 4.83663512680532e-16j),
        ],
    ],
    ("refl", "phi"): [
        [
            (1.000644780021258 + 0.9513002218945596j),
            (1.0785788217012138 - 0.6932334510493225j),
        ],
        [
            (0.9551954962361684 + 0.6989967078067943j),
            (1.028722513559535 - 0.3734290777266752j),
        ],
        [
            (3.3306690738754696e-16 + 1.1102230246251565e-16j),
            (1.6653345369377348e-16 - 1.1102230246251565e-16j),
        ],
    ],
    ("sommerfeld", "theta"): [
        [
            (-2.024337244192792 + 0.28301770457112524j),
            (1.0079005370041179 + 0.595463750596719j),
        ],
        [
            (-2.159574468951825 - 0.607087195846127j),
            (-1.026281105175534 - 0.8278265112727713j),
        ],
        [
            (-8.600139172102991e-16 - 4.124456793128058e-16j),
            (-8.406856879770307e-16 - 4.83663512680532e-16j),
        ],
    ],
    ("sommerfeld", "phi"): [
        [
            (1.000644780021258 + 0.9513002218945596j),
            (1.0785788217012138 - 0.6932334510493225j),
        ],
        [
            (0.9551954962361684 + 0.6989967078067943j),
            (1.028722513559535 - 0.3734290777266752j),
        ],
        [
            (3.3306690738754696e-16 + 1.1102230246251565e-16j),
            (1.6653345369377348e-16 - 1.1102230246251565e-16j),
        ],
    ],
}


@pytest.mark.parametrize("name", sorted(_G3_GROUNDS))
def test_g3_an_above_ground_pattern_is_bit_identical_to_the_pin(name):
    """Not "close": EQUAL. The served-vs-stock exactness gates compare
    printed digits, so a change that moved an above-ground pattern in the
    16th place would redden them somewhere far from here."""
    m_theta, m_phi = _far_moments(
        _G3_MID,
        _G3_MOMENT,
        _G3_K,
        _G3_THETA,
        _G3_PHI,
        _G3_GROUNDS[name],
        0.0,
        7.0e6,
    )
    assert np.array_equal(m_theta, np.array(_G3_PIN[(name, "theta")]))
    assert np.array_equal(m_phi, np.array(_G3_PIN[(name, "phi")]))


@pytest.mark.parametrize("name", sorted(_G3_GROUNDS))
def test_g3_the_transmitted_path_is_never_entered_without_a_buried_element(
    name, monkeypatch
):
    """The pin above says the numbers did not move; this says WHY, and keeps
    saying it if someone re-pins the numbers."""

    def explode(*_args, **_kwargs):
        raise AssertionError("the transmitted path ran on an all-above deck")

    monkeypatch.setattr(_far_readout, "transmitted_moments", explode)
    _far_moments(
        _G3_MID,
        _G3_MOMENT,
        _G3_K,
        _G3_THETA,
        _G3_PHI,
        _G3_GROUNDS[name],
        0.0,
        7.0e6,
    )


def test_g3_an_element_exactly_in_the_plane_stays_above(monkeypatch):
    """A ground contact's last midpoint, or a radial hub, can land in the
    plane to the bit. It is the image that represents such an element, not
    the transmitted factors — and a tolerance of exactly zero would send it
    the other way on the first rounding error."""

    def explode(*_args, **_kwargs):
        raise AssertionError("an in-plane element took the transmitted path")

    monkeypatch.setattr(_far_readout, "transmitted_moments", explode)
    mid = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    moment = np.array([[1.0 + 0j, 0j, 0j], [0j, 0j, 1.0 + 0j]])
    _far_moments(
        mid,
        moment,
        _G3_K,
        _G3_THETA,
        _G3_PHI,
        Ground("sommerfeld", 13.0, 0.005),
        0.0,
        7.0e6,
    )


# ----------------------------------------------------------------------
# G4 — the two limits
# ----------------------------------------------------------------------


def test_the_horizon_kills_every_factor(record_property):
    """At ``θ = 90°`` the transmitted wave has nowhere to go, and the
    finite-ground image pattern vanishes on the same row (NEC prints
    -999.99 there). Approaching it, each factor falls like ``cosθ``."""
    _eps, k_p, k_m, _w = medium(7e6, *SOIL_A)
    at_horizon = transmitted_factors(np.array([0.5 * math.pi]), k_p, k_m)
    worst = float(max(abs(f[0]) for f in at_horizon))
    record_property("horizon_max_factor", f"{worst:.3e}")
    assert worst < 1e-14

    near = transmitted_factors(np.radians(np.array([89.5, 89.9])), k_p, k_m)
    expected = math.cos(math.radians(89.9)) / math.cos(math.radians(89.5))
    for factor in near:
        ratio = abs(factor[1]) / abs(factor[0])
        assert 0.18 <= ratio <= 0.22, f"{ratio:.4f} against cos ratio {expected:.4f}"


@pytest.mark.parametrize(
    "kind,needle",
    [
        ("pec", "field inside a perfect conductor"),
        ("refl", "UPPER half-space alone"),
    ],
)
def test_a_below_element_over_a_ground_with_no_lower_medium_raises(kind, needle):
    """Both fronts refuse these decks at FILL time, each with its own
    sentence, so nothing reaches here by a route a user can take. The guard
    is for the route a CALLER can take — and the failure it prevents is the
    quiet one: imaging the element would print a plausible pattern for a
    source that cannot exist there."""
    mid = np.array([[0.0, 0.0, -0.5]])
    moment = np.array([[1.0 + 0j, 0j, 0j]])
    with pytest.raises(ValueError, match="no lower medium") as excinfo:
        _far_moments(
            mid,
            moment,
            _G3_K,
            _G3_THETA,
            _G3_PHI,
            Ground(kind, 13.0, 0.005),
            0.0,
            7.0e6,
        )
    assert needle in str(excinfo.value)


def test_a_mixed_deck_is_the_sum_of_its_two_paths(record_property):
    """An elevated feed over a buried counterpoise is the common shape, and
    the split is per ELEMENT. Two claims in one: the below element does not
    reach the image sum (so the above deck's own answer is unchanged by
    burying something), and the total is the plain sum of the two paths."""
    ground = Ground("sommerfeld", *SOIL_A)
    _eps, k_p, k_m, _w = medium(7e6, *SOIL_A)
    theta = np.radians(np.array([15.0, 55.0, 85.0]))
    phi = np.radians(np.array([0.0, 90.0, 200.0]))
    above_mid = np.array([[0.0, 0.0, 4.0], [1.0, -1.0, 0.75]])
    above_moment = np.array([[0.3 + 0.1j, 0j, 1.0 + 0j], [0j, 0.5 - 0.2j, 0.25 + 0j]])
    below_mid = np.array([[0.0, 0.0, -0.2], [2.0, 0.5, -0.9]])
    below_moment = np.array([[1.0 + 0j, 0j, 0j], [0j, 0.4 + 0.6j, -0.7 + 0j]])

    args = (k_p, theta, phi, ground, 0.0, 7.0e6)
    only_above = _far_moments(above_mid, above_moment, *args)
    mixed = _far_moments(
        np.vstack([above_mid, below_mid]),
        np.vstack([above_moment, below_moment]),
        *args,
    )
    transmitted = transmitted_moments(
        below_mid, below_moment, k_p, k_m, theta, phi, 0.0
    )
    for got, above, below in zip(mixed, only_above, transmitted):
        assert np.allclose(got, above + below, rtol=0.0, atol=1e-15)
    lifted = np.max(np.abs(np.array(mixed) - np.array(only_above)))
    record_property("mixed_deck_below_contribution", f"{float(lifted):.3e}")
    assert lifted > 1e-3, "the buried half must actually move the pattern"
