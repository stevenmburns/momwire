"""The far-zone readout — one owner.

Current moments to E(THETA)/E(PHI), the PEC/Fresnel ground image, the
linear/circular cliff split, the mixed-potential near field, the
axial-ratio/tilt/sense polarisation ellipse and the gain-dB floor: two seams
need this arithmetic — the NEC-2 portal's ``RP``/``NE``/``NH`` tables and the
EZNEC seam's pattern and near-field requests — and until momwire#719 U1 only
one of them owned it. The second seam paid for the first copy by reaching
across a package boundary for eight private names out of
``momwire.portal._portal`` (the ranked extraction backlog, momwire#429, is
what marked that reach as debt rather than as done). This module is the
payoff: the physics moves to a home neither seam is privileged over, and
``momwire.portal._portal`` imports it back for its own internal use and
re-exports the same names, so ``from momwire.portal._portal import
_far_moments``-style call sites are unaffected.

Every derivation comment below is unmoved from its old home in
``portal/_portal.py`` — only the module boundary is new.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Free-space impedance and permittivity. This module is ``EPS0``'s one
# owner too: the portal's ANTENNA ENVIRONMENT block also needs it (the
# complex dielectric constant line) and imports it from here through the
# same compat block that re-exports the readout — a second spelling is the
# thing that drifts (momwire#643).
ETA0 = 376.730_313_668
EPS0 = 8.854_187_817e-12

# NEC-2's two degenerate-value thresholds. They are both spelt 1e-20 and they
# are NOT the same test — a fact that hides completely while every pattern
# fixture is taken at one range and one wavelength, and stops hiding the
# moment an ``RFLD = 0`` deck arrives (issue #802).
#
# * ``_GAIN_FLOOR2`` is ``DB10``'s — ``nec2dx.f``, ``FUNCTION DB10``:
#   ``IF (X.LT.1.D-20) GO TO 2 ... 2 DB10=-999.99``. It clamps the LINEAR
#   POWER GAIN, the number about to be logged, so a direction below -200 dB
#   prints -999.99. Gain never depended on the range, so this floor is
#   range-free too.
# * ``_FIELD_FLOOR2`` is the polarisation block's — ``RDPAT``:
#   ``IF (ETHM2.GT.1.D-20.OR.EPHM2.GT.1.D-20) GO TO 11``, and the fall-through
#   sets ``ISENS=HBLK``. That blanks AXIAL/TILT/SENSE, and a blank SENSE is
#   exactly what makes a row 11 tokens instead of 12. It is applied to the
#   field as ``FFLD`` returns it — BEFORE the ``*WLAM`` and the ``*EXRM``
#   (``EXRM=1./RFLD``) that turn it into the volts-per-metre the table prints
#   — so it is a fixed bar on the antenna, not on the reading.
#   :func:`_pattern_lines` rescales the printed field back to that basis
#   before testing it.
#
# Read off ``dipole_rp_pattern.out`` (E_theta = 5.4196E-15 at theta = 180 ->
# -999.99 and blank) and confirmed against ``dipole_rp_crossed_quadrature.out``
# (E_theta = 2.7098E-15 -> VERTC -999.99 but SENSE still LINEAR, because
# E_phi is large). Grammar doc §4.14.
_FIELD_FLOOR2 = 1.0e-20
_GAIN_FLOOR2 = 1.0e-20
_GAIN_FLOOR_DB = -999.99


@dataclass(frozen=True)
class Ground:
    """The deck's ``GN`` card, reduced to what both the printout and momwire
    need. ``kind`` is one of free / pec / refl / sommerfeld."""

    kind: str = "free"
    eps_r: float = 0.0
    sigma: float = 0.0

    @classmethod
    def from_model(cls, ground) -> Ground:
        """The printout's view of ``DeckModel.ground``.

        The model speaks momwire's ground vocabulary — ``None`` / ``"pec"`` /
        ``("finite-fast" | "finite", eps_r, sigma)`` — and this record is the
        NEC-facing half: the four words the ANTENNA ENVIRONMENT block prints
        and the two constants that go with them. The ``GN`` card's own reading
        (which type means which model, and the radial-screen refusal) lives in
        the dialect now, not here.
        """
        if ground is None:
            return cls("free")
        if ground == "pec":
            return cls("pec")
        model, eps_r, sigma = ground
        return cls("refl" if model == "finite-fast" else "sommerfeld", eps_r, sigma)


def _image_moments(mid, moment, ground_z):
    """The geometric PEC image of a set of current moments across ``z = z0``.

    Horizontal components flip, the vertical one does not — the standard
    image, and the same convention ``engines/momwire.py::_evaluate_M_perp``
    uses, so the finite-ground Fresnel step below can be written as a
    correction to it.
    """
    mid_img = mid.copy()
    mid_img[:, 2] = 2.0 * ground_z - mid[:, 2]
    return mid_img, moment * np.array([-1.0, -1.0, 1.0])


def _image_coeffs(eps_r, sigma, freq_hz, rx, ry, rz):
    """``(rho_h, rho_v)`` — the IMAGE-CURRENT multipliers for one medium.

    These are ``FFLD``'s ``RRH`` and ``-RRV``, not the textbook Fresnel pair:
    they multiply the geometric image (horizontal components already flipped
    by :func:`_image_moments`), which is why perfect ground is ``(-1, +1)``
    here and ``RRV=RRH=-(1.,0.)`` in NEC. Algebraically identical — NEC writes
    the ratio ``ZRATI = 1./SQRT(EPSC)`` (theory manual eq. 179's ``Z_R``)
    where this writes ``q = sqrt(eps_c - sin²θ)``, and ``ZRATI*ZRSIN`` reduces
    to ``q/eps_c`` — but the sign convention is ours, and the mapping is the
    thing to check first if a reflected field ever comes out inverted.

    The pair is applied the way theory-manual eq. 181 composes them,
    ``E_R = R_V·E_I + (R_H - R_V)(E_I·p̂)p̂`` with ``p̂`` normal to the plane of
    incidence — ``FFLD``'s ``CDP=(TIX*PHX+TIY*PHY)*(RRH-RRV)`` lines.

    Passing ``eps_r = sigma = 0`` (a ``GD`` with an empty second medium) makes
    ``q`` collapse to ``j·sinθ`` and the vertical ratio a 0/0 at the zenith.
    The oracle does the same thing from the other side — ``zrati2`` goes to
    infinity and its whole pattern table prints ``nan`` — so nothing here
    tries to rescue a deck that asks for a cliff into vacuum.
    """
    omega = 2.0 * math.pi * freq_hz
    eps_c = eps_r - 1j * sigma / (omega * EPS0)
    q = np.sqrt(eps_c - rx * rx - ry * ry)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rz - q) / (rz + q), (eps_c * rz - q) / (eps_c * rz + q)


def _cliff_medium_2(mid, theta, phi, ground_z, mode, edge_distance):
    """``(n_theta, n_phi, n_element)`` mask: is this element's reflection point
    on the FAR side of the cliff?

    NEC picks the medium per SEGMENT and per DIRECTION, at that segment's own
    specular point on the ground. For an element at height ``z`` radiating
    towards ``theta``, the reflection point is ``dr = z·tan(theta)`` out along
    the azimuth, so its ground coordinates are ``(x + dr·cosφ, y + dr·sinφ)``
    — and the edge it is compared against is

    * ``RP 2``, LINEAR cliff: the line ``x = CLT``, so only the x coordinate
      counts. An azimuth parallel to the edge never crosses it and an azimuth
      pointing away from it never does either (``d`` goes negative).
    * ``RP 3``, CIRCULAR cliff: the circle ``r = CLT``, so the radius counts
      and every azimuth crosses alike.

    The comparison is ``FFLD``'s exactly, including its tie-break:
    ``IF ((CL-D).LE.0.) GO TO 15`` takes medium 2, so ``(CL - D) > 0`` keeps
    medium 1 and a point landing precisely ON the edge takes medium 2. The
    specular-point construction ``DR=Z(I)*TTHET`` / ``D=DR*PHY+X(I)`` /
    ``D=SQRT(D*D+(Y(I)-DR*PHX)**2)`` is that routine's, with ``TTHET=TAN(THET)``,
    ``PHX=-SIN(PHI)`` and ``PHY=COS(PHI)`` set at its head.

    **Validity.** This is NEC-2's own cliff model, and it is a geometric-optics
    one: a single specular bounce off whichever flat half-plane the reflection
    point happens to sit on, with no diffraction at the edge and no shadowing
    by the step. It is trustworthy where the edge is many wavelengths from
    both the antenna and the reflection point, and it is visibly discontinuous
    across the angle where the reflection point crosses — which is a property
    of the model, not of this implementation, and shows up identically in the
    oracle's own table.
    """
    height = mid[:, 2] - ground_z
    dr = np.tan(theta)[:, None] * height[None, :]  # (n_theta, n_element)
    along = dr[:, None, :] * np.cos(phi)[None, :, None] + mid[None, None, :, 0]
    if mode == 3:
        across = mid[None, None, :, 1] + dr[:, None, :] * np.sin(phi)[None, :, None]
        along = np.hypot(along, across)
    return along >= edge_distance


def _cliff_image_moments(
    mid, moment, k, theta, phi, basis, ground, ground_z, freq_hz, mode, cliff
):
    """The reflected far-field moment under ``RP 2`` / ``RP 3``.

    The flat-ground path applies one reflection coefficient to the whole image
    sum. A cliff cannot: two elements of the same antenna can reflect off
    different media in the same direction, so the sum has to be split first
    and weighted after. That is ``FFLD``'s inner loop, vectorised — split the
    image carrier by :func:`_cliff_medium_2`, give each half its own
    ``(rho_h, rho_v)``, add.

    The far side also carries an extra phase. Its surface is ``CHT`` below
    medium 1's (signed, POSITIVE = lower — the fixture that isolates the
    field is ``mininec_vertical_rp3_ch``, momwire#487), so its image sits
    ``2·CHT`` further along the vertical and the ray to it is
    ``2·CHT·cos(theta)`` longer:
    ``FFLD`` spells that ``DARG=-TP*2.*CH*ROZ`` (``TP`` = 2π, ``CH`` = ``CHT``
    in wavelengths, ``ROZ`` = cos θ) and adds it to the image phase, which is
    the same ``exp(-2jk·CHT·cosθ)`` applied here.
    """
    rhat, h_hat, v_hat = basis
    rx, ry, rz = rhat[..., 0], rhat[..., 1], rhat[..., 2]
    mid_img, moment_img = _image_moments(mid, moment, ground_z)
    carrier = np.exp(1j * k * np.einsum("ijc,nc->ijn", rhat, mid_img))

    beyond = _cliff_medium_2(mid, theta, phi, ground_z, mode, cliff.edge_distance)
    step = np.exp(-2j * k * cliff.height * np.cos(theta))
    far = np.einsum(
        "ijn,nc->ijc", carrier * np.where(beyond, step[:, None, None], 0.0), moment_img
    )
    carrier *= ~beyond
    near = np.einsum("ijn,nc->ijc", carrier, moment_img)

    # Medium 1 is whatever the GN card said. Perfect ground is the one case
    # FFLD does not run through the Fresnel formula at all, and the second
    # medium never gets that shortcut: a GD states eps/sigma and nothing else,
    # so it is always a reflection coefficient even under GN 1.
    if ground.kind == "pec":
        near_coeffs = (-1.0, 1.0)
    else:
        near_coeffs = _image_coeffs(ground.eps_r, ground.sigma, freq_hz, rx, ry, rz)
    far_coeffs = _image_coeffs(cliff.eps_r2, cliff.sigma2, freq_hz, rx, ry, rz)

    total = np.zeros_like(near)
    for half, (rho_h, rho_v) in ((near, near_coeffs), (far, far_coeffs)):
        half_h = np.sum(half * h_hat, axis=-1)
        half_v = np.sum(half * v_hat, axis=-1)
        total += (rho_v * half_v)[..., None] * v_hat
        total -= (rho_h * half_h)[..., None] * h_hat
    return total


def _far_moments(mid, moment, k, theta, phi, ground, ground_z, freq_hz, cliff=None):
    """Complex ``(M_theta, M_phi)`` on the ``theta`` x ``phi`` grids (radians).

    ``M = Σ I_n dl_n exp(+j k r̂·r_n)`` is the far-field current moment; the
    radiated field is ``E = -j η k /(4π) · e^(-jkr)/r · M_perp``. The engine's
    ``_evaluate_M_perp`` computes ``|M_perp|²`` for the gain plot and throws
    the components away; a NEC printout needs them, because it reports
    E(THETA) and E(PHI) magnitude AND phase and splits the gain into VERTC
    (theta) and HORIZ (phi). Same physics, same ground handling, components
    kept.

    ``cliff`` is ``(mode, SecondMedium)`` when the ``RP`` card asked for one of
    the cliff modes, and ``None`` otherwise. It only ever reaches the image
    term — a cliff with no ground image is not a cliff, which is why NEC still
    prints the FAR FIELD GROUND PARAMETERS block for an ``RP 2`` in free space
    and still moves no number.
    """
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    rx = sin_t[:, None] * cos_p[None, :]
    ry = sin_t[:, None] * sin_p[None, :]
    rz = np.broadcast_to(cos_t[:, None], rx.shape)
    rhat = np.stack([rx, ry, rz], axis=-1)

    cos_p_g = np.broadcast_to(cos_p[None, :], rx.shape)
    sin_p_g = np.broadcast_to(sin_p[None, :], rx.shape)
    cos_t_g = np.broadcast_to(cos_t[:, None], rx.shape)
    sin_t_g = np.broadcast_to(sin_t[:, None], rx.shape)
    # NEC's spherical basis: theta_hat points away from +z, phi_hat is
    # azimuthal. Both are already perpendicular to rhat, so projecting M onto
    # them IS projecting M_perp onto them.
    theta_hat = np.stack([cos_t_g * cos_p_g, cos_t_g * sin_p_g, -sin_t_g], axis=-1)
    phi_hat = np.stack([-sin_p_g, cos_p_g, np.zeros_like(rx)], axis=-1)

    def moments_of(centres, weights):
        phase = k * np.einsum("ijc,nc->ijn", rhat, centres)
        return np.einsum("ijn,nc->ijc", np.exp(1j * phase), weights)

    # The reflected wave's own polarisation basis: h along phi_hat, v the
    # in-plane partner. PEC is rho_h = -1, rho_v = +1, so the Fresnel step
    # below is written as a correction to the PEC image exactly as the engine
    # does.
    h_hat = phi_hat
    v_hat = np.stack([-cos_p_g * cos_t_g, -sin_p_g * cos_t_g, sin_t_g], axis=-1)

    m_direct = moments_of(mid, moment)
    if ground is None or ground.kind == "free":
        total = m_direct
    elif cliff is not None:
        mode, second = cliff
        total = m_direct + _cliff_image_moments(
            mid,
            moment,
            k,
            theta,
            phi,
            (rhat, h_hat, v_hat),
            ground,
            ground_z,
            freq_hz,
            mode,
            second,
        )
    else:
        mid_img, moment_img = _image_moments(mid, moment, ground_z)
        m_img = moments_of(mid_img, moment_img)
        if ground.kind == "pec":
            total = m_direct + m_img
        else:
            m_img_h = np.sum(m_img * h_hat, axis=-1)
            m_img_v = np.sum(m_img * v_hat, axis=-1)
            rho_h, rho_v = _image_coeffs(
                ground.eps_r, ground.sigma, freq_hz, rx, ry, rz
            )
            m_refl = (rho_v * m_img_v)[..., None] * v_hat - (rho_h * m_img_h)[
                ..., None
            ] * h_hat
            total = m_direct + m_refl
    return np.sum(total * theta_hat, axis=-1), np.sum(total * phi_hat, axis=-1)


def _element_fields(points, elements, k, radius, magnetic):
    """E (or H) at ``points`` from the solved current, in MIXED-POTENTIAL form.

    ``elements`` is ``(mid, moment, nodes, delta)``: element midpoints, their
    current moments ``p = I·dl``, the mesh NODES between them, and the current
    STEP ``ΔI = I_in - I_out`` at each node — the discrete continuity charge
    ``q = ΔI/(jω)``. With ``G = e^{-jkR}/R``,

        E = -j·ηk/(4π)·Σ p_n G_n  -  j·η/(4πk)·Σ ΔI_m (1+jkR_m)/R_m² · G_m·R̂_m
        H = 1/(4π)·Σ (p_n × R̂_n)·(jk/R_n + 1/R_n²)·G_n

    the first E term being ``-jωA`` and the second ``-∇Φ``. In the radiation
    zone the pair collapses to ``-j·ηk/(4πr)·e^(-jkr)·M_perp``, the same
    prefactor :func:`_far_moments` is normalised against, so the near-field and
    pattern tables are one physics.

    **Why not a chain of Hertzian point dipoles.** That form is algebraically
    simpler and agrees with this one everywhere off the structure — but each
    element carries its own ±q pair separated by dl, and a sample point a
    fraction of an element away sees that pair's 1/R³ term with nothing to
    cancel it: an observation point ON the wire came out at 1.7E+05 V/m
    against nec2c's 1.2E-02. Splitting current and charge puts the charge
    where it physically is (the nodes) and makes ΔI small wherever the current
    is smooth, so adjacent nodes cancel the way the continuous integral does.

    ``R`` is still the thin-wire regularised ``sqrt(|R|² + a²)``: the sample is
    taken on the conductor SURFACE rather than its axis, momwire's own
    convention (``rho_eval`` in the sinusoidal kernel). Grammar doc §11.
    """
    mid, moment, nodes, delta = elements

    def geometry(sources):
        rvec = points[:, None, :] - sources[None, :, :]
        r = np.sqrt(np.sum(rvec * rvec, axis=-1) + radius * radius)
        return rvec / r[..., None], r, np.exp(-1j * k * r) / (4.0 * math.pi * r)

    rhat, r, green = geometry(mid)
    if magnetic:
        cross = np.cross(np.broadcast_to(moment, (len(points),) + moment.shape), rhat)
        weight = (1j * k + 1.0 / r) * green
        return np.sum(weight[..., None] * cross, axis=1)

    e_vector = -1j * ETA0 * k * np.sum(green[..., None] * moment[None, :, :], axis=1)
    q_rhat, q_r, q_green = geometry(nodes)
    scalar = -1j * ETA0 / k * (delta[None, :] * (1.0 + 1j * k * q_r) / q_r * q_green)
    return e_vector + np.sum(scalar[..., None] * q_rhat, axis=1)


def _phase_deg(value: complex) -> float:
    """The one spelling of phase-in-degrees for the readout AND the portal's
    row formatters (``fmt_pattern_row``, ``fmt_near_field_row``), which stay
    in the portal because they format NEC's column layout rather than compute
    the field — they import this back through the same compat block that
    re-exports the readout."""
    return math.degrees(math.atan2(value.imag, value.real))


def _polarisation(
    e_theta: complex,
    e_phi: complex,
    floor_scale: float = 1.0,
    linear_below: float = 5.0e-5,
) -> tuple[float, float, str]:
    """``(axial_ratio, tilt_deg, sense)`` for one direction's polarisation
    ellipse — the AXIAL RATIO / TILT / SENSE columns.

    ``floor_scale`` converts the PRINTED field back to the amplitude ``FFLD``
    returned, which is the basis NEC's blank-column test is written in (see
    ``_FIELD_FLOOR2``). It is 1 for a table read out at the wavelength's own
    scale and ``RFLD/lambda`` for one read out at a range; everything else
    here is scale-free, so it reaches the floor test and nothing more.

    With ``a = |E_theta|``, ``b = |E_phi|`` and ``δ = arg(E_phi) - arg(E_theta)``
    wrapped to ±180°, the ellipse semi-axes are
    ``0.5·[a²+b² ± sqrt(a⁴+b⁴+2a²b²cos2δ)]`` and the tilt is
    ``½·atan2(2ab·cosδ, a²-b²)`` — the same pair ``RDPAT`` forms as
    ``TILTA=.5*ATGN2(TSTOR2,TSTOR1)``.
    The axial ratio is minor/major, signed by ``sinδ``, so linear
    polarisation prints 0.0000 and circular ±1.0000.

    Sense is calibrated against the oracle, not derived: a crossed pair fed
    ``EX ... 1. 0.`` / ``EX ... 0. 1.`` gives ``δ = +90°`` at the zenith and
    nec2c prints ``AXIAL RATIO 1.0000 ... LEFT``
    (``dipole_rp_crossed_quadrature``), so positive ``sinδ`` is LEFT here.

    ``linear_below`` is the bar under which the ratio PRINTS as zero, and it
    is the caller's because the two doors print it to different widths — the
    portal's ``{axial:12.4f}`` rounds to zero below 5e-5, the EZNEC seam's
    ``{axial:11.5f}`` below 5e-6. Under it the row reads ``0.0000`` and the
    sense is forced to ``LINEAR``, so the number column and the label cannot
    contradict each other.

    That is a rule, not a tolerance (momwire#578). The bar used to be a flat
    ``1e-8``, which is BELOW where the ratio of a linear row actually lands:
    ``sin δ`` there is the difference of two nearly-equal phases, so it moves
    with thread count and process history, and 46 rows across the EZNEC
    corpus flipped between ``LINEAR``, ``RIGHT`` and ``LEFT`` run to run —
    every one of them printing an axial ratio of ``0.00000``. A user-visible
    polarisation LABEL is not dust to be squinted past. Both oracles agree
    with the rule and neither is being contradicted by it: across 662 nec2c
    fixture rows and 13169 EZNEC capture rows, no row prints a zero axial
    ratio beside a sense other than blank or ``LINEAR``.
    """
    a, b = abs(e_theta), abs(e_phi)
    raw2 = floor_scale * floor_scale
    if a * a * raw2 <= _FIELD_FLOOR2 and b * b * raw2 <= _FIELD_FLOOR2:
        return 0.0, 0.0, ""
    delta = _phase_deg(e_phi) - _phase_deg(e_theta)
    delta = (delta + 180.0) % 360.0 - 180.0
    a2, b2 = a * a, b * b
    root = math.sqrt(
        max(
            a2 * a2 + b2 * b2 + 2.0 * a2 * b2 * math.cos(math.radians(2.0 * delta)), 0.0
        )
    )
    major2 = 0.5 * (a2 + b2 + root)
    minor2 = max(0.5 * (a2 + b2 - root), 0.0)
    tilt = 0.5 * math.degrees(
        math.atan2(2.0 * a * b * math.cos(math.radians(delta)), a2 - b2)
    )
    if major2 <= 0.0:
        return 0.0, tilt, "LINEAR"
    ratio = math.sqrt(minor2 / major2)
    sin_d = math.sin(math.radians(delta))
    if ratio < linear_below:
        return 0.0, tilt, "LINEAR"
    return (
        ratio if sin_d >= 0 else -ratio,
        tilt,
        "LEFT" if sin_d >= 0 else "RIGHT",
    )


def _gain_db(power_gain: float) -> float:
    """A gain column in dB, with nec2c's degenerate floor.

    This is NEC's ``DB10``: ``X < 1.D-20 -> -999.99``, applied to the LINEAR POWER
    GAIN about to be logged rather than to the field. ``dipole_rp_pattern``
    prints the floor for a direction whose E(THETA) is 5.4196E-15 because that
    direction's gain is around -220 dB, not because the field is small — a
    distinction the fixtures could not show while every pattern was taken at
    one range, and one that ``dipole_rp_gain_only`` now pins: the gain columns
    of an ``RFLD = 0`` table are identical to the same deck's at 1000 m, while
    every E column moves by three decades (issue #802).
    """
    if power_gain < _GAIN_FLOOR2:
        return _GAIN_FLOOR_DB
    return 10.0 * math.log10(power_gain)
