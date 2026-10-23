"""The near field of a deck with wires on BOTH sides of a finite ground, and
of any deck at a point IN the ground (momwire#1653 unit 1: #524 phase 3,
#570's NE/NH half, #1257).

Two spellings, one readout. `_near_buried_core.cpp` is the C++ one, which
the standalone executables link and which `_near_readout` calls through
`_accelerators.near_buried_parts` whenever it is built (`cxx_parts`), so a
launcher and an executable print the same bytes. This module's numpy
functions are the REFERENCE it is gated against
(tests/test_near_buried_1653.py, per regime) and the route without an
accelerator; the docstrings below describe both.

The readout contract
--------------------
Every point field in this tree is a sum over SOURCE elements, and which
terms a (source, observer) pair contributes depends on which side of the
interface each sits on. The four regimes, and the CONTRACT each one keeps:

    source  observer  terms                                       returns
    above   above     direct(k0) + C2*image(k0) + reflected rem.  REMAINDER
    below   below     direct(km) + Am*image(km) + below rem.      REMAINDER
    below   above     transmitted                                 WHOLE
    above   below     transmitted (reciprocity transpose)         WHOLE

C2 = (e-1)/(e+1), Am = (1-e)/(1+e), e the complex relative permittivity
e_r - j sigma/(w e0), km = k0 sqrt(e).

A REMAINDER regime hands back only the smooth Sommerfeld remainder and its
CALLER adds direct and coefficient*image in the regime's own medium
(`_field_point.reflected_field_at` for the first row,
`_sommerfeld_below.remainder_field_below` for the second). A WHOLE regime
hands back the entire field across the interface and the caller adds
NOTHING: there is no direct ray from a buried source to an observer in air,
and no image either. Feeding a whole field through a remainder entry point
would add a direct and an image term the physics does not have, which is
the double count momwire#570 named. So the two kinds are two entry points,
not one signature with a flag: `_field_point.reflected_field_at` still
REFUSES the crossing regimes, and the whole field lives here, in
`transmitted_field_at`, whose signature takes the knot charges the remainder
entry points never see.

The transmitted field
---------------------
The crossing fill's own designed kernels (`_near_interface`, the six
{U, V, W, dzW, dzpV, dzpW} and the point family {gRhoV, gzV, gRhoW, gzpV})
evaluated by direct contour integration at each (rho_eff, z, z'), z >= 0 the
height of the ABOVE point and z' <= 0 the height of the BELOW one, whichever
of the two is the source. Read off momwire#956's dyad for a current moment
p = I dl and the knot charge steps dI = I_in - I_out, with
D = (observer - source) horizontal and rho_eff = hypot(rho, a):

    E = c1 sum_e [ U p_h + z^ (k0^2 V p_z + (p_h . D/rho_eff) gRhoW - p_z dzW)
                   - p_z (D/rho_eff) gRhoW ]
      - c1 sum_n dI_n [ (D/rho_eff) gRhoV + z^ dV ]

dV = gzV for an observer above and gzpV for one below. It is ONE formula for
both directions (`_crossing_fill.point_observer_block` writes the two
tested forms; transposing the forward dyad onto the reversed observer gives
the same vector), and the observer gradient of the V term is what carries
the charge, so the knot charges enter exactly where `_element_fields` puts
them in free space. No grid and no ladder: a point anywhere on either side
is served, the corner z = z' = 0 included, and the far columns ride the
steepest-descent path (momwire#1595 D1b).

H is the curl of E over -j w mu0. The charge term and -p_z grad W are
gradients and drop out, so only the remaining part is differenced, by a
central stencil of `TRANSMITTED_CURL_STEP_WAVELENGTHS`, one-sided in z where
the stencil would cross the interface. The kernels are direct contour
evaluations at rtol 1e-10 with no interpolation noise to amplify, which is
why this step can be ten times finer than the reflected remainder's.

The below/below regime
----------------------
direct(km) + Am*image(km) through the mixed-potential element sum written
in the medium (jw mu0 on A, 1/(jw e0 e) on the scalar potential: the buried
fill's own split, `_below_interface.compute_Z_operator_buried`), plus the
vector remainder `_sommerfeld_below.remainder_field_below` over a point
spacing grid. Its H is the analytic in-medium element H plus the curl of the
remainder by central differences, one-sided in z near the interface.

Which side a source is on
-------------------------
An element by its midpoint; a knot by the wire it belongs to. A crossing
node sits exactly at z = 0 and carries one knot per wire meeting there
(`element_currents` does not merge them), so each side's charge goes through
its own side's regime, the split the crossing fill itself makes. An observer
at z >= ground_z is above.
"""

from __future__ import annotations

import math

import numpy as np

from . import _field_point, _near_interface, _sommerfeld_below
from ._far_readout import ETA0, _image_moments
from ._sommerfeld_below import _c1_moment, image_coefficient_below, k_medium

_MU0 = _field_point._MU0

# The transmitted curl's central-difference step, in free-space wavelengths.
# FD truncation is (k step)^2/6 of scale on a field of scale 1/k and
# (step/d)^2/6 on one varying over a distance d to its source: 1e-4 wl is
# 6.6e-8 and, at 7 MHz (step 4.3 mm), 1.4e-4 at d = 0.15 m. The kernels'
# own error is ~1e-10 relative, which a difference amplifies by 1/(k step)
# ~ 1.6e3: 1.6e-7. Measured, not assumed: see the curl gate in
# tests/test_near_buried_1653.py.
TRANSMITTED_CURL_STEP_WAVELENGTHS = 1e-4

# The below remainder's: the reflected remainder's own rule
# (`_field_point.CURL_STEP_WAVELENGTHS`), in the IN-MEDIUM wavelength,
# because that is the scale the below surfaces are tabulated on.
BELOW_CURL_STEP_WAVELENGTHS = _field_point.CURL_STEP_WAVELENGTHS

# A point in the soil past the below grid's cap reads a mesh segment as one
# element when the segment's phase extent (k0 horizontally, |k_m|
# vertically) is under this: (0.1)^2/24 = 4e-4 of the segment's share.
BELOW_FAR_SEGMENT_PHASE = 0.1

# The transmitted kernels' contour tolerance: the crossing fill's own.
_RTOL = 1e-10

# The C++ readout (`_near_buried_core.cpp`, bound as
# `_accelerators.near_buried_parts`): what `_near_readout._near_field_buried`
# answers through when it is built, so the Python launchers and the
# standalone executables read the same code. False takes this module's numpy
# spelling, the reference the C++ route is gated against
# (tests/test_near_buried_1653.py).
_CXX = True


def cxx_parts():
    """The C++ readout as a callable, or None (no accelerator, or `_CXX`
    off): `f(points, elements, per_wire, subdiv, eps_t, ground_z, k, omega,
    radius, magnetic)` -> (n, 3), the transmitted field of the buried
    sources at a point at or above the plane, below_field_at + the
    transmitted field of the above sources at a point below it."""
    from ._accel import acc

    if not _CXX or acc is None or not getattr(acc, "near_buried_1653", False):
        return None

    def run(
        points, elements, per_wire, subdiv, eps_t, ground_z, k, omega, radius, magnetic
    ):
        mid, moment, nodes, delta = elements
        return acc.near_buried_parts(
            np.ascontiguousarray(points, dtype=float),
            np.ascontiguousarray(mid, dtype=float),
            np.ascontiguousarray(moment, dtype=np.complex128),
            np.ascontiguousarray(nodes, dtype=float),
            np.ascontiguousarray(delta, dtype=np.complex128),
            np.asarray(per_wire, dtype=np.int64),
            int(subdiv),
            complex(eps_t),
            float(ground_z),
            float(k),
            float(omega),
            float(_MU0),
            float(radius),
            bool(magnetic),
            _near_interface._physical_cpu_count(),
        )

    return run


# --------------------------------------------------------------------------
# sides
# --------------------------------------------------------------------------


def element_counts(solver):
    """Per-wire, per-edge mesh segment counts of `solver.element_currents`'
    walk, in its wire order: the caller's wires, through a dedup map when
    the solver dropped a copy (momwire#1042). A wire's element count at
    resampling `subdiv` is its sum times `subdiv`."""
    per_edge = solver.n_per_edge_per_wire
    dedup = getattr(solver, "_wire_dedup", None)
    if dedup is not None:
        per_edge = dedup.to_input(
            list(per_edge), lambda r, v, rev: list(v)[::-1] if rev else v
        )
    return [[int(n) for n in npe] for npe in per_edge]


def source_sides(elements, counts, ground_z):
    """`(mid_below, node_below)` boolean masks: an element by its midpoint,
    a knot by its wire's elements (module docstring). A wire is wholly on
    one side here (a card crossing the plane is split there before it is
    solved, momwire#1281); a wire whose elements straddle is a ValueError,
    because its knot in the plane would have no side."""
    mid, _moment, nodes, _delta = elements
    mid_below = mid[:, 2] < ground_z
    node_below = np.zeros(nodes.shape[0], dtype=bool)
    if sum(counts) != mid.shape[0] or sum(counts) + len(counts) != nodes.shape[0]:
        raise ValueError(
            "element counts do not match the element walk: "
            f"{sum(counts)} elements / {len(counts)} wires against "
            f"{mid.shape[0]} mids and {nodes.shape[0]} knots"
        )
    e0 = 0
    for w, m in enumerate(counts):
        side = mid_below[e0 : e0 + m]
        if m and side.any() != side.all():
            raise ValueError(
                f"wire {w} has elements on both sides of the interface; a "
                "crossing card is split in the plane before it is solved"
            )
        node_below[e0 + w : e0 + w + m + 1] = bool(m and side[0])
        e0 += m
    return mid_below, node_below


# --------------------------------------------------------------------------
# one side's sources
# --------------------------------------------------------------------------


class Sources:
    """One side's source terms: element moments at their midpoints and knot
    steps at their knots, the `(mid, moment, nodes, delta)` of the wires
    `keep_wire` names. Every observer reads every one of them; what makes
    that affordable is the sheet plan (`_plan`), not a coarser reading - a
    far cluster read as one element is off by (k L)^2/24 of its share,
    and at the in-medium wavenumber that is a percent for one 0.5 m
    segment (measured 1.3 % at 50 m on a 5 m buried dipole)."""

    def __init__(self, elements, counts, keep_wire, subdiv=1):
        mid, moment, nodes, delta = elements
        f_idx, a_idx = [], []
        e0 = 0
        for w, m in enumerate(counts):
            if keep_wire[w]:
                f_idx.append(np.arange(e0, e0 + m))
                a_idx.append(np.arange(e0 + w, e0 + w + m + 1))
            e0 += m
        f = np.concatenate(f_idx) if f_idx else np.zeros(0, dtype=int)
        a = np.concatenate(a_idx) if a_idx else np.zeros(0, dtype=int)
        self.mid = np.asarray(mid, dtype=float)[f]
        self.moment = np.asarray(moment, dtype=complex)[f]
        self.nodes = np.asarray(nodes, dtype=float)[a]
        self.delta = np.asarray(delta, dtype=complex)[a]
        # Mesh segments (`subdiv` consecutive elements each): their summed
        # moments at their midpoints, for `_below_remainder`'s far route.
        sub = max(int(subdiv or 1), 1)
        g = self.mid.shape[0] // sub
        self.seg_moment = self.moment[: g * sub].reshape(g, sub, 3).sum(axis=1)
        self.seg_mid = self.mid[: g * sub].reshape(g, sub, 3).mean(axis=1)
        span = self.mid[: g * sub].reshape(g, sub, 3)
        ext = np.abs(span[:, -1] - span[:, 0]) * (sub / max(sub - 1, 1))
        self.seg_h = np.hypot(ext[:, 0], ext[:, 1])
        self.seg_v = ext[:, 2]
        self.whole = g * sub == self.mid.shape[0]

    @property
    def empty(self):
        return self.mid.shape[0] == 0

    def moment_pairs(self, points):
        """`(obs, pos, moment)`, every (point, element) pair; `obs` indexes
        `points`."""
        n, m = points.shape[0], self.mid.shape[0]
        obs = np.repeat(np.arange(n), m)
        return obs, np.tile(self.mid, (n, 1)), np.tile(self.moment, (n, 1))

    def charge_pairs(self, points):
        """`(obs, pos, dI)`, every (point, knot) pair, dI = I_in - I_out."""
        n, m = points.shape[0], self.nodes.shape[0]
        obs = np.repeat(np.arange(n), m)
        return obs, np.tile(self.nodes, (n, 1)), np.tile(self.delta, n)


# --------------------------------------------------------------------------
# the transmitted (WHOLE) field
# --------------------------------------------------------------------------


def _pair_geometry(obs, src, ground_z, observers_above, a):
    """Flat pairs: (D_x/rho_eff, D_y/rho_eff, rho, z, z')."""
    dx = obs[:, 0] - src[:, 0]
    dy = obs[:, 1] - src[:, 1]
    rho = np.hypot(dx, dy)
    z_o = obs[:, 2] - ground_z
    z_s = src[:, 2] - ground_z
    z, zp = (z_o, z_s) if observers_above else (z_s, z_o)
    rho_eff = _near_interface.radius_fold(rho, a)
    return dx / rho_eff, dy / rho_eff, rho, np.maximum(z, 0.0), np.minimum(zp, 0.0)


def _scatter(n, obs, values):
    out = np.zeros((n, 3), dtype=complex)
    for i in range(3):
        out[:, i] = np.bincount(obs, weights=values[i].real, minlength=n) + 1j * (
            np.bincount(obs, weights=values[i].imag, minlength=n)
        )
    return out


def _transmitted_moment_terms(points, sources, eps_t, ground_z, k, a, above, memo):
    """sum over the source moments a point reads of the moment part of E/c1
    (module docstring), split into its curl-carrying part and the
    -p_z grad W gradient: two (N, 3)."""
    n = points.shape[0]
    obs, pos, mom = sources.moment_pairs(points)
    if not obs.size:
        z = np.zeros((n, 3), dtype=complex)
        return z, z.copy()
    gx, gy, rho, z, zp = _pair_geometry(points[obs], pos, ground_z, above, a)
    six = _near_interface.radius_tables(eps_t, k, rho, z, zp, a, rtol=_RTOL, memo=memo)
    pk = _near_interface.point_radius_tables(
        eps_t, k, rho, z, zp, a, plan=memo.sheet_plan
    )
    U, V, gRhoW = six["U"], six["V"], pk["gRhoW"]
    px, py, pz = mom[:, 0], mom[:, 1], mom[:, 2]
    z_ng = k * k * V * pz + (px * gx + py * gy) * gRhoW
    if not above:
        # -p_z (D gRhoW + z dzW) is -p_z grad_o W + p_z z (dzpW - dzW) when
        # the observer is the below point (its z-derivative is d/dz').
        z_ng = z_ng + pz * (six["dzpW"] - six["dzW"])
    curl_part = _scatter(n, obs, (U * px, U * py, z_ng))
    dz_obs = six["dzW"] if above else six["dzpW"]
    grad_part = _scatter(n, obs, (-pz * gx * gRhoW, -pz * gy * gRhoW, -pz * dz_obs))
    return curl_part, grad_part


def _transmitted_charge_term(points, sources, eps_t, ground_z, k, a, above, memo):
    """-sum over the knots a point reads of dI grad_o V: (N, 3), over c1."""
    n = points.shape[0]
    obs, pos, d_i = sources.charge_pairs(points)
    if not obs.size:
        return np.zeros((n, 3), dtype=complex)
    gx, gy, rho, z, zp = _pair_geometry(points[obs], pos, ground_z, above, a)
    pk = _near_interface.point_radius_tables(
        eps_t, k, rho, z, zp, a, plan=memo.sheet_plan
    )
    d_v = pk["gzV"] if above else pk["gzpV"]
    q = -d_i
    return _scatter(n, obs, (q * gx * pk["gRhoV"], q * gy * pk["gRhoV"], q * d_v))


def _plan(points, sources, eps_t, ground_z, k, a, above, step, magnetic):
    """One readout's memo, carrying the crossing fill's own `SheetPlan`
    (momwire#1224 option A) over every (observer, source) row it may ask:
    a NE grid is a plane of observers over a screen's plane of sources, so
    its rows sit on a sheet and are read off it (gated at ~1e-11 there,
    `MOMWIRE_NEAR_INTERFACE_SHEET=0` the exact route)."""
    obs = points
    if magnetic:
        obs = np.concatenate(
            [points]
            + [
                points + m * step * np.eye(3)[i]
                for i in range(3)
                for m in (-2, -1, 1, 2)
            ]
        )
        obs = obs[(obs[:, 2] >= ground_z) if above else (obs[:, 2] <= ground_z)]
    src = np.concatenate([sources.mid, sources.nodes])
    o = obs.copy()
    sp = src.copy()
    o[:, 2] -= ground_z
    sp[:, 2] -= ground_z
    memo = _near_interface.TripleMemo()
    memo.sheet_plan = _near_interface.sheet_plan(
        eps_t, k, [(o, sp) if above else (sp, o)], a
    )
    return memo


def _check_sides(points, sources, ground_z, observers_above, what):
    if not points.shape[0] or not sources.shape[0]:
        return
    if observers_above:
        ok = points[:, 2].min() >= ground_z and sources[:, 2].max() <= ground_z
    else:
        ok = points[:, 2].max() <= ground_z and sources[:, 2].min() >= ground_z
    if not ok:
        raise ValueError(
            f"the transmitted field needs the observers and the {what} on "
            "opposite sides of the interface"
        )


def transmitted_field_at(
    points,
    sources,
    eps_t,
    ground_z,
    k,
    omega,
    radius,
    *,
    observers_above: bool,
    magnetic: bool = False,
    mu: float = _MU0,
    step: float | None = None,
):
    """The WHOLE field across the interface at `points` (all on one side)
    from `sources` (a `Sources`, all on the other): (N, 3) complex. Nothing
    is to be added to it (module docstring)."""
    points = np.asarray(points, dtype=float)
    out = np.zeros((points.shape[0], 3), dtype=complex)
    if not points.shape[0] or sources.empty:
        return out
    _check_sides(points, sources.mid, ground_z, observers_above, "source elements")
    _check_sides(points, sources.nodes, ground_z, observers_above, "source knots")
    gz = float(ground_z)
    c1 = _c1_moment(omega, mu)
    if step is None:
        step = TRANSMITTED_CURL_STEP_WAVELENGTHS * (2.0 * math.pi / k)
    memo = _plan(points, sources, eps_t, gz, k, radius, observers_above, step, magnetic)
    if not magnetic:
        curl_part, grad_part = _transmitted_moment_terms(
            points, sources, eps_t, gz, k, radius, observers_above, memo
        )
        charge = _transmitted_charge_term(
            points, sources, eps_t, gz, k, radius, observers_above, memo
        )
        return c1 * (curl_part + grad_part + charge)

    def e_ng(pts, _centre):
        return (
            c1
            * _transmitted_moment_terms(
                pts, sources, eps_t, gz, k, radius, observers_above, memo
            )[0]
        )

    return _curl(e_ng, points, step, gz, observers_above) / (-1j * omega * mu)


def _curl(field, points, step, ground_z, above):
    """curl of `field` at `points` by central differences, the z one taken
    one-sided (second order, on the point's own side) wherever the central
    stencil would reach across `ground_z`. `field(pts, centre)` is told each
    offset's stencil centre."""
    n = points.shape[0]
    height = points[:, 2] - ground_z
    one_sided = (height < step) if above else (-height < step)
    away = 1.0 if above else -1.0
    stacks, centres = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            shift = np.zeros(3)
            shift[axis] = sign * step
            # a one-sided point's central z offsets are not read across the
            # plane: they stand at the point itself and go unused
            off = points + shift
            if axis == 2:
                off = np.where(one_sided[:, None], points, off)
            stacks.append(off)
            centres.append(points)
    # The one-sided z stencil, f'(0) = (-3 f0 + 4 f1 - f2) / (2 step), away
    # from the interface: three more reads, at the one-sided points only.
    side = points[one_sided]
    for m in (0.0, 1.0, 2.0):
        stacks.append(side + np.array([0.0, 0.0, m * away * step]))
        centres.append(side)
    values = field(np.concatenate(stacks), np.concatenate(centres))
    v = [values[i * n : (i + 1) * n] for i in range(6)]
    d = [(v[2 * a] - v[2 * a + 1]) / (2.0 * step) for a in range(3)]
    m1 = side.shape[0]
    if m1:
        w = [values[6 * n + i * m1 : 6 * n + (i + 1) * m1] for i in range(3)]
        d[2][one_sided] = away * (-3.0 * w[0] + 4.0 * w[1] - w[2]) / (2.0 * step)
    return np.stack(
        [
            d[1][:, 2] - d[2][:, 1],
            d[2][:, 0] - d[0][:, 2],
            d[0][:, 1] - d[1][:, 0],
        ],
        axis=-1,
    )


# --------------------------------------------------------------------------
# below/below: direct + Am*image in the medium, plus the remainder
# --------------------------------------------------------------------------


def medium_element_fields(points, elements, k, eps_t, radius, magnetic):
    """`_far_readout._element_fields` written in the lower medium: the
    vector potential at k_m with mu0, the scalar potential at k_m over e."""
    mid, moment, nodes, delta = elements
    k_m = k_medium(complex(eps_t), float(k))

    def geometry(sources):
        rvec = points[:, None, :] - sources[None, :, :]
        r = np.sqrt(np.sum(rvec * rvec, axis=-1) + radius * radius)
        return rvec / r[..., None], r, np.exp(-1j * k_m * r) / (4.0 * math.pi * r)

    rhat, r, green = geometry(mid)
    if magnetic:
        cross = np.cross(np.broadcast_to(moment, (len(points),) + moment.shape), rhat)
        weight = (1j * k_m + 1.0 / r) * green
        return np.sum(weight[..., None] * cross, axis=1)
    e_vector = -1j * ETA0 * k * np.sum(green[..., None] * moment[None, :, :], axis=1)
    q_rhat, q_r, q_green = geometry(nodes)
    scalar = (
        -1j
        * ETA0
        / (k * eps_t)
        * (delta[None, :] * (1.0 + 1j * k_m * q_r) / q_r * q_green)
    )
    return e_vector + np.sum(scalar[..., None] * q_rhat, axis=1)


def _depth_sum_r1(points, mid, ground_z):
    d_o = ground_z - points[:, 2]
    d_s = ground_z - mid[:, 2]
    rho = np.hypot(
        points[:, None, 0] - mid[None, :, 0], points[:, None, 1] - mid[None, :, 1]
    )
    return np.sqrt(rho * rho + (d_o[:, None] + d_s[None, :]) ** 2)


def _below_grid(points, mid, eps_t, ground_z, k, omega, mu):
    """The point-spacing below grid sized for `points`, or None when every
    point is past its tabulation cap (`_sommerfeld_below.below_r1_cap`)."""
    r1 = _depth_sum_r1(points, mid, ground_z)
    cap = _sommerfeld_below.below_r1_cap(k_medium(complex(eps_t), float(k)))
    inside = r1[r1 <= cap]
    if not inside.size:
        return None
    return _sommerfeld_below.get_grid_below(
        eps_t,
        k,
        1.001 * float(np.max(inside)),
        omega,
        mu=mu,
        spacing=_sommerfeld_below._SOMM_BELOW_SPACING_POINT,
    )


class _DirectBelow:
    """The four below surfaces by direct contour evaluation, read through
    the grid's `eval` signature: `remainder_field_below`'s source of
    surfaces for an observer past the grid's cap. The cap is a statement
    about Z (`_SOMM_BELOW_R1_CAP_LAMBDA_M`); a point there is a question
    the grid was never sized for, so it is answered exactly instead."""

    regime = "below"

    def __init__(self, eps_t, k, omega, mu, r1_max):
        self.eps_t, self.k, self.omega, self.mu = eps_t, k, omega, mu
        self.r1_max = float(r1_max)

    def eval(self, R1, theta):
        return _sommerfeld_below.iv_surfaces_direct_below(
            self.eps_t, self.k, R1, theta, omega=self.omega, mu=self.mu
        )


def _below_remainder(points, sources, eps_t, ground_z, k, omega, mu, grid, centre):
    """`remainder_field_below` at `points`, through `grid` for every point
    whose sources all lie inside its tabulation and by direct evaluation
    (`_DirectBelow`) for the rest. Decided at `centre` (a curl stencil's
    middle) with the stencil's reach as margin, so a curl never differences
    the grid against the contour."""
    k_m = k_medium(complex(eps_t), float(k))
    mid, moment = sources.mid, sources.moment
    reach = float(np.max(np.linalg.norm(points - centre, axis=1)))
    r1 = _depth_sum_r1(centre, mid, ground_z)
    far = (
        np.ones(points.shape[0], dtype=bool)
        if grid is None
        else np.max(r1, axis=1) + 2.0 * reach > grid.r1_max
    )
    out = np.zeros((points.shape[0], 3), dtype=complex)
    if np.any(~far):
        out[~far] = _sommerfeld_below.remainder_field_below(
            points[~far], mid, moment, ground_z, k, k_m, grid
        )
    if np.any(far):
        # Each pair here is a contour evaluation of its own. Past the grid's
        # cap the remainder is the lateral wave, whose phase across a source
        # runs at k0 horizontally and k_m vertically, so a mesh segment
        # small on BOTH scales is read as one element (its summed moment at
        # its midpoint): the dropped phase term is (phase extent)^2/24.
        m_far, p_far = mid, moment
        if sources.whole and sources.seg_mid.shape[0]:
            phase = float(np.max(k * sources.seg_h + abs(k_m) * sources.seg_v))
            if phase <= BELOW_FAR_SEGMENT_PHASE:
                m_far, p_far = sources.seg_mid, sources.seg_moment
        r1_max = 1.001 * float(np.max(_depth_sum_r1(points[far], m_far, ground_z)))
        direct = _DirectBelow(eps_t, k, omega, mu, r1_max)
        out[far] = _sommerfeld_below.remainder_field_below(
            points[far], m_far, p_far, ground_z, k, k_m, direct
        )
    return out


def below_field_at(
    points,
    elements,
    sources,
    eps_t,
    ground_z,
    k,
    omega,
    radius,
    *,
    magnetic: bool = False,
    mu: float = _MU0,
    step: float | None = None,
):
    """E (or H) at `points` below the interface from below-interface
    `elements` (the same set as `sources`, a `Sources`, which the remainder
    reads): direct + Am*image in the medium, plus the remainder."""
    points = np.asarray(points, dtype=float)
    mid, moment, nodes, delta = elements
    out = np.zeros((points.shape[0], 3), dtype=complex)
    if not points.shape[0] or not mid.shape[0]:
        return out
    gz = float(ground_z)
    k_m = k_medium(complex(eps_t), float(k))
    a_m = image_coefficient_below(eps_t)
    direct = medium_element_fields(points, elements, k, eps_t, radius, magnetic)
    mid_img, moment_img = _image_moments(mid, moment, gz)
    nodes_img = nodes.copy()
    nodes_img[:, 2] = 2.0 * gz - nodes[:, 2]
    image = medium_element_fields(
        points, (mid_img, moment_img, nodes_img, -delta), k, eps_t, radius, magnetic
    )
    if step is None:
        step = BELOW_CURL_STEP_WAVELENGTHS * (2.0 * math.pi / abs(k_m))
    # Sized over every point the curl may read, so all six offsets read ONE
    # interpolant (`_field_point.reflected_h_field_at`'s rule).
    sized = points
    if magnetic:
        sized = np.concatenate(
            [points]
            + [
                points + sign * step * np.eye(3)[axis]
                for axis in range(3)
                for sign in (1, -1)
            ]
            + [points - 2.0 * step * np.eye(3)[2]]
        )
    grid = _below_grid(sized, mid, eps_t, gz, k, omega, mu)

    def remainder(pts, centre):
        # A point a hair under the plane is as deep as the curl lets it be.
        pts = pts.copy()
        pts[:, 2] = np.minimum(pts[:, 2], gz - 1e-12 * max(1.0, abs(gz)) - 1e-12)
        return _below_remainder(pts, sources, eps_t, gz, k, omega, mu, grid, centre)

    if magnetic:
        rem = _curl(remainder, points, step, gz, False) / (-1j * omega * mu)
    else:
        rem = remainder(points, points)
    return direct + (a_m * image + rem)
