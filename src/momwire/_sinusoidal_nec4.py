"""The point-matched sinusoidal basis with PER-END charge factors: the NEC-4
dialect's radius-step junction (momwire#1644).

`SinusoidalSolver._basis_coefs` is NEC-2's basis, NEC-2 theory manual
Eqs 43-64, in which every segment carries ONE charge constant
`a_const = 1/(ln(2/(k a)) - gamma)` at both of its ends. The NEC-4 theory
manual (UCRL-MA-109338 Pt. II, Section 3.2, (3-5) to (3-37)) writes the same
basis with a constant per segment END, a_i^- and a_i^+, because NEC-4 takes
them from a junction solve (Section 3.4, `_nec4_junction`) rather than from
the segment's radius. Only the junction solve makes the two ends differ; at
every node it does not solve, both ends keep the log-rule constant.

The algebra is NEC-2's with the substitutions the manual's per-end form
makes, in momwire's NEC-2 shape set (A + B sin k xi + C cos k xi, A = -1):

    D   = (P- P+ + a- a+) sin kD + (P- a+ - P+ a-) cos kD
    Q-  = ( a+ (1 - cos kD) - P+ sin kD) / D
    Q+  = (-a- (1 - cos kD) - P- sin kD) / D
    B   = (a- Q- + a+ Q+) sin(kD/2) / sin kD
    C   = (a- Q- - a+ Q+) cos(kD/2) / sin kD

with P-/P+ summing each neighbour's factor AT THE SHARED NODE times
tan(k D_j / 2). An end branch reads the one factor of its connected end. A
neighbour entry reads the neighbour's factor at the shared node, which is its
end 2 (`a_plus`) when the neighbour's natural arc runs into the node (sigma =
+1 on an N- edge, -1 on an N+ edge) and its end 1 (`a_minus`) otherwise.

`A + C` keeps `_basis_coefs`' cancellation-free spelling (momwire#606 /
#799), generalised the same way: with S = sin(kD/2), C2 = cos(kD/2),
s4 = sin(kD/4) and dP = P- a+ - P+ a-,

    A + C = [4 a- a+ S s4^2 - 2 dP s4^2 - 2 S C2 P- P+ + 2 dP S^2] / D,

which is the NEC-2 identity when a- = a+ (verified symbolically, the
numerator difference vanishes identically).

Scope: free space and grounds that keep one real wavenumber. A deck that
solves at a per-segment wavenumber (a crossing, momwire#1223) or a complex
one (buried) never reaches here: `SinusoidalSolver._nec4_end_factors` returns
None for it and the NEC-2 basis answers, as it does for every deck with no
solved node -- which is what keeps the NEC-4 option bit-identical to the
NEC-2 basis on uniform-radius two-wire structures.
"""

from __future__ import annotations

import math

import numpy as np

from . import _nec4_junction

__all__ = ["end_factors", "basis_coefs_per_end", "solved_nodes"]


def solved_nodes(junctions, radius_per_wire, grounded):
    """The junction indices the NEC-4 rule solves: every ungrounded junction
    of three or more members, and every two-member junction whose members'
    radii differ. A two-member junction at one radius keeps the uniform
    (log-rule) factors, as NEC-4.2 is measured to."""
    out = []
    for j, members in enumerate(junctions):
        if j in grounded or len(members) < 2:
            continue
        radii = {float(radius_per_wire[w]) for w, _end in members}
        if len(members) >= 3 or len(radii) > 1:
            out.append(j)
    return out


def end_factors(geom, k, a_const, junctions, radius_per_wire, nodes):
    """`(a_minus, a_plus)` per segment, real: `a_const` at every end, the
    junction solve's factors at the members of each node in `nodes`."""
    n = geom["n_segs"]
    base = np.broadcast_to(np.asarray(a_const, dtype=np.float64), (n,))
    a_minus = base.copy()
    a_plus = base.copy()
    seg_t = geom["seg_tangents"]
    seg_h = geom["seg_h"]
    first = geom["wire_first"]
    last = geom["wire_last"]
    for j in nodes:
        segs = []
        dirs = []
        for w, end in junctions[j]:
            if end == "start":
                s = first[w]
                t = seg_t[s]
                dirs.append((float(t[0]), float(t[1]), float(t[2])))
            else:
                s = last[w]
                t = seg_t[s]
                dirs.append((-float(t[0]), -float(t[1]), -float(t[2])))
            segs.append((s, end))
        radii = [float(radius_per_wire[w]) for w, _end in junctions[j]]
        lengths = [float(seg_h[s]) for s, _end in segs]
        try:
            factors = _nec4_junction.junction_charge_factors(
                float(k), dirs, radii, lengths, float(base[segs[0][0]])
            )
        except ZeroDivisionError:
            # A singular node: members that coincide (two copies of one wire
            # left in place, say) have no unique charge split. It keeps the
            # log rule, as an unsolved node does.
            continue
        if not all(math.isfinite(f) for f in factors):
            continue
        for (s, end), f in zip(segs, factors, strict=True):
            if end == "start":
                a_minus[s] = f
            else:
                a_plus[s] = f
    return a_minus, a_plus


def basis_coefs_per_end(geom, k, a_minus, a_plus, x_end):
    """`_basis_coefs`' CSR-by-segment view for per-end factors (real `k`).

    `x_end` is each segment's end-cap factor X at its free end(s), NEC-4's
    (3-12) / (3-13): I(end) = +-(X/k) dI/ds there. NEC-2 is X = 0; NEC-4's
    flat cap gives I'/I = 2/a, (4-36), i.e. X = k a / 2 (`end_cap_x`). The
    end branches below are the manual's (3-30) to (3-38) in this lane's
    shape set, and reduce to `_basis_coefs`' X = 0 forms."""
    seg_h = geom["seg_h"]
    n_segs = geom["n_segs"]
    nm_basis = geom["nm_basis"]
    nm_seg = geom["nm_seg"]
    nm_sigma = geom["nm_sigma"]
    np_basis = geom["np_basis"]
    np_seg = geom["np_seg"]
    np_sigma = geom["np_sigma"]
    nm_count = geom["nm_count"]
    np_count = geom["np_count"]

    kd_arr = k * np.asarray(seg_h, dtype=np.float64)
    sin_kd = np.sin(kd_arr)
    cos_kd = np.cos(kd_arr)
    sin_kd_2 = np.sin(0.5 * kd_arr)
    cos_kd_2 = np.cos(0.5 * kd_arr)
    sin_kd_4 = np.sin(0.25 * kd_arr)
    recip_gap = _recip_sin_gap(kd_arr)
    tan_half = sin_kd_2 / cos_kd_2

    # Each neighbour's factor at the node it shares with the basis.
    a_nm = np.where(nm_sigma > 0, a_plus[nm_seg], a_minus[nm_seg])
    a_np = np.where(np_sigma > 0, a_minus[np_seg], a_plus[np_seg])

    P_minus = np.zeros(n_segs, dtype=np.float64)
    P_plus = np.zeros(n_segs, dtype=np.float64)
    np.add.at(P_minus, nm_basis, a_nm * tan_half[nm_seg])
    np.add.at(P_plus, np_basis, -(a_np * tan_half[np_seg]))

    ground_minus = geom["ground_minus"]
    ground_plus = geom["ground_plus"]
    if ground_minus.any():
        P_minus = P_minus + np.where(ground_minus, a_minus * tan_half, 0.0)
    if ground_plus.any():
        P_plus = P_plus - np.where(ground_plus, a_plus * tan_half, 0.0)

    crossing_minus = geom.get("crossing_minus")
    crossing_plus = geom.get("crossing_plus")
    if crossing_minus is None:
        crossing_minus = np.zeros(n_segs, dtype=bool)
        crossing_plus = np.zeros(n_segs, dtype=bool)
    has_minus = (nm_count > 0) | ground_minus | crossing_minus
    has_plus = (np_count > 0) | ground_plus | crossing_plus

    am = a_minus
    ap = a_plus
    D = (P_minus * P_plus + am * ap) * sin_kd + (P_minus * ap - P_plus * am) * cos_kd
    D_safe = np.where(D != 0, D, 1.0)
    sin_kd_safe = np.where(sin_kd != 0, sin_kd, 1.0)
    one_m_cos_kd = 2.0 * sin_kd_2 * sin_kd_2
    Q_minus = (ap * one_m_cos_kd - P_plus * sin_kd) / D_safe
    Q_plus = (-am * one_m_cos_kd - P_minus * sin_kd) / D_safe
    A_i0 = np.full(n_segs, -1.0)
    B_i0 = (am * Q_minus + ap * Q_plus) * sin_kd_2 / sin_kd_safe
    C_i0 = (am * Q_minus - ap * Q_plus) * cos_kd_2 / sin_kd_safe
    dP = P_minus * ap - P_plus * am
    s4sq = sin_kd_4 * sin_kd_4
    AC_i0 = (
        4.0 * am * ap * sin_kd_2 * s4sq
        - 2.0 * dP * s4sq
        - 2.0 * sin_kd_2 * cos_kd_2 * P_minus * P_plus
        + 2.0 * dP * sin_kd_2 * sin_kd_2
    ) / D_safe

    only_plus = has_plus & ~has_minus
    only_minus = has_minus & ~has_plus
    iso = ~has_minus & ~has_plus
    if only_plus.any() or only_minus.any() or iso.any():
        X = x_end
        denom_x = cos_kd - X * sin_kd
        denom_x_safe = np.where(denom_x != 0, denom_x, 1.0)
        # 1 - cos kD + X sin kD, the end-branch numerator (3-33) / (3-37).
        gap_x = one_m_cos_kd + X * sin_kd
        cap_s = sin_kd_2 + X * cos_kd_2
        cap_c = cos_kd_2 - X * sin_kd_2
        qpe1_denom = (ap + X * P_plus) * sin_kd + (ap * X - P_plus) * cos_kd
        qpe1_denom_safe = np.where(qpe1_denom != 0, qpe1_denom, 1.0)
        Q_plus_e1 = -gap_x / qpe1_denom_safe
        B_e1 = (sin_kd_2 + ap * Q_plus_e1 * cap_c) / denom_x_safe
        C_e1 = (cos_kd_2 + ap * Q_plus_e1 * cap_s) / denom_x_safe
        qme2_denom = (am - X * P_minus) * sin_kd + (am * X + P_minus) * cos_kd
        qme2_denom_safe = np.where(qme2_denom != 0, qme2_denom, 1.0)
        Q_minus_e2 = gap_x / qme2_denom_safe
        B_e2 = (-sin_kd_2 + am * Q_minus_e2 * cap_c) / denom_x_safe
        C_e2 = (cos_kd_2 - am * Q_minus_e2 * cap_s) / denom_x_safe
        iso_den = cos_kd_2 - X * sin_kd_2
        iso_den_safe = np.where(iso_den != 0, iso_den, 1.0)
        C_iso = 1.0 / iso_den_safe

        Q_minus = np.where(only_plus | iso, 0.0, Q_minus)
        Q_minus = np.where(only_minus, Q_minus_e2, Q_minus)
        Q_plus = np.where(only_minus | iso, 0.0, Q_plus)
        Q_plus = np.where(only_plus, Q_plus_e1, Q_plus)
        B_i0 = np.where(only_plus, B_e1, B_i0)
        B_i0 = np.where(only_minus, B_e2, B_i0)
        B_i0 = np.where(iso, 0.0, B_i0)
        C_i0 = np.where(only_plus, C_e1, C_i0)
        C_i0 = np.where(only_minus, C_e2, C_i0)
        C_i0 = np.where(iso, C_iso, C_i0)
        # A + C = C - 1 over the common denominator, every term no larger
        # than the answer: cos(kD/2) - cos kD by sum-to-product, as
        # `_basis_coefs` spells it, and the X terms never cancel.
        cos_gap_e = 2.0 * np.sin(0.75 * kd_arr) * sin_kd_4 + X * sin_kd
        AC_e1 = (cos_gap_e + ap * Q_plus_e1 * cap_s) / denom_x_safe
        AC_e2 = (cos_gap_e - am * Q_minus_e2 * cap_s) / denom_x_safe
        AC_iso = (2.0 * s4sq + X * sin_kd_2) / iso_den_safe
        AC_i0 = np.where(only_plus, AC_e1, AC_i0)
        AC_i0 = np.where(only_minus, AC_e2, AC_i0)
        AC_i0 = np.where(iso, AC_iso, AC_i0)

    if ground_minus.any():
        Qg = np.where(ground_minus, am * Q_minus, 0.0)
        A_i0 = A_i0 + Qg / sin_kd_safe
        B_i0 = B_i0 - Qg / (2.0 * cos_kd_2)
        C_i0 = C_i0 - Qg / (2.0 * sin_kd_2)
        AC_i0 = AC_i0 + Qg * recip_gap
    if ground_plus.any():
        Qg = np.where(ground_plus, ap * Q_plus, 0.0)
        A_i0 = A_i0 - Qg / sin_kd_safe
        B_i0 = B_i0 - Qg / (2.0 * cos_kd_2)
        C_i0 = C_i0 + Qg / (2.0 * sin_kd_2)
        AC_i0 = AC_i0 - Qg * recip_gap

    self_seg = np.arange(n_segs, dtype=np.int64)
    nm_aQ = a_nm * Q_minus[nm_basis]
    nm_A = nm_aQ / sin_kd[nm_seg]
    nm_B = nm_aQ / (2.0 * cos_kd_2[nm_seg])
    nm_C = -nm_aQ / (2.0 * sin_kd_2[nm_seg])
    nm_AC = nm_aQ * recip_gap[nm_seg]
    np_aQ = a_np * Q_plus[np_basis]
    np_A = -np_aQ / sin_kd[np_seg]
    np_B = np_aQ / (2.0 * cos_kd_2[np_seg])
    np_C = np_aQ / (2.0 * sin_kd_2[np_seg])
    np_AC = -np_aQ * recip_gap[np_seg]

    all_seg = np.concatenate([self_seg, nm_seg, np_seg])
    all_basis = np.concatenate([self_seg, nm_basis, np_basis])
    all_A = np.concatenate([A_i0, nm_A, np_A]).astype(np.complex128)
    all_B = np.concatenate([B_i0, nm_B, np_B]).astype(np.complex128)
    all_C = np.concatenate([C_i0, nm_C, np_C]).astype(np.complex128)
    all_AC = np.concatenate([AC_i0, nm_AC, np_AC]).astype(np.complex128)
    all_sigma = np.concatenate([np.ones(n_segs, dtype=np.int8), nm_sigma, np_sigma])
    order = np.argsort(all_seg, kind="stable")
    counts = np.ones(n_segs, dtype=np.int64) + nm_count + np_count
    starts = np.empty(n_segs + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])
    return {
        "starts": starts,
        "jbasis": all_basis[order],
        "A": all_A[order],
        "B": all_B[order],
        "C": all_C[order],
        "AC": all_AC[order],
        "sigma": all_sigma[order],
    }


def _recip_sin_gap(kd):
    # Imported late: `sinusoidal` imports this module.
    from .sinusoidal import _recip_sin_gap as impl

    return impl(kd)


# ---------------------------------------------------------------------------
# The NEC-4 thin-wire kernel (theory manual Section 4.1)
# ---------------------------------------------------------------------------
#
# NEC-4 puts each segment's current on a tube at the SOURCE segment's radius
# and matches on the observer's axis, R = sqrt(rho^2 + a_src^2 + (z - z')^2)
# with rho the axis-to-axis distance ((4-26) to (4-29)), where NEC-2 puts a
# filament on the source axis and matches on the observer's surface (the
# observer's radius in R, `_pair_geometry`). With the radius taken from the
# source, the two segments either side of a radius step see the node through
# different R, so the point charges NEC-2's closed forms carry at every
# segment end no longer cancel there; NEC-4 drops them (its basis already
# satisfies Kirchhoff's law, Section 2.1), which is (4-14) to (4-21): the
# NEC-2 fields of this lane (`_field_components_bcast`) less their end-charge
# terms. Its radial forms take the bare 1/rho.
#
# In this lane's shapes, unit current, pref_z = j eta/(4 pi k), G = e^{-jkR}/R,
# at the source ends z' = +-H with dz = z - z':
#   const : E_z = -pref_z k^2 int G dz'               E_rho = 0
#   sin   : E_z =  pref_z k [G cos kz']               E_rho = -(j eta/4 pi)
#                    [G dz cos kz' / rho - j e^{-jkR} sin kz' / rho]
#   cos   : E_z = -pref_z k [G sin kz']               E_rho = -(j eta/4 pi)
#                    [-G dz sin kz' / rho - j e^{-jkR} cos kz' / rho]
# each [.] the difference between the z' = +H and z' = -H ends.

# The radial projection (rho_vec . t_obs)/rho^2 is taken as zero below this
# fraction of the source half-length: rho_vec . t_obs vanishes with rho on a
# collinear pair, and rounding must not leave a 0/0 there.
_RHO_FLOOR = 1e-9


def nec4_field_components(k, eta, obs_c, obs_t, src_c, src_t, src_hh, src_a, gx, gw):
    """The NEC-4 kernel's tangential fields, `(Phi_const, Phi_sin, Phi_cos)`,
    observers (M, 3) against sources (N, 3); numpy reference for the C++
    `sinusoidal_field_tensor_nec4`. `src_a` is each SOURCE segment's radius,
    `src_hh` its half length."""
    from ._stable import asinh_diff, expm1_neg_j_from_half

    oc = obs_c[:, None, :]
    ot = obs_t[:, None, :]
    sc = src_c[None, :, :]
    st = src_t[None, :, :]
    rvec = oc - sc
    z = np.einsum("...d,...d->...", rvec, st)
    rho_vec = rvec - z[..., None] * st
    rho2 = np.einsum("...d,...d->...", rho_vec, rho_vec)
    rho = np.sqrt(rho2)
    td = (ot * st).sum(axis=-1)
    rdot = (rho_vec * ot).sum(axis=-1)
    H = np.broadcast_to(src_hh[None, :], z.shape)
    a2 = np.broadcast_to((src_a * src_a)[None, :], z.shape)
    b2 = rho2 + a2
    dz2 = z - H
    dz1 = z + H
    r2 = np.sqrt(b2 + dz2 * dz2)
    r1 = np.sqrt(b2 + dz1 * dz1)
    e2 = np.exp(-1j * k * r2)
    e1 = np.exp(-1j * k * r1)
    g2 = e2 / r2
    g1 = e1 / r1
    pref_z = 1j * eta / (4.0 * np.pi * k)
    pre_rho = -1j * eta / (4.0 * np.pi)

    int_inv_r = asinh_diff(-dz1, -dz2, b2, r1, r2)
    zq = H[..., None] * gx
    dzq = z[..., None] - zq
    rq = np.sqrt(b2[..., None] + dzq * dzq)
    reg = expm1_neg_j_from_half(-0.5 * k * rq) / rq
    int_g = int_inv_r + np.einsum("...q,q->...", reg, gw) * H

    sin2 = np.sin(k * H)
    cos2 = np.cos(k * H)
    ez_c = -pref_z * (k * k) * int_g
    ez_s = pref_z * k * (g2 * cos2 - g1 * cos2)
    ez_o = -pref_z * k * (g2 * sin2 + g1 * sin2)

    on = rho > _RHO_FLOOR * H
    proj = np.where(on, rdot / np.where(on, rho2, 1.0), 0.0)
    # [.]_{-H}^{+H}; sin(-kH) = -sin2, cos(-kH) = cos2.
    br_s = (g2 * dz2 * cos2 - 1j * e2 * sin2) - (g1 * dz1 * cos2 + 1j * e1 * sin2)
    br_o = (-g2 * dz2 * sin2 - 1j * e2 * cos2) - (g1 * dz1 * sin2 - 1j * e1 * cos2)
    er_s = pre_rho * proj * br_s
    er_o = pre_rho * proj * br_o
    return td * ez_c, td * ez_s + er_s, td * ez_o + er_o


# ---------------------------------------------------------------------------
# End caps (theory manual Section 4.4, Appendix E)
# ---------------------------------------------------------------------------
#
# NEC-4 closes every free wire end with a flat cap carrying a uniform surface
# charge that keeps current and charge continuous from the wire onto it. The
# basis takes X = k a / 2 at the end ((4-36) in (3-12) / (3-13)); the cap's
# charge, (4-35), rho_c = j S I(end) / (omega pi a^2), S = +1 when the
# segment's reference direction points away from the cap (end 1) and -1 at
# end 2, adds its scalar-potential field (4-37) to the end basis function's
# column. On the cap's axis the field is (E-3), in closed form; off it,
# Appendix E's expansion of the potential to second order in the radius,
# whose gradient is (E-1) / (E-2) (re-derived; the potential is
# C0 e^{-jkR0}/R0 [1 + a^2 (rho^2 (3 + 3jkR0 - k^2 R0^2)
# - 2 (1 + jkR0) R0^2) / (8 R0^4)] with C0 = q / (4 pi eps), q = rho_c pi a^2).
# The vector potential of the cap current is neglected, as the manual does.
# No source-gap cap ((4-38) to (4-40)): licensed NEC-4.2 is measured not to
# carry one.

# On-axis below this fraction of the radius: a collinear observer has rho = 0
# up to rounding.
_CAP_AXIS_FRACTION = 1e-6


def end_cap_x(geom, k, seg_radius):
    """X = k a / 2 on every segment with a free end (no neighbour, no ground
    contact, no crossing node on that side), 0 elsewhere."""
    free_m, free_p = free_ends(geom)
    return np.where(free_m | free_p, 0.5 * k * seg_radius, 0.0)


def free_ends(geom):
    """`(end-1 free, end-2 free)` per segment."""
    n = geom["n_segs"]
    zeros = np.zeros(n, dtype=bool)
    cm = geom.get("crossing_minus", zeros)
    cp = geom.get("crossing_plus", zeros)
    free_m = (geom["nm_count"] == 0) & ~geom["ground_minus"] & ~cm
    free_p = (geom["np_count"] == 0) & ~geom["ground_plus"] & ~cp
    return free_m, free_p


def end_caps(geom, k, seg_view, seg_radius):
    """`[(segment, cap point, outward normal, radius, C0)]` for every free
    end, C0 = q / (4 pi eps) per unit eta: j S I(end) / (4 pi k), with I(end)
    the segment's own basis function at the end. (`eta` multiplies it at
    the field.) The basis is real at a real k, so I(end) is read off the
    real parts, in scalar float arithmetic (the C++ fill repeats it)."""
    free_m, free_p = free_ends(geom)
    seg_c = geom["seg_centers"]
    seg_t = geom["seg_tangents"]
    seg_h = geom["seg_h"]
    starts = seg_view["starts"]
    jb = seg_view["jbasis"]
    out = []
    for i in np.flatnonzero(free_m | free_p):
        lo, hi = int(starts[i]), int(starts[i + 1])
        e = lo + int(np.flatnonzero(jb[lo:hi] == i)[0])
        A = float(seg_view["A"][e].real)
        B = float(seg_view["B"][e].real)
        C = float(seg_view["C"][e].real)
        h = float(seg_h[i])
        u = 0.5 * k * h
        c = [float(v) for v in seg_c[i]]
        t = [float(v) for v in seg_t[i]]
        for free, sgn in ((free_m[i], -1.0), (free_p[i], 1.0)):
            if not free:
                continue
            i_end = A + B * math.sin(sgn * u) + C * math.cos(u)
            S = -sgn  # +1 at end 1, -1 at end 2
            off = sgn * 0.5 * h
            pos = np.array([c[0] + off * t[0], c[1] + off * t[1], c[2] + off * t[2]])
            normal = np.array([sgn * t[0], sgn * t[1], sgn * t[2]])
            c0 = complex(0.0, S * i_end / (4.0 * math.pi * k))
            out.append((int(i), pos, normal, float(seg_radius[i]), c0))
    return out


def cap_field(k, eta, obs_c, obs_t, pos, normal, a, c0):
    """Tangential field at the observers of one charged flat cap, (M,): the
    C++ `sinusoidal_cap_field` when the extension is loaded (the C++ fill
    calls the same code), `cap_field_numpy` otherwise."""
    acc = _accel()
    if acc is not None:
        return acc.sinusoidal_cap_field(
            np.ascontiguousarray(obs_c, dtype=np.float64),
            np.ascontiguousarray(obs_t, dtype=np.float64),
            np.ascontiguousarray(pos, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(a),
            float(k),
            float(eta),
            complex(c0),
        )
    return cap_field_numpy(k, eta, obs_c, obs_t, pos, normal, a, c0)


def _accel():
    from ._accel import acc

    return acc if acc is not None and hasattr(acc, "sinusoidal_cap_field") else None


def cap_field_numpy(k, eta, obs_c, obs_t, pos, normal, a, c0):
    """`cap_field`'s numpy reference."""
    from ._stable import expm1_neg_j

    d = obs_c - pos[None, :]
    h = d @ normal
    perp = d - h[:, None] * normal[None, :]
    rho = np.sqrt(np.einsum("md,md->m", perp, perp))
    tn = obs_t @ normal
    tr = np.einsum("md,md->m", perp, obs_t)  # rho * (rho_hat . t)
    pref = eta * c0
    out = np.zeros(obs_c.shape[0], dtype=np.complex128)
    on = rho <= _CAP_AXIS_FRACTION * a
    if on.any():
        hh = h[on]
        ah = np.abs(hh)
        ra = np.sqrt(a * a + hh * hh)
        dd = a * a / (ra + ah)  # Ra - |h|
        # E_h = (2 C0 / a^2) sgn(h) e^{-jk|h|} [1 - (|h|/Ra) e^{-jk(Ra-|h|)}]
        br = dd / ra - (ah / ra) * expm1_neg_j(k * dd)
        eh = (2.0 / (a * a)) * np.sign(hh) * np.exp(-1j * k * ah) * br
        out[on] = pref * eh * tn[on]
    off = ~on
    if off.any():
        hh = h[off]
        rr = rho[off]
        r2 = hh * hh + rr * rr
        r0 = np.sqrt(r2)
        jk = 1j * k * r0
        one = 1.0 + jk
        k2r2 = k * k * r2
        jk3 = 1j * k * k2r2 * r0
        base = 8.0 * r2 * r2 * one
        p_h = base + a * a * (
            3.0 * one * (3.0 * rr * rr - 2.0 * hh * hh)
            + 2.0 * k2r2 * (hh * hh - 2.0 * rr * rr)
            - jk3 * rr * rr
        )
        p_r = base + a * a * (
            3.0 * one * (rr * rr - 4.0 * hh * hh)
            + 2.0 * k2r2 * (2.0 * hh * hh - rr * rr)
            - jk3 * rr * rr
        )
        g = np.exp(-1j * k * r0) / (8.0 * r2 * r2 * r2 * r0)
        # E_h = C0 h g p_h, E_rho = C0 rho g p_r; E . t = E_h tn + E_rho tr/rho
        out[off] = pref * g * (hh * p_h * tn[off] + p_r * tr[off])
    return out
