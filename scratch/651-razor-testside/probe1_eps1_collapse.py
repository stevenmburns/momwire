"""Razor test-side probe (momwire#651, 2026-09-02): at eps_tilde = 1, does the
crossing trunk (`_crossing_fill`, post-#801) with a RAZOR-BLADE test axis and
razor's tents as the source basis reproduce razor's own free-space cross block
between the two wires of `crossing_deck(level=1)`?

Three steps, all against razor's free-space Z as the truth:
  1. interior rows x interior columns          -> 6.6e-6 relative, ratio 1.000000
  2. the junction tent's COLUMN                 -> 7.2e-9 vs razor's below-wing piece;
                                                   node end terms contribute 0.000 to razor rows
  3. the junction tent's ROW, above half        -> 5.3e-5 vs razor's kernel chopped at the
     (endpoint IN the plane)                       node, PROVIDED the trunk's corner is not
                                                   applied to razor rows (it adds 1.9e5)

The razor test axis is written in `axis_data`'s own language: F = 1 on each
row's own path, Fd = 0, ends = the path's two centroids with the T2 signs.
Findings and the route decision are on momwire#651.

    python scratch/651-razor-testside/probe1_eps1_collapse.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from test_crossing_serve_524 import A_WIRE, crossing_deck

from momwire import RazorSolver
from momwire import _crossing_fill as CF

warnings.simplefilter("ignore")

deck = crossing_deck(1)
fs = {
    k: v for k, v in deck.items() if k not in ("ground_z", "ground_eps", "ground_model")
}
rs = RazorSolver(**fs, nec5_quadrature=False)
geom = rs._build_geometry()
k = rs.k
omega = rs.omega
n_basis = geom["n_basis_total"]
seg_off = np.asarray(geom["seg_offsets"])
bas_off = np.asarray(geom["basis_offsets"])
seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
wing_seg, wing_rise, wing_sigma = (
    geom["wing_seg"],
    geom["wing_rise"],
    geom["wing_sigma"],
)
print(
    f"razor: n_seg={seg_h.size} n_basis={n_basis} (interior {geom['n_basis_interior']}) k={k:.5f}"
)

# --- razor's tents as BasisPolynomials -------------------------------------
supp = np.zeros((n_basis, 2), dtype=np.int64)
polys = np.zeros((n_basis, 2, 2))
for n in range(n_basis):
    for j in range(2):
        s, rise, sig = wing_seg[n, j], wing_rise[n, j], wing_sigma[n, j]
        supp[n, j] = s
        h = seg_h[s]
        polys[n, j] = sig * (
            np.array([0.0, 1.0 / h]) if rise else np.array([1.0, -1.0 / h])
        )
basis = CF.BasisPolynomials(supp, polys, 1)
ageom = CF.AxisGeometry(seg_p0, seg_p0 + seg_h[:, None] * seg_t, seg_h, seg_t, seg_off)
medium = CF.buried_medium((1.0, 0.0), omega, rs.eps, k)
print("medium at eps_tilde=1:", medium)
ctx = CF.CrossingContext(basis, ageom, medium, 0.0, float(A_WIRE), omega, rs.mu, rs.eps)

b_idx = np.arange(seg_off[0], seg_off[1])  # wire 0 = below
a_idx = np.arange(seg_off[1], seg_off[2])  # wire 1 = above
B = CF.axis_data(ctx, b_idx)

# --- razor-blade test axis on the above wire ------------------------------
pts, tans, wts = rs._testing_paths(geom)  # (n_basis, 2q, 3/3/-)
cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
q = pts.shape[1] // 2
rows_above = np.arange(bas_off[1], bas_off[2])  # interior knots of wire 1
cols_below = np.arange(bas_off[0], bas_off[1])  # interior tents of wire 0
nodes, tl, wl, F, segof, ends = [], [], [], [], [], []
for m in rows_above:
    nodes.append(pts[m])
    tl.append(tans[m])
    wl.append(wts[m])
    f = np.zeros((n_basis, 2 * q))
    f[m] = 1.0
    F.append(f)
    segof.append(
        np.concatenate([np.full(q, wing_seg[m, 0]), np.full(q, wing_seg[m, 1])])
    )
    e = np.zeros(n_basis)
    e[m] = 1.0
    ends.append((cent[wing_seg[m, 0]], -1.0, e))  # c_before: path starts here
    ends.append((cent[wing_seg[m, 1]], +1.0, e))  # c_after: path ends here
A = dict(
    nodes=np.concatenate(nodes),
    t=np.concatenate(tl),
    w=np.concatenate(wl),
    F=np.concatenate(F, axis=1),
    Fd=np.zeros((n_basis, len(rows_above) * 2 * q)),
    ends=ends,
    n_basis=n_basis,
    segof=np.concatenate(segof),
)

t_ab = CF.cross_complete_block(ctx, A, B)

# --- razor's own free-space Z ------------------------------------------------
prep = rs._assemble_Z_prepare(geom)
Z = rs._assemble_Z_from_prepared(geom, prep, k, omega)
ref = Z[np.ix_(rows_above, cols_below)]
got = -t_ab[np.ix_(rows_above, cols_below)]
scale = np.abs(ref).max()
print(f"block {ref.shape}; max|Z_ref| = {scale:.4e}")
print(f"max|got - ref| / max|ref| = {np.abs(got - ref).max() / scale:.3e}")
ratio = got / ref
big = np.abs(ref) > 1e-3 * scale
print(
    f"elementwise got/ref over |ref|>1e-3 max: median {np.median(ratio[big].real):+.6f}{np.median(ratio[big].imag):+.6f}j, "
    f"spread {np.abs(ratio[big] - np.median(ratio[big])).max():.3e}"
)
# nearest-to-node entry and a far entry, printed raw
i0 = rows_above[0] - rows_above[0]
j0 = cols_below[-1] - cols_below[0]
print(f"near-node entry ref {ref[i0, j0]:.6e}  got {got[i0, j0]:.6e}")
print(f"far entry       ref {ref[-1, 0]:.6e}  got {got[-1, 0]:.6e}")

# ===== step 2: the junction tent's column =====================================
jn = n_basis - 1
print(
    "\n--- step 2: junction tent",
    jn,
    "wings",
    wing_seg[jn],
    "rise",
    wing_rise[jn],
    "sigma",
    wing_sigma[jn],
)
# razor's below-wing-only piece: zero the above wing (sigma=0 zeroes T1, T2 and its half-path)
g2 = dict(geom)
g2["wing_sigma"] = geom["wing_sigma"].copy()
g2["wing_sigma"][jn, 1] = 0.0
Z_A = rs._assemble_Z_from_prepared(g2, rs._assemble_Z_prepare(g2), k, omega)
col_full = Z[rows_above, jn]
col_Aonly = Z_A[rows_above, jn]
col_trunk = -t_ab[rows_above, jn]
sc = np.abs(col_full).max()
print(f"max|col_full| {sc:.4e}")
print(
    f"trunk vs razor below-wing-only : max|diff|/max|col_full| = {np.abs(col_trunk - col_Aonly).max() / sc:.3e}"
)
print(
    f"trunk vs razor full tent       : max|diff|/max|col_full| = {np.abs(col_trunk - col_full).max() / sc:.3e}"
)
# what the trunk's source-end term at the node is worth here
B_noend = dict(B)
B_noend["ends"] = [e for e in B["ends"] if abs(e[0][2]) > 1e-9]
t_noend = CF.cross_complete_block(ctx, A, B_noend)
print(
    f"node end-term share of the column: {np.abs(t_ab[rows_above, jn] - t_noend[rows_above, jn]).max() / sc:.3e}"
)
# the same three numbers for the junction ROW would need the node path; step 3.

# ===== step 3: the junction ROW's above half (endpoint IN the plane) ==========
print("\n--- step 3: junction row", jn, "above half: node -> cent(B)")
g3 = dict(geom)
g3["wing_sigma"] = geom["wing_sigma"].copy()
g3["wing_sigma"][jn, 0] = 0.0
Z_B = rs._assemble_Z_from_prepared(g3, rs._assemble_Z_prepare(g3), k, omega)
row_ref = Z_B[
    jn, cols_below
]  # razor: above half-path only, against below interior tents
# the trunk's node row: nodes = the above half of path 28 (second q points), ends = node (-1), cent(B) (+1)
pts_n, tans_n, wts_n = pts[jn, q:], tans[jn, q:], wts[jn, q:]
node_pt = rs._knot_points(geom)[jn]
e = np.zeros(n_basis)
e[jn] = 1.0
f = np.zeros((n_basis, q))
f[jn] = 1.0
A3 = dict(
    nodes=pts_n,
    t=tans_n,
    w=wts_n,
    F=f,
    Fd=np.zeros((n_basis, q)),
    n_basis=n_basis,
    segof=np.full(q, wing_seg[jn, 1]),
    ends=[(node_pt, -1.0, e), (cent[wing_seg[jn, 1]], +1.0, e)],
)
t3 = CF.cross_complete_block(ctx, A3, B)
row_trunk = -t3[jn, cols_below]
sr = np.abs(row_ref).max()
print(f"max|row_ref| {sr:.4e}")
print(
    f"trunk (corner fires on the in-plane end) vs razor: max|diff|/max = {np.abs(row_trunk - row_ref).max() / sr:.3e}"
)
# the corner only touches the (row 28, col 28) entry; the interior columns see the BT term at z=0+.
print(
    f"row entries  ref  {row_ref[-1]:.6e}  trunk {row_trunk[-1]:.6e}   (col nearest the node)"
)
print(
    f"(28,28) entry: trunk {-t3[jn, jn]:.6e}  razor-full {Z[jn, jn]:.6e}  razor-B-half {Z_B[jn, jn]:.6e}"
)
B3 = dict(B)
B3["ends"] = [ee for ee in B["ends"] if abs(ee[0][2]) > 1e-9]
t3n = CF.cross_complete_block(ctx, A3, B3)
print(f"(28,28) without the node source-end (no corner, no SW/SQ): {-t3n[jn, jn]:.6e}")

# ===== step 3b: the honest half-row: T1(above half) + T2(node -> cent B) ======
print("\n--- step 3b: razor's own kernel, chopped at the node")
T2t = Z_A + Z_B - Z  # the full-path T2 term, in Z units (both endpoints = centroids)
T1B_Z = Z_B - T2t  # jωμ·T1 of the above half alone, in Z units
a_src = rs._kernel_radius(geom)
obs = np.array([node_pt, cent[wing_seg[jn, 1]]])  # c_before = node, c_after = cent(B)
M0 = rs._seg_moments_from_prepared(
    rs._seg_moments_prepare(obs, geom, a_src), k, 2, need_m1=False
)[0]
dM0 = M0[1] - M0[0]
s_a, s_b, q_a, q_b = prep["s_a"], prep["s_b"], prep["q_a"], prep["q_b"]
T2h = dM0[s_a] * q_a + dM0[s_b] * q_b  # per source basis n
razor_half = T1B_Z[jn] - T2h / (1j * omega * rs.eps)
rr = razor_half[cols_below]
print(
    f"half-row (interior cols): trunk vs razor-kernel: max|diff|/max|ref| = {np.abs(row_trunk - rr).max() / np.abs(rr).max():.3e}"
)
print(f"  nearest-node entry  razor {rr[-1]:.6e}   trunk {row_trunk[-1]:.6e}")
# T1 alone: trunk with no A ends (and B's node end removed so no SW/SQ/corner) vs razor's T1B
A3n = dict(A3)
A3n["ends"] = []
t1only = -CF.cross_complete_block(ctx, A3n, B3)[jn, cols_below]
print(
    f"T1 alone: trunk vs razor: {np.abs(t1only - T1B_Z[jn, cols_below]).max() / np.abs(T1B_Z[jn, cols_below]).max():.3e}"
)
# (28,28) for the cross block = below wing of tent 28 as the only source: T1B against wing A + T2 half with q_a only
row28_T1 = T1B_Z[
    jn, jn
]  # includes the ABOVE wing as source too (same medium) -- not comparable directly
print(
    f"(28,28) trunk w/o node source-end {-t3n[jn, jn]:.6e}; razor T2-half below-wing-only term {(-dM0[s_a[jn]] * q_a[jn] / (1j * omega * rs.eps)):.6e}"
)
