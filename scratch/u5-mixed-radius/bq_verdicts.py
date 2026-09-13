"""Read step (b') against its registered predictions (MEASUREMENTS.md)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = ("control", "ratio2", "ratio4", "ratio05")
LENGTHS = (0.30, 0.60, 1.20)


def load(cfg):
    rows = json.loads((HERE / f"bq_{cfg}.json").read_text())["rows"]
    return {(row["L"], row["r"]): row for row in rows}


def z_of(row, which):
    rec = row["node_B"]
    return complex(rec["z"] if which == "split" else rec["vc_keep"]["z"])


def dz(row, which):
    return complex(row["nec5"]) - z_of(row, which)


def rich(rows, L, which):
    return 2 * dz(rows[(L, 4)], which) - dz(rows[(L, 2)], which)


def step_ratio(rows, L, which):
    z1, z2, z4 = (z_of(rows[(L, r)], which) for r in (1, 2, 4))
    return abs(z4 - z2) / abs(z2 - z1)


def verdict(ok):
    return "HIT" if ok else "MISSED"


data = {cfg: load(cfg) for cfg in CONFIGS}
ctl = data["control"]
band = {
    L: max(abs(dz(ctl[(L, 4)], "split") - dz(ctl[(L, 2)], "split")), 0.5)
    for L in LENGTHS
}

for which, pid in (("split", "Pq.1"), ("keep", "Pq.1k")):
    print(f"== {pid} ({'node_B' if which == 'split' else 'VC_keep(node_B)'})")
    ok = True
    for cfg in CONFIGS[1:]:
        for L in LENGTHS:
            d_mix = rich(data[cfg], L, which)
            d_ctl = rich(ctl, L, "split")
            diff = abs(d_mix - d_ctl)
            r4 = abs(dz(data[cfg][(L, 4)], which) - dz(ctl[(L, 4)], "split"))
            ok &= diff <= band[L]
            print(
                f"{cfg:8s} L={L:.2f} dZinf {d_mix:.4f} control {d_ctl:.4f} "
                f"|diff| {diff:.3f} band {band[L]:.3f} "
                f"{'inside' if diff <= band[L] else 'OUTSIDE'} | r=4 alone {r4:.3f}"
            )
    print(f"{pid} {verdict(ok)}")

print("== Pq.2 (|node_B - VC_keep| <= 0.1 Ohm, every rung)")
worst = max(
    abs(z_of(row, "split") - z_of(row, "keep"))
    for cfg in CONFIGS
    for row in data[cfg].values()
)
print(f"worst {worst:.4f} -> {verdict(worst <= 0.1)}")

print("== Pq.3 (node_B KCL <= 1e-5 at r = 4 and falling from r = 1)")
ok = True
for cfg in CONFIGS:
    for L in LENGTHS:
        k = [data[cfg][(L, r)]["node_B"]["kcl_rel"] for r in (1, 2, 4)]
        good = k[2] <= 1e-5 and k[2] < k[0]
        ok &= good
        print(
            f"{cfg:8s} L={L:.2f} kcl r1 {k[0]:.2e} r2 {k[1]:.2e} r4 {k[2]:.2e} {'ok' if good else 'NO'}"
        )
print(f"Pq.3 {verdict(ok)}")

print("== Pq.4 (momwire step ratio within +-0.15 of the control's)")
ok = True
for L in LENGTHS:
    q_c = step_ratio(ctl, L, "split")
    for cfg in CONFIGS[1:]:
        q = step_ratio(data[cfg], L, "split")
        good = abs(q - q_c) <= 0.15
        ok &= good
        print(
            f"{cfg:8s} L={L:.2f} ratio {q:.3f} control {q_c:.3f} {'ok' if good else 'NO'}"
        )
print(f"Pq.4 {verdict(ok)}")
