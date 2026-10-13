"""momwire#1170 stage 0: the reference comparison, from the recorded runs.

usage: python scratch/1170-stage0/table.py

Reads `results/momwire_*.jsonl` (written by `runset.sh` -> `refrun.py`, one
line per deck and mode) and the engines' printed impedances below, and
prints the tables in README.md. The engines were run as black boxes on the
same deck files (`run_engine.sh`); their printouts are not committed, only
the impedance each printed under ANTENNA INPUT PARAMETERS.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    rows = {}
    for line in (HERE / "results" / name).read_text().splitlines():
        d = json.loads(line)
        rows[(d["deck"][:-4], d["mode"])] = d
    return rows


chg = load("momwire_change_7481b353.jsonl")
base = load("momwire_base_afe9e92e.jsonl")

# NEC-4.2 (nec42cl-serial, release 7768648, on the laptop) for the .nec4
# decks; clean-room NEC-5 (nec5cl-cleanroom, on Rocketlake) for the .nec5
# ones. 2026-10-13.
REF = {
    "g8hul_x1.nec4": ("NEC-4.2", 2925.33 + 257.284j),
    "g8hul_x3.nec4": ("NEC-4.2", 2873.61 - 450.905j),
    "g8hul_nt_x1.nec4": ("NEC-4.2", 59.8084 + 5.21897j),
    "g8hul_nt_x3.nec4": ("NEC-4.2", 58.7375 - 9.23229j),
    "brv_x1.nec4": ("NEC-4.2", 101.158 - 57.8815j),
    "brv_x2.nec4": ("NEC-4.2", 102.117 - 60.0176j),
    "brv_x4.nec4": ("NEC-4.2", 100.993 - 58.7559j),
    "fat_x1.nec4": ("NEC-4.2", 111.239 - 66.1858j),
    "fat_x2.nec4": ("NEC-4.2", 111.033 - 66.7439j),
    "g8hul_x1.nec5": ("NEC-5", 2725.4 + 753.78j),
    "g8hul_x3.nec5": ("NEC-5", 2920.7 - 250.10j),
    "brv_x1.nec5": ("NEC-5", 112.25 - 76.157j),
    "brv_x2.nec5": ("NEC-5", 100.19 - 62.253j),
    "brv_x4.nec5": ("NEC-5", 100.42 - 60.122j),
    "fat_x1.nec5": ("NEC-5", 111.77 - 69.726j),
    "fat_x2.nec5": ("NEC-5", 111.28 - 68.855j),
}


def z(row):
    """The feed impedance: the solver's own port matrix at full precision,
    except on the two decks fed through a network, where the printed source
    port is the quantity the engine prints too."""
    if "_nt_" in row["deck"]:
        return complex(*row["ports"][0]["z"])
    return complex(*row["z_ports_full"][0][0])


print("today's fallback is the forced-reduced run, bit for bit")
print("(base tree as served == change tree forced reduced, solver port Z):")
for key in sorted(k for k in REF if (k, "served") in base):
    b, c = base[(key, "served")], chg[(key, "reduced")]
    assert b["kernels"][0]["extended_kernel"] is False, key
    print(f"  {key:18s} {b['z_ports_full'] == c['z_ports_full']}")

print()
print(
    "| deck | reference | reference Z | all reduced (today) | extended above "
    "| shift | gap before | gap after | moves |"
)
print("|---|---|---|---|---|---|---|---|---|")
for key, (eng, zr) in REF.items():
    zs, zd = z(chg[(key, "served")]), z(chg[(key, "reduced")])
    g0, g1 = abs(zd - zr), abs(zs - zr)
    print(
        f"| {key} | {eng} | {zr:.6g} | {zd:.4f} | {zs:.4f} | {zs - zd:+.4f} | "
        f"{g0:.4f} | {g1:.4f} | {'toward' if g1 < g0 else 'AWAY'} by {abs(g1 - g0):.4f} |"
    )

print()
print(
    "| deck | wires extended / all | air pairs: handed, extended "
    "| soil pairs: handed, extended | forced reduced: air, soil |"
)
print("|---|---|---|---|---|")
for key in REF:
    k, r = chg[(key, "served")], chg[(key, "reduced")]
    w = k["kernels"][0]["wires_extended"]
    air, soil = k["census"]["air"], k["census"]["soil"]
    print(
        f"| {key} | {sum(w)} / {len(w)} | {air[0]}, {air[1]} | {soil[0]}, {soil[1]} | "
        f"{r['census']['air']}, {r['census']['soil']} |"
    )
