"""NEC-4.2 on buried_radius_probe's decks (and the same dipole in free space):
does the lane's own reference see a buried radius-step junction like NEC-5?
Usage: python nec42_buried_step.py"""

from __future__ import annotations

from ref_ladder import NEC5, NEC42, run


def deck(a_l, a_r, n, eng, free):
    z = 0.0 if free else -0.3
    lines = [
        "CM step",
        "CE",
        f"GW 1 {n} -3 0 {z} 0 0 {z} {a_l}",
        f"GW 2 {n} 0 0 {z} 3 0 {z} {a_r}",
    ]
    if free:
        lines += ["GE 0" if eng == "nec42" else "GE 0 0"]
    elif eng == "nec42":
        lines += ["GE -1", "GN 2 0 0 0 13 0.005 NOFILE"]
    else:
        lines += ["GE -1 0", "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE"]
    lines += [
        "EX 0 2 1 0 1 0" if eng == "nec42" else f"EX 0 1 {n} 2 1 0",
        "FR 0 1 0 0 14 0",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


for free in (False, True):
    for n in (20, 40):
        for eng, exe in (("nec42", NEC42), ("nec5", NEC5)):
            zs = {}
            for name, ar in {
                "a1": (1e-3, 1e-3),
                "a05": (5e-4, 5e-4),
                "step": (5e-4, 1e-3),
            }.items():
                (R, X), _ = run(exe, deck(*ar, n, eng, free))
                zs[name] = complex(R, X)
            c = zs["a1"]
            print(
                "free" if free else "buried",
                n,
                eng,
                f"a1 {c:.2f}",
                " ".join(f"{k} {z - c:+.2f}" for k, z in zs.items() if k != "a1"),
                flush=True,
            )
