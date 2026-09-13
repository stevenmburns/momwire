"""Re-solve #1027's banked NEC-5 rows on the binary named by NEC5_EXE.

`rod_rows_1027.jsonl` is #1027's own row file, copied unchanged. Each row is
rebuilt from its parameters, its deck is hashed and compared with the banked
deck sha, and the printed NEC-5 impedance is compared with the banked one.
A row passes only if both are identical.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/nec5_binary_check.py [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import rod_ladder as rl

from antennaknobs.engines.nec5 import NEC5Engine

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exe = os.environ["NEC5_EXE"]
    exe_sha = hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    lines = (HERE / "rod_rows_1027.jsonl").read_text().splitlines()
    rows = [json.loads(ln) for ln in lines if ln.strip()]
    out = []
    for r in rows:
        b = rl.build(
            r["soil"], r["rod_len"], r["rod_depth"], r["nn"], refine=r["refine"]
        )
        g = ("finite", b.design_eps_r, b.design_sigma)
        deck = NEC5Engine(b, ground=g).deck([b.freq])
        deck_sha = hashlib.sha256(deck.encode()).hexdigest()[:12]
        z = complex(NEC5Engine(b, ground=g).impedance()[0])
        rec = dict(
            rod_len=r["rod_len"],
            refine=r["refine"],
            deck_sha=deck_sha,
            deck_same=deck_sha == r["deck_sha"],
            nec5=str(z),
            banked=str(complex(r["nec5_R"], r["nec5_X"])),
            z_same=(z.real == r["nec5_R"] and z.imag == r["nec5_X"]),
        )
        out.append(rec)
        print(
            f"L={r['rod_len']:4.2f} ref={r['refine']} deck "
            f"{'same' if rec['deck_same'] else 'DIFF'}  Z {z:.5g} vs {rec['banked']}  "
            f"{'same' if rec['z_same'] else 'DIFF'}",
            flush=True,
        )
    n_ok = sum(r["deck_same"] and r["z_same"] for r in out)
    print(f"{n_ok}/{len(out)} rows identical on {exe} (sha256 {exe_sha})")
    if args.out is not None:
        args.out.write_text(
            json.dumps(dict(nec5_exe=exe, nec5_sha256=exe_sha, rows=out), indent=1)
        )


if __name__ == "__main__":
    main()
