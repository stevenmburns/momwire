"""Regenerate `src/momwire/_leggauss_table.py` from numpy's `leggauss`.

The table freezes the Gauss-Legendre rules momwire asks for, so the solver's
quadrature no longer depends on the host's LAPACK eigensolver (see
`momwire._quadrature`). It must be generated on the REFERENCE platform —
Linux x86_64 (Skylake), the numpy pinned in the header — so that Linux
results do not move; on that platform the table is bit-identical to
`numpy.polynomial.legendre.leggauss`, which `tests/test_leggauss_table.py`
asserts.

Usage (momwire venv, Linux x86_64):

    python scripts/gen_leggauss_table.py [--max N]

Each rule is stored as its non-negative half (numpy symmetrises nodes and
weights exactly, so the other half is the mirror image) in C99 hex-float
spelling, which round-trips through `float.fromhex` to the bit.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "momwire" / "_leggauss_table.py"
DEFAULT_MAX = 192

HEADER = '''"""Frozen Gauss-Legendre rules, orders 1..{n_max}. GENERATED — do not edit.

Written by ``scripts/gen_leggauss_table.py`` from
``numpy.polynomial.legendre.leggauss`` (numpy {numpy}, {machine} {system}).
On that platform every entry is bit-identical to numpy's own rule; see
`momwire._quadrature` for why the solver reads this instead.

``HALVES[n] = (nodes, weights)``: the non-negative half of the rule, nodes
ascending (the middle node 0 included when ``n`` is odd), each a string of
space-separated hex floats (`float.fromhex`, or C99 ``%a``, to the bit).
"""

N_MAX = {n_max}
NUMPY_VERSION = "{numpy}"

HALVES = {{
'''


def _from_half(n, xs, ws):
    # `momwire._quadrature._from_half`, repeated here because importing
    # momwire needs the table this script writes (module-level rules).
    xh = np.array([float.fromhex(v) for v in xs.split()], dtype=np.float64)
    wh = np.array([float.fromhex(v) for v in ws.split()], dtype=np.float64)
    m = n // 2
    return np.concatenate([-xh[::-1][:m], xh]), np.concatenate([wh[::-1][:m], wh])


def half(n):
    x, w = np.polynomial.legendre.leggauss(n)
    m = n // 2
    return x[m:], w[m:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max", type=int, default=DEFAULT_MAX)
    args = ap.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        sys.exit("generate on the reference platform (Linux x86_64) only")
    lines = [
        HEADER.format(
            n_max=args.max,
            numpy=np.__version__,
            machine=platform.machine(),
            system=platform.system(),
        )
    ]
    for n in range(1, args.max + 1):
        xh, wh = half(n)
        x, w = np.polynomial.legendre.leggauss(n)
        xs = " ".join(float(v).hex() for v in xh)
        ws = " ".join(float(v).hex() for v in wh)
        rx, rw = _from_half(n, xs, ws)
        if rx.tobytes() != x.tobytes() or rw.tobytes() != w.tobytes():
            sys.exit(f"order {n}: numpy's rule is not exactly symmetric")
        lines.append(f'    {n}: (\n        "{xs}",\n        "{ws}",\n    ),\n')
    lines.append("}\n")
    OUT.write_text("".join(lines))
    print(f"wrote {OUT} (orders 1..{args.max})")


if __name__ == "__main__":
    main()
