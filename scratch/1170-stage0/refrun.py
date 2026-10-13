"""momwire#1170 stage 0 reference runs: one deck through a momwire EZNEC slot.

usage: refrun.py <tree> <deck> <nec4|nec5> <basis> <served|reduced>

`served` is the tree as it serves the deck. `reduced` forces the default
extended kernel off for the whole deck (the pre-stage-0 fallback) by making the
front end's refusal question answer "yes" and its partial question "no".
Prints one JSON line: the port rows of the printout, and what every solver the
front end constructed says about its own kernel, per wire.
"""

import json
import os
import re
import sys
import time
import warnings

tree, deck, dialect, basis, mode = sys.argv[1:6]
tree = os.path.abspath(tree)
sys.path.insert(0, os.path.join(tree, "src"))
warnings.simplefilter("ignore")

import momwire  # noqa: E402

assert momwire.__file__.startswith(os.path.join(tree, "src")), momwire.__file__
import momwire._accel as _acc  # noqa: E402
from momwire import bspline  # noqa: E402
from momwire import _bspline_kernels as _k  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.deck import _solver  # noqa: E402
from momwire.eznec import render  # noqa: E402

if mode == "reduced":
    _solver.extended_kernel_default_refusal = lambda *a, **k: "forced reduced (refrun)"
    if hasattr(_solver, "extended_kernel_partial"):
        _solver.extended_kernel_partial = lambda *a, **k: None

built = []
_init = BSplineSolver.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    built.append(self)


BSplineSolver.__init__ = init

# The solver's own port matrix, at full precision (the printout rounds).
ymats = []
_cps = BSplineSolver.compute_port_solution


def cps(self, *a, **k):
    import numpy as np

    sol = _cps(self, *a, **k)
    zm = np.linalg.inv(np.atleast_2d(np.asarray(sol.y, dtype=complex)))
    ymats.append([[float(v.real), float(v.imag)] for v in np.diag(zm)])
    return sol


BSplineSolver.compute_port_solution = cps

# What the moment kernel was handed, per medium: pairs, and pairs extended.
census = {"air": [0, 0], "soil": [0, 0]}
_off = bspline._seg_seg_full_moments_offedge


def off(sl_i, sr_i, sl_j, sr_j, a, k, *args, **kwargs):
    import numpy as np

    n_i, n_j = np.asarray(sl_i).shape[0], np.asarray(sl_j).shape[0]
    ek = kwargs.get("ek")
    row = census["soil" if complex(k).imag != 0.0 else "air"]
    row[0] += n_i * n_j
    row[1] += 0 if ek is None else int(_k._ek_pair_mask(ek, n_i, n_j).sum())
    return _off(sl_i, sr_i, sl_j, sr_j, a, k, *args, **kwargs)


bspline._seg_seg_full_moments_offedge = off

text = open(deck, encoding="latin-1").read()
t0 = time.perf_counter()
printout = render(text, basis=basis, dialect=dialect)
wall = time.perf_counter() - t0

number = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]\d+)?")
lines = printout.splitlines()
ports = []
errors = [ln.strip() for ln in lines if "NEC ERROR" in ln or "ERROR" in ln][:3]
for i, ln in enumerate(lines):
    if "ANTENNA INPUT PARAMETERS" in ln:
        for row in lines[i + 1 : i + 12]:
            c = number.findall(row)
            if len(c) in (11, 12):
                o = len(c) - 5
                ports.append(
                    dict(tag=int(c[0]), seg=int(c[1]), z=[float(c[o]), float(c[o + 1])])
                )
        break

kernels = []
for s in built:
    wires = getattr(s, "extended_kernel_wires", None)
    kernels.append(
        dict(
            extended_kernel=bool(s.extended_kernel),
            wires_extended=None if wires is None else [bool(w) for w in wires()],
            n_segs=int(s._build_geometry()["n_segs_total"]),
        )
    )
print(
    json.dumps(
        dict(
            deck=os.path.basename(deck),
            dialect=dialect,
            basis=basis,
            mode=mode,
            tree=os.path.basename(tree),
            ports=ports,
            errors=errors,
            kernels=kernels,
            z_ports_full=ymats,
            census=census,
            accel=[bool(_acc.LOADED), _acc.VARIANT],
            wall=round(wall, 2),
        )
    )
)
