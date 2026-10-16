"""momwire-razor2p's fallback, instrumented: each run `_serve._finish` finishes
writes its arrays to MW_EXE_PYARRAYS (`<run>_<name>.bin`, complex128 in C
order), as the executable's C++ route writes them to MW_EXE_ARRAYS.

The executable execs this (MW_EXE_PYSTARTUP) after publishing its library
and before answering, so every array here was computed on the executable's
OpenBLAS through `momwire._lapack`; `coeffs` is recomputed exactly as
`_finish` computes it, through the same seam. `exe_compare.py` uses it; it is
never part of the executable.
"""

import json
import os

import numpy as np

from momwire import _lapack
from momwire.eznec import _serve

_DIR = os.environ["MW_EXE_PYARRAYS"]
_orig_state, _orig_finish = _serve._port_state, _serve._finish
_seen = {}
_count = [0]


def _port_state(*args, **kwargs):
    _seen["state"] = state = _orig_state(*args, **kwargs)
    return state


def _finish(deck, prepared):
    data = _orig_finish(deck, prepared)
    st = _seen.pop("state")
    coeffs = _lapack.matmul(
        prepared.solution.coeffs,
        _lapack.matmul(_serve._transform(prepared.mesh), st.v_gap),
    )
    named = {
        "y": prepared.solution.y,
        "coeffs": coeffs,
        "v_applied": st.v_applied,
        "v_gap": st.v_gap,
        "i_port": st.i_port,
        "i_source": st.i_source,
    }
    k = _count[0]
    # Proof the run was on the executable's library: the seam's host calls.
    with open(os.path.join(_DIR, "host_calls.json"), "w") as f:
        f.write(json.dumps(_lapack.HOST_CALLS))
    _count[0] += 1
    for name, value in named.items():
        with open(os.path.join(_DIR, f"{k}_{name}.bin"), "wb") as f:
            f.write(np.ascontiguousarray(value, dtype=np.complex128).tobytes())
    return data


_serve._port_state, _serve._finish = _port_state, _finish
