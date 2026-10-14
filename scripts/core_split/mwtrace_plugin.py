"""pytest plugin: hash every native accelerator call's inputs and outputs.

    MWTRACE_OUT=<dir> PYTHONPATH=scripts/core_split \
        python -m pytest tests/ -m "<lane>" -p mwtrace_plugin

Wraps every public callable of `_accel.acc` and the two near-interface
companions, and writes one JSONL file per xdist worker into <dir>: test node
id, call index, function, input hash, output hash. The output hash covers the
results AND every array argument after the call, so in-place writes count;
an exception is recorded by type. cmp_traces.py compares two such runs.
"""

import hashlib
import json
import os

import numpy as np

_state = {"node": None, "idx": 0, "fh": None}


def _feed(h, x, depth=0):
    if isinstance(x, np.ndarray):
        h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode())
        h.update(
            np.ascontiguousarray(x).view(np.uint8).reshape(-1).data if x.size else b""
        )
    elif isinstance(x, (bool, int, float, complex, str, bytes, type(None))):
        h.update(repr(x).encode())
    elif isinstance(x, (tuple, list)) and depth < 3:
        h.update(b"(")
        for e in x:
            _feed(h, e, depth + 1)
        h.update(b")")
    elif isinstance(x, dict) and depth < 3:
        for k in sorted(x, key=repr):
            h.update(repr(k).encode())
            _feed(h, x[k], depth + 1)
    else:
        h.update(type(x).__name__.encode())


def _wrap(name, fn):
    def w(*args, **kw):
        hi = hashlib.sha1()
        _feed(hi, args)
        _feed(hi, kw)
        try:
            res = fn(*args, **kw)
        except Exception as exc:  # noqa: BLE001
            _emit(name, hi.hexdigest(), "EXC:" + type(exc).__name__)
            raise
        ho = hashlib.sha1()
        _feed(ho, res)
        _feed(ho, args)
        _feed(ho, kw)
        _emit(name, hi.hexdigest(), ho.hexdigest())
        return res

    w.__name__ = getattr(fn, "__name__", name)
    w.__doc__ = getattr(fn, "__doc__", None)
    w.__wrapped__ = fn
    return w


def _emit(name, hin, hout):
    fh = _state["fh"]
    if fh is None or _state["node"] is None:
        return
    fh.write(json.dumps([_state["node"], _state["idx"], name, hin, hout]) + "\n")
    _state["idx"] += 1


def pytest_configure(config):
    out = os.environ.get("MWTRACE_OUT")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
    _state["fh"] = open(os.path.join(out, f"{wid}.jsonl"), "w", buffering=1 << 16)
    from momwire import _accel

    mods = [("", _accel.acc)]
    for comp in ("_near_interface_accel", "_near_interface_point_accel"):
        try:
            mods.append((comp + ".", _accel.import_companion(comp)))
        except Exception:  # noqa: BLE001
            pass
    n = 0
    for tag, m in mods:
        if m is None:
            continue
        for name in dir(m):
            if name.startswith("_"):
                continue
            f = getattr(m, name)
            if callable(f) and not isinstance(f, type):
                setattr(m, name, _wrap(tag + name, f))
                n += 1
    _state["n_wrapped"] = n


def pytest_runtest_setup(item):
    _state["node"] = item.nodeid
    _state["idx"] = 0


def pytest_unconfigure(config):
    if _state["fh"] is not None:
        _state["fh"].close()
