"""Persistent (cross-process) store for the Sommerfeld tabulations (momwire#1224).

The in-process caches in `_sommerfeld` (`_NORM_CACHE` / `_GRID_CACHE`) make a
repeat solve free, but every FRESH process pays the fill again: ~0.68 s per
buried lane on the 10-01 profile, against NEC-5's whole 0.84 s run. NEC-5
itself persists its Sommerfeld table beside the deck (SOMMPD.NEX), so a disk
level here is parity, not a trick. It sits BELOW the in-process caches, which
are unchanged: a process asks disk only on an in-process miss, and keeps what
disk gave it in memory exactly as if it had filled it.

What is persisted, per family:

* above/above -- the frequency-independent normalized masters (`_norm_master`),
  the expensive level of the two; the views over them stay in-process.
* below/below -- the physical grid (`get_grid_below`; that family has no
  normalized master). Its band regions fill LAZILY, so a grid is re-written
  after a lazy region fill (`note_mutated`) and the next process starts with
  every region the last one reached.
* below->above (transmitted) -- the physical grid (`get_grid_below_above`).

Contract:

* **A hit is bit-identical to a fresh fill.** Arrays go to `.npz` exactly (no
  lossy format); scalars are written as `float.hex`, numpy scalars keep their
  dtype. A load checks the manifest's identity (format, code fingerprint,
  state tag, exact key) and every array's shape, dtype and sha256. Anything
  short of all of that is a miss, and the refill overwrites the bad file.
* **Keyed by everything that sets the bytes.** The exact in-process key, plus
  a CODE fingerprint (sha256 of the fill modules' source and of the loaded
  C++ accelerator binary, with the numpy / scipy / Python versions), plus a
  STATE tag read at call time (every upper-case module constant of the fill
  modules -- several are env-overridable and tests monkeypatch them -- and the
  accelerator routing flags). A hashed fingerprint rather than a hand-bumped
  version, because momwire's submodule pointer runs ahead of its PyPI release
  with the SAME version string (see `momwire/__init__.py`), and a constant
  tuned without a bump would otherwise serve a stale table silently. A
  comment-only edit costs one refill; that is the price, and it is cheap.
* **Atomic and safe under concurrent writers.** Write to a unique temp file in
  the cache dir, then `os.replace`. Two writers on one key both produce valid
  files (the fills are deterministic), and the last rename wins. A reader
  reads the whole file into memory first, so it never sees a partial write.
* **Never an error.** Any failure on the disk path is a miss or a skipped
  write. An unwritable directory logs ONE warning per process and the process
  runs on the in-process caches alone, exactly as before this module.

Location: `MOMWIRE_SOMM_CACHE_DIR`, else the platform cache dir --
`%LOCALAPPDATA%\\momwire\\Cache\\sommerfeld` on Windows,
`~/Library/Caches/momwire/sommerfeld` on macOS,
`$XDG_CACHE_HOME/momwire/sommerfeld` (default `~/.cache/...`) elsewhere.
`MOMWIRE_SOMM_CACHE=0` (or off/false/no) disables it.

Eviction: least-recently-USED by mtime (a hit refreshes the file's mtime),
run after every write, until the directory's `.npz` total is under
`MOMWIRE_SOMM_CACHE_MAX_MB` (default 512). The file just written is never
evicted by its own write. Orphaned temp files older than an hour are removed
on the same pass. Files keyed by an older fingerprint are never read again
and age out the same way.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import time
import uuid
import weakref
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1

ENV_DIR = "MOMWIRE_SOMM_CACHE_DIR"
ENV_ENABLE = "MOMWIRE_SOMM_CACHE"
ENV_MAX_MB = "MOMWIRE_SOMM_CACHE_MAX_MB"
DEFAULT_MAX_MB = 512.0
_TMP_STALE_S = 3600.0
_OFF = ("0", "off", "false", "no")
_MANIFEST = "__manifest__"

_log = logging.getLogger("momwire.sommerfeld_cache")

# One warning per process per directory that refused a write.
_WRITE_WARNED: set = set()
# Grids that mutate after the fill (the below family's lazy regions): which
# file they belong to, so `note_mutated` can re-write it.
_TRACKED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_CODE_FP: list = []  # [fingerprint or None], computed once per process
# Hit / miss / write counters, for tests and benchmarks (never for control).
STATS = {"hits": 0, "misses": 0, "writes": 0, "rejected": 0}


class _Unpersistable(TypeError):
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """False when `MOMWIRE_SOMM_CACHE` is 0/off/false/no. Read per call."""
    return os.environ.get(ENV_ENABLE, "1").strip().lower() not in _OFF


def cache_dir() -> Path:
    """The directory the store reads and writes (not created here)."""
    override = os.environ.get(ENV_DIR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "momwire" / "Cache" / "sommerfeld"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "momwire" / "sommerfeld"
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "momwire" / "sommerfeld"


def max_bytes() -> int:
    try:
        mb = float(os.environ.get(ENV_MAX_MB) or DEFAULT_MAX_MB)
    except ValueError:
        mb = DEFAULT_MAX_MB
    return int(max(mb, 0.0) * (1 << 20))


# ---------------------------------------------------------------------------
# Identity: code fingerprint, call-time state tag, key
# ---------------------------------------------------------------------------


def _fill_modules():
    from . import (
        _accel,
        _constants,
        _sommerfeld,
        _sommerfeld_below,
        _sommerfeld_transmitted,
    )

    return (
        _sommerfeld,
        _sommerfeld_below,
        _sommerfeld_transmitted,
        _constants,
        _accel,
        sys.modules[__name__],
    )


def _hash_file(h, path) -> None:
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)


def _compute_code_fingerprint():
    import scipy

    h = hashlib.sha256()
    h.update(
        repr(
            (
                FORMAT_VERSION,
                sys.implementation.name,
                tuple(sys.version_info[:3]),
                np.__version__,
                scipy.__version__,
                sys.byteorder,
            )
        ).encode()
    )
    frozen = bool(getattr(sys, "frozen", False))
    for mod in _fill_modules():
        h.update(mod.__name__.encode())
        try:
            _hash_file(h, mod.__file__)
        except (OSError, TypeError):
            # A frozen bundle carries bytecode, not source. The executable
            # itself is then the code identity: a new build is a new file.
            if not frozen:
                return None
            st = os.stat(sys.executable)
            h.update(repr((st.st_size, st.st_mtime_ns)).encode())
    from ._accel import acc

    so = getattr(acc, "__file__", None) if acc is not None else None
    h.update(repr(so is not None).encode())
    if so is not None:
        _hash_file(h, so)
    return h.hexdigest()


def code_fingerprint():
    """sha256 over the fill's source and binary, once per process; None when
    it cannot be established (the store is then bypassed)."""
    if not _CODE_FP:
        try:
            _CODE_FP.append(_compute_code_fingerprint())
        except Exception:  # noqa: BLE001 — identity bookkeeping never fails a solve
            _CODE_FP.append(None)
    return _CODE_FP[0]


def _const_repr(v):
    if isinstance(v, np.ndarray):
        return ("nd", v.dtype.str, v.shape, hashlib.sha256(v.tobytes()).hexdigest())
    if isinstance(v, (tuple, list)):
        return tuple(_const_repr(x) for x in v)
    if isinstance(v, (set, frozenset)):
        return ("set", tuple(sorted(repr(_const_repr(x)) for x in v)))
    if isinstance(v, (bool, int, float, complex, str, np.generic)) or v is None:
        return repr(v)
    raise _Unpersistable


def state_tag() -> str:
    """The fill-relevant module state, read NOW: every upper-case constant of
    the fill modules (env overrides and monkeypatches included) and the
    accelerator routing each family will take."""
    from . import _sommerfeld as s
    from . import _sommerfeld_below as b
    from . import _sommerfeld_transmitted as t

    parts = []
    for mod in (s, b, t):
        for name, v in sorted(vars(mod).items()):
            bare = name.lstrip("_")
            if not bare or bare != bare.upper() or not bare[0].isalpha():
                continue
            try:
                parts.append((mod.__name__, name, _const_repr(v)))
            except _Unpersistable:
                continue
        parts.append((mod.__name__, "_acc", mod._acc is None))
    parts.append(("routing", b._use_below_accel(), t._use_transmitted_accel()))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _identity(key, extra):
    fp = code_fingerprint()
    if fp is None:
        return None
    ident = {
        "format": FORMAT_VERSION,
        "code": fp,
        "state": state_tag(),
        "key": repr((key, extra)),
    }
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    family = "".join(c if c.isalnum() else "_" for c in str(key[0]))
    return f"{family}-{digest[:40]}.npz", ident


# ---------------------------------------------------------------------------
# Exact (de)serialization of a grid's __dict__
# ---------------------------------------------------------------------------


def _enc(v, arrays):
    if v is None:
        return None
    t = type(v)
    if t is bool:
        return {"b": v}
    if t is int:
        return {"i": v}
    if t is float:
        return {"f": v.hex()}
    if t is complex:
        return {"c": [v.real.hex(), v.imag.hex()]}
    if t is str:
        return {"s": v}
    if isinstance(v, (np.ndarray, np.generic)):
        arr = np.asarray(v)
        if arr.dtype.hasobject:
            raise _Unpersistable(arr.dtype)
        name = f"a{len(arrays)}"
        arrays[name] = arr
        return {"g" if isinstance(v, np.generic) else "a": name}
    if t is tuple:
        return {"t": [_enc(x, arrays) for x in v]}
    if t is list:
        return {"l": [_enc(x, arrays) for x in v]}
    if t is dict and all(type(k) is str for k in v):
        # Sorted, so array numbering is canonical: a loaded grid's __dict__
        # comes back in sorted order, a fresh one in insertion order.
        return {"d": {k: _enc(v[k], arrays) for k in sorted(v)}}
    raise _Unpersistable(t)


def _dec(e, arrays):
    if e is None:
        return None
    ((tag, v),) = e.items()
    if tag == "b":
        return bool(v)
    if tag == "i":
        return int(v)
    if tag == "f":
        return float.fromhex(v)
    if tag == "c":
        return complex(float.fromhex(v[0]), float.fromhex(v[1]))
    if tag == "s":
        return str(v)
    if tag == "a":
        return arrays[v]
    if tag == "g":
        return arrays[v][()]
    if tag == "t":
        return tuple(_dec(x, arrays) for x in v)
    if tag == "l":
        return [_dec(x, arrays) for x in v]
    if tag == "d":
        return {k: _dec(x, arrays) for k, x in v.items()}
    raise ValueError(f"unknown tag {tag!r}")


def _classes():
    from ._sommerfeld import SommerfeldGrid
    from ._sommerfeld_below import SommerfeldGridBelow
    from ._sommerfeld_transmitted import TransmittedGrid

    return {
        c.__module__ + ":" + c.__qualname__: c
        for c in (SommerfeldGrid, SommerfeldGridBelow, TransmittedGrid)
    }


def _array_digest(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def grid_digest(obj) -> str:
    """sha256 over a grid's whole state -- every attribute, its exact type
    and every array's dtype, shape and bytes. Two grids with equal digests
    are interchangeable bit for bit; the bit-identity gates compare these."""
    arrays: dict = {}
    attrs = _enc(dict(vars(obj)), arrays)
    h = hashlib.sha256(type(obj).__qualname__.encode())
    h.update(json.dumps(attrs, sort_keys=True).encode())
    for name in sorted(arrays, key=lambda n: int(n[1:])):
        a = arrays[name]
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _serialize(obj, ident) -> bytes:
    cls = type(obj)
    cname = cls.__module__ + ":" + cls.__qualname__
    if cname not in _classes():
        raise _Unpersistable(cls)
    arrays: dict = {}
    attrs = _enc(dict(vars(obj)), arrays)
    manifest = dict(ident)
    manifest["class"] = cname
    manifest["attrs"] = attrs
    manifest["arrays"] = {
        name: [a.dtype.str, list(a.shape), _array_digest(a)]
        for name, a in arrays.items()
    }
    buf = io.BytesIO()
    payload = {name: a for name, a in arrays.items()}
    payload[_MANIFEST] = np.frombuffer(
        json.dumps(manifest, sort_keys=True).encode(), dtype=np.uint8
    )
    np.savez(buf, **payload)
    return buf.getvalue()


def _deserialize(data: bytes, ident):
    """The grid in `data`, or None if anything at all fails to verify."""
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        manifest = json.loads(bytes(npz[_MANIFEST]).decode())
        for field in ("format", "code", "state", "key"):
            if manifest.get(field) != ident[field]:
                return None
        cls = _classes().get(manifest.get("class"))
        if cls is None:
            return None
        spec = manifest["arrays"]
        if set(spec) != set(npz.files) - {_MANIFEST}:
            return None
        arrays = {}
        for name, (dtype, shape, digest) in spec.items():
            arr = npz[name]
            if arr.dtype.str != dtype or list(arr.shape) != shape:
                return None
            if _array_digest(arr) != digest:
                return None
            if not arr.flags.writeable:
                arr = arr.copy()
            arrays[name] = arr
    attrs = _dec(manifest["attrs"], arrays)
    if type(attrs) is not dict:
        return None
    obj = object.__new__(cls)
    obj.__dict__.update(attrs)
    return obj


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def _warn_write(d: Path, exc: BaseException) -> None:
    if str(d) in _WRITE_WARNED:
        return
    _WRITE_WARNED.add(str(d))
    _log.warning(
        "momwire: Sommerfeld disk cache at %s is not writable (%s); "
        "grids stay cached in this process only. Set %s to move it or "
        "%s=0 to turn it off.",
        d,
        exc,
        ENV_DIR,
        ENV_ENABLE,
    )


def _load(path: Path, ident):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    try:
        obj = _deserialize(data, ident)
    except Exception:  # noqa: BLE001 — corrupt/truncated/foreign: a miss, never served
        obj = None
    if obj is None:
        STATS["rejected"] += 1
        _log.debug("momwire: ignoring unusable Sommerfeld cache file %s", path)
        return None
    try:
        os.utime(path)  # the LRU clock
    except OSError:
        pass
    return obj


def _store(path: Path, ident, obj) -> None:
    d = path.parent
    if str(d) in _WRITE_WARNED:
        return  # refused once already: this process stays in-memory only
    try:
        data = _serialize(obj, ident)
    except _Unpersistable as exc:
        _log.debug("momwire: grid not persistable (%s); kept in memory", exc)
        return
    tmp = d / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        d.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        # A replace refused because a reader holds the target open (Windows)
        # is a lost race, not an unwritable directory.
        if not path.exists():
            _warn_write(d, exc)
        return
    STATS["writes"] += 1
    _evict(d, keep=path)


def _evict(d: Path, keep: Path) -> None:
    limit = max_bytes()
    now = time.time()
    files = []
    try:
        entries = list(os.scandir(d))
    except OSError:
        return
    for ent in entries:
        try:
            st = ent.stat()
        except OSError:
            continue
        if ent.name.endswith(".tmp"):
            if now - st.st_mtime > _TMP_STALE_S:
                try:
                    os.unlink(ent.path)
                except OSError:
                    pass
            continue
        if ent.name.endswith(".npz"):
            files.append((st.st_mtime, st.st_size, ent.path))
    total = sum(f[1] for f in files)
    for _mtime, size, p in sorted(files):
        if total <= limit:
            break
        if os.path.abspath(p) == os.path.abspath(keep):
            continue
        try:
            os.unlink(p)
        except OSError:
            continue
        total -= size


# ---------------------------------------------------------------------------
# The entry points the grid getters call
# ---------------------------------------------------------------------------


def fetch_or_fill(key, fill, *, extra=(), track=False):
    """The grid for `key` from disk, else `fill()` -- written back to disk.

    `key` is the getter's exact in-process key (its first element the regime
    tag); `extra` carries fill arguments the in-process key leaves implicit.
    `track` registers the grid for `note_mutated` (lazy region fills).
    Exceptions from `fill` propagate unchanged and nothing is written.
    """
    if not enabled():
        return fill()
    try:
        named = _identity(key, extra)
    except Exception:  # noqa: BLE001 — no identity means no disk, never an error
        named = None
    if named is None:
        return fill()
    name, ident = named
    path = cache_dir() / name
    obj = _load(path, ident)
    if obj is not None:
        STATS["hits"] += 1
    else:
        STATS["misses"] += 1
        obj = fill()
        try:
            _store(path, ident, obj)
        except Exception:  # noqa: BLE001 — the disk is an optimization; the fill stands
            pass
    if track:
        try:
            _TRACKED[obj] = (path, ident)
        except TypeError:
            pass
    return obj


def note_mutated(obj) -> None:
    """Re-write a tracked grid after it filled more of itself."""
    entry = _TRACKED.get(obj)
    if entry is None or not enabled():
        return
    try:
        _store(entry[0], entry[1], obj)
    except Exception:  # noqa: BLE001 — a lost re-write costs one lazy refill later
        pass
