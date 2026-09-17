"""Which machine produced a record (momwire#1029 phase 1).

The ladder's bars are #1067's, and #1067 was measured on the Skylake box — so
a G3 row that does not say which box it came from cannot be compared with
them. Every record this arc writes carries this dict, read from the process
rather than typed in, because a hand-written box label survives being copied
to a different one and a read one does not.
"""

from __future__ import annotations

import os
import platform
import resource
import subprocess


def _threads():
    """What the numeric stack was actually told, not what the box has."""
    return {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _mem_total_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def provenance():
    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    return {
        "hostname": platform.node(),
        "cpu": _cpu_model(),
        "nproc": os.cpu_count(),
        "mem_total_gb": _mem_total_gb(),
        "as_cap_gb": None if soft == resource.RLIM_INFINITY else round(soft / 2**30, 1),
        "threads": _threads(),
        "python": platform.python_version(),
        "numpy": _numpy_version(),
        "blas": _blas(),
    }


def _numpy_version():
    try:
        import numpy

        return numpy.__version__
    except ImportError:
        return None


def _blas():
    """The BLAS numpy is linked against — a timing row means little without
    it, and the two boxes need not agree."""
    try:
        import numpy

        cfg = numpy.show_config(mode="dicts")
        return cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except (ImportError, TypeError, AttributeError, KeyError):
        return None


def accel_variant():
    """Which accelerator the momwire on PYTHONPATH resolved — a timing row
    taken against the baseline build is not the same measurement as one taken
    against AVX2, and the two boxes need not pick the same one."""
    try:
        from momwire import _accel

        mod = getattr(_accel, "acc", None)
        return None if mod is None else getattr(mod, "__name__", str(mod))
    except Exception:  # noqa: BLE001 — provenance must never fail a gate
        return None


def git_head(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
