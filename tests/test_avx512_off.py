"""AVX-512 off: momwire's answers must not depend on whether the CPU has it.

`momwire_serve_client.pin_cpu_env` sets NPY_DISABLE_CPU_FEATURES (numpy's
AVX-512 dispatch targets) and OPENBLAS_CORETYPE=Haswell on an AVX-512 CPU,
before numpy is imported, at every entry point momwire owns: the thin
clients, the frozen engine, the resident pool's workers, the standalone
executable's embedded Python, and this test session (tests/conftest.py).

The session checks below hold on EVERY box: on a CPU without AVX-512 numpy
has no AVX-512 target to run and OpenBLAS no AVX-512 core to pick, and on one
with it the pin has turned both off. So a red here on an AVX-512 box means
something imported numpy before the conftest could pin it.
"""

from __future__ import annotations

import os
import sys

import pytest

import momwire_serve_client as mech

# OpenBLAS's names for its AVX-512 cores, as threadpoolctl reports them.
_AVX512_CORES = {"skylakex", "cooperlake", "sapphirerapids"}


def _ours(name: str) -> bool:
    """Whether the session's value of ``name`` is the pin's, i.e. no user
    exported their own (a user's value is respected, so these checks would
    measure their choice rather than the pin)."""
    want = mech.cpu_env({}, avx512=True)[name]
    return os.environ.get(name, want) == want


def test_numpy_runs_no_avx512_target_in_this_session():
    if not _ours(mech.NPY_DISABLE_ENV) or mech.NPY_ENABLE_ENV in os.environ:
        pytest.skip("NPY_*_CPU_FEATURES exported by the user")
    from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__

    # The AVX-512 targets this numpy dispatches to. A disabled one reads False
    # (the CPU features beneath it, AVX512F and the rest, still read True:
    # numpy only stops dispatching); on a CPU without AVX-512 none was on.
    targets = [n for n in __cpu_dispatch__ if "AVX512" in n or n == "X86_V4"]
    assert targets, __cpu_dispatch__
    on = [n for n in targets if __cpu_features__.get(n)]
    assert on == [], on


def test_openblas_runs_no_avx512_core_in_this_session():
    if not _ours(mech.OPENBLAS_CORETYPE_ENV):
        pytest.skip("OPENBLAS_CORETYPE exported by the user")
    threadpoolctl = pytest.importorskip("threadpoolctl")
    import numpy  # noqa: F401 - loads numpy's OpenBLAS
    import scipy.linalg  # noqa: F401 - loads SciPy's OpenBLAS

    cores = {
        (pool.get("filepath"), str(pool.get("architecture")).lower())
        for pool in threadpoolctl.threadpool_info()
        if pool.get("internal_api") == "openblas"
    }
    assert cores, "no OpenBLAS loaded"
    assert not {c for c in cores if c[1] in _AVX512_CORES}, cores


def test_nothing_is_set_on_a_cpu_without_avx512():
    assert mech.cpu_env({}, avx512=False) == {}
    environ = {"PATH": "/bin"}
    assert mech.cpu_env(environ, avx512=False) == {}


def test_both_are_set_on_an_avx512_cpu():
    added = mech.cpu_env({}, avx512=True)
    assert added[mech.OPENBLAS_CORETYPE_ENV] == "Haswell"
    assert "X86_V4" in added[mech.NPY_DISABLE_ENV].split() or "AVX512F" in (
        added[mech.NPY_DISABLE_ENV].split()
    )


def test_a_user_value_is_never_replaced():
    # An empty NPY_DISABLE_CPU_FEATURES and a named core are how a user asks
    # for the AVX-512 paths back.
    environ = {mech.NPY_DISABLE_ENV: "", mech.OPENBLAS_CORETYPE_ENV: "SkylakeX"}
    assert mech.cpu_env(environ, avx512=True) == {}
    # numpy refuses to start with both DISABLE and ENABLE set.
    added = mech.cpu_env({mech.NPY_ENABLE_ENV: "X86_V3"}, avx512=True)
    assert mech.NPY_DISABLE_ENV not in added


def test_pin_updates_the_environment_it_is_given(monkeypatch):
    monkeypatch.setattr(mech, "cpu_has_avx512", lambda: True)
    environ = {mech.OPENBLAS_CORETYPE_ENV: "Zen"}
    added = mech.pin_cpu_env(environ)
    assert list(added) == [mech.NPY_DISABLE_ENV]
    assert environ == {mech.OPENBLAS_CORETYPE_ENV: "Zen", **added}


@pytest.mark.parametrize(
    "version, names",
    [
        ("2.5.4", "X86_V4 AVX512_ICL AVX512_SPR"),
        ("2.4.0", "X86_V4 AVX512_ICL AVX512_SPR"),
        ("2.4.0rc1", "X86_V4 AVX512_ICL AVX512_SPR"),
        ("3.0.0.dev0+git1", "X86_V4 AVX512_ICL AVX512_SPR"),
        (
            "2.3.5",
            "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX "
            "AVX512_CNL AVX512_ICL AVX512_SPR",
        ),
        (
            "2.2.6",
            "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX "
            "AVX512_CNL AVX512_ICL",
        ),
        (
            "1.26.4",
            "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX "
            "AVX512_CNL AVX512_ICL",
        ),
    ],
)
def test_numpy_feature_names_follow_the_release(version, names):
    # The dispatch lists measured from numpy 1.26.4, 2.0.2, 2.2.6, 2.3.5,
    # 2.4.0 and 2.5.4's own __cpu_dispatch__: a name a release does not
    # dispatch would earn an ImportWarning.
    assert mech.numpy_avx512_features(version) == names


def test_an_unknown_numpy_gets_both_vocabularies():
    names = mech.numpy_avx512_features(None).split()
    assert len(names) == len(set(names))
    assert {"X86_V4", "AVX512F", "AVX512_SKX", "AVX512_ICL"} <= set(names)


def test_this_numpy_dispatches_every_name_the_pin_gives_it():
    """No ImportWarning from the installed numpy: every name the pin would
    hand it is one of its dispatch targets."""
    import numpy

    try:
        from numpy._core._multiarray_umath import __cpu_dispatch__
    except ImportError:  # numpy 1.x
        from numpy.core._multiarray_umath import __cpu_dispatch__
    names = mech.numpy_avx512_features(numpy.__version__).split()
    assert set(names) <= set(__cpu_dispatch__), (names, __cpu_dispatch__)


def test_the_pool_hands_its_workers_the_pin(monkeypatch):
    from momwire.serve import _pool

    seen = {}

    class _Stop(Exception):
        pass

    def fake_popen(command, **kwargs):
        seen["env"] = kwargs.get("env")
        raise _Stop

    monkeypatch.setattr(mech, "cpu_has_avx512", lambda: True)
    monkeypatch.delenv(mech.OPENBLAS_CORETYPE_ENV, raising=False)
    monkeypatch.setattr(_pool.subprocess, "Popen", fake_popen)

    class _Log:
        def write(self, _text):
            pass

        def flush(self):
            pass

    pool = _pool.WorkerPool(
        {"factory": "x:y", "kwargs": {}},
        _Log(),
        max_workers=1,
        idle_timeout=1.0,
        cores=1,
        command=[sys.executable, "-c", "pass"],
        env={"MOMWIRE_PROBE": "1"},
    )
    pool._spawn()
    assert seen["env"]["MOMWIRE_PROBE"] == "1"
    assert seen["env"][mech.OPENBLAS_CORETYPE_ENV] == "Haswell"


def test_the_eznec_client_pins_before_anything_else(monkeypatch):
    import momwire_eznec_client

    calls = []
    monkeypatch.setattr(mech, "pin_cpu_env", lambda *a: calls.append(a) or {})
    assert momwire_eznec_client.main(["-version"]) == 0
    assert calls == [()]
