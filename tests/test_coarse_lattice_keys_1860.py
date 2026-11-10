"""The above-ground Sommerfeld masters on coarser lattice keys (momwire#1860,
the opt-in feature `coarse_lattice_keys`).

Im(eps_t) goes as 1/f, so a 0.75..1.25 f0 sweep crosses about five rungs of
the 1.1 ladder (`_sommerfeld._SOMM_EPS_IM_BUCKET`), and a pair's two masters
are keyed to the lattice of its demanding rung: each crossing stores the
shared rung under a new lattice and rebuilds both masters. On, the lattice is
that rung rounded up to a multiple of `_SOMM_LATTICE_COARSE_RUNGS`
(`_sommerfeld._coarse_lattice`), so the pairs between two multiples share a
lattice and a crossing reuses a master. The two-rung blend is unchanged.

What is held:

- off by default, the keyword wins over the variable, and off is the
  shipped keying (the lattice is the pair's demanding rung, the keys untagged);
- the coarse lattice is a multiple of the coarse step, never below the pair's
  own demanding rung, and an off-ladder value is its own lattice;
- on and off never share a master or a view, in process or on disk;
- over a 30-point sweep's eps_t the masters on are far fewer than off, each
  rung on one lattice per coarse step it touches;
- a swept deck solved on builds fewer masters than off, with the printed
  input impedance at every frequency within `ZIN_RTOL` (nec5-razor-2p,
  nec5-bspline, nec42-bspline, nec42-sinusoidal).
"""

from __future__ import annotations

import inspect
import math
import re

import numpy as np
import pytest

from momwire import _features, _somm_disk_cache
from momwire import _sommerfeld as sm
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver

from test_crossing_serve_524 import crossing_deck

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

ENV = "MOMWIRE_X_COARSE_LATTICE_KEYS"

# The stated tolerance: |Zin(on) - Zin(off)| / |Zin(off)|, the registry's
# `drift`. Measured on Skylake (the PR's table) over 30-point sweeps of
# efhw_lo, efhw_hi, a buried hub, a rod Beverage and the ACA screen, all four
# engines.
ZIN_RTOL = _features.FEATURES["coarse_lattice_keys"].drift

_EPS0 = 8.8541878128e-12
_C0 = 299792458.0


def _eps(er, sigma, f):
    return complex(er, -sigma / (2.0 * math.pi * f * _EPS0))


def _clear():
    sm._GRID_CACHE.clear()
    sm._NORM_CACHE.clear()


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    _clear()
    yield
    _clear()


def test_registry_entry():
    f = _features.FEATURES["coarse_lattice_keys"]
    assert (f.reach, f.kind, f.env, f.kwarg, f.issue) == (
        1,
        "python",
        ENV,
        "coarse_lattice_keys",
        1860,
    )
    assert "tests/test_coarse_lattice_keys_1860.py" in f.paths
    assert f.offer and f.label and f.summary


def test_off_by_default_and_the_keyword_wins(monkeypatch):
    assert not _features.enabled("coarse_lattice_keys")
    assert not BSplineSolver(**crossing_deck())._coarse_lattice_keys
    monkeypatch.setenv(ENV, "1")
    assert BSplineSolver(**crossing_deck())._coarse_lattice_keys
    assert not BSplineSolver(
        **crossing_deck(), coarse_lattice_keys=False
    )._coarse_lattice_keys
    # Every solver that reaches the above-ground grid takes the keyword.
    for cls in (BSplineSolver, RazorSolver, SinusoidalSolver):
        assert "coarse_lattice_keys" in inspect.signature(cls.__init__).parameters


def _rung(n):
    return complex(13.0, -((1.0 + sm._SOMM_EPS_IM_BUCKET) ** n))


@pytest.mark.parametrize("n", [-7, -4, -1, 0, 1, 3, 4, 5, 17, 20, 23])
def test_coarse_lattice_rounds_up_to_the_coarse_step(n):
    every = sm._SOMM_LATTICE_COARSE_RUNGS
    lat = sm._coarse_lattice(_rung(n))
    m = round(math.log(-lat.imag, 1.0 + sm._SOMM_EPS_IM_BUCKET))
    assert lat.real == 13.0
    assert m % every == 0 and n <= m < n + every
    if n % every == 0:
        assert lat == _rung(n)


def test_off_ladder_values_are_their_own_lattice():
    for e in (complex(13.0, 0.0), complex(13.0, 0.5), complex(-1.0, -3.0)):
        assert sm._coarse_lattice(e) == e


def _sweep(npts=30):
    f0 = 3.6e6
    return [f0 * (0.75 + 0.5 * i / (npts - 1)) for i in range(npts)]


def _views(on, freqs, sigma=0.005):
    for f in freqs:
        k2 = 2.0 * math.pi * f / _C0
        sm.get_grid(
            _eps(13.0, sigma, f),
            k2,
            0.35 * 2.0 * math.pi / k2,
            2.0 * math.pi * f,
            lazy=True,
            coarse_keys=on,
        )


def test_off_is_the_shipped_keying():
    _views(False, _sweep())
    for nkey in sm._NORM_CACHE:
        assert len(nkey) == 4 and nkey[0] == "above"
        rung, lattice = nkey[1], nkey[3]
        # A pair's lattice is its demanding rung: the rung itself or the
        # next one up.
        step = 1.0 + sm._SOMM_EPS_IM_BUCKET
        assert lattice.real == rung.real
        assert lattice == rung or abs(lattice.imag / rung.imag - step) < 1e-12
    for key in sm._GRID_CACHE:
        assert sm._COARSE_TAG not in key


def test_on_and_off_never_share_a_master_or_a_view(monkeypatch, tmp_path):
    f = 3.6e6
    _views(False, [f])
    off_m, off_v = set(sm._NORM_CACHE), set(sm._GRID_CACHE)
    _views(True, [f])
    on_m = set(sm._NORM_CACHE) - off_m
    on_v = set(sm._GRID_CACHE) - off_v
    assert on_m and on_v
    assert all(k[-1] == sm._COARSE_TAG for k in on_m | on_v)
    assert not any(sm._COARSE_TAG in k for k in off_m | off_v)
    # The disk store names a file by the key: the two never meet there.
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "1")
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE_DIR", str(tmp_path))
    names_off = {_somm_disk_cache._identity(k, ())[0] for k in off_m}
    names_on = {_somm_disk_cache._identity(k, ())[0] for k in on_m}
    assert names_off and names_on and not names_off & names_on


def test_a_coarse_lattice_equal_to_the_shipped_one_is_still_its_own_master():
    # Im(eps_t) on a rung that is a multiple of the coarse step: on and off
    # read the same rung on the same lattice, and still not the same master.
    every = sm._SOMM_LATTICE_COARSE_RUNGS
    e = _rung(8 * every)
    f = 3.6e6
    k2 = 2.0 * math.pi * f / _C0
    args = (e, k2, 0.35 * 2.0 * math.pi / k2, 2.0 * math.pi * f)
    g_off = sm.get_grid(*args, lazy=True, coarse_keys=False)
    g_on = sm.get_grid(*args, lazy=True, coarse_keys=True)
    assert g_on is not g_off
    keys = set(sm._NORM_CACHE)
    assert len(keys) == 2 and {(k[1], k[3]) for k in keys} == {(e, e)}
    off = [k for k in keys if len(k) == 4]
    on = [k for k in keys if len(k) == 5]
    assert len(off) == len(on) == 1 and off[0] + (sm._COARSE_TAG,) == on[0]


def test_a_sweep_builds_far_fewer_masters_on():
    _views(False, _sweep())
    off = set(sm._NORM_CACHE)
    _clear()
    _views(True, _sweep())
    on = set(sm._NORM_CACHE)
    rungs = {k[1] for k in on}
    assert {k[1] for k in off} == rungs
    # Off: the shared rung of every crossing is stored twice.
    assert len(off) >= 2 * len(rungs) - 2
    # On: one master per rung per coarse lattice it is read on.
    lattices = {k[3] for k in on}
    assert len(on) <= len(rungs) + len(lattices) - 1
    assert len(on) < len(off)


# A 40 m horizontal dipole 6 m over a lossy ground, swept 3.0..4.5 MHz: the
# eps_t of 13 S, 5 mS crosses four rungs, and every master is above-ground.
DIPOLE = """CM dipole over Sommerfeld ground, swept
CE
GW 1 21 -20 0 6 20 0 6 0.001
GE 1
GN 2 0 0 0 13 0.005
EX 0 1 11 0 1 0
FR 0 7 0 0 3.0 0.25
XQ
EN
"""

_FLOAT = re.compile(r"[-+]?\d\.\d+E[-+]\d+")


def _zins(text):
    """Every frequency's first input impedance, from either dialect's table."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if "INPUT PARAMETERS" in line:
            for r in lines[i + 1 : i + 8]:
                f = _FLOAT.findall(r)
                if len(f) >= 6:
                    out.append(complex(float(f[4]), float(f[5])))
                    break
    assert out, "no input impedance in the printout"
    return out


def _render(deck, basis, dialect):
    from momwire.dropin.nec5 import _shell

    # Level 1, as antennaknobs runs it: the shell's render without its
    # drop-in context.
    return _shell.render.__wrapped__(deck, basis=basis, dialect=dialect)


@pytest.fixture
def built(monkeypatch):
    """The masters that filled nodes, by (rung, lattice)."""
    seen = set()
    real = sm.SommerfeldGrid.fill_nodes

    def fill_nodes(self, *a, **kw):
        seen.add((self.eps_t, self.lattice_eps))
        return real(self, *a, **kw)

    monkeypatch.setattr(sm.SommerfeldGrid, "fill_nodes", fill_nodes)
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "0")
    return seen


@pytest.mark.parametrize(
    "basis,dialect",
    [
        # ~20-35 s each on Skylake: the masters are filled whole.
        pytest.param("razor-2p", "nec5", marks=pytest.mark.slow),
        pytest.param("bspline", "nec5", marks=pytest.mark.slow),
        pytest.param("bspline", "nec4", marks=pytest.mark.slow),
        pytest.param("sinusoidal", "nec4", marks=pytest.mark.slow),
    ],
)
def test_a_swept_deck_reuses_masters_and_zin_agrees(built, monkeypatch, basis, dialect):
    z_off = _zins(_render(DIPOLE, basis, dialect))
    n_off = len(built)
    built.clear()
    _clear()
    monkeypatch.setenv(ENV, "1")
    z_on = _zins(_render(DIPOLE, basis, dialect))
    n_on = len(built)
    assert len(z_on) == len(z_off) == 7
    assert 0 < n_on < n_off, (n_on, n_off)
    for a, b in zip(z_on, z_off, strict=True):
        assert abs(a - b) <= ZIN_RTOL * abs(b), (basis, dialect, a, b)
    assert np.isfinite(z_on).all()
