"""The deck front ends' bs2 asks for the exact ring kernel, "auto" (momwire#1428).

antennaknobs passes ``exact_kernel="auto"`` to bs2 by default (its #1965);
the NEC-5 seam SimNEC and EZNEC drive (``momwire-nec5-bspline``) passed
nothing, so it solved with the class default (off) and the two front ends
disagreed on fat wires: Dan's TaperedDipole at 2000 segs/lambda read
X = -2.511 ohm through the seam against -0.144 ohm from bs2 with "auto"
(momwire#1422). The roster now binds "auto" on the ``bspline`` name.

The gates drive the REAL seams -- the resident server's connection handler
(``momwire.eznec._resident._connection``, the function the warm daemon runs
per deck and the worker pool builds by name), the one-shot ``render``, the
NEC-4.2 slot and the nec2 portal's ``run_deck`` -- and count the exact-kernel
correction where it runs, so a green result is one where the branch was
reached, not one where nothing moved. The bit-identity half is the reason
"auto" can be a default at all: below h/a 3 nothing engages, so every
ordinary mesh prints the bytes it printed before.
"""

from __future__ import annotations

import io
import re
import socket
import threading
import warnings
from types import MappingProxyType

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire.deck import _solver
from momwire.deck._solver import BASES, NEC2_BASES
from momwire.eznec import _resident, _serve, _shell
from momwire.eznec._nec4 import render as render_nec4

# A half-metre dipole at 299.7925 MHz (lambda = 1 m), the shape of capture
# 0010 but twelve segments, so NEC-5's node 6 is the centre. h = 41.7 mm.
# Thin (a = 0.5 mm, capture 0010's own): h/a 83, auto stays off. Fat
# (a = 20 mm): h/a 2.08 < 3, auto engages. Near (a = 13 mm): h/a 3.21,
# fat enough to tempt but above the threshold, so it must stay off too.
_DECK = (
    "CM synthetic fat-wire dipole (momwire#1428)\nCE\n"
    "GW 1,12,0.,-.25,0.,0.,.25,0.,{a}\nGE 0,-1\n"
    "FR 0,1,0,0,299.7925\nGN -1\nEX 4,1,6,0,1.414214,0.\nPQ 0\n"
    "RP 0,1,37,1000,90.,0.,0.,10.,0.\nEN\n"
)
THIN, FAT, NEAR = 0.0005, 0.02, 0.013

# A fat dipole in the nec2 dialect, for the portal and the NEC-4.2 slot
# (EX 0 on segment 6, the centre of eleven; h/a 2.27).
_NEC2 = (
    "CM synthetic fat-wire dipole (momwire#1428)\nCE\n"
    "GW 1 11 0 -0.25 0 0 0.25 0 {a}\nGE 0\n"
    "EX 0 1 6 0 1 0\nFR 0 1 0 0 299.7925\nXQ\nEN\n"
)

_ROSTER_OFF = MappingProxyType(
    {**BASES, "bspline": (BSplineSolver, MappingProxyType({}))}
)


def _deck(a: float) -> str:
    return _DECK.format(a=f"{a:g}")


@pytest.fixture
def corrections(monkeypatch) -> list[int]:
    """How many times the exact-kernel correction ran, and on how many
    coaxial groups' worth of segments (one entry per fill)."""
    seen: list[int] = []
    original = BSplineSolver._add_exact_kernel_correction

    def counted(self, Z, geom, supp_seg, polys, k):
        seen.append(int(geom["n_segs_total"]))
        return original(self, Z, geom, supp_seg, polys, k)

    monkeypatch.setattr(BSplineSolver, "_add_exact_kernel_correction", counted)
    return seen


@pytest.fixture
def built(monkeypatch) -> list[BSplineSolver]:
    """Every solver the NEC-5 seam constructs, as constructed."""
    solvers: list = []
    original = _serve._solver_for

    def spy(*args, **kwargs):
        solver = original(*args, **kwargs)
        solvers.append(solver)
        return solver

    monkeypatch.setattr(_serve, "_solver_for", spy)
    return solvers


def _roster_without_auto(monkeypatch) -> None:
    """The red control: the roster as it was before momwire#1428."""
    monkeypatch.setattr(_solver, "BASES", _ROSTER_OFF)


def _serve_resident(deck: str, basis: str = "bspline") -> str:
    """One deck through the warm daemon's connection handler, over a real
    socket pair, exactly as the thin client sends it: the bytes, then EOF."""
    ours, theirs = socket.socketpair()
    log = io.StringIO()

    def handle() -> None:
        # The daemon's accept loop owns the socket and closes it after the
        # handler returns; the handler only closes its own text wrappers.
        try:
            _resident._connection(theirs, 1, log, threading.Lock(), basis=basis)
        finally:
            theirs.close()

    worker = threading.Thread(target=handle, daemon=True)
    worker.start()
    try:
        ours.sendall(deck.replace("\n", "\r\n").encode("latin-1"))
        ours.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := ours.recv(65536):
            chunks.append(chunk)
    finally:
        ours.close()
        worker.join(timeout=120)
    assert not worker.is_alive(), log.getvalue()
    return b"".join(chunks).decode("latin-1")


_E = re.compile(r"[-+]?\d\.\d+E[-+]\d+")


def _printed_z(printout: str) -> complex:
    """The ANTENNA INPUT PARAMETERS row's impedance (5 significant figures)."""
    assert "NEC ERROR" not in printout, printout[-2000:]
    lines = printout.splitlines()
    start = next(i for i, ln in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in ln)
    row = next(f for ln in lines[start + 1 :] if len(f := _E.findall(ln)) >= 6)
    return complex(float(row[4]), float(row[5]))


def _direct(a: float, exact_kernel) -> complex:
    """bs2 constructed directly on the same antenna, spelled as the NEC-5
    dialect spells a node feed: two six-segment halves meeting at the centre
    node, driven by a node gap there, under the dialect's kernel default
    (extended, momwire#1326)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = BSplineSolver(
            wires=[
                np.array([(0.0, -0.25, 0.0), (0.0, 0.0, 0.0)]),
                np.array([(0.0, 0.0, 0.0), (0.0, 0.25, 0.0)]),
            ],
            n_per_edge_per_wire=[[6], [6]],
            feeds=[],
            node_gaps=[(1, "start", 1.0 + 0j)],
            # The dialect's own c (299.8 MHz*m), as the seam meshes it.
            wavelength=_serve.SPEED_OF_LIGHT_MHZ_M / 299.7925,
            wire_radius=a,
            extended_kernel=True,
            exact_kernel=exact_kernel,
        )
        return complex(np.asarray(solver.compute_impedance()[0]).ravel()[0])


def _close(printed: complex, exact: complex) -> bool:
    """A printed value against a solver's, to the printout's 5 figures."""
    return abs(printed.real - exact.real) <= 1e-4 * max(1.0, abs(exact.real)) and abs(
        printed.imag - exact.imag
    ) <= 1e-4 * max(1.0, abs(exact.imag))


# --------------------------------------------------------------------------
# the roster


def test_the_roster_binds_auto_on_bs2_and_nowhere_else():
    assert BASES["bspline"][1]["exact_kernel"] == "auto"
    assert NEC2_BASES["bspline"][1]["exact_kernel"] == "auto"
    others = {name for name, (_c, kw) in BASES.items() if "exact_kernel" in kw}
    assert others == {"bspline"}
    # The name it binds must be a choice the class declares.
    assert "auto" in BSplineSolver.EXACT_KERNEL_CHOICES


# --------------------------------------------------------------------------
# the NEC-5 seam: the resident daemon's handler and the one-shot render


def test_the_resident_seam_runs_the_exact_kernel_on_a_fat_wire(corrections, built):
    printout = _serve_resident(_deck(FAT))
    assert corrections == [12], corrections
    (solver,) = built
    assert solver.exact_kernel_requested == "auto"
    assert solver.exact_kernel is True
    z_auto, z_off = _direct(FAT, "auto"), _direct(FAT, False)
    # The two kernels must be told apart by the printout's own precision, or
    # the match below would prove nothing.
    assert not _close(z_auto, z_off), (z_auto, z_off)
    assert _close(_printed_z(printout), z_auto), (_printed_z(printout), z_auto)


def test_red_control_without_the_binding_the_seam_is_off(
    monkeypatch, corrections, built
):
    _roster_without_auto(monkeypatch)
    printout = _serve_resident(_deck(FAT))
    assert corrections == []
    (solver,) = built
    assert solver.exact_kernel is False
    assert _close(_printed_z(printout), _direct(FAT, False))
    assert not _close(_printed_z(printout), _direct(FAT, "auto"))


@pytest.mark.parametrize("a", [THIN, NEAR])
def test_an_ordinary_mesh_prints_the_same_bytes(monkeypatch, corrections, built, a):
    """h/a >= 3: "auto" resolves off, and the printout is byte-for-byte the
    one the roster without the binding prints."""
    ours = _shell.render(_deck(a), basis="bspline")
    assert corrections == []
    assert [s.exact_kernel for s in built] == [False]
    assert [s.exact_kernel_requested for s in built] == ["auto"]
    _roster_without_auto(monkeypatch)
    before = _shell.render(_deck(a), basis="bspline")
    assert ours == before


def test_the_one_shot_render_and_the_daemon_agree(corrections):
    assert _shell.render(_deck(FAT), basis="bspline") == _serve_resident(
        _deck(FAT)
    ).replace("\r\n", "\n")
    assert corrections == [12, 12]


@pytest.mark.parametrize("basis", ["bspline-d1", "razor-2p", "sinusoidal"])
def test_other_bases_are_untouched(corrections, basis):
    _shell.render(_deck(FAT), basis=basis)
    assert corrections == []


# --------------------------------------------------------------------------
# the NEC-4.2 slot and the nec2 portal read the same roster


def test_the_nec42_slot_runs_it_too(corrections):
    text = _NEC2.format(a=f"{FAT:g}")
    printout = render_nec4(text, basis="bspline")
    assert "NEC ERROR" not in printout
    assert corrections == [11]


def test_the_portal_runs_it_too(corrections):
    from momwire.portal import _portal

    out, err = io.StringIO(), io.StringIO()
    with _portal.engine_scope():
        rc = _portal.main(
            ["--basis", "bspline"],
            stdin=io.StringIO(_NEC2.format(a=f"{FAT:g}")),
            stdout=out,
            stderr=err,
        )
    assert rc == 0, err.getvalue()
    assert "ANTENNA INPUT PARAMETERS" in out.getvalue(), out.getvalue()[-2000:]
    assert corrections == [11]
