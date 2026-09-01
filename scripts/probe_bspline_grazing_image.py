"""momwire#631 — bspline's grazing residual is TWO under-resolved quadratures.

Where #631 left it
------------------
#630 keyed razor's Sommerfeld remainder source order to the grazing height and
razor-nec5 converged on 0033 (1.46 / 0.90 / 0.94 / 0.94 / 0.80 % at N = 5..41).
bspline did not, and the issue recorded three things that together looked like
a defect pointing AWAY from the ground terms:

  * raising ``n_qp_sommerfeld`` to 192 does not rescue it (83.6 % at N = 25),
  * its PEC control converges (24.17 -> 3.42 %) while ``GN 0`` diverges,
  * yet its ground-correction ENTRIES are right (the near-diagonal band test
    lands on 0.0389/0.0396/0.0392 against the 0.0392 limit at every mesh).

All three readings were taken on 0033 — five wires, a junction and a screen —
except the band test, which was taken on the one-wire reproducer.  Splitting
those two decks apart is what this probe does, and it overturns the framing.

``--mode split`` — one wire vs 0033 under refinement, PEC control beside the
finite-ground column so each deck carries its own basis-error baseline.  On
the ONE WIRE at 1.09e-4 lambda, bspline's PEC control is not converged at all:

  n            4       8      16      24      32
  bspline PEC  998.67  458.80 195.12  112.07  73.04   (%)
  razor  PEC     0.00    0.00   0.00    0.00   0.00

nec5cl says Z(GN 1) = 0.0000 - 598.49j at n = 4 and bspline says +5378.43j —
the reactance sign is flipped.  #631's "PEC converges" was 0033's column, where
the non-grazing vertical dominates Z and dilutes the radials.

``--mode pec`` — and it is GRAZING-keyed, with no soil in the deck at all.
Free space is the control (height-invariant by translation, so this column
must not move) and PEC is the measurement:

  h/lambda        1e-1   3e-2   1e-2   3e-3   1e-3   1.09e-4
  n=4  free       5.67   5.67   5.67   5.67   5.67   5.67
  n=4  PEC        5.68   5.51   4.56   5.99  37.16   998.67
  n=16 free       1.21   1.21   1.21   1.21   1.21   1.21
  n=16 PEC        1.21   1.19   1.09   0.83   2.07   195.12

razor is 0.00-0.01 % in every cell of both columns.  The PEC column tracks the
free-space basis error down to 3e-3 lambda and then detonates.  So the defect
reproduces over a CLOSED-FORM image: no Sommerfeld surface, no interpolation
grid, no soil.  That also explains "entries right, answer wrong" — the band
test's denominator is Z(``GN 1``) - Z(``GN -1``), so a broken image term sits
in both its numerator and its denominator and divides out.  **The band test is
blind to this defect by construction**, which matters because #510 §5 banked it
as a candidate CI gate.

Where it comes from
-------------------
``_build_J_image_blocks`` states the premise in prose::

    The image is always far enough from the original that the analytic
    same-edge static + reg split doesn't apply — full off-edge quadrature
    handles every (i, j) pair uniformly.

and integrates every image pair at a fixed ``n_qp_pair`` (default 4).  At
grazing a segment's own image sits 2h away — 3.6 cm under a 2.48 m segment at
n = 16 — so the premise fails exactly where the error appeared.  Same defect
SHAPE as #510 (a geometric premise baked into a fixed order, true where the
unit was gated), in a different code path: the exact image rather than Q.

``--mode image`` raises the order on the IMAGE BLOCK ALONE:

  n=16, h/lambda   delta/2h       4       8      16      32      64     128
  free space          --       1.21    1.21    1.21    1.21    1.21    1.21
  PEC 1e-2           0.76      1.09    1.09    1.09    1.09    1.09    1.09
  PEC 1e-3           7.57      2.07    0.58    0.46    0.46    0.46    0.46
  PEC 1.09e-4       69.42    195.12   80.03   25.87    5.48    0.56    0.09

Free space and the clean rung are FLAT — the hook cannot touch a deck with no
image block, and a well-resolved image does not care about the order.  Only
the grazing rungs move, and the order they need tracks delta/2h at a roughly
constant ratio (~2.1x at 7.57 and 30.3, ~1.8x at 69.4), which is #630's
``C * len / R_min`` keying with R_min = 2h.

``--mode cross`` — **and ONE knob is not enough.**  Image order against
remainder order, one wire, n = 16, h/lambda = 1.09e-4, ``GN 0``:

  image n_qp | somm=3   somm=12   somm=48   somm=192
           4 | 393.38    320.81    306.22     305.68
          32 | 164.02     30.74     10.28       9.55
         128 | 152.01     21.60      1.34       0.62

Neither edge works and the corner does.  The PEC half of the same cross is
flat across ``n_qp_sommerfeld`` at every image order (195.12 / 5.48 / 0.09),
which is the cross's own control — a PEC deck has no remainder to resolve.

So the finding is::

    bspline's grazing error is TWO under-resolved quadratures, not one: the
    exact-image block (`n_qp_pair`, premised on the image being far) and the
    Sommerfeld remainder (`n_qp_sommerfeld`, #630's premise).  Keying either
    alone leaves 152 % / 306 %; keying both leaves 0.62 %.

and #631's "raising n_qp_sommerfeld to 192 does not rescue bspline" was true
but was moving one of two broken terms.  razor needed only one knob because
razor's PEC is exact — 0.00 % at every mesh and every height above.

What a fix has to deal with
---------------------------
The C++ off-edge kernel refuses ``n_qp > 8`` ("scratch buffer size") and the
regular same-edge kernel refuses ``n_qp**2 > 64``, so the orders this probe
measures are reachable only through the numpy fallback.  A shipped fix cannot
simply raise the order the way #630 did; it needs either a wider C++ scratch
buffer or an accepted fallback cost on grazing decks.  Cost is not free even
then: at delta/2h = 69 the measured requirement is ~128 points per segment
against a default of 4.

Instruments
-----------
The image-order hook goes on the KERNEL, not on ``_build_J_image_blocks``:
bspline reaches its image by more than one assembly path (that function for
the dense route, ``_accumulate_Z_image_chunked`` for the chunked one) and
#510 §6's trap was a per-class hook that never fired.  Both call
``_seg_seg_full_moments_offedge``, and bspline binds that name at import, so
the patch goes on ``momwire.bspline``'s copy.  Image calls are told apart by
the mirror itself — source side below z = 0, observer side above — and the
wrapper COUNTS its hits so a run that silently misses cannot pass for a null
result.  Forced orders are read in the function BODY, never as a default
argument, because a module constant bound at import is what made #510's
negative control pass for the wrong reason.

Needs the antennaknobs venv and
``NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl``.

    scripts/probe_bspline_grazing_image.py --mode cross
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_grazing_height_floor import (
    FREQ_MHZ,
    RADIAL_LEN,
    RADIUS,
    WL,
    _num,
    deck,
    momwire_z,
)

TRUNKS = ("bspline", "razor-nec5")
SOIL = (13.0, 0.005)

PEC_HEIGHTS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1.09e-4)
IMAGE_ORDERS = (4, 8, 16, 32, 64, 128)
SOMM_ORDERS = (3, 12, 48, 192)


def one_wire(h, *, n_seg, ground, length=RADIAL_LEN):
    """One horizontal wire at height `h`, driven at the CENTRE node.

    `wire_deck` in the sibling probe drives node 2, which sits at
    x = L/n_seg and therefore walks toward the wire end as the mesh refines —
    fine for a fixed-mesh sweep, wrong for a refinement study, which would
    then be changing the antenna as well as the mesh.  Even `n_seg` puts a
    node exactly on x = L/2 (node k is at (k-1)/n_seg * L), so the feed is
    the same physical point at every mesh here.

    `ground` is "free" (`GE 0,0`, no `GN` card), "pec" (`GN 1`, the
    closed-form image) or "gn0" (`GN 0`, Sommerfeld over average soil).
    """
    if n_seg % 2:
        raise ValueError(f"n_seg must be even to have a centre node, got {n_seg}")
    node = n_seg // 2 + 1
    if ground == "free":
        ge, gn = "GE 0,0", ""
    elif ground == "pec":
        ge, gn = "GE 1,-1", "GN 1,0,0,0\n"
    else:
        ge, gn = "GE 1,-1", f"GN 0,0,0,0,{_num(SOIL[0])},{_num(SOIL[1])},1.,0.\n"
    return (
        "CM momwire#631 — one horizontal wire, grazing, centre-driven\n"
        f"CM h = {h:.6g} m = {h / WL:.4g} wl, n = {n_seg}, {ground}\nCE\n"
        f"GW 1,{n_seg},0.,0.,{_num(h)},{_num(length)},0.,{_num(h)},{_num(RADIUS)}\n"
        f"{ge}\n"
        f"FR 0,1,0,0,{_num(FREQ_MHZ)}\n{gn}"
        f"EX 0,1,{node},0,1.414214,0.\nXQ 0\nEN\n"
    )


class ImageOrderHook:
    """Force the quadrature order on bspline's IMAGE block alone.

    Hooked at the kernel because bspline reaches its image by more than one
    assembly path; see the module docstring.  `hits` counts firings so a
    silent miss cannot be read as a null result.
    """

    def __init__(self):
        from momwire import _bspline_kernels as bk
        from momwire import bspline as bs

        self._bk = bk
        self._bs = bs
        self._orig = bs._seg_seg_full_moments_offedge
        self.order = None
        self.hits = 0

    def __enter__(self):
        self._bs._seg_seg_full_moments_offedge = self._patched
        return self

    def __exit__(self, *exc):
        self._bs._seg_seg_full_moments_offedge = self._orig
        return False

    def _patched(self, seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, n_qp, **kw):
        n = self.order  # read in the BODY — a default argument binds at import
        if n is not None:
            zj = np.asarray(seg_l_j)[:, 2]
            zi = np.asarray(seg_l_i)[:, 2]
            if zj.max() < 0.0 <= zi.min():  # source mirrored, observer real
                self.hits += 1
                bk = self._bk
                save = (bk._HAVE_BSPLINE_ACCEL, bk._HAVE_BSPLINE_OFFEDGE_EK_ACCEL)
                # The C++ kernel refuses n_qp > 8; the orders that matter here
                # are only reachable through the numpy fallback.
                bk._HAVE_BSPLINE_ACCEL = False
                bk._HAVE_BSPLINE_OFFEDGE_EK_ACCEL = False
                try:
                    return self._orig(
                        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, int(n), **kw
                    )
                finally:
                    (
                        bk._HAVE_BSPLINE_ACCEL,
                        bk._HAVE_BSPLINE_OFFEDGE_EK_ACCEL,
                    ) = save
        return self._orig(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, n_qp, **kw)


class SommOrderHook:
    """Force `n_qp_sommerfeld` on every BSplineSolver this run builds."""

    def __init__(self):
        from momwire import BSplineSolver

        self._cls = BSplineSolver
        self._orig = BSplineSolver.__init__
        self.order = None

    def __enter__(self):
        hook = self

        def patched(inner, *a, **kw):
            if hook.order is not None:
                kw["n_qp_sommerfeld"] = int(hook.order)
            return hook._orig(inner, *a, **kw)

        self._cls.__init__ = patched
        return self

    def __exit__(self, *exc):
        self._cls.__init__ = self._orig
        return False


def _engine():
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    return NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)


def _err(z, z_ref):
    return 100.0 * abs(z - z_ref) / abs(z_ref)


def mode_split(eng) -> list[dict]:
    """One wire vs 0033 under refinement, PEC control beside `GN 0`.

    Every mesh-divergence number in #631 is 0033's; the band test that came
    out clean is the one wire's.  If the one wire's PEC control is ALSO broken
    the framing "PEC converges, finite ground diverges" belongs to 0033 only.
    """
    h = 1.09e-4 * WL
    cases = (
        (
            "one wire (centre-driven)",
            (4, 8, 16, 24, 32),
            lambda n, g: one_wire(h, n_seg=n, ground=g),
        ),
        (
            "0033 (junction + screen)",
            (5, 9, 15, 25),
            lambda n, g: deck(h, n_seg=n, pec=(g == "pec")),
        ),
    )
    rows = []
    for title, meshes, build in cases:
        print(f"\n{'=' * 78}\n{title}   h/lambda = 1.09e-4\n{'=' * 78}")
        hdr = f"{'n':>4} | {'ground':>6} | {'nec5cl':>21} | " + " | ".join(
            f"{t:>21} {'err%':>8}" for t in TRUNKS
        )
        print(hdr)
        print("-" * len(hdr))
        for n in meshes:
            for ground in ("pec", "gn0"):
                text = build(n, ground)
                z_ref = complex(eng.run_deck(text)[0][0][2])
                line = (
                    f"{n:>4} | {ground:>6} | {z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j |"
                )
                row = dict(deck=title, n=n, ground=ground)
                for trunk in TRUNKS:
                    z = momwire_z(text, trunk)
                    if z is None:
                        line += f" {'REFUSED':>21} {'':>8} |"
                        continue
                    e = _err(z, z_ref)
                    line += f" {z.real:>10.4f}{z.imag:>+10.4f}j {e:>8.2f} |"
                    row[f"{trunk}_err_pct"] = e
                print(line, flush=True)
                rows.append(row)
    return rows


def mode_pec(eng, meshes) -> list[dict]:
    """Is bspline's PEC error grazing-keyed, with no soil in the deck?

    Free space is the control: the wire is translation-invariant with no
    ground, so that column must not move with height.  If it does, the sweep
    is wrong rather than the basis.
    """
    rows = []
    for n in meshes:
        for ground in ("free", "pec"):
            print(f"\n{'=' * 78}\nn = {n}   {ground}\n{'=' * 78}")
            hdr = f"{'h/lambda':>10} | {'nec5cl':>21} | " + " | ".join(
                f"{t:>21} {'err%':>8}" for t in TRUNKS
            )
            print(hdr)
            print("-" * len(hdr))
            for hw in PEC_HEIGHTS:
                text = one_wire(hw * WL, n_seg=n, ground=ground)
                z_ref = complex(eng.run_deck(text)[0][0][2])
                line = f"{hw:>10.3g} | {z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j |"
                row = dict(n=n, ground=ground, h_over_wl=hw)
                for trunk in TRUNKS:
                    z = momwire_z(text, trunk)
                    if z is None:
                        line += f" {'REFUSED':>21} {'':>8} |"
                        continue
                    e = _err(z, z_ref)
                    line += f" {z.real:>10.4f}{z.imag:>+10.4f}j {e:>8.2f} |"
                    row[f"{trunk}_err_pct"] = e
                print(line, flush=True)
                rows.append(row)
    return rows


def mode_image(eng, meshes, orders) -> list[dict]:
    """Raise the order on the image block alone and watch the grazing rungs.

    Free space (no image block at all) and the clean rung are the controls;
    both must stay flat while the grazing rungs collapse.
    """
    rows = []
    cases = (("free", 1.09e-4), ("pec", 1e-2), ("pec", 1e-3), ("pec", 1.09e-4))
    with ImageOrderHook() as hook:
        _positive_control(hook)
        for n in meshes:
            delta = RADIAL_LEN / n
            print(f"\n{'=' * 78}\nn_seg = {n}   (segment {delta:.4g} m)\n{'=' * 78}")
            for ground, hw in cases:
                h = hw * WL
                text = one_wire(h, n_seg=n, ground=ground)
                z_ref = complex(eng.run_deck(text)[0][0][2])
                extra = (
                    "" if ground == "free" else f"   delta/2h = {delta / (2 * h):.4g}"
                )
                print(
                    f"\n{ground:>4}  h/lambda = {hw:<9.3g}{extra}\n"
                    f"   nec5cl = {z_ref.real:.4f}{z_ref.imag:+.4f}j"
                )
                print(f"   {'image n_qp':>10} | {'bspline':>23} | {'err%':>9}")
                print("   " + "-" * 48)
                for nq in orders:
                    hook.order, hook.hits = int(nq), 0
                    z = momwire_z(text, "bspline")
                    fired = hook.hits
                    hook.order = None
                    if z is None:
                        continue
                    flag = "" if (fired or ground == "free") else "  <- HOOK MISSED"
                    e = _err(z, z_ref)
                    print(
                        f"   {nq:>10} | {z.real:>10.4f}{z.imag:>+11.4f}j | "
                        f"{e:>9.2f}{flag}",
                        flush=True,
                    )
                    rows.append(
                        dict(
                            n=n,
                            ground=ground,
                            h_over_wl=hw,
                            image_n_qp=int(nq),
                            hits=fired,
                            err_pct=e,
                        )
                    )
    return rows


def mode_cross(eng, n_seg, orders, somm_orders) -> list[dict]:
    """Image order AGAINST remainder order — is either one enough alone?

    #631 recorded that `n_qp_sommerfeld` = 192 does not rescue bspline.  That
    was measured with the image order still at its default, so it moved one of
    two terms.  The PEC half of this cross is the control: a PEC deck has no
    remainder, so its rows must be flat across `n_qp_sommerfeld`.
    """
    h = 1.09e-4 * WL
    delta = RADIAL_LEN / n_seg
    rows = []
    with ImageOrderHook() as ihook, SommOrderHook() as shook:
        _positive_control(ihook)
        for ground in ("pec", "gn0"):
            text = one_wire(h, n_seg=n_seg, ground=ground)
            z_ref = complex(eng.run_deck(text)[0][0][2])
            print(f"\n{'=' * 78}")
            print(
                f"{ground.upper()}   n = {n_seg}, h/lambda = 1.09e-4, "
                f"delta/2h = {delta / (2 * h):.4g}"
            )
            print(f"nec5cl = {z_ref.real:.4f}{z_ref.imag:+.4f}j\n{'=' * 78}")
            hdr = f"{'image n_qp':>11} |" + "".join(
                f"   somm={s:<4} err%" for s in somm_orders
            )
            print(hdr)
            print("-" * len(hdr))
            for iq in orders:
                line = f"{iq:>11} |"
                for sq in somm_orders:
                    ihook.order, ihook.hits, shook.order = int(iq), 0, int(sq)
                    z = momwire_z(text, "bspline")
                    fired = ihook.hits
                    ihook.order = shook.order = None
                    if z is None:
                        line += f"{'REFUSED':>17}"
                        continue
                    if not fired:
                        line += f"{'HOOK MISSED':>17}"
                        continue
                    e = _err(z, z_ref)
                    line += f"{e:>17.2f}"
                    rows.append(
                        dict(
                            ground=ground,
                            n=n_seg,
                            image_n_qp=int(iq),
                            n_qp_sommerfeld=int(sq),
                            err_pct=e,
                        )
                    )
                print(line, flush=True)
    return rows


def _positive_control(hook):
    """Assert the image hook FIRES and CHANGES the answer before reading it."""
    probe = one_wire(1.09e-4 * WL, n_seg=4, ground="pec")
    hook.order, hook.hits = 4, 0
    z_lo, hits_lo = momwire_z(probe, "bspline"), hook.hits
    hook.order, hook.hits = 64, 0
    z_hi, hits_hi = momwire_z(probe, "bspline"), hook.hits
    hook.order = None
    if not hits_lo or not hits_hi:
        raise SystemExit(
            f"POSITIVE CONTROL FAILED: the image hook never fired "
            f"({hits_lo}, {hits_hi} hits) — bspline reached its image by a "
            f"path this patch does not cover, and a flat sweep would be the "
            f"patch's silence rather than the solver's."
        )
    if z_lo is None or z_hi is None or z_lo == z_hi:
        raise SystemExit(
            f"POSITIVE CONTROL FAILED: image order 4 and 64 both gave {z_lo} — "
            f"the order is not reaching the quadrature."
        )
    print(
        f"positive control OK: image hook fired {hits_lo}/{hits_hi} times; "
        f"order 4 -> {z_lo:.4f}, 64 -> {z_hi:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=("split", "pec", "image", "cross"), default="cross"
    )
    p.add_argument("--meshes", type=int, nargs="*", default=None)
    p.add_argument("--orders", type=int, nargs="*", default=None)
    p.add_argument("--somm-orders", type=int, nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    eng = _engine()
    if args.mode == "split":
        rows = mode_split(eng)
    elif args.mode == "pec":
        rows = mode_pec(eng, args.meshes or (4, 16))
    elif args.mode == "image":
        rows = mode_image(eng, args.meshes or (4, 16), args.orders or IMAGE_ORDERS)
    else:
        rows = mode_cross(
            eng,
            (args.meshes or (16,))[0],
            args.orders or (4, 32, 128),
            args.somm_orders or SOMM_ORDERS,
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
