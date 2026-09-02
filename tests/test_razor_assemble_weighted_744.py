"""The razor fill's C++ WEIGHTED T1 assembly (momwire#744) against numpy.

#780 moved the UNWEIGHTED T1 assembly into C++ and said so in its own
docstring: free space and the folding grounds, i.e. `w_A_fn is None`. The
finite-ground weighted branch kept numpy because it carries a per-pair weight
table that signature had no room for -- and with #742's moment kernel and
#780's assembler both serving, that branch was what was left. Measured
threaded on this box at N=801 over a refl-coef ground, the weight windows were
40.8% of the fill's wall time and the contraction around them another ~28%.

The measurement has to be THREADED, which is the trap the issue's own comment
records: `razor_seg_moments` is built `-fopenmp` and the numpy windows are
not, so an `OMP_NUM_THREADS=1` profile handicaps the kernel alone and reads
the windows at 10% -- "not worth it", and an artifact of the pin.

WHAT THE KERNEL DOES, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------------
It forms the per-pair A-term window INSIDE the tile -- the
`specular_pair_tables` -> `fresnel_rho` -> `a_term_weights` chain, term for
term and in the same multi-step order -- so the `(n_obs_chunk, n_seg)` window
plane the numpy closure materialises is never built. It computes the two
columns each basis function reads rather than the whole plane, which is MORE
window arithmetic (~2x, since each wing segment is referenced by about two
basis functions) traded for the plane's memory traffic and for OpenMP.

It serves exactly ONE ground: refl-coef. Two are refused, for different
reasons, and both refusals are the interesting part of this module.

A COMPOSING (sommerfeld) ground would fuse trivially -- its window is the
constant C2 on the same mirrored tangent dot -- and an earlier draft of #744
served it, for a measured 1.26x on that deck. It is refused because C2 reaches
Z *through the windows* by design, and a fill that reads
`PotentialGround.image_coefficient` and applies it itself doubles the exact-
image half. `test_the_consumer_never_applies_the_image_coefficient_itself`
pins that with a ground that lies about its coefficient while its weights stay
honest, and it caught the draft. That win is still on the table; it needs C2
routed without the consumer reading it, which is a change to the ground
interface rather than to this kernel.

A ground whose per-pair weights are not the stock Fresnel pair is refused too
-- the radial-wire screen's `standard_fresnel = False` row (architecture doc
6.1) reaches the same `a_term_weights` with screen-modified rho_v / rho_h, and
a kernel that hard-codes the stock pair would serve it silently and WRONGLY.
`test_a_non_fresnel_ground_is_refused` is that boundary, gated on the selector
rather than on an answer, because the screen ground does not exist yet to
produce one.

What this module gates:

  * **the paths agree** on every configuration the kernel serves -- both
    grounds, both quadrature lanes, two meshes. NOT bitwise, deliberately, and
    for the reason `test_razor_assemble_accel_780` states at length: the repo
    does not pin cross-build bit equality, and momwire#781 was the cost of
    ignoring that. It happens that this kernel and numpy agree to 0.0e+00 on
    this box -- the same strided-reduction accident #780 records, since the
    path-point sum is still `reshape(...).sum(axis=1)` over a strided axis --
    but complex `sqrt` and complex division here are libstdc++'s where numpy
    uses its own, so a different libm is exactly where that would stop being
    true. The bars sit far above the measured agreement to leave room for it.

  * **the kernel actually runs.** The #822 lesson: the agreement tests would
    pass just as happily if the dispatch silently stopped dispatching, so the
    count is asserted nonzero on and zero off.

  * **the unweighted path is untouched.** #744 adds a twin; it does not edit
    `razor_assemble_t1`. The unweighted branch must still take that kernel and
    still produce what it produced.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from momwire import razor as _razor
from momwire.razor import RazorSolver

C0 = 299792458.0
LAM = C0 / 7.0e6

LANES = {"nec5": {"nec5_quadrature": True}, "gauss-legendre": {}}
GROUNDS = ["refl-coef", "sommerfeld"]


def _dipole(n, *, ground_model=None, **kw):
    deck = dict(
        wires=[np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + LAM / 2]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=0.005,
        wavelength=LAM,
        feeds=[(0, LAM / 4, 1 + 0j)],
    )
    if ground_model is not None:
        deck["ground_z"] = 0.0
        deck["ground_eps"] = (13.0, 0.005)
        deck["ground_model"] = ground_model
    deck.update(kw)
    return deck


def _solve(monkeypatch, deck, *, accel, **kw):
    """One solve with the WEIGHTED assembly kernel on or off.

    Only `_HAVE_RAZOR_WEIGHTED_ACCEL` is flipped -- never `_FORCE_NUMPY`,
    which also disables #742's moments and #780's unweighted assembly, so a
    comparison against it would measure three kernels and attribute the whole
    difference to this one. #780's module records making exactly that mistake.
    """
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_WEIGHTED_ACCEL", accel)
    z, _ = RazorSolver(**deck, **kw).compute_impedance()
    return complex(np.asarray(z).ravel()[0])


@pytest.mark.parametrize("lane", sorted(LANES))
@pytest.mark.parametrize("model", GROUNDS)
@pytest.mark.parametrize("n", [80, 160])
def test_both_grounds_and_lanes_agree_with_numpy(monkeypatch, lane, model, n):
    """Every configuration the kernel serves, against the numpy closure.

    `sommerfeld` is parametrised here deliberately even though the gate
    refuses it: flipping the flag on a ground the kernel does not serve must
    be a no-op, and that is worth a bar rather than an assumption.
    """
    deck = _dipole(n, ground_model=model)
    kw = LANES[lane]
    z_np = _solve(monkeypatch, deck, accel=False, **kw)
    z_acc = _solve(monkeypatch, deck, accel=True, **kw)
    rel = abs(z_acc - z_np) / abs(z_np)
    assert rel < 1e-11, f"{model}/{lane}/N={n}: Zin {z_acc} vs {z_np} (rel {rel:.3e})"


def test_the_kernel_actually_runs(monkeypatch):
    """Two tells: a nonzero count with the flag on, zero with it off."""
    if not _razor._HAVE_RAZOR_WEIGHTED_ACCEL:
        pytest.skip("build carries no razor_weighted_744 kernel")
    from momwire import _accelerators as _acc

    calls = {"n": 0}
    real = _acc.razor_assemble_t1_weighted

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(_acc, "razor_assemble_t1_weighted", counting)
    monkeypatch.setattr(
        _razor._acc, "razor_assemble_t1_weighted", counting, raising=False
    )

    deck = _dipole(80, ground_model="refl-coef")

    calls["n"] = 0
    _solve(monkeypatch, deck, accel=True)
    on = calls["n"]

    calls["n"] = 0
    _solve(monkeypatch, deck, accel=False)
    off = calls["n"]

    assert on > 0, "the weighted assembly kernel never ran with the flag on"
    assert off == 0, f"the kernel ran {off} times with the flag off"


def test_the_unweighted_path_is_untouched(monkeypatch):
    """#744 adds a twin; it does not edit `razor_assemble_t1`.

    Free space has no ground object at all, so the weighted gate cannot fire
    there. Flipping it must therefore change nothing -- if it does, the twin
    has reached a branch that is not its own.
    """
    deck = _dipole(120)
    z_off = _solve(monkeypatch, deck, accel=False)
    z_on = _solve(monkeypatch, deck, accel=True)
    assert z_on == z_off, f"free space moved with the weighted gate: {z_on} vs {z_off}"


# ----------------------------------------------------------------------
# the selector, which is where the scope boundary is actually enforced
# ----------------------------------------------------------------------


def _ground(mode, *, eps_tilde=complex(13.0, -0.5), standard_fresnel=True):
    """A stand-in carrying only the three attributes the selector reads."""
    return types.SimpleNamespace(
        mode=mode, eps_tilde=eps_tilde, standard_fresnel=standard_fresnel
    )


def test_the_refl_coef_ground_is_served():
    assert _razor._weighted_accel_serves(_ground("fold")) is True


def test_a_composing_ground_is_refused():
    """C2 reaches Z through the WINDOWS, never through the consumer.

    An earlier draft served this ground and doubled the exact-image half,
    which `test_the_consumer_never_applies_the_image_coefficient_itself`
    caught. The refusal is the fix, and it is pinned here so a later
    optimisation cannot quietly undo it without also answering the C2
    routing question.
    """
    assert _razor._weighted_accel_serves(_ground("compose")) is False


def test_free_space_and_pec_select_nothing():
    """No ground object, and a ground with no eps_tilde (PEC), are both the
    unweighted branch -- the selector must not claim them."""
    assert _razor._weighted_accel_serves(None) is False
    assert _razor._weighted_accel_serves(_ground("fold", eps_tilde=None)) is False


def test_a_non_fresnel_ground_is_refused():
    """The radial-wire screen's row: `weighted=True, standard_fresnel=False`.

    Its rho_v / rho_h are screen-modified and reach the same
    `a_term_weights`, so the numpy closure serves it correctly by reading the
    ground's own functions. The kernel hard-codes the stock Fresnel pair, so
    it must decline -- silently serving this one would be a wrong answer, not
    a slow one. There is no such ground in the tree yet, which is exactly why
    this is gated on the selector and not on an impedance.
    """
    screen = _ground("fold", standard_fresnel=False)
    assert _razor._weighted_accel_serves(screen) is False
