"""momwire#935: re-measure the Wynn-fallback truncation ladder.

The refusal message quotes "~4e-9 just under the floor to ~9e-4 an octave
below it". Those were 0.08 deg and 0.023 deg when the floor was 0.1. With the
floor at 0.05 the same sentence points at different angles, so the figures are
re-measured rather than re-interpolated.

Method: the capped run against a reference with the cap lifted to 40000, at
the same point. The cap reaches the accelerated contour as a module-global
lookup inside `_six_below_accel`, so patching `below._MAX_TAIL_PANELS` is what
takes effect -- patching `_tail_below.__defaults__` does NOT, because the
numpy path is not the one that runs. Measured the wrong way first: the
"reference" came back refused, quoting 8000, which is the tell.
"""

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOIL = (13.0, 0.005)
_SURF = below._SURF_KEYS


def _set_cap(n):
    below._MAX_TAIL_PANELS = n


def at_unrefused(th_deg, cap, r1_over_lam=1.0, f=7e6):
    """The value the contour WOULD return at `cap`, refusal suppressed.

    Below the floor the point of the measurement is precisely the number the
    Wynn fallback produces, which the #841 refusal exists to withhold. So the
    refusal is stubbed for the duration -- this is the one context where
    reading it is the right thing to do, and it is a scratch probe, not the
    library.
    """
    orig_refuse = below._refuse_if_capped
    below._refuse_if_capped = lambda *a, **k: None
    try:
        return at(th_deg, cap, r1_over_lam, f)
    finally:
        below._refuse_if_capped = orig_refuse


def at(th_deg, cap, r1_over_lam=1.0, f=7e6):
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    eps_t = _ground_refl.eps_tilde(SOIL, om, EPS0)
    lam_m = below.lambda_medium(eps_t, k2)
    _set_cap(cap)
    return below.iv_surfaces_direct_below(
        eps_t,
        k2,
        np.array([r1_over_lam * lam_m]),
        np.radians([th_deg]),
        rtol=1e-9,
        omega=om,
    )


orig = below._MAX_TAIL_PANELS
print(f"shipped cap = {orig}")
print("\ntheta    rel err of the capped value vs a 40000-panel reference")
for th in (0.08, 0.05, 0.04, 0.03, 0.025, 0.023):
    served = True
    try:
        at(th, orig)
    except ValueError:
        served = False
    got = at_unrefused(th, orig)
    ref = at_unrefused(th, 40000)
    scale = max(abs(complex(ref[k][0])) for k in _SURF)
    err = max(abs(complex(got[k][0]) - complex(ref[k][0])) for k in _SURF) / scale
    tag = "served" if served else "REFUSED"
    print(f"  {th:6.3f}   {err:.3e}   {tag}")
_set_cap(orig)
