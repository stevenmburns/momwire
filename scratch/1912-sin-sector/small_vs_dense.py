"""momwire#1912: max relative route-vs-dense difference per small deck (Z or Y
and coefficients), off the test module's deck builders. usage:
small_vs_dense.py PATH/TO/tests/test_rotational_sinusoidal_1912.py"""

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("t", sys.argv[1])
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
rev = t.reverse(*t.cage(), [3, 2 + t.N + 2, 2 + 2 * t.N + 2])
revl = t.reverse(*t.cage(lift=t.LIFT), [3, 2 + t.N + 2, 2 + 2 * t.N + 2])
cases = [
    ("cage4 free Z", (*t.cage(), [t.FEED]), {}, False),
    (
        "cage4 somm Z",
        (*t.cage(lift=t.LIFT), [t.FEED]),
        dict(ground="sommerfeld"),
        False,
    ),
    ("cage6 free Z", (*t.cage(6), [t.FEED]), {}, False),
    ("cage4 posts free Y", (*t.cage(), [t.FEED, *t.posts()]), {}, True),
    (
        "cage4 posts somm Y",
        (*t.cage(lift=t.LIFT), [t.FEED, *t.posts()]),
        dict(ground="sommerfeld"),
        True,
    ),
    ("reversed free Z", (*rev, [t.FEED]), {}, False),
    ("reversed free Y", (*rev, [t.FEED, *t.posts(reversed_=(1,))]), {}, True),
    (
        "reversed somm Y",
        (*revl, [t.FEED, *t.posts(reversed_=(1,))]),
        dict(ground="sommerfeld"),
        True,
    ),
    ("parasitic4 free Z", t.parasitic(), {}, False),
    ("parasitic4 somm Z", t.parasitic(lift=t.LIFT), dict(ground="sommerfeld"), False),
    ("screen pec Z", t.screen(droop=0.05), dict(ground="pec"), False),
    ("screen refl Z", t.screen(droop=0.05), dict(ground="refl-coef"), False),
    ("screen somm Z", t.screen(droop=0.05), dict(ground="sommerfeld"), False),
    ("top-hat contact somm Z", t.top_hat(), dict(ground="sommerfeld"), False),
    ("buried somm Z", t.buried(), dict(ground="sommerfeld"), False),
    (
        "1029 screen 4 somm Z",
        t.screen_1029(4),
        dict(ground="sommerfeld", wavelength=t.WL_1029),
        False,
    ),
    (
        "1029 screen 16 somm Z",
        t.screen_1029(16),
        dict(ground="sommerfeld", wavelength=t.WL_1029),
        False,
    ),
    ("detached mixed somm Z", t.detached(), dict(ground="sommerfeld"), False),
]
for name, (w, npe, feeds), kw, ports in cases:
    s, d = t.pair(w, npe, feeds, **kw)
    if ports:
        a, b = s.compute_port_solution(), d.compute_port_solution()
        print(
            f"{name:26s} Y {t.rel(a.y, b.y):.2e} coeffs {t.rel(a.coeffs, b.coeffs):.2e}"
        )
    else:
        (za, ca), (zb, cb) = s.compute_impedance(), d.compute_impedance()
        print(f"{name:26s} Z {t.rel(za, zb):.2e} coeffs {t.rel(ca, cb):.2e}")
