# Vendored: scipy/xsf

- Upstream: https://github.com/scipy/xsf
- Commit: `261e034dd644cbe0e5373c1bc599abfcaae539f0` (upstream date 2026-08-26)
- Vendored: 2026-08-27, for momwire#680 U2 (the near-interface C++ twin)
- License: BSD-3-Clause (`LICENSE` here; bundled-code licenses in
  `LICENSES_bundled.txt`)
- Contents: `include/` copied verbatim from the upstream commit, plus the
  three license/readme files. Nothing else from the upstream repo (tests,
  build files) is vendored, and nothing in `include/` is modified.

What we use and why: `xsf::cyl_hankel_1` / `cyl_hankel_2` at COMPLEX
argument (scipy.special's own Amos translation — the rotated tail rays of
`_near_interface.six_point` need H₀ off the real axis, which Boost.Math
does not provide) plus real-argument J₀/J₁ via `xsf::cyl_bessel_j`. The
library is header-only C++17; `_near_interface_accel.cpp` compiles with
`-Iextern/xsf/include` and there is no link-time or install-time footprint.

Underflow contract (load-bearing for the quiet-panel rule): on Amos
underflow (`nz != 0`) xsf maps to `SF_ERROR_UNDERFLOW`, which does NOT
NaN the value — the exact 0.0 comes back, which is precisely the
"underflow panels are quiet, the tail IS zero" semantics the Python walk
pinned. Only DOMAIN/OVERFLOW/NO_RESULT produce NaN.

Parity discipline: values here were measured bit-identical to the
*installed* scipy 1.18.0 on spot checks, but the gates are RELATIVE
(1e-12), never bit — scipy versions drift and cross-build bit equality is
a banned pin (house rule, momwire#249/#270).

To re-vendor: copy `include/` from a newer upstream commit, update the
commit hash and dates here, and re-run the near-interface parity ledger
(`momwire/tests/test_near_interface_accel_680.py`).
