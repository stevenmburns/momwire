"""Bank the same-edge accelerator's square answers as a fixture (momwire#968).

Run against a build of the commit BEFORE a change to those kernels; the fixture
is then what `tests/test_same_edge_window_968.py` compares a rebuilt tree to.
That is the momwire#762 protocol — bit-identity is established by rebuilding and
comparing, never by reading a diff and concluding nothing moved.

    git stash && make build
    python scripts/bank_same_edge_oracle.py tests/data/same_edge_oracle_968.npz
    git stash pop && make build && pytest tests/test_same_edge_window_968.py

It caught a real one: writing the closed-form Toeplitz loop out twice in one
translation unit changed GCC's inlining of `J_static_dispatch` and moved the
square answer by 5.3e-15 relative on a handful of entries.
"""

import itertools
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from momwire._accel import acc as _acc  # noqa: E402
from momwire._bspline_kernels import _seg_seg_reg_geometry  # noqa: E402

H, A, A_EK = 0.37, 5e-4, 1.7e-3
KS = np.ascontiguousarray(np.array([0.31, 0.77, 1.9], dtype=np.float64))


def main(dest):
    out = {}
    for N, max_d, n_qp in itertools.product((3, 7, 12), (1, 2), (2, 4)):
        geo = _seg_seg_reg_geometry(np.arange(N + 1) * H, A, max_d, n_qp)
        R = np.ascontiguousarray(geo["R"], dtype=np.float64)
        wu = np.ascontiguousarray(geo["wu_pow"], dtype=np.float64)
        out[f"reg/{N}/{max_d}/{n_qp}"] = _acc.seg_seg_reg_moments_bspline_swept(
            R, wu, KS
        )
        out[f"regek/{N}/{max_d}/{n_qp}"] = _acc.seg_seg_reg_moments_bspline_swept_ek(
            R, wu, KS, A_EK
        )
        out[f"stat/{N}/{max_d}"] = _acc.seg_seg_static_moments_bspline_uniform(
            H, A, N, max_d
        )
        out[f"statek/{N}/{max_d}"] = _acc.seg_seg_static_moments_bspline_uniform_ek(
            H, A, N, max_d, A_EK
        )
    # THE FINGERPRINT IS PART OF THE FIXTURE. Bit-identity is a claim about
    # THIS toolchain rebuilding the same source, not a portable property: the
    # closed forms and the reg kernel come out with different last bits under
    # a different compiler and libm. A fixture banked on Linux/GCC failed on
    # macOS in CI, correctly. The gate skips unless the fingerprint matches.
    out["_fingerprint"] = np.array(
        [
            platform.system(),
            platform.machine(),
            platform.python_version(),
            np.__version__,
        ],
        dtype=object,
    )
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print(f"banked {len(out) - 1} arrays to {dest} for {out['_fingerprint']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/data/same_edge_oracle_968.npz")
