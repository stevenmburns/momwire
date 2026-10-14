"""The T1 assemblers refuse per-basis arrays and an M0 that do not fit (#1475).

`razor_assemble_t1` and `razor_assemble_t1_weighted` index `h_a`, `h_b`,
`fall_a` and `fall_b` by basis and M0 by wing segment. Before #1475 neither
length was checked: a short array (an empty one included) or an M0 with fewer
columns than M1 was accepted, the kernel read past its end, and T1 came back
holding whatever was there, different from one run to the next. Production
callers always pass consistent arrays, so no solve moved; the read was still
reachable through the public binding.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel

pytestmark = pytest.mark.skipif(
    _accel.acc is None or not hasattr(_accel.acc, "razor_assemble_t1_weighted"),
    reason="no compiled razor T1 assemblers",
)

N_U, N_SEG, N_BASIS, N_PATH = 6, 9, 7, 2
N_OBS = N_U  # one moment row per observer: no obs_row


def _arrays():
    rng = np.random.default_rng(1475)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)  # noqa: E731
    return {
        "M0": cplx(N_U, N_SEG),
        "M1": cplx(N_U, N_SEG),
        "s_a": rng.integers(0, N_SEG, N_BASIS),
        "s_b": rng.integers(0, N_SEG, N_BASIS),
        "h_a": rng.uniform(0.1, 1, N_BASIS),
        "h_b": rng.uniform(0.1, 1, N_BASIS),
        "fall_a": rng.random(N_BASIS) < 0.5,
        "fall_b": rng.random(N_BASIS) < 0.5,
        "t_out": rng.standard_normal((N_OBS, 3)),
        "td_a": rng.standard_normal((3, N_BASIS)),
        "td_b": rng.standard_normal((3, N_BASIS)),
        "sig_a": rng.choice([-1.0, 1.0], N_BASIS),
        "sig_b": rng.choice([-1.0, 1.0], N_BASIS),
        "obs_c": rng.standard_normal((N_OBS, 3)) - [0, 0, 3],
        "src_c": rng.standard_normal((N_SEG, 3)),
        "src_t": rng.standard_normal((N_SEG, 3)),
        "w": rng.uniform(0, 1, N_OBS),
    }


def _plain(a):
    return _accel.acc.razor_assemble_t1(
        a["M0"], a["M1"], a["s_a"], a["s_b"], a["h_a"], a["h_b"],
        a["fall_a"], a["fall_b"], a["t_out"], a["td_a"], a["td_b"], a["w"],
        N_PATH,
    )  # fmt: skip


def _weighted(a):
    return _accel.acc.razor_assemble_t1_weighted(
        a["M0"], a["M1"], a["s_a"], a["s_b"], a["h_a"], a["h_b"],
        a["fall_a"], a["fall_b"], a["sig_a"], a["sig_b"], a["obs_c"],
        a["t_out"], a["src_c"], a["src_t"], a["w"], N_PATH, 0, 13 - 0.4j, 0.0,
        1.0 + 0j,
    )  # fmt: skip


ASSEMBLERS = pytest.mark.parametrize(
    "assemble", [_plain, _weighted], ids=["unweighted", "weighted"]
)


@ASSEMBLERS
def test_consistent_arrays_still_assemble(assemble):
    a = _arrays()
    got = assemble(a)
    assert got.shape == (N_OBS // N_PATH, N_BASIS)
    assert np.array_equal(got, assemble(_arrays()))


@ASSEMBLERS
@pytest.mark.parametrize("name", ["h_a", "h_b", "fall_a", "fall_b"])
@pytest.mark.parametrize("length", [0, N_BASIS - 1, N_BASIS + 1])
def test_a_per_basis_array_of_the_wrong_length_is_refused(assemble, name, length):
    a = _arrays()
    a[name] = np.resize(a[name], length)
    with pytest.raises(RuntimeError, match="per-basis arrays disagree on n_basis"):
        assemble(a)


@ASSEMBLERS
@pytest.mark.parametrize("n_cols", [1, N_SEG - 1, N_SEG + 1])
def test_an_m0_whose_columns_are_not_m1s_is_refused(assemble, n_cols):
    a = _arrays()
    a["M0"] = np.resize(a["M0"], (N_U, n_cols))
    with pytest.raises(RuntimeError, match="M0 and M1 disagree on columns"):
        assemble(a)
