"""The accelerator loader (`momwire._accel`) must warn — not silently degrade —
when a *built* C++ extension fails to import, while staying quiet when it was
never built. Regression guard for the fallback that used to be invisible: on
Linux a static-TLS clash (an old pynec-accel vendoring its own libgomp), on
macOS a missing Homebrew libomp — either way momwire would drop to the slow
pure-Python path with no signal at all. The warning is platform-aware, so this
also pins that the right remediation hint is shown for each OS."""

import sys
import warnings

import numpy as np
import pytest

import momwire
from momwire import BSplineSolver, _accel
from momwire import _bspline_kernels as _bk


def _force_import_failure(monkeypatch):
    """Make `from . import _accelerators` raise ImportError inside _load().

    A None entry in sys.modules makes the import fail, but `from package import
    sub` first checks for an already-bound attribute on the package — so the
    real (already-imported) module must be detached too. monkeypatch restores
    both after the test.
    """
    monkeypatch.setitem(sys.modules, "momwire._accelerators", None)
    monkeypatch.delattr(momwire, "_accelerators", raising=False)


def test_extension_is_built_in_this_install():
    # The test suite runs against a compiled wheel/editable install, so the
    # detector must see the extension; otherwise the warn-vs-quiet logic below
    # is testing the wrong branch.
    assert _accel._extension_built() is True


def test_clean_load_reports_accelerated():
    assert momwire.accelerated is True
    assert _accel.acc is not None


def _warn_message(monkeypatch) -> str:
    """Force the built-but-unloadable path and return the single RuntimeWarning."""
    _force_import_failure(monkeypatch)
    monkeypatch.setattr(_accel, "_extension_built", lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod, loaded = _accel._load()
    assert (mod, loaded) == (None, False)
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(msgs) == 1
    return msgs[0]


def test_built_but_unloadable_warns(monkeypatch):
    # Host-platform run: always names the fallback and the OS-appropriate hint.
    msg = _warn_message(monkeypatch)
    assert "pure-Python" in msg
    expected = "brew install libomp" if sys.platform == "darwin" else "static-TLS"
    assert expected in msg


def test_warning_hint_is_platform_specific(monkeypatch):
    # macOS -> brew/libomp hint, no Linux static-TLS noise; Linux -> the reverse.
    monkeypatch.setattr(sys, "platform", "darwin")
    mac = _warn_message(monkeypatch)
    assert "brew install libomp" in mac and "static-TLS" not in mac

    monkeypatch.setattr(sys, "platform", "linux")
    lin = _warn_message(monkeypatch)
    assert "static-TLS" in lin and "brew install libomp" not in lin


def test_the_windows_hint_names_the_runtime_windows_actually_needs(monkeypatch):
    # momwire#737: Windows used to fall through to the Linux branch and advise
    # `apt install libgomp1`, on a box with no apt and a different missing
    # file. The extensions are built with /openmp:llvm, so what is missing is
    # LLVM's libomp140.x86_64.dll — the name a user has to be able to search
    # for — and never the vcomp140.dll an MSVC build would suggest.
    monkeypatch.setattr(sys, "platform", "win32")
    win = _warn_message(monkeypatch)
    assert "libomp140.x86_64.dll" in win
    assert "apt install" not in win and "brew install" not in win


def test_not_built_is_silent(monkeypatch):
    # Same import failure, but the extension was never built -> pure-Python is
    # the expected, unremarkable outcome; no warning.
    _force_import_failure(monkeypatch)
    monkeypatch.setattr(_accel, "_extension_built", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod, loaded = _accel._load()

    assert (mod, loaded) == (None, False)
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


# ---------------------------------------------------------------------------
# momwire#769 / #762 — the quadrature ceilings
# ---------------------------------------------------------------------------
#
# Before #769 the capped C++ pair kernels were simply called and raised
# `RuntimeError: n_qp > 8 not supported`. That turned a slow-but-correct answer
# into an unhandled exception on exactly the crossing/lossy-soil class that
# needs the order (#760), while #696 shipped a warning telling users to raise
# the very knob that crashed.
#
# #762 then TILED the six off-edge kernels, so that path has no ceiling left at
# all and high order runs accelerated. What remains is the SAME-EDGE reg kernel,
# which #762 did not tile — and which #769 missed entirely, because its refusal
# is worded differently ("n_qp^2 must be <= 64") from the off-edge kernels'.
# So the fallback machinery is now exercised through `n_qp_pair_same_edge`.
#
# The off-edge deck below is BENT on purpose. A straight wire has one edge, so
# since #759 it never enters the off-edge kernel at all and would pass
# vacuously — that deck-dependence is half of what #769 was about.

_STRAIGHT = np.array([[0.0, -5.291, 0.0], [0.0, 5.291, 0.0]])
_BENT = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])


def _solver(wire, **knobs):
    return BSplineSolver(
        wires=[wire],
        nsegs=12,
        wavelength=22.0,
        wire_radius=0.0005,
        degree=2,
        feed_wire_index=0,
        feed_arclength=5.0,
        **knobs,
    )


def test_both_ceilings_are_read_off_the_extension_not_respelled():
    """The Python guard and the C++ kernels must not be able to disagree.

    Two constants because they are two kernels with two different limits.
    Collapsing them is exactly how #769 came to miss the same-edge path.
    """
    from momwire import _accelerators

    assert _accelerators.BSPLINE_MAX_N_QP == _accel.MAX_N_QP
    assert _accelerators.BSPLINE_SAME_EDGE_MAX_N_QP == _accel.SAME_EDGE_MAX_N_QP


def test_the_offedge_ceiling_is_gone_since_762():
    """#762 tiled all six off-edge kernels, so the guard must never divert."""
    assert _accel.MAX_N_QP > 10**6


def test_serves_n_qp_is_a_predicate_about_a_named_ceiling():
    assert _accel.serves_n_qp(8, "off-edge pair", cap=8) is True
    with pytest.warns(RuntimeWarning, match="ceiling of 8"):
        assert _accel.serves_n_qp(9, "off-edge pair", cap=8) is False


def test_the_eligibility_flag_is_what_gates_the_warning():
    """A caller already on numpy for another reason must not be told about a
    ceiling it never reached.

    Not hypothetical: the repo's own complex-k and Sommerfeld truth references
    ask for n_qp of 12, 64 and 256 precisely because they want the reference
    implementation. Warning there trains people to ignore the warning that
    matters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _accel.serves_n_qp(9, "x", eligible=False, cap=8) is False
    with pytest.warns(RuntimeWarning):
        assert _accel.serves_n_qp(9, "x", eligible=True, cap=8) is False


@pytest.mark.skipif(not _accel.LOADED, reason="no accelerator to run high order ON")
@pytest.mark.parametrize("n_qp", [9, 16, 33, 64])
def test_offedge_high_order_runs_accelerated_and_never_falls_back(n_qp):
    """#762's headline. 9 and 33 are deliberate: they are the first orders past
    a whole number of 64-wide chunks (81 and 1089 pairs), so they exercise the
    short trailing chunk rather than only exact multiples."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Z, _ = _solver(_BENT, n_qp_pair=n_qp).compute_impedance()
    assert np.isfinite(Z.real) and np.isfinite(Z.imag)


@pytest.mark.skipif(not _accel.LOADED, reason="nothing to compare against")
def test_tiled_offedge_agrees_with_numpy_across_the_chunk_boundary(monkeypatch):
    """Tiling is a reassociation, so C++ and numpy are close rather than equal;
    what matters is that crossing the 64-pair boundary does not change that.
    n_qp=8 is one full chunk, n_qp=9 is two."""
    for n_qp in (8, 9):
        Z_acc, _ = _solver(_BENT, n_qp_pair=n_qp).compute_impedance()
        with monkeypatch.context() as mp:
            mp.setattr(_bk, "_HAVE_BSPLINE_ACCEL", False)
            mp.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
            Z_np, _ = _solver(_BENT, n_qp_pair=n_qp).compute_impedance()
        assert abs(Z_acc - Z_np) < 1e-9, (n_qp, Z_acc, Z_np)


@pytest.mark.skipif(not _accel.LOADED, reason="no accelerator to fall back FROM")
def test_same_edge_over_its_ceiling_solves_instead_of_raising_and_says_so():
    """The site #769 missed. Before this it raised
    `RuntimeError: n_qp too large (n_qp^2 must be <= 64)`."""
    over = _accel.SAME_EDGE_MAX_N_QP + 8
    with pytest.warns(RuntimeWarning, match="same-edge reg"):
        Z, _ = _solver(_STRAIGHT, n_qp_pair_same_edge=over).compute_impedance()
    assert np.isfinite(Z.real) and np.isfinite(Z.imag)


@pytest.mark.skipif(not _accel.LOADED, reason="no accelerator")
def test_same_edge_at_its_ceiling_nothing_warns_and_the_fast_path_runs():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Z, _ = _solver(
            _STRAIGHT, n_qp_pair_same_edge=_accel.SAME_EDGE_MAX_N_QP
        ).compute_impedance()
    assert np.isfinite(Z.real)


@pytest.mark.skipif(not _accel.LOADED, reason="nothing to compare against")
def test_the_same_edge_fallback_really_is_the_numpy_path(monkeypatch):
    """Falling back must land on the reference implementation, not somewhere
    else that merely fails to raise."""
    over = _accel.SAME_EDGE_MAX_N_QP + 8
    with pytest.warns(RuntimeWarning):
        Z_routed, _ = _solver(_STRAIGHT, n_qp_pair_same_edge=over).compute_impedance()

    # No `pytest.warns` here, and that is the point: with the accelerator off
    # the ceiling is not what moved the work, so this run is silent.
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_REG_SWEPT_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_REG_SWEPT_EK_ACCEL", False)
    Z_numpy, _ = _solver(_STRAIGHT, n_qp_pair_same_edge=over).compute_impedance()

    assert abs(Z_routed - Z_numpy) < 1e-9, (Z_routed, Z_numpy)
