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
# momwire#769 — the quadrature ceiling routes to numpy instead of raising
# ---------------------------------------------------------------------------
#
# Before #769 the capped C++ pair kernels were simply called and raised
# `RuntimeError: n_qp > 8 not supported`. That turned a slow-but-correct answer
# into an unhandled exception on exactly the crossing/lossy-soil class that
# needs the order (#760), while #696 shipped a warning telling users to raise
# the very knob that crashed.
#
# The deck below is BENT on purpose. A straight wire has one edge, so since #759
# it never enters the off-edge kernel at all and would pass this file vacuously
# — that deck-dependence is half of what #769 is about.


def _bent(n_qp):
    wire = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    return BSplineSolver(
        wires=[wire],
        nsegs=12,
        wavelength=22.0,
        wire_radius=0.0005,
        degree=2,
        n_qp_pair=n_qp,
        feed_wire_index=0,
        feed_arclength=5.0,
    )


def test_max_n_qp_is_read_off_the_extension_not_respelled():
    """The Python guard and the C++ kernels must not be able to disagree.

    #762 lifts the ceiling by editing one `constexpr` in _accel_common.h; this
    is what makes that edit sufficient. If the export ever disappears, the
    getattr default silently takes over — so pin that the export exists.
    """
    from momwire import _accelerators

    assert _accelerators.BSPLINE_MAX_N_QP == _accel.MAX_N_QP


def test_serves_n_qp_is_a_predicate_about_the_ceiling():
    assert _accel.serves_n_qp(_accel.MAX_N_QP, "off-edge pair") is True
    with pytest.warns(RuntimeWarning, match="exceeds the accelerated"):
        assert _accel.serves_n_qp(_accel.MAX_N_QP + 1, "off-edge pair") is False


@pytest.mark.skipif(not _accel.LOADED, reason="no accelerator to fall back FROM")
def test_over_the_ceiling_solves_instead_of_raising_and_says_so():
    with pytest.warns(RuntimeWarning, match=r"exceeds the accelerated .*ceiling of 8"):
        Z, _ = _bent(_accel.MAX_N_QP + 8).compute_impedance()
    assert np.isfinite(Z.real) and np.isfinite(Z.imag)


@pytest.mark.skipif(not _accel.LOADED, reason="no accelerator to fall back FROM")
def test_at_the_ceiling_nothing_warns_and_the_fast_path_still_runs():
    """The guard must not cost the accelerated path anything at legal orders."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Z, _ = _bent(_accel.MAX_N_QP).compute_impedance()
    assert np.isfinite(Z.real)


@pytest.mark.skipif(not _accel.LOADED, reason="nothing to compare against")
def test_the_fallback_really_is_the_numpy_path(monkeypatch):
    """Falling back must land on the reference implementation, not somewhere
    else that merely fails to raise. #758's converged anchors were banked on
    the numpy path precisely because it reproduces the C++ one bit-identically
    below the ceiling, so above it the two must be the SAME computation."""
    over = _accel.MAX_N_QP + 8
    with pytest.warns(RuntimeWarning):
        Z_routed, _ = _bent(over).compute_impedance()

    # No `pytest.warns` here, and that is the point: with the accelerator off
    # the ceiling is not what moved the work, so this run is silent.
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    Z_numpy, _ = _bent(over).compute_impedance()

    assert abs(Z_routed - Z_numpy) < 1e-9, (Z_routed, Z_numpy)


@pytest.mark.skipif(not _accel.LOADED, reason="no accelerator to be ineligible for")
def test_no_warning_when_the_ceiling_was_not_the_reason(monkeypatch):
    """A caller already on numpy for another reason must not be told about a
    ceiling it never reached.

    This is not hypothetical: the repo's own complex-k and Sommerfeld truth
    references call `_seg_seg_full_moments_offedge` with n_qp of 12, 64 and 256
    precisely because they want the reference implementation. Warning there
    trains people to ignore the warning that matters.
    """
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _bent(_accel.MAX_N_QP + 8).compute_impedance()


def test_the_eligibility_flag_is_what_gates_the_warning():
    over = _accel.MAX_N_QP + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _accel.serves_n_qp(over, "off-edge pair", eligible=False) is False
    with pytest.warns(RuntimeWarning):
        assert _accel.serves_n_qp(over, "off-edge pair", eligible=True) is False
