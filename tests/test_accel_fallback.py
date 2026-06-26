"""The accelerator loader (`momwire._accel`) must warn — not silently degrade —
when a *built* C++ extension fails to import, while staying quiet when it was
never built. Regression guard for the fallback that used to be invisible: on
Linux a static-TLS clash (an old pynec-accel vendoring its own libgomp), on
macOS a missing Homebrew libomp — either way momwire would drop to the slow
pure-Python path with no signal at all. The warning is platform-aware, so this
also pins that the right remediation hint is shown for each OS."""

import sys
import warnings

import momwire
from momwire import _accel


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
