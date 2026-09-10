"""The CPU-feature chooser in `momwire._accel` (momwire#1032).

`import momwire` used to kill the interpreter outright on a CPU without AVX2 —
STATUS_ILLEGAL_INSTRUCTION (0xC000001D) on Windows, SIGILL on Linux, no
traceback, no output. The wheels are compiled `/arch:AVX2` (MSVC) and
`-mavx2 -mfma` (otherwise), and `_load()` guarded the import with
`except ImportError`; a hardware fault is not an exception, so the pure-Python
fallback that exists for a *missing* extension never ran. A user on the QRZ
thread lost three days to it on a machine where the server simply exited.

The fix is the double build: both extensions are compiled twice from the same
sources, and the loader picks by CPU feature BEFORE importing either. So the
property under test is not "falls back" — on such a CPU momwire is still
accelerated, by the baseline build — it is **never imports the AVX2 one**.

Nothing here can be integration-tested on this fleet (every box has AVX2), so
these are unit tests around the decision, plus a real parse of real
`/proc/cpuinfo`-shaped files, plus one test that the two builds on this box
actually differ.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import pytest

import momwire
from momwire import _accel

# ---------------------------------------------------------------------------
# The chain: which variants are tried, in what order
# ---------------------------------------------------------------------------


def test_an_avx2_cpu_prefers_avx2_but_can_fall_back(monkeypatch):
    monkeypatch.delenv(_accel._FORCE_VARIANT_ENV, raising=False)
    assert _accel._variants_to_try(True) == (_accel._AVX2, _accel._SSE2, _accel._LEGACY)


@pytest.mark.parametrize("verdict", [False, None])
def test_avx2_is_never_tried_when_it_is_not_known_to_be_safe(monkeypatch, verdict):
    """The whole fix, as a property of the chain.

    `False` is the user's CPU. `None` is every CPU we could not classify, and
    it gets the SAME answer on purpose: guessing upward kills the interpreter
    with no output, guessing downward costs speed. Those are not comparable
    mistakes, so unknown is never optimistic.
    """
    monkeypatch.delenv(_accel._FORCE_VARIANT_ENV, raising=False)
    chain = _accel._variants_to_try(verdict)

    assert _accel._AVX2 not in chain
    assert chain == (_accel._SSE2, _accel._LEGACY)


def test_the_force_override_pins_one_variant(monkeypatch):
    """Test-only knob, and the reason the baseline build can be exercised at
    all on a fleet where every box would otherwise choose AVX2."""
    for name, expected in (
        ("avx2", _accel._AVX2),
        ("sse2", _accel._SSE2),
        ("legacy", _accel._LEGACY),
    ):
        monkeypatch.setenv(_accel._FORCE_VARIANT_ENV, name)
        assert _accel._variants_to_try(True) == (expected,)


def test_a_bogus_force_value_warns_and_chooses_normally(monkeypatch):
    monkeypatch.setenv(_accel._FORCE_VARIANT_ENV, "avx512")
    with pytest.warns(RuntimeWarning, match="avx512"):
        chain = _accel._variants_to_try(False)
    assert chain == (_accel._SSE2, _accel._LEGACY)


# ---------------------------------------------------------------------------
# What _load() does with the chain
# ---------------------------------------------------------------------------


def _recording_loader(monkeypatch, present: str):
    """Record which module names `_load()` ASKS for, and serve only `present`.

    Imports NOTHING — not the other variant, and not the served one either.

    Two 34 MB extensions built from identical sources, each with its own OpenMP
    state, in one interpreter is not a configuration the product ever creates:
    the chooser picks once, at import. Creating it in a test destabilised the
    whole suite — cancellation and abort tests that pass in isolation began
    failing under xdist, a different set on each run. Measured twice, because
    the first fix was not enough: dropping the *other* variant still left this
    helper importing the *served* one, and the suite stayed red until it
    imported neither. With this file deselected the same lane is green.

    So the decision is asserted here, on a stub, and the real loading of a real
    extension is left to the subprocess test at the bottom, where a second
    variant cannot reach this interpreter.
    """
    asked: list[str] = []

    class _Stub:
        """Not the real extension: importing it is the thing to avoid."""

        def __init__(self, name):
            self.__name__ = name

    def spy(base_name, suffix):
        asked.append(f"{base_name}{suffix}")
        if suffix != present:
            return None
        return _Stub(f"momwire.{base_name}{suffix}")

    monkeypatch.setattr(_accel, "_import_variant", spy)
    return asked


def test_an_unsupported_cpu_loads_the_baseline_not_the_avx2_build(monkeypatch):
    """On the user's machine momwire stays ACCELERATED — by the SSE2 build."""
    monkeypatch.delenv(_accel._FORCE_VARIANT_ENV, raising=False)
    monkeypatch.setattr(_accel, "_cpu_supports_extension", lambda: False)
    asked = _recording_loader(monkeypatch, "_sse2")

    _mod, loaded, variant = _accel._load()

    assert (loaded, variant) == (True, "sse2")
    assert "_accelerators_avx2" not in asked, (
        "the AVX2 build was imported on a CPU without AVX2"
    )


def test_an_avx2_cpu_loads_the_avx2_build(monkeypatch):
    monkeypatch.delenv(_accel._FORCE_VARIANT_ENV, raising=False)
    monkeypatch.setattr(_accel, "_cpu_supports_extension", lambda: True)
    asked = _recording_loader(monkeypatch, "_avx2")

    _mod, loaded, variant = _accel._load()

    assert (loaded, variant) == (True, "avx2")
    assert asked[0] == "_accelerators_avx2"


def test_require_accel_raises_when_nothing_in_the_chain_loads(monkeypatch):
    """A build/CI box that ends up unaccelerated should fail, not degrade.

    NOTE this is a NEW import-time meaning for the variable: before #1032 it
    was read only by setup.py, at build time.
    """
    # The chain must come from the CPU verdict, not from an override the
    # surrounding lane happens to have set — this test names what it tried.
    monkeypatch.delenv(_accel._FORCE_VARIANT_ENV, raising=False)
    monkeypatch.setattr(_accel, "_cpu_supports_extension", lambda: False)
    monkeypatch.setattr(_accel, "_import_variant", lambda *a: None)
    monkeypatch.setattr(_accel, "_extension_built", lambda *_: True)
    monkeypatch.setenv("MOMWIRE_REQUIRE_ACCEL", "1")

    with pytest.raises(RuntimeError) as exc:
        _accel._load()

    assert "MOMWIRE_REQUIRE_ACCEL" in str(exc.value)
    assert "sse2" in str(exc.value)  # names what it tried


def test_a_pure_python_install_stays_silent(monkeypatch):
    """Nothing built for this platform is the expected case, not a problem."""
    monkeypatch.delenv(_accel._FORCE_VARIANT_ENV, raising=False)
    monkeypatch.setattr(_accel, "_cpu_supports_extension", lambda: True)
    monkeypatch.setattr(_accel, "_import_variant", lambda *a: None)
    monkeypatch.setattr(_accel, "_extension_built", lambda *_: False)
    monkeypatch.delenv("MOMWIRE_REQUIRE_ACCEL", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod, loaded, variant = _accel._load()

    assert (mod, loaded, variant) == (None, False, None)
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


# ---------------------------------------------------------------------------
# The two extensions must come from the SAME variant
# ---------------------------------------------------------------------------


def test_the_companion_follows_the_chosen_variant():
    """An AVX2 `_near_interface_accel` beside an SSE2 `_accelerators` faults on
    exactly the CPU the split exists for — and from the module nobody is
    looking at. One decision, read by both.

    Compared as module NAMES built from the variant's suffix, not as a label
    against a name: `"legacy"` is the unsuffixed extension, so an `endswith`
    on the label is false there for a correctly-loaded module. That is the
    single-variant configuration macOS and every non-x86 build ship, and it is
    the one this box cannot choose on its own — it cost a red `test-macos` on
    the certification dispatch, which is the lane that exists to catch it.
    """
    from momwire import _near_interface as ni

    if momwire.accelerator_variant is None:
        pytest.skip("pure-Python install: no variant to agree about")
    suffix = dict(
        (label, sfx) for label, sfx in (_accel._AVX2, _accel._SSE2, _accel._LEGACY)
    )[momwire.accelerator_variant]
    assert _accel.acc.__name__ == f"momwire._accelerators{suffix}"
    if ni._nia is not None:
        assert ni._nia.__name__ == f"momwire._near_interface_accel{suffix}"


def test_the_historic_name_still_resolves():
    """`from momwire import _accelerators` is what this repo's own tests say,
    and what anything outside it would have said. It now means "the variant
    that loaded", which is the only honest meaning once there are two."""
    if momwire.accelerator_variant is None:
        pytest.skip("pure-Python install")
    from momwire import _accelerators

    assert _accelerators is _accel.acc


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------


def test_macos_carries_no_requirement(monkeypatch):
    """setup.py's darwin branch passes no -mavx2/-mfma on EITHER arch — it is
    the simple-pragmas port — so macOS is not affected at all, not merely
    arm64. A guard that answered False there would slow down working Macs."""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _accel._cpu_supports_extension() is None


@pytest.mark.parametrize("machine", ["aarch64", "arm64", "ppc64le", "s390x", "riscv64"])
def test_non_x86_carries_no_requirement(monkeypatch, machine):
    """setup.py never inspects `platform.machine()`, so its `-mavx2` branch is
    written for x86 but reached on any non-Windows non-macOS platform. On a CPU
    with no AVX2 to look for, absence must read as "not applicable"."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_accel.platform, "machine", lambda: machine)
    assert _accel._cpu_supports_extension() is None


HASWELL = (
    "processor\t: 0\nvendor_id\t: GenuineIntel\n"
    "flags\t\t: fpu vme de pse tsc msr pae sse2 avx avx2 fma bmi1 bmi2\n"
)
PRE_AVX2 = (
    "processor\t: 0\nvendor_id\t: GenuineIntel\n"
    "flags\t\t: fpu vme de pse tsc msr pae sse2 ssse3 sse4_1 sse4_2 avx\n"
)


@pytest.mark.parametrize(
    "text,expected",
    [
        (HASWELL, True),
        (PRE_AVX2, False),
        # AVX2 without FMA: setup.py passes -mfma too, so the pair is required.
        (HASWELL.replace(" fma", ""), False),
        # A file with no flag line at all answers "cannot tell", not "no".
        ("processor\t: 0\nvendor_id\t: GenuineIntel\n", None),
    ],
)
def test_the_linux_flag_parse(monkeypatch, tmp_path, text, expected):
    path = tmp_path / "cpuinfo"
    path.write_text(text)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_accel.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_accel, "_CPUINFO_PATH", str(path))
    assert _accel._cpu_supports_extension() is expected


def test_no_procfs_cannot_tell(monkeypatch, tmp_path):
    """A container, a BSD, a locked-down sandbox: answer None, take baseline."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_accel.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_accel, "_CPUINFO_PATH", str(tmp_path / "absent"))
    assert _accel._cpu_supports_extension() is None


def _explode():
    raise RuntimeError("boom")


def test_the_guard_never_raises(monkeypatch):
    """The one property that matters more than the answer: a guard that can
    raise out of `import momwire` is worse than the crash it prevents.

    `platform.machine()` is the call nobody would write a handler for, which is
    why it is the one used here. Two steps on purpose: first prove the stub
    really does raise (so the second assertion is not vacuous), then prove the
    guard swallows it and answers None.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_accel.platform, "machine", _explode)

    with pytest.raises(RuntimeError):
        _accel._cpu_supports_extension_uncaught()

    assert _accel._cpu_supports_extension() is None


# ---------------------------------------------------------------------------
# This box, for real
# ---------------------------------------------------------------------------


def test_this_box_is_answered_consistently():
    """Whatever this CPU is, the check and the load must agree.

    Skipped under the force override, which exists precisely to break this
    agreement: a forced-sse2 run on an AVX2 box is a CPU that says yes and a
    loader that was told to say no, and asserting they match there would fail
    the lane the override exists to enable.
    """
    if os.environ.get(_accel._FORCE_VARIANT_ENV):
        pytest.skip("the variant is pinned by the environment, not by this CPU")
    verdict = _accel._cpu_supports_extension()
    if verdict is True:
        assert momwire.accelerator_variant in ("avx2", None)
    else:
        assert momwire.accelerator_variant != "avx2"


def test_only_one_variant_is_ever_loaded_in_this_process():
    """The tripwire for the mistake this file made twice.

    The chooser picks once, at import. Two builds of the same extension live in
    one interpreter only if a TEST puts them there — identical symbols, two
    OpenMP runtimes, 68 MB — and when this file did that, cancellation and
    abort tests elsewhere in the suite began failing under xdist, a different
    set each run and every one of them green in isolation. Nothing asserted it,
    so it took two rounds of measurement to find. Now something asserts it.
    """
    loaded = sorted(n for n in sys.modules if n.startswith("momwire._accelerators_"))
    assert len(loaded) <= 1, (
        f"more than one accelerator variant is loaded in this interpreter: "
        f"{loaded}. A test is importing a variant the chooser did not pick; "
        f"use a stub for the decision and a subprocess for a real load."
    )


@pytest.mark.parametrize("variant", ["avx2", "sse2"])
def test_each_variant_imports_in_a_fresh_interpreter(variant):
    """Not a mock: start a real interpreter, force the variant, and confirm the
    extension it names actually loads and reports itself.

    This is the closest a box with AVX2 can get to the user's machine — it
    proves the SSE2 build is a real, loadable, distinct extension rather than a
    second copy of the AVX2 one under a different filename, which is precisely
    what a shared object-file tree would have produced.
    """
    if not _accel._extension_built(f"_{variant}"):
        pytest.skip(f"the {variant} variant is not built in this checkout")
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import momwire; print(momwire.accelerator_variant)",
        ],
        env={**os.environ, _accel._FORCE_VARIANT_ENV: variant},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == variant
