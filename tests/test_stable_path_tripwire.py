"""The stable-path tripwire: flags off, every printout byte-identical to main's.

The feature-tier contract has one global rule: with every experimental flag
OFF (`momwire._features`), the stable path prints what main printed. The
flags live inside existing files, so nothing structural enforces that; this
test does. It clears every ``MOMWIRE_X_*`` variable and every alias the
registry knows, then renders the two golden corpora `scripts/golden_printouts.py`
writes, through the entry EZNEC launches (`momwire.dropin.nec5._shell.render`):

* the NEC-5 corpus (tests/fixtures/golden_printouts: bspline and razor-2p
  through the NEC-5 slot), and
* the stable-path corpus (tests/fixtures/stable_path_goldens: sin42 and bs42
  through the NEC-4.2 slot, Sinusoidal-Galerkin through both slots, more
  finite grounds and a buried-radial hub for bspline and razor-2p).

Regime, as the generator pins it: one BLAS/OpenMP thread, the Sommerfeld disk
store off (tests/conftest.py), AVX-512 off (the same), and the accelerator
variant the Linux default lane loads, avx2.

What is compared follows `tests/test_golden_printouts.py`, the existing rule,
and adds no tolerance: on the stack that wrote a corpus (CPU, variant, numpy,
scipy, BLAS kernel, as its manifest records them) every byte; on any other
Linux avx2 stack, every byte after `undefined_blanked` on both sides, which
blanks only the cells the arithmetic does not define (momwire#578's boundary).
The masks are the generator's: line 2's VERSION (the NEC-5 corpus masks the
whole stamp, so line 2 is checked here field by field instead), and in the
NEC-4.2 slot its wall-clock readings.

Linux x86_64 avx2 only: no corpus was written on macOS, Windows or sse2, and
`test_golden_printouts.py` records that those stacks need their own
measurement first.
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import pytest

from golden_compare import undefined_blanked

import momwire
from momwire import _exact_kernel_series, _features

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "golden_printouts", ROOT / "scripts" / "golden_printouts.py"
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

NEC5_MANIFEST = json.loads(G.MANIFEST.read_text())
STABLE_MANIFEST = json.loads(G.STABLE_MANIFEST.read_text())

# The variant the Linux default lane loads, and the one both corpora record.
VARIANT = "avx2"
assert NEC5_MANIFEST["regime"]["accelerator_variant"] == VARIANT
assert STABLE_MANIFEST["regime"]["accelerator_variant"] == VARIANT

_WHERE = (sys.platform, platform.machine(), momwire.accelerator_variant)
pytestmark = [
    pytest.mark.xdist_group("stable_path_tripwire"),
    pytest.mark.skipif(
        _WHERE[:2] != ("linux", "x86_64") or _WHERE[2] != VARIANT,
        reason=(
            "the stable-path tripwire runs on Linux x86_64 with the avx2 "
            f"accelerator (the default lane); this is {_WHERE[0]} {_WHERE[1]} "
            f"{_WHERE[2]}, on which no golden corpus was written"
        ),
    ),
]

NEC5 = "nec5-corpus"
STABLE = "stable"
CASES = (
    *((NEC5, deck, "nec5", basis) for deck, basis in G.CASES),
    *((STABLE, deck, dialect, basis) for deck, dialect, basis in G.STABLE_CASES),
)
REGENERATE = {
    NEC5: "python scripts/golden_printouts.py",
    STABLE: "python scripts/golden_printouts.py --corpus stable",
}

# What has to match for the every-byte claim (test_golden_printouts' rule).
_STACK = ("system", "machine", "cpu", "accelerator_variant", "numpy", "scipy")


def _stack(regime: dict) -> tuple:
    blas = sorted(
        (p.get("internal_api"), p.get("version"), p.get("architecture"))
        for p in regime["threadpools_during_render"]
        if p.get("user_api") == "blas"
    )
    return (*(regime.get(k) for k in _STACK), tuple(blas))


def _flag_envs() -> set[str]:
    names = {v for v in os.environ if v.startswith(_features.ENV_PREFIX)}
    for f in _features.FEATURES.values():
        names.update((f.env, *f.aliases))
    return names


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for name in _flag_envs():
        monkeypatch.delenv(name, raising=False)
    assert _features.active() == []
    # Read once, at import: a variable cleared now cannot turn it off.
    assert not _exact_kernel_series.ENABLED, (
        "exact_kernel_series was on when momwire was imported; unset "
        "MOMWIRE_X_EXACT_KERNEL_SERIES (and MOMWIRE_EXACT_KERNEL_SERIES) and rerun"
    )


@pytest.fixture(scope="module")
def current_stack() -> tuple:
    return _stack(G.regime())


def _golden(corpus: str, deck: str, dialect: str, basis: str) -> str:
    if corpus == NEC5:
        path = G.golden_path(deck, basis)
    else:
        path = G.stable_golden_path(deck, dialect, basis)
    return path.read_bytes().decode("latin-1")


def _render(corpus: str, deck: str, dialect: str, basis: str) -> str:
    if corpus == STABLE:
        return G.render_stable(deck, dialect, basis)
    raw = G.render_raw(deck, basis, dialect)
    if "NEC ERROR" in raw:
        raise AssertionError(f"{deck} [{basis}] refused")
    fields = G.stamp_fields(raw)
    # The NEC-5 corpus masks line 2 whole, so the flags-off shape of the
    # stamp is held here: exactly momwire <version> <basis> <variant>.
    if len(fields) != 4 or fields[0] != "momwire" or fields[2:] != [basis, VARIANT]:
        raise _Line2(raw.split("\n")[1])
    return G.mask_stamp(raw, basis)


class _Line2(Exception):
    """Line 2 of a NEC-5 corpus render is not the flags-off stamp."""


def _dump(corpus: str, deck: str, dialect: str, basis: str, text: str) -> None:
    root = os.environ.get("MOMWIRE_403_DUMP")
    if not root:
        return
    name = f"{dialect}-{basis}__" + deck.removesuffix(".nec").replace("/", "__")
    path = Path(root) / "stable_path_tripwire" / corpus / f"{name}.out"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("latin-1"))


def mismatch(
    corpus: str, deck: str, dialect: str, basis: str, current_stack: tuple
) -> str | None:
    """None when the case renders its golden; else the failure text: the
    first differing lines as a unified diff, and how to regenerate."""
    golden = _golden(corpus, deck, dialect, basis)
    try:
        text = _render(corpus, deck, dialect, basis)
    except _Line2 as exc:
        return (
            f"{deck} [{dialect} {basis}]: line 2 is {exc.args[0]!r}, not the "
            f"flags-off stamp ' momwire <version> {basis} {VARIANT}'"
        )
    if text == golden:
        return None
    manifest = NEC5_MANIFEST if corpus == NEC5 else STABLE_MANIFEST
    same_stack = current_stack == _stack(manifest["regime"])
    want, got = golden, text
    if not same_stack:
        want, got = undefined_blanked(golden), undefined_blanked(text)
        if want == got:
            return None
    _dump(corpus, deck, dialect, basis, text)
    diff = list(
        difflib.unified_diff(
            want.split("\n"),
            got.split("\n"),
            "golden",
            "rendered (every MOMWIRE_X_ flag off)",
            n=1,
            lineterm="",
        )
    )
    shown = "\n".join(diff[:24]) + ("\n..." if len(diff) > 24 else "")
    claim = "every byte" if same_stack else "every defined cell"
    return (
        f"{deck} [{dialect} {basis}] moved on the stable path ({claim}):\n"
        f"{shown}\n\n"
        f"If this move is intended, regenerate with `{REGENERATE[corpus]}` "
        "and state the measured ΔZ in the PR."
    )


def test_the_stable_manifest_is_the_corpus():
    """Every stable case has a golden, the manifest names exactly those, and
    the decks and printouts are the ones it recorded."""
    named = [(e["deck"], e["dialect"], e["basis"]) for e in STABLE_MANIFEST["entries"]]
    assert named == list(G.STABLE_CASES)
    on_disk = sorted(
        p.relative_to(G.STABLE).as_posix() for p in G.STABLE.glob("*/*.out")
    )
    assert on_disk == sorted(e["golden"] for e in STABLE_MANIFEST["entries"])
    for e in STABLE_MANIFEST["entries"]:
        assert G._sha((G.FIXTURES / e["deck"]).read_bytes()) == e["deck_sha256"]
        golden = (G.STABLE / e["golden"]).read_bytes()
        assert G._sha(golden) == e["printout_sha256"], e["golden"]


@pytest.mark.parametrize(
    ("corpus", "deck", "dialect", "basis"),
    CASES,
    ids=[f"{c}:{d}[{dl}-{b}]" for c, d, dl, b in CASES],
)
def test_flags_off_printout_is_byte_identical(
    corpus, deck, dialect, basis, current_stack
):
    failure = mismatch(corpus, deck, dialect, basis, current_stack)
    if failure is not None:
        pytest.fail(failure, pytrace=False)


# The self-test: one Python-kind feature on, one deck. The tripwire must see
# it, on line 2 (the feature stamp), whatever the feature does to the numbers.
_SELF_TEST = ("eznec_nec42/0228_dipole-10m-free.nec", "nec4", "bspline")


def test_the_tripwire_fires_when_a_flag_is_on(monkeypatch, current_stack):
    assert _features.FEATURES["offset_reuse"].kind == "python"
    assert (*_SELF_TEST[1:],) in {(dl, b) for _, dl, b in G.STABLE_CASES}
    deck, dialect, basis = _SELF_TEST
    assert mismatch(STABLE, deck, dialect, basis, current_stack) is None
    monkeypatch.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    assert _features.active() == ["offset_reuse"]
    golden_line2 = _golden(STABLE, deck, dialect, basis).split("\n")[1]
    line2 = G.render_stable(deck, dialect, basis).split("\n")[1]
    assert line2 != golden_line2
    assert line2.startswith(golden_line2 + " +offset_reuse"), line2
    failure = mismatch(STABLE, deck, dialect, basis, current_stack)
    assert failure is not None and "+offset_reuse" in failure, failure
    assert REGENERATE[STABLE] in failure


def test_the_nec5_corpus_line2_check_fires_when_a_flag_is_on(
    monkeypatch, current_stack
):
    """The NEC-5 corpus masks its whole stamp, so its flags-off line-2 check
    is what sees a flag there."""
    deck, basis = G.CASES[0]
    monkeypatch.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    failure = mismatch(NEC5, deck, "nec5", basis, current_stack)
    assert failure is not None and "+offset_reuse" in failure, failure
