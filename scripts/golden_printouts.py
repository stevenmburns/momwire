"""Golden printouts: momwire's OWN NEC-5 printouts, frozen at one commit.

Step 0 of the python-free arc. Every gate the EZNEC seam had before this was
either a renderer round-trip of the licensed engine's captures, a layout check
with the solved cells masked, or momwire-against-momwire reproducibility in one
process. None of them pins momwire's own NUMBERS, so moving a unit of the
pipeline into C++ had nothing to be byte-identical TO. This is that thing: a
corpus of printouts rendered through the very entry EZNEC launches
(`momwire.eznec._shell.render`), for both bases a drop-in launcher serves
(``bspline``, the `momwire-nec5` default, and ``razor-2p``), over the NEC-5
fixture decks the seam serves in about a second or less.

    python scripts/golden_printouts.py            # rewrite tests/fixtures/golden_printouts/
    python scripts/golden_printouts.py --check    # render, compare, list every moved cell

A second corpus, the STABLE-PATH one (``--corpus stable``, written to
tests/fixtures/stable_path_goldens/), widens the same record past the NEC-5
seam's two bases: the NEC-4.2 slot on the point-matched sinusoidal (sin42)
and B-spline (bs42) bases, Sinusoidal-Galerkin (a Python engine) on both
slots, more finite grounds for bspline and razor-2p, and a buried-radial
deck. With the NEC-5 corpus it is what `tests/test_stable_path_tripwire.py`
holds byte-identical with every experimental flag off (`momwire._features`):

    python scripts/golden_printouts.py --corpus stable           # rewrite it
    python scripts/golden_printouts.py --corpus stable --check   # compare

The regime is pinned, because the numbers depend on it. BLAS and OpenMP run
ONE thread (`threadpoolctl`) for every render, here and in the test: OpenBLAS's
one-thread LU rounds differently from its multi-thread path (the reason
`serve/_pool.py` floors the pool at two), and one thread is what an xdist test
worker runs. The Sommerfeld disk store is off (`MOMWIRE_SOMM_CACHE=0`), as the
test session has it. The manifest records the commit, the accelerator lane,
the BLAS library and its kernel architecture, numpy/scipy, and the machine.

Line 2 of a printout is the engine stamp, which names the momwire version and
lane; it is masked (in every run of a multi-run printout, where it repeats),
and nothing else is. The timing cells (``FILL=``, ``FACTOR=``, ``RUN TIME``)
are not masked because this seam prints them as fixed zeros; a printout that
starts timing for real fails here, which is the point.

Regenerate ONLY at a commit whose printouts are meant to move, and review the
``--check`` listing before you do: the corpus is a record of what the Python
pipeline printed, and rewriting it to make a red test green deletes the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = FIXTURES / "golden_printouts"
MANIFEST = GOLDEN / "manifest.json"

BASES = ("bspline", "razor-2p")

# Deck paths relative to tests/fixtures: NEC-5 fixture decks the seam serves
# on both bases, chosen for cost. Measured on one core (Haswell i7-4770K,
# Sommerfeld disk store off): every case renders in under 2 s, except that the
# first elevated-wire-over-Sommerfeld deck in a process builds a ~21 s table
# the later ones reuse (the test builds it in a module fixture), and the
# corpus renders in ~30 s. No printout is over ~150 kB a basis. Left out: the
# eight decks the seam refuses; near-duplicates of a family already here; and
# decks that each paid a cold table of their own in a one-process sweep -- the
# 0011/0029/0030 coax dipoles and 0033/0034 elevated radials (~21 s), the
# buried and crossing decks (~17 s a basis), both grazing decks (~8 s), 0192
# and dipole-ne1-ground (~8 s), the WA7ARK ground rod (~6 s). Not covered,
# then: buried wires, the exact ring kernel (no deck here has a segment shorter
# than three radii) and NX chains.
DECKS = (
    "eznec/decks/0000_cardioid-l-network-feed.nec",
    "eznec/decks/0002_4-square-array-w-feed-system.nec",
    "eznec/decks/0003_4-square-array-w-feed-system.nec",
    "eznec/decks/0014_network-connection-test.nec",
    "eznec/decks/0017_network-connection-test.nec",
    "eznec/decks/0019_vertical-over-real-ground.nec",
    "eznec/decks/0023_4-square-array-l-ntwk-feed.nec",
    "eznec/decks/0025_4-sq-l-ntwrk-z-match.nec",
    "eznec/decks/0026_cardioid-with-feed-system.nec",
    "eznec/decks/0027_cardioid-with-feed-system.nec",
    "eznec/decks/0028_17-10m-log-per-arrl-ant-book.nec",
    "eznec/decks/0031_40-meter-four-square-array.nec",
    "eznec/decks/0032_cardioid.nec",
    "eznec/decks/0036_dipole-in-free-space.nec",
    "eznec/decks/0037_dipole-in-free-space.nec",
    "eznec/decks/0043_vertical-over-real-ground.nec",
    "eznec/decks/0063_15m-quad-ant-book-p-12-2.nec",
    "eznec/decks/0064_15m-quad-ant-book-p-12-2.nec",
    "eznec/decks/0067_nbs-yagi-ant-book-p-18-7.nec",
    "eznec/decks/0068_nbs-yagi-ant-book-p-18-7.nec",
    "eznec/decks/0069_w8jk-with-0-1-wavelength-sp.nec",
    "eznec/decks/0070_w8jk-with-0-1-wavelength-sp.nec",
    "eznec/decks/0075_field-day-special-jun-84-qst.nec",
    "eznec/decks/0077_field-day-special-jun-84-qst.nec",
    "eznec/decks/0079_k5rp.nec",
    "eznec/decks/0081_n4pc-loop-cq-dec-1990.nec",
    "eznec/decks/0083_n4pc-loop-cq-dec-1990.nec",
    "eznec/decks/0113_back-yard-dipole.nec",
    "eznec/decks/0115_back-yard-dipole.nec",
    "eznec/decks/0116_40-meter-four-square-array.nec",
    "eznec/decks/0120_cardioid-l-network-feed.nec",
    "eznec_endcode_1092/invvee_apex.nec",
    "eznec_endcode_1092/synthetic_ex_endcode.nec",
    "eznec_gyrator_1134/Cardioidmodnec2.nec",
    "eznec_gyrator_1134/gyrator_current.nec",
    "eznec_ld01_1085/synthetic_ld01.nec",
    "eznec_ld23_1088/synthetic_ld23.nec",
    "eznec_ld5_1082/synthetic_ld5.nec",
    "eznec_ld5_virtual_1096/synthetic_ld5_virtual.nec",
    "eznec_load_probes_1110/0183.nec",
    "eznec_load_probes_1110/0202.nec",
    "eznec_multi_run_1237/two-port-ground.nec",
    "eznec_multi_run_1237/two-port-xq.nec",
    "eznec_multi_run_1237/two-port.nec",
    "eznec_multi_voltage_network_1105/0187.nec",
    "eznec_multi_voltage_network_1105/V_0120_two_voltage_through_network.nec",
    "eznec_ne_spherical_1257/dipole-ne1-free.nec",
    "eznec_one_field_ge_1116/WA7ARK-OCF-Load-Xfmr-TL.nec",
    "eznec_one_field_ge_1116/WA7ARK-OCF-LoadOnly.nec",
    "eznec_probe_ex_1099/synthetic_probe_ex.nec",
    "eznec_probe_ex_1099/synthetic_probe_ex_noprobe.nec",
    "eznec_simnec_1230/dipole-ex5.nec",
    "eznec_simnec_1230/dipole-rp-norange.nec",
    "eznec_simnec_1230/dipole-rp-range.nec",
)

# Every (deck, basis) in the order the corpus is rendered, which is also the
# order the test renders it: in-process caches (the Sommerfeld tables) are
# warm or cold the same way on both sides.
CASES = tuple((deck, basis) for deck in DECKS for basis in BASES)

STAMP_MASK = "<engine stamp>"

# --- the stable-path corpus ---------------------------------------------------
#
# (deck, dialect, basis), deck relative to tests/fixtures. In-repo fixtures
# only: EZNEC's own example decks (eznec/decks), the decks EZNEC's NEC-4.2
# slot wrote and the probes written for that slot (eznec_nec42; the probes
# that reproduce a user's deck, p37 and p91-p96, are left out), and the
# buried B-spline printout fixtures. Every case renders in about a second or
# less on one core (Rocketlake, Sommerfeld disk store off).
STABLE = FIXTURES / "stable_path_goldens"
STABLE_MANIFEST = STABLE / "manifest.json"
VERSION_MASK = "<version>"
CLOCK_MASK = "<clock>"

# The NEC-4.2 slot prints the wall clock, as the licensed engine does: the
# FILL= and FACTOR= seconds under MATRIX TIMING and the RUN TIME trailer (the
# NEC-5 seam prints those as fixed zeros, and its corpus keeps them). A clock
# is not a number of the solve, so in this slot's printouts those three
# readings, and nothing else, are masked (`_timeless` in
# tests/test_eznec_nec42_printout_1295.py blanks the same two lines).
_CLOCK = re.compile(r"(FILL=|FACTOR=|^ RUN TIME =)( *)\d+\.\d+", re.MULTILINE)

# The NEC-4.2 slot, sin42 and bs42 alike: fifteen of the slot's seventeen
# captures (0226 and 0236 repeat 0225 and 0223), its two multi-source decks,
# and fifteen probes. Grounds: free, GN 1, GN 1 with GD (the reflection-
# coefficient far field), GN 2 and GN 3 (Sommerfeld), and a buried wire
# under each Sommerfeld form (0239 under GN 3, p9 under GN 2).
_NEC42_DECKS = (
    "eznec_nec42/0223_dipole-ex6.nec",
    "eznec_nec42/0224_dipole-ex0.nec",
    "eznec_nec42/0225_dipole-ld4.nec",
    "eznec_nec42/0227_dipole-ld5-copper.nec",
    "eznec_nec42/0228_dipole-10m-free.nec",
    "eznec_nec42/0229_dipole-10m-gn1.nec",
    "eznec_nec42/0230_dipole-10m-gn1-gd.nec",
    "eznec_nec42/0231_dipole-10m-gn2.nec",
    "eznec_nec42/0232_dipole-10m-gn3.nec",
    "eznec_nec42/0233_dipole-13p9mhz-xq.nec",
    "eznec_nec42/0234_cardioid-l-network.nec",
    "eznec_nec42/0235_foursquare-tl.nec",
    "eznec_nec42/0237_dipole-ex6-elevation.nec",
    "eznec_nec42/0238_dipole-ex6-3d.nec",
    "eznec_nec42/0239_dipole-buried-gn3.nec",
    "eznec_nec42/multisource/m1_cardioid-2-current-sources.nec",
    "eznec_nec42/multisource/m2_foursquare-4-current-sources.nec",
    "eznec_nec42/probes/p1_junction.nec",
    "eznec_nec42/probes/p2_three_frequencies.nec",
    "eznec_nec42/probes/p3_rlc_loads.nec",
    "eznec_nec42/probes/p4_near_field.nec",
    "eznec_nec42/probes/p5_gd_xq.nec",
    "eznec_nec42/probes/p6_voltage_source_on_network.nec",
    "eznec_nec42/probes/p7_gn2_7mhz.nec",
    "eznec_nec42/probes/p8_two_runs.nec",
    "eznec_nec42/probes/p9_above_and_buried_gn2.nec",
    "eznec_nec42/probes/p10_fat_dipole.nec",
    "eznec_nec42/probes/p13_ne1_gn3.nec",
    "eznec_nec42/probes/p15_ne1_gn2.nec",
    "eznec_nec42/probes/p16_rp_range_gn3.nec",
    "eznec_nec42/probes/p29_rp_range_gn1.nec",
    "eznec_nec42/probes/p31_rp_below_horizon_gn3.nec",
)

# Sinusoidal-Galerkin through the NEC-5 seam: free space, networks, loads,
# GN 1, bare GD (EZNEC's "MININEC type" ground: PEC currents and a
# reflection-coefficient far field) and GN 0 (Sommerfeld).
_SG_NEC5_DECKS = (
    "eznec/decks/0036_dipole-in-free-space.nec",
    "eznec/decks/0014_network-connection-test.nec",
    "eznec/decks/0063_15m-quad-ant-book-p-12-2.nec",
    "eznec/decks/0067_nbs-yagi-ant-book-p-18-7.nec",
    "eznec/decks/0019_vertical-over-real-ground.nec",
    "eznec/decks/0032_cardioid.nec",
    "eznec/decks/0000_cardioid-l-network-feed.nec",
    "eznec/decks/0020_vertical-over-real-ground.nec",
    "eznec/decks/0084_n4pc-loop-cq-dec-1990.nec",
    "eznec/decks/0079_k5rp.nec",
    "eznec/decks/0113_back-yard-dipole.nec",
    "eznec/decks/0059_back-yard-inverted-vee.nec",
)

# Sinusoidal-Galerkin in the NEC-4.2 slot, one deck per ground form.
_SG_NEC42_DECKS = (
    "eznec_nec42/0229_dipole-10m-gn1.nec",
    "eznec_nec42/0230_dipole-10m-gn1-gd.nec",
    "eznec_nec42/0231_dipole-10m-gn2.nec",
    "eznec_nec42/0232_dipole-10m-gn3.nec",
)

# bspline and razor-2p beyond the NEC-5 corpus: two more bare-GD
# (reflection-coefficient) decks, two more GN 0 (Sommerfeld) decks, and one
# small buried-radial deck (an eight-radial hub), which the NEC-5 corpus has
# none of.
_NEC5_EXTRA_DECKS = (
    "eznec/decks/0020_vertical-over-real-ground.nec",
    "eznec/decks/0084_n4pc-loop-cq-dec-1990.nec",
    "eznec/decks/0059_back-yard-inverted-vee.nec",
    "eznec/decks/0080_k5rp.nec",
    "bspline_printout_buried/ab_hub8.nec",
)

STABLE_CASES = (
    *((d, "nec4", "sinusoidal") for d in _NEC42_DECKS),
    *((d, "nec4", "bspline") for d in _NEC42_DECKS),
    *((d, "nec5", "sinusoidal-galerkin") for d in _SG_NEC5_DECKS),
    *((d, "nec4", "sinusoidal-galerkin") for d in _SG_NEC42_DECKS),
    *((d, "nec5", b) for d in _NEC5_EXTRA_DECKS for b in BASES),
)


def stable_golden_path(deck: str, dialect: str, basis: str) -> Path:
    name = deck.removesuffix(".nec").replace("/", "__") + ".out"
    return STABLE / f"{dialect}-{basis}" / name


def render_raw(deck: str, basis: str, dialect: str = "nec5") -> str:
    """One deck's printout as the EZNEC entry writes it, line 2 and all, in
    the pinned regime: one BLAS/OpenMP thread, and the below/below sweep
    state forgotten first (it is per-process, `tests/conftest.py`), so a
    case answers the same whichever cases ran before it."""
    from threadpoolctl import threadpool_limits

    from momwire.eznec import _shell

    below = sys.modules.get("momwire._sommerfeld_below")
    if below is not None:
        below.reset_sweep_state()
    text = _shell.read_deck(FIXTURES / deck)
    if text is None:
        raise FileNotFoundError(FIXTURES / deck)
    with threadpool_limits(limits=1):
        return _shell.render(text, basis=basis, dialect=dialect)


def stamp_fields(text: str) -> list[str]:
    """Line 2's fields: ``momwire <version> <basis> <variant>``, and after
    them whatever the stamp adds (an experimental feature's ``+name``)."""
    return text.split("\n")[1].split()


def mask_version(text: str, basis: str) -> str:
    """``text`` with only the VERSION on line 2 masked, wherever that line
    repeats. Basis, variant and any feature tail stay, so a printout with an
    experimental flag on differs from its golden on line 2."""
    lines = text.split("\n")
    stamp = lines[1]
    fields = stamp.split()
    if fields[:1] != ["momwire"] or len(fields) < 4 or fields[2] != basis:
        raise AssertionError(f"engine stamp {stamp!r} does not name basis {basis!r}")
    masked = stamp.replace(f" {fields[1]} ", f" {VERSION_MASK} ", 1)
    return "\n".join(masked if line == stamp else line for line in lines)


def render_stable(deck: str, dialect: str, basis: str) -> str:
    printout = render_raw(deck, basis, dialect)
    if "NEC ERROR" in printout or "INPUT PARAMETERS" not in printout:
        raise AssertionError(
            f"{deck} [{dialect} {basis}] refused: the corpus holds served decks only"
        )
    if dialect == "nec4":
        printout = _CLOCK.sub(rf"\1\2{CLOCK_MASK}", printout)
    return mask_version(printout, basis)


def golden_path(deck: str, basis: str) -> Path:
    return GOLDEN / basis / (deck.removesuffix(".nec").replace("/", "__") + ".out")


def mask_stamp(text: str, basis: str) -> str:
    """``text`` with its engine stamp masked wherever it appears.

    The stamp is line 2, and a multi-run printout repeats it in every run's
    header. It has to NAME the basis: a render that silently answered on the
    default basis would otherwise pass as the other one."""
    lines = text.split("\n")
    stamp = lines[1]
    if basis not in stamp.split():
        raise AssertionError(f"engine stamp {stamp!r} does not name basis {basis!r}")
    return "\n".join(STAMP_MASK if line == stamp else line for line in lines)


def render(deck: str, basis: str) -> str:
    """One deck's printout, as the EZNEC entry renders it, in the pinned regime."""
    from threadpoolctl import threadpool_limits

    from momwire.eznec import _shell

    text = _shell.read_deck(FIXTURES / deck)
    if text is None:
        raise FileNotFoundError(FIXTURES / deck)
    with threadpool_limits(limits=1):
        printout = _shell.render(text, basis=basis)
    if "NEC ERROR" in printout:
        raise AssertionError(
            f"{deck} [{basis}] refused: the golden corpus holds served decks only"
        )
    return mask_stamp(printout, basis)


def regime() -> dict:
    """What the numbers depend on, as far as this process can see it."""
    import numpy
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits

    from importlib.metadata import version

    import momwire

    with threadpool_limits(limits=1):
        pools = [
            {
                k: (Path(v).name if k == "filepath" else v)
                for k, v in pool.items()
                if k
                in (
                    "user_api",
                    "internal_api",
                    "version",
                    "architecture",
                    "num_threads",
                    "filepath",
                )
            }
            for pool in threadpool_info()
        ]
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "momwire": version("momwire"),
        "accelerator_variant": momwire.accelerator_variant,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": cpu or platform.processor(),
        "threadpools_during_render": pools,
        "somm_disk_cache": os.environ.get("MOMWIRE_SOMM_CACHE"),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate() -> dict:
    dirty = _git("status", "--porcelain", "--", "src")
    if dirty:
        raise SystemExit(
            "src/ has uncommitted changes; a golden corpus must name the commit that printed it:\n"
            + dirty
        )
    for basis in BASES:
        (GOLDEN / basis).mkdir(parents=True, exist_ok=True)
        for stale in (GOLDEN / basis).glob("*.out"):
            stale.unlink()
    entries = []
    for deck, basis in CASES:
        started = time.perf_counter()
        text = render(deck, basis)
        print(f"{time.perf_counter() - started:7.2f} s  {deck} [{basis}]", flush=True)
        path = golden_path(deck, basis)
        path.write_bytes(text.encode("latin-1"))
        entries.append(
            {
                "deck": deck,
                "basis": basis,
                "golden": path.relative_to(GOLDEN).as_posix(),
                "deck_sha256": _sha((FIXTURES / deck).read_bytes()),
                "printout_sha256": _sha(text.encode("latin-1")),
            }
        )
    manifest = {
        "commit": _git("rev-parse", "HEAD"),
        "generator": "scripts/golden_printouts.py",
        "bases": list(BASES),
        "stamp_mask": STAMP_MASK,
        "regime": regime(),
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def generate_stable() -> dict:
    dirty = _git("status", "--porcelain", "--", "src")
    if dirty:
        raise SystemExit(
            "src/ has uncommitted changes; a golden corpus must name the commit that printed it:\n"
            + dirty
        )
    from momwire import _features

    if _features.active():
        raise SystemExit(
            f"experimental features are on ({', '.join(_features.active())}); "
            "the stable-path corpus is the flags-off printout"
        )
    for stale in STABLE.glob("*/*.out"):
        stale.unlink()
    entries = []
    for deck, dialect, basis in STABLE_CASES:
        started = time.perf_counter()
        text = render_stable(deck, dialect, basis)
        print(
            f"{time.perf_counter() - started:7.2f} s  {deck} [{dialect} {basis}]",
            flush=True,
        )
        path = stable_golden_path(deck, dialect, basis)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("latin-1"))
        entries.append(
            {
                "deck": deck,
                "dialect": dialect,
                "basis": basis,
                "golden": path.relative_to(STABLE).as_posix(),
                "deck_sha256": _sha((FIXTURES / deck).read_bytes()),
                "printout_sha256": _sha(text.encode("latin-1")),
            }
        )
    manifest = {
        "commit": _git("rev-parse", "HEAD"),
        "generator": "scripts/golden_printouts.py --corpus stable",
        "version_mask": VERSION_MASK,
        "regime": regime(),
        "entries": entries,
    }
    STABLE_MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _check_stable() -> int:
    sys.path.insert(0, str(ROOT / "tests"))
    from golden_compare import moved_cells

    moved = 0
    for deck, dialect, basis in STABLE_CASES:
        path = stable_golden_path(deck, dialect, basis)
        golden = path.read_bytes().decode("latin-1")
        text = render_stable(deck, dialect, basis)
        if text == golden:
            continue
        cells = moved_cells(golden, text)
        moved += 1
        print(f"{deck} [{dialect} {basis}]: {len(cells)} cells moved")
        for cell in cells[:20]:
            print("   ", cell)
    print(f"{moved} of {len(STABLE_CASES)} printouts moved")
    return 1 if moved else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    ap.add_argument(
        "--corpus",
        choices=("nec5", "stable"),
        default="nec5",
        help="nec5: tests/fixtures/golden_printouts (the default); "
        "stable: tests/fixtures/stable_path_goldens",
    )
    args = ap.parse_args(argv)
    if args.corpus == "stable":
        if args.check:
            return _check_stable()
        manifest = generate_stable()
        print(f"wrote {len(manifest['entries'])} printouts at {manifest['commit']}")
        return 0
    if args.check:
        sys.path.insert(0, str(ROOT / "tests"))
        from golden_compare import moved_cells

        moved = 0
        for deck, basis in CASES:
            golden = golden_path(deck, basis).read_bytes().decode("latin-1")
            text = render(deck, basis)
            if text == golden:
                continue
            cells = moved_cells(golden, text)
            moved += 1
            print(f"{deck} [{basis}]: {len(cells)} cells moved")
            for cell in cells[:20]:
                print("   ", cell)
        print(f"{moved} of {len(CASES)} printouts moved")
        return 1 if moved else 0
    manifest = generate()
    print(f"wrote {len(manifest['entries'])} printouts at {manifest['commit']}")
    return 0


if __name__ == "__main__":
    # The regime, before anything imports numpy or the accelerator. The test
    # session sets the same Sommerfeld switch in tests/conftest.py.
    os.environ["MOMWIRE_SOMM_CACHE"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # AVX-512 off, as every entry point and the test session have it: the
    # corpus must not depend on whether the CPU has it.
    import momwire_serve_client

    momwire_serve_client.pin_cpu_env()
    sys.exit(main())
