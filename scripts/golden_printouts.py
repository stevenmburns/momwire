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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    args = ap.parse_args(argv)
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
    sys.exit(main())
