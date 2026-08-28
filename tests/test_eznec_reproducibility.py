"""The four bars of :mod:`eznec_reproducibility`, each against its own void
(stevenmburns/momwire#578).

The module states a boundary; these gates are what keep it a measurement.
Every void below is recomputed from the LIVE corpus rather than recorded, so
growing the corpus past a bar — or retuning a bar out of its void — fails
here, naming which one, instead of silently starting to blank readings or
quietly stopping to blank dust.

The last gate is the one the issue exists for: the corpus rendered twice at
different thread counts, byte-identical inside the boundary.
"""

from __future__ import annotations

import collections
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import eznec_reproducibility as R
from test_eznec_serve import MANIFEST, corpus

# How much of each void a bar has to leave on both sides. A bar that sits
# within a decade of either edge is not in the middle of anything.
CLEARANCE = 10.0


def _network_rows(text: str) -> list[list[float]]:
    rows = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if R._NETWORK_HEAD not in line:
            continue
        j = i + R._SKIP[R._NETWORK_HEAD] + 1
        while j < len(lines) and lines[j].strip():
            v = [R._f(c) for c in R._cells(lines[j])]
            if any(x is None for x in v):
                break
            rows.append(v)
            j += 1
    return rows


def _wire_rows(text: str) -> list[tuple[str, float]]:
    out = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        head = next((h for h in R._WIRE_HEADS if h in line), None)
        if head is None:
            continue
        j = i + R._SKIP[head] + 1
        while j < len(lines) and lines[j].strip():
            fields = lines[j].split()
            mag = R._f(fields[-2]) if len(fields) > R._WIRE_GEOMETRY_FIELDS else None
            if mag is None:
                break
            out.append((head, mag))
            j += 1
    return out


def _void(below: list[float], above: list[float], bar: float, name: str) -> None:
    """`bar` has to sit inside (max(below), min(above)) with CLEARANCE either
    side. Both edges are asserted: one keeps the bar off real readings, the
    other keeps it over the dust."""
    top, floor = max(below, default=0.0), min(above, default=math.inf)
    assert top * CLEARANCE < bar, (
        f"{name}: the bar has dropped onto the dust it must clear "
        f"(largest is {top:.4e}, bar is {bar:.4e})"
    )
    assert bar * CLEARANCE < floor, (
        f"{name}: the bar has risen into the readings it must spare "
        f"(weakest is {floor:.4e}, bar is {bar:.4e})"
    )


@pytest.mark.integration
def test_the_undefined_current_bar_sits_in_sixteen_empty_decades():
    """Rule 1. A network current either is a current or is the zero an open
    stub carries; the corpus has nothing in between."""
    cur = [
        abs(complex(v[R._I_RE], v[R._I_IM]))
        for text in corpus().values()
        for v in _network_rows(text)
    ]
    dust = [c for c in cur if c < R.UNDEFINED_NETWORK_CURRENT]
    real = [c for c in cur if c >= R.UNDEFINED_NETWORK_CURRENT]
    assert dust, "no deck exercises the open-stub row any more"
    _void(dust, real, R.UNDEFINED_NETWORK_CURRENT, "UNDEFINED_NETWORK_CURRENT")


@pytest.mark.integration
def test_the_cancellation_bar_sits_in_six_empty_decades():
    """Rule 2. An imaginary part is either a reactance or the crumb of a
    cancellation an NT card made exactly real."""
    ratio = [
        abs(v[im]) / abs(v[re])
        for text in corpus().values()
        for v in _network_rows(text)
        for re, im in ((R._Z_RE, R._Z_IM), (R._Y_RE, R._Y_IM))
        if v[re] and v[im]
    ]
    dust = [r for r in ratio if r < R.CANCELLATION_CRUMB]
    real = [r for r in ratio if r >= R.CANCELLATION_CRUMB]
    assert dust, "no deck pins a network impedance real any more"
    _void(dust, real, R.CANCELLATION_CRUMB, "CANCELLATION_CRUMB")


@pytest.mark.integration
def test_the_wire_table_bar_sits_in_four_empty_decades():
    """Rule 3, on the axis that separates: magnitude as a fraction of the
    deck's OWN peak in the same table. Absolute magnitude does not — the
    corpus carries wire-table dust at every scale from 1e-23 up."""
    rel = []
    for text in corpus().values():
        rows = _wire_rows(text)
        peak: dict[str, float] = collections.defaultdict(float)
        for head, mag in rows:
            peak[head] = max(peak[head], mag)
        rel += [mag / peak[head] for head, mag in rows if peak[head] > 0 and mag > 0]
    dust = [r for r in rel if r < R.WIRE_TABLE_DUST]
    real = [r for r in rel if r >= R.WIRE_TABLE_DUST]
    assert dust, "no deck carries a dead wire any more"
    _void(dust, real, R.WIRE_TABLE_DUST, "WIRE_TABLE_DUST")


@pytest.mark.integration
def test_the_wire_loss_bar_sits_in_thirty_one_empty_decades():
    """Rule 4. Zeros are excluded from BOTH sides: 54 decks print an exact
    0.0000E+00 because their wires are lossless, which is an answer."""
    frac = []
    for text in corpus().values():
        power = loss = None
        for line in text.split("\n"):
            if "INPUT POWER" in line and "=" in line:
                power = R._f(line.split("=")[1].split()[0])
            if "WIRE LOSS" in line and "=" in line:
                loss = R._f(line.split("=")[1].split()[0])
        if power and loss:
            frac.append(abs(loss) / abs(power))
    dust = [x for x in frac if x < R.WIRE_LOSS_DUST]
    real = [x for x in frac if x >= R.WIRE_LOSS_DUST]
    assert dust, "no deck prints a wire-loss crumb any more"
    _void(dust, real, R.WIRE_LOSS_DUST, "WIRE_LOSS_DUST")


@pytest.mark.integration
def test_the_boundary_stays_narrow():
    """What it costs, as a number rather than as a promise.

    56 lines in 6 of 80 decks, all of them carrying an open stub or a dead
    wire. If a rule starts reaching decks outside this set, the bar moved or
    the physics did, and either wants looking at rather than accepting.
    """
    touched = {
        cid: sum(
            1
            for a, b in zip(text.split("\n"), R.blank_undefined(text).split("\n"))
            if a != b
        )
        for cid, text in corpus().items()
    }
    hit = {cid: n for cid, n in touched.items() if n}
    assert set(hit) == {"0012", "0014", "0016", "0017", "0018", "0028"}, sorted(hit)
    assert sum(hit.values()) == 56, hit


def test_an_exact_zero_wire_loss_is_an_answer_and_not_a_crumb():
    """The control for rule 4's `value` truth-test. A lossless deck's zero is
    reproducible and has to stay inside the claim; blanking it would narrow
    the gate across 54 of the 80 decks for nothing."""
    text = "\n".join(
        [
            "     INPUT POWER   = 1.0000E+02 WATTS",
            "     WIRE LOSS     = 0.0000E+00 WATTS",
        ]
    )
    assert R.blank_undefined(text) == text
    crumb = text.replace("0.0000E+00", "4.1635E-54")
    assert R._MASK in R.blank_undefined(crumb)


def test_a_small_but_real_reactance_survives_the_crumb_rule():
    """The control for rule 2. The bar is 1e-11 of the row's own real part,
    which is six decades under the weakest ratio any capture prints — a
    genuinely small reactance beside a large resistance is still a reading."""
    cell = "{:>12}".format
    row = "   1     5 1" + "".join(
        cell(x)
        for x in [
            "-1.0000E+02",
            "5.0000E+01",
            "7.5000E-01",
            "-5.0000E-01",
            "-2.0000E+02",
            "-2.0000E-08",  # 1e-10 of the real part: a reading, not a crumb
            "-5.0000E-03",
            "1.0000E-09",
            "-8.5000E+01",
        ]
    )
    text = "\n".join([R._NETWORK_HEAD, "", "", "", row])
    assert R.blank_undefined(text) == text


@pytest.mark.integration
@pytest.mark.slow
def test_the_corpus_is_byte_identical_across_thread_counts_inside_the_boundary():
    """**The gate momwire#578 exists for.**

    Two renders of the whole corpus, one per thread count, in child processes
    because `OMP_NUM_THREADS` is read at the first parallel region and cannot
    be moved afterwards. Without the boundary this differs on 8 lines in
    `0017`/`0018`; with it, nowhere.

    This is what makes the four bars a gate rather than a derivation someone
    once ran: a fifth undefined quantity appearing in any table fails here
    before it can be argued about.
    """
    script = textwrap.dedent(
        """
        import os, sys
        os.environ["OMP_NUM_THREADS"] = sys.argv[1]
        sys.path.insert(0, sys.argv[2])
        from test_eznec_serve import MANIFEST, deck_text, render
        from eznec_reproducibility import blank_undefined
        out = []
        for entry in MANIFEST["captures"]:
            text = render(deck_text(entry["id"]))
            out.append(entry["id"])
            out.append(blank_undefined(text))
        sys.stdout.write("\\n".join(out))
        """
    )
    here = str(Path(__file__).parent)
    runs = [
        subprocess.run(
            [sys.executable, "-c", script, threads, here],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for threads in ("1", "8")
    ]
    if runs[0] != runs[1]:
        diff = [
            (i + 1, a, b)
            for i, (a, b) in enumerate(zip(runs[0].split("\n"), runs[1].split("\n")))
            if a != b
        ]
        pytest.fail(
            f"{len(diff)} lines move with thread count outside the boundary:\n"
            + "\n".join(f"  {i}: {a!r}\n  -> {b!r}" for i, a, b in diff[:8])
        )
    assert len(MANIFEST["captures"]) == 80, "the corpus grew; re-derive the bars"
