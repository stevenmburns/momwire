"""MOMWIRE_TIMES's two outputs, read and checked (shared by the tests).

* the ``times:`` line on stderr (`momwire_serve_client.times_line`, and the
  standalone executables' `_mw_times.h`): every key, in one order; every
  ``*_ms`` cell non-negative; the cells and ``other_ms`` adding up to
  ``total_ms`` to the line's rounding;
* the printout's ``MATRIX TIMING`` (``FILL=`` / ``FACTOR=``) and ``RUN TIME``
  lines: measured seconds where momwire otherwise prints zeros, and nothing
  else in the printout moved.
"""

from __future__ import annotations

import re

import momwire_serve_client as _mech

KEYS = (
    ["route", "threads", "runs"]
    + [f"{c}_ms" for c in _mech.TIMES_CELLS]
    + ["other_ms", "total_ms", "ground_built", "ground_disk", "ground_blended"]
)

_FILL = re.compile(
    r"^( {24}FILL=)( *-?\d+\.\d{3})( SEC\.,  FACTOR=)( *-?\d+\.\d{3})( SEC\.)\r?$"
)
_RUN_TIME = re.compile(r"^( RUN TIME =)( *-?\d+\.\d{3})( SEC\. \(WALL CLOCK\))?\r?$")
_ENV_HEADING = " " * 30 + "- - - MOMWIRE_TIMES ENVIRONMENT - - -"
# the heading, a blank, seven settings and three blanks
_ENV_LINES = 12
_PHASE = re.compile(
    r"^ {24}(GROUND TABLES= *\d+\.\d{3} SEC\.  \(BUILT \d+, FROM DISK CACHE \d+, "
    r"ANCHOR-BLENDED \d+\)"
    r"|FILL: \d+ msec \(\d+ threads\) FACTOR: \d+ msec \(\d+ threads\)"
    r"|SOLUTION \(PORTS\+CURRENTS\+POWER\)= *\d+\.\d{3} SEC\."
    r"|RADIATED FIELD \(RP\)= *\d+\.\d{3} SEC\."
    r"|NEAR FIELD \(NE/NH\)= *\d+\.\d{3} SEC\.)\r?$"
)
_PHASES_PER_RUN = 5


def timed_extras(text: str | bytes) -> tuple[list[list[str]], int, int]:
    """MOMWIRE_TIMES's added lines: each environment block's settings, how
    many phase lines, and how many RUN TIME lines say WALL CLOCK."""
    if isinstance(text, bytes):
        text = text.decode("latin-1")
    lines = text.splitlines()
    envs, phases, wall = [], 0, 0
    for k, line in enumerate(lines):
        if line == _ENV_HEADING:
            envs.append([x.strip() for x in lines[k + 2 : k + 9]])
        elif _PHASE.match(line):
            phases += 1
        elif (m := _RUN_TIME.match(line)) and m.group(3):
            wall += 1
    return envs, phases, wall


def times_lines(stderr: str | bytes) -> list[dict]:
    """Every ``times:`` line in ``stderr``, parsed and checked."""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("latin-1")
    out = []
    for line in stderr.splitlines():
        if line.startswith("times:"):
            assert [item.partition("=")[0] for item in line.split()[1:]] == KEYS, line
            cells = _mech.parse_times_line(line)
            check(cells)
            out.append(cells)
    return out


def check(cells: dict) -> None:
    """One line's arithmetic: non-negative, and summing to its total."""
    stages = [cells[f"{c}_ms"] for c in _mech.TIMES_CELLS]
    assert all(v >= 0.0 for v in stages), cells
    assert cells["other_ms"] >= 0.0 and cells["total_ms"] > 0.0, cells
    # each cell rounded to 0.001 ms: the sum to 0.001 per cell
    assert abs(sum(stages) + cells["other_ms"] - cells["total_ms"]) <= 1e-3 * (
        len(stages) + 2
    ), cells
    for key in ("threads", "runs", "ground_built", "ground_disk", "ground_blended"):
        assert cells[key] >= 0, cells


def printout_timing(text: str | bytes) -> tuple[list[tuple[float, float]], list[float]]:
    """The printout's ``(FILL, FACTOR)`` pairs and its ``RUN TIME`` values."""
    if isinstance(text, bytes):
        text = text.decode("latin-1")
    fills, runs = [], []
    for line in text.splitlines():
        if m := _FILL.match(line):
            fills.append((float(m.group(2)), float(m.group(4))))
        elif m := _RUN_TIME.match(line):
            runs.append(float(m.group(2)))
    return fills, runs


def zero_timing(text: str | bytes) -> str:
    """The printout as momwire prints it when MOMWIRE_TIMES is off: its
    timing cells as zeros (each keeps its width), and the environment block,
    the phase lines and RUN TIME's WALL CLOCK left out."""
    if isinstance(text, bytes):
        text = text.decode("latin-1")

    def zero(cell: str) -> str:
        return f"{0.0:{len(cell)}.3f}"

    out = []
    skip = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        end = line[len(body) :]
        if skip:
            skip -= 1
            continue
        if body == _ENV_HEADING:
            skip = _ENV_LINES - 1
            continue
        if _PHASE.match(body):
            continue
        if m := _FILL.match(body):
            body = (
                m.group(1)
                + zero(m.group(2))
                + m.group(3)
                + zero(m.group(4))
                + m.group(5)
            )
        elif m := _RUN_TIME.match(body):
            body = m.group(1) + zero(m.group(2))
        out.append(body + end)
    return "".join(out)
