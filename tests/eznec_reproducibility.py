"""The boundary of this seam's reproducibility claim (stevenmburns/momwire#578).

Two printouts of one deck are asserted byte-identical — across thread counts,
and across process history (``test_eznec_shell``'s residency gate). Four
printed quantities cannot honour that, and this module is where the seam says
so, once, with the measurement behind each.

They are not a tolerance. Each is a number the arithmetic does not define:
a quotient whose divisor is zero to sixteen decades, the imaginary part of a
cancellation that is exactly real, a phase whose magnitude is dust, a power
that is the sum of nothing. The reference engine prints its own rounding
crumb in the same cells — ``0017``'s tag-4 row reads ``-3.3851E+25
1.6763E+25`` in the capture — so this is not momwire being less careful than
EZNEC; it is one place where neither of them has a number to print.

Measured over the whole 80-deck corpus rendered at ``OMP_NUM_THREADS`` 1 and
8. **8 lines in 2 decks move; every bar below sits in a void with nothing in
it.** The rule each bar states is deliberately wider than the 8 lines: it
excludes what is UNDEFINED, not what was observed to move. Ten network rows
match rule 1 and only two of them moved — the other eight are the same
quotient of the same dust and happen to agree today. A boundary drawn around
the observation rather than around the arithmetic would be luck, and would
re-open the moment a machine reassociated a sum differently.

---

**1. A network row whose current is zero.** ``UNDEFINED_NETWORK_CURRENT``

``0017``/``0018`` hang wire 4 on two ``LD 4,4,*,0,1.E+10,0.`` loads — a
deliberate open — and connect it with ``NT`` cards. Its currents solve to
~1e-32 A, and the printed impedance is V/I with both operands at dust, so it
comes out ~1e+25 Ω with an undetermined phase: on ``0017`` the real and
imaginary parts TRANSPOSE between runs, on ``0018`` the magnitude moves 6x.
1e+25 cannot be floored — it is not small. Only its operands are.

    |I| on the ten rows that match      1.4e-33 … 3.4e-32 A
    |I| on the other 125 in the corpus  1.376e-16 A and up

Sixteen empty decades. The bar sits in the middle on a log scale.

**2. The imaginary part of an impedance an NT card pinned real.**
``CANCELLATION_CRUMB``

A network port's impedance is fixed by its card: ``0017`` prints -1.0000E+02
and -2.0000E+02, exactly real. The imaginary part is then a cancellation of
two equal sums and comes out at ±1e-14 or at a signed zero depending on the
order the reduction ran in. The sign of a zero is not a number — the same
statement ``test_eznec_serve._NETWORK_LOSS_DUST`` already makes about a line
whose presence a crumb decides.

    |Z_imag| / |Z_real| where it is a crumb    0 … 1.3e-16
    the next |Z_imag| / |Z_real| in the corpus 2.3e-07

Six empty decades. Applied per CELL, not per row: everything else on these
rows is a legitimate reading and stays inside the claim.

**3. A wire-table row whose current is dust.** ``WIRE_TABLE_DUST``

Wire 4's own current and charge rows. Their MAGNITUDE is reproducible to
every printed digit; the PHASE and the real/imaginary split are not, because
a cancellation can hold |z| steady while arg(z) walks. Absolute magnitude
does not separate these — the corpus carries wire-table dust at every scale
from 1e-23 up — but magnitude as a fraction of the DECK'S OWN PEAK in the
same table does, and cleanly:

    decade histogram of magnitude / deck peak, 4694 rows
      1e-10      9 rows      the weakest legitimate readings
      1e-11      0
      1e-12      0
      1e-13      0
      1e-14      0
      1e-15     34 rows and down, to 1e-22   dust

Four empty decades, and the physical reading agrees with the arithmetic one:
nothing twelve decades under a deck's own peak current is a measurement.

**4. A wire loss that is the sum of nothing.** ``WIRE_LOSS_DUST``

``0018`` prints ``WIRE LOSS = 4.1635E-54 WATTS`` against a 195 W input.
Fifty-four decks print an exact ``0.0000E+00`` here; five print this crumb;
the next nonzero is ``0028``'s 7.9e-25 and then real losses at 3e-09 and up.

    wire loss / input power, the crumb cluster   1.2e-56 … 1.8e-56
    the next nonzero in the corpus               7.9e-25

Thirty-one empty decades. Same shape as ``_NETWORK_LOSS_DUST``, one table
over, and derived separately because it is in a different unit.
"""

from __future__ import annotations

# Each bar sits in the middle of its measured void on a log scale.
# tests/test_eznec_reproducibility.py asserts every one of them still does.
UNDEFINED_NETWORK_CURRENT = 1e-24  # amps; void 3.4e-32 … 1.4e-16
CANCELLATION_CRUMB = 1e-11  # of the cell's own real part; void 1.3e-16 … 2.3e-7
WIRE_TABLE_DUST = 1e-13  # of the deck's peak in that table; void 1e-15 … 1e-10
WIRE_LOSS_DUST = 1e-40  # of input power; void 1.8e-56 … 7.9e-25

_MASK = "<undefined>"

_NETWORK_HEAD = "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -"
_WIRE_HEADS = ("- - - Wire Currents - - -", "- - - Wire Charge Densities - - -")

# Lines between a heading and its first row, matching test_eznec_serve._TABLES.
_SKIP = {_NETWORK_HEAD: 3, _WIRE_HEADS[0]: 5, _WIRE_HEADS[1]: 5}

# A network row is 12 characters of TAG/SEG and then nine 12-character cells:
# V real/imag, I real/imag, Z real/imag, Y real/imag, POWER.
_CELL0, _CELLW = 12, 12
_I_RE, _I_IM, _Z_RE, _Z_IM, _Y_RE, _Y_IM = 2, 3, 4, 5, 6, 7
# Where a wire row stops being geometry (seg, tag, x, y, z, length) and starts
# being a number the solver produced.
_WIRE_GEOMETRY_FIELDS = 6


def _cells(line: str) -> list[str]:
    return [line[_CELL0 + _CELLW * k : _CELL0 + _CELLW * (k + 1)] for k in range(9)]


def _f(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _blank(cells: list[str], which: range | tuple[int, ...]) -> None:
    for k in which:
        cells[k] = _MASK.rjust(_CELLW)


def _network_row(line: str) -> str:
    cells = _cells(line)
    v = [_f(c) for c in cells]
    if any(x is None for x in v):
        return line
    if abs(complex(v[_I_RE], v[_I_IM])) < UNDEFINED_NETWORK_CURRENT:
        # Rule 1. The current itself is one of the dust operands, so the claim
        # stops there rather than at the quotient: everything right of VOLTAGE
        # is derived from a divisor that is zero.
        _blank(cells, range(_I_RE, 9))
        return line[:_CELL0] + "".join(cells)
    # Rule 2, per cell. Z and Y independently: a card can pin either.
    crumbs = [
        im
        for re, im in ((_Z_RE, _Z_IM), (_Y_RE, _Y_IM))
        if abs(v[im]) < CANCELLATION_CRUMB * abs(v[re])
    ]
    if crumbs:
        _blank(cells, tuple(crumbs))
        return line[:_CELL0] + "".join(cells)
    return line


def _wire_rows(rows: list[str]) -> list[str]:
    """Rule 3, resolved against the peak of the rows handed in — which is why
    a whole table is normalized at once and not a line at a time."""
    mags = [
        _f(r.split()[-2]) if len(r.split()) > _WIRE_GEOMETRY_FIELDS else None
        for r in rows
    ]
    known = [m for m in mags if m is not None]
    if not known:
        return rows
    peak = max(known)
    if peak <= 0.0:
        return rows
    out = []
    for row, mag in zip(rows, mags):
        if mag is not None and mag < WIRE_TABLE_DUST * peak:
            head = row.split()[:_WIRE_GEOMETRY_FIELDS]
            width = row.index(head[-1]) + len(head[-1])
            out.append(row[:width] + "  " + _MASK)
        else:
            out.append(row)
    return out


def blank_undefined(text: str) -> str:
    """Replace every cell this seam does not claim to reproduce.

    Applied to BOTH sides of every comparison, exactly as
    ``test_eznec_serve.mask`` is: it narrows what is asserted, and can only
    hide a difference that both rules above call undefined.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    input_power = 0.0
    while i < len(lines):
        line = lines[i]
        if "INPUT POWER" in line and "=" in line:
            value = _f(line.split("=")[1].split()[0])
            input_power = abs(value) if value is not None else 0.0
        if "WIRE LOSS" in line and "=" in line:
            # Rule 4.
            value = _f(line.split("=")[1].split()[0])
            # `value` must be NONZERO: 54 of the 80 decks have lossless wires
            # and print an exact 0.0000E+00 here, which is an answer and not a
            # crumb. Blanking those would narrow the claim for nothing.
            if (
                value
                and input_power > 0.0
                and abs(value) < WIRE_LOSS_DUST * input_power
            ):
                out.append(line.split("=")[0] + "= " + _MASK)
                i += 1
                continue
        heading = next((h for h in _SKIP if h in line), None)
        if heading is None:
            out.append(line)
            i += 1
            continue
        stop = i + _SKIP[heading] + 1
        out.extend(lines[i:stop])
        body = []
        while stop < len(lines) and lines[stop].strip():
            body.append(lines[stop])
            stop += 1
        if heading == _NETWORK_HEAD:
            out.extend(_network_row(r) for r in body)
        else:
            out.extend(_wire_rows(body))
        i = stop
    return "\n".join(out)
