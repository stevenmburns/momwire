"""momwire#1695: a GW tag two wires share, as the licensed NEC-5 reads it.

Dan AC6LA's 4nec2 quad repeats a tag. The licensed NEC-5 (nec5cl-simd-d0c1383,
black box, Haswell) was probed on the square below, a 2.8 m-a-side quad in
free space fed at node 6 of tag 1, with tag 2 on two sides:

- a node-addressed load (``LD 4`` at node 6 of tag 2) lands on the FIRST wire
  carrying the tag: Z = 124.40 - j0.984 ohms, the same as the deck with
  unique tags;
- a tag-wide load (``LD 5`` on tag 2, sigma 1000 S/m) reaches EVERY wire
  carrying it: 1056.6 + j8.29 ohms, against 362.9 - j153.1 with only one
  side lossy;
- the Wire Currents table prints tag 2 on all 22 of their elements.
"""

import re

import pytest

from momwire.eznec._shell import render

_QUAD = (
    "CM repeated tag\nCE\n"
    "GW 1 11 0 -2.8 -2.8 0 2.8 -2.8 1e-3\n"
    "GW 2 11 0 2.8 -2.8 0 2.8 2.8 1e-3\n"
    "GW {third} 11 0 2.8 2.8 0 -2.8 2.8 1e-3\n"
    "GW 4 11 0 -2.8 2.8 0 -2.8 -2.8 1e-3\n"
    "GE 0\n{load}\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.1 0\nXQ\nEN\n"
)
_LD4 = "LD 4 2 6 0 50 0"
_LD5 = "LD 5 2 0 0 1e3"


def _z(out: str) -> complex:
    rows = out.split("ANTENNA INPUT PARAMETERS", 1)[1].splitlines()
    row = next(r for r in rows if re.match(r"\s+1\s+\d+\s+\d\s", r))
    f = row.split()
    return complex(float(f[7]), float(f[8]))


def _deck(third: int, load: str) -> str:
    return _QUAD.format(third=third, load=load)


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
def test_a_node_address_on_the_tag_names_its_first_wire(basis):
    shared = render(_deck(2, _LD4), basis=basis)
    unique = render(_deck(3, _LD4), basis=basis)
    assert "NEC ERROR" not in shared
    # the second tag-2 wire is meshed exactly as the unique deck's tag 3
    assert _z(shared) == _z(unique)


def test_razor_reads_the_licensed_engines_first_wire_load():
    z = _z(render(_deck(2, _LD4), basis="razor-2p"))
    assert z == pytest.approx(124.40 - 0.984j, abs=0.02)


def test_a_tag_wide_load_reaches_every_wire_carrying_the_tag():
    shared = _z(render(_deck(2, _LD5), basis="razor-2p"))
    one_side = _z(render(_deck(3, _LD5), basis="razor-2p"))
    assert shared == pytest.approx(1056.6 + 8.29j, rel=2e-3)
    assert one_side == pytest.approx(362.9 - 153.1j, rel=1e-2)


def test_the_currents_table_prints_the_shared_tag_on_both_wires():
    out = render(_deck(2, _LD4), basis="razor-2p")
    table = out.split("Wire Currents", 1)[1]
    tags = [int(m) for m in re.findall(r"^\s+\d+\s+(\d+)\s+[-0-9]", table, re.M)]
    assert tags[:44] == [1] * 11 + [2] * 22 + [4] * 11
