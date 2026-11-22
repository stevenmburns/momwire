"""The NEC-5 slot: a wire listed twice that cannot be merged is refused by
name (momwire#2107).

The NEC-5 slot keeps #1042's merge. A copy it cannot merge (its segment
count differs from its twin's) printed an INTERNAL ERROR frame around the
constructor's ValueError, on bs2 and razor2p alike; it is a refusal by name
now, in the deck's GW tags.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest

SMALL = pathlib.Path(__file__).resolve().parent / "fixtures" / "razor_printout_small"

NEC5_UNMERGEABLE = {
    # GW 3 has 7 segments against GW 2's 9
    "dup_counts_differ_raises.nec": (
        "NEC ERROR - the GW wires of tags 2 and 3 are one wire listed twice, "
        "and their segment counts differ, so they cannot be merged into one "
        "conductor: delete the copy (tag 3), or make the two copies identical "
        "- momwire#1042"
    ),
}


@pytest.mark.parametrize("basis", ["bs2", "razor2p"])
@pytest.mark.parametrize("name", list(NEC5_UNMERGEABLE))
def test_the_nec5_slot_refuses_an_unmergeable_copy_by_name(name, basis):
    from momwire.dropin.nec5 import _shell

    text = (SMALL / name).read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _shell.render(text, basis=basis)
    assert "INTERNAL ERROR" not in out
    assert NEC5_UNMERGEABLE[name] in out, out[-600:]


@pytest.mark.parametrize("basis", ["bs2", "razor2p"])
def test_the_nec5_slot_refuses_the_fuzzed_twin_by_name(basis):
    # dup_twin_fed_raises_h1705: two copies of one span, written with
    # fractional tags (the reader takes 2.5 as 2 and -0.5 as 0); refused,
    # not an INTERNAL ERROR frame
    from momwire.dropin.nec5 import _shell

    text = (SMALL / "dup_twin_fed_raises_h1705.nec").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _shell.render(text, basis=basis)
    assert "INTERNAL ERROR" not in out
    assert (
        "NEC ERROR - the GW wires of tags 2 and 0 are one wire listed twice, "
        "and their segment counts differ, so they cannot be merged into one "
        "conductor: delete the copy (tag 0), or make the two copies identical "
        "- momwire#1042"
    ) in out, out[-600:]
