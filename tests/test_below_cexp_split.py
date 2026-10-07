"""batch_below's held divide-out modulus (`below_divide_out_held`).

The below/below projection's lane path takes the divide-out's exponential as
exp(Re z) * sincos(Im z) with exp(Re z) held across the pairs of one depth
sum -- glibc's own cexp, spelled out -- but only where a one-time self-check
found this process's libm agrees with std::exp(complex) bit for bit. The
lanes-vs-scalar gates (tests/test_below_lanes_1290.py) then compare the held
route against the scalar stages' std::exp; this file pins that the held route
is the one those gates exercise on the glibc builds, so they cannot pass by
comparing the old route with itself.
"""

import sys

import pytest

from momwire._accel import acc

pytestmark = pytest.mark.skipif(acc is None, reason="needs the C++ accelerator")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc builds only")
def test_the_held_route_is_live_on_glibc():
    assert hasattr(acc, "below_cexp_split_ok")
    assert acc.below_cexp_split_ok()
