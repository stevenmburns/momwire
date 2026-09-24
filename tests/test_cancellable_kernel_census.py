"""Every accelerator kernel that takes `cancel_flag` is in `_CANCELLABLE_KERNELS`.

A kernel that polls the flag but is not listed raises the C++
`AcceleratorAborted` instead of `momwire.SolveAborted`, and every caller's
`except SolveAborted` misses it. That hole was patched one kernel at a time
(momwire#245, #259, #269, #1006) until AK#1712 found nine more at once, the
Sommerfeld remainder block among them. This census reads each exported
kernel's pybind signature, so a new kernel that takes the flag fails here
until it is listed.
"""

from __future__ import annotations

import pytest

from momwire import _accel


@pytest.mark.skipif(_accel.acc is None, reason="accelerators not built")
def test_every_kernel_taking_cancel_flag_translates_its_abort():
    takes = {
        name
        for name in dir(_accel.acc)
        if "cancel_flag" in (getattr(getattr(_accel.acc, name), "__doc__", "") or "")
    }
    missing = sorted(takes - set(_accel._CANCELLABLE_KERNELS))
    assert not missing, f"kernels take cancel_flag but are not listed: {missing}"
    # The census is not vacuous: the known kernels are found by it.
    assert {"sommerfeld_remainder_bspline_Q", "assemble_Z_bspline"} <= takes


@pytest.mark.skipif(_accel.acc is None, reason="accelerators not built")
def test_listed_kernels_are_wrapped():
    for name in _accel._CANCELLABLE_KERNELS:
        fn = getattr(_accel.acc, name, None)
        if fn is not None:
            assert hasattr(fn, "__wrapped__"), f"{name} is listed but not wrapped"
