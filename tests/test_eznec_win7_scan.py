"""The Windows 7 import gate's allow-list, without a bundle (momwire#1456).

`scripts/eznec_freeze/scan_win7_imports.py` runs only on the Windows freeze
lane, against a frozen bundle.  Its decision per import is the pure
`classify()`, so the allow-list is pinned here: what passes, what fails, and
that an API set Windows 7 lacks passes only when the bundle ships it beside
the importer -- and then only if it forwards somewhere Windows 7 has.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCAN = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "eznec_freeze"
    / "scan_win7_imports.py"
)

PATH_SET = "api-ms-win-core-path-l1-1-0.dll"


def _scan():
    spec = importlib.util.spec_from_file_location("eznec_freeze_scan_win7", SCAN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_post_win7_api_set_passes_only_when_shipped_beside_its_importer():
    classify = _scan().classify
    # The PythonWin7 shim: a real implementation, shipped, no forward.
    assert classify(PATH_SET, "PathCchCombineEx", shipped_beside=True) is None
    # The release bundle's shape: python.org's python312.dll, no shim.
    assert classify(PATH_SET, "PathCchCombineEx", shipped_beside=False)


def test_a_shipped_forwarder_is_judged_by_its_target():
    classify = _scan().classify
    synch = "api-ms-win-core-synch-l1-2-0.dll"
    # An app-local UCRT forwarder pointing at a function Windows 7 has.
    assert (
        classify(
            synch,
            "InitializeCriticalSectionEx",
            True,
            "kernel32.InitializeCriticalSectionEx",
        )
        is None
    )
    # The same forwarder pointing at one Windows 7 lacks fails as a direct
    # import would.
    assert classify(synch, "WaitOnAddress", True, "kernelbase.WaitOnAddress")
    # Not shipped: fails whatever it would have forwarded to.
    assert classify(synch, "InitializeCriticalSectionEx", False)


def test_kb2533623_functions_pass_and_other_post_win7_functions_fail():
    classify = _scan().classify
    for name in ("AddDllDirectory", "RemoveDllDirectory", "SetDefaultDllDirectories"):
        assert classify("KERNEL32.dll", name, shipped_beside=False) is None
    # What the 0.76.0 release bundle's MSVCP140.dll (14.51) imports.
    for name in ("CopyFile2", "CreateFile2", "GetSystemTimePreciseAsFileTime"):
        assert classify("KERNEL32.dll", name, shipped_beside=False), name
    assert classify("KERNEL32.dll", "CreateFileW", shipped_beside=False) is None


def test_the_widened_function_list_fails():
    classify = _scan().classify
    for name in (
        "IsWow64Process2",
        "GetTempPath2W",
        "GetTempPath2A",
        "GetMachineTypeAttributes",
    ):
        assert classify("KERNEL32.dll", name, shipped_beside=False), name
    assert classify("bcryptprimitives.dll", "ProcessPrng", shipped_beside=False)


def test_a_dll_windows_7_lacks_fails_unless_shipped_beside():
    classify = _scan().classify
    for dll in ("combase.dll", "SHCORE.dll", "kernel.appcore.dll"):
        assert classify(dll, "Anything", shipped_beside=False), dll
        assert classify(dll, "Anything", shipped_beside=True) is None, dll
    # A DLL Windows 7 has passes by name; only its functions are judged.
    assert classify("ole32.dll", "CoInitializeEx", shipped_beside=False) is None


def test_win7_api_sets_and_the_ucrt_pass():
    classify = _scan().classify
    assert (
        classify("api-ms-win-core-synch-l1-1-0.dll", "x", shipped_beside=False) is None
    )
    assert (
        classify("api-ms-win-crt-runtime-l1-1-0.dll", "_initterm", shipped_beside=False)
        is None
    )
