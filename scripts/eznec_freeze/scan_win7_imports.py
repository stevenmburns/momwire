"""Gate: no executable in the Windows 7 bundle imports an API Windows 7 SP1 lacks (momwire#1456).

The Win7 bundle is frozen with the community PythonWin7 interpreter so that it
starts on Windows 7, but nothing on a CI runner can RUN Windows 7.  This scan is
the only Win7 check before a user does: it walks every .exe/.dll/.pyd in the
bundle and reads both the import and the delay-import tables, failing on

* an API-set DLL (``api-ms-win-*``) outside the sets Windows 7 SP1 provides,
* a direct import of a DLL that does not exist on Windows 7 (combase.dll,
  shcore.dll, kernel.appcore.dll, ...; ``POST_WIN7_DLLS``), and
* a named function from the post-Windows-7 list below,

unless the hit is on the ALLOW-LIST.  ``api-ms-win-crt-*`` is not a hit: that
is the UCRT, which the README lists as a prerequisite (KB2999226, or the
VC++ 2015-2022 x64 redistributable) unless the bundle carries it app-locally.

An API set Windows 7 lacks DOES pass when the bundle ships a DLL of that name
beside the file importing it, because Windows 7 then loads the shipped file:
that is how the PythonWin7 PathCch shim and Microsoft's app-local UCRT
forwarders (``api-ms-win-core-synch-l1-2-0.dll`` and friends) work.  A shipped
forwarder is followed to its target, and the target is judged like a direct
import.

A DLL on the post-Windows-7 list passes on the same terms: only when the
bundle ships a DLL of that name beside the importer.

THE FUNCTION AND DLL LISTS ARE CURATED, NOT EXHAUSTIVE.  There is no Windows 7 SP1
kernel32 export table on the runner to diff against, so the list names the
post-Win7 functions known to have broken Python-on-Win7 builds and their
neighbours (PathCch*, WaitOnAddress, GetSystemTimePreciseAsFileTime, thread
descriptions, DPI, CPU sets, ...).  An import outside it passes this gate and
can still fail on Windows 7.  A static scan also cannot see a function looked
up at run time through GetProcAddress.  A real Windows 7 run is the only proof.

ALLOW-LIST, and why each entry is safe:

* ``python312.dll`` -> ``api-ms-win-core-path-l1-1-0.dll`` (PathCch*): the
  PythonWin7 build ships its own implementation of that DLL, and the rule
  above passes it only because the shim sits beside ``python312.dll``.
* ``AddDllDirectory`` / ``RemoveDllDirectory`` / ``SetDefaultDllDirectories``:
  Windows 7 SP1 has them with KB2533623, which the README requires.

Usage: ``python scripts/eznec_freeze/scan_win7_imports.py dist/momwire``
"""

from __future__ import annotations

import sys
from pathlib import Path

# The API sets Windows 7 SP1 provides: the l1-1-0 core sets, plus the few
# security/service sets it also carries.  Any other api-ms-win-* is newer.
WIN7_API_SETS = frozenset(
    {
        f"api-ms-win-core-{name}-l1-1-0"
        for name in (
            "console datetime debug delayload errorhandling fibers file handle heap "
            "interlocked io libraryloader localization localregistry memory misc "
            "namedpipe processenvironment processthreads profile rtlsupport string "
            "synch sysinfo threadpool util xstate"
        ).split()
    }
    | {
        "api-ms-win-core-processthreads-l1-1-1",
        "api-ms-win-security-base-l1-1-0",
        "api-ms-win-security-lsalookup-l1-1-0",
        "api-ms-win-service-core-l1-1-0",
        "api-ms-win-service-management-l1-1-0",
        "api-ms-win-service-management-l2-1-0",
        "api-ms-win-service-winsvc-l1-1-0",
    }
)

# Functions introduced after Windows 7 (kernel32/kernelbase/user32/shcore and
# friends).  Curated -- see the module docstring.
POST_WIN7_FUNCTIONS = frozenset(
    """
    GetSystemTimePreciseAsFileTime SetThreadDescription GetThreadDescription
    WaitOnAddress WakeByAddressSingle WakeByAddressAll CreateFile2 CopyFile2
    GetCurrentThreadStackLimits PrefetchVirtualMemory GetOverlappedResultEx
    InitializeSynchronizationBarrier EnterSynchronizationBarrier DeleteSynchronizationBarrier
    GetProcessMitigationPolicy SetProcessMitigationPolicy SetThreadInformation
    GetThreadInformation SetProcessInformation GetProcessInformation
    GetSystemCpuSetInformation GetProcessDefaultCpuSets SetThreadSelectedCpuSets
    DiscardVirtualMemory OfferVirtualMemory ReclaimVirtualMemory VirtualAlloc2
    MapViewOfFile3 MapViewOfFileNuma2 UnmapViewOfFile2 CompareObjectHandles
    GetCurrentPackageId GetCurrentPackageFullName GetPackageFamilyName
    GetCurrentPackagePath GetPackagePathByFullName LoadPackagedLibrary
    QueryUnbiasedInterruptTimePrecise QueryInterruptTimePrecise QueryInterruptTime
    GetSystemTimeAdjustmentPrecise SetThreadpoolTimerEx SetThreadpoolWaitEx
    GetMemoryErrorHandlingCapabilities CreateFileMappingFromApp MapViewOfFileFromApp
    VirtualProtectFromApp GetFirmwareType GetIntegratedDisplaySize
    PathCchCombine PathCchCombineEx PathCchCanonicalize PathCchCanonicalizeEx
    PathCchSkipRoot PathCchStripToRoot PathCchRemoveFileSpec PathCchAppend
    PathCchAppendEx PathCchFindExtension PathCchIsRoot PathAllocCombine
    PathAllocCanonicalize PathCchRemoveBackslash PathCchAddBackslash
    GetDpiForMonitor SetProcessDpiAwareness GetProcessDpiAwareness
    SetProcessDpiAwarenessContext GetDpiForWindow GetDpiForSystem
    SetThreadDpiAwarenessContext GetThreadDpiAwarenessContext GetSystemMetricsForDpi
    AdjustWindowRectExForDpi EnableNonClientDpiScaling
    GetAwarenessFromDpiAwarenessContext AreDpiAwarenessContextsEqual
    IsValidDpiAwarenessContext GetSystemDpiForProcess SystemParametersInfoForDpi
    CreatePseudoConsole ResizePseudoConsole ClosePseudoConsole GetFileInformationByName
    SetDefaultDllDirectories AddDllDirectory RemoveDllDirectory
    IsWow64Process2 GetTempPath2W GetTempPath2A ProcessPrng GetMachineTypeAttributes
    """.split()
)

# DLLs that do not exist on Windows 7 at all, so a direct import of one fails
# there whatever it names -- unless the bundle ships a DLL of that name beside
# the importer.  Curated, like the function list: these are the ones known to
# be Windows 8 or later, not a census of System32.
POST_WIN7_DLLS = frozenset(
    {
        "combase.dll",  # Windows 8: COM base, split out of ole32
        "shcore.dll",  # Windows 8.1: shell core (DPI awareness, streams)
        "kernel.appcore.dll",  # Windows 8: app-model core
        "twinapi.appcore.dll",  # Windows 8: app-model runtime
        "windows.storage.dll",  # Windows 8: storage runtime split out of shell32
        "dcomp.dll",  # Windows 8: DirectComposition
        "d3d12.dll",  # Windows 10
        "dxcore.dll",  # Windows 10
    }
)

# On Windows 7 SP1 only with KB2533623, which the Win7 README requires.
KB2533623_FUNCTIONS = frozenset(
    {"SetDefaultDllDirectories", "AddDllDirectory", "RemoveDllDirectory"}
)


def classify(
    dll: str,
    function: str | None,
    shipped_beside: bool,
    forward_to: str | None = None,
) -> str | None:
    """Why one import is a FAILURE, or None if it passes.

    ``dll`` is the imported DLL, ``function`` the imported name (None for an
    ordinal import), ``shipped_beside`` whether a file of that name sits in
    the importing file's directory, and ``forward_to`` -- when it does and
    that file FORWARDS the name -- the forward's target, ``"KERNEL32.Name"``.
    Pure, so the allow-list is testable without a bundle.

    An API set Windows 7 lacks passes only when the bundle ships a DLL of
    that name beside its importer: Windows 7's loader knows no such set, so
    it loads the shipped file from the application directory.  That is how
    both the PythonWin7 PathCch shim and Microsoft's app-local UCRT
    forwarders work.  A shipped forwarder is then judged by where it
    forwards: to a function Windows 7 has, it passes; to one it lacks, it
    fails exactly as a direct import would.
    """
    stem = dll.lower().removesuffix(".dll")
    if stem.startswith("api-ms-win-crt-"):
        return None  # the UCRT: a README prerequisite, or shipped app-locally
    if stem.startswith("api-ms-win-") and stem not in WIN7_API_SETS:
        if not shipped_beside:
            return f"imports API set {dll}, which Windows 7 SP1 lacks, and the bundle does not ship it"
        if forward_to and "." in forward_to:
            target_dll, target = forward_to.rsplit(".", 1)
            why = classify(f"{target_dll}.dll", target, shipped_beside=False)
            if why:
                return f"imports {dll}!{function}, forwarded to {forward_to}: {why}"
        return None
    if dll.lower() in POST_WIN7_DLLS and not shipped_beside:
        return f"imports {dll}, which does not exist on Windows 7, and the bundle does not ship it"
    if function in KB2533623_FUNCTIONS:
        return None
    if function in POST_WIN7_FUNCTIONS:
        return f"imports {dll}!{function}, which Windows 7 SP1 lacks"
    return None


def _forwards(path: Path) -> dict[str, str]:
    """{exported name: forward target} for the forwarded exports of one DLL."""
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]]
    )
    export = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
    if export is None:
        return {}
    return {
        sym.name.decode(errors="replace"): sym.forwarder.decode(errors="replace")
        for sym in export.symbols
        if sym.name and sym.forwarder
    }


def scan(root: Path) -> tuple[int, list[str]]:
    """(files scanned, failure lines) for every PE file under ``root``."""
    import pefile  # PyInstaller's own dependency, so present wherever we freeze

    wanted = [
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
    ]
    scanned, failures = 0, []
    forwards: dict[Path, dict[str, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".exe", ".dll", ".pyd") or not path.is_file():
            continue
        scanned += 1
        rel = path.relative_to(root)
        try:
            pe = pefile.PE(str(path), fast_load=True)
            pe.parse_data_directories(directories=wanted)
        except pefile.PEFormatError as exc:
            failures.append(f"{rel}: not a readable PE file ({exc})")
            continue
        for attr in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
            for entry in getattr(pe, attr, []):
                dll = entry.dll.decode(errors="replace")
                beside = path.parent / dll
                shipped = beside.is_file()
                if shipped and beside not in forwards:
                    forwards[beside] = _forwards(beside)
                names = [
                    imp.name.decode(errors="replace") if imp.name else None
                    for imp in entry.imports
                ]
                for name in names or [None]:
                    target = forwards.get(beside, {}).get(name) if shipped else None
                    why = classify(dll, name, shipped, target)
                    if why:
                        failures.append(f"{rel}: {why}")
    return scanned, sorted(set(failures))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python scripts/eznec_freeze/scan_win7_imports.py <bundle dir>")
        return 2
    root = Path(argv[0])
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 2
    scanned, failures = scan(root)
    print(f"scanned {scanned} PE files under {root} (import + delay-import tables)")
    for line in failures:
        print(f"FAIL {line}")
    if failures:
        print(f"{len(failures)} post-Windows-7 import(s) outside the allow-list")
        return 1
    if scanned == 0:
        print("ERROR: no PE files found; is this the bundle directory?")
        return 1
    print(
        "ok   no post-Windows-7 imports outside the allow-list (curated list; see docstring)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
