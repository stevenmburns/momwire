"""check_link.py's instruction scan (momwire#2090): no AVX-family instruction
in an sse2 object of the fat MSVC build, read from `dumpbin /disasm` text.

The scan replaces the whole-corpus SDE Westmere pass, so its parser is the
gate: a VEX mnemonic, a ymm/zmm/k register operand or a BMI instruction
must be found, and the baseline's own instructions, continuation lines and
mangled names that merely contain `k1` must not be.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_link", ROOT / "scripts" / "windows_fat" / "check_link.py"
)
CL = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("check_link", CL)
_spec.loader.exec_module(CL)

CLEAN = """\
Dump of file mw_fill.obj

File Type: COFF OBJECT

?fill@mw@@YAXPEAN@Z:
  0000000000000000: 48 89 5C 24 08     mov         qword ptr [rsp+8],rbx
  0000000000000005: 66 0F 10 04 11     movupd      xmm0,xmmword ptr [rcx+rdx]
  000000000000000A: F2 0F 59 05 00 00  mulsd       xmm0,mmword ptr [__real@3ff0000000000000]
                    00 00
  0000000000000012: E8 00 00 00 00     call        ?g@k1@@YAXXZ
  0000000000000017: 66 0F 28 05 00 00  movapd      xmm0,xmmword ptr [__ymm@00000000]
                    00 00
  000000000000001F: 0F 00 E0           verr        ax
  0000000000000022: F3 48 0F BC C1     tzcnt       rax,rcx
  0000000000000027: C3                 ret
"""


def test_a_clean_sse2_object_has_none():
    assert CL.avx_instructions(CLEAN) == []


GUARDED = """\
?find@std@@YAPEBGPEBG0G@Z:
  0000000000000000: 83 3D 00 00 00 00  cmp         dword ptr [__isa_available],2
                    02
  0000000000000007: 7C 20              jl          0000000000000029
  0000000000000030: C5 FE 6F 09        vmovdqu     ymm1,ymmword ptr [r9]
  0000000000000034: C5 F8 77           vzeroupper
  0000000000000037: C3                 ret
?bare@mw@@YANN@Z:
  0000000000000000: C4 E2 F1 AB D9     vfmsub213sd xmm3,xmm3,xmm1
  0000000000000005: C3                 ret
"""


def test_a_guarded_function_is_allowed_and_a_bare_one_is_not():
    hits = CL.avx_instructions(GUARDED)
    assert [(h[0], h[2], h[4]) for h in hits] == [
        ("?find@std@@YAPEBGPEBG0G@Z", "vmovdqu", True),
        ("?find@std@@YAPEBGPEBG0G@Z", "vzeroupper", True),
        ("?bare@mw@@YANN@Z", "vfmsub213sd", False),
    ]


@pytest.mark.parametrize(
    "line, mnem",
    [
        (
            "  0000000000000040: C5 FD 10 04 11     vmovupd     ymm0,ymmword ptr [rcx+rdx]",
            "vmovupd",
        ),
        (
            "  0000000000000040: C4 E2 F1 A8 C2     vfmadd213pd xmm0,xmm1,xmm2",
            "vfmadd213pd",
        ),
        ("  0000000000000040: C5 F8 77           vzeroupper", "vzeroupper"),
        ("  0000000000000040: C4 E2 F3 F6 C2     mulx        rax,rcx,rdx", "mulx"),
        ("  0000000000000040: C4 E2 71 F7 C2     shlx        eax,edx,ecx", "shlx"),
        ("  0000000000000040: C5 F8 92 C8        kmovw       k1,eax", "kmovw"),
    ],
)
def test_an_avx_instruction_is_found(line, mnem):
    (hit,) = CL.avx_instructions(CLEAN + line + "\n")
    assert hit[1:3] == ("0000000000000040", mnem) and hit[4] is False


CALL_GUARDED = """\
?find_trivial@@YAPEBX0PEBXG@Z:
  0000000000000000: E8 00 00 00 00     call        ?_Use_avx2@std@@YA_NXZ
  0000000000000005: 84 C0              test        al,al
  0000000000000007: 74 20              je          0000000000000029
  0000000000000030: C5 FE 6F 09        vmovdqu     ymm1,ymmword ptr [r9]
  0000000000000034: C3                 ret
"""


def test_a_call_to_a_cpu_check_helper_is_a_guard():
    """The STL's vectorized algorithms ask through a helper (`_Use_avx2`)."""
    ((fn, _a, mnem, _o, guarded),) = CL.avx_instructions(CALL_GUARDED)
    assert (mnem, guarded) == ("vmovdqu", True)


def test_an_unguarded_function_keeps_its_first_instructions_for_the_report():
    CL.avx_instructions.preambles.clear()
    CL.avx_instructions(GUARDED)  # its `?bare@...` has no check
    lines = CL.avx_instructions.preambles["?bare@mw@@YANN@Z"]
    assert lines and "vfmsub213sd" in lines[0]
