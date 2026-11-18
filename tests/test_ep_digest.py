"""The layout-blind digest behind windows-static.yml's MSVC preprocessing check."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ep_digest", ROOT / "scripts" / "windows_static" / "ep_digest.py"
)
ED = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ep_digest", ED)
_spec.loader.exec_module(ED)

TEXT = 'double f(double x);\n\n\nextern "C" double g(double);\n'


def test_layout_does_not_move_the_digest():
    for same in (
        'double f(double x);\nextern "C" double g(double);\n',
        '\n\n\ndouble f(double x);\r\n\r\n  extern  "C"\tdouble g(double);  \r\n\n',
    ):
        assert ED.digest(same) == ED.digest(TEXT)


def test_a_guard_that_lets_a_declaration_through_moves_the_digest():
    active = TEXT + "#pragma omp declare simd notinbranch simdlen(4)\n"
    assert ED.digest(active) != ED.digest(TEXT)
    active = TEXT + 'extern "C" double exp(double);\n'
    assert ED.digest(active) != ED.digest(TEXT)


def test_a_changed_token_moves_the_digest():
    assert ED.digest(TEXT.replace("f(double x)", "f(float x)")) != ED.digest(TEXT)
