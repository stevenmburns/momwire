"""The C++ Gauss-Legendre header is the frozen Python table, regenerated.

`src/momwire/_leggauss_table.h` is written by `scripts/gen_leggauss_header.py`
from `src/momwire/_leggauss_table.py` (the table `_quadrature.leggauss`
reads), the table's hex strings copied as C99 hex-float literals. A header
that is not that script's output, byte for byte, is stale: the C++ front half
would read other rules than the Python solver. The rules themselves are
compared bit for bit through the binding in `test_razor_cxx_front.py`.
"""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_header_is_the_table_regenerated():
    spec = importlib.util.spec_from_file_location(
        "gen_leggauss_header", ROOT / "scripts" / "gen_leggauss_header.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "src" / "momwire" / "_leggauss_table.h").read_text() == mod.render()
