"""`depends=` must list every header the extension actually includes — #824.

setuptools' `build_ext` recompiles an object when a listed source or a listed
DEPENDENCY is newer than it. A header it was never told about is neither, so
editing one recompiles nothing: the stale object is relinked, the .so is
re-copied, and `build_ext --inplace` reports success. The binary then stops
matching its own source while every test that reads the source keeps passing.

momwire#568 paid for that once (a benchmark round comparing a stale .so
against numpy, reading 6x where the code had 19x) and setup.py's comment has
said "every new inline header belongs here" ever since. It happened anyway:
`_bspline_static_far_inline.h` (#808) and `_stable_inline.h` (#799) were both
absent from `_ACCEL_HEADERS` until #824.

#824 is why a prose instruction is not enough. The symptom was not a missing
speedup but a WRONG ANSWER that read as a hardware story: the C++ and numpy
lanes of the far static-moment table disagreed 600x over their 1e-14 bar, in
a band sitting exactly on the regime switch, reproducibly, on one workstation
— and passed on two CI runners and a second laptop. All three of those build
cold, which is the whole of the discriminator; the workstation was carrying
an `_accel_bspline.o` compiled from a mid-development revision of a header
that no later build had any reason to recompile. A stale binary survives
`git checkout main`, so it presents as "fails on pristine main".

So this test derives the requirement from the sources rather than trusting
anyone to remember it: every local header reachable, transitively, from an
extension's sources must appear in that extension's `depends=`.
"""

from __future__ import annotations

import ast
import glob
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETUP = ROOT / "setup.py"
PKG = ROOT / "src" / "momwire"

# `#include "foo.h"` — the quoted form only. Angle-bracket includes are system
# or vendored-by-glob headers and are not per-file dependencies here.
_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def _literal_lists():
    """The `depends=` lists out of setup.py, without importing it.

    Importing would need pybind11 and would run the platform flag logic; the
    lists are plain literals, so `ast` reads them exactly and cheaply. For
    `_NEAR_HEADERS = [...] + sorted(glob.glob(...))` only the literal left
    operand is taken — the vendored xsf headers are covered by the glob, and
    `test_the_vendored_glob_is_not_empty` checks that it actually matches.
    """
    tree = ast.parse(SETUP.read_text(), filename=str(SETUP))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                value = value.left
            if not isinstance(value, ast.List):
                continue
            try:
                out[target.id] = [ast.literal_eval(e) for e in value.elts]
            except ValueError:
                continue
    return out


LISTS = _literal_lists()

# (extension, its sources, its declared depends). The near-interface twin's
# sources are spelled inline at its Pybind11Extension call rather than bound
# to a name, so they are spelled inline here too.
EXTENSIONS = [
    ("momwire._accelerators", LISTS["_ACCEL_SOURCES"], LISTS["_ACCEL_HEADERS"]),
    (
        "momwire._near_interface_accel",
        ["src/momwire/_near_interface_accel.cpp"],
        LISTS["_NEAR_HEADERS"],
    ),
]


def _local_include_closure(sources):
    """Every header under src/momwire reachable from `sources` by #include.

    Transitive: a header pulled in only through another header is just as much
    a rebuild trigger as one named in the .cpp, and `depends=` is a flat list
    with no notion of indirection.
    """
    seen: set[str] = set()
    stack = [ROOT / s for s in sources]
    while stack:
        path = stack.pop()
        if not path.exists():
            continue
        for name in _INCLUDE.findall(path.read_text()):
            target = PKG / name
            if not target.exists():
                continue  # vendored (xsf/…) — covered by the glob, not by name
            rel = target.relative_to(ROOT).as_posix()
            if rel not in seen:
                seen.add(rel)
                stack.append(target)
    return seen


@pytest.mark.parametrize("ext,sources,depends", EXTENSIONS, ids=lambda v: str(v)[:40])
def test_every_included_header_is_a_declared_dependency(ext, sources, depends):
    missing = sorted(_local_include_closure(sources) - set(depends))
    assert not missing, (
        f"{ext}: header(s) included but not in depends=, so editing one "
        f"recompiles nothing and the stale object is silently relinked "
        f"(momwire#824, #568): {missing}"
    )


@pytest.mark.parametrize("ext,sources,depends", EXTENSIONS, ids=lambda v: str(v)[:40])
def test_declared_dependencies_all_exist(ext, sources, depends):
    """A path that has been renamed away is not a dependency, it is a typo —
    and it fails open, exactly like an absent one."""
    gone = sorted(d for d in depends if not (ROOT / d).exists())
    assert not gone, f"{ext}: depends= names files that do not exist: {gone}"


def test_the_vendored_glob_is_not_empty():
    """`_NEAR_HEADERS` covers extern/xsf by glob rather than by name, so the
    literal list above is allowed to omit those — but only while the glob
    still matches something. A re-vendor that moved the tree would leave the
    near extension with no header dependencies at all, silently."""
    assert glob.glob(str(ROOT / "extern/xsf/include/xsf/**/*.h"), recursive=True)


def test_the_inline_headers_are_all_accounted_for():
    """The other direction: no `*_inline.h` sits in the package unreferenced.

    An inline header that no TU includes is either dead or — the case that
    costs something — included from somewhere this test does not look.
    """
    declared = {h for _, _, d in EXTENSIONS for h in d}
    reachable = set().union(*(_local_include_closure(s) for _, s, _ in EXTENSIONS))
    orphans = sorted(
        p.relative_to(ROOT).as_posix()
        for p in PKG.glob("*_inline.h")
        if p.relative_to(ROOT).as_posix() not in reachable | declared
    )
    assert not orphans, f"inline headers no extension includes: {orphans}"
