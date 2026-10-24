"""momwire 0.80.0: every refusal the NEC-5 serve layer can print is reached by
a fixture deck that the C++-vs-Python gates run.

Each refusal exists twice, once in Python (``eznec/_serve.py``,
``eznec/_shell.py``) and once in C++ (``_nec5_seam_core.cpp``, the razor and
bspline seam cores, and the deck core's stepping and sequence rungs). The
gates compare the two word for word, but only on decks that reach the
refusal: ``test_nec5_razor_seam_core``'s fixture sweep, the standalone
printout tests and ``scripts/core_split/exe_compare.py`` all walk
``tests/fixtures``. A refusal no fixture reaches can be changed on one side
and nothing notices. This file closes that gap in both directions:

1. every refusal sentence the Python side can emit, enumerated from the
   source (each ``_REFUSE_*`` / ``_STEPPING_*`` constant and every literal
   ``raise ServeRefusal(...)``), is printed by the C++ seam -- razor-2p or
   bspline, route ``refused``, never ``declined`` -- for at least one deck
   under ``tests/fixtures``. The existing gates then hold that sentence
   equal to Python's. A sentence no deck can reach sits in :data:`ALLOWED`
   with its reason, and each reason is checked here rather than taken on
   trust;
2. every refusal sentence the C++ side carries exists on the Python side.

A new refusal on either side without a fixture deck fails here. Add a deck
under ``tests/fixtures/refusal_coverage_0800/`` (synthetic, one refusal
each) or, if no deck can reach it, an :data:`ALLOWED` entry whose reason this
file verifies.

The deck READER's refusals (``deck/_nec5.py`` against ``_nec5_deck_core``)
are not enumerated here: ``test_nec5_deck_core.py`` holds the two readers
byte-identical on their own.
"""

from __future__ import annotations

import ast
import re
import string
from pathlib import Path

import pytest

from momwire.deck._solver import basis_entry
from momwire.eznec import _serve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "momwire"
FIXTURES = ROOT / "tests" / "fixtures"
OWN = FIXTURES / "refusal_coverage_0800"

PYTHON_SOURCES = (SRC / "eznec" / "_serve.py", SRC / "eznec" / "_shell.py")
CXX_SOURCES = (
    SRC / "_nec5_seam_core.cpp",
    SRC / "_nec5_razor_seam_core.cpp",
    SRC / "_nec5_bspline_seam_core.cpp",
    SRC / "_nec5_deck_core.cpp",
)
# The two bases the C++ seams port, and their in-process hooks.
SEAMS = {"razor-2p": "nec5_razor_seam", "bspline": "nec5_bspline_seam"}
NAMED = re.compile(r"_(REFUSE|STEPPING)_[A-Z0-9_]+")

# A refusal no deck reaches on razor-2p or bspline, by its key in
# :func:`python_refusals`, with the reason. Each reason is verified below
# (`test_every_allowance_holds`), so an entry cannot outlive its reason.
ALLOWED = {
    # -- basis-gated: printed only for a basis neither C++ seam ports; both
    # ported bases have the capability (checked against the roster) and the
    # C++ seams carry no copy of the sentence (checked against the sources)
    "_check_basis_can_host: {},{} needs a series source AT a node where {} wir": (
        "node_gaps"
    ),
    "_REFUSE_CROSSING_SOURCE_NO_KNOT": "knot_feeds",
    "_check_basis_can_host: {},{} puts a series source AT a node, and basis {}": (
        "knot_feeds"
    ),
    "_check_basis_can_host: {},{} stands a wire END in a FINITE ground plane a": (
        "contact"
    ),
    "_check_basis_can_host: wire {} runs below a FINITE ground plane and basis": (
        "buried"
    ),
    "_prepare: basis {} cannot answer this dialect: its printout ": "charge_support",
    "_prepare: {} and basis {} does not serve wire loading: {}": "wire_loading",
    # -- defence in depth: the deck reader refuses the same deck first, with
    # its own sentence (the named deck proves it on both seams)
    "index_of: no GW card declares tag {}": "reader:no_wire_declares_tag.nec",
    "_site_for: no wire piece carries node {} of tag {}": (
        "reader:node_past_the_wire.nec"
    ),
    # -- the crossing junction's own grounded case: a crossing junction with
    # no wire above or none below is not a crossing, and the only shape left
    # (a member IN the plane) is refused earlier, by _REFUSE_IN_PLANE_WIRE.
    # Its sentence is word for word the plain grounded junction's, which
    # `grounded_junction.nec` reaches, so the C++ copy is gated all the same.
    "_crossing_site: {},{} addresses a node where several wires meet IN": (
        "same-sentence:_site_for: {},{} addresses a node where several wires meet IN"
    ),
}

# The raise sites that relay a sentence made elsewhere rather than write one.
RELAYS = {
    # the scope ladder's answer, every rung a `_REFUSE_*` constant above
    "serve_runs: reason #1": "ladder",
    "_prepare: reason #1": "ladder",
    # `basis_entry`'s ValueError: an unknown basis name, the launcher's
    # argument and never a deck's
    "_prepare: str(exc) #1": "basis name",
    # the solver's NotImplementedError (the crossing serve's scope): its
    # sentence is the solver's, and the C++ front declines those decks by
    # name (`front:buried-fill`, `front:bspline-buried-refusal-*`), so the
    # launcher prints Python's
    "_prepare: str(exc) #2": "solver scope",
}


def _regex(parts):
    """A sentence's regex from its literal parts, ``None`` for a field."""
    return "".join("(.*?)" if p is None else re.escape(p) for p in parts)


def _key(parts, width=50):
    return "".join("{}" if p is None else p for p in parts)[:width]


def _format_parts(text):
    out = []
    for literal, field, _spec, _conv in string.Formatter().parse(text):
        out.append(literal)
        if field is not None:
            out.append(None)
    return out


def _joined_parts(node):
    if isinstance(node, ast.Constant):
        return [node.value]
    return [v.value if isinstance(v, ast.Constant) else None for v in node.values]


def python_refusals():
    """``(sentences, relays)``: every refusal sentence the Python side can
    emit as ``{key: regex}``, and every raise site that relays one."""
    sentences, relays = {}, []
    for path in PYTHON_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and NAMED.fullmatch(node.targets[0].id)
            ):
                text = ast.literal_eval(node.value)
                sentences[node.targets[0].id] = _regex(_format_parts(text))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            seen = {}
            for node in ast.walk(func):
                if not (
                    isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call)
                    and getattr(node.exc.func, "id", None) == "ServeRefusal"
                ):
                    continue
                arg = node.exc.args[0]
                if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                    parts = _joined_parts(arg)
                    sentences[f"{func.name}: {_key(parts)}"] = _regex(parts)
                elif (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "format"
                    and isinstance(arg.func.value, ast.Name)
                    and NAMED.fullmatch(arg.func.value.id)
                ):
                    continue  # the constant, enumerated above
                else:
                    text = ast.unparse(arg)
                    seen[text] = seen.get(text, 0) + 1
                    relays.append(f"{func.name}: {text} #{seen[text]}")
    return sentences, relays


def _outcomes(acc, seam, path):
    """Every refusal the C++ seam prints for one deck (any run), and its
    route."""
    answer = getattr(acc, seam)(path.read_bytes())
    found = [answer["message"]] if answer["route"] == "refused" else []
    for structure in answer.get("structures", []):
        # razor's hook gives each structure as a dict of runs, bspline's as
        # the list of runs itself
        runs = structure.get("runs", []) if isinstance(structure, dict) else structure
        for run in runs:
            if run.get("kind") == "refused":
                found.append(run["message"])
    return answer["route"], found


@pytest.fixture(scope="module")
def printed():
    """``{sentence: {(basis, deck), ...}}`` over every fixture deck the
    gates walk, both C++ seams."""
    from momwire import _accel

    acc = _accel.acc
    if acc is None or not all(hasattr(acc, s) for s in SEAMS.values()):
        pytest.skip("no accelerator with the C++ seams loaded")
    decks = [
        p
        for p in sorted(FIXTURES.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    out = {}
    for path in decks:
        for basis, seam in SEAMS.items():
            _route, found = _outcomes(acc, seam, path)
            for message in found:
                out.setdefault(message, set()).add((basis, path.name))
    return out


def test_the_enumeration_finds_the_refusals():
    """Not vacuous: the walk sees the named constants and the raise sites,
    and every relay is one this file has read."""
    sentences, relays = python_refusals()
    assert sum(1 for k in sentences if NAMED.fullmatch(k)) >= 20, sorted(sentences)
    assert sum(1 for k in sentences if ": " in k) >= 10, sorted(sentences)
    assert sorted(relays) == sorted(RELAYS), (
        "a raise site relays a sentence made elsewhere; read where it comes "
        f"from and add it to RELAYS: {sorted(set(relays) ^ set(RELAYS))}"
    )


def test_every_python_refusal_is_printed_by_the_cxx_seam_on_a_fixture(printed):
    sentences, _ = python_refusals()
    unknown = sorted(set(ALLOWED) - set(sentences))
    assert not unknown, f"ALLOWED names refusals that no longer exist: {unknown}"
    missing = []
    for key, regex in sentences.items():
        hits = [m for m in printed if re.fullmatch(regex, m, re.S)]
        if key in ALLOWED:
            continue
        if not hits:
            missing.append(key)
    assert not missing, (
        "these refusals are reached by no fixture deck on either C++ seam, so "
        "a change to one side's sentence goes unnoticed: add a deck under "
        f"{OWN.relative_to(ROOT)} (or an ALLOWED entry with a checked "
        f"reason): {missing}"
    )


def test_every_cxx_printed_refusal_is_a_python_sentence(printed):
    """The reverse over what the gates actually see: every refusal either
    C++ seam prints on a fixture matches a Python sentence -- or the deck
    reader's, which its own gate holds."""
    sentences, _ = python_refusals()
    reader = _cxx_literals([SRC / "deck" / "_nec5.py"], python=True)
    stray = []
    for message in printed:
        if any(re.fullmatch(rx, message, re.S) for rx in sentences.values()):
            continue
        if any(len(f) >= 20 and f in message for f in reader):
            continue
        stray.append(message[:120])
    assert not stray, stray


def test_every_allowance_holds(printed):
    sentences, _ = python_refusals()
    cxx = "\n".join(_cxx_literals(CXX_SOURCES))
    for key, why in ALLOWED.items():
        regex = sentences[key]
        if why.startswith("same-sentence:"):
            other = why.split(":", 1)[1]
            assert sentences[other] == regex, (key, other)
            assert any(re.fullmatch(regex, m, re.S) for m in printed), other
            continue
        assert not any(re.fullmatch(regex, m, re.S) for m in printed), (
            f"{key} is ALLOWED as unreachable, but a fixture reaches it: "
            "drop the allowance"
        )
        if why.startswith("declined:"):
            from momwire import _accel
            from momwire.eznec._shell import render

            deck = OWN / why.split(":", 1)[1]
            for seam in SEAMS.values():
                answer = getattr(_accel.acc, seam)(deck.read_bytes())
                assert answer["route"] == "declined", (deck, seam, answer["route"])
                assert answer["message"] == "4nec2-cards", (
                    deck,
                    seam,
                    answer["message"],
                )
            assert re.search(regex, render(deck.read_text()), re.S), deck
            continue
        if why.startswith("reader:"):
            deck = OWN / why.split(":", 1)[1]
            reached = [
                m
                for m, where in printed.items()
                if any(d == deck.name for _, d in where)
            ]
            assert reached, deck
            assert not any(re.fullmatch(regex, m, re.S) for m in reached), deck
            assert all(
                {("razor-2p", deck.name), ("bspline", deck.name)} <= printed[m]
                for m in reached
            ), deck
        else:
            # basis-gated: both ported bases have the capability, and no C++
            # seam carries the sentence
            for basis in SEAMS:
                solver_class, _ = basis_entry(basis)
                assert _has(solver_class.capabilities, why), (key, basis)
            longest = max((p for p in re.split(r"\(\.\*\?\)", regex)), key=len).replace(
                "\\", ""
            )
            assert longest and longest not in cxx, (
                f"{key}: the C++ side now carries this sentence, so it is "
                "reachable there: give it a fixture deck"
            )


def _has(caps, cell):
    """Whether a basis' capabilities serve what an allowance's sentence
    refuses."""
    if cell == "node_gaps":
        return bool(caps.node_gaps)
    if cell == "knot_feeds":
        return bool(caps.knot_feeds)
    if cell == "contact":
        return caps.refusal("contact", "sommerfeld") is None
    if cell == "buried":
        return (
            caps.refusal("buried") is None
            and caps.refusal("buried", "crossing_junction") is None
        )
    if cell == "charge_support":
        return bool(set(caps.axes.get("charge_support", ())) - {"point"})
    assert cell == "wire_loading", cell
    return caps.refusal("wire_loading") is None


def test_the_relays_are_what_they_say():
    """The ladder relay prints only the named constants: `refusal` and its
    rungs return a constant's text or None."""
    tree = ast.parse((SRC / "eznec" / "_serve.py").read_text(encoding="utf-8"))
    rungs = {
        f.name: f
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef) and f.name.endswith("refusal")
    }
    assert {"refusal", "_drive_refusal", "_geometry_refusal"} <= set(rungs)
    for name, func in rungs.items():
        for node in ast.walk(func):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            v = node.value
            if isinstance(v, ast.Constant):
                assert v.value is None, name
            elif isinstance(v, ast.Name):
                assert v.id.startswith("_REFUSE_") or v.id in {
                    "drive",
                    "geometry",
                    "near",
                    "mixed",
                }, (name, v.id)
            else:
                assert (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Attribute)
                    and v.func.attr == "format"
                    and isinstance(v.func.value, ast.Name)
                    and (
                        NAMED.fullmatch(v.func.value.id)
                        or v.func.value.id == "template"
                    )
                ), (name, ast.unparse(v))


# --- the reverse: C++ sentences exist in Python -----------------------------

_LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _cxx_literals(paths, python=False):
    """String fragments: for C++, adjacent literals merged (as the compiler
    merges them), split wherever an expression interrupts; for Python, every
    literal and every f-string's literal parts."""
    out = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if python:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    out.append(node.value)
            continue
        out += _cxx_fragments(_COMMENT.sub("", text))
    return out


def _unescape(s):
    return re.sub(
        r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s
    )


def _cxx_fragments(code):
    frags, pos, current = [], 0, None
    for m in _LITERAL.finditer(code):
        between = code[pos : m.start()]
        if current is not None and between.strip() == "":
            current += _unescape(m.group(1))
        else:
            if current is not None:
                frags.append(current)
            current = _unescape(m.group(1))
        pos = m.end()
    if current is not None:
        frags.append(current)
    return frags


def _cxx_refusal_regions(code):
    """The parts of a C++ file that make a refusal sentence: a
    ``Refusal{...}`` / ``SeamRefusal{...}`` throw, a ``REFUSE_*`` constant,
    and the body of a ``refuse_*`` / ``*_refusal`` / ``sequence_read``
    function."""
    code = _COMMENT.sub("", code)
    regions = []
    for m in re.finditer(r"\b(?:Seam)?Refusal\{", code):
        regions.append(_balanced(code, m.end() - 1))
    for m in re.finditer(r"\bREFUSE_[A-Z0-9_]+\s*=", code):
        regions.append(code[m.end() : code.index(";", m.end())])
    for m in re.finditer(
        r"^[\w:<>&\s\*]*?\b(refuse_\w+|\w+_refusal|sequence_read)\s*\([^;{]*\)\s*\{",
        code,
        re.M,
    ):
        regions.append(_balanced(code, m.end() - 1))
    return regions


def _balanced(code, start):
    depth, i = 0, start
    while True:
        c = code[i]
        if c == '"':
            i = _LITERAL.match(code, i).end()
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return code[start : i + 1]
        i += 1


def test_every_cxx_refusal_sentence_exists_in_python():
    # the deck core's reader rungs answer to the Python reader
    python = _cxx_literals((*PYTHON_SOURCES, SRC / "deck" / "_nec5.py"), python=True)
    python += [
        p for t in python for p in re.split(r"\{[^{}]*\}", t)
    ]  # a template's literal runs
    fragments, missing = 0, []
    for path in CXX_SOURCES:
        for region in _cxx_refusal_regions(path.read_text(encoding="utf-8")):
            for frag in _cxx_fragments(region):
                # a sentence's words, not a decline name, a mnemonic or a
                # separator
                if frag.count(" ") < 3:
                    continue
                fragments += 1
                if not any(frag in p for p in python):
                    missing.append(f"{path.name}: {frag[:100]!r}")
    assert fragments >= 40, fragments
    assert not missing, (
        f"the C++ side carries refusal text the Python side does not: {missing}"
    )


def test_the_serve_layer_names_its_refusal_constants_as_this_file_reads_them():
    """A refusal constant named off the pattern would escape the
    enumeration: every module constant whose text the seam raises is
    `_REFUSE_*` or `_STEPPING_*`."""
    for path in PYTHON_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "ServeRefusal"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                assert node.args[0].id in {"reason"} or NAMED.fullmatch(
                    node.args[0].id
                ), (path.name, node.args[0].id)
    assert _serve.ServeRefusal  # the name this file walks for
