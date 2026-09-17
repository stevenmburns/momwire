"""momwire#1100: a refusal must survive the printout's latin-1 codec.

``eznec/_shell.py`` writes the printout with ``_CODEC = "latin-1"``.  Seven
``nec5``-dialect refusal messages carried an em dash (U+2014), so when one
fired the user never read it: the writer raised ``UnicodeEncodeError`` and
the printout said ``INTERNAL ERROR IN MOMWIRE ENGINE`` instead (Dan AC6LA's
spiral loop, 2026-09-17, on an ``LD`` the dialect refuses).  The regression
was silent because the refusal tests compare the exception's text, never the
bytes the shell writes.

Two gates: every string constant that can become a refusal encodes under the
shell's codec (raise sites, the ``_REFUSE_*`` sentences, the off-vocabulary
mnemonic table), and one refusal rendered end to end reads as its sentence.
"""

from __future__ import annotations

import ast
from pathlib import Path

from momwire.deck import _nec5
from momwire.eznec import _serve, _shell
from momwire.eznec._shell import render

DECK = (
    "CM refusal encoding\nCE\n"
    "GW 1,21,0.,-5.,10.,0.,5.,10.,.001\n"
    "GE 0,-1\n"
    "LD 6,1,5,0,0.,-500.\n"
    "FR 0,1,0,0,14.\n"
    "GN -1\n"
    "EX 0,1,11,0,1.414214,0.\n"
    "XQ 0\n"
    "EN\n"
)


def _message_constants(path: Path) -> list[tuple[int, str]]:
    """Every string constant that can reach a refusal in `path`: inside a
    `raise`, assigned to a `_REFUSE_*` name, or a value of a module-level
    dict keyed by two-letter card mnemonics (the off-vocabulary table)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []

    def strings(node: ast.AST):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                found.append((sub.lineno, sub.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            strings(node)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n.startswith("_REFUSE_") for n in names):
                strings(node.value)
            elif isinstance(node.value, ast.Dict) and node.value.keys:
                keys = node.value.keys
                if all(
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and len(k.value) == 2
                    and k.value.isupper()
                    for k in keys
                ):
                    strings(node.value)
    return found


def test_every_refusal_string_encodes_under_the_printout_codec():
    offenders = []
    for module in (_nec5, _serve):
        for lineno, text in _message_constants(Path(module.__file__)):
            try:
                text.encode(_shell._CODEC)
            except UnicodeEncodeError as exc:
                offenders.append((Path(module.__file__).name, lineno, exc.object))
    assert not offenders, offenders


def test_a_refused_card_reads_as_its_sentence_not_as_an_internal_error():
    printout = render(DECK)
    assert "INTERNAL ERROR" not in printout
    assert (
        " ***** NEC ERROR - LD type 6 is not part of this engine's nec5 dialect"
        in printout
    )
    # and the bytes the shell would write are the same text
    printout.encode(_shell._CODEC)
