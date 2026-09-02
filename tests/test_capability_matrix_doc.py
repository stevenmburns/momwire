"""`docs/capability-matrix.md` is generated, and every False cell is reasoned.

Two gates, and they answer different questions.

The first is drift: the document is rendered from the `capabilities` rows by
`scripts/capability_matrix.py`, and a row edit that does not regenerate it
ships a table that disagrees with the code. `--check` is the script's own
comparison rather than a second implementation of it here, because a test that
re-rendered the document itself would only pin this file against that one.

The second is the property the document exists to have. `Capabilities.refusal`
generates ``"<cell> is not supported by this solver"`` for a False cell with no
`refusals` entry — a placeholder that says nothing a consumer can plan around,
and momwire#396 goal 3 is the unit that cleared the nine of them the tree
carried. Nothing stops the tenth from arriving with the next False cell, so it
is a gate and not a cleanup: every False cell in every row must have a recorded
reason, and it must not be the generated one.
"""

import subprocess
import sys

import pytest

from momwire._capabilities import _AXES

import scripts.capability_matrix as gen
from scripts.capability_matrix import CLASSES, DOC, main


def test_the_committed_document_matches_the_declarations():
    assert main(["--check"]) == 0, (
        "docs/capability-matrix.md has drifted from the rows it is generated "
        "from — run `python scripts/capability_matrix.py`"
    )


def test_the_script_runs_as_a_script():
    """`--check` is a CI-shaped invocation, so it is exercised as one: an
    import-time failure or a broken `__main__` guard would leave the gate
    above passing while the documented command did not run."""
    done = subprocess.run(
        # Absolute: the working directory a lane runs from is not this
        # file's business, and `pythonpath = ["."]` puts the repo root on
        # sys.path for the import above but says nothing about cwd.
        [sys.executable, gen.__file__, "--check"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert DOC.name in done.stdout


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_no_false_cell_falls_through_to_the_generated_default(cls):
    caps = cls.capabilities
    generic = []
    for cell in (*_AXES, "pec", "refl-coef", "sommerfeld"):
        reason = caps.refusal(cell)
        if reason is not None and reason == f"{cell} is not supported by this solver":
            generic.append(cell)
    assert not generic, (
        f"{cls.__name__} refuses {generic} with `refusal()`'s generated "
        "placeholder. A False cell owes a reason — what is missing, whether it "
        "is a not-yet or a never, and where to go instead — because this prose "
        "reaches a user verbatim through the exception, the seam printouts and "
        "antennaknobs' host dialogs."
    )


def test_every_recorded_reason_is_prose_rather_than_a_label():
    """The weaker failure the gate above cannot see, and the one it cannot
    reach at all: a COMBINATION key is only ever answered through
    `refusal(a, b)`, so the generated default never stands in for one and the
    parametrized test above never looks at it. An entry that merely spells the
    cell name back would pass everything else.

    Length is a crude proxy and deliberately so — the shortest real reason in
    the tree is the enrichment/wire-loading one at ~120 characters, and a cell
    name plus "is not supported" is under 60."""
    for cls in CLASSES:
        for key, reason in cls.capabilities.refusals.items():
            assert len(reason) >= 60, (cls.__name__, key, reason)
            assert reason != f"{key} is not supported by this solver", (
                cls.__name__,
                key,
            )
