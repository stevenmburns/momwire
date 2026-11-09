"""`--dist loadgroup` really groups: every xdist_group item carries its suffix.

The default lane runs `-n auto --dist loadgroup` (pyproject addopts), and the
groups are load-bearing twice over: tests/conftest.py's hook puts the portal
files on one worker because they share server state (momwire#403), and the
module-fixture files on one worker so their fixture is paid once, not once
per worker. On a worker, xdist marks a grouped item by rewriting its nodeid to
``<nodeid>@<group>``; the controller schedules by that suffix and nothing
else.

That rewrite can lapse with no failure anywhere. pytest 9.2.0 (2026-10-27)
moved a node's id to ``_id``, so pytest-xdist 3.8.0's ``item._nodeid = ...``
set a dead attribute and loadgroup degraded to plain load: the PR lane's
`test` job ran ~19 min where it had run ~9, with the summed call time flat,
because four workers each compiled test_razor_core_standalone's ~80 C++
objects at once in its module fixture (pytest-dev/pytest-xdist#1386;
pyproject pins pytest below 9.2 for it). This test is the tripwire: it reads
the worker's own collection, so it holds the hook path and xdist's rewrite
together, whatever the next pytest or xdist changes.
"""

from __future__ import annotations

import pytest

# Grouped itself, so the check is never vacuous: run alone, it still holds
# xdist's rewrite on its own nodeid; in the lane it also holds every item
# tests/conftest.py's hook grouped.
pytestmark = pytest.mark.xdist_group("loadgroup_tripwire")


def _groups(item) -> str:
    names = {
        str(m.args[0] if m.args else m.kwargs.get("name", "default"))
        for m in item.iter_markers("xdist_group")
    }
    return "_".join(sorted(names))


def test_every_grouped_item_carries_its_group_in_its_nodeid(request):
    config = request.config
    # On a worker xdist resets `dist` to "no" and keeps the controller's
    # choice as `loadgroup` (xdist/remote.py), which is what its own rewrite
    # reads.
    if not hasattr(config, "workerinput") or not getattr(
        config.option, "loadgroup", False
    ):
        pytest.skip("not an xdist worker under --dist loadgroup")
    grouped = [(item, _groups(item)) for item in request.session.items]
    grouped = [(item, g) for item, g in grouped if g]
    assert request.node in [item for item, _ in grouped]
    lapsed = [item.nodeid for item, g in grouped if not item.nodeid.endswith(f"@{g}")]
    assert not lapsed, (
        f"{len(lapsed)} of {len(grouped)} grouped items have no @group suffix, "
        "so --dist loadgroup is scheduling them as plain load (the pytest 9.2.0 "
        "/ pytest-xdist 3.8.0 lapse, pytest-dev/pytest-xdist#1386); first: "
        f"{lapsed[:3]}"
    )
