"""`make lint` must run the ruff CI pins — momwire#997.

The lint target used to go through `$(PY)`, i.e. `$(PYTHON) -m ruff`. `PYTHON`
resolves, by design, to "an activated venv, then this repo's .venv, then THE
PARENT REPO'S .venv, then python3" — that last-but-one step is deliberate and
correct for build and test targets, where you want the interpreter that will
import momwire. Inside antennaknobs' submodule it resolves to antennaknobs'
venv, which is the right answer for `make build` and the wrong one for lint:
which environment imports momwire has nothing to do with which ruff should
judge its source.

So `make lint` ran whatever ruff happened to be installed wherever `PYTHON`
landed, while CI pins an exact version. Both venvs carry 0.16.5 today, so
nothing was actually mislinted — the defect is that nothing was ENFORCING it,
and "the local gate ran a different tool than CI" is a failure this repo has
met before.

WHY NOT A FILE-COUNT TRIPWIRE. The first design for this test compared the
count ruff prints against `git ls-files '*.py'`. That premise is wrong: since
0.16 the formatter's scope includes Markdown, so on this tree ruff reports 496
where `*.py` is 444 and `*.md` is 66 — the number is py + md minus the config's
own exclusions. A test asserting it would have to re-model both the extension
set and every `extend-exclude`, i.e. become a second source of truth for
ruff's file discovery that drifts every time either changes. The version is
the thing worth pinning, and it is exactly checkable.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _makefile_pin() -> str:
    text = (ROOT / "Makefile").read_text()
    m = re.search(r"^RUFF_VERSION\s*=\s*(\S+)\s*$", text, re.M)
    assert m, "Makefile has no RUFF_VERSION"
    return m.group(1)


def _ci_pins() -> list[str]:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    return re.findall(r"ruff==(\S+?)\s*$", text, re.M)


def test_g997_the_makefile_and_ci_pin_the_same_ruff():
    """Two hand-written version strings in two files is how they drift — the
    same shape as the `_BSPLINE_ACCEL_MAX_D` constant that sat at 2 for a
    release after the tables moved to 3 (#999). Make cannot read the workflow
    and the workflow cannot read Make, so a test holds them equal."""
    ci = _ci_pins()
    assert ci, "ci.yml no longer pins ruff==<version>"
    assert set(ci) == {_makefile_pin()}, (
        f"Makefile pins ruff {_makefile_pin()}, ci.yml pins {sorted(set(ci))}"
    )


def test_g997_the_lint_target_does_not_go_through_PY():
    """`$(PY) ruff` is the defect itself: it makes the tool version a property
    of whichever venv `PYTHON` resolved to. The target may fall back to it, but
    only after checking the version, so the guarantee holds on both paths."""
    text = (ROOT / "Makefile").read_text()
    body = text[text.index("\nlint:") :]
    body = body[: body.index("\n\n")] if "\n\n" in body else body
    assert "uvx ruff@$(RUFF_VERSION)" in body, (
        "the lint target should run the pinned ruff through uvx, which needs no venv"
    )
    if "$(PY) ruff" in body:
        assert 'if [ "$$have" != "$(RUFF_VERSION)" ]' in body, (
            "the $(PY) fallback must verify the version before using it, or it "
            "reintroduces exactly what #997 removed"
        )
