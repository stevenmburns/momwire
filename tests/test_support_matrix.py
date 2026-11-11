"""scripts/support_matrix.py: which checks a pull request needs (the
`required` check), and the verdict on them."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "support_matrix" / "check_runs.json"

pytest.importorskip("yaml")
_spec = importlib.util.spec_from_file_location(
    "support_matrix", ROOT / "scripts" / "support_matrix.py"
)
SM = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("support_matrix", SM)  # dataclasses look the module up
_spec.loader.exec_module(SM)

REGISTRY = (ROOT / SM.REGISTRY).read_text(encoding="utf-8")
WORKFLOWS = SM.load_workflows()
HIERARCHY = SM.load_hierarchy()
STANDALONE_LANES = {
    "standalone.yml:linux-artifact",
    "standalone.yml:linux",
    "standalone.yml:linux-nec42",
    "standalone.yml:linux-nec42-bs",
    "standalone.yml:windows",
    "standalone.yml:windows-nopython",
    "standalone.yml:macos-nopython",
}
ALWAYS = {"ci.yml:lint", "ci.yml:test", "ci.yml:test-pynec"}


def _select(files, registry=REGISTRY, base=None, hierarchy=None):
    return SM.select(
        files,
        head_registry=registry,
        base_registry=base,
        hierarchy=hierarchy or HIERARCHY,
        workflows=WORKFLOWS,
    )


def _level_jobs(upto):
    return {j for lv in HIERARCHY["level"][:upto] for j in lv["jobs"]}


def _with_reach(name, reach, registry=REGISTRY):
    """The registry with `name`'s reach set to `reach`."""
    pat = re.compile(rf'(name="{name}",\s*reach=)\d+')
    out, n = pat.subn(rf"\g<1>{reach}", registry)
    assert n == 1
    return out


# --- the selection rules ---------------------------------------------------------


def test_a_feature_only_diff_selects_its_levels_pr_jobs_only():
    feats, _ = SM.read_registry(REGISTRY)
    f = feats["exact_kernel_series"]
    assert f.reach == 1
    # Its own C++ unit: the native filter and standalone's src/** still RUN,
    # but a reach-1 feature is held to level 1's PR jobs alone.
    sel = _select(["src/momwire/_accel_exact_kernel_series.cpp"])
    assert set(sel.required) == ALWAYS
    assert set(sel.features) == {"exact_kernel_series"}
    assert sel.stable_files == []
    assert set(sel.required) <= _level_jobs(1) | ALWAYS
    # A deeper reach pulls in the deeper levels' PR jobs that run.
    deeper = _with_reach("exact_kernel_series", 7)
    sel = _select(["src/momwire/_accel_exact_kernel_series.cpp"], deeper)
    # Levels 5-7: test-macos and the bs2/razor exe lanes, not the nec42
    # ones (levels 8, 9), nor linux-artifact (a build, no level).
    assert set(sel.required) == ALWAYS | {"ci.yml:test-macos"} | (
        STANDALONE_LANES
        - {
            "standalone.yml:linux-artifact",
            "standalone.yml:linux-nec42",
            "standalone.yml:linux-nec42-bs",
        }
    )
    assert set(sel.required) <= _level_jobs(7) | ALWAYS


def test_a_stable_cpp_diff_selects_the_standalone_lanes():
    sel = _select(["src/momwire/_razor_core.cpp"])
    assert STANDALONE_LANES <= set(sel.required)
    assert "ci.yml:test-macos" in sel.required  # the native filter
    assert ALWAYS <= set(sel.required)
    assert not sel.features


def test_a_docs_only_diff_selects_lint_and_test():
    sel = _select(["docs/notes.md", "README.md"])
    assert ALWAYS <= set(sel.required)
    assert set(sel.required) - ALWAYS <= {"ci.yml:changes"}


def test_a_site_diff_also_selects_the_site_build():
    assert "site.yml:build" in _select(["site/src/x.md"]).required


def test_a_mixed_diff_keeps_the_stable_gates():
    sel = _select(["tests/test_pair_reuse_1809.py", "standalone/route.cpp"])
    assert set(sel.features) == {"pair_reuse"}
    assert sel.stable_files == ["standalone/route.cpp"]
    assert STANDALONE_LANES <= set(sel.required)


def test_a_reach_9_feature_selects_every_pr_running_job():
    stable = _with_reach("pair_reuse", len(HIERARCHY["level"]))
    files = ["tests/test_pair_reuse_1809.py"]
    sel = _select(files, stable)
    assert set(sel.required) == SM.running_jobs(WORKFLOWS, files) | ALWAYS
    # A wide diff claimed by the feature: everything that runs on a PR.
    wide, n = re.subn(
        r'(name="pair_reuse",.*?)paths=\(.*?\),',
        r'\g<1>paths=("**",),',
        stable,
        count=1,
        flags=re.DOTALL,
    )
    assert n == 1
    files = ["src/momwire/_razor_core.cpp", "setup.py", "site/x.md"]
    sel = _select(files, wide)
    assert sel.stable_files == []
    assert set(sel.required) == SM.running_jobs(WORKFLOWS, files) | ALWAYS
    assert STANDALONE_LANES | {"ci.yml:test-macos", "site.yml:build"} <= set(
        sel.required
    )


def test_a_registry_only_edit_touches_that_feature():
    edited = _with_reach("razor_cxx", 7)
    sel = _select([SM.REGISTRY], edited, base=REGISTRY)
    assert sel.features == {"razor_cxx": ["registry entry edited"]}
    assert sel.stable_files == []  # nothing outside FEATURES changed
    added = REGISTRY.replace(
        "    )\n}\n",
        '        Feature(name="probe", reach=1, env="MOMWIRE_X_PROBE",'
        ' kind="python", issue=1, since="2026-11-06"),\n    )\n}\n',
        1,
    )
    assert added != REGISTRY
    sel = _select([SM.REGISTRY], added, base=REGISTRY)
    assert sel.features == {"probe": ["added to the registry"]}
    # A change to the module outside FEATURES is stable code.
    other = REGISTRY.replace('ENV_PREFIX = "MOMWIRE_X_"', 'ENV_PREFIX = "MOMWIRE_Y_"')
    sel = _select([SM.REGISTRY], other, base=REGISTRY)
    assert sel.stable_files == [SM.REGISTRY]
    assert STANDALONE_LANES <= set(sel.required)


def test_nothing_selected_is_push_only():
    widest = ["src/momwire/x.cpp", "setup.py", "site/a", "standalone/a"]
    running = SM.running_jobs(WORKFLOWS, widest)
    for wf in ("wheels.yml", "eznec-dropin.yml", "nightly.yml", "deploy-docs.yml"):
        assert not any(j.startswith(wf + ":") for j in running), wf
    for job in (
        "ci.yml:test-slow",
        "ci.yml:test-crossgate",
        "ci.yml:test-memgate",
        "ci.yml:test-integration",
        "standalone.yml:release-linux",
        SM.SELF,
    ):
        assert job not in running


# --- the workflows and the names ---------------------------------------------------


def test_every_hierarchy_job_exists_in_a_workflow():
    jobs = SM.all_jobs(WORKFLOWS)
    named = [j for lv in HIERARCHY["level"] for j in lv["jobs"]]
    named += list(HIERARCHY["unassigned"])
    assert named
    assert set(named) <= set(jobs), set(named) - set(jobs)
    for job in ALWAYS:
        assert job in jobs


def _real_names():
    runs = json.loads(FIXTURE.read_text())["check_runs"]
    return {(r["workflow"], r["name"]) for r in runs}


def test_every_selectable_job_maps_to_a_real_check_name():
    real = _real_names()
    jobs = SM.all_jobs(WORKFLOWS)
    widest = ["src/momwire/x.cpp", "setup.py", "site/a", "standalone/a"]
    for key in SM.running_jobs(WORKFLOWS, widest) | ALWAYS:
        job = jobs[key]
        names = {n for w, n in real if w == job.workflow}
        for pat in job.check_names():
            assert any(re.fullmatch(pat, n) for n in names), (key, pat)


def test_names_map_for_matrix_jobs_and_skipped_templates():
    jobs = SM.all_jobs(WORKFLOWS)
    real = _real_names()
    slow = jobs["ci.yml:test-slow"]
    assert slow.check_names() == [re.escape(f"test-slow ({i}/3)") for i in (1, 2, 3)]
    assert ("ci.yml", slow.skipped_name()) in real  # skipped on a PR, unexpanded
    freeze = jobs["eznec-dropin.yml:freeze"]
    for pat in freeze.check_names():
        assert any(re.fullmatch(pat, n) for w, n in real if w == "eznec-dropin.yml")
    smoke = jobs["wheels.yml:wheel_smoke"]  # a matrix read from another job
    assert any(
        re.fullmatch(p, "smoke 3.12 wheel on macos-15") for p in smoke.check_names()
    )


def test_feature_paths_name_real_files():
    feats, _ = SM.read_registry(REGISTRY)
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    tracked = out.splitlines()
    for f in feats.values():
        assert f.paths, f"{f.name} has no paths"
        for g in f.paths:
            assert any(SM.glob_match(g, p) for p in tracked), (f.name, g)


def test_the_registry_reader_agrees_with_the_module():
    from momwire import _features as F

    feats, _ = SM.read_registry(REGISTRY)
    assert {n: (f.reach, f.paths) for n, f in feats.items()} == {
        n: (f.reach, f.paths) for n, f in F.FEATURES.items()
    }


@pytest.mark.parametrize(
    ("pattern", "path", "hit"),
    [
        ("**/*.md", "README.md", True),
        ("**/*.md", "a/b/c.md", True),
        ("src/**", "src/momwire/x.py", True),
        ("tests/test_standalone_*.py", "tests/test_standalone_x.py", True),
        ("tests/test_standalone_*.py", "tests/sub/test_standalone_x.py", False),
        ("scripts/probe_*.py", "scripts/probe_a.py", True),
        ("src/**/*.cpp", "src/momwire/a.cpp", True),
        ("src/**/*.cpp", "src/a.h", False),
    ],
)
def test_glob(pattern, path, hit):
    assert SM.glob_match(pattern, path) is hit


def test_dorny_some_is_true_for_any_file():
    # The `code` filter's predicate-quantifier is the default, `some`: `**`
    # holds for every file, so a docs-only diff runs the lanes. Mirrored.
    pats = ["**", "!docs/**", "!**/*.md"]
    assert SM.dorny_filter_match(pats, ["docs/a.md"], "some")
    assert not SM.dorny_filter_match(pats, ["docs/a.md"], "every")
    assert SM.dorny_filter_match(pats, ["src/a.py"], "every")


# --- the verdict -------------------------------------------------------------------


def _req(*keys):
    jobs = SM.all_jobs(WORKFLOWS)
    return [
        {
            "job": k,
            "check_names": jobs[k].check_names(),
            "skipped_name": jobs[k].skipped_name(),
        }
        for k in keys
    ]


def _judge(required, checks, **kw):
    args = dict(
        finished_workflows=set(),
        seen_workflows={"ci.yml", "standalone.yml"},
        appear_expired=False,
        deadline_expired=False,
    )
    args.update(kw)
    return SM.judge(required, checks, **args)


def C(wf, name, conclusion="success", status="completed", id=1):
    return SM.Check(wf, name, status, conclusion, id)


def test_judge_passes_when_every_selected_check_succeeded():
    v = _judge(
        _req("ci.yml:lint", "ci.yml:test"), [C("ci.yml", "lint"), C("ci.yml", "test")]
    )
    assert v.state == "pass"


def test_judge_a_skipped_selected_check_is_a_failure():
    v = _judge(_req("ci.yml:test-macos"), [C("ci.yml", "test-macos", "skipped")])
    assert v.state == "fail"
    slow = _req("ci.yml:test-slow")
    v = _judge(slow, [C("ci.yml", "test-slow (${{ matrix.shard }}/3)", "skipped")])
    assert v.state == "fail"
    assert "skipped" in v.lines[0]


def test_judge_other_failures_and_unselected_checks():
    req = _req("ci.yml:test")
    for bad in ("failure", "cancelled", "timed_out", "neutral", "action_required"):
        assert _judge(req, [C("ci.yml", "test", bad)]).state == "fail", bad
    # An unselected check's failure is not this check's business.
    v = _judge(req, [C("ci.yml", "test"), C("ci.yml", "test-macos", "failure")])
    assert v.state == "pass"
    # Same name, other workflow: not the selected job.
    v = _judge(_req("ci.yml:changes"), [C("wheels.yml", "changes")])
    assert v.state == "wait"


def test_judge_waits_then_times_out():
    req = _req("standalone.yml:linux")
    running = [C("standalone.yml", "linux", None, "in_progress")]
    assert _judge(req, running).state == "wait"
    assert _judge(req, []).state == "wait"  # not started yet (needs linux-artifact)
    assert _judge(req, running, deadline_expired=True).state == "fail"
    v = _judge(req, [], finished_workflows={"standalone.yml"})
    assert v.state == "fail"  # its workflow finished without it
    v = _judge(req, [], seen_workflows=set(), appear_expired=True)
    assert v.state == "fail"  # its workflow never started


def test_judge_the_newest_attempt_wins():
    req = _req("ci.yml:test")
    old_fail = C("ci.yml", "test", "failure", id=1)
    assert _judge(req, [old_fail, C("ci.yml", "test", id=2)]).state == "pass"
    assert (
        _judge(
            req, [C("ci.yml", "test", id=1), C("ci.yml", "test", "failure", id=2)]
        ).state
        == "fail"
    )


def test_judge_needs_every_matrix_cell():
    req = _req("ci.yml:test-slow")
    two = [C("ci.yml", f"test-slow ({i}/3)") for i in (1, 2)]
    assert _judge(req, two).state == "wait"
    three = two + [C("ci.yml", "test-slow (3/3)")]
    assert _judge(req, three).state == "pass"


def test_an_advisory_job_runs_on_a_pr_but_is_never_required():
    # ci.yml:consumer-contract (momwire#1882) runs on a code PR, so the stable
    # path would select it; [advisory] in hierarchy.toml is what keeps it out.
    job = "ci.yml:consumer-contract"
    assert job in HIERARCHY["advisory"]
    assert job not in SM.running_jobs(WORKFLOWS, ["src/momwire/x.py"])
    assert job not in _select(["src/momwire/x.py", "setup.py"]).required
