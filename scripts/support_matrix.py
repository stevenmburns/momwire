#!/usr/bin/env python3
"""Which CI checks a pull request needs, and whether they have passed.

Branch protection on main names ONE check, ``required`` (the job in
``.github/workflows/required.yml``). That job runs this script twice:

``select``
    Reads the support hierarchy (``support/hierarchy.toml``), the feature
    registry (``src/momwire/_features.py``, read with ``ast`` so nothing is
    imported or compiled) and the PR's changed files (``git diff
    base...head``), and prints, as JSON, the jobs the PR must pass and why.

``wait``
    Polls the check-runs API for the PR's head SHA until every selected job
    has completed, then fails unless each concluded ``success``. A selected
    job that was SKIPPED fails: a skipped required job must never read as a
    pass.

The rules (``support/README.md`` has the prose):

* **always**: ``ci.yml:lint``, ``ci.yml:test`` (the Linux default lane, which
  runs the stable-path tripwire) and ``ci.yml:test-pynec``.
* **a touched feature** of reach N: every job of levels 1..N that runs on
  this pull request. A feature is touched when a changed file matches one of
  its ``paths``, or when the PR adds the feature or edits its registry entry.
  Reach == the deepest level (stable) needs every job that runs.
* **stable path** (a changed file no feature claims): the jobs that would run
  on a PR made of those files alone -- today's behaviour, computed from the
  workflows' own ``on.pull_request.paths`` and ``changes`` path filters, which
  this script parses rather than copies.
* never a job that does not run on this pull request (push-, schedule-,
  dispatch- and tag-only lanes): the nightly certifies those.

Whether a job runs is decided by evaluating the workflows themselves: the
``on:`` trigger and its path filters, every ``dorny/paths-filter`` step, and
each job's ``if:`` and ``needs:``, for ``event_name == 'pull_request'``.
"""

from __future__ import annotations

import argparse
import ast
import functools
import itertools
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
HIERARCHY = ROOT / "support" / "hierarchy.toml"
REGISTRY = "src/momwire/_features.py"

# The aggregator itself: never selected, never waited on.
SELF = "required.yml:required"
ALWAYS = ("ci.yml:lint", "ci.yml:test", "ci.yml:test-pynec")

# --- globs ---------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _glob_re(pattern: str) -> re.Pattern[str]:
    """A path glob as GitHub's ``paths`` and dorny/paths-filter read it:
    ``**`` crosses directories (``**/`` may match none), ``*`` and ``?`` do
    not, and dotfiles match like any other name."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(pattern: str, path: str) -> bool:
    return bool(_glob_re(pattern).match(path))


def github_paths_match(patterns: list[str], files: list[str]) -> bool:
    """``on.<event>.paths``: a file is in when the LAST pattern it matches is
    positive; the workflow runs when any file is in."""
    for f in files:
        inside = False
        for p in patterns:
            if p.startswith("!"):
                if glob_match(p[1:], f):
                    inside = False
            elif glob_match(p, f):
                inside = True
        if inside:
            return True
    return False


def dorny_filter_match(patterns: list[str], files: list[str], quantifier: str) -> bool:
    """One dorny/paths-filter rule. Each pattern is a predicate (``!x`` is
    "does not match x"); a file matches the rule when SOME (the default) or
    EVERY predicate holds, and the rule is true when any file matches.

    Note what the default means for ``['**', '!docs/**', ...]``: ``**`` holds
    for every file, so the rule is true for any non-empty diff. That is how
    the filter behaves on GitHub today (a docs-only PR runs ``test`` in
    full), and this mirrors it rather than what the comments intend."""

    def pred(p: str, f: str) -> bool:
        return not glob_match(p[1:], f) if p.startswith("!") else glob_match(p, f)

    agg = all if quantifier == "every" else any
    return any(agg(pred(p, f) for p in patterns) for f in files)


# --- a small evaluator for GitHub expressions ------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<op>\|\||&&|==|!=|<=|>=|[!()<>,])|'(?P<str>(?:[^']|'')*)'"
    r"|(?P<num>-?\d+(?:\.\d+)?)|(?P<id>[A-Za-z_][A-Za-z0-9_.\-*]*))"
)


class ExprError(ValueError):
    pass


def _tokens(text: str) -> list[tuple[str, object]]:
    out, i = [], 0
    text = text.strip()
    while i < len(text):
        m = _TOKEN.match(text, i)
        if not m or m.end() == i:
            raise ExprError(f"cannot read expression at {text[i:]!r}")
        i = m.end()
        if m.group("op"):
            out.append(("op", m.group("op")))
        elif m.group("str") is not None:
            out.append(("lit", m.group("str").replace("''", "'")))
        elif m.group("num"):
            out.append(("lit", float(m.group("num"))))
        elif m.group("id"):
            out.append(("id", m.group("id")))
    return out


def _truthy(v: object) -> bool:
    return v not in (None, False, "", 0, 0.0)


def _eq(a: object, b: object) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def evaluate(text: str, ctx: dict, status: dict) -> object:
    """Evaluate a GitHub expression (``${{ }}`` optional). `ctx` resolves
    dotted names; `status` gives ``success()``/``always()``/... An unknown
    name raises, so a workflow this does not understand fails loudly."""
    text = text.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    toks = _tokens(text)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else (None, None)

    def take(kind=None, val=None):
        nonlocal pos
        t = peek()
        if (kind and t[0] != kind) or (val and t[1] != val):
            raise ExprError(f"expected {val or kind} in {text!r}, got {t}")
        pos += 1
        return t

    def or_():
        v = and_()
        while peek() == ("op", "||"):
            take()
            rhs = and_()
            v = v if _truthy(v) else rhs
        return v

    def and_():
        v = cmp_()
        while peek() == ("op", "&&"):
            take()
            rhs = cmp_()
            v = rhs if _truthy(v) else v
        return v

    def cmp_():
        v = unary()
        while peek()[0] == "op" and peek()[1] in ("==", "!="):
            op = take()[1]
            rhs = unary()
            v = _eq(v, rhs) if op == "==" else not _eq(v, rhs)
        return v

    def unary():
        if peek() == ("op", "!"):
            take()
            return not _truthy(unary())
        return primary()

    def primary():
        kind, val = peek()
        if kind == "op" and val == "(":
            take()
            v = or_()
            take("op", ")")
            return v
        if kind == "lit":
            take()
            return val
        if kind == "id":
            take()
            if peek() == ("op", "("):
                take()
                args = []
                while peek() != ("op", ")"):
                    args.append(or_())
                    if peek() == ("op", ","):
                        take()
                take("op", ")")
                return _call(val, args)
            if val in ("true", "false"):
                return val == "true"
            if val == "null":
                return None
            return _resolve(val)
        raise ExprError(f"unexpected {val!r} in {text!r}")

    def _call(name, args):
        name = name.lower()
        if name in status:
            return status[name]
        if name == "startswith":
            return str(args[0] or "").lower().startswith(str(args[1]).lower())
        if name == "endswith":
            return str(args[0] or "").lower().endswith(str(args[1]).lower())
        if name == "contains":
            return str(args[1]).lower() in str(args[0] or "").lower()
        if name == "fromjson":
            return Opaque(f"fromJSON({args[0]!r})")
        raise ExprError(f"unknown function {name}() in {text!r}")

    def _resolve(name):
        for key in sorted(ctx, key=len, reverse=True):
            if name == key:
                return ctx[key]
            if key.endswith(".*") and name.startswith(key[:-1]):
                return ctx[key]
        raise ExprError(f"unknown context {name!r} in {text!r}")

    v = or_()
    if pos != len(toks):
        raise ExprError(f"trailing tokens in {text!r}")
    return v


@dataclass(frozen=True)
class Opaque:
    """A value known only at run time (``fromJSON(needs.x.outputs.y)``)."""

    what: str


# --- the workflows ---------------------------------------------------------------


@dataclass
class Job:
    workflow: str
    id: str
    spec: dict

    @property
    def key(self) -> str:
        return f"{self.workflow}:{self.id}"

    @property
    def needs(self) -> list[str]:
        n = self.spec.get("needs") or []
        return [n] if isinstance(n, str) else list(n)

    def check_names(self) -> list[str]:
        """The check-run names this job reports when it runs: one per
        matrix cell. A cell that cannot be expanded statically (a matrix
        read from another job's output) is a regex fragment, ``.*``."""
        template = self.spec.get("name")
        cells = _matrix_cells((self.spec.get("strategy") or {}).get("matrix"))
        names = []
        for cell in cells:
            if template is None:
                if cell is None:
                    names.append(re.escape(self.id))
                elif cell:
                    vals = ", ".join(_matrix_str(v) for v in cell.values())
                    names.append(re.escape(f"{self.id} ({vals})"))
                else:
                    names.append(re.escape(self.id) + r" \(.*\)")
            else:
                names.append(_expand_name(template, cell or {}))
        return names

    def skipped_name(self) -> str:
        """The name a SKIPPED job reports: GitHub does not expand a matrix it
        never ran, so the template is printed verbatim."""
        return self.spec.get("name") or self.id


def _matrix_str(v) -> str:
    return (
        json.dumps(v)
        if isinstance(v, (dict, list))
        else str(v).lower()
        if isinstance(v, bool)
        else str(v)
    )


def _matrix_cells(matrix) -> list[dict | None]:
    """Static matrix cells: [None] for no matrix; [{}] when the matrix comes
    from an expression (unknown cells)."""
    if matrix is None:
        return [None]
    if isinstance(matrix, str):
        return [{}]
    axes = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
    if any(not isinstance(v, list) for v in axes.values()):
        return [{}]
    cells = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
    if cells == [{}]:
        cells = []
    for ex in matrix.get("exclude") or []:
        cells = [c for c in cells if any(c.get(k) != v for k, v in ex.items())]
    for inc in matrix.get("include") or []:
        hit = False
        for c in cells:
            if all(c.get(k) == v for k, v in inc.items() if k in axes):
                c.update({k: v for k, v in inc.items() if k not in axes})
                hit = True
        if not hit:
            cells.append(dict(inc))
    return cells or [{}]


def _expand_name(template: str, cell: dict) -> str:
    """A job ``name:`` with its ``${{ }}`` filled from `cell`, as a regex."""
    out, last = [], 0
    for m in re.finditer(r"\$\{\{(.*?)\}\}", template):
        out.append(re.escape(template[last : m.start()]))
        try:
            ctx = {f"matrix.{k}": v for k, v in cell.items()}
            v = evaluate(m.group(1), ctx, {})
            out.append(
                ".*"
                if isinstance(v, Opaque)
                else re.escape(_matrix_str(v) if v is not None else "")
            )
        except ExprError:
            out.append(".*")
        last = m.end()
    out.append(re.escape(template[last:]))
    return "".join(out)


@dataclass
class Workflow:
    file: str
    spec: dict
    jobs: dict[str, Job] = field(default_factory=dict)

    @property
    def on(self) -> dict:
        on = self.spec.get("on", self.spec.get(True))  # YAML 1.1 reads `on` as True
        if isinstance(on, str):
            return {on: None}
        if isinstance(on, list):
            return dict.fromkeys(on)
        return on or {}


def load_workflows(root: Path = ROOT) -> dict[str, Workflow]:
    out = {}
    for path in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        wf = Workflow(path.name, spec)
        for jid, jspec in (spec.get("jobs") or {}).items():
            wf.jobs[jid] = Job(path.name, jid, jspec or {})
        out[path.name] = wf
    return out


def all_jobs(workflows: dict[str, Workflow]) -> dict[str, Job]:
    return {j.key: j for wf in workflows.values() for j in wf.jobs.values()}


def _pr_context() -> dict:
    return {
        "github.event_name": "pull_request",
        "github.ref": "refs/pull/0/merge",
        "github.ref_type": "branch",
        "github.base_ref": "main",
        "github.head_ref": "branch",
        "github.repository": "stevenmburns/momwire",
        "github.event.pull_request.*": "",
        "inputs.*": None,
        "vars.*": None,
        "secrets.*": "",
        "runner.os": "Linux",
    }


def workflow_triggers_on_pr(wf: Workflow, files: list[str]) -> bool:
    on = wf.on
    if "pull_request" not in on:
        return False
    pr = on["pull_request"] or {}
    branches = pr.get("branches")
    if branches is not None and not any(glob_match(b, "main") for b in branches):
        return False
    if "paths" in pr and not github_paths_match(pr["paths"], files):
        return False
    if "paths-ignore" in pr and all(
        any(glob_match(p, f) for p in pr["paths-ignore"]) for f in files
    ):
        return False
    return True


def _job_outputs(job: Job, files: list[str], ctx: dict) -> dict[str, object]:
    """A job's ``outputs:``, resolving every dorny/paths-filter step against
    `files`. Other step outputs are unknown (Opaque)."""
    steps: dict[str, object] = {}
    for step in job.spec.get("steps") or []:
        sid = step.get("id")
        if not sid:
            continue
        ran = "if" not in step or _truthy(evaluate(str(step["if"]), ctx, _ALL_OK))
        if str(step.get("uses", "")).startswith("dorny/paths-filter"):
            w = step.get("with") or {}
            rules = yaml.safe_load(w.get("filters") or "") or {}
            q = w.get("predicate-quantifier", "some")
            for name, pats in rules.items():
                pats = [pats] if isinstance(pats, str) else pats
                val = "true" if ran and dorny_filter_match(pats, files, q) else "false"
                steps[f"steps.{sid}.outputs.{name}"] = val if ran else ""
        else:
            steps[f"steps.{sid}.outputs.*"] = Opaque(f"steps.{sid}")
    out = {}
    for name, expr in (job.spec.get("outputs") or {}).items():
        out[name] = evaluate(str(expr), {**ctx, **steps}, _ALL_OK)
    return out


_ALL_OK = {"success": True, "always": True, "cancelled": False, "failure": False}


def runs_on_pr(wf: Workflow, files: list[str]) -> dict[str, bool]:
    """For each job of `wf`, whether it runs on a pull request whose diff is
    `files` (assuming every job it needs succeeds)."""
    if not files or not workflow_triggers_on_pr(wf, files):
        return dict.fromkeys(wf.jobs, False)
    base = _pr_context()
    ran: dict[str, bool] = {}
    outputs: dict[str, dict] = {}

    def visit(jid: str) -> bool:
        if jid in ran:
            return ran[jid]
        job = wf.jobs[jid]
        needs_ran = all(visit(n) for n in job.needs)
        ctx = dict(base)
        for n in job.needs:
            for k, v in outputs.get(n, {}).items():
                ctx[f"needs.{n}.outputs.{k}"] = v
            ctx[f"needs.{n}.result"] = "success" if ran[n] else "skipped"
            ctx[f"needs.{n}.outputs.*"] = ""
        cond = job.spec.get("if")
        status = {**_ALL_OK, "success": needs_ran}
        if cond is None:
            ok = needs_ran
        else:
            text = str(cond)
            uses_status = re.search(r"\b(success|always|cancelled|failure)\s*\(", text)
            v = _truthy(evaluate(text, ctx, status))
            ok = v if uses_status else (needs_ran and v)
        ran[jid] = ok
        if ok:
            outputs[jid] = _job_outputs(job, files, ctx)
        return ok

    for jid in wf.jobs:
        visit(jid)
    return ran


def running_jobs(workflows: dict[str, Workflow], files: list[str]) -> set[str]:
    out = set()
    for wf in workflows.values():
        for jid, ok in runs_on_pr(wf, files).items():
            if ok:
                out.add(f"{wf.file}:{jid}")
    out.discard(SELF)
    # Visible on the PR, never required (hierarchy.toml [advisory]).
    out -= set(load_hierarchy().get("advisory", {}))
    return out


# --- the hierarchy and the registry ------------------------------------------------


def load_hierarchy(path: Path = HIERARCHY) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FeatureEntry:
    name: str
    reach: int
    paths: tuple[str, ...]
    dump: str  # the entry's AST, to tell an edited entry


def _features_node(tree: ast.Module):
    for i, node in enumerate(tree.body):
        target = (
            getattr(node, "target", None)
            or (getattr(node, "targets", None) or [None])[0]
        )
        if isinstance(target, ast.Name) and target.id == "FEATURES":
            return i, node
    raise ValueError("no FEATURES assignment in the registry")


def read_registry(source: str) -> tuple[dict[str, FeatureEntry], str]:
    """The registry's features, and the AST of everything else in the
    module (to tell a change confined to FEATURES). Read with ``ast``: the
    aggregator imports nothing from momwire and builds nothing."""
    if not source.strip():
        return {}, ""
    tree = ast.parse(source)
    i, node = _features_node(tree)
    feats = {}
    for call in ast.walk(node.value):
        if not (
            isinstance(call, ast.Call) and getattr(call.func, "id", None) == "Feature"
        ):
            continue
        fields = [
            "name",
            "reach",
            "env",
            "kind",
            "issue",
            "since",
            "kwarg",
            "aliases",
            "paths",
        ]
        kw = dict(zip(fields, (ast.literal_eval(a) for a in call.args)))
        kw.update({k.arg: ast.literal_eval(k.value) for k in call.keywords})
        feats[kw["name"]] = FeatureEntry(
            kw["name"], int(kw["reach"]), tuple(kw.get("paths", ())), ast.dump(call)
        )
    rest = ast.Module(body=tree.body[:i] + tree.body[i + 1 :], type_ignores=[])
    return feats, ast.dump(rest)


# --- selection ---------------------------------------------------------------------


@dataclass
class Selection:
    required: dict[str, list[str]] = field(default_factory=dict)  # job -> reasons
    features: dict[str, list[str]] = field(
        default_factory=dict
    )  # feature -> why touched
    stable_files: list[str] = field(default_factory=list)
    unreachable: dict[str, str] = field(
        default_factory=dict
    )  # level job -> why not required

    def add(self, job: str, why: str) -> None:
        self.required.setdefault(job, [])
        if why not in self.required[job]:
            self.required[job].append(why)


def select(
    files: list[str],
    *,
    head_registry: str,
    base_registry: str | None = None,
    hierarchy: dict | None = None,
    workflows: dict[str, Workflow] | None = None,
) -> Selection:
    hierarchy = hierarchy or load_hierarchy()
    workflows = workflows or load_workflows()
    levels = hierarchy["level"]
    deepest = len(levels)
    head, head_rest = read_registry(head_registry)
    # None: the base registry is the head's. "": there was none (a PR from
    # before the registry existed, or the one that added it).
    base, base_rest = (
        (head, head_rest) if base_registry is None else read_registry(base_registry)
    )

    sel = Selection()
    for job in ALWAYS:
        sel.add(job, "always")

    claimed: set[str] = set()
    for name, f in head.items():
        hits = [p for p in files if any(glob_match(g, p) for g in f.paths)]
        if hits:
            sel.features.setdefault(name, []).append(f"paths: {', '.join(hits)}")
            claimed.update(hits)
        if REGISTRY not in files:
            continue
        if name not in base:
            sel.features.setdefault(name, []).append("added to the registry")
        elif base[name].dump != f.dump:
            sel.features.setdefault(name, []).append("registry entry edited")
    # A registry edit confined to FEATURES is the touched features' business;
    # any other change to the module is stable code.
    if REGISTRY in files and head_rest == base_rest:
        claimed.add(REGISTRY)

    running = running_jobs(workflows, files)
    for name in sel.features:
        reach = head[name].reach
        for number, level in enumerate(levels[:reach], start=1):
            for job in level["jobs"]:
                if job in running:
                    sel.add(
                        job,
                        f"feature {name} (reach {reach}): level {number} {level['id']}",
                    )
                else:
                    sel.unreachable.setdefault(
                        job,
                        f"feature {name}: level {number} job does not run on this PR",
                    )
        if reach == deepest:
            for job in sorted(running):
                sel.add(job, f"feature {name} is stable (reach {reach}): every PR job")

    sel.stable_files = [f for f in files if f not in claimed]
    for job in sorted(running_jobs(workflows, sel.stable_files)):
        sel.add(job, "stable path: runs today for " + _sample(sel.stable_files))
    for job in sel.required:
        sel.unreachable.pop(job, None)
    return sel


def _sample(files: list[str], n: int = 3) -> str:
    more = f" (+{len(files) - n} more)" if len(files) > n else ""
    return ", ".join(files[:n]) + more


def selection_json(sel: Selection, workflows: dict[str, Workflow]) -> dict:
    jobs = all_jobs(workflows)
    return {
        "required": [
            {
                "job": key,
                "check_names": jobs[key].check_names(),
                "skipped_name": jobs[key].skipped_name(),
                "why": why,
            }
            for key, why in sorted(sel.required.items())
        ],
        "features_touched": sel.features,
        "stable_files": sel.stable_files,
        "not_required_level_jobs": dict(sorted(sel.unreachable.items())),
    }


# --- waiting -------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    workflow: str  # file name, e.g. "ci.yml"
    name: str
    status: str  # queued / in_progress / completed
    conclusion: str | None
    id: int = 0


@dataclass(frozen=True)
class Verdict:
    state: str  # "pass", "fail", "wait"
    lines: tuple[str, ...]


def judge(
    required: list[dict],
    checks: list[Check],
    *,
    finished_workflows: set[str],
    seen_workflows: set[str],
    appear_expired: bool,
    deadline_expired: bool,
) -> Verdict:
    """The verdict on the selected jobs, given the check runs seen so far.

    * a selected job passes only when every check run it reports completed
      with ``success``; SKIPPED, neutral, cancelled, timed out all fail;
    * a selected job whose workflow finished without reporting it fails, as
      does one whose workflow never started before `appear_expired`;
    * otherwise wait, until `deadline_expired`.
    Re-runs: the newest check run (largest id) of a name wins."""
    latest: dict[tuple[str, str], Check] = {}
    for c in checks:
        k = (c.workflow, c.name)
        if k not in latest or c.id > latest[k].id:
            latest[k] = c
    lines, failed, pending = [], False, False
    for r in required:
        wf = r["job"].split(":", 1)[0]
        mine = [c for (w, _), c in latest.items() if w == wf]
        skipped = [
            c for c in mine if c.name == r["skipped_name"] and c.conclusion == "skipped"
        ]
        cells = []
        for pat in r["check_names"]:
            rx = re.compile(pat + r"\Z")
            cells.append([c for c in mine if rx.match(c.name)])
        if skipped and not any(cells):
            lines.append(
                f"FAIL {r['job']}: skipped (a skipped required job is a failure)"
            )
            failed = True
            continue
        bad = [
            c
            for cell in cells
            for c in cell
            if c.status == "completed" and c.conclusion != "success"
        ]
        if bad:
            what = ", ".join(f"{c.name}={c.conclusion}" for c in bad)
            lines.append(f"FAIL {r['job']}: {what}")
            failed = True
            continue
        missing = [p for p, cell in zip(r["check_names"], cells) if not cell]
        running = [c for cell in cells for c in cell if c.status != "completed"]
        if not missing and not running:
            lines.append(f"ok   {r['job']}")
            continue
        if missing and wf in finished_workflows:
            lines.append(f"FAIL {r['job']}: {wf} finished without reporting {missing}")
            failed = True
        elif missing and wf not in seen_workflows and appear_expired:
            lines.append(f"FAIL {r['job']}: {wf} never started on this commit")
            failed = True
        elif deadline_expired:
            lines.append(
                f"FAIL {r['job']}: still {'missing' if missing else 'running'} at the deadline"
            )
            failed = True
        else:
            lines.append(
                f"wait {r['job']}: {'not reported yet' if missing else 'running'}"
            )
            pending = True
    state = "fail" if failed else ("wait" if pending else "pass")
    return Verdict(state, tuple(lines))


def _api(url: str, token: str) -> dict:
    if not url.startswith("https://api.github.com/"):
        raise ValueError(f"not a GitHub API URL: {url}")
    req = urllib.request.Request(  # noqa: S310 — https-only, checked above
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — as above
        return json.load(resp)


def fetch(repo: str, sha: str, token: str, own_run_id: str | None):
    """The head SHA's check runs, each tagged with its workflow file, and
    which workflows have a pull_request run (started / finished). Two calls
    (plus pages): check-runs, and actions/runs to map suites to workflows."""
    api = f"https://api.github.com/repos/{repo}"
    runs = _api(
        f"{api}/actions/runs?head_sha={sha}&event=pull_request&per_page=100", token
    )
    suite_wf, seen, finished = {}, set(), set()
    for run in sorted(runs.get("workflow_runs", []), key=lambda r: r["id"]):
        if own_run_id and str(run["id"]) == str(own_run_id):
            continue
        wf = Path(run["path"]).name
        suite_wf[run["check_suite_id"]] = wf
        seen.add(wf)
        if run["status"] == "completed":
            finished.add(wf)
        else:
            finished.discard(wf)  # a newer attempt is still going
    checks, page = [], 1
    while True:
        data = _api(f"{api}/commits/{sha}/check-runs?per_page=100&page={page}", token)
        for c in data["check_runs"]:
            wf = suite_wf.get(c["check_suite"]["id"])
            if wf is None or f"{wf}:{c['name']}" == SELF:
                continue
            checks.append(Check(wf, c["name"], c["status"], c["conclusion"], c["id"]))
        if len(data["check_runs"]) < 100:
            break
        page += 1
    finished.discard("required.yml")
    return checks, seen, finished


def wait(args) -> int:
    required = json.loads(Path(args.selection).read_text())["required"]
    token = os.environ["GITHUB_TOKEN"]
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        checks, seen, finished = fetch(args.repo, args.sha, token, args.own_run_id)
        v = judge(
            required,
            checks,
            finished_workflows=finished,
            seen_workflows=seen,
            appear_expired=elapsed > args.appear_minutes * 60,
            deadline_expired=elapsed > args.deadline_minutes * 60,
        )
        print(f"--- {elapsed / 60:.0f} min: {v.state}", flush=True)
        for line in v.lines:
            print("   ", line, flush=True)
        if v.state != "wait":
            return 0 if v.state == "pass" else 1
        time.sleep(args.interval)


# --- command line --------------------------------------------------------------------


def _git(*a: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *a], check=True, capture_output=True, text=True
    ).stdout


def _show(rev: str, path: str) -> str:
    """`path` at `rev`, or "" when it did not exist there."""
    try:
        return _git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return ""


def changed_files(base: str, head: str | None) -> list[str]:
    """``git diff base...head``; with no head, the merge base against the
    working tree (tracked files, committed or not)."""
    if head is None:
        rng = [_git("merge-base", base, "HEAD").strip()]
    else:
        rng = [f"{base}...{head}"]
    return [f for f in _git("diff", "--name-only", *rng).splitlines() if f]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="print the PR's required jobs as JSON")
    s.add_argument("--base", required=True, help="base commit (the PR's base SHA)")
    s.add_argument("--head", help="head commit (default: the working tree)")
    s.add_argument(
        "--files",
        help="a file listing the changed paths (default: git diff base...head)",
    )
    s.add_argument("--out", help="also write the JSON here")
    w = sub.add_parser("wait", help="wait for the selected checks on a SHA")
    w.add_argument("--selection", required=True)
    w.add_argument("--repo", required=True)
    w.add_argument("--sha", required=True)
    w.add_argument("--own-run-id")
    w.add_argument("--interval", type=int, default=180)
    w.add_argument("--appear-minutes", type=float, default=30)
    w.add_argument("--deadline-minutes", type=float, default=150)
    args = ap.parse_args(argv)
    if args.cmd == "wait":
        if args.interval < 120:
            ap.error("--interval below 120 s would spend the API rate limit")
        return wait(args)

    files = (
        Path(args.files).read_text().split()
        if args.files
        else changed_files(args.base, args.head)
    )
    merge_base = _git("merge-base", args.base, args.head or "HEAD").strip()
    if args.head is None:
        head_reg = (ROOT / REGISTRY).read_text(encoding="utf-8")
    else:
        head_reg = _show(args.head, REGISTRY)
    base_reg = _show(merge_base, REGISTRY)
    workflows = load_workflows()
    sel = select(
        files, head_registry=head_reg, base_registry=base_reg, workflows=workflows
    )
    text = json.dumps({"files": files, **selection_json(sel, workflows)}, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
