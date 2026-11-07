"""Triage one Skylake night against the previous green night on the same box.

    python triage.py NIGHT_DIR [--base GREEN_NIGHT_DIR] [--known known.json]

Writes NIGHT_DIR/triage.json and prints a summary. The severity is PROPOSED,
never acted on: nothing here reverts, files or merges anything.

    severe    revert within the day
    moderate  open an issue, fix before the next release
    minor     add to the known list (known.json)
    known     matched an entry of the known list; does not turn a night red

A night is red when any finding is severe or moderate.

Findings, by kind (cell = <variant>-T<threads>):

    mismatch            the executable SERVED (route cxx:*) and its printout is not
                        Python's                                         severe
    exe-declines        the executable declined a deck Python serves     moderate,
                        severe when the green night's executable served it
    exe-no-answer       the executable timed out or wrote no printout    moderate,
                        severe when new
    py-error            Python raised past the seam, or timed out        moderate,
                        severe when new
    variant             Python ran a variant other than the cell's       severe
    py-drift            (avx2-T1) Python's printout moved from the green night:
                        text or a number beyond Z_REL                    severe
                        a number within Z_REL                            moderate
                        only last-digit flips (a rounding boundary)      minor
    sweep-outcome       a sweep case ok -> not ok                        severe
                        any other outcome change                         moderate
    sweep-z             Z moved by more than Z_REL (relative)            severe
                        by less, but moved                               moderate
    sweep-mem/-time     analyze.py's MEM and TIME thresholds             moderate
    harness             a phase of the night failed (build, a cell, the sweep)
                                                                         severe
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# "Z moves above a few x 1e-12" (Steve, 2026-11-06: "probably tight but we
# will start with that"). No repo constant names this bar.
Z_REL = 5e-12
# analyze.py's MEM and TIME thresholds (antennaknobs scratch/pretag-sweep).
WALL_REL, WALL_ABS = 0.10, 0.5
RSS_REL, RSS_ABS = 0.10, 30.0

RECOMMEND = {
    "severe": "revert within the day",
    "moderate": "open an issue; fix before the next release",
    "minor": "add to the known list",
    "known": "on the known list",
}
RED = ("severe", "moderate")


def load_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def audit_rows(night: Path) -> dict[str, dict[tuple[str, str], dict]]:
    """{cell: {(exe, deck): row}}"""
    out = {}
    for p in sorted(night.glob("audit-*.jsonl")):
        cell = p.stem[len("audit-") :]
        out[cell] = {(r["exe"], r["deck"]): r for r in load_jsonl(p)}
    return out


class Printouts:
    """A night's saved avx2-T1 Python printouts, from the directory or the
    tarball the night was packed into."""

    def __init__(self, night: Path):
        self.dir = night / "py-avx2-T1"
        self.tar = night / "py-avx2-T1.tar.xz"
        self._tf = None

    def get(self, key: str) -> bytes | None:
        p = self.dir / f"{key}.py"
        if p.is_file():
            return p.read_bytes()
        if self.tar.is_file():
            if self._tf is None:
                self._tf = tarfile.open(self.tar)
            try:
                return self._tf.extractfile(f"py-avx2-T1/{key}.py").read()
            except KeyError:
                return None
        return None


def _ratio(ref: bytes, got: bytes):
    # beside this file in the installed harness (nightly.py copies it there),
    # in scripts/core_split in a checkout
    sys.path[:0] = [str(HERE), str(HERE.parent / "core_split")]
    import printout_tolerance as pt

    return pt.ratio(ref, got)


def finding(kind, severity, detail, **where) -> dict:
    f = dict(kind=kind, severity=severity, detail=detail, **where)
    f["id"] = "|".join(
        str(where.get(k, "")) for k in ("cell", "exe", "deck", "case")
    ).strip("|")
    f["id"] = f"{kind}|{f['id']}"
    return f


def triage_audit(night: Path, base: Path | None) -> list[dict]:
    rows = audit_rows(night)
    brows = audit_rows(base) if base else {}
    out = []
    for cell, recs in rows.items():
        variant = cell.split("-")[0]
        bcell = brows.get(cell, {})
        for (exe, deck), r in sorted(recs.items()):
            w = dict(cell=cell, exe=exe, deck=deck)
            b = bcell.get((exe, deck))
            route = r.get("route", "none")
            if r.get("py_variant") != variant:
                out.append(
                    finding(
                        "variant",
                        "severe",
                        f"Python ran {r.get('py_variant')}, not {variant}",
                        **w,
                    )
                )
            if r.get("py_error"):
                new = b is not None and not b.get("py_error")
                out.append(
                    finding(
                        "py-error",
                        "severe" if new else "moderate",
                        r["py_error"].split(":")[0],
                        **w,
                    )
                )
                continue
            if route in ("none", "timeout") or r.get("exe_outcome") == "empty":
                new = b is not None and b.get("route") not in ("none", "timeout")
                out.append(
                    finding(
                        "exe-no-answer",
                        "severe" if new else "moderate",
                        f"route {route}, rc {r.get('exe_rc')}",
                        **w,
                    )
                )
                continue
            if route.startswith("refused:"):
                if r.get("py_outcome") == "served":
                    was_served = b is not None and b.get("route", "").startswith("cxx")
                    out.append(
                        finding(
                            "exe-declines",
                            "severe" if was_served else "moderate",
                            route[len("refused:") :][:120]
                            + (
                                "; the green night's executable served it"
                                if was_served
                                else ""
                            ),
                            **w,
                        )
                    )
            elif not r.get("identical_masked"):
                det = f"first differing line {r.get('diff_line')}"
                det += f"; exe {r.get('exe_outcome')}, Python {r.get('py_outcome')}"
                out.append(finding("mismatch", "severe", det, **w))
    # Python drift night over night: the every-night cell only.
    cell = "avx2-T1"
    if base and cell in rows and cell in brows:
        tonight, green = Printouts(night), Printouts(base)
        for key2, r in sorted(rows[cell].items()):
            b = brows[cell].get(key2)
            if b is None or r.get("py_error") or b.get("py_error"):
                continue
            if r["py_sha"] == b["py_sha"]:
                continue
            w = dict(cell=cell, exe=key2[0], deck=key2[1])
            ref, got = green.get(b.get("key", "")), tonight.get(r.get("key", ""))
            ratio = _ratio(ref, got) if ref is not None and got is not None else None
            if ratio is None:
                sev, det = "severe", "the printout's text changed"
            elif ratio > Z_REL:
                sev, det = "severe", f"a number moved {ratio:.2e} (bar {Z_REL:g})"
            elif ratio > 0:
                sev, det = "moderate", f"a number moved {ratio:.2e}, within {Z_REL:g}"
            else:
                sev, det = (
                    "minor",
                    "last printed digits flipped only (a rounding boundary)",
                )
            out.append(finding("py-drift", sev, det, **w))
    return out


def _zrel(zb, zc) -> float:
    if zb is None or zc is None or len(zb) != len(zc):
        return float("inf")
    worst = 0.0
    for (br, bi), (cr, ci) in zip(zb, zc, strict=True):
        b, c = complex(br, bi), complex(cr, ci)
        worst = max(worst, abs(c - b) / max(abs(b), 1e-300))
    return worst


def sweep_rows(night: Path) -> dict[str, dict]:
    return {r["id"]: r for r in load_jsonl(night / "sweep.jsonl")}


def triage_sweep(night: Path, base: Path | None, meta: dict, bmeta: dict) -> list[dict]:
    cand = sweep_rows(night)
    if not cand or not base:
        return []
    if meta.get("ak_sha") != bmeta.get("ak_sha"):
        return []  # a new antennaknobs pin re-baselines the sweep (night.json says so)
    old = sweep_rows(base)
    out = []
    for cid in sorted(set(old) & set(cand)):
        b, c = old[cid], cand[cid]
        w = dict(case=cid)
        if b["outcome"] != c["outcome"]:
            sev = "severe" if b["outcome"] == "ok" else "moderate"
            msg = (c.get("message") or "")[:120]
            out.append(
                finding(
                    "sweep-outcome",
                    sev,
                    f"{b['outcome']} -> {c['outcome']}: {msg}",
                    **w,
                )
            )
            continue
        if c["outcome"] != "ok":
            continue
        zr = _zrel(b.get("z"), c.get("z"))
        if zr > Z_REL:
            out.append(finding("sweep-z", "severe", f"Z moved {zr:.2e} relative", **w))
        elif zr > 0:
            out.append(
                finding(
                    "sweep-z", "moderate", f"Z moved {zr:.2e}, within {Z_REL:g}", **w
                )
            )
        if b.get("mode") != c.get("mode"):
            continue
        bm, cm = b.get("ru_maxrss_mb") or 0, c.get("ru_maxrss_mb") or 0
        if cm > bm * (1 + RSS_REL) and cm - bm > RSS_ABS:
            out.append(
                finding("sweep-mem", "moderate", f"{bm:.0f} -> {cm:.0f} MB", **w)
            )
        bw, cw = b.get("wall_s") or 0, c.get("wall_s") or 0
        if c.get("mode") == "timed" and cw > bw * (1 + WALL_REL) and cw - bw > WALL_ABS:
            out.append(
                finding("sweep-time", "moderate", f"{bw:.2f} -> {cw:.2f} s", **w)
            )
    return out


def load_known(p: Path) -> list[dict]:
    return json.loads(p.read_text())["known"] if p.is_file() else []


def apply_known(findings: list[dict], known: list[dict]) -> None:
    for f in findings:
        for k in known:
            if k["kind"] != f["kind"]:
                continue
            target = f.get("deck") or f.get("case") or ""
            if not fnmatch.fnmatchcase(target, k.get("match", "*")):
                continue
            if not fnmatch.fnmatchcase(f.get("exe", ""), k.get("exe", "*")):
                continue
            if not fnmatch.fnmatchcase(f.get("cell", ""), k.get("cell", "*")):
                continue
            f["proposed"] = f["severity"]
            f["severity"] = "known"
            f["known_note"] = k.get("note", "")
            break


def triage(night: Path, base: Path | None, known: Path) -> dict:
    meta = json.loads((night / "night.json").read_text())
    bmeta = json.loads((base / "night.json").read_text()) if base else {}
    findings = []
    for phase, err in (meta.get("errors") or {}).items():
        findings.append(
            finding("harness", "severe", f"{phase}: {err}"[:300], case=phase)
        )
    findings += triage_audit(night, base)
    findings += triage_sweep(night, base, meta, bmeta)
    apply_known(findings, load_known(known))
    for f in findings:
        f["recommend"] = RECOMMEND[f["severity"]]
    red = any(f["severity"] in RED for f in findings)
    counts = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    result = dict(
        status="red" if red else "green",
        base=bmeta.get("sha"),
        base_date=bmeta.get("date"),
        counts=counts,
        findings=findings,
    )
    (night / "triage.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("night", type=Path)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--known", type=Path, default=HERE / "known.json")
    a = ap.parse_args()
    r = triage(a.night, a.base, a.known)
    print(f"{r['status']}: {r['counts']} (base {r['base']})")
    for f in r["findings"]:
        print(f"  {f['severity']:8} {f['id']}: {f['detail']}")
    return 1 if r["status"] == "red" else 0


if __name__ == "__main__":
    sys.exit(main())
