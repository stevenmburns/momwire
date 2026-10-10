"""Outcome diff of two junit XML files: tests whose outcome differs, and
tests present in one only. usage: junit_diff.py BASE.xml CHANGE.xml"""

import sys
import xml.etree.ElementTree as ET


def outcomes(p):
    out = {}
    for tc in ET.parse(p).getroot().iter("testcase"):
        key = f"{tc.get('classname')}::{tc.get('name')}"
        kids = {c.tag for c in tc}
        out[key] = (
            "failed"
            if kids & {"failure", "error"}
            else "skipped"
            if "skipped" in kids
            else "passed"
        )
    return out


a, b = outcomes(sys.argv[1]), outcomes(sys.argv[2])
moved = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
for k in moved:
    print(f"MOVED {k}: {a[k]} -> {b[k]}")
only_b = sorted(b.keys() - a.keys())
only_a = sorted(a.keys() - b.keys())
print(
    f"base {len(a)}  change {len(b)}  moved {len(moved)}  new {len(only_b)}  gone {len(only_a)}"
)
for k in only_a:
    print("GONE", k)
print("new by file:", sorted({k.split("::")[0] for k in only_b}))
