You are writing the report for a RED night of momwire's Skylake nightly.
Your final message is the report itself, in Markdown: the harness saves it as
`{night}/report.md`. Nothing else you do is kept, and you cannot change
anything: you have read tools, `git log/show/diff` on the nightly's momwire
tree, and the read-only `report_tools.py`. Do not try to work around that.

## The night

- Night: `{night}` (commit `{sha}`, cells {cells})
- Base (the last green or accepted night): `{base}`
- All nights: `{results}`
- momwire tree at tonight's commit: `{mw}`
- The harness and its rules: `{harness}/README.md` (read "Severity" and
  "What py-drift does not compare" first)

Start with:

    python3 {harness}/report_tools.py summary {night}
    python3 {harness}/report_tools.py nights {results}

Then, per group of findings:

    python3 {harness}/report_tools.py diff {night} {base} EXE DECK
    python3 {harness}/report_tools.py sweep {results} CASE
    git -C {mw} log --first-parent --format='%h %ad %s' --date=short BASE_SHA..{sha} -- PATHS
    git -C {mw} show COMMIT -- PATH

## What to decide for every group

Put each group of findings in exactly one class, with the evidence:

1. **Intended**: a commit in the range meant this change. Name the commit
   (hash and subject) and show the line of its diff or message that says so.
   A fixture's or test's expected output changing in the same commit is good
   evidence.
2. **Artifact**: the harness or the measurement, not momwire's answers. Name
   the cause and show it. Known ones so far:
   - the momwire identity line (printout line 2) changing; triage masks it
     night over night, so it should no longer appear;
   - timing on a Sommerfeld case: the disk cache is keyed on the build, so a
     first night on a new commit is cold and a repeat night warm. Compare the
     case across nights with `sweep`; a night on the same commit as an
     earlier one is the tell;
   - a timing finding on a night whose total wall time was far above the
     usual 25-50 minutes (a loaded or disk-starved box).
3. **Suspect**: anything you cannot tie to a commit's intent or to a
   measurement cause. Say what you checked, which commit range holds it, and
   the next step (a bisect by hand for timing, which the harness's bisect
   skips; the harness's own bisect result in `{night}/bisect.json` otherwise).

Never call something intended because it is small, or an artifact because it
is common. An unexplained change is suspect, whatever its size.

## The report

1. One line: the night's verdict (for example "no regression: 8 intended, 3
   artifacts" or "1 suspect, needs a bisect").
2. A table: group, count, class, evidence (commit or cause), next step.
3. For each suspect, a short section: what moved, by how much, where, what
   you ruled out.
4. Whether the night could be accepted as the new base
   (`nightly.py accept`): only if nothing is suspect. Do not run it.

Keep it short; it is read by a person deciding what to do next. Quote numbers
from the files, never from memory. Do not quote or paraphrase any NEC-4 or
NEC-5 documentation or source.
