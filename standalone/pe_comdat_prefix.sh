#!/bin/bash
# Give a prefixed OpenBLAS archive's PE COMDAT sections their symbols' new
# names (MinGW-w64 only; windows-static.yml):
#
#     standalone/pe_comdat_prefix.sh ARCHIVE PREFIX
#
# build_openblas.sh renames every global symbol `<prefix><name>`, and on ELF
# that is the whole job. On PE, gcc puts each `.refptr.<var>` stub (the
# indirection it emits for external data) in its own COMDAT section NAMED
# after the symbol, `.rdata$.refptr.<var>`, and GNU ld folds COMDAT sections
# by that name across the whole link. Both archives carry the same names
# (`.rdata$.refptr.gotoblas`, ...), so the second archive's sections were
# dropped as duplicates of the first's and its renamed stubs
# (`mwn_.refptr.gotoblas`, ...) left undefined. Here each section `<base>$<s>`
# whose `<prefix><s>` the archive defines becomes `<base>$<prefix><s>`.
# The section's own COMDAT symbol (same name) is renamed with it. Sections
# only: no code or data changes. Idempotent (a renamed section's suffix
# already carries the prefix, so `<prefix><prefix><s>` matches nothing).
#
# Also: the `.drectve` sections go. OpenBLAS's objects ask the linker to
# `-export:` a few entry points (zgemm_, openblas_set_num_threads, ...) by
# their UNprefixed names, which no longer exist, so the link fails with
# "cannot export". An executable exports nothing anyway. The script first
# checks that `-export:` is every directive there (an `-aligncomm:` would
# matter); anything else fails it.
set -euo pipefail
A=${1:?usage: pe_comdat_prefix.sh ARCHIVE PREFIX}
P=${2:?usage: pe_comdat_prefix.sh ARCHIVE PREFIX}
T=$(mktemp -d)
nm -g --defined-only "$A" 2>/dev/null | awk 'NF == 3 { print $3 }' | sort -u > "$T/defined"
objdump -h "$A" | awk '$1 ~ /^[0-9]+$/ && $2 ~ /\$/ { print $2 }' | sort -u > "$T/sections"
awk -v p="$P" 'NR == FNR { def[$1] = 1; next }
    { i = index($0, "$"); s = substr($0, i + 1)
      if ((p s) in def) {
          print "--rename-section=" $0 "=" substr($0, 1, i) p s
          print "--redefine-sym=" $0 "=" substr($0, 1, i) p s
      } }' \
    "$T/defined" "$T/sections" > "$T/args"
# the linker directives: -export: only (see above), then dropped
objdump -s -j .drectve "$A" 2>/dev/null | awk '/^ [0-9a-f]+ / { print substr($0, 44) }' | tr -d '\n' |
    grep -o -- '-[a-z]*:' | sort | uniq -c > "$T/directives" || true
cat "$T/directives"
if grep -qv -- ' -export:$' "$T/directives"; then
    echo "$A: .drectve holds directives other than -export:" >&2
    exit 1
fi
if [ -s "$T/directives" ]; then
    echo "--remove-section=.drectve" >> "$T/args"
fi
n=$(wc -l < "$T/args")
if [ "$n" -gt 0 ]; then
    objcopy @"$T/args" "$A"
fi
echo "$A: $n objcopy arguments (COMDAT sections and their symbols renamed $P, .drectve dropped)"
rm -rf "$T"
