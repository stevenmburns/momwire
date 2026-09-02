#!/usr/bin/env bash
# Drive one momwire portal entry point through SimNECDriver in the launch
# shapes that differ between "it works from my terminal" and "SimNEC says
# NEC Failure Code": the terminal's environment, a GUI app's near-empty
# environment, a path with a space, a name-selected engine, and a name that
# selects nothing (which must fail, with the exit code and message shown).
#
#     bash scripts/simnec_driver/run_shapes.sh /abs/path/momwire-nec2c-bspline [deck ...]
#
# Exit status is the number of shapes that did not behave.
set -u

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
driver="$here/SimNECDriver.java"
engine=${1:?engine path}
shift
decks=("$@")
if [[ ${#decks[@]} -eq 0 && -d "$here/../../tests/fixtures/nec_portal" ]]; then
  decks=("$here/../../tests/fixtures/nec_portal/dipole_free_space.deck"
         "$here/../../tests/fixtures/nec_portal/dipole_rp_pattern.deck")
fi

java=$(command -v java)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
bad=0

shape() { # shape <name> <expected: pass|fail> <command...>
  local name=$1 expect=$2; shift 2
  echo "################  $name  (expect $expect)"
  "$@"
  local rc=$?
  echo
  if [[ $expect == pass && $rc -ne 0 ]] || [[ $expect == fail && $rc -eq 0 ]]; then
    echo "!!!!!!!!!!!!!!!!  $name: expected $expect, driver exited $rc"
    bad=$((bad + 1))
  fi
  echo
}

# 1. The terminal's own environment.
shape "terminal environment" pass \
  "$java" "$driver" "$engine" "${decks[@]}" --repeat 2

# 2. What a GUI-launched app on macOS inherits: no shell rc, the stock PATH.
shape "GUI-launch environment" pass \
  env -i HOME="$HOME" USER="${USER:-runner}" LOGNAME="${LOGNAME:-runner}" \
      SHELL=/bin/sh TMPDIR="${TMPDIR:-/tmp}" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
      "$java" "$driver" "$engine" "${decks[@]}"

# 3. A path with a space, which SimNEC double-quotes on this route.
mkdir -p "$work/dir with space"
ln -s "$engine" "$work/dir with space/momwire-nec2c-bspline"
shape "path with a space" pass \
  "$java" "$driver" "$work/dir with space/momwire-nec2c-bspline" "${decks[@]}"

# 4. Selecting a different basis by the file name alone (momwire#528).
ln -s "$engine" "$work/momwire-nec2c-razor-2p"
shape "name-selected engine (razor-2p)" pass \
  "$java" "$driver" "$work/momwire-nec2c-razor-2p" "${decks[@]}"

# 5. A name that selects nothing: momwire exits 3 at the probe and says why.
ln -s "$engine" "$work/momwire-nec2c-bogus"
shape "name selecting no basis" fail \
  "$java" "$driver" "$work/momwire-nec2c-bogus" --probe-only

# 6. A dead interpreter: the shebang points at a Python that is gone.
printf '#!%s/no/such/python\n' "$work" > "$work/momwire-nec2c-dead"
chmod +x "$work/momwire-nec2c-dead"
shape "entry point whose interpreter is gone" fail \
  "$java" "$driver" "$work/momwire-nec2c-dead" --probe-only

if [[ $bad -eq 0 ]]; then echo "ALL SHAPES BEHAVED"; else echo "$bad SHAPE(S) MISBEHAVED"; fi
exit $bad
