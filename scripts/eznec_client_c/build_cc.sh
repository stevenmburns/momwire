#!/bin/sh
# Build the native EZNEC thin client with a POSIX cc (momwire#718 phase 3).
#
#     scripts/eznec_client_c/build_cc.sh <output-path>
#
# One compiler invocation and no build system: the client is a single
# translation unit with no dependency beyond libc, which is the property that
# makes it start in milliseconds and the property a Makefile would only
# obscure.  The Windows half of the same source is built by build_msvc.bat.
#
# The momwire version is a compile-time define because it is a HASH INPUT
# (the server key is scoped by `eznec.<major>.<minor>`), so the exe must
# carry the version of the tree that built it and must not go looking for one
# at run time.  Read from the installed distribution's metadata first — that
# is what `momwire_serve_client.dist_version` reads — with pyproject.toml
# behind it for a source tree with nothing installed.
#
# -Werror because every warning this program can raise is about a buffer or a
# conversion, and both are the failure modes of a process that runs as an
# engine on somebody else's machine.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
out=${1:-momwire-eznec}
cc=${CC:-cc}
python=${PYTHON:-python3}

version=$("$python" -c 'from importlib.metadata import version; print(version("momwire"))' 2>/dev/null || true)
if [ -z "$version" ]; then
    version=$(sed -n 's/^version = "\([^"]*\)".*/\1/p' "$here/../../pyproject.toml" | head -1)
fi
if [ -z "$version" ]; then
    echo "build_cc.sh: cannot determine the momwire version" >&2
    exit 1
fi

major=${version%%.*}
rest=${version#*.}
minor=${rest%%.*}
case "$major$minor" in
    *[!0-9]*|"") echo "build_cc.sh: unusable version '$version'" >&2; exit 1 ;;
esac

exec "$cc" -std=c11 -Wall -Wextra -Werror -O2 \
    -DMOMWIRE_VERSION_MAJOR="$major" \
    -DMOMWIRE_VERSION_MINOR="$minor" \
    -o "$out" "$here/momwire_eznec_client.c"
