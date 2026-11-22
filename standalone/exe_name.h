// The fat executable's own file name, read for the variant a deprecated
// name asks for (momwire#2090).
//
// Up to 0.96.0 the Windows zip shipped each engine twice, one build per
// variant, as `momwire-nec5-<basis>-avx2.exe` and `-sse2.exe`. From 0.97.0 it
// ships one fat program per engine, `momwire-nec5-<basis>.exe`, which picks
// the variant by CPU, and the old names as copies of it: a copy whose name
// ends in `-avx2` or `-sse2` runs that variant, as MOMWIRE_FORCE_VARIANT
// would (the variable wins when both are given), so an EZNEC, SimNEC or 4nec2
// engine path to an old name computes what it computed before. Line 2 of its
// printout then ends in "(renamed from <old name>)", as #2091's deprecated
// basis spellings end in theirs.
//
// Header-only and free of the build's other headers, so
// tests/test_standalone_exe_name.py compiles it on its own.
#pragma once

#include <string>

namespace mw_exe_name {

// "avx2" or "sse2" when the base name of `path` (either slash; ".exe" in any
// case dropped) ends in "-avx2" or "-sse2", in any case; "" otherwise.
inline std::string variant_from_path(const std::string &path) {
    const std::size_t slash = path.find_last_of("/\\");
    std::string stem = slash == std::string::npos ? path : path.substr(slash + 1);
    for (char &c : stem)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    const std::string exe = ".exe";
    if (stem.size() > exe.size() && stem.compare(stem.size() - exe.size(), exe.size(), exe) == 0)
        stem.resize(stem.size() - exe.size());
    for (const char *v : {"avx2", "sse2"}) {
        const std::string suffix = std::string("-") + v;
        if (stem.size() > suffix.size() && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
            return v;
    }
    return "";
}

// Line 2's note for an executable run under a deprecated variant name:
// "(renamed from momwire-nec5-razor2p-sse2)", or "" for no variant.
inline std::string deprecation_note(const std::string &launcher, const std::string &variant) {
    if (variant.empty()) return "";
    return "(renamed from " + launcher + "-" + variant + ")";
}

}  // namespace mw_exe_name
