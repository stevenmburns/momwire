// The C++ route's entry points, in std types only. route.cpp is compiled
// TWICE: as `mw::exe` with OpenMP (the threaded fill), and as `mws::exe` with
// `-Dmw=mws` and no OpenMP at all, so a one-thread answer never enters the
// OpenMP runtime (its first use costs 2-8 ms on Windows). Both copies are the
// same source with the same arithmetic; the corpus gate holds each to the
// launcher's bytes (1 thread: the serial copy; more: the threaded one).
//
// No include guard on purpose: momwire_razor2p.cpp includes it once per
// namespace (the second time with `mw` defined as `mws`).
#include <string>

namespace mw {
namespace exe {

// The largest solve (basis functions) the deck's served runs prepare, or -1
// when the structure half does not serve it (the deck then goes elsewhere).
long long largest_solve(const std::string &latin1_text);

// The Sommerfeld grid's persistent store for this copy (serial or threaded):
// `tag` is the executable's identity, its version and build
// (`somm_core::configure_disk_cache`).
void configure_grid_cache(const std::string &tag);

// `momwire-nec5-razor-2p`'s printout for this deck, on the executable's own
// OpenBLAS (the `sys._mw_host_lapack` route). True: `printout` holds the
// file's bytes. False: declined, `reason` says why. With `arrays_dir`, each
// finished run's arrays are written there (the gate's).
bool answer(const std::string &latin1_text, const std::string &version, const std::string &variant,
            const std::string &arrays_dir, std::string &printout, std::string &reason);

}  // namespace exe
}  // namespace mw
