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
#include <vector>

namespace mw {
namespace exe {

// The largest solve (basis functions) the deck's served runs prepare, or -1
// when the structure half does not serve it (the deck then goes elsewhere).
// `open_end`: the text is a stream's piece (momwire#1512). `crossing_pairs`,
// when given: the largest buried crossing block a run fills, its above
// segments times its below segments (0 for a deck with none), the size
// policy's second measure (`threads_for`). `below_pairs`, when given: the
// largest wholly-buried (below-plane) run's segment pairs, its segments
// squared (0 for a deck with none), the third.
// `buried_runs`, when given: how many of the served runs are buried ones
// (a buried stepping FR's frequencies), the frequency-parallel policy's
// measure (`freq_workers_for`).
long long largest_solve(const std::string &latin1_text, bool open_end = false,
                        long long *crossing_pairs = nullptr, long long *below_pairs = nullptr,
                        long long *buried_runs = nullptr);

// A buried stepping FR's frequencies solved up to `n` at once, each on one
// thread (`nec5::set_frequency_workers`); 1 solves them in turn.
void set_frequency_workers(int n);
// The frequencies this copy has solved concurrently (`nec5::frequencies_presolved`).
long long frequencies_presolved();
// Whether this copy takes a sweep with the features it was asked for
// (`nec5::frequency_sweep_allowed`).
bool frequency_sweep_allowed();

// The Sommerfeld grid's persistent store for this copy (serial or threaded):
// `tag` is the executable's identity, its version and build
// (`somm_core::configure_disk_cache`).
void configure_grid_cache(const std::string &tag);

// `momwire-nec5-razor-2p`'s printout for this deck, on the executable's own
// OpenBLAS (the `sys._mw_host_lapack` route). True: `printout` holds the
// file's bytes. False: declined, `reason` says why. With `arrays_dir`, each
// finished run's arrays are written there (the gate's).
// `failed`: declined because a C++ exception escaped (`failure_kind` is
// "MemoryError" for std::bad_alloc, else "RuntimeError"; `failure_message`
// its what()), which the no-Python build reports as an INTERNAL ERROR.
// `open_end`: the text is a stream's piece (momwire#1512, `razor_printout`);
// `served`: every structure solved and printed (no refusal, nothing raised),
// so a stream goes on after it.
// The experimental features (`momwire._features` names) the executable
// accepted from its environment, for this copy's answers from now on: each
// asked of the buried fills (`ni_eval::request_feature`) and named on line 2
// of every printout unless `stamped` is false (a default-on feature, which
// line 2 does not name). False when this copy carries a name not (nothing of
// that name is asked).
bool configure_features(const std::vector<std::string> &names, bool stamped = true);
// The default-on features (requested from the start, `ni_eval`) that this
// executable runs off: set off in its environment, or outside its reach.
bool withdraw_features(const std::vector<std::string> &names);
// razor-2p's fill-pair reuse plans (momwire#1971, `razor::reuse::cache_counters`)
// this copy has built, how many of those built came back declined, and how
// many it found, process totals: the MW_EXE_TIMES gate line's, so a test can
// tell the reuse ran (built - declined >= 1: a declined plan fills as off;
// nec5-bspline's and nec42's routes never build one).
void reuse_plans(long long &built, long long &declined, long long &found);

bool answer(const std::string &latin1_text, const std::string &version, const std::string &variant,
            const std::string &arrays_dir, std::string &printout, std::string &reason,
            bool *failed = nullptr, std::string *failure_kind = nullptr,
            std::string *failure_message = nullptr, std::string *raised_kind = nullptr,
            bool open_end = false, bool *served = nullptr);

}  // namespace exe
}  // namespace mw
