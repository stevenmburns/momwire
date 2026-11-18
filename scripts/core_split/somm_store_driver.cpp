// The executables' Sommerfeld store with no solver in front of it
// (momwire#2014): one process of one build, reading and writing tables the way
// a run does, so a test can stand several builds side by side in seconds.
//
//     somm_store_driver TAG ROUNDS TABLES KB SEED
//
// Each round asks for one of TABLES tables of KB kibibytes: a read that comes
// back whole is a hit, a missing file is a miss and writes it. A file that
// comes back with the wrong bytes is `corrupt`, which the store's contract
// says cannot happen. Prints one line: the counters, as JSON.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "_somm_grid_core.h"

namespace {

std::string table_bytes(const std::string &name, size_t kb) {
    std::string b = name + "\n";
    b.resize(kb * 1024 - sizeof(uint64_t), '\xa5');
    const uint64_t sum = somm_core::disk_hash(b);
    b.append(reinterpret_cast<const char *>(&sum), sizeof sum);
    return b;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: somm_store_driver TAG ROUNDS TABLES KB SEED\n");
        return 2;
    }
    const std::string tag = argv[1];
    const long rounds = std::atol(argv[2]);
    const int tables = std::atoi(argv[3]);
    const size_t kb = static_cast<size_t>(std::atol(argv[4]));
    std::mt19937 pick(static_cast<unsigned>(std::atol(argv[5])));
    somm_core::configure_disk_cache(tag);
    uint64_t hits = 0, misses = 0, corrupt = 0;
    for (long r = 0; r < rounds; r++) {
        const std::string name = "master-" + std::to_string(pick() % tables) + ".grid";
        const std::string want = table_bytes(name, kb);
        std::string got;
        if (somm_core::disk_read(name, got)) {
            if (got == want)
                hits++;
            else
                corrupt++;
        } else {
            misses++;
            somm_core::disk_write(name, want);
        }
    }
    const somm_core::DiskStats s = somm_core::disk_stats();
    std::printf(
        "{\"hits\": %llu, \"misses\": %llu, \"corrupt\": %llu, \"evicted\": %llu, \"retired\": %llu}\n",
        static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses),
        static_cast<unsigned long long>(corrupt), static_cast<unsigned long long>(s.evicted),
        static_cast<unsigned long long>(s.retired));
    return 0;
}
