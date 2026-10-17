// The exact kernel's coaxial pair moments with no Python
// (`_exact_kernel_coaxial_core.h`, momwire#1517 step 0): reads one group's
// inputs from DIR, writes the C++ core's answers there for
// `tests/test_razor_core_standalone.py` to hold to the extension's bytes.
//
//   DIR/x0.bin sgn.bin h.bin           float64, n each
//   DIR/xg.bin wg.bin xs.bin ws.bin wls.bin   the quadrature rules
//   DIR/z.bin                          float64, ring kernel points
//   DIR/params.txt   a k nd max_cached q symmetric n_threads r0 r1
//   ->  DIR/drv_rows.bin   (nd, nd, r1 - r0, n) complex128, C order, for
//                          rows [0, r0) then [r0, r1) (the cache warm)
//       DIR/drv_pm.bin     every (i, j)'s pair moments from the rows'
//                          geometry, nd x nd each, far = -1, 0, 1
//       DIR/drv_ring.bin   K(z); DIR/drv_split_a.bin / drv_split_b.bin
//                          K's split at the points with |z| < 6a
//       DIR/drv_counts.txt evaluated cached
#include "_exact_kernel_coaxial_core.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using mw::ek::cd;

static std::vector<double> read_doubles(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<double> v(b.size() / sizeof(double));
    std::copy(b.data(), b.data() + v.size() * sizeof(double), reinterpret_cast<char *>(v.data()));
    return v;
}

static void write_cd(const std::string &path, const std::vector<cd> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(cd)));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: exact_kernel_coaxial_driver DIR\n";
        return 2;
    }
    const std::string dir = argv[1];
    const std::vector<double> x0 = read_doubles(dir + "/x0.bin"), sgn = read_doubles(dir + "/sgn.bin"),
                              h = read_doubles(dir + "/h.bin"), z = read_doubles(dir + "/z.bin");
    mw::ek::PointRules rules;
    rules.xg = read_doubles(dir + "/xg.bin");
    rules.wg = read_doubles(dir + "/wg.bin");
    rules.xs = read_doubles(dir + "/xs.bin");
    rules.ws = read_doubles(dir + "/ws.bin");
    rules.wls = read_doubles(dir + "/wls.bin");
    double a, k, q;
    int nd, symmetric, n_threads;
    long long max_cached, r0, r1;
    {
        std::ifstream f(dir + "/params.txt");
        f >> a >> k >> nd >> max_cached >> q >> symmetric >> n_threads >> r0 >> r1;
        if (!f) {
            std::cerr << "bad params.txt\n";
            return 2;
        }
    }
    try {
        const int64_t n = static_cast<int64_t>(h.size());
        mw::ek::CoaxialRows rows(x0.data(), sgn.data(), h.data(), n, a, k, nd, max_cached, q,
                                 symmetric != 0, rules, n_threads);
        std::vector<cd> out(static_cast<size_t>(nd * nd * r0 * n)), out2(static_cast<size_t>(nd * nd * (r1 - r0) * n));
        rows.rows(0, r0, out.data());
        rows.rows(r0, r1, out2.data());
        out.insert(out.end(), out2.begin(), out2.end());
        write_cd(dir + "/drv_rows.bin", out);
        {
            std::ofstream c(dir + "/drv_counts.txt");
            c << rows.evaluated() << " " << rows.cached() << "\n";
        }
        std::vector<cd> pm;
        std::vector<cd> m(static_cast<size_t>(nd * nd));
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++)
                for (int far = -1; far <= 1; far++) {
                    const double c = x0[j] - x0[i];
                    mw::ek::coaxial_pair_moments(h[i], h[j], c, sgn[i], sgn[j], a, k, nd, rules, far,
                                                 m.data());
                    pm.insert(pm.end(), m.begin(), m.end());
                }
        write_cd(dir + "/drv_pm.bin", pm);
        std::vector<cd> ring(z.size());
        mw::ek::ring_kernel(z.data(), static_cast<int64_t>(z.size()), a, k, ring.data());
        write_cd(dir + "/drv_ring.bin", ring);
        std::vector<double> near;
        for (double v : z)
            if (std::fabs(v) < 6.0 * a) near.push_back(v);
        std::vector<cd> A(near.size()), B(near.size());
        mw::ek::ring_kernel_split(near.data(), static_cast<int64_t>(near.size()), a, k, A.data(),
                                  B.data());
        write_cd(dir + "/drv_split_a.bin", A);
        write_cd(dir + "/drv_split_b.bin", B);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
