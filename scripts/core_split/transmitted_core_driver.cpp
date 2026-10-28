// The transmitted Sommerfeld family with no Python (momwire#1718 S12): read
// requests on stdin and print `_somm_transmitted_core`'s answers, every
// float as a C99 hex literal, for `tests/test_razor_core_standalone.py` to
// compare with `_sommerfeld_transmitted`'s own, field for field.
//
//     transmitted_core_driver < requests
//
// Requests (whitespace-separated tokens, floats as hex or decimal):
//
//   grid EPS_R EPS_I K2 R_MAX ZP_MIN ZP_MAX OMEGA MU HAS_R_MIN R_MIN
//       -> "grid" and the state (`n_zp n_r n_th`, then `zp_min zp_max r_min
//          r_max th_min dth lnr0 lnz0 dlnz`), then every value (re im), C
//          order; or "refused <name>"
//   proj EPS_R EPS_I K2 R_MAX ZP_MIN ZP_MAX OMEGA MU R_MIN GZ K_P KM_R KM_I
//        TRANSPOSED A B, then A rows (x y z tx ty tz) above, B rows below
//       -> "proj <refusal or ->", then A*B values (re im), row major
//   plan GZ K_P KM_R KM_I NA NB, then NA rows (x y z) above, NB below
//       -> "plan r_cross_max r_cross_min zp_min zp_max" or "refused <name>"
#include "_somm_transmitted_core.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

double num(std::istream &in) {
    std::string t;
    in >> t;
    return std::strtod(t.c_str(), nullptr);
}

void hex(double x) { std::printf(" %a", x); }

}  // namespace

int main() {
    std::string cmd;
    while (std::cin >> cmd) {
        try {
            if (cmd == "grid") {
                const double er = num(std::cin), ei = num(std::cin), k2 = num(std::cin), rmax = num(std::cin);
                const double zlo = num(std::cin), zhi = num(std::cin), om = num(std::cin), mu = num(std::cin);
                const bool has = num(std::cin) != 0.0;
                const double rmin = num(std::cin);
                const somm_trans::Grid g =
                    somm_trans::make_grid({er, ei}, k2, rmax, zlo, zhi, om, mu, has, rmin, 0);
                std::printf("grid %lld %lld %lld", static_cast<long long>(g.n_zp), static_cast<long long>(g.n_r),
                            static_cast<long long>(g.n_th));
                for (double v : {g.zp_min, g.zp_max, g.r_min, g.r_max, g.th_min, g.dth, g.lnr0, g.lnz0, g.dlnz})
                    hex(v);
                for (const auto &v : g.vals) {
                    hex(v.real());
                    hex(v.imag());
                }
                std::printf("\n");
            } else if (cmd == "proj") {
                const double er = num(std::cin), ei = num(std::cin), k2 = num(std::cin), rmax = num(std::cin);
                const double zlo = num(std::cin), zhi = num(std::cin), om = num(std::cin), mu = num(std::cin);
                const double rmin = num(std::cin), gz = num(std::cin), kp = num(std::cin);
                const double kmr = num(std::cin), kmi = num(std::cin);
                const bool tr = num(std::cin) != 0.0;
                const size_t A = static_cast<size_t>(num(std::cin)), B = static_cast<size_t>(num(std::cin));
                std::vector<double> pa(3 * A), ta(3 * A), pb(3 * B), tb(3 * B);
                for (size_t i = 0; i < A; i++) {
                    for (int c = 0; c < 3; c++) pa[3 * i + c] = num(std::cin);
                    for (int c = 0; c < 3; c++) ta[3 * i + c] = num(std::cin);
                }
                for (size_t i = 0; i < B; i++) {
                    for (int c = 0; c < 3; c++) pb[3 * i + c] = num(std::cin);
                    for (int c = 0; c < 3; c++) tb[3 * i + c] = num(std::cin);
                }
                const somm_trans::Grid &g = somm_trans::get_grid({er, ei}, k2, rmax, zlo, zhi, om, mu, rmin, 0);
                std::vector<std::complex<double>> out(A * B);
                somm_trans::Extremes ext;
                somm_trans::project(g, pa.data(), ta.data(), A, pb.data(), tb.data(), B, gz, kp, {kmr, kmi}, tr,
                                    out.data(), ext);
                const std::string why = (A && B) ? somm_trans::eval_refusal(g, ext) : std::string();
                std::printf("proj %s", why.empty() ? "-" : why.c_str());
                for (const auto &v : out) {
                    hex(v.real());
                    hex(v.imag());
                }
                std::printf("\n");
            } else if (cmd == "plan") {
                const double gz = num(std::cin), kp = num(std::cin), kmr = num(std::cin), kmi = num(std::cin);
                const size_t na = static_cast<size_t>(num(std::cin)), nb = static_cast<size_t>(num(std::cin));
                std::vector<double> a(3 * na), b(3 * nb);
                for (double &v : a) v = num(std::cin);
                for (double &v : b) v = num(std::cin);
                const somm_trans::CrossPlan p = somm_trans::cross_extents(a.data(), na, b.data(), nb, gz, kp,
                                                                          {kmr, kmi});
                std::printf("plan");
                for (double v : {p.r_cross_max, p.r_cross_min, p.zp_min, p.zp_max}) hex(v);
                std::printf("\n");
            } else {
                std::cerr << "unknown request " << cmd << "\n";
                return 2;
            }
        } catch (const somm_trans::Refused &e) {
            std::printf("refused %s\n", e.what());
        }
        std::fflush(stdout);
    }
    return 0;
}
