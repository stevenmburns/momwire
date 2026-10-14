// A standalone C++ program (no Python, no pybind11, no numpy) that runs one
// razor row window through three core kernels in sequence:
//
//     seg_moments -> (M0, M1) -> assemble_t1 -> t1 -> t2_rows -> rows of Z
//
// M0, M1 and t1 are std::vector buffers that never leave C++. The inputs a
// real RazorSolver solve handed the bound kernels for its first window are
// read from <dir> (written by capture_chain.py); the outputs are written back
// for a byte-for-byte comparison against what the Python chain produced.
//
// Build: link the SAME objects the extension links (the core TUs compiled by
// setup.py with the extension's flags), plus -fopenmp -lmvec.
#include "_razor_core.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

struct Arr {
    std::string dtype;
    std::vector<ptrdiff_t> shape;
    std::vector<char> bytes;
};

static std::map<std::string, Arr> load(const std::string &dir) {
    std::map<std::string, Arr> m;
    std::ifstream man(dir + "/manifest.txt");
    std::string line;
    while (std::getline(man, line)) {
        std::istringstream is(line);
        std::string name;
        Arr a;
        int nd;
        is >> name >> a.dtype >> nd;
        a.shape.resize(nd);
        for (int k = 0; k < nd; k++) is >> a.shape[k];
        std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
        a.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        m[name] = a;
    }
    return m;
}

template <class T, int N>
static mw::view<const T, N> V(const Arr &a) {
    if ((int)a.shape.size() != N) throw std::runtime_error("rank mismatch");
    ptrdiff_t shp[N > 0 ? N : 1], st[N > 0 ? N : 1];
    ptrdiff_t acc = sizeof(T);
    for (int k = N - 1; k >= 0; k--) {
        shp[k] = a.shape[k];
        st[k] = acc;
        acc *= a.shape[k];
    }
    return mw::view<const T, N>(reinterpret_cast<const T *>(a.bytes.data()), shp, st);
}

static double scalar(const Arr &a) { double d; std::memcpy(&d, a.bytes.data(), 8); return d; }

static void dump(const std::string &path, const void *p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char *>(p), n);
}

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "usage: driver <dir> [reps]\n"; return 2; }
    const std::string dir = argv[1];
    const int reps = argc > 2 ? std::atoi(argv[2]) : 1;
    auto A = load(dir);
    using cd = mw::cd;
    using namespace mw;

    const auto obs = V<double, 2>(A["sm_obs"]);
    const size_t n_obs = obs.shape(0), n_seg = A["sm_seg_h"].shape[0];
    const bool need_m1 = scalar(A["sm_need_m1"]) != 0.0;
    std::vector<cd> M0(n_obs * n_seg), M1(need_m1 ? n_obs * n_seg : 0);
    const size_t n_path = (size_t)scalar(A["t1_n_path"]);
    const size_t n_basis = A["t1_s_a"].shape[0];
    const size_t n_rows = A["t1_wts"].shape[0] / n_path;
    std::vector<cd> t1(n_rows * n_basis);
    // `out` column-major, as RazorSolver's Z is (a window of a Fortran array).
    std::vector<cd> rows(n_rows * n_basis);
    const ptrdiff_t oshape[2] = {(ptrdiff_t)n_rows, (ptrdiff_t)n_basis};
    const ptrdiff_t ostr[2] = {(ptrdiff_t)sizeof(cd), (ptrdiff_t)(sizeof(cd) * n_rows)};

    // MW_FRESH=1: allocate M0, M1 and t1 afresh every repetition (uninitialised,
    // as numpy.empty is), so the kernels first-touch new pages the way the
    // fresh outputs of a Python call do. Unset: the buffers are reused.
    const bool fresh = std::getenv("MW_FRESH") != nullptr;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<cd[]> fM0, fM1, ft1;
        cd *pM0 = M0.data(), *pM1 = M1.data(), *pt1 = t1.data();
        if (fresh) {
            fM0.reset(new cd[M0.size()]);
            fM1.reset(new cd[M1.size() ? M1.size() : 1]);
            ft1.reset(new cd[t1.size()]);
            pM0 = fM0.get();
            pM1 = fM1.get();
            pt1 = ft1.get();
        }
        razor::seg_moments(obs, V<double, 2>(A["sm_seg_p0"]), V<double, 2>(A["sm_seg_t"]),
                           V<double, 1>(A["sm_seg_h"]), V<double, 1>(A["sm_a"]),
                           V<double, 1>(A["sm_xg"]), V<double, 1>(A["sm_wg"]),
                           scalar(A["sm_k"]), need_m1, V<int64_t, 1>(A["sm_group_i"]),
                           V<int64_t, 1>(A["sm_group_j"]), V<double, 1>(A["sm_a_ek"]), 0,
                           V<double, 1>(A["sm_b_obs"]),
                           view<cd, 2>::contiguous(pM0, {(ptrdiff_t)n_obs, (ptrdiff_t)n_seg}),
                           view<cd, 2>::contiguous(pM1, {(ptrdiff_t)(need_m1 ? n_obs : 0),
                                                               (ptrdiff_t)(need_m1 ? n_seg : 0)}));
        const ptrdiff_t mshape[2] = {(ptrdiff_t)n_obs, (ptrdiff_t)n_seg};
        const ptrdiff_t mstr[2] = {(ptrdiff_t)(sizeof(cd) * n_seg), (ptrdiff_t)sizeof(cd)};
        const view<const cd, 2> vM0(pM0, mshape, mstr), vM1(pM1, mshape, mstr);
        razor::assemble_t1(vM0, vM1, V<int64_t, 1>(A["t1_s_a"]), V<int64_t, 1>(A["t1_s_b"]),
                           V<double, 1>(A["t1_h_a"]), V<double, 1>(A["t1_h_b"]),
                           V<bool, 1>(A["t1_fall_a"]), V<bool, 1>(A["t1_fall_b"]),
                           V<double, 2>(A["t1_t_out"]), V<double, 2>(A["t1_td_a"]),
                           V<double, 2>(A["t1_td_b"]), V<double, 1>(A["t1_wts"]), n_path,
                           V<int64_t, 1>(A["t1_obs_row"]),
                           view<cd, 2>::contiguous(pt1, {(ptrdiff_t)n_rows, (ptrdiff_t)n_basis}));
        const view<const cd, 2> vt1 = view<const cd, 2>::contiguous(
            pt1, {(ptrdiff_t)n_rows, (ptrdiff_t)n_basis});
        razor::t2_rows(view<cd, 2>(rows.data(), oshape, ostr), &vt1,
                       V<cd, 2>(A["t2_M0w"]), V<cd, 2>(A["t2_M0k"]),
                       V<int64_t, 1>(A["t2_plus"]), V<int64_t, 1>(A["t2_minus"]),
                       V<int64_t, 1>(A["t1_s_a"]), V<int64_t, 1>(A["t1_s_b"]),
                       V<double, 1>(A["t2_q_a"]), V<double, 1>(A["t2_q_b"]),
                       scalar(A["t2_c_a_imag"]), scalar(A["t2_c_phi_imag"]));
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt < best) best = dt;
        if (fresh && r == reps - 1) {
            std::memcpy(M0.data(), pM0, M0.size() * sizeof(cd));
            std::memcpy(M1.data(), pM1, M1.size() * sizeof(cd));
            std::memcpy(t1.data(), pt1, t1.size() * sizeof(cd));
        }
    }
    dump(dir + "/drv_M0.bin", M0.data(), M0.size() * sizeof(cd));
    dump(dir + "/drv_M1.bin", M1.data(), M1.size() * sizeof(cd));
    dump(dir + "/drv_t1.bin", t1.data(), t1.size() * sizeof(cd));
    dump(dir + "/drv_rows_F.bin", rows.data(), rows.size() * sizeof(cd));
    std::printf("n_obs=%zu n_seg=%zu n_basis=%zu n_rows=%zu n_path=%zu best_us=%.2f\n",
                n_obs, n_seg, n_basis, n_rows, n_path, best * 1e6);
    return 0;
}
