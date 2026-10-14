// A standalone C++ program (no Python, no pybind11, no numpy) that runs one
// razor row window through three core kernels in sequence:
//
//     seg_moments -> (M0, M1) -> assemble_t1 -> T1 -> t2_rows -> rows of Z
//
// M0, M1 and T1 are std::vector buffers that never leave C++. The window's
// inputs are read from <dir> (written by capture_window.py's
// write_driver_inputs); the four results are written back as
// <dir>/drv_{M0,M1,t1,rows}.bin, the Z rows column-major as RazorSolver's Z
// is, for a byte-for-byte comparison with the Python chain over the same
// inputs (tests/test_razor_core_standalone.py).
//
// Build it from the core TUs alone, with the extension's flags and no Python
// include path; the test does exactly that.
#include "_razor_core.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Arr {
    std::string dtype;
    std::vector<ptrdiff_t> shape;
    std::vector<char> bytes;
};

std::map<std::string, Arr> load(const std::string &dir) {
    std::map<std::string, Arr> m;
    std::ifstream man(dir + "/manifest.txt");
    std::string line;
    while (std::getline(man, line)) {
        std::istringstream is(line);
        std::string name;
        Arr a;
        int nd = 0;
        is >> name >> a.dtype >> nd;
        a.shape.resize(nd);
        for (int k = 0; k < nd; k++) is >> a.shape[k];
        std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
        a.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        m[name] = a;
    }
    return m;
}

const Arr &get(const std::map<std::string, Arr> &A, const std::string &name) {
    auto it = A.find(name);
    if (it == A.end()) throw std::runtime_error("window has no " + name);
    return it->second;
}

// A C-contiguous view of a loaded array.
template <class T, int N>
mw::view<const T, N> V(const std::map<std::string, Arr> &A, const std::string &name) {
    const Arr &a = get(A, name);
    if (static_cast<int>(a.shape.size()) != N) throw std::runtime_error(name + ": rank");
    ptrdiff_t shp[N], st[N];
    ptrdiff_t acc = sizeof(T);
    for (int k = N - 1; k >= 0; k--) {
        shp[k] = a.shape[k];
        st[k] = acc;
        acc *= a.shape[k];
    }
    if (static_cast<size_t>(acc) != a.bytes.size()) throw std::runtime_error(name + ": size");
    return mw::view<const T, N>(reinterpret_cast<const T *>(a.bytes.data()), shp, st);
}

double scalar(const std::map<std::string, Arr> &A, const std::string &name) {
    const Arr &a = get(A, name);
    if (a.dtype != "<f8" || a.bytes.size() != 8) throw std::runtime_error(name + ": not a scalar");
    double d;
    std::memcpy(&d, a.bytes.data(), 8);
    return d;
}

void dump(const std::string &path, const void *p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char *>(p), static_cast<std::streamsize>(n));
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: razor_chain_driver <dir>\n";
        return 2;
    }
    using mw::cd;
    using mw::view;
    const std::string dir = argv[1];
    try {
        const auto A = load(dir);
        const auto obs = V<double, 2>(A, "sm_obs");
        const ptrdiff_t n_obs = obs.shape(0), n_seg = V<double, 1>(A, "sm_seg_h").shape(0);
        const bool need_m1 = scalar(A, "sm_need_m1") != 0.0;
        std::vector<cd> M0(n_obs * n_seg), M1(need_m1 ? n_obs * n_seg : 0);
        mw::razor::seg_moments(
            obs, V<double, 2>(A, "sm_seg_p0"), V<double, 2>(A, "sm_seg_t"),
            V<double, 1>(A, "sm_seg_h"), V<double, 1>(A, "sm_a"), V<double, 1>(A, "sm_xg"),
            V<double, 1>(A, "sm_wg"), scalar(A, "sm_k"), need_m1,
            V<int64_t, 1>(A, "sm_group_i"), V<int64_t, 1>(A, "sm_group_j"),
            V<double, 1>(A, "sm_a_ek"), 0, V<double, 1>(A, "sm_b_obs"),
            view<cd, 2>::contiguous(M0.data(), {n_obs, n_seg}),
            view<cd, 2>::contiguous(M1.data(), {need_m1 ? n_obs : 0, need_m1 ? n_seg : 0}));

        const auto n_path = static_cast<size_t>(scalar(A, "t1_n_path"));
        const ptrdiff_t n_basis = V<int64_t, 1>(A, "t1_s_a").shape(0);
        const ptrdiff_t n_rows = V<double, 1>(A, "t1_wts").shape(0) / n_path;
        std::vector<cd> T1(n_rows * n_basis);
        mw::razor::assemble_t1(
            view<const cd, 2>::contiguous(M0.data(), {n_obs, n_seg}),
            view<const cd, 2>::contiguous(M1.data(), {n_obs, n_seg}),
            V<int64_t, 1>(A, "t1_s_a"), V<int64_t, 1>(A, "t1_s_b"), V<double, 1>(A, "t1_h_a"),
            V<double, 1>(A, "t1_h_b"), V<bool, 1>(A, "t1_fall_a"), V<bool, 1>(A, "t1_fall_b"),
            V<double, 2>(A, "t1_t_out"), V<double, 2>(A, "t1_td_a"),
            V<double, 2>(A, "t1_td_b"), V<double, 1>(A, "t1_wts"), n_path,
            V<int64_t, 1>(A, "t1_obs_row"),
            view<cd, 2>::contiguous(T1.data(), {n_rows, n_basis}));

        // The Z rows column-major, written through their strides.
        std::vector<cd> rows(n_rows * n_basis);
        const ptrdiff_t rshape[2] = {n_rows, n_basis};
        const ptrdiff_t rstrides[2] = {static_cast<ptrdiff_t>(sizeof(cd)),
                                       static_cast<ptrdiff_t>(sizeof(cd)) * n_rows};
        const view<const cd, 2> t1 = view<const cd, 2>::contiguous(T1.data(), {n_rows, n_basis});
        mw::razor::t2_rows(view<cd, 2>(rows.data(), rshape, rstrides), &t1,
                           V<cd, 2>(A, "t2_M0w"), V<cd, 2>(A, "t2_M0k"),
                           V<int64_t, 1>(A, "t2_plus"), V<int64_t, 1>(A, "t2_minus"),
                           V<int64_t, 1>(A, "t2_s_a"), V<int64_t, 1>(A, "t2_s_b"),
                           V<double, 1>(A, "t2_q_a"), V<double, 1>(A, "t2_q_b"),
                           scalar(A, "t2_c_a_imag"), scalar(A, "t2_c_phi_imag"));

        dump(dir + "/drv_M0.bin", M0.data(), M0.size() * sizeof(cd));
        dump(dir + "/drv_M1.bin", M1.data(), M1.size() * sizeof(cd));
        dump(dir + "/drv_t1.bin", T1.data(), T1.size() * sizeof(cd));
        dump(dir + "/drv_rows.bin", rows.data(), rows.size() * sizeof(cd));
    } catch (const std::exception &e) {
        std::cerr << "razor_chain_driver: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
