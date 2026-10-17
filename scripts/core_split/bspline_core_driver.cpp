// The B-spline kernels with no Python (`_bspline_core.h`, momwire#1517 step
// 0), on the default host: reads a segment set from DIR, writes the core's
// answers there for `tests/test_razor_core_standalone.py` to hold to the
// extension's bytes.
//
//   DIR/seg_l.bin seg_r.bin   float64 (n, 3) each, C order
//   DIR/gl_t.bin gl_w.bin     the Gauss-Legendre rule on [0, 1]
//   DIR/params.txt            a k max_d h N
//   ->  DIR/drv_full.bin      seg_seg_full_moments_bspline (lanes where built)
//       DIR/drv_full_ref.bin  the same with reference=true (the per-pair walk)
//       DIR/drv_static.bin    seg_seg_static_moments_bspline_uniform(h, a, N, max_d)
#include "_bspline_core.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace nd = mw::nd;

static std::vector<double> read_doubles(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<double> v(b.size() / sizeof(double));
    std::copy(b.data(), b.data() + v.size() * sizeof(double), reinterpret_cast<char *>(v.data()));
    return v;
}

template <class T, int F>
static void write_array(const std::string &path, const nd::array_t<T, F> &a) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(a.data()),
            static_cast<std::streamsize>(a.size() * static_cast<nd::ssize_t>(sizeof(T))));
}

using CF = nd::array_t<double, nd::array::c_style | nd::array::forcecast>;

static CF borrow(std::vector<double> &v, std::vector<nd::ssize_t> shape) {
    nd::Buffer b;
    b.ptr = v.data();
    b.ndim = static_cast<int>(shape.size());
    nd::ssize_t acc = sizeof(double);
    for (int k = b.ndim - 1; k >= 0; k--) {
        b.shape[k] = shape[static_cast<size_t>(k)];
        b.strides[k] = acc;
        acc *= shape[static_cast<size_t>(k)];
    }
    return CF::borrow(b);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: bspline_core_driver DIR\n";
        return 2;
    }
    const std::string dir = argv[1];
    std::vector<double> sl = read_doubles(dir + "/seg_l.bin"), sr = read_doubles(dir + "/seg_r.bin"),
                        gt = read_doubles(dir + "/gl_t.bin"), gw = read_doubles(dir + "/gl_w.bin");
    double a, k, h;
    int max_d;
    size_t N;
    {
        std::ifstream f(dir + "/params.txt");
        f >> a >> k >> max_d >> h >> N;
        if (!f) {
            std::cerr << "bad params.txt\n";
            return 2;
        }
    }
    try {
        const nd::ssize_t n = static_cast<nd::ssize_t>(sl.size() / 3);
        CF l = borrow(sl, {n, 3}), r = borrow(sr, {n, 3});
        CF t = borrow(gt, {static_cast<nd::ssize_t>(gt.size())}),
           w = borrow(gw, {static_cast<nd::ssize_t>(gw.size())});
        write_array(dir + "/drv_full.bin",
                    mw::bspline::seg_seg_full_moments_bspline(l, r, l, r, a * a, k, max_d, t, w, false));
        write_array(dir + "/drv_full_ref.bin",
                    mw::bspline::seg_seg_full_moments_bspline(l, r, l, r, a * a, k, max_d, t, w, true));
        write_array(dir + "/drv_static.bin",
                    mw::bspline::seg_seg_static_moments_bspline_uniform(h, a, N, max_d, 0, 0));
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
