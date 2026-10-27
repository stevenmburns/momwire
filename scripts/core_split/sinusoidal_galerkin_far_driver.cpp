// The Sinusoidal-Galerkin far fills with no Python (`_sinusoidal_galerkin_core.h`,
// momwire#1718 S1), on the default host: replays calls the extension served,
// read from DIR/calls.txt, and writes the core's answers beside them for
// `tests/test_razor_core_standalone.py` to hold to the extension's bytes.
//
//   DIR/calls.txt   one call per line: "<function> <out prefix> <arg>..."
//                   with each arg one of
//                     A:<file>:<d0>[x<d1>...]   a float64 array, C order
//                     L:<file>:<d0>[x<d1>...]   an int64 array
//                     W:<file>:<d0>[x<d1>...]   a complex128 array
//                     D:<hex bits>              a double
//                     Z:<hex bits>:<hex bits>   a complex
//                     U:<n>                     a cancel flag (0)
//                     B:<0|1>                   a bool
//                     N                         no fold destination (`out=None`)
//                     F:<f0>:<f1>:<f2>:<nnz>x<N>  three complex128 fold
//                                               destinations' starting values,
//                                               followed by the scale (a Z)
//   ->  DIR/<prefix>_0.bin _1.bin _2.bin   the three complex128 (nnz, N) arrays
//                                          (folding: the destinations after it)
#include "_sinusoidal_galerkin_core.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace nd = mw::nd;
namespace sgc = mw::sinusoidal_galerkin;
using cd = std::complex<double>;
using CF = nd::array_t<double, nd::array::c_style | nd::array::forcecast>;
using CL = nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast>;
using CW = nd::array_t<cd, nd::array::c_style | nd::array::forcecast>;
using Out = std::tuple<nd::array_t<cd>, nd::array_t<cd>, nd::array_t<cd>>;

namespace {

std::vector<char> read_bytes(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

double from_bits(const std::string &hex) {
    std::uint64_t u = std::stoull(hex, nullptr, 16);
    double d;
    std::memcpy(&d, &u, sizeof d);
    return d;
}

std::vector<nd::ssize_t> dims_of(const std::string &s) {
    std::vector<nd::ssize_t> shape;
    std::stringstream dims(s);
    std::string d;
    while (std::getline(dims, d, 'x')) shape.push_back(std::stoll(d));
    return shape;
}

// One call's arguments, kept alive while the call runs.
struct Args {
    std::vector<std::vector<char>> storage;
    std::vector<std::string> specs;
    std::size_t next = 0;
    std::string dir;
    sgc::GalerkinFoldBlock fold;

    const std::string &take(char kind) {
        const std::string &s = specs.at(next++);
        if (s.empty() || s[0] != kind) throw std::runtime_error("argument " + s + " is not of kind " + kind);
        return s;
    }
    // A C-ordered buffer over FILE's bytes, of the shape DIMS.
    nd::Buffer buffer_of(const std::string &file, const std::string &dims, std::size_t itemsize) {
        storage.push_back(read_bytes(dir + "/" + file));
        nd::Buffer b;
        b.ptr = storage.back().data();
        std::vector<nd::ssize_t> shape = dims_of(dims);
        b.ndim = static_cast<int>(shape.size());
        nd::ssize_t acc = static_cast<nd::ssize_t>(itemsize);
        for (int k = b.ndim - 1; k >= 0; k--) {
            b.shape[k] = shape[static_cast<std::size_t>(k)];
            b.strides[k] = acc;
            acc *= shape[static_cast<std::size_t>(k)];
        }
        return b;
    }
    nd::Buffer buffer(const std::string &s, std::size_t itemsize) {
        std::size_t a = s.find(':', 2);
        return buffer_of(s.substr(2, a - 2), s.substr(a + 1), itemsize);
    }
    CF a() { return CF::borrow(buffer(take('A'), sizeof(double))); }
    CL l() { return CL::borrow(buffer(take('L'), sizeof(int64_t))); }
    CW w() { return CW::borrow(buffer(take('W'), sizeof(cd))); }
    double d() { return from_bits(take('D').substr(2)); }
    cd z() {
        const std::string &s = take('Z');
        std::size_t c = s.find(':', 2);
        return cd(from_bits(s.substr(2, c - 2)), from_bits(s.substr(c + 1)));
    }
    std::uintptr_t u() { return static_cast<std::uintptr_t>(std::stoull(take('U').substr(2))); }
    bool b() { return take('B').substr(2) == "1"; }
    // `N`, or `F:...` and its scale: the fold block, or null.
    const sgc::GalerkinFoldBlock *f() {
        const std::string &s = specs.at(next);
        if (s == "N") {
            next++;
            return nullptr;
        }
        take('F');
        std::vector<std::string> part;
        std::stringstream ss(s.substr(2));
        std::string p;
        while (std::getline(ss, p, ':')) part.push_back(p);
        if (part.size() != 4) throw std::runtime_error("a fold is three files and a shape: " + s);
        fold.dst_const = nd::array_t<cd>::borrow(buffer_of(part[0], part[3], sizeof(cd)));
        fold.dst_sin = nd::array_t<cd>::borrow(buffer_of(part[1], part[3], sizeof(cd)));
        fold.dst_cos = nd::array_t<cd>::borrow(buffer_of(part[2], part[3], sizeof(cd)));
        cd scale = z();
        fold.s_re = scale.real();
        fold.s_im = scale.imag();
        return &fold;
    }
};

Out dispatch(const std::string &fn, Args &x) {
    if (fn == "sinusoidal_galerkin_far_fill" || fn == "sinusoidal_galerkin_far_fill_ek") {
        CF oc = x.a(), ot = x.a(), orad = x.a(), sc = x.a(), st = x.a(), hh = x.a();
        double k = x.d(), eta = x.d();
        CF t = x.a(), w = x.a();
        CW we = x.w();
        CL starts = x.l();
        if (fn == "sinusoidal_galerkin_far_fill") {
            std::uintptr_t cancel = x.u();
            const sgc::GalerkinFoldBlock *fold = x.f();
            bool reference = x.b();
            return sgc::sinusoidal_galerkin_far_fill(oc, ot, orad, sc, st, hh, k, eta, t, w, we, starts, cancel,
                                                     fold, reference);
        }
        CF src_a = x.a();
        CL g_obs = x.l(), g_src = x.l();
        CF ek_gx = x.a(), ek_gw = x.a();
        std::uintptr_t cancel = x.u();
        const sgc::GalerkinFoldBlock *fold = x.f();
        return sgc::sinusoidal_galerkin_far_fill_ek(oc, ot, orad, sc, st, hh, k, eta, t, w, we, starts, src_a, g_obs,
                                                    g_src, ek_gx, ek_gw, cancel, fold);
    }
    if (fn == "sinusoidal_galerkin_far_fill_cplx") {
        CF oc = x.a(), ot = x.a(), orad = x.a(), sc = x.a(), st = x.a(), hh = x.a();
        cd k = x.z(), eta = x.z();
        CF t = x.a(), w = x.a();
        CW we = x.w();
        CL starts = x.l();
        std::uintptr_t cancel = x.u();
        const sgc::GalerkinFoldBlock *fold = x.f();
        bool reference = x.b();
        return sgc::sinusoidal_galerkin_far_fill_cplx(oc, ot, orad, sc, st, hh, k, eta, t, w, we, starts, cancel,
                                                      fold, reference);
    }
    throw std::runtime_error("unknown function " + fn);
}

void write(const std::string &path, const nd::array_t<cd> &a) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(a.data()), static_cast<std::streamsize>(a.size() * 16));
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: sinusoidal_galerkin_far_driver DIR\n";
        return 2;
    }
    const std::string dir = argv[1];
    std::ifstream calls(dir + "/calls.txt");
    std::string line;
    try {
        while (std::getline(calls, line)) {
            std::stringstream ss(line);
            std::string fn, prefix, spec;
            ss >> fn >> prefix;
            Args x;
            x.dir = dir;
            while (ss >> spec) x.specs.push_back(spec);
            Out r = dispatch(fn, x);
            if (x.next != x.specs.size()) throw std::runtime_error(prefix + ": arguments left over");
            write(dir + "/" + prefix + "_0.bin", std::get<0>(r));
            write(dir + "/" + prefix + "_1.bin", std::get<1>(r));
            write(dir + "/" + prefix + "_2.bin", std::get<2>(r));
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
