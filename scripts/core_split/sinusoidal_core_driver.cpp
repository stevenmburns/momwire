// The sinusoidal point-matched kernels with no Python (`_sinusoidal_core.h`,
// momwire#1607 B1), on the default host: replays calls the extension served,
// read from DIR/calls.txt, and writes the core's answers beside them for
// `tests/test_razor_core_standalone.py` to hold to the extension's bytes.
//
//   DIR/calls.txt   one call per line: "<function> <out prefix> <arg>..."
//                   with each arg one of
//                     A:<file>:<d0>[x<d1>...]   a float64 array, C order
//                     I:<file>:<d0>[x<d1>...]   an int8 array
//                     D:<hex bits>              a double
//                     Z:<hex bits>:<hex bits>   a complex
//                     U:<n>                     a cancel flag (0)
//                     B:<0|1>                   a bool
//   ->  DIR/<prefix>_0.bin _1.bin _2.bin   the three complex128 tensors
#include "_sinusoidal_core.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace nd = mw::nd;
using cd = std::complex<double>;
using CF = nd::array_t<double, nd::array::c_style | nd::array::forcecast>;
using CI8 = nd::array_t<int8_t, nd::array::c_style | nd::array::forcecast>;
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

// One call's arguments, kept alive while the call runs.
struct Args {
    std::vector<std::vector<char>> storage;
    std::vector<std::string> specs;
    std::size_t next = 0;
    std::string dir;

    const std::string &take(char kind) {
        const std::string &s = specs.at(next++);
        if (s.empty() || s[0] != kind) throw std::runtime_error("argument " + s + " is not of kind " + kind);
        return s;
    }
    nd::Buffer buffer(const std::string &s, std::size_t itemsize) {
        std::size_t a = s.find(':', 2);
        storage.push_back(read_bytes(dir + "/" + s.substr(2, a - 2)));
        nd::Buffer b;
        b.ptr = storage.back().data();
        std::vector<nd::ssize_t> shape;
        std::stringstream dims(s.substr(a + 1));
        std::string d;
        while (std::getline(dims, d, 'x')) shape.push_back(std::stoll(d));
        b.ndim = static_cast<int>(shape.size());
        nd::ssize_t acc = static_cast<nd::ssize_t>(itemsize);
        for (int k = b.ndim - 1; k >= 0; k--) {
            b.shape[k] = shape[static_cast<std::size_t>(k)];
            b.strides[k] = acc;
            acc *= shape[static_cast<std::size_t>(k)];
        }
        return b;
    }
    CF a() { return CF::borrow(buffer(take('A'), sizeof(double))); }
    CI8 i8() { return CI8::borrow(buffer(take('I'), sizeof(int8_t))); }
    double d() { return from_bits(take('D').substr(2)); }
    cd z() {
        const std::string &s = take('Z');
        std::size_t c = s.find(':', 2);
        return cd(from_bits(s.substr(2, c - 2)), from_bits(s.substr(c + 1)));
    }
    std::uintptr_t u() { return static_cast<std::uintptr_t>(std::stoull(take('U').substr(2))); }
    bool b() { return take('B').substr(2) == "1"; }
};

Out dispatch(const std::string &fn, Args &x) {
    namespace s = mw::sinusoidal;
    if (fn == "sinusoidal_field_tensor") {
        CF oc = x.a(), ot = x.a(), sc = x.a(), st = x.a(), h = x.a();
        double a = x.d(), k = x.d(), eta = x.d();
        CF t = x.a(), w = x.a();
        return s::sinusoidal_field_tensor(oc, ot, sc, st, h, a, k, eta, t, w, x.u());
    }
    if (fn == "sinusoidal_field_tensor_refl") {
        CF oc = x.a(), ot = x.a(), sc = x.a(), st = x.a(), h = x.a();
        double a = x.d(), k = x.d(), eta = x.d();
        CF t = x.a(), w = x.a(), cos_th = x.a(), px = x.a(), py = x.a(), tm = x.a(), tn = x.a();
        cd eps = x.z();
        return s::sinusoidal_field_tensor_refl(oc, ot, sc, st, h, a, k, eta, t, w, cos_th, px, py, tm, tn, eps, x.u());
    }
    if (fn == "sinusoidal_field_tensor_ek") {
        CF oc = x.a(), ot = x.a(), sc = x.a(), st = x.a(), h = x.a();
        double a = x.d(), k = x.d(), eta = x.d();
        CF t = x.a(), w = x.a(), src_a = x.a();
        CI8 i1 = x.i8(), i2 = x.i8();
        return s::sinusoidal_field_tensor_ek(oc, ot, sc, st, h, a, k, eta, t, w, src_a, i1, i2, x.u());
    }
    if (fn == "sinusoidal_field_tensor_ek_refl") {
        CF oc = x.a(), ot = x.a(), sc = x.a(), st = x.a(), h = x.a();
        double a = x.d(), k = x.d(), eta = x.d();
        CF t = x.a(), w = x.a(), src_a = x.a();
        CI8 i1 = x.i8(), i2 = x.i8();
        CF cos_th = x.a(), px = x.a(), py = x.a(), tm = x.a(), tn = x.a();
        cd eps = x.z();
        return s::sinusoidal_field_tensor_ek_refl(oc, ot, sc, st, h, a, k, eta, t, w, src_a, i1, i2, cos_th, px, py,
                                                  tm, tn, eps, x.u());
    }
    if (fn == "sinusoidal_field_tensor_cplx") {
        CF oc = x.a(), ot = x.a(), orad = x.a(), sc = x.a(), st = x.a(), h = x.a();
        cd k = x.z(), eta = x.z();
        CF t = x.a(), w = x.a();
        bool folded = x.b();
        return s::sinusoidal_field_tensor_cplx(oc, ot, orad, sc, st, h, k, eta, t, w, folded, x.u());
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
        std::cerr << "usage: sinusoidal_core_driver DIR\n";
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
