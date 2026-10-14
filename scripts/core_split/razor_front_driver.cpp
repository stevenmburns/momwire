// geometry -> currents with no Python: razor-2p's front half
// (`_razor_front_core.h`), fill and LU (`_razor_solve_core.h`) driven from a
// plain-text deck.
//
//     razor_front_driver DECK.txt LIBOPENBLAS.so OUTDIR [REPEATS]
//
// With REPEATS it also times the whole chain that many times over, warm (one
// SolveContext, its pool reused, as a served process would hold it), and
// prints the medians in ms: "build fill lu total" -- build is the front half
// (construction, geometry, tables, port columns).
//
// LIBOPENBLAS.so supplies ddot / zgetrf / zgetrs (tried as `scipy_<name>_`,
// then `<name>_`): pointing it at SciPy's bundled OpenBLAS makes the driver
// call the very routines the Python route reaches, which is what lets
// tests/test_razor_core_standalone.py require the same BYTES. Nothing of
// Python is loaded; the library is OpenBLAS.
//
// The deck is one directive per line ('#' starts a comment):
//
//     wavelength 22.0
//     radius 0.002 [0.004 ...]   one value, or one per wire
//     ground 0.0                  a perfect ground at that z (optional)
//     ek 1                        the extended kernel (default 0)
//     exact auto|off|on           the exact ring kernel option (default auto)
//     wire c1 [c2 ...] : x y z  x y z [x y z ...]   segment counts per edge,
//                                                   then the anchors
//     feed W ARC|mid RE IM        a gap feed on wire W
//     gap W start|end RE IM       a series EMF at a junction member
//
// It writes OUTDIR/front_Z.bin (Z before the LU, column-major complex128),
// OUTDIR/front_X.bin (the coefficients, n_basis x n_ports, column-major) and
// OUTDIR/front_meta.txt ("n_basis n_ports info", or "decline REASON", or
// "refused KIND MESSAGE").
#include "_razor_front_core.h"
#include "_razor_solve_core.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using mw::cd;

namespace {

void *sym(void *lib, const char *name) {
    std::string prefixed = std::string("scipy_") + name + "_";
    void *p = dlsym(lib, prefixed.c_str());
    if (p == nullptr) p = dlsym(lib, (std::string(name) + "_").c_str());
    if (p == nullptr) {
        std::cerr << "no " << name << " in the library\n";
        std::exit(2);
    }
    return p;
}

mw::razor::FrontInputs read_deck(const char *path) {
    mw::razor::FrontInputs in;
    in.counts_given = true;
    in.radius_shape.clear();
    std::ifstream f(path);
    if (!f) {
        std::cerr << "cannot read " << path << "\n";
        std::exit(2);
    }
    std::string line;
    std::vector<double> radii;
    while (std::getline(f, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream s(line);
        std::string key;
        if (!(s >> key)) continue;
        if (key == "wavelength") {
            std::string v;
            s >> v;
            in.wavelength = std::strtod(v.c_str(), nullptr);
        } else if (key == "radius") {
            std::string v;
            while (s >> v) radii.push_back(std::strtod(v.c_str(), nullptr));
        } else if (key == "ground") {
            std::string v;
            s >> v;
            in.has_ground = true;
            in.ground_z = std::strtod(v.c_str(), nullptr);
        } else if (key == "ek") {
            int v = 0;
            s >> v;
            in.extended_kernel = v != 0;
        } else if (key == "exact") {
            std::string v;
            s >> v;
            in.exact_kernel = v == "auto"  ? mw::razor::ExactKernel::automatic
                              : v == "on" ? mw::razor::ExactKernel::on
                                          : mw::razor::ExactKernel::off;
        } else if (key == "wire") {
            mw::razor::EdgeCounts ec;
            ec.kind = mw::razor::EdgeCounts::list;
            std::string tok;
            while (s >> tok && tok != ":") ec.v.push_back(std::atoll(tok.c_str()));
            int64_t n = 0;
            while (s >> tok) {
                in.points.push_back(std::strtod(tok.c_str(), nullptr));
                n++;
            }
            in.wire_points.push_back(n / 3);
            in.counts.push_back(ec);
        } else if (key == "feed") {
            mw::razor::FeedSpec fs;
            std::string arc, re, im;
            s >> fs.wire >> arc >> re >> im;
            fs.midpoint = arc == "mid";
            if (!fs.midpoint) fs.arc = std::strtod(arc.c_str(), nullptr);
            fs.volts = cd(std::strtod(re.c_str(), nullptr), std::strtod(im.c_str(), nullptr));
            in.feeds.push_back(fs);
        } else if (key == "gap") {
            mw::razor::NodeGapSpec g;
            std::string end, re, im;
            s >> g.wire >> end >> re >> im;
            g.end = end == "start" ? mw::razor::End::start : mw::razor::End::end;
            g.volts = cd(std::strtod(re.c_str(), nullptr), std::strtod(im.c_str(), nullptr));
            in.node_gaps.push_back(g);
        }
    }
    in.radius = radii;
    if (radii.size() > 1) in.radius_shape.push_back(static_cast<int64_t>(radii.size()));
    return in;
}

void write(const std::string &path, const void *p, size_t bytes) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    std::fwrite(p, 1, bytes, f);
    std::fclose(f);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "usage: razor_front_driver DECK.txt LIBOPENBLAS.so OUTDIR [REPEATS]\n";
        return 2;
    }
    void *lib = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        std::cerr << dlerror() << "\n";
        return 2;
    }
    mw::razor::Blas blas;
    blas.ddot = reinterpret_cast<mw::razor::ddot_fn>(sym(lib, "ddot"));
    mw::razor::Lapack lp;
    lp.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(sym(lib, "zgetrf"));
    lp.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(sym(lib, "zgetrs"));
    const std::string out = argv[3];

    const mw::razor::FrontInputs in = read_deck(argv[1]);
    mw::razor::Front front;
    std::ofstream meta(out + "/front_meta.txt");
    try {
        const std::string why = front.build(in, blas);
        if (!why.empty()) {
            meta << "decline " << why << "\n";
            return 0;
        }
    } catch (const mw::razor::FrontError &e) {
        meta << "refused "
             << (e.kind == mw::razor::FrontErrorKind::value_error ? "ValueError"
                                                                  : "NotImplementedError")
             << " " << e.what() << "\n";
        return 0;
    }
    const int64_t n = front.n_basis, m = front.n_ports;
    std::vector<cd> X(static_cast<size_t>(n * m));
    for (int64_t j = 0; j < m; j++)
        for (int64_t i = 0; i < n; i++) X[j * n + i] = cd(front.cols[i * m + j], 0.0);
    mw::razor::SolveContext ctx;
    ctx.fill(front.solve_inputs(0));
    write(out + "/front_Z.bin", ctx.z(), sizeof(cd) * static_cast<size_t>(n * n));
    const int info = ctx.lu_solve(lp, X.data(), static_cast<int>(m));
    write(out + "/front_X.bin", X.data(), sizeof(cd) * X.size());
    meta << n << " " << m << " " << info << "\n";
    if (argc == 5) {
        const int reps = std::atoi(argv[4]);
        typedef std::chrono::steady_clock clk;
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::vector<double> tb, tf, tl, tt;
        for (int r = 0; r < reps; r++) {
            const auto t0 = clk::now();
            mw::razor::Front f;
            f.build(in, blas);
            std::vector<cd> x(static_cast<size_t>(f.n_basis * f.n_ports));
            for (int64_t j = 0; j < f.n_ports; j++)
                for (int64_t i = 0; i < f.n_basis; i++)
                    x[j * f.n_basis + i] = cd(f.cols[i * f.n_ports + j], 0.0);
            const auto t1 = clk::now();
            ctx.fill(f.solve_inputs(0));
            const auto t2 = clk::now();
            ctx.lu_solve(lp, x.data(), static_cast<int>(f.n_ports));
            const auto t3 = clk::now();
            tb.push_back(ms(t0, t1));
            tf.push_back(ms(t1, t2));
            tl.push_back(ms(t2, t3));
            tt.push_back(ms(t0, t3));
        }
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            const size_t k = v.size();
            return k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
        };
        std::printf("%.6f %.6f %.6f %.6f\n", med(tb), med(tf), med(tl), med(tt));
    }
    return 0;
}
