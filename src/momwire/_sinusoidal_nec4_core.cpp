// The NEC-4 dialect's radius-step junction, Python-free (see
// `_sinusoidal_nec4_core.h`). Built -ffp-contract=off like every core TU:
// each a*b+c below rounds twice, as written.

#define _USE_MATH_DEFINES

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <complex>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "_leggauss_table.h"
#include "_mw_core.h"
#include "_mw_nd.h"
#include "_sinusoidal_nec4_core.h"
#include "_stable_inline.h"

namespace py = mw::nd;
typedef std::complex<double> cd;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> F64;

namespace {

// Scalar libm, never a vector clone. This TU includes `_mw_core.h`, whose
// `omp declare simd` set lets GCC vectorize a loop of std::sin / std::cos /
// std::log onto libmvec, whose results differ from scalar libm in the last
// bits -- and whether a loop vectorizes is the optimizer's call, made per
// build (setup.py's -ffp-contract=off note names the hazard). A call through
// a function the optimizer may not inline cannot be vectorized, so every
// transcendental here is libm's scalar one: the junction solve is then
// Python's `math` to the bit, and the kernels' bits are one build's whatever
// the vectorizer decides.
#if defined(_MSC_VER)
#define MW_NEC4_SCALAR __declspec(noinline)
#else
#define MW_NEC4_SCALAR __attribute__((noinline))
#endif
MW_NEC4_SCALAR double s_sin(double x) { return std::sin(x); }
MW_NEC4_SCALAR double s_cos(double x) { return std::cos(x); }
MW_NEC4_SCALAR double s_log(double x) { return std::log(x); }
MW_NEC4_SCALAR double s_asinh(double x) { return std::asinh(x); }

// `_sinusoidal_nec4._RHO_FLOOR`
const double kRhoFloor = 1e-9;
// `_sinusoidal_nec4._CAP_AXIS_FRACTION`
const double kCapAxisFraction = 1e-6;

}  // namespace

namespace mw {
namespace sinusoidal {

// ---------------------------------------------------------------------------
// The NEC-4 kernel (`_sinusoidal_nec4.nec4_field_components`). Per pair, at
// the source ends z' = +-H with dz = z - z', G = e^{-jkR}/R and
// R = sqrt(rho^2 + a_src^2 + dz^2):
//   E_z  : const -j(eta k/4pi) int G;  sin j(eta/4pi) cos kH (G2 - G1);
//          cos -j(eta/4pi) sin kH (G2 + G1)
//   E_rho: -j(eta/4pi) (rho . t_obs)/rho^2 times the bracket of
//          [G dz cos kz' - j e sin kz'] (sin) or [-G dz sin kz' - j e cos kz']
//          (cos), the const shape carrying none.
std::tuple<py::array_t<cd>, py::array_t<cd>, py::array_t<cd>>
sinusoidal_field_tensor_nec4(F64 obs_centers, F64 obs_tangents, F64 src_centers, F64 src_tangents, F64 seg_h,
                             F64 src_a, double k, double eta, F64 gl_t, F64 gl_w, uintptr_t cancel_flag) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    auto sa = src_a.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    if (oc.shape(1) != 3 || ot.shape(1) != 3 || sc.shape(1) != 3 || st.shape(1) != 3)
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    if (oc.shape(0) != ot.shape(0)) throw std::runtime_error("obs_centers and obs_tangents must have matching N");
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0) || sc.shape(0) != sa.shape(0))
        throw std::runtime_error("src arrays must all have matching N");
    if (glt.shape(0) != glw.shape(0)) throw std::runtime_error("gl_t and gl_w must have matching length");
    const size_t M = oc.shape(0), N = sc.shape(0), nq = glt.shape(0);

    py::array_t<cd> Phi_const({M, N});
    py::array_t<cd> Phi_sin({M, N});
    py::array_t<cd> Phi_cos({M, N});
    auto pc = Phi_const.mutable_unchecked<2>();
    auto ps = Phi_sin.mutable_unchecked<2>();
    auto po = Phi_cos.mutable_unchecked<2>();

    py::gil_scoped_release release;

    std::vector<double> Hn(N), s2n(N), c2n(N), a2n(N), gx(nq), gw(nq);
    for (size_t n = 0; n < N; n++) {
        Hn[n] = 0.5 * sh(n);
        s2n[n] = s_sin(k * Hn[n]);
        c2n[n] = s_cos(k * Hn[n]);
        a2n[n] = sa(n) * sa(n);
    }
    for (size_t q = 0; q < nq; q++) {
        gx[q] = glt(q);
        gw[q] = glw(q);
    }
    const double pz = eta / (4.0 * M_PI * k);  // pref_z = j pz
    const double pr = eta / (4.0 * M_PI);      // pre_rho = -j pr
    const double kk = k * k;

    MW_CANCEL_SETUP(cancel_flag);
#pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        MW_CANCEL_POLL();
        const double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        const double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);
        for (size_t n = 0; n < N; n++) {
            const double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            const double rvx = cmx - sc(n, 0), rvy = cmy - sc(n, 1), rvz = cmz - sc(n, 2);
            const double z = rvx * tnx + rvy * tny + rvz * tnz;
            const double qx = rvx - z * tnx, qy = rvy - z * tny, qz = rvz - z * tnz;
            const double rho2 = qx * qx + qy * qy + qz * qz;
            const double rho = std::sqrt(rho2);
            const double td = tmx * tnx + tmy * tny + tmz * tnz;
            const double rdot = qx * tmx + qy * tmy + qz * tmz;
            const double H = Hn[n];
            const double b2 = rho2 + a2n[n];
            const double dz2 = z - H, dz1 = z + H;
            const double r2 = std::sqrt(b2 + dz2 * dz2), r1 = std::sqrt(b2 + dz1 * dz1);
            const double e2r = s_cos(k * r2), e2i = -s_sin(k * r2);
            const double e1r = s_cos(k * r1), e1i = -s_sin(k * r1);
            const double g2r = e2r / r2, g2i = e2i / r2, g1r = e1r / r1, g1i = e1i / r1;

            // int G dz' = int 1/R (closed form) + int (e^{-jkR} - 1)/R (rule)
            double rre = 0.0, rim = 0.0;
            for (size_t q = 0; q < nq; q++) {
                const double dzq = z - H * gx[q];
                const double rq = std::sqrt(b2 + dzq * dzq);
                const double half = -0.5 * k * rq;
                const double hs = s_sin(half), hc = s_cos(half);
                rre += gw[q] * ((-2.0 * hs * hs) / rq);
                rim += gw[q] * ((2.0 * hs * hc) / rq);
            }
            const double ig_re = stable_asinh_diff(-dz1, -dz2, b2, r1, r2) + H * rre;
            const double ig_im = H * rim;

            const double s2 = s2n[n], c2 = c2n[n];
            // const: -j pz k^2 int G
            const cd ez_c(pz * kk * ig_im, -(pz * kk * ig_re));
            // sin: j pz k cos kH (G2 - G1)
            const double ds_re = g2r - g1r, ds_im = g2i - g1i;
            const double as = pz * k * c2;
            const cd ez_s(-(as * ds_im), as * ds_re);
            // cos: -j pz k sin kH (G2 + G1)
            const double so_re = g2r + g1r, so_im = g2i + g1i;
            const double ao = pz * k * s2;
            const cd ez_o(ao * so_im, -(ao * so_re));

            const double proj = rho > kRhoFloor * H ? rdot / rho2 : 0.0;
            const double bs_re = (g2r * dz2 * c2 + e2i * s2) - (g1r * dz1 * c2 - e1i * s2);
            const double bs_im = (g2i * dz2 * c2 - e2r * s2) - (g1i * dz1 * c2 + e1r * s2);
            const double bo_re = (-(g2r * dz2 * s2) + e2i * c2) - (g1r * dz1 * s2 + e1i * c2);
            const double bo_im = (-(g2i * dz2 * s2) - e2r * c2) - (g1i * dz1 * s2 - e1r * c2);
            const double pp = pr * proj;
            const cd er_s(pp * bs_im, -(pp * bs_re));
            const cd er_o(pp * bo_im, -(pp * bo_re));

            pc(m, n) = cd(td * ez_c.real(), td * ez_c.imag());
            ps(m, n) = cd(td * ez_s.real() + er_s.real(), td * ez_s.imag() + er_s.imag());
            po(m, n) = cd(td * ez_o.real() + er_o.real(), td * ez_o.imag() + er_o.imag());
        }
    }
    MW_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}

// ---------------------------------------------------------------------------
// The charged flat end cap (`_sinusoidal_nec4.cap_field`): on its axis the
// closed form (E-3), off it the gradient of Appendix E's second-order
// potential, (E-1) / (E-2).
py::array_t<cd> sinusoidal_cap_field(F64 obs_centers, F64 obs_tangents, F64 pos, F64 normal, double a, double k,
                                     double eta, cd c0) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto ps = pos.unchecked<1>();
    auto nm = normal.unchecked<1>();
    if (oc.shape(1) != 3 || ot.shape(1) != 3 || oc.shape(0) != ot.shape(0))
        throw std::runtime_error("observer arrays must have matching shape (M, 3)");
    if (ps.shape(0) != 3 || nm.shape(0) != 3) throw std::runtime_error("pos and normal must have shape (3,)");
    const size_t M = oc.shape(0);
    py::array_t<cd> out({M});
    auto o = out.mutable_unchecked<1>();
    const cd pref(eta * c0.real(), eta * c0.imag());
    const double nx = nm(0), ny = nm(1), nz = nm(2);
    const double a2 = a * a;
    for (size_t m = 0; m < M; m++) {
        const double dx = oc(m, 0) - ps(0), dy = oc(m, 1) - ps(1), dz = oc(m, 2) - ps(2);
        const double h = dx * nx + dy * ny + dz * nz;
        const double px = dx - h * nx, py_ = dy - h * ny, pz = dz - h * nz;
        const double rho = std::sqrt(px * px + py_ * py_ + pz * pz);
        const double tn = ot(m, 0) * nx + ot(m, 1) * ny + ot(m, 2) * nz;
        const double tr = px * ot(m, 0) + py_ * ot(m, 1) + pz * ot(m, 2);
        cd e;
        if (rho <= kCapAxisFraction * a) {
            const double ah = std::fabs(h);
            const double ra = std::sqrt(a2 + h * h);
            const double dd = a2 / (ra + ah);  // Ra - |h|
            // dd/Ra - (|h|/Ra)(e^{-jk dd} - 1)
            const double sh = s_sin(0.5 * (k * dd));
            const double em_re = -2.0 * sh * sh, em_im = -s_sin(k * dd);
            const double w = ah / ra;
            const double br_re = dd / ra - w * em_re, br_im = -(w * em_im);
            const double sg = h > 0.0 ? 1.0 : (h < 0.0 ? -1.0 : 0.0);
            const double f = (2.0 / a2) * sg;
            const double ph_re = s_cos(k * ah), ph_im = -s_sin(k * ah);
            const double eh_re = f * (ph_re * br_re - ph_im * br_im);
            const double eh_im = f * (ph_re * br_im + ph_im * br_re);
            e = cd(eh_re * tn, eh_im * tn);
        } else {
            const double r2 = h * h + rho * rho;
            const double r0 = std::sqrt(r2);
            const double kr = k * r0;
            const double k2r2 = k * k * r2;
            const double jk3 = k * k2r2 * r0;  // imaginary part of jk^3 R0^3
            const double h2 = h * h, p2 = rho * rho;
            // p = 8 R0^4 (1 + jkR0) + a^2 [3 (1 + jkR0) A + 2 k^2 R0^2 B - j k^3 R0^3 rho^2]
            const double base = 8.0 * r2 * r2;
            const double ah_ = 3.0 * p2 - 2.0 * h2, bh = h2 - 2.0 * p2;
            const double ar_ = p2 - 4.0 * h2, brr = 2.0 * h2 - p2;
            const double ph_re = base + a2 * (3.0 * ah_ + 2.0 * k2r2 * bh);
            const double ph_im = base * kr + a2 * (3.0 * kr * ah_ - jk3 * p2);
            const double pr_re = base + a2 * (3.0 * ar_ + 2.0 * k2r2 * brr);
            const double pr_im = base * kr + a2 * (3.0 * kr * ar_ - jk3 * p2);
            const double inv = 1.0 / (8.0 * r2 * r2 * r2 * r0);
            const double gr = s_cos(kr) * inv, gi = -s_sin(kr) * inv;
            // g (h p_h tn + p_r tr)
            const double s_re = h * ph_re * tn + pr_re * tr;
            const double s_im = h * ph_im * tn + pr_im * tr;
            e = cd(gr * s_re - gi * s_im, gr * s_im + gi * s_re);
        }
        o(m) = cd(pref.real() * e.real() - pref.imag() * e.imag(), pref.real() * e.imag() + pref.imag() * e.real());
    }
    return out;
}

// ---------------------------------------------------------------------------
// The junction-charge solve: `_nec4_junction`, statement for statement.

namespace {

const double kEulerGamma = 0.5772156649015329;
const double kPi = 3.141592653589793;  // math.pi
const double kHalfPi = 0.5 * kPi;
const double kTwoOverPi = 2.0 / kPi;
const int kBesselTerms = 40;
const double kBesselMaxArg = 8.0;
const int kGlOrder = 16;  // JUNCTION_GL_ORDER

struct Rule {
    std::vector<double> x, w;
};

// `_quadrature.leggauss(16)` (`_from_half`): the negative half mirrored,
// then the stored half.
const Rule &gl_rule() {
    static const Rule r = [] {
        Rule t;
        const int lo = leggauss_table::OFFSET[kGlOrder - 1], hi = leggauss_table::OFFSET[kGlOrder];
        const int h = hi - lo;
        const bool odd = (kGlOrder % 2) == 1;
        for (int i = h - 1; i >= (odd ? 1 : 0); i--) {
            t.x.push_back(-leggauss_table::NODES[lo + i]);
            t.w.push_back(leggauss_table::WEIGHTS[lo + i]);
        }
        for (int i = 0; i < h; i++) {
            t.x.push_back(leggauss_table::NODES[lo + i]);
            t.w.push_back(leggauss_table::WEIGHTS[lo + i]);
        }
        return t;
    }();
    return r;
}

double phi0(double k, double b, double z, double lo, double hi) {
    const Rule &g = gl_rule();
    const double b2 = b * b;
    double total = s_asinh((hi - z) / b) - s_asinh((lo - z) / b);
    double ends[3];
    int np = 0;
    ends[np++] = lo;
    if (lo < z && z < hi) ends[np++] = z;
    ends[np++] = hi;
    for (int piece = 0; piece + 1 < np; piece++) {
        const double p = ends[piece], q = ends[piece + 1];
        const double half = 0.5 * (q - p);
        const double mid = 0.5 * (q + p);
        double acc = 0.0;
        for (std::size_t i = 0; i < g.x.size(); i++) {
            const double u = mid + half * g.x[i] - z;
            const double r = std::sqrt(b2 + u * u);
            const double sh = s_sin(0.5 * k * r);
            acc = acc + g.w[i] * (-2.0 * sh * sh / r);
        }
        total = total + acc * half;
    }
    return total;
}

double bessel_y0(double x) {
    const double q = 0.25 * x * x;
    double term = 1.0, j0 = 1.0, s = 0.0, harmonic = 0.0;
    for (int m = 1; m <= kBesselTerms; m++) {
        term = -term * q / static_cast<double>(m * m);
        harmonic = harmonic + 1.0 / static_cast<double>(m);
        j0 = j0 + term;
        s = s - harmonic * term;
        if (std::fabs(term) < 1e-18) break;
    }
    return kTwoOverPi * ((s_log(0.5 * x) + kEulerGamma) * j0 + s);
}

double triangle(double k, double b, double z, double d) {
    const double r_hi = std::sqrt(b * b + (d - z) * (d - z));
    const double r_lo = std::sqrt(b * b + z * z);
    return (d - z) * phi0(k, b, z, 0.0, d) + (s_sin(k * r_lo) - s_sin(k * r_hi)) / k;
}

// `_full_triangle`: the n = 3 expansion's middle function, 0 at the node,
// d at s = d, 0 at s = 2d.
double full_triangle(double k, double b, double z, double d, double tri) {
    return d * phi0(k, b, z, 0.0, d) - tri + triangle(k, b, z - d, d);
}

// `_density`: member density at distance s from the node (or at it).
double density(const double *q, double d, double s, bool at_node) {
    if (at_node) return (q[0] + q[2]) * d;
    const double f1 = s < d ? d - s : 0.0;
    double f2;
    if (s <= d)
        f2 = s;
    else if (s < 2.0 * d)
        f2 = 2.0 * d - s;
    else
        f2 = 0.0;
    return q[0] * f1 + q[1] * f2 + q[2] * d;
}

double semi_infinite(double k, double b, double z) {
    double out = -kHalfPi * bessel_y0(k * b);
    if (z > 0.0)
        out = out + phi0(k, b, z, 0.0, z);
    else if (z < 0.0)
        out = out - phi0(k, b, z, z, 0.0);
    return out;
}

std::string fmt3g(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.3g", v);
    return buf;
}

}  // namespace

std::vector<double> junction_charge_factors(double k, const std::vector<std::array<double, 3>> &dirs,
                                            const std::vector<double> &radii, const std::vector<double> &lengths,
                                            double first_factor) {
    const std::size_t m = radii.size();
    const std::size_t n = 3 * m;
    // One length for every member: the mean of the junction segments'.
    double dsum = 0.0;
    for (std::size_t l = 0; l < m; l++) dsum = dsum + lengths[l];
    const double dl = dsum / static_cast<double>(m);
    std::vector<std::vector<double>> mat;
    for (std::size_t i = 0; i < m; i++) {
        const double ss[3] = {0.5 * dl, dl, 2.0 * dl};
        for (double s : ss) {
            const double p0 = s * dirs[i][0], p1 = s * dirs[i][1], p2 = s * dirs[i][2];
            std::vector<double> row;
            for (std::size_t l = 0; l < m; l++) {
                const auto &d = dirs[l];
                const double z = p0 * d[0] + p1 * d[1] + p2 * d[2];
                const double x0 = p0 - z * d[0], x1 = p1 - z * d[1], x2 = p2 - z * d[2];
                const double a = radii[l];
                const double b = std::sqrt(a * a + (x0 * x0 + x1 * x1 + x2 * x2));
                if (k * b > kBesselMaxArg)
                    throw std::domain_error(
                        "a junction member lies too far from another member's axis for the NEC-4 "
                        "junction-charge solve (k b = " +
                        fmt3g(k * b) + " > 8); shorten the segments at the junction (momwire#1644)");
                const double tri = triangle(k, b, z, dl);
                row.push_back(tri);
                row.push_back(full_triangle(k, b, z, dl, tri));
                row.push_back(dl * semi_infinite(k, b, z));
            }
            mat.push_back(row);
        }
    }
    std::vector<double> rhs(n, 1.0);
    for (std::size_t c = 0; c < n; c++) {
        std::size_t p = c;
        double best = std::fabs(mat[c][c]);
        for (std::size_t r = c + 1; r < n; r++) {
            const double v = std::fabs(mat[r][c]);
            if (v > best) {
                best = v;
                p = r;
            }
        }
        if (best == 0.0) throw std::runtime_error("singular junction-charge system");
        if (p != c) {
            std::swap(mat[c], mat[p]);
            std::swap(rhs[c], rhs[p]);
        }
        const double piv = mat[c][c];
        for (std::size_t r = c + 1; r < n; r++) {
            const double f = mat[r][c] / piv;
            for (std::size_t cc = c; cc < n; cc++) mat[r][cc] = mat[r][cc] - f * mat[c][cc];
            rhs[r] = rhs[r] - f * rhs[c];
        }
    }
    std::vector<double> x(n, 0.0);
    for (std::size_t rr = n; rr-- > 0;) {
        double acc = rhs[rr];
        for (std::size_t cc = rr + 1; cc < n; cc++) acc = acc - mat[rr][cc] * x[cc];
        x[rr] = acc / mat[rr][rr];
    }
    std::vector<double> dens(m);
    const bool at_node = m == 2;
    for (std::size_t l = 0; l < m; l++) dens[l] = density(&x[3 * l], dl, 0.5 * lengths[l], at_node);
    const double scale = first_factor / dens[0];
    std::vector<double> out{first_factor};
    for (std::size_t l = 1; l < m; l++) out.push_back(dens[l] * scale);
    return out;
}

}  // namespace sinusoidal
}  // namespace mw
