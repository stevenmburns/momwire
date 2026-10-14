#include "_accel_common.h"
#include "_mw_binding.h"
#include "_razor_core.h"

// razor section of the accelerator: the BINDING half (explore/core-binding-
// split). The kernels, their documentation and their arithmetic live in
// `_razor_core.cpp`, which includes no Python header. What stays here is only
// what is Python's: numpy arrays in (with the forcecast/c_style conversion
// the signatures always had), numpy arrays allocated for the results (so
// tracemalloc still sees them), the GIL released around the call, and the
// names, docstrings and keyword arguments Python calls.

using mw::binding::in;
using mw::binding::out;
using mw::binding::opt;
using cdarr = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;
using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using barr = py::array_t<bool, py::array::c_style | py::array::forcecast>;

template <bool COMPLEX_K, class K>
static std::pair<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
seg_moments_binding(darr obs, darr seg_p0, darr seg_t, darr seg_h, darr a,
                    darr xg, darr wg, K k, bool need_m1, iarr group_i,
                    iarr group_j, darr a_ek, uintptr_t cancel_flag,
                    py::object b_obs_obj) {
    // The pre-split checks that read ndim, in their order; every other shape
    // check is the core's.
    if (obs.ndim() != 2 || obs.shape(1) != 3)
        throw std::runtime_error("obs must have shape (n_obs, 3)");
    const size_t n_seg = (size_t)seg_h.shape(0);
    if (seg_p0.ndim() != 2 || seg_p0.shape(1) != 3 ||
        (size_t)seg_p0.shape(0) != n_seg)
        throw std::runtime_error("seg_p0 must have shape (n_seg, 3)");
    if (seg_t.ndim() != 2 || seg_t.shape(1) != 3 ||
        (size_t)seg_t.shape(0) != n_seg)
        throw std::runtime_error("seg_t must have shape (n_seg, 3)");
    darr b_obs;
    if (!b_obs_obj.is_none()) b_obs = b_obs_obj.cast<darr>();
    const size_t n_obs = (size_t)obs.shape(0);
    py::array_t<std::complex<double>> M0({n_obs, n_seg});
    py::array_t<std::complex<double>> M1(
        need_m1 ? std::vector<size_t>{n_obs, n_seg} : std::vector<size_t>{0, 0});
    // Views are formed while the GIL is held; the core runs without it.
    auto v_obs = in<double, 2>(obs), v_p0 = in<double, 2>(seg_p0), v_t = in<double, 2>(seg_t);
    auto v_h = in<double, 1>(seg_h), v_a = in<double, 1>(a);
    auto v_xg = in<double, 1>(xg), v_wg = in<double, 1>(wg);
    auto v_gi = opt<int64_t>(group_i), v_gj = opt<int64_t>(group_j);
    auto v_aek = opt<double>(a_ek), v_b = opt<double>(b_obs);
    auto v_m0 = out<std::complex<double>, 2>(M0), v_m1 = out<std::complex<double>, 2>(M1);
    {
        py::gil_scoped_release release;
        if (COMPLEX_K)
            mw::razor::seg_moments_cplx(v_obs, v_p0, v_t, v_h, v_a, v_xg, v_wg,
                                        std::complex<double>(k), need_m1, v_gi, v_gj,
                                        v_aek, cancel_flag, v_b, v_m0, v_m1);
        else
            mw::razor::seg_moments(v_obs, v_p0, v_t, v_h, v_a, v_xg, v_wg,
                                   std::real(std::complex<double>(k)), need_m1, v_gi,
                                   v_gj, v_aek, cancel_flag, v_b, v_m0, v_m1);
    }
    return {M0, M1};
}

static std::pair<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
razor_seg_moments(darr obs, darr seg_p0, darr seg_t, darr seg_h, darr a, darr xg,
                  darr wg, double k, bool need_m1, iarr group_i, iarr group_j,
                  darr a_ek, uintptr_t cancel_flag = 0, py::object b_obs = py::none()) {
    return seg_moments_binding<false>(obs, seg_p0, seg_t, seg_h, a, xg, wg, k, need_m1,
                                      group_i, group_j, a_ek, cancel_flag, b_obs);
}

static std::pair<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
razor_seg_moments_cplx(darr obs, darr seg_p0, darr seg_t, darr seg_h, darr a,
                       darr xg, darr wg, std::complex<double> k, bool need_m1,
                       iarr group_i, iarr group_j, darr a_ek,
                       uintptr_t cancel_flag = 0, py::object b_obs = py::none()) {
    // Im k > 0 is refused by the core before it reads anything.
    return seg_moments_binding<true>(obs, seg_p0, seg_t, seg_h, a, xg, wg, k, need_m1,
                                     group_i, group_j, a_ek, cancel_flag, b_obs);
}

static size_t t1_rows_or_zero(size_t n_obs, size_t n_path) {
    return (n_path != 0 && n_obs % n_path == 0) ? n_obs / n_path : 0;
}

static py::array_t<std::complex<double>>
razor_assemble_t1(cdarr M0, cdarr M1, iarr s_a, iarr s_b, darr h_a, darr h_b,
                  barr fall_a, barr fall_b, darr t_out, darr td_a, darr td_b,
                  darr wts, size_t n_path, iarr obs_row) {
    const size_t n_obs = static_cast<size_t>(in<double, 1>(wts).shape(0));
    py::array_t<std::complex<double>> T1(
        {t1_rows_or_zero(n_obs, n_path), static_cast<size_t>(in<int64_t, 1>(s_a).shape(0))});
    auto vm0 = in<std::complex<double>, 2>(M0), vm1 = in<std::complex<double>, 2>(M1);
    auto vsa = in<int64_t, 1>(s_a), vsb = in<int64_t, 1>(s_b);
    auto vha = in<double, 1>(h_a), vhb = in<double, 1>(h_b);
    auto vfa = in<bool, 1>(fall_a), vfb = in<bool, 1>(fall_b);
    auto vto = in<double, 2>(t_out), vta = in<double, 2>(td_a), vtb = in<double, 2>(td_b);
    auto vw = in<double, 1>(wts);
    auto vrow = opt<int64_t>(obs_row);
    auto vout = out<std::complex<double>, 2>(T1);
    {
        py::gil_scoped_release release;
        mw::razor::assemble_t1(vm0, vm1, vsa, vsb, vha, vhb, vfa, vfb, vto, vta, vtb,
                               vw, n_path, vrow, vout);
    }
    return T1;
}

static py::array_t<std::complex<double>>
razor_assemble_t1_weighted(cdarr M0, cdarr M1, iarr s_a, iarr s_b, darr h_a,
                           darr h_b, barr fall_a, barr fall_b, darr sig_a,
                           darr sig_b, darr obs_c, darr obs_t, darr src_c,
                           darr src_t, darr wts, size_t n_path, int window_kind,
                           std::complex<double> eps_t, double ground_z,
                           std::complex<double> coefficient, iarr obs_row) {
    const size_t n_obs = static_cast<size_t>(in<double, 1>(wts).shape(0));
    py::array_t<std::complex<double>> T1(
        {t1_rows_or_zero(n_obs, n_path), static_cast<size_t>(in<int64_t, 1>(s_a).shape(0))});
    auto vm0 = in<std::complex<double>, 2>(M0), vm1 = in<std::complex<double>, 2>(M1);
    auto vsa = in<int64_t, 1>(s_a), vsb = in<int64_t, 1>(s_b);
    auto vha = in<double, 1>(h_a), vhb = in<double, 1>(h_b);
    auto vfa = in<bool, 1>(fall_a), vfb = in<bool, 1>(fall_b);
    auto vga = in<double, 1>(sig_a), vgb = in<double, 1>(sig_b);
    auto voc = in<double, 2>(obs_c), vot = in<double, 2>(obs_t);
    auto vsc = in<double, 2>(src_c), vst = in<double, 2>(src_t);
    auto vw = in<double, 1>(wts);
    auto vrow = opt<int64_t>(obs_row);
    auto vout = out<std::complex<double>, 2>(T1);
    {
        py::gil_scoped_release release;
        mw::razor::assemble_t1_weighted(vm0, vm1, vsa, vsb, vha, vhb, vfa, vfb, vga, vgb,
                                        voc, vot, vsc, vst, vw, n_path, window_kind,
                                        eps_t, ground_z, coefficient, vrow, vout);
    }
    return T1;
}


void register_razor(py::module_ &m) {
    // The capability flag lives here, in the TU that defines the symbol it
    // vouches for (#710 review): a build carrying `razor_fill_742` carries
    // this kernel.
    m.attr("razor_fill_742") = true;

    m.def("razor_seg_moments", &razor_seg_moments,
          "Fused, tiled segment-moment fill for the razor-blade formulation "
          "(momwire#742). Returns (M0, M1), each (n_obs, n_seg) complex, with "
          "M0 = int_0^h g dtau and M1 = int_0^h tau g dtau over every source "
          "segment at every observer — the closed-form static moments of "
          "_kernel_moments plus the Gauss-Legendre smooth remainder "
          "(exp(-jkR)-1)/(4 pi R), computed together so the (n_obs, n_seg, "
          "n_qp) distance table the numpy path retains is never formed. M1 is "
          "returned as a (0, 0) array when need_m1 is false. group_i / "
          "group_j are the extended kernel's coaxial-equal-radius labels "
          "(_ek_pair_mask: pair (p, s) extends iff group_i[p] == group_j[s] "
          ">= 0); pass empty arrays for the reduced kernel. a is the axis "
          "frame's regularising radius per SOURCE segment and a_ek the EK "
          "radius, both as (n_seg,) columns.",
          py::arg("obs"), py::arg("seg_p0"), py::arg("seg_t"), py::arg("seg_h"),
          py::arg("a"), py::arg("xg"), py::arg("wg"), py::arg("k"),
          py::arg("need_m1"), py::arg("group_i"), py::arg("group_j"),
          py::arg("a_ek"), py::arg("cancel_flag") = 0,
          py::arg("b_obs") = py::none());

    // momwire#796, on its OWN flag for the same reason `razor_fill_742` is:
    // a .so built before this lands exports the real-k kernel and not this,
    // and the Python gate must be able to tell the difference rather than
    // assume one symbol implies the other.
    m.attr("razor_cplx_796") = true;
    m.def("razor_seg_moments_cplx", &razor_seg_moments_cplx,
          "The in-medium (complex k) twin of razor_seg_moments (momwire#796). "
          "Identical in shape and meaning; k is a complex wavenumber with "
          "Im k <= 0, and exp(-jkR) is continued as the real scale factor "
          "exp(Im(k) R) on the trig part, so the reduced branch's `- 1` and "
          "the k-free statics are untouched while the EK branch's C1/C2 pick "
          "up their terms in Im(k) R. Raises ValueError on Im k > 0, the "
          "growing-exponential branch this kernel will not evaluate.",
          py::arg("obs"), py::arg("seg_p0"), py::arg("seg_t"), py::arg("seg_h"),
          py::arg("a"), py::arg("xg"), py::arg("wg"), py::arg("k"),
          py::arg("need_m1"), py::arg("group_i"), py::arg("group_j"),
          py::arg("a_ek"), py::arg("cancel_flag") = 0,
          py::arg("b_obs") = py::none());

    // Design D4's vector bracket, exported for its accuracy test alone
    // (tests/test_razor_cplx_brackets.py): e^{-jkR} - 1 at every R, through
    // the same padded call the kernel above makes. Not a solver entry point.
    m.def("razor_cplx_brackets",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> R,
             std::complex<double> k) {
              const size_t n = (size_t)R.size();
              py::array_t<std::complex<double>> out(n);
              mw::razor::cplx_brackets(R.data(), n, k, out.mutable_data());
              return out;
          },
          "Design D4's vectorised e^{-jkR} - 1 (Im k <= 0), for its tests.",
          py::arg("R"), py::arg("k"));

    // momwire#780. Declared beside the moments kernel it pairs with.
    m.attr("razor_assemble_780") = true;
    m.def("razor_assemble_t1", &razor_assemble_t1,
          "Fused T1 assembly for the razor-blade formulation (momwire#780). "
          "Gathers both wings out of (M0, M1), applies the falling-wing "
          "correction, contracts each testing-path point's tangent with the "
          "source tangents, weights, and sums over the path points -- without "
          "forming the (n_obs, n_basis) complex intermediate the numpy path "
          "builds and immediately reduces. Returns T1 of shape "
          "(n_obs / n_path, n_basis). `n_path` is the only thing that differs "
          "between the two quadrature lanes (2 under nec5_quadrature, "
          "2*n_qp_path under Gauss-Legendre), so one kernel serves both. "
          "Unweighted integrand only: free space and folding grounds, i.e. "
          "the `w_A_fn is None` branch.",
          py::arg("M0"), py::arg("M1"), py::arg("s_a"), py::arg("s_b"),
          py::arg("h_a"), py::arg("h_b"), py::arg("fall_a"), py::arg("fall_b"),
          py::arg("t_out"), py::arg("td_a"), py::arg("td_b"), py::arg("wts"),
          py::arg("n_path"), py::arg("obs_row") = py::array_t<int64_t>());
    m.attr("razor_t1_rows_1290") = true;

    // momwire#744, on its OWN flag for the same reason every kernel above is:
    // a .so built before this lands exports the unweighted assembler and not
    // this, and the Python gate must be able to tell rather than assume the
    // #780 symbol implies this one.
    m.attr("razor_weighted_744") = true;
    m.def("razor_assemble_t1_weighted", &razor_assemble_t1_weighted,
          "The WEIGHTED twin of razor_assemble_t1 (momwire#744). Same gather, "
          "falling-wing correction, path weighting and path-point sum, with "
          "the real tangent contraction replaced by the complex per-pair "
          "A-term weight window -- formed inside the tile from the observer "
          "and source geometry, so the (n_obs_chunk, n_seg) window plane the "
          "numpy closure materialises is never built. "
          "`window_kind` is the ground's own answer about which window it "
          "is, carried in the FusedWindowRule that PotentialGround."
          "fused_window_rule returns (momwire#806): 0 is the stock "
          "specular_pair_tables -> fresnel_rho -> a_term_weights chain from "
          "eps_t and ground_z, 1 is `coefficient` times the mirrored tangent "
          "dot, which is a composing (sommerfeld) ground's window. The "
          "coefficient travels in the rule rather than being read off the "
          "ground by the caller -- C2 reaches Z through the windows by "
          "design, and a fill that applies image_coefficient itself doubles "
          "the exact-image half. A ground whose weights are not the stock "
          "Fresnel pair (standard_fresnel = False) returns no rule and never "
          "arrives here. Returns T1 of shape (n_obs / n_path, n_basis).",
          py::arg("M0"), py::arg("M1"), py::arg("s_a"), py::arg("s_b"),
          py::arg("h_a"), py::arg("h_b"), py::arg("fall_a"), py::arg("fall_b"),
          py::arg("sig_a"), py::arg("sig_b"), py::arg("obs_c"),
          py::arg("obs_t"), py::arg("src_c"), py::arg("src_t"),
          py::arg("wts"), py::arg("n_path"), py::arg("window_kind"),
          py::arg("eps_t"), py::arg("ground_z"), py::arg("coefficient"),
          py::arg("obs_row") = py::array_t<int64_t>());
}

