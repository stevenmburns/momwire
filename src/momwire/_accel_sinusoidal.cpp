#include "_accel_common.h"
#include "_mw_nd_binding.h"
#include "_sinusoidal_core.h"
#include "_sinusoidal_galerkin_core.h"

#include <string>
#include "_accel_sinusoidal_cplx.h"

// sinusoidal section of the former _accelerators.cpp monolith (momwire#687).

// The point-matched field tensors (`sinusoidal_field_tensor` and its
// _refl / _ek / _ek_refl / _cplx twins) live in `_sinusoidal_core.cpp`,
// Python-free, since momwire#1607 B1, and the Sinusoidal-Galerkin far fills
// (`sinusoidal_galerkin_far_fill` and its _ek / _cplx twins) in
// `_sinusoidal_galerkin_core.cpp` since momwire#1718 S1. What remains here is
// Python's: the names and arguments Python calls, numpy arrays in with the
// forcecast / c_style conversion the signatures always had, the check of the
// far fills' `out` argument, and the host (`_mw_nd_binding.h`) through which
// a core allocates its outputs with numpy and releases the GIL where it
// always did.

namespace sgc = mw::sinusoidal_galerkin;

// ===========================================================================
// The Sinusoidal-Galerkin far fills: the BINDING half (momwire#1718 S1).

// The far fills' `out`, checked: the three numpy destinations and the scale.
// The core folds into the same buffers (`sgc::GalerkinFoldBlock`, borrowed
// from these), and the binding hands these same arrays back.
struct GalerkinFoldArg {
    py::array_t<std::complex<double>> dst_const, dst_sin, dst_cos;
    double s_re, s_im;
};
// One of the three fold destinations, checked (momwire#356). It has to be
// exactly the array the allocating path would have returned — (nnz, N)
// complex128, C-contiguous and writeable — because the kernel folds into its
// buffer in place. A forcecast `array_t` would silently accept a copy here
// and the fold would land in the copy, so the cast below is the strict one
// and the shape/layout checks are explicit.
static py::array_t<std::complex<double>> galerkin_fold_dest(
    py::handle h, py::ssize_t nnz, py::ssize_t N, const char *which
) {
    if (!py::isinstance<py::array_t<std::complex<double>>>(h)) {
        throw std::runtime_error(
            std::string("out[") + which + "] must be a complex128 array");
    }
    auto a = py::reinterpret_borrow<py::array_t<std::complex<double>>>(h);
    if (a.ndim() != 2 || a.shape(0) != nnz || a.shape(1) != N) {
        throw std::runtime_error(
            std::string("out[") + which + "] must have shape (nnz, N)");
    }
    if ((a.flags() & py::array::c_style) == 0 || !a.writeable()) {
        throw std::runtime_error(
            std::string("out[") + which +
            " must be C-contiguous and writeable");
    }
    return a;
}

// `out` (a 3-sequence of destinations, or None) and `scale` into a fold
// block. Returns false for `out=None`, which is the pre-#356 behaviour: the
// fill allocates and returns its own triple and never touches a caller array.
static bool galerkin_fold_block(
    const py::object &out, std::complex<double> scale,
    py::ssize_t nnz, py::ssize_t N, GalerkinFoldArg &blk
) {
    if (out.is_none()) return false;
    auto seq = py::cast<py::sequence>(out);
    if (py::len(seq) != 3) {
        throw std::runtime_error(
            "out must be (const, sin, cos) — three (nnz, N) complex arrays");
    }
    blk.dst_const = galerkin_fold_dest(seq[0], nnz, N, "const");
    blk.dst_sin = galerkin_fold_dest(seq[1], nnz, N, "sin");
    blk.dst_cos = galerkin_fold_dest(seq[2], nnz, N, "cos");
    blk.s_re = scale.real();
    blk.s_im = scale.imag();
    return true;
}

// The core's view of a checked `out`, or null for `out=None`.
static const sgc::GalerkinFoldBlock *galerkin_fold_view(
    bool folding, const GalerkinFoldArg &arg, sgc::GalerkinFoldBlock &view
) {
    if (!folding) return nullptr;
    view.dst_const = ndb::in(arg.dst_const);
    view.dst_sin = ndb::in(arg.dst_sin);
    view.dst_cos = ndb::in(arg.dst_cos);
    view.s_re = arg.s_re;
    view.s_im = arg.s_im;
    return &view;
}

// A far fill's result: folding, the caller's own three arrays (the core
// filled them in place); not folding, the three the core allocated through
// the NumpyHost.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
galerkin_far_fill_result(
    bool folding, const GalerkinFoldArg &arg,
    const std::tuple<mw::nd::array_t<std::complex<double>>,
                     mw::nd::array_t<std::complex<double>>,
                     mw::nd::array_t<std::complex<double>>> &got
) {
    if (folding) return std::make_tuple(arg.dst_const, arg.dst_sin, arg.dst_cos);
    return ndb::out_tuple(got);
}
// The complex entry point. Separate from `sinusoidal_galerkin_far_fill` by
// design (momwire#980 step E): the real symbol's signature and branch
// conditions are untouched, so its bytes are unchanged and that is checked by
// rebuilding and hashing the fill's output rather than by reading a diff.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    std::complex<double> k, std::complex<double> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    uintptr_t cancel_flag = 0,
    py::object out = py::none(),
    std::complex<double> scale = std::complex<double>(1.0, 0.0),
    bool reference = false
) {
    GalerkinFoldArg fold;
    bool folding = galerkin_fold_block(
        out, scale, w_entry.shape(0), src_centers.shape(0), fold);
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    sgc::GalerkinFoldBlock view;
    return galerkin_far_fill_result(folding, fold, sgc::sinusoidal_galerkin_far_fill_cplx(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(obs_radius), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(src_hh), k, eta, ndb::in(gl_t), ndb::in(gl_w), ndb::in(w_entry), ndb::in(starts), cancel_flag,
        galerkin_fold_view(folding, fold, view), reference));
}


// The reduced entry point. Byte-frozen against its pre-#246 build: the
// instantiation below has WITH_EK false, so not one line of the delta is
// compiled into it (gate G-C2). momwire#356's `out`/`scale` are additive and
// default to `None`/1, which reaches `galerkin_far_fill_impl` with a null
// fold block — the allocating path, unchanged.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    uintptr_t cancel_flag = 0,
    py::object out = py::none(),
    std::complex<double> scale = std::complex<double>(1.0, 0.0),
    bool reference = false
) {
    GalerkinFoldArg fold;
    bool folding = galerkin_fold_block(
        out, scale, w_entry.shape(0), src_centers.shape(0), fold);
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    sgc::GalerkinFoldBlock view;
    return galerkin_far_fill_result(folding, fold, sgc::sinusoidal_galerkin_far_fill(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(obs_radius), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(src_hh), k, eta, ndb::in(gl_t), ndb::in(gl_w), ndb::in(w_entry), ndb::in(starts), cancel_flag,
        galerkin_fold_view(folding, fold, view), reference));
}

// The extended-kernel twin (momwire#246 unit C). Same arguments plus the EK
// payload — one source radius per segment, the pair rule's GROUP LABELS
// (momwire#358: per test segment and per source segment, the mask derived in
// the sweep), and the delta quadrature's composite rule — and the same three
// arrays out.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_obs,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_src,
    py::array_t<double, py::array::c_style | py::array::forcecast> ek_gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> ek_gw,
    uintptr_t cancel_flag = 0,
    py::object out = py::none(),
    std::complex<double> scale = std::complex<double>(1.0, 0.0)
) {
    // The payload's checks come first, as they always did: the core makes
    // them (`galerkin_ek_block`), here before `out` is checked and again in
    // the fill, which discards nothing a first pass did not already accept.
    sgc::galerkin_ek_block(ndb::in(src_a), ndb::in(group_obs), ndb::in(group_src),
                           ndb::in(ek_gx), ndb::in(ek_gw),
                           src_centers.shape(0), starts.shape(0));
    GalerkinFoldArg fold;
    bool folding = galerkin_fold_block(
        out, scale, w_entry.shape(0), src_centers.shape(0), fold);
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    sgc::GalerkinFoldBlock view;
    return galerkin_far_fill_result(folding, fold, sgc::sinusoidal_galerkin_far_fill_ek(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(obs_radius), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(src_hh), k, eta, ndb::in(gl_t), ndb::in(gl_w), ndb::in(w_entry), ndb::in(starts),
        ndb::in(src_a), ndb::in(group_obs), ndb::in(group_src), ndb::in(ek_gx), ndb::in(ek_gw), cancel_flag,
        galerkin_fold_view(folding, fold, view)));
}


// ==========================================================================
// Sommerfeld ground: batched six-integral evaluation (sommerfeld-perf-plan
// Phase 3). A faithful port of _sommerfeld.py's `_six_integrals` machinery
// (same 24-point Gauss rule, same adaptive bisection, same fig 13/14
// contours and tail handling), evaluated per (rho, h) node with OpenMP
// across nodes — the nodes are independent, and the pure-Python fill spends
// ~75% of its time on interpreter/ufunc dispatch this port removes.
//
// Clean-room note: ported from momwire's own _sommerfeld.py (itself
// implemented from the public-domain NEC-2 theory manual equations); the
// complex-argument Bessel/Hankel functions below are implemented from the
// Abramowitz & Stegun 9.1/9.2 ascending series and asymptotic expansions.
// No GPL Sommerfeld source (nec2c, nec2++/PyNEC, somnec) was consulted.
// ==========================================================================


// ===========================================================================
// The point-matched field tensors: the BINDING half (momwire#1607 B1). The
// kernels live in `_sinusoidal_core.cpp`, Python-free; what stays here is
// Python's: the names and arguments Python calls, numpy arrays in with the
// forcecast / c_style conversion the signatures always had, and the host
// (`_mw_nd_binding.h`) through which the core allocates its outputs with
// numpy and releases the GIL where it always did.

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    uintptr_t cancel_flag = 0
) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(seg_h), a, k, eta, ndb::in(gl_t), ndb::in(gl_w), cancel_flag));
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor_refl(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(seg_h), a, k, eta, ndb::in(gl_t), ndb::in(gl_w), ndb::in(cos_th), ndb::in(px), ndb::in(py_), ndb::in(tm_p), ndb::in(tn_p), eps_t, cancel_flag));
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    uintptr_t cancel_flag = 0
) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor_ek(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(seg_h), a, k, eta, ndb::in(gl_t), ndb::in(gl_w), ndb::in(src_a), ndb::in(ind1), ndb::in(ind2), cancel_flag));
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor_ek_refl(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(seg_h), a, k, eta, ndb::in(gl_t), ndb::in(gl_w), ndb::in(src_a), ndb::in(ind1), ndb::in(ind2), ndb::in(cos_th), ndb::in(px), ndb::in(py_), ndb::in(tm_p), ndb::in(tn_p), eps_t, cancel_flag));
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    std::complex<double> k, std::complex<double> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool folded,
    uintptr_t cancel_flag = 0
) {
    ndb::NumpyHost host;
    mw::nd::HostScope scope(host);
    return ndb::out_tuple(mw::sinusoidal::sinusoidal_field_tensor_cplx(
        ndb::in(obs_centers), ndb::in(obs_tangents), ndb::in(obs_radius), ndb::in(src_centers), ndb::in(src_tangents), ndb::in(seg_h), k, eta, ndb::in(gl_t), ndb::in(gl_w), folded, cancel_flag));
}

void register_sinusoidal(py::module_ &m) {
    // Capability flag, not a value (momwire#258). The two EKSCX entry points
    // dropped their build-wide `want_swapped` argument when the IRA arm went
    // per pair, and a STALE extension still exports both symbols under the
    // old arity — so `hasattr` alone would hand the new caller a TypeError
    // instead of the graceful numpy fallback the guards exist to give.
    // `sinusoidal.py` requires this attribute before it claims either
    // accelerator; an older build simply lacks it and takes the numpy
    // reference, which carries the same per-pair fix. Set HERE, beside the
    // bindings it vouches for (#710 review).
    m.attr("ek_ira_per_pair") = true;

    m.def("sinusoidal_field_tensor", &sinusoidal_field_tensor,
          "Tangential field tensor for the NEC2 three-term basis. Returns "
          "(Phi_const, Phi_sin, Phi_cos), each (M, N) complex. obs_*/src_* "
          "can be the same arrays (free-space build) or src_* mirrored "
          "(PEC image build).",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_refl", &sinusoidal_field_tensor_refl,
          "Reflection-coefficient finite-ground variant of "
          "sinusoidal_field_tensor: src_* are the MIRRORED image sources, "
          "and the tangential projection applies NEC's Fresnel field dyad "
          "t_m·D·E = rho_v·(t_m·E) − (rho_v+rho_h)·(t_m·p̂)(E·p̂) with "
          "rho_v/rho_h computed in-kernel at each pair's specular angle "
          "from eps_t and the k-independent tables (cos_th, px, py, tm_p, "
          "tn_p — see SinusoidalSolver._image_refl_band). Returns "
          "(Phi_const, Phi_sin, Phi_cos), each (M, N) complex.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cos_th"), py::arg("px"), py::arg("py"),
          py::arg("tm_p"), py::arg("tn_p"),
          py::arg("eps_t"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_ek", &sinusoidal_field_tensor_ek,
          "NEC extended-thin-wire-kernel variant of sinusoidal_field_tensor "
          "(EKSCX, nec2-1.2.1.2.f:3170-3234). Same (Phi_const, Phi_sin, "
          "Phi_cos) contract and the same observer-side scalar radius `a`, "
          "plus three per-SOURCE tables: `src_a` (NEC's BX, the source "
          "conductor radius the O(a²) expansion is about) and the int8 "
          "per-end gating codes `ind1`/`ind2` from "
          "SinusoidalSolver._ek_gating (0/1 -> GXX, 2 -> GX). "
          "EKSCX's IRA is NOT an argument: it is resolved per pair inside "
          "the kernel from that pair's own `rhx < src_a`, the same "
          "comparison that orders (rh, b) — see momwire#258, which removed "
          "the build-wide `want_swapped` scalar #245 took. "
          "src_* may be the MIRRORED image sources (the gating tables are "
          "unchanged there, as in NEC's KSYMP loop).",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("src_a"), py::arg("ind1"), py::arg("ind2"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_ek_refl", &sinusoidal_field_tensor_ek_refl,
          "Extended-thin-wire-kernel (EKSCX) variant of "
          "sinusoidal_field_tensor_refl: the EK tables of "
          "sinusoidal_field_tensor_ek (src_a / ind1 / ind2, with EKSCX's IRA "
          "resolved per pair in-kernel) "
          "with the Fresnel field dyad projection tail of "
          "sinusoidal_field_tensor_refl (cos_th, px, py, tm_p, tn_p, eps_t) "
          "in place of the plain tangential one. src_* are the MIRRORED "
          "image sources. Returns (Phi_const, Phi_sin, Phi_cos), each "
          "(M, N) complex.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("src_a"), py::arg("ind1"), py::arg("ind2"),
          py::arg("cos_th"), py::arg("px"), py::arg("py"),
          py::arg("tm_p"), py::arg("tn_p"),
          py::arg("eps_t"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_galerkin_far_fill", &sinusoidal_galerkin_far_fill,
          "Fused far fill for SinusoidalGalerkinSolver: the plainly-projected "
          "Eqs 76-79 tensor at each test segment's nq Gauss points, reduced "
          "on the fly against the test weights. Returns (contrib_const, "
          "contrib_sin, contrib_cos), each (nnz, N) complex, where nnz is the "
          "CSR support-entry count (`starts` is that CSR's per-test-segment "
          "row index). src_* are the geometry's segments (free-space block) "
          "or the MIRRORED ones (PEC image block); the reflection-coefficient "
          "and Sommerfeld blocks stay on the numpy path. With `out` — three "
          "(nnz, N) complex128 C-contiguous arrays — nothing is allocated "
          "and each entry is folded on as `out += scale * value`, scale on "
          "the LEFT of the complex product; the same three arrays come back. "
          "`out=None` with `scale=1` is the pre-momwire#356 behaviour to the "
          "bit. `reference=True` runs the per-pair scalar assembly instead "
          "of the staged vector one (perf item 9); the two are "
          "bit-identical, and the flag exists for the test that says so.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("src_hh"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("w_entry"), py::arg("starts"),
          py::arg("cancel_flag") = 0,
          py::arg("out") = py::none(),
          py::arg("scale") = std::complex<double>(1.0, 0.0),
          py::arg("reference") = false);
    m.def("sinusoidal_galerkin_far_fill_cplx",
          &sinusoidal_galerkin_far_fill_cplx,
          "Complex-wavenumber twin of sinusoidal_galerkin_far_fill "
          "(momwire#980 step E): the same three arrays for an IN-MEDIUM "
          "k_m = k0*sqrt(eps_tilde) and its eta_m, both std::complex. A "
          "separate body rather than a template instantiation, so the real "
          "symbol's bytes are unchanged — see momwire#991 for what differs "
          "and the gate that would retire the copy. Requires Im k <= 0 (the "
          "e^{+jwt} convention) and raises otherwise rather than conjugating. "
          "No extended-kernel twin: EK stays real-k, and its Bessel-polynomial "
          "split assumes jkR is purely imaginary. `reference=True` runs the "
          "per-pair scalar assembly instead of the staged vector one "
          "(momwire#1224); the two are bit-identical, and the flag exists for "
          "the test that says so.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("src_hh"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("w_entry"), py::arg("starts"),
          py::arg("cancel_flag") = 0,
          py::arg("out") = py::none(),
          py::arg("scale") = std::complex<double>(1.0, 0.0),
          py::arg("reference") = false);
    m.def("sg_real_far_fill_calls",
          []() {
              auto calls = sgc::sg_real_far_fill_calls();
              return std::make_tuple(calls.first, calls.second);
          },
          "(staged, reference): how many reduced (EK-off) "
          "sinusoidal_galerkin_far_fill calls each body has served in this "
          "process (perf item 9). The bodies' outputs are bit-identical, so "
          "this is the only way to see which one production took.");
    m.def("sg_cplx_far_fill_calls",
          []() {
              auto calls = sgc::sg_cplx_far_fill_calls();
              return std::make_tuple(calls.first, calls.second);
          },
          "(staged, reference): how many sinusoidal_galerkin_far_fill_cplx "
          "calls each body has served in this process (momwire#1224). The "
          "bodies' outputs are bit-identical, so this is the only way to "
          "see which one production took.");
    m.def("sg_cplx_phase_sweep",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> d,
             std::complex<double> k, bool pad) {
              // The far fill's Stage B on its own (momwire#1224), padded and
              // aligned exactly as the fill calls it, for the tests to pin
              // against scalar references. Returns (ea, em, s, h).
              // `pad=false` is the red control: the sweep over exactly n
              // entries, whose last n mod 4 take the loop's scalar tail.
              auto dv = d.unchecked<1>();
              const size_t n = (size_t)dv.shape(0);
              const size_t nv = (n + SG_CPLX_LANES - 1) / SG_CPLX_LANES
                                * SG_CPLX_LANES;
              std::vector<double> buf(5 * nv + 4, 0.0);
              double *b = buf.data();
              b += ((32 - (reinterpret_cast<uintptr_t>(b) & 31)) & 31)
                   / sizeof(double);
              for (size_t i = 0; i < n; i++) b[i] = dv(i);
              sg_cplx_phase_sweep(b, pad ? nv : n, k.real(), k.imag(),
                                  b + nv, b + 2 * nv, b + 3 * nv, b + 4 * nv);
              py::array_t<double> ea(n), em(n), s(n), h(n);
              double *outs[4] = {ea.mutable_data(), em.mutable_data(),
                                 s.mutable_data(), h.mutable_data()};
              for (int t = 0; t < 4; t++)
                  std::copy(b + (t + 1) * nv, b + (t + 1) * nv + n, outs[t]);
              return py::make_tuple(ea, em, s, h);
          },
          "The complex-k Galerkin far fill's vector sweep (momwire#1224): "
          "for distances d and complex k, (exp(a), expm1(a), sin(y), "
          "sin(y/2)) with a = Im(k)*d, y = Re(k)*d.",
          py::arg("d"), py::arg("k"), py::arg("pad") = true);

    m.def("sinusoidal_field_tensor_cplx", &sinusoidal_field_tensor_cplx,
          "Complex-wavenumber twin of sinusoidal_field_tensor (momwire#1222): "
          "the point-matched (3, M, N) tangential-field tensor at an IN-MEDIUM "
          "k_m and eta_m, both std::complex, with a per-OBSERVER radius and "
          "`folded` choosing the third shape (false: cos k(xi); true: "
          "cos k(xi) - 1, #205). Requires Im k <= 0 and raises otherwise.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("seg_h"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"), py::arg("folded"),
          py::arg("cancel_flag") = 0);

    m.def("sinusoidal_galerkin_far_fill_ek", &sinusoidal_galerkin_far_fill_ek,
          "Extended-kernel twin of sinusoidal_galerkin_far_fill "
          "(momwire#246): the same fused far fill, plus the folded EK delta "
          "on the pairs the group labels select. `src_a` is one radius per "
          "SOURCE segment; `group_obs` (one label per TEST segment, i.e. "
          "len(starts) - 1) and `group_src` (one per source segment) are the "
          "pair rule's coaxial-and-equal-radius labels, from which the sweep "
          "derives eligibility as `g_obs[m] >= 0 and g_obs[m] == g_src[n]` "
          "(momwire#358 — the (obs rows, N) mask this replaced was constant "
          "over each test segment's observer rows). `ek_gx`/`ek_gw` are the "
          "composite sinh-mapped rule "
          "`SinusoidalSolver._ek_delta_rule` built — passed in, not rebuilt "
          "here, so both backends integrate the delta on the same nodes. "
          "Returns the same three (nnz, N) arrays, and takes the same "
          "momwire#356 `out`/`scale` fold destination.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("src_hh"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("w_entry"), py::arg("starts"),
          py::arg("src_a"), py::arg("group_obs"), py::arg("group_src"),
          py::arg("ek_gx"), py::arg("ek_gw"),
          py::arg("cancel_flag") = 0,
          py::arg("out") = py::none(),
          py::arg("scale") = std::complex<double>(1.0, 0.0));
}
