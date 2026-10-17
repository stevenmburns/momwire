// razor-2p's k-dependent solve in one C++ call: see `_razor_solve_core.h`.
//
// Each window replays `RazorSolver._source_block_rows` on its
// `t1_on_centroids` branch, step for step:
//
//   need    = unique(s_a[lo:hi] ++ s_b[lo:hi])          (sorted)
//   M0, M1  = razor_seg_moments(cent[need], <source set>, need_m1=True)
//   obs_row = searchsorted(need, [s_a, s_b] interleaved per row)
//   T1      = razor_assemble_t1(M0, M1, ..., obs_row=obs_row)
//   plus    = searchsorted(need, s_b[lo:hi]); minus = searchsorted(need, s_a)
//             with minus = -1 on a grounded row
//   rows    = razor_t2_rows(out, T1, M0w = M0, M0k = (0, n_seg), plus, minus,
//                           ..., Im c_A, Im c_Phi)     (fused: c_A T1 - T2/c_Phi)
//
// The real block's rows go straight into Z's window; the image block's go
// into a scratch window that is then subtracted (`_schedule.fold_rows`'s
// `out[lo:hi] -= rows`). Python walks every real window and then every image
// window; here each window does both. Every entry is computed by the same
// kernels from the same operands and combined by the same one subtraction,
// so the order moves no value. The `_m0w` gather of the Python route is the
// identity on this branch (its runs of `need` are consecutive slices of the
// shared M0, concatenated back in order), so M0w IS M0.
//
// Over the composing ground (momwire#1479 S3) the image window is the same
// replay with the ground's three differences, in `_source_block_rows`' order:
//
//   T1      = razor_assemble_t1_weighted(M0, M1, ..., path points and their
//             tangents, centroids and segment tangents, CONSTANT_MIRROR,
//             eps~, ground_z, C2, obs_row=obs_row)
//   M0w     = M0 * w_Phi, w_Phi = C2 everywhere (numpy's fused complex
//             product, `_cmul`'s formula: the CPUs the gates run on fuse it)
//   Q       = razor_q_rows(remainder_moment_reduce(remainder_field_proj_batch(
//             path points, their tangents, the remainder's source nodes,
//             ...), W), arange, ...)
//   rows    = razor_t2_rows(..., M0w, ...) + Q      (`C2 img + Q`, then the
//             one `out[lo:hi] -= rows`)
#include "_study_timers.h"
#include "_razor_solve_core.h"

#include "_razor_core.h"
#include "_somm_proj_core.h"
#include "_somm_grid_core.h"

#include <cstring>
#include <unordered_map>

namespace mw {
namespace razor {

namespace {

template <class T, int N>
view<T, N> c_view(T *p, std::initializer_list<ptrdiff_t> shape) {
    return view<T, N>::contiguous(p, shape);
}

// Rows [r0, r1) of a C-contiguous-in-rows 2-D view (any row stride).
template <class T>
view<T, 2> row_slice(const view<T, 2> &v, ptrdiff_t r0, ptrdiff_t r1) {
    ptrdiff_t shp[2] = {r1 - r0, v.n[1]};
    return view<T, 2>(reinterpret_cast<T *>(reinterpret_cast<char *>(
                          const_cast<typename std::remove_const<T>::type *>(v.ptr)) +
                                            r0 * v.st[0]),
                      shp, v.st);
}

template <class T>
view<T, 1> slice1(const view<T, 1> &v, ptrdiff_t r0, ptrdiff_t r1) {
    ptrdiff_t shp[1] = {r1 - r0};
    return view<T, 1>(reinterpret_cast<T *>(reinterpret_cast<char *>(
                          const_cast<typename std::remove_const<T>::type *>(v.ptr)) +
                                            r0 * v.st[0]),
                      shp, v.st);
}

// numpy's fused complex product (`_cmul.cmul`'s formula) and its complex
// division (`CDOUBLE_divide`, Smith's), as the run path spells them.
inline cd np_cmul(cd a, cd b) {
    return cd(std::fma(a.real(), b.real(), -(a.imag() * b.imag())),
              std::fma(a.real(), b.imag(), a.imag() * b.real()));
}
inline cd np_cdiv(cd a, cd b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}

void moments(const SolveInputs &in, view<const double, 2> obs, const SourceSet &src,
             bool need_m1, view<const int64_t, 1> gi, view<const double, 1> bo, view<cd, 2> m0,
             view<cd, 2> m1) {
    STUDY_T(in.k_complex ? "razor.moments_cplx" : "razor.moments_real");
    if (in.k_complex)
        seg_moments_cplx(obs, src.seg_p0, src.seg_t, in.seg_h, in.a, in.xg, in.wg, in.k_c, need_m1,
                         gi, src.group_j, src.a_ek, in.cancel_flag, bo, m0, m1);
    else
        seg_moments(obs, src.seg_p0, src.seg_t, in.seg_h, in.a, in.xg, in.wg, in.k, need_m1, gi,
                    src.group_j, src.a_ek, in.cancel_flag, bo, m0, m1);
}

// `_distinct_observers(o_p, o_t)`: the rows (point, tangent) equal to the
// BIT (so -0.0 is not 0.0) grouped in first-appearance order. `first` gets
// each group's first row, `inv` each row's group; false (both untouched)
// when every row is distinct, as `(None, None)` answers there.
bool distinct_rows(const double *pts, const double *tans, ptrdiff_t m, std::vector<int64_t> &first,
                   std::vector<int64_t> &inv) {
    if (m < 2) return false;
    struct Key {
        uint64_t b[6];
        bool operator==(const Key &o) const { return std::memcmp(b, o.b, sizeof b) == 0; }
    };
    struct Hash {
        size_t operator()(const Key &k) const {
            uint64_t h = 1469598103934665603ull;
            for (uint64_t v : k.b) h = (h ^ v) * 1099511628211ull;
            return static_cast<size_t>(h ^ (h >> 29));
        }
    };
    std::unordered_map<Key, int64_t, Hash> seen;
    seen.reserve(static_cast<size_t>(m));
    first.clear();
    if (inv.size() < static_cast<size_t>(m)) inv.resize(static_cast<size_t>(m));
    for (ptrdiff_t o = 0; o < m; o++) {
        Key k;
        std::memcpy(k.b, pts + 3 * o, 3 * sizeof(double));
        std::memcpy(k.b + 3, tans + 3 * o, 3 * sizeof(double));
        auto it = seen.emplace(k, static_cast<int64_t>(first.size()));
        if (it.second) first.push_back(o);
        inv[static_cast<size_t>(o)] = it.first->second;
    }
    return static_cast<ptrdiff_t>(first.size()) < m;
}

void poll(uintptr_t flag) {
    const volatile int32_t *p = reinterpret_cast<const volatile int32_t *>(flag);
    if (p && *p) throw AbortedError{};
}

}  // namespace

void SolveContext::window(const SolveInputs &in, const SourceSet &src, ptrdiff_t lo,
                          ptrdiff_t hi, bool image) {
    const ptrdiff_t n = n_, nb = hi - lo, n_seg = in.seg_h.shape(0);
    const int64_t *sa = in.s_a.data(), *sb = in.s_b.data();

    // need: the sorted distinct centroids the window's rows read, and each
    // one's position in it (mark_), which is searchsorted on a sorted set.
    reserve(mark_, static_cast<size_t>(n_seg));
    std::fill(mark_.begin(), mark_.begin() + n_seg, int64_t(-1));
    for (ptrdiff_t r = lo; r < hi; r++) {
        mark_[sa[r]] = 0;
        mark_[sb[r]] = 0;
    }
    reserve(need_, static_cast<size_t>(n_seg));
    ptrdiff_t n_need = 0;
    for (ptrdiff_t s = 0; s < n_seg; s++)
        if (mark_[s] == 0) {
            mark_[s] = n_need;
            need_[n_need++] = s;
        }

    // The centroid token taken at `need` (`_FusedMoments.take`): the observer
    // rows and, under the EK, their labels and tubes.
    reserve(obs_, static_cast<size_t>(n_need * 3));
    for (ptrdiff_t p = 0; p < n_need; p++)
        for (int c = 0; c < 3; c++) obs_[p * 3 + c] = in.cent(need_[p], c);
    view<const int64_t, 1> gi;
    view<const double, 1> bo;
    if (in.group_i.size() > 0) {
        reserve(gi_, static_cast<size_t>(n_need));
        for (ptrdiff_t p = 0; p < n_need; p++) gi_[p] = in.group_i(need_[p]);
        gi = c_view<const int64_t, 1>(gi_.data(), {n_need});
    }
    if (in.b_obs.size() > 0) {
        reserve(bo_, static_cast<size_t>(n_need));
        for (ptrdiff_t p = 0; p < n_need; p++) bo_[p] = in.b_obs(need_[p]);
        bo = c_view<const double, 1>(bo_.data(), {n_need});
    }

    const size_t plane = static_cast<size_t>(n_need * n_seg);
    reserve(m0_, plane);
    reserve(m1_, plane);
    auto vm0 = c_view<cd, 2>(m0_.data(), {n_need, n_seg});
    auto vm1 = c_view<cd, 2>(m1_.data(), {n_need, n_seg});
    moments(in, c_view<const double, 2>(obs_.data(), {n_need, 3}), src, true, gi, bo, vm0, vm1);
    counters.seg_moments_calls++;

    // T1, with each path point reading its centroid's moment row.
    const ptrdiff_t np = static_cast<ptrdiff_t>(in.n_path);
    reserve(obs_row_, static_cast<size_t>(nb * np));
    for (ptrdiff_t r = 0; r < nb; r++) {
        obs_row_[2 * r] = mark_[sa[lo + r]];
        obs_row_[2 * r + 1] = mark_[sb[lo + r]];
    }
    reserve(t1_, static_cast<size_t>(nb * n));
    auto vt1 = c_view<cd, 2>(t1_.data(), {nb, n});
    const bool compose = image && in.compose;
    if (compose) {
        STUDY_T("win.assemble_t1_weighted");
        const Compose &c = in.comp;
        assemble_t1_weighted(view<const cd, 2>(vm0), view<const cd, 2>(vm1), in.s_a, in.s_b,
                             in.h_a, in.h_b, in.fall_a, in.fall_b, c.sig_a, c.sig_b,
                             row_slice(c.obs_pts, lo * np, hi * np),
                             row_slice(c.obs_tans, lo * np, hi * np), c.src_c, c.src_t,
                             slice1(in.wts, lo * np, hi * np), in.n_path,
                             /* WINDOW_RULE_CONSTANT_MIRROR */ 1, c.eps_t, c.ground_z, c.c2,
                             c_view<const int64_t, 1>(obs_row_.data(), {nb * np}), vt1);
    } else {
        STUDY_T("win.assemble_t1");
        assemble_t1(view<const cd, 2>(vm0), view<const cd, 2>(vm1), in.s_a, in.s_b, in.h_a,
                    in.h_b, in.fall_a, in.fall_b, row_slice(in.tans, lo * np, hi * np),
                    src.td_a, src.td_b, slice1(in.wts, lo * np, hi * np), in.n_path,
                    c_view<const int64_t, 1>(obs_row_.data(), {nb * np}), vt1);
    }
    counters.assemble_t1_calls++;

    // T2's row references into M0 (`_t2_tables`), grounded rows dropping
    // their A centroid.
    reserve(plus_, static_cast<size_t>(nb));
    reserve(minus_, static_cast<size_t>(nb));
    for (ptrdiff_t r = 0; r < nb; r++) {
        plus_[r] = mark_[sb[lo + r]];
        minus_[r] = mark_[sa[lo + r]];
    }
    for (ptrdiff_t g = 0; g < in.grounded.shape(0); g++) {
        const int64_t row = in.grounded(g);
        if (row >= lo && row < hi) minus_[row - lo] = -1;
    }
    // A chopped row (`_t2_tables`): "A" is knot - centroid(A), "B" is
    // centroid(B) - knot, the knot being row n_need + (its chop index) of the
    // stacked table [M0w; M0k]. After the grounded rows, as Python orders it.
    const ptrdiff_t n_chop = in.chop_rows.shape(0);
    for (ptrdiff_t q = 0; q < n_chop; q++) {
        const int64_t row = in.chop_rows(q);
        if (row < lo || row >= hi) continue;
        if (in.chop_keep_a(q)) {
            plus_[row - lo] = n_need + q;
            minus_[row - lo] = mark_[sa[row]];
        } else {
            plus_[row - lo] = mark_[sb[row]];
            minus_[row - lo] = n_need + q;
        }
    }
    const std::vector<cd> &m0k_tab = image ? m0k_img_ : m0k_;
    ptrdiff_t k_shape[2] = {n_chop, n_seg}, k_st[2] = {static_cast<ptrdiff_t>(sizeof(cd) * n_seg),
                                                       static_cast<ptrdiff_t>(sizeof(cd))};
    view<const cd, 2> m0k(n_chop ? m0k_tab.data() : nullptr, k_shape, k_st);
    const view<const cd, 2> t1c(vt1);
    view<cd, 2> dst;
    if (!image) {
        // Z's window: column-major, so the rows are the short stride.
        ptrdiff_t shp[2] = {nb, n}, st[2] = {static_cast<ptrdiff_t>(sizeof(cd)),
                                             static_cast<ptrdiff_t>(sizeof(cd) * n)};
        dst = view<cd, 2>(z_.data() + lo, shp, st);
    } else {
        reserve(img_, static_cast<size_t>(nb * n));
        dst = c_view<cd, 2>(img_.data(), {nb, n});
    }
    view<const cd, 2> m0w(vm0);
    if (compose) {
        // `M0c[a - c0 : b - c0] *= w_Phi`: each part one fused multiply-add.
        const cd c2 = in.comp.c2;
        reserve(m0c_, plane);
        const cd *src0 = m0_.data();
        cd *dst0 = m0c_.data();
        for (size_t i = 0; i < plane; i++) {
            const double ar = src0[i].real(), ai = src0[i].imag();
            dst0[i] = cd(std::fma(ar, c2.real(), -(ai * c2.imag())),
                         std::fma(ar, c2.imag(), ai * c2.real()));
        }
        m0w = c_view<const cd, 2>(m0c_.data(), {n_need, n_seg});
    }
    if (!in.unfused) {
        STUDY_T("win.t2_rows");
        t2_rows(dst, &t1c, m0w, m0k,
                c_view<const int64_t, 1>(plus_.data(), {nb}),
                c_view<const int64_t, 1>(minus_.data(), {nb}), in.s_a, in.s_b, in.q_a, in.q_b,
                in.c_a_imag, in.c_phi_imag);
    } else {
        // T2 alone, then `np.multiply(c_A, t1)`, `np.divide(t2, c_Phi)`,
        // `np.subtract(t1, t2)`, elementwise in that order.
        STUDY_T("win.t2_rows_unfused+elem");
        t2_rows(dst, nullptr, m0w, m0k,
                c_view<const int64_t, 1>(plus_.data(), {nb}),
                c_view<const int64_t, 1>(minus_.data(), {nb}), in.s_a, in.s_b, in.q_a, in.q_b,
                in.c_a_imag, in.c_phi_imag);
        for (ptrdiff_t r = 0; r < nb; r++)
            for (ptrdiff_t j = 0; j < n; j++) {
                const cd t1v = np_cmul(in.c_a, vt1(r, j));
                const cd t2v = np_cdiv(dst(r, j), in.c_phi);
                dst(r, j) = cd(t1v.real() - t2v.real(), t1v.imag() - t2v.imag());
            }
    }
    counters.t2_rows_calls++;
    if (compose) {
        // The remainder over the window's path points (`field_windows`,
        // `distinct=True`), its arc moments, and the Q rows, added before
        // the one minus: `rows_T1 += Q_rows`.
        // Below (`RemainderBelow`), only the window's bit-distinct (point,
        // tangent) rows are projected and each row reads its group's moments
        // (`distinct=True`'s `(moments, inverse)`): under nec5_quadrature each
        // wing centroid is observed by both bases ending on it. A row's
        // projection is its own point's and tangent's, so the gathered rows
        // are the rows the whole window gave.
        const Compose &c = in.comp;
        const ptrdiff_t m_obs = nb * np, s_src = c.rem_src.shape(0);
        const double *w_pts = &c.obs_pts(lo * np, 0), *w_tans = &c.obs_tans(lo * np, 0);
        ptrdiff_t m_u = m_obs;
        reserve(q_row_, static_cast<size_t>(m_obs));
        bool gathered = false;
        if (c.proj && distinct_rows(w_pts, w_tans, m_obs, need_dist_, q_row_)) {
            m_u = static_cast<ptrdiff_t>(need_dist_.size());
            reserve(dpt_, static_cast<size_t>(3 * m_u));
            reserve(dtan_, static_cast<size_t>(3 * m_u));
            for (ptrdiff_t u = 0; u < m_u; u++)
                for (int c3 = 0; c3 < 3; c3++) {
                    dpt_[3 * u + c3] = w_pts[3 * need_dist_[u] + c3];
                    dtan_[3 * u + c3] = w_tans[3 * need_dist_[u] + c3];
                }
            w_pts = dpt_.data();
            w_tans = dtan_.data();
            gathered = true;
        }
        if (!gathered)
            for (ptrdiff_t o = 0; o < m_obs; o++) q_row_[o] = o;
        reserve(proj_, static_cast<size_t>(m_u * s_src));
        if (c.proj)
            c.proj(w_pts, w_tans, m_u, proj_.data());
        else
            somm_core::field_proj_batch(*c.grid, c.n_reg, m_u, s_src, w_pts, w_tans,
                                        c.rem_src.data(), c.rem_t_src.data(), c.ground_z, in.k,
                                        proj_.data(), in.cancel_flag, true);
        if (somm_core::lazy_check())
            somm_core::check_read(proj_.data(), static_cast<size_t>(m_u * s_src),
                                  c.proj ? "the below compose window" : "the above compose window");
        reserve(mom_, static_cast<size_t>(m_u * n_seg * 2));
        somm_core::moment_reduce(proj_.data(), c.W, 2, n_seg, c.n_qp, m_u, mom_.data());
        reserve(q_, static_cast<size_t>(nb * n));
        STUDY_T("win.q_rows+add");
        q_rows(c_view<const cd, 3>(mom_.data(), {m_u, n_seg, 2}),
               c_view<const int64_t, 1>(q_row_.data(), {m_obs}), in.s_a, in.s_b, in.h_a,
               in.h_b, in.fall_a, in.fall_b, c.sig_a, c.sig_b,
               slice1(in.wts, lo * np, hi * np), np, c_view<cd, 2>(q_.data(), {nb, n}));
        cd *im = img_.data();
        const cd *q = q_.data();
        for (size_t i = 0, e = static_cast<size_t>(nb * n); i < e; i++) im[i] += q[i];
        counters.compose_windows++;
    }
    if (image) {
        STUDY_T("win.image_subtract");
        // `out[lo:hi] -= rows`: one complex subtraction per entry.
        cd *z = z_.data();
        const cd *im = img_.data();
        for (ptrdiff_t j = 0; j < n; j++) {
            cd *col = z + j * n + lo;
            for (ptrdiff_t r = 0; r < nb; r++) col[r] -= im[r * n + j];
        }
        counters.image_windows++;
    }
}

// `_seg_moments_from_prepared(t2_chop_chunks, k, n_chop, need_m1=False)`
// against one source set, and over the composing ground's image times its
// w_Phi (C2): `M0k * w_Phi_k`, numpy's fused product.
void SolveContext::knot_moments(const SolveInputs &in, const SourceSet &src, bool image,
                                std::vector<cd> &out) {
    const ptrdiff_t n_chop = in.chop_rows.shape(0), n_seg = in.seg_h.shape(0);
    out.assign(static_cast<size_t>(n_chop * n_seg), cd(0.0, 0.0));
    if (n_chop == 0) return;
    ptrdiff_t z_shape[2] = {0, 0}, z_st[2] = {static_cast<ptrdiff_t>(sizeof(cd)),
                                              static_cast<ptrdiff_t>(sizeof(cd))};
    view<cd, 2> none(nullptr, z_shape, z_st);
    moments(in, in.chop_pts, src, false, view<const int64_t, 1>(), view<const double, 1>(),
            c_view<cd, 2>(out.data(), {n_chop, n_seg}), none);
    counters.seg_moments_calls++;
    if (image && in.compose) {
        const cd c2 = in.comp.c2;
        for (cd &v : out) {
            const double ar = v.real(), ai = v.imag();
            v = cd(std::fma(ar, c2.real(), -(ai * c2.imag())), std::fma(ar, c2.imag(), ai * c2.real()));
        }
    }
}

void SolveContext::fill(const SolveInputs &in) { fill(in, nullptr); }

void SolveContext::fill(const SolveInputs &in, const std::function<void(cd *)> &between) {
    STUDY_T(in.k_complex ? "razor.fill_part_below" : "razor.fill_part_above");
    const auto t0 = std::chrono::steady_clock::now();
    const ptrdiff_t n = in.s_a.shape(0);
    const char *who = "razor_solve";
    if (in.n_path != 2)
        throw std::invalid_argument("razor_solve: only the NEC-5 path rule (n_path = 2)");
    if (in.tans.shape(0) != n * 2 || in.wts.shape(0) != n * 2)
        throw std::invalid_argument("razor_solve: tans / wts need n_path rows per basis");
    if (in.window_rows < 1)
        throw std::invalid_argument("razor_solve: window_rows must be >= 1");
    if (!in.unfused && !(in.c_phi_imag != 0.0))
        throw std::invalid_argument("razor_solve: c_Phi must be nonzero");
    require_contiguous(in.s_a, who, "s_a");
    require_contiguous(in.s_b, who, "s_b");
    const ptrdiff_t n_seg = in.seg_h.shape(0);
    for (ptrdiff_t j = 0; j < n; j++)
        if (in.s_a(j) < 0 || in.s_a(j) >= n_seg || in.s_b(j) < 0 || in.s_b(j) >= n_seg)
            throw std::invalid_argument("razor_solve: wing segment index out of range");
    if (in.cent.shape(0) != n_seg || in.cent.shape(1) != 3)
        throw std::invalid_argument("razor_solve: cent must be (n_seg, 3)");
    if (in.group_i.size() > 0 && in.group_i.shape(0) != n_seg)
        throw std::invalid_argument("razor_solve: group_i must be (n_seg,)");
    if (in.b_obs.size() > 0 && in.b_obs.shape(0) != n_seg)
        throw std::invalid_argument("razor_solve: b_obs must be (n_seg,)");
    if (in.compose) {
        const Compose &c = in.comp;
        if (!in.has_image || (c.grid == nullptr && !c.proj) || c.W == nullptr || c.n_qp < 1)
            throw std::invalid_argument("razor_solve: a composing ground needs its image and remainder");
        if (c.obs_pts.shape(0) != n * 2 || c.obs_tans.shape(0) != n * 2 ||
            c.src_c.shape(0) != n_seg || c.src_t.shape(0) != n_seg ||
            c.sig_a.shape(0) != n || c.sig_b.shape(0) != n ||
            c.rem_src.shape(0) != n_seg * c.n_qp || c.rem_t_src.shape(0) != n_seg * c.n_qp)
            throw std::invalid_argument("razor_solve: the composing ground's tables are mis-shaped");
        require_contiguous(c.obs_pts, who, "obs_pts");
        require_contiguous(c.obs_tans, who, "obs_tans");
        require_contiguous(c.rem_src, who, "rem_src");
        require_contiguous(c.rem_t_src, who, "rem_t_src");
    }

    if (in.chop_rows.shape(0) > 0) {
        if (in.chop_keep_a.shape(0) != in.chop_rows.shape(0) ||
            in.chop_pts.shape(0) != in.chop_rows.shape(0))
            throw std::invalid_argument("razor_solve: the chopped rows' tables are mis-shaped");
        knot_moments(in, in.real, false, m0k_);
        if (in.has_image) knot_moments(in, in.image, true, m0k_img_);
    } else {
        m0k_.clear();
        m0k_img_.clear();
    }
    n_ = n;
    const size_t nn = static_cast<size_t>(n * n);
    if (z_.size() < nn) {
        z_.resize(nn);
        counters.pool_grows++;
    }
    if (!between) {
        for (ptrdiff_t lo = 0; lo < n; lo += in.window_rows) {
            const ptrdiff_t hi = std::min(n, lo + in.window_rows);
            poll(in.cancel_flag);
            window(in, in.real, lo, hi, false);
            if (in.has_image) window(in, in.image, lo, hi, true);
            counters.windows++;
        }
    } else {
        for (ptrdiff_t lo = 0; lo < n; lo += in.window_rows) {
            poll(in.cancel_flag);
            window(in, in.real, lo, std::min(n, lo + in.window_rows), false);
            counters.windows++;
        }
        poll(in.cancel_flag);
        between(z_.data());
        if (in.has_image)
            for (ptrdiff_t lo = 0; lo < n; lo += in.window_rows) {
                poll(in.cancel_flag);
                window(in, in.image, lo, std::min(n, lo + in.window_rows), true);
            }
    }
    counters.solves++;
    counters.fill_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void SolveContext::release() {
    n_ = 0;
    std::vector<cd>().swap(z_);
    std::vector<cd>().swap(m0_);
    std::vector<cd>().swap(m1_);
    std::vector<cd>().swap(t1_);
    std::vector<cd>().swap(img_);
    std::vector<cd>().swap(m0c_);
    std::vector<cd>().swap(proj_);
    std::vector<cd>().swap(mom_);
    std::vector<cd>().swap(q_);
    std::vector<int64_t>().swap(q_row_);
    std::vector<int64_t>().swap(need_dist_);
    std::vector<double>().swap(dpt_);
    std::vector<double>().swap(dtan_);
    std::vector<double>().swap(obs_);
    std::vector<double>().swap(bo_);
    std::vector<int64_t>().swap(need_);
    std::vector<int64_t>().swap(gi_);
    std::vector<int64_t>().swap(obs_row_);
    std::vector<int64_t>().swap(plus_);
    std::vector<int64_t>().swap(minus_);
    std::vector<int64_t>().swap(mark_);
    std::vector<int>().swap(ipiv_);
}

void SolveContext::set_z(const cd *z, ptrdiff_t n) {
    n_ = n;
    z_.assign(z, z + static_cast<size_t>(n * n));
}

void SolveContext::add_at(const int64_t *rows, const int64_t *cols, const cd *vals, size_t n) {
    cd *z = z_.data();
    for (size_t i = 0; i < n; i++) {
        cd &e = z[static_cast<size_t>(rows[i]) + static_cast<size_t>(cols[i]) * static_cast<size_t>(n_)];
        e = cd(e.real() + vals[i].real(), e.imag() + vals[i].imag());
    }
}

int SolveContext::lu_solve(const Lapack &lp, cd *b, int n_rhs) {
    STUDY_T("lu_solve");
    if (lp.getrf == nullptr || lp.getrs == nullptr)
        throw std::invalid_argument("razor_solve: no LAPACK routines were supplied");
    const auto t0 = std::chrono::steady_clock::now();
    int n = static_cast<int>(n_);
    cd *z = z_.data();
    // `lu_factor(check_finite=True)`'s refusal, in its words.
    const size_t nn = static_cast<size_t>(n_) * static_cast<size_t>(n_);
    for (size_t i = 0; i < nn; i++)
        if (!std::isfinite(z[i].real()) || !std::isfinite(z[i].imag()))
            throw std::invalid_argument("array must not contain infs or NaNs");
    if (ipiv_.size() < static_cast<size_t>(n)) ipiv_.resize(static_cast<size_t>(n));
    int lda = n, info = 0;
    lp.getrf(&n, &n, z, &lda, ipiv_.data(), &info);
    if (info < 0)
        throw std::invalid_argument("illegal value in " + std::to_string(-info) +
                                    "th argument of internal getrf (lu_factor)");
    // `_bspline._lu_solve`'s singular check: any exactly-zero U(i, i).
    for (int i = 0; i < n; i++)
        if (z[static_cast<size_t>(i) * n + i] == cd(0.0, 0.0)) return i + 1;
    char trans = 'N';
    int nrhs = n_rhs, ldb = n;
    lp.getrs(&trans, &n, &nrhs, z, &lda, ipiv_.data(), b, &ldb, &info);
    if (info != 0)
        throw std::invalid_argument("illegal value in " + std::to_string(-info) +
                                    "th argument of internal gesv|posv");
    counters.lu_calls++;
    counters.lu_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

}  // namespace razor
}  // namespace mw
