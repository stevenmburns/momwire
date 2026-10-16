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
#include "_razor_solve_core.h"

#include "_razor_core.h"

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
    seg_moments(c_view<const double, 2>(obs_.data(), {n_need, 3}), src.seg_p0, src.seg_t,
                in.seg_h, in.a, in.xg, in.wg, in.k, true, gi, src.group_j, src.a_ek,
                in.cancel_flag, bo, vm0, vm1);
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
    assemble_t1(view<const cd, 2>(vm0), view<const cd, 2>(vm1), in.s_a, in.s_b, in.h_a,
                in.h_b, in.fall_a, in.fall_b, row_slice(in.tans, lo * np, hi * np),
                src.td_a, src.td_b, slice1(in.wts, lo * np, hi * np), in.n_path,
                c_view<const int64_t, 1>(obs_row_.data(), {nb * np}), vt1);
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
    ptrdiff_t k_shape[2] = {0, n_seg}, k_st[2] = {static_cast<ptrdiff_t>(sizeof(cd) * n_seg),
                                                  static_cast<ptrdiff_t>(sizeof(cd))};
    view<const cd, 2> m0k(nullptr, k_shape, k_st);
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
    t2_rows(dst, &t1c, view<const cd, 2>(vm0), m0k,
            c_view<const int64_t, 1>(plus_.data(), {nb}),
            c_view<const int64_t, 1>(minus_.data(), {nb}), in.s_a, in.s_b, in.q_a, in.q_b,
            in.c_a_imag, in.c_phi_imag);
    counters.t2_rows_calls++;
    if (image) {
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

void SolveContext::fill(const SolveInputs &in) { fill(in, nullptr); }

void SolveContext::fill(const SolveInputs &in, const std::function<void(cd *)> &between) {
    const auto t0 = std::chrono::steady_clock::now();
    const ptrdiff_t n = in.s_a.shape(0);
    const char *who = "razor_solve";
    if (in.n_path != 2)
        throw std::invalid_argument("razor_solve: only the NEC-5 path rule (n_path = 2)");
    if (in.tans.shape(0) != n * 2 || in.wts.shape(0) != n * 2)
        throw std::invalid_argument("razor_solve: tans / wts need n_path rows per basis");
    if (in.window_rows < 1)
        throw std::invalid_argument("razor_solve: window_rows must be >= 1");
    if (!(in.c_phi_imag != 0.0))
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

void SolveContext::add_at(const int64_t *rows, const int64_t *cols, const cd *vals, size_t n) {
    cd *z = z_.data();
    for (size_t i = 0; i < n; i++) {
        cd &e = z[static_cast<size_t>(rows[i]) + static_cast<size_t>(cols[i]) * static_cast<size_t>(n_)];
        e = cd(e.real() + vals[i].real(), e.imag() + vals[i].imag());
    }
}

int SolveContext::lu_solve(const Lapack &lp, cd *b, int n_rhs) {
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
