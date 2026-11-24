#include "_accel_common.h"

#include "_sin_pair_reuse_core.h"

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// momwire#1916 part B: `pair_reuse`'s gather fused into the Φ → Z reduction
// (`_sin_pair_reuse.reduce_band`), so a band's per-pair Φ is never
// materialised: each observer row's three Φ rows are gathered by class index
// into a per-thread buffer, the near pairs written over them, and the row
// reduced from there — necpp's shape for its fill-pair reuse.
//
// What it replaces: `_sin_pair_reuse.scatter` (three `np.take`s into three
// band-sized Φ, then the near pairs) followed by `_acc.reduce_phi_band`; on
// the ELT whip at 8 threads (Rocketlake) 0.119 s + 0.078 s per frequency,
// the gather being memory traffic through three Φ the size of the band.
//
// THE BITS ARE THAT ROUTE'S, by construction. The gather is a copy:
// `np.take(rep[s], cls, mode="clip")` is the clamped class's value, and the
// near pairs then overwrite theirs, as `scatter` writes them after the take.
// Under the PEC image the image block is gathered the same way and
// subtracted, `Φ -= Φ_img`, one IEEE subtraction per part as numpy's complex
// `-=`. The reduction is `reduce_phi_band`'s (`_accel_phi_reduce.cpp`)
// written out again, in `_sin_pair_reuse_core.cpp`, where the C++ fill's
// reuse route runs the same copy: per output element a zero plus its column's stored
// entries of M in stored order, one unfused multiply-add each, then
// `(A + B) + C`. Rows are independent, so the thread count moves nothing;
// LANES rows share each step of the walk over M, each its own chain.
// `-ffp-contract=off` (setup.py) and the MSVC pragmas below keep every
// product and sum its own rounding. Gated bit for bit against scatter +
// reduce_phi_band (tests/test_sin_pair_reuse_1916.py).
//
// momwire#1969: a CODED class map (`signs` given, for the direct block as a
// keyword, for the image as the tuple's fifth element) holds class << 2 |
// code per pair, the code the pair's tangent reversals against its class's
// canonical orientation. The gather takes class `entry >> 2` (clamped as
// before) and negates each shape whose `signs[code][shape]` is negative,
// an exact negation as `scatter`'s np.negative, so the bits stay that
// route's. Without `signs`, the map is read as before.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace sin_pair_reuse {

using cd = std::complex<double>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<int32_t, py::array::c_style>;
using C128 = py::array_t<cd, py::array::c_style>;

using Csc = mw::sinusoidal::pair_reuse::Csc;
using Block = mw::sinusoidal::pair_reuse::Block;

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

// A coded block's sign table, (4, 3) float64, read as negations.
static void set_signs(Block &b, const py::object &signs, const char *what) {
    if (signs.is_none()) return;
    F64 t = signs.cast<F64>();
    if (t.ndim() != 2 || t.shape(0) != 4 || t.shape(1) != 3)
        throw std::runtime_error(std::string("reduce_band: ") + what +
                                 " signs must be (4, 3) float64");
    const double *v = t.data();
    for (int c = 0; c < 4; ++c)
        for (int s = 0; s < 3; ++s) b.neg[c][s] = v[3 * c + s] < 0.0;
    b.coded = true;
}

static Block block_of(const I32 &cls, const C128 &rep, const I64 &near_idx,
                      const C128 &near_val, py::ssize_t rows, py::ssize_t n_src,
                      const char *what) {
    if (cls.ndim() != 2 || cls.shape(0) != rows || cls.shape(1) != n_src)
        throw std::runtime_error(std::string("reduce_band: ") + what +
                                 " classes must be int32 (rows of out, n_src)");
    if (rep.ndim() != 2 || rep.shape(1) != 3 || rep.shape(0) < 1)
        throw std::runtime_error(std::string("reduce_band: ") + what +
                                 " class values must be (n_cls >= 1, 3) complex128");
    if (near_idx.ndim() != 1 || near_val.ndim() != 2 || near_val.shape(1) != 3 ||
        near_val.shape(0) != near_idx.shape(0))
        throw std::runtime_error(std::string("reduce_band: ") + what +
                                 " near pairs must be (k,) int64 and (k, 3) complex128");
    Block b;
    b.cls = cls.data();
    b.rep = reinterpret_cast<const double *>(rep.data());
    b.n_cls = rep.shape(0);
    b.near_idx = near_idx.data();
    b.near_val = reinterpret_cast<const double *>(near_val.data());
    b.n_near = near_idx.shape(0);
    const int64_t total = static_cast<int64_t>(rows) * static_cast<int64_t>(n_src);
    for (int64_t e = 0; e < b.n_near; ++e) {
        if (b.near_idx[e] < 0 || b.near_idx[e] >= total)
            throw std::runtime_error("reduce_band: a near pair lies outside the band");
        if (e && b.near_idx[e] <= b.near_idx[e - 1])
            throw std::runtime_error("reduce_band: near pairs must be sorted and distinct");
    }
    return b;
}

static void reduce_band(py::array out, I32 cls, C128 rep, I64 near_idx, C128 near_val,
                        py::object image, I64 ptr0, I64 ind0, py::array val0, I64 ptr1,
                        I64 ind1, py::array val1, I64 ptr2, I64 ind2, py::array val2,
                        int n_threads, py::object signs) {
    const auto cplx = py::dtype::of<cd>();
    if (out.ndim() != 2 || !out.dtype().is(cplx))
        throw std::runtime_error("reduce_band: out must be a 2-D complex128 array");
    if (!out.writeable()) throw std::runtime_error("reduce_band: out is read-only");
    const py::ssize_t rows = out.shape(0), cols = out.shape(1);
    const py::ssize_t cs = static_cast<py::ssize_t>(sizeof(cd));
    if (cols > 1 && out.strides(1) != cs)
        throw std::runtime_error("reduce_band: out rows must be contiguous");
    const py::ssize_t n_src = cls.ndim() == 2 ? cls.shape(1) : -1;
    Block d = block_of(cls, rep, near_idx, near_val, rows, n_src, "the direct");
    set_signs(d, signs, "the direct");
    Block im;
    bool has_image = !image.is_none();
    // Keep the image arrays alive (and their conversions owned) to the end.
    I32 icls;
    C128 irep, inval;
    I64 inidx;
    if (has_image) {
        auto t = image.cast<py::tuple>();
        if (t.size() != 4 && t.size() != 5)
            throw std::runtime_error(
                "reduce_band: image must be (classes, values, near index, near values"
                "[, signs])");
        icls = t[0].cast<I32>();
        irep = t[1].cast<C128>();
        inidx = t[2].cast<I64>();
        inval = t[3].cast<C128>();
        im = block_of(icls, irep, inidx, inval, rows, n_src, "the image");
        if (t.size() == 5) set_signs(im, t[4], "the image");
    }
    I64 ptr[3] = {ptr0, ptr1, ptr2}, ind[3] = {ind0, ind1, ind2};
    py::array val[3] = {val0, val1, val2};
    Csc M[3];
    for (int c = 0; c < 3; c++) {
        const py::array &v = val[c];
        if (v.ndim() != 1 || !v.dtype().is(cplx) || (v.size() > 1 && v.strides(0) != cs))
            throw std::runtime_error("reduce_band: M data must be contiguous complex128");
        if (ptr[c].ndim() != 1 || ptr[c].size() != cols + 1 || ind[c].ndim() != 1 ||
            ind[c].size() != v.size())
            throw std::runtime_error(
                "reduce_band: M must be CSC with out's columns (indptr cols + 1, "
                "indices as long as data)");
        const int64_t *p = ptr[c].data(), *ix = ind[c].data();
        if (p[0] < 0 || p[cols] > static_cast<int64_t>(v.size()))
            throw std::runtime_error("reduce_band: indptr out of range");
        for (py::ssize_t j = 0; j < cols; j++)
            if (p[j + 1] < p[j]) throw std::runtime_error("reduce_band: indptr not monotone");
        for (py::ssize_t e = 0; e < ind[c].size(); e++)
            if (ix[e] < 0 || ix[e] >= n_src)
                throw std::runtime_error("reduce_band: index out of range");
        M[c] = Csc{p, ix, static_cast<const double *>(v.data())};
    }
    char *O = static_cast<char *>(out.mutable_data());
    const py::ssize_t os = out.strides(0);
    {
        py::gil_scoped_release nogil;
        mw::sinusoidal::pair_reuse::reduce_band(O, os, rows, cols, n_src, d, has_image ? &im : nullptr, M,
                                                n_threads);
    }
}


// momwire#1911 / #2063: the plans both routes run (`_sin_pair_reuse_core.h`),
// for the Python route (`_sin_pair_reuse.py`) and the tests. A plan's arrays
// are read in place: numpy views over the plan, which they keep alive.
namespace pr = mw::sinusoidal::pair_reuse;
namespace core = mw::pair_reuse;

using Ptr = std::shared_ptr<const pr::Plan>;

static py::array_t<int64_t> i64_of(const std::vector<int64_t> &v) {
    py::array_t<int64_t> a(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

static py::list blocks_of(const std::vector<pr::NearBlock> &bs) {
    py::list out;
    for (const pr::NearBlock &b : bs)
        out.append(py::make_tuple(b.m0, b.m1, i64_of(b.ns), i64_of(b.r), i64_of(b.c), b.p0, b.p1));
    return out;
}

// A read-only view of `n` (or rows x cols) elements at `data`, owned by `p`.
template <class T>
static py::array view(const Ptr &p, const T *data, std::vector<py::ssize_t> shape, py::dtype dt) {
    py::capsule owner(new Ptr(p), [](void *q) { delete static_cast<Ptr *>(q); });
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t st = static_cast<py::ssize_t>(sizeof(T));
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = st;
        st *= shape[i];
    }
    py::array a(dt, shape, strides, data, owner);
    a.attr("flags").attr("writeable") = false;
    return a;
}

struct PlanHandle {
    Ptr plan;
    bool built = false;
};

struct PlanRef {
    std::weak_ptr<const pr::Plan> plan;
};

// The fill report's wire map (momwire#2016) of one plan: (W, W, 3) int64,
// per observer wire and source wire the pairs computed, reused and near
// (`pair_reuse::wire_map`), or None once the plan is gone.
static py::object wire_map(const PlanRef &r, py::array_t<int32_t, py::array::c_style | py::array::forcecast> row,
                           py::array_t<int32_t, py::array::c_style | py::array::forcecast> col, int64_t n_wires) {
    const Ptr p = r.plan.lock();
    if (p == nullptr) return py::none();
    if (row.size() != p->M || col.size() != p->N || n_wires < 1)
        throw std::invalid_argument("fill_report: one wire per row and per source");
    for (py::ssize_t i = 0; i < row.size(); i++)
        if (row.data()[i] < 0 || row.data()[i] >= n_wires) throw std::invalid_argument("fill_report: wire out of range");
    for (py::ssize_t i = 0; i < col.size(); i++)
        if (col.data()[i] < 0 || col.data()[i] >= n_wires) throw std::invalid_argument("fill_report: wire out of range");
    const py::ssize_t W = static_cast<py::ssize_t>(n_wires);
    py::array_t<int64_t> out({W, W, py::ssize_t(3)});
    int64_t *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        std::fill(o, o + W * W * 3, int64_t(0));
        core::wire_map(p->fam, row.data(), col.data(), n_wires, o);
    }
    return out;
}

static py::tuple sin_pair_reuse_plan(F64 obs_c, F64 obs_t, F64 src_c, F64 src_t, F64 seg_h, F64 a_obs,
                                     I64 src_type, I64 obs_type, int64_t rows, py::object src_rev,
                                     py::object allowance, int workers, py::object ground_z, py::object obs_rev,
                                     int64_t band_bytes, bool shared, int keep) {
    const py::ssize_t M = obs_c.shape(0), N = src_c.shape(0);
    if (obs_c.ndim() != 2 || obs_c.shape(1) != 3 || obs_t.ndim() != 2 || obs_t.shape(0) != M ||
        src_c.ndim() != 2 || src_c.shape(1) != 3 || src_t.ndim() != 2 || src_t.shape(0) != N ||
        seg_h.size() != N || a_obs.size() != M || src_type.size() != N || obs_type.size() != M)
        throw std::runtime_error("sin_pair_reuse_plan: (M, 3) observers, (N, 3) sources, per-segment columns");
    pr::CensusInputs in;
    in.M = M;
    in.N = N;
    in.obs_c = obs_c.data();
    in.obs_t = obs_t.data();
    in.src_c = src_c.data();
    in.src_t = src_t.data();
    in.seg_h = seg_h.data();
    in.a_obs = a_obs.data();
    in.src_type = src_type.data();
    in.obs_type = obs_type.data();
    I64 srev, orev;
    if (!src_rev.is_none()) {
        srev = src_rev.cast<I64>();
        if (srev.size() != N) throw std::runtime_error("sin_pair_reuse_plan: src_rev per source");
        in.src_rev = srev.data();
    }
    if (!obs_rev.is_none()) {
        orev = obs_rev.cast<I64>();
        if (orev.size() != M) throw std::runtime_error("sin_pair_reuse_plan: obs_rev per observer");
        in.obs_rev = orev.data();
    }
    in.rows = rows;
    in.allowance = allowance.is_none() ? -1 : allowance.cast<int64_t>();
    in.workers = workers > 0 ? workers : pr::workers();
    in.band_bytes_per_pair = band_bytes;
    const bool ground = !ground_z.is_none();
    if (ground) in.ground_z = ground_z.cast<double>();
    Ptr p;
    bool built = true;
    std::string why;
    {
        py::gil_scoped_release nogil;
        if (shared)
            p = pr::shared_plan(in, ground, &built, &why, keep);
        else
            p = ground ? pr::build_ground_plan(in, &why) : pr::build_plan(in, &why);
    }
    if (p == nullptr) return py::make_tuple(py::none(), built, why);
    return py::make_tuple(PlanHandle{p, built}, built, py::str(""));
}

static py::tuple sin_pair_reuse_types(F64 seg_h, F64 a_obs, py::object src_a, py::object ind1, py::object ind2,
                                      py::object seg_t) {
    using I8 = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;
    const py::ssize_t n = seg_h.size();
    F64 sa;
    I8 i1, i2;
    if (!src_a.is_none()) {
        sa = src_a.cast<F64>();
        i1 = ind1.cast<I8>();
        i2 = ind2.cast<I8>();
    }
    const pr::Types t = pr::reversible_segment_types(n, seg_h.data(), a_obs.data(),
                                                     src_a.is_none() ? nullptr : sa.data(),
                                                     src_a.is_none() ? nullptr : i1.data(),
                                                     src_a.is_none() ? nullptr : i2.data());
    if (seg_t.is_none()) return py::make_tuple(i64_of(t.src), i64_of(t.obs), i64_of(t.src_rev));
    F64 st = seg_t.cast<F64>();
    const pr::GroundTypes g = pr::reversible_ground_types(t, n, st.data());
    return py::make_tuple(i64_of(g.src), i64_of(g.obs), i64_of(g.src_rev), i64_of(g.obs_rev));
}

static py::tuple sin_pair_reuse_ground_types(I64 src_type, I64 obs_type, I64 src_rev, F64 seg_t) {
    const py::ssize_t n = src_type.size();
    if (obs_type.size() != n || src_rev.size() != n || seg_t.size() != 3 * n)
        throw std::runtime_error("sin_pair_reuse_ground_types: per-segment types and (n, 3) tangents");
    pr::Types t;
    t.src.assign(src_type.data(), src_type.data() + n);
    t.obs.assign(obs_type.data(), obs_type.data() + n);
    t.src_rev.assign(src_rev.data(), src_rev.data() + n);
    const pr::GroundTypes g = pr::reversible_ground_types(t, n, seg_t.data());
    return py::make_tuple(i64_of(g.src), i64_of(g.obs), i64_of(g.src_rev), i64_of(g.obs_rev));
}

static py::tuple sin_pair_reuse_keys(F64 obs_c, F64 obs_t, F64 src_c, F64 src_t, F64 seg_h, F64 a_obs,
                                     I64 src_type, I64 obs_type, py::object ground_z, py::object src_rev,
                                     py::object obs_rev) {
    pr::CensusInputs in;
    in.M = obs_c.shape(0);
    in.N = src_c.shape(0);
    if (obs_t.shape(0) != in.M || src_t.shape(0) != in.N || seg_h.size() != in.N || a_obs.size() != in.M ||
        src_type.size() != in.N || obs_type.size() != in.M)
        throw std::runtime_error("sin_pair_reuse_keys: (M, 3) observers, (N, 3) sources, per-segment columns");
    in.obs_c = obs_c.data();
    in.obs_t = obs_t.data();
    in.src_c = src_c.data();
    in.src_t = src_t.data();
    in.seg_h = seg_h.data();
    in.a_obs = a_obs.data();
    in.src_type = src_type.data();
    in.obs_type = obs_type.data();
    I64 srev, orev;
    if (!src_rev.is_none()) {
        srev = src_rev.cast<I64>();
        in.src_rev = srev.data();
    }
    if (!obs_rev.is_none()) {
        orev = obs_rev.cast<I64>();
        in.obs_rev = orev.data();
    }
    const bool ground = !ground_z.is_none();
    if (ground) in.ground_z = ground_z.cast<double>();
    std::vector<int64_t> words;
    std::vector<int8_t> codes;
    std::vector<uint8_t> near;
    pr::pair_keys(in, ground, words, codes, near);
    const py::ssize_t P = static_cast<py::ssize_t>(codes.size()), W = ground ? 6 : 5;
    py::array_t<int64_t> w({P, W});
    std::copy(words.begin(), words.end(), w.mutable_data());
    py::array_t<int8_t> c(P);
    std::copy(codes.begin(), codes.end(), c.mutable_data());
    py::array_t<bool> nr(P);
    for (py::ssize_t i = 0; i < P; i++) nr.mutable_data()[i] = near[static_cast<size_t>(i)] != 0;
    return py::make_tuple(w, c, nr);
}

static py::tuple sin_pair_reuse_canonical(py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows,
                                          int ia, int ib, int it, py::object near, int64_t n_src, py::object ds,
                                          py::object dob) {
    if (rows.ndim() != 2 || rows.shape(1) < 1 || rows.shape(1) > 8)
        throw std::runtime_error("sin_pair_reuse_canonical: (n, width) int64 key rows");
    const py::ssize_t n = rows.shape(0), W = rows.shape(1);
    for (int c : {ia, ib, it})
        if (c < 1 || c >= W) throw std::runtime_error("sin_pair_reuse_canonical: column out of range");
    py::array_t<int64_t> out({n, W});
    std::copy(rows.data(), rows.data() + n * W, out.mutable_data());
    using U8 = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
    U8 nm;
    I64 d1, d2;
    if (!near.is_none()) nm = near.cast<U8>();
    if (!ds.is_none()) d1 = ds.cast<I64>();
    if (!dob.is_none()) d2 = dob.cast<I64>();
    if ((!near.is_none() && nm.size() != n) || (!ds.is_none() && d1.size() != n_src) ||
        (!dob.is_none() && d2.size() * n_src != n) || n_src < 1 || n % n_src != 0)
        throw std::runtime_error("sin_pair_reuse_canonical: near per row, ds per source, do per observer");
    py::array_t<int8_t> codes(n);
    pr::canonical_codes(out.mutable_data(), static_cast<int>(W), n, ia, ib, it, near.is_none() ? nullptr : nm.data(),
                        n_src, ds.is_none() ? nullptr : d1.data(), dob.is_none() ? nullptr : d2.data(),
                        codes.mutable_data());
    return py::make_tuple(out, codes);
}

// TEST HOOK: the band budget alone over `n_bands` bands, the census holding
// `base` + `per` bytes per band folded and its floor `floor_per` per band.
static py::dict band_budget(int64_t n_bands, int64_t budget, int64_t band_bytes, int workers, int64_t base,
                            int64_t per, int64_t floor_per) {
    core::BandBudget bb(budget, band_bytes, workers, n_bands, base);
    int64_t started = 0, folded = 0;
    bool ok = true;
    for (int64_t i = 0; i < n_bands && ok; i++) {
        if (!bb.start(i)) {
            ok = false;
            break;
        }
        started++;
        folded++;
        ok = bb.fold(base + per * folded, floor_per * folded);
    }
    py::dict d;
    d["peak"] = ok ? py::object(py::int_(bb.peak())) : py::none();
    d["started"] = started;
    d["folded"] = folded;
    d["max_in_flight"] = bb.max_in_flight();
    return d;
}

static py::dict sin_pair_reuse_stats(bool reset) {
    if (reset) pr::reset_counters();
    const core::Counters c = pr::counters();
    py::dict d;
    d["built"] = c.built;
    d["declined"] = c.declined;
    d["found"] = c.found;
    d["last"] = pr::last_outcome();
    py::dict census;
    for (const pr::CensusEntry &e : pr::last_census()) census[py::str(e.block)] = py::make_tuple(e.pairs, e.classes, e.near);
    d["census"] = census;
    d["allowance"] = pr::plan_allowance_override();
    d["workers"] = pr::workers();
    return d;
}

static void sin_pair_reuse_configure(py::object allowance, py::object small_free, py::object small_ground,
                                     py::object workers, bool clear, py::object probe, py::object cache_bytes) {
    if (!allowance.is_none()) pr::set_plan_allowance(allowance.cast<int64_t>());
    if (!small_free.is_none() || !small_ground.is_none())
        pr::set_small_deck_segs(small_free.is_none() ? pr::small_deck_segs(false) : small_free.cast<int64_t>(),
                                small_ground.is_none() ? pr::small_deck_segs(true) : small_ground.cast<int64_t>());
    if (!workers.is_none()) pr::set_workers(workers.cast<int>());
    if (!probe.is_none()) pr::set_probe(probe.cast<int>());
    if (!cache_bytes.is_none()) core::set_cache_bytes(cache_bytes.cast<int64_t>());
    if (clear) pr::clear_shared_plans();
}

}  // namespace sin_pair_reuse

void register_sin_pair_reuse(py::module_ &m) {
    m.def("pair_reuse_reduce_band", &sin_pair_reuse::reduce_band,
          "pair_reuse's gather fused into the Φ → Z reduction, to the bit of "
          "`_sin_pair_reuse.scatter` then `reduce_phi_band`: per observer row, "
          "Φ gathered from the class values (n_cls, 3) by the int32 class map "
          "(clamped, as np.take mode='clip'), the band's near pairs (sorted "
          "band-local flat index, (k, 3) values) written over it, the image "
          "block's likewise subtracted when `image` is (classes, values, near "
          "index, near values[, signs]), then reduced against the three CSC Ms. "
          "OpenMP over out's rows; n_threads > 0 sets the team. momwire#1916. "
          "`signs` (4, 3) float64 marks the direct block's map coded, class << 2 "
          "| code, each shape negated where its code's sign is negative "
          "(momwire#1969).",
          py::arg("out"), py::arg("cls"), py::arg("rep"), py::arg("near_idx"),
          py::arg("near_val"), py::arg("image"), py::arg("indptr0"), py::arg("indices0"),
          py::arg("data0"), py::arg("indptr1"), py::arg("indices1"), py::arg("data1"),
          py::arg("indptr2"), py::arg("indices2"), py::arg("data2"),
          py::arg("n_threads") = 0, py::arg("signs") = py::none());
    using sin_pair_reuse::PlanHandle;
    using sin_pair_reuse::PlanRef;
    using sin_pair_reuse::Ptr;
    using sin_pair_reuse::view;
    namespace pr = mw::sinusoidal::pair_reuse;
    py::class_<PlanRef>(m, "SinReusePlanRef", "A sinusoidal reuse plan held weakly (the fill report's).")
        .def_property_readonly("alive", [](const PlanRef &r) { return !r.plan.expired(); },
                               "Whether the process's cache (or a fill) still holds the plan.")
        .def("wire_map", &sin_pair_reuse::wire_map, py::arg("wire_of_row"), py::arg("wire_of_col"),
             py::arg("n_wires"),
             "The (W, W, 3) int64 map of observer wire x source wire x (computed, reused, near), or None "
             "when the plan is gone (`pair_reuse::wire_map`, momwire#2016).");
    py::class_<PlanHandle>(m, "SinReusePlan",
                           "A sinusoidal fill-pair reuse plan (momwire#1911, #2063): its arrays as read-only "
                           "views that keep it alive.")
        .def_readonly("built", &PlanHandle::built, "Whether this call's census built the plan.")
        .def_property_readonly("ground", [](const PlanHandle &h) { return h.plan->ground; })
        .def_property_readonly("cls", [](const PlanHandle &h) {
            const Ptr &p = h.plan;
            return view(p, p->cls().data(), {static_cast<py::ssize_t>(p->M), static_cast<py::ssize_t>(p->N)},
                        py::dtype::of<int32_t>());
        })
        .def_property_readonly("rep", [](const PlanHandle &h) {
            const Ptr &p = h.plan;
            return view(p, p->rep.data(), {static_cast<py::ssize_t>(p->rep.size())}, py::dtype::of<int64_t>());
        })
        .def_property_readonly("rep_code", [](const PlanHandle &h) -> py::object {
            const Ptr &p = h.plan;
            if (!p->coded) return py::none();
            return view(p, p->fam.rep_code.data(), {static_cast<py::ssize_t>(p->fam.rep_code.size())},
                        py::dtype::of<int8_t>());
        })
        .def_property_readonly("code_bits", [](const PlanHandle &h) { return h.plan->coded ? pr::CODE_BITS : 0; })
        .def_property_readonly("near_flat", [](const PlanHandle &h) {
            const Ptr &p = h.plan;
            return view(p, p->near_flat().data(), {static_cast<py::ssize_t>(p->near_flat().size())},
                        py::dtype::of<int64_t>());
        })
        .def_property_readonly("near_blocks", [](const PlanHandle &h) { return sin_pair_reuse::blocks_of(h.plan->near_blocks); })
        .def_property_readonly("rep_blocks", [](const PlanHandle &h) { return sin_pair_reuse::blocks_of(h.plan->rep_blocks); })
        .def_property_readonly("src_c", [](const PlanHandle &h) {
            const Ptr &p = h.plan;
            return view(p, p->src_c.data(), {static_cast<py::ssize_t>(p->src_c.size() / 3), py::ssize_t(3)},
                        py::dtype::of<double>());
        })
        .def_property_readonly("src_t", [](const PlanHandle &h) {
            const Ptr &p = h.plan;
            return view(p, p->src_t.data(), {static_cast<py::ssize_t>(p->src_t.size() / 3), py::ssize_t(3)},
                        py::dtype::of<double>());
        })
        .def_property_readonly("runs", [](const PlanHandle &h) {
            py::list runs;
            for (const auto &r : h.plan->runs) runs.append(py::make_tuple(r.c0, r.c1, r.g_obs));
            return runs;
        })
        .def_property_readonly("census_peak_bytes", [](const PlanHandle &h) { return h.plan->census_peak_bytes; })
        .def_property_readonly("plan_nbytes", [](const PlanHandle &h) { return h.plan->plan_nbytes(); })
        .def_property_readonly("fill_nbytes", [](const PlanHandle &h) { return h.plan->fill_nbytes(); })
        .def("ref", [](const PlanHandle &h) { return PlanRef{h.plan}; }, "The plan, held weakly.");
    m.def("sin_pair_reuse_plan", &sin_pair_reuse::sin_pair_reuse_plan, py::arg("obs_c"), py::arg("obs_t"),
          py::arg("src_c"), py::arg("src_t"), py::arg("seg_h"), py::arg("a_obs"), py::arg("src_type"),
          py::arg("obs_type"), py::arg("rows"), py::arg("src_rev") = py::none(), py::arg("allowance") = py::none(),
          py::arg("workers") = 0, py::arg("ground_z") = py::none(), py::arg("obs_rev") = py::none(),
          py::arg("band_bytes") = -1, py::arg("shared") = false, py::arg("keep") = 1,
          "momwire#1911 / #2063 / #2123: the sinusoidal census and plan (with ground_z, the reflected "
          "term's): (SinReusePlan or None where it declines, whether a census ran, the decline's reason or "
          "''). `allowance` is the kept plan's bytes (None: 4 B per pair, `sin_pair_reuse_kept_bytes` per "
          "class the 4/3x cap allows and per near pair). `shared` reads and keeps the process's cache (`keep`: its newest entries kept whatever "
          "their size, 2 for a fill's second block); workers > 0 sets the census's worker count, "
          "band_bytes >= 0 its band bytes per pair.");
    m.def("sin_pair_reuse_types", &sin_pair_reuse::sin_pair_reuse_types, py::arg("seg_h"), py::arg("a_obs"),
          py::arg("src_a") = py::none(), py::arg("ind1") = py::none(), py::arg("ind2") = py::none(),
          py::arg("seg_t") = py::none(),
          "momwire#1911: `reversible_segment_types`; with seg_t, `reversible_ground_types` of those.");
    m.def("sin_pair_reuse_ground_types", &sin_pair_reuse::sin_pair_reuse_ground_types, py::arg("src_type"),
          py::arg("obs_type"), py::arg("src_rev"), py::arg("seg_t"),
          "momwire#1911: `reversible_ground_types` of given types.");
    m.def("sin_pair_reuse_keys", &sin_pair_reuse::sin_pair_reuse_keys, py::arg("obs_c"), py::arg("obs_t"),
          py::arg("src_c"), py::arg("src_t"), py::arg("seg_h"), py::arg("a_obs"), py::arg("src_type"),
          py::arg("obs_type"), py::arg("ground_z") = py::none(), py::arg("src_rev") = py::none(),
          py::arg("obs_rev") = py::none(),
          "TESTS: every pair's key words (M*N, 5 or 6), code and near flag, as the census keys them.");
    m.def("sin_pair_reuse_canonical", &sin_pair_reuse::sin_pair_reuse_canonical, py::arg("rows"), py::arg("ia"),
          py::arg("ib"), py::arg("it"), py::arg("near") = py::none(), py::arg("n_src") = 1,
          py::arg("ds") = py::none(), py::arg("do") = py::none(),
          "TESTS: the canonical orientation of key rows (n, width): (the keyed rows, their codes).");
    m.def("pair_reuse_band_budget", &sin_pair_reuse::band_budget, py::arg("n_bands"), py::arg("budget"),
          py::arg("band_bytes"), py::arg("workers"), py::arg("base") = 0, py::arg("per") = 0,
          py::arg("floor_per") = 0,
          "TESTS: `pair_reuse::BandBudget` over a census holding base + per bytes a band folded: "
          "peak (None: stopped), started, folded, max_in_flight.");
    m.def("sin_pair_reuse_stats", &sin_pair_reuse::sin_pair_reuse_stats, py::arg("reset") = false,
          "momwire#1911: the reuse counters (plans built, of those declined, found; both routes), the C++ "
          "fill's last outcome ('', 'served' or the decline reason) and census per block; reset first.");
    m.def("sin_pair_reuse_configure", &sin_pair_reuse::sin_pair_reuse_configure, py::arg("allowance") = py::none(),
          py::arg("small_free") = py::none(), py::arg("small_ground") = py::none(), py::arg("workers") = py::none(),
          py::arg("clear") = false, py::arg("probe") = py::none(), py::arg("cache_bytes") = py::none(),
          "TEST HOOK (momwire#1911): the C++ fill's kept-plan allowance per plan (<= 0 restores the derived "
          "one, momwire#2123), small-deck thresholds "
          "(< 0 restores each), census worker count (<= 0: the OpenMP team's), the adversarial probe (0 none), "
          "the plan cache's bytes (< 0 restores it, 0 keeps none); clear drops the kept plans.");
    // momwire#2123: what a plan keeps per class (direct, reflected) and per
    // near pair, the allowance's per-engine costs, from the plan's layout.
    m.attr("sin_pair_reuse_kept_bytes") =
        py::make_tuple(pr::KEPT_BYTES_PER_CLASS, pr::GROUND_KEPT_BYTES_PER_CLASS, pr::NEAR_KEPT_BYTES);
    m.attr("sin_pair_reuse_cxx") = true;
    m.attr("sin_pair_reuse_2063") = true;
    m.attr("pair_reuse_reduce_1916") = true;
    m.attr("pair_reuse_reduce_1969") = true;
}
