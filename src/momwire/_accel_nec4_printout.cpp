// The NEC-4.2 printout writer's binding (`_nec4_printout_core.cpp` is the
// writer, with no Python in it; momwire#1607 A3). Test-facing only: the slot
// still prints with `eznec/_nec4_printout.py`, which stays the reference.
// This TU reads a Python `Nec4Printout` (the dataclass `eznec._nec4.solve`
// fills) into the C++ record, field by field, so `tests/test_nec4_printout_core.py`
// can hand the C++ writer exactly the numbers Python's own writer got.
#include "_accel_common.h"
#include "_nec4_printout_core.h"
#include "_sinusoidal_galerkin_run_core.h"
#include "_sinusoidal_run_core.h"

#include <complex>
#include <string>
#include <vector>

namespace {

using namespace mw::nec4;

bool none(const py::handle &h) { return h.is_none(); }
double fl(const py::handle &h) { return h.cast<double>(); }
long long in(const py::handle &h) { return h.cast<long long>(); }
std::complex<double> cx(const py::handle &h) { return h.cast<std::complex<double>>(); }
std::string st(const py::handle &h) { return h.cast<std::string>(); }  // UTF-8
py::object at(const py::handle &h, const char *name) { return h.attr(name); }
std::string type_name(const py::handle &h) { return py::str(py::type::of(h).attr("__name__")); }

void three(const py::handle &h, double out[3]) {
    py::tuple t = py::reinterpret_borrow<py::object>(h).cast<py::tuple>();
    for (int k = 0; k < 3; k++) out[k] = t[static_cast<std::size_t>(k)].cast<double>();
}

PrintLoadRow load_row(const py::handle &h) {
    PrintLoadRow r;
    r.tag = in(at(h, "tag"));
    r.node_from = in(at(h, "node_from"));
    r.node_thru = in(at(h, "node_thru"));
    auto opt = [&](const char *name, bool &has, double &value) {
        py::object v = at(h, name);
        has = !none(v);
        if (has) value = fl(v);
    };
    opt("resistance", r.has_resistance, r.resistance);
    opt("reactance", r.has_reactance, r.reactance);
    opt("inductance", r.has_inductance, r.inductance);
    opt("capacitance", r.has_capacitance, r.capacitance);
    opt("conductivity", r.has_conductivity, r.conductivity);
    r.kind = st(at(h, "kind"));
    return r;
}

PrintPortRow port_row(const py::handle &h) {
    PrintPortRow r;
    r.tag = in(at(h, "tag"));
    r.segment = in(at(h, "segment"));
    r.end_index = in(at(h, "end_index"));
    r.voltage = cx(at(h, "voltage"));
    r.current = cx(at(h, "current"));
    r.impedance = cx(at(h, "impedance"));
    r.admittance = cx(at(h, "admittance"));
    r.power = fl(at(h, "power"));
    return r;
}

ValueRow value_row(const py::handle &h) {
    ValueRow r;
    r.number = in(at(h, "number"));
    r.tag = in(at(h, "tag"));
    three(at(h, "centre"), r.centre);
    r.length = fl(at(h, "length"));
    r.real = fl(at(h, "real"));
    r.imag = fl(at(h, "imag"));
    r.magnitude = fl(at(h, "magnitude"));
    r.phase_deg = fl(at(h, "phase_deg"));
    r.end = at(h, "end").cast<bool>();
    return r;
}

PrintNetworkRow network_row(const py::handle &h) {
    PrintNetworkRow r;
    r.line = type_name(h) == "LineRow";
    r.tag_from = in(at(h, "tag_from"));
    r.segment_from = in(at(h, "segment_from"));
    r.tag_to = in(at(h, "tag_to"));
    r.segment_to = in(at(h, "segment_to"));
    if (r.line) {
        r.z0 = fl(at(h, "z0"));
        r.length_m = fl(at(h, "length_m"));
        r.shunt_a = cx(at(h, "shunt_a"));
        r.shunt_b = cx(at(h, "shunt_b"));
        r.crossed = at(h, "crossed").cast<bool>();
    } else {
        r.y11 = cx(at(h, "y11"));
        r.y12 = cx(at(h, "y12"));
        r.y22 = cx(at(h, "y22"));
    }
    return r;
}

PrintNearBlock near_block(const py::handle &h) {
    PrintNearBlock b;
    b.magnetic = at(h, "magnetic").cast<bool>();
    for (const py::handle &row : at(h, "rows")) {
        PrintNearRow r;
        three(at(row, "point"), r.point);
        three(at(row, "magnitudes"), r.magnitudes);
        three(at(row, "phases_deg"), r.phases_deg);
        b.rows.push_back(r);
    }
    return b;
}

PrintPatternBlock pattern_block(const py::handle &h) {
    PrintPatternBlock b;
    for (const py::handle &row : at(h, "rows")) {
        py::tuple t = py::reinterpret_borrow<py::object>(row).cast<py::tuple>();
        PrintPatternRow r;
        r.theta_deg = t[0].cast<double>();
        r.phi_deg = t[1].cast<double>();
        r.vert_db = t[2].cast<double>();
        r.hor_db = t[3].cast<double>();
        r.total_db = t[4].cast<double>();
        r.axial_ratio = t[5].cast<double>();
        r.tilt_deg = t[6].cast<double>();
        r.sense = t[7].cast<std::string>();
        r.e_theta_magnitude = t[8].cast<double>();
        r.e_theta_phase_deg = t[9].cast<double>();
        r.e_phi_magnitude = t[10].cast<double>();
        r.e_phi_phase_deg = t[11].cast<double>();
        b.rows.push_back(r);
    }
    py::object avg = at(h, "average_power_gain");
    b.has_average = !none(avg);
    if (b.has_average) {
        b.average_power_gain = fl(avg);
        b.solid_angle_pi = fl(at(h, "solid_angle_pi"));
        b.power_radiated_4pi = fl(at(h, "power_radiated_4pi"));
    }
    py::object range = at(h, "range_m");
    b.has_range = !none(range);
    if (b.has_range) {
        b.range_m = fl(range);
        b.range_phase_deg = fl(at(h, "range_phase_deg"));
    }
    py::object lower = at(h, "lower_range_mag");
    b.has_lower = !none(lower);
    if (b.has_lower) {
        b.lower_range_mag = fl(lower);
        b.lower_range_phase_deg = fl(at(h, "lower_range_phase_deg"));
    }
    b.directive = at(h, "directive").cast<bool>();
    return b;
}

void tables(const py::handle &h, PrintRun &r) {
    for (const py::handle &n : at(h, "near_fields")) r.near_fields.push_back(near_block(n));
    py::object g = at(h, "far_field_ground");
    r.has_far_field_ground = !none(g);
    if (r.has_far_field_ground) {
        r.far_field_ground.kind = st(at(g, "kind"));
        r.far_field_ground.edge_distance = fl(at(g, "edge_distance"));
        r.far_field_ground.height = fl(at(g, "height"));
        r.far_field_ground.eps_r2 = fl(at(g, "eps_r2"));
        r.far_field_ground.sigma2 = fl(at(g, "sigma2"));
    }
    for (const py::handle &p : at(h, "patterns")) r.patterns.push_back(pattern_block(p));
}

PrintRun run(const py::handle &h) {
    PrintRun r;
    r.report = type_name(h) == "Nec4Report";
    py::object through = at(h, "echo_through");
    r.has_echo_through = !none(through);
    if (r.has_echo_through) r.echo_through = in(through);
    if (!r.report) {
        py::object pre = at(h, "preamble");
        r.has_preamble = !none(pre);
        if (r.has_preamble) {
            PrintPreamble &p = r.preamble;
            p.frequency_mhz = fl(at(pre, "frequency_mhz"));
            p.wavelength_m = fl(at(pre, "wavelength_m"));
            p.environment = st(at(pre, "environment"));
            py::object g = at(pre, "ground");
            p.has_ground = !none(g);
            if (p.has_ground) {
                p.eps_r = fl(at(g, "eps_r"));
                p.sigma = fl(at(g, "sigma"));
                p.eps_c = cx(at(g, "eps_c"));
            }
            for (const py::handle &l : at(pre, "loads")) p.loads.push_back(load_row(l));
            p.fill_seconds = fl(at(pre, "fill_seconds"));
            p.factor_seconds = fl(at(pre, "factor_seconds"));
            py::object file = at(pre, "ground_file");
            p.has_ground_file = !none(file);
            if (p.has_ground_file) p.ground_file = st(file);
        }
        for (const py::handle &n : at(h, "networks")) r.networks.push_back(network_row(n));
        for (const py::handle &n : at(h, "network_excitation")) r.network_excitation.push_back(port_row(n));
        for (const py::handle &n : at(h, "sources")) r.sources.push_back(port_row(n));
        py::object currents = at(h, "currents");
        r.has_currents = !none(currents);
        if (r.has_currents)
            for (const py::handle &v : currents) r.currents.push_back(value_row(v));
        py::object charges = at(h, "charges");
        r.has_charges = !none(charges);
        if (r.has_charges)
            for (const py::handle &v : charges) r.charges.push_back(value_row(v));
        py::object power = at(h, "power");
        r.power.input_power = fl(at(power, "input_power"));
        r.power.radiated_power = fl(at(power, "radiated_power"));
        r.power.wire_loss = fl(at(power, "wire_loss"));
        r.power.efficiency_percent = fl(at(power, "efficiency_percent"));
        py::object loss = at(power, "network_loss");
        r.power.has_network_loss = !none(loss);
        if (r.power.has_network_loss) r.power.network_loss = fl(loss);
    }
    tables(h, r);
    return r;
}

Nec4Printout printout_of(const py::handle &h) {
    Nec4Printout d;
    for (const py::handle &w : at(h, "wires")) {
        WireTableRow r;
        const std::string kind = type_name(w);
        if (kind == "ScaleRow") {
            r.kind = WireTableRow::scale;
            r.factor = fl(at(w, "factor"));
        } else if (kind == "ReflectRow") {
            r.kind = WireTableRow::reflect;
            py::object planes = at(w, "planes");
            for (int c = 0; c < 3; c++) r.planes[c] = py::cast<bool>(planes[py::int_(c)]);
            r.tag_increment = in(at(w, "tag_increment"));
        } else if (kind == "MoveRow") {
            r.kind = WireTableRow::move;
            r.tag_increment = in(at(w, "tag_increment"));
            r.copies = in(at(w, "copies"));
            three(at(w, "rotation_deg"), r.rotation_deg);
            three(at(w, "translation"), r.translation);
            r.first_tag = in(at(w, "first_tag"));
            r.move_segments = in(at(w, "segments"));
        } else if (kind == "ArcRow") {
            r.kind = WireTableRow::arc;
            r.number = in(at(w, "number"));
            r.arc_radius = fl(at(w, "arc_radius"));
            r.angle1_deg = fl(at(w, "angle1_deg"));
            r.angle2_deg = fl(at(w, "angle2_deg"));
            r.radius = fl(at(w, "radius"));
            r.segments = in(at(w, "segments"));
            r.first = in(at(w, "first"));
            r.last = in(at(w, "last"));
            r.tag = in(at(w, "tag"));
        } else if (kind == "HelixRow") {
            r.kind = WireTableRow::helix;
            r.number = in(at(w, "number"));
            r.turns = fl(at(w, "turns"));
            r.length = fl(at(w, "length"));
            py::object hr = at(w, "helix_radii"), wr = at(w, "wire_radii");
            r.helix_radii[0] = fl(hr[py::int_(0)]);
            r.helix_radii[1] = fl(hr[py::int_(1)]);
            r.wire_radii[0] = fl(wr[py::int_(0)]);
            r.wire_radii[1] = fl(wr[py::int_(1)]);
            r.wire_length = fl(at(w, "wire_length"));
            r.segments = in(at(w, "segments"));
            r.first = in(at(w, "first"));
            r.last = in(at(w, "last"));
            r.tag = in(at(w, "tag"));
        } else if (kind == "CatenaryRow") {
            r.kind = WireTableRow::catenary;
            r.icat = in(at(w, "icat"));
            r.rhm = fl(at(w, "rhm"));
            r.zm = fl(at(w, "zm"));
            r.length = fl(at(w, "length"));
            r.wire_length = fl(at(w, "chord_length"));
            py::object row = at(w, "wire");
            r.number = in(at(row, "number"));
            three(at(row, "end1"), r.end1);
            three(at(row, "end2"), r.end2);
            r.radius = fl(at(row, "radius"));
            r.segments = in(at(row, "segments"));
            r.first = in(at(row, "first"));
            r.last = in(at(row, "last"));
            r.tag = in(at(row, "tag"));
        } else {
            r.number = in(at(w, "number"));
            three(at(w, "end1"), r.end1);
            three(at(w, "end2"), r.end2);
            r.radius = fl(at(w, "radius"));
            r.segments = in(at(w, "segments"));
            r.first = in(at(w, "first"));
            r.last = in(at(w, "last"));
            r.tag = in(at(w, "tag"));
        }
        d.wires.push_back(r);
    }
    d.ge_flag = in(at(h, "ge_flag"));
    d.total_segments = in(at(h, "total_segments"));
    for (const py::handle &j : at(h, "junctions")) {
        std::vector<long long> members;
        for (const py::handle &m : j) members.push_back(in(m));
        d.junctions.push_back(members);
    }
    for (const py::handle &s : at(h, "segments")) {
        SegmentDataRow r;
        r.number = in(at(s, "number"));
        three(at(s, "centre"), r.centre);
        r.length = fl(at(s, "length"));
        r.alpha_deg = fl(at(s, "alpha_deg"));
        r.beta_deg = fl(at(s, "beta_deg"));
        r.radius = fl(at(s, "radius"));
        r.i_minus = in(at(s, "i_minus"));
        r.i_plus = in(at(s, "i_plus"));
        r.tag = in(at(s, "tag"));
        d.segments.push_back(r);
    }
    for (const py::handle &r : at(h, "runs")) d.runs.push_back(run(r));
    d.run_seconds = fl(at(h, "run_seconds"));
    d.cell_segments = in(at(h, "cell_segments"));
    d.symmetry_flag = in(at(h, "symmetry_flag"));
    d.symmetry_planes = in(at(h, "symmetry_planes"));
    return d;
}

mw::nec5::EngineStamp stamp_of(const std::string &version, const std::string &variant, const std::string &marker) {
    mw::nec5::EngineStamp s;
    s.version = version;
    s.variant = variant;
    s.marker = marker;
    return s;
}

}  // namespace

void register_nec4_printout(py::module_ &m) {
    m.def(
        "nec4_render_printout",
        [](py::bytes latin1_echo, py::object data, const std::string &basis, const std::string &version,
           const std::string &variant, const std::string &marker, bool closing) -> std::string {
            return render_nec4_printout(std::string(latin1_echo), printout_of(data), basis,
                                        stamp_of(version, variant, marker), closing);
        },
        py::arg("latin1_echo"), py::arg("data"), py::arg("basis"), py::arg("version"), py::arg("variant"),
        py::arg("marker"), py::arg("closing") = true,
        "The C++ NEC-4.2 writer's text (UTF-8) for a Python Nec4Printout.");
    m.def(
        "nec4_refusal_printout",
        [](py::bytes latin1_text, const std::string &basis, const std::string &version, const std::string &variant,
           const std::string &marker) -> py::object {
            std::string out;
            if (!refusal_printout(std::string(latin1_text), basis, stamp_of(version, variant, marker), out))
                return py::none();
            return py::str(out);
        },
        py::arg("latin1_text"), py::arg("basis"), py::arg("version"), py::arg("variant"), py::arg("marker"),
        "The slot's whole printout (UTF-8) when the C++ seam or build refuses the deck "
        "before any solve, else None.");
    // The NEC-4.2 slot's run half over the sinusoidal basis (momwire#1607
    // B4a). `blas`: numpy's ILP64 cblas_zgemm / cblas_zgemv / cblas_zdotu_sub /
    // zgesv, SciPy's LP64 zgetrf / zgetrs, then numpy's cblas_dgemm /
    // cblas_dgemv / cblas_ddot, and SciPy's LP64 zgesvx (the network
    // solve), as addresses.
    m.def(
        "sinusoidal_nec4_printout",
        [](py::bytes latin1_text, std::vector<uintptr_t> blas, const std::string &version,
           const std::string &variant, const std::string &marker) -> py::tuple {
            if (blas.size() != 10) throw py::value_error("blas: ten addresses");
            mw::sinusoidal::FillLinalg la;
            la.np.zgemm = reinterpret_cast<mw::nec5::cblas_zgemm_fn>(blas[0]);
            la.np.zgemv = reinterpret_cast<mw::nec5::cblas_zgemv_fn>(blas[1]);
            la.np.zdotu_sub = reinterpret_cast<mw::nec5::cblas_zdotu_sub_fn>(blas[2]);
            la.np.zgesv = reinterpret_cast<mw::nec5::zgesv_fn>(blas[3]);
            la.lu.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(blas[4]);
            la.lu.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(blas[5]);
            la.np.dgemm = reinterpret_cast<mw::nec5::cblas_dgemm_fn>(blas[6]);
            la.np.dgemv = reinterpret_cast<mw::nec5::cblas_dgemv_fn>(blas[7]);
            la.np.ddot = reinterpret_cast<mw::nec5::cblas_ddot_fn>(blas[8]);
            la.np.zgesvx = reinterpret_cast<mw::nec5::zgesvx_fn>(blas[9]);
            const std::string text = latin1_text;
            const mw::nec5::EngineStamp st = stamp_of(version, variant, marker);
            std::string out, why;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = mw::sinusoidal::printout(text, st, la, out, why);
            }
            return py::make_tuple(ok, py::str(ok ? out : why));
        },
        py::arg("latin1_text"), py::arg("blas"), py::arg("version"), py::arg("variant"), py::arg("marker"),
        "(True, the slot's printout) for basis 'sinusoidal' from the C++ run half, "
        "or (False, why it declines).");
    // The Sinusoidal-Galerkin basis on the same run half (momwire#1718 S3).
    m.def(
        "sinusoidal_galerkin_nec4_printout",
        [](py::bytes latin1_text, std::vector<uintptr_t> blas, const std::string &version,
           const std::string &variant, const std::string &marker) -> py::tuple {
            if (blas.size() != 10) throw py::value_error("blas: ten addresses");
            mw::sinusoidal::FillLinalg la;
            la.np.zgemm = reinterpret_cast<mw::nec5::cblas_zgemm_fn>(blas[0]);
            la.np.zgemv = reinterpret_cast<mw::nec5::cblas_zgemv_fn>(blas[1]);
            la.np.zdotu_sub = reinterpret_cast<mw::nec5::cblas_zdotu_sub_fn>(blas[2]);
            la.np.zgesv = reinterpret_cast<mw::nec5::zgesv_fn>(blas[3]);
            la.lu.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(blas[4]);
            la.lu.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(blas[5]);
            la.np.dgemm = reinterpret_cast<mw::nec5::cblas_dgemm_fn>(blas[6]);
            la.np.dgemv = reinterpret_cast<mw::nec5::cblas_dgemv_fn>(blas[7]);
            la.np.ddot = reinterpret_cast<mw::nec5::cblas_ddot_fn>(blas[8]);
            la.np.zgesvx = reinterpret_cast<mw::nec5::zgesvx_fn>(blas[9]);
            const std::string text = latin1_text;
            const mw::nec5::EngineStamp st = stamp_of(version, variant, marker);
            std::string out, why;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = mw::sinusoidal_galerkin::printout(text, st, la, out, why);
            }
            return py::make_tuple(ok, py::str(ok ? out : why));
        },
        py::arg("latin1_text"), py::arg("blas"), py::arg("version"), py::arg("variant"), py::arg("marker"),
        "(True, the slot's printout) for basis 'sinusoidal-galerkin' from the C++ run half, "
        "or (False, why it declines).");
}
