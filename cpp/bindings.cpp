#include <array>
#include <cstdint>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "attenuation.hpp"
#include "compton.hpp"
#include "direction.hpp"
#include "geometry.hpp"
#include "kinematics.hpp"
#include "random.hpp"
#include "source.hpp"
#include "transport.hpp"
#include "vec3.hpp"

namespace py = pybind11;
using namespace albedo;

namespace {

py::array_t<double> sample_klein_nishina_array(double energy, py::ssize_t n, std::uint64_t seed) {
    Random rng(seed);
    py::array_t<double> out(n);
    auto r = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) r(i) = sample_klein_nishina(energy, rng);
    return out;
}

py::array_t<double> isotropic_directions(py::ssize_t n, std::uint64_t seed) {
    Random rng(seed);
    py::array_t<double> out({n, py::ssize_t{3}});
    auto r = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        const Vec3 d = isotropic_direction(rng);
        r(i, 0) = d.x;
        r(i, 1) = d.y;
        r(i, 2) = d.z;
    }
    return out;
}

Vec3 unit(const std::array<double, 3>& v) {
    const Vec3 u{v[0], v[1], v[2]};
    const double n = norm(u);
    if (n == 0.0) throw std::invalid_argument("direction must be non-zero");
    return (1.0 / n) * u;
}

Vec3 point(const std::array<double, 3>& v) { return {v[0], v[1], v[2]}; }

std::array<double, 3> rotate_py(const std::array<double, 3>& d, double cos_theta, double phi) {
    const Vec3 v = rotate({d[0], d[1], d[2]}, cos_theta, phi);
    return {v.x, v.y, v.z};
}

}  // namespace

namespace pybind11::detail {
// Accept any 3-element sequence where a Vec3 is expected.
template <>
struct type_caster<Vec3> {
    PYBIND11_TYPE_CASTER(Vec3, const_name("Vec3"));
    bool load(handle src, bool convert) {
        make_caster<std::array<double, 3>> inner;
        if (!inner.load(src, convert)) return false;
        const auto& v = cast_op<const std::array<double, 3>&>(inner);
        value = {v[0], v[1], v[2]};
        return true;
    }
    static handle cast(const Vec3& v, return_value_policy, handle) {
        return py::make_tuple(v.x, v.y, v.z).release();
    }
};
}  // namespace pybind11::detail

PYBIND11_MODULE(_core, m) {
    m.doc() = "ALBEDO transport core";

    m.attr("ELECTRON_MASS") = kElectronMass;

    m.def("compton_energy", py::vectorize(&compton_energy),
          py::arg("energy"), py::arg("cos_theta"),
          "Scattered photon energy (keV) for incident energy (keV) and cos(theta).");

    py::class_<AttenuationTable>(m, "AttenuationTable")
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<double>&>(),
             py::arg("energy"), py::arg("mu_incoherent"), py::arg("mu_photoelectric"))
        .def("mu_incoherent", py::vectorize(&AttenuationTable::mu_incoherent),
             py::arg("energy"), "Linear incoherent coefficient (cm^-1).")
        .def("mu_photoelectric", py::vectorize(&AttenuationTable::mu_photoelectric),
             py::arg("energy"), "Linear photoelectric coefficient (cm^-1).")
        .def("mu_total", py::vectorize(&AttenuationTable::mu_total),
             py::arg("energy"), "Incoherent + photoelectric (cm^-1).")
        .def_property_readonly("energy_min", &AttenuationTable::energy_min)
        .def_property_readonly("energy_max", &AttenuationTable::energy_max);

    // Sampling routines exposed for testing and teaching. The transport
    // loop calls the C++ functions directly.
    m.def("sample_klein_nishina", &sample_klein_nishina_array,
          py::arg("energy"), py::arg("n"), py::arg("seed"),
          "Draw n values of cos(theta) from the Klein-Nishina distribution.");
    m.def("isotropic_directions", &isotropic_directions,
          py::arg("n"), py::arg("seed"),
          "Draw n unit vectors uniformly distributed on the sphere, shape (n, 3).");
    m.def("rotate", &rotate_py,
          py::arg("direction"), py::arg("cos_theta"), py::arg("phi"),
          "Deflect a unit vector by polar angle theta and azimuth phi.");

    py::class_<Geometry>(m, "Geometry")
        .def(py::init<>())
        .def("add_cylinder", &Geometry::add_cylinder,
             py::arg("radius"), py::arg("z_min"), py::arg("z_max"), py::arg("material"),
             "Append a solid cylinder; returns its volume index.")
        .def("locate", [](const Geometry& g, const std::array<double, 3>& p) {
                 return g.locate(point(p));
             },
             py::arg("point"), "Volume index containing the point, or -1 outside the world.")
        .def("material", &Geometry::material, py::arg("volume"))
        .def("distance_to_boundary",
             [](const Geometry& g, const std::array<double, 3>& p, const std::array<double, 3>& d) {
                 return g.distance_to_boundary(point(p), unit(d));
             },
             py::arg("point"), py::arg("direction"))
        .def("trace",
             [](const Geometry& g, const std::array<double, 3>& p, const std::array<double, 3>& d) {
                 py::list out;
                 for (const auto& s : trace(g, point(p), unit(d))) {
                     out.append(py::make_tuple(s.volume, s.length));
                 }
                 return out;
             },
             py::arg("point"), py::arg("direction"),
             "Straight path until leaving the world, as (volume, length) pairs.")
        .def("__len__", &Geometry::size);

    m.attr("VACUUM") = kVacuum;

    py::class_<LineSampler>(m, "LineSampler")
        .def(py::init<const std::vector<double>&, const std::vector<double>&>(),
             py::arg("energy"), py::arg("intensity"))
        .def("sample",
             [](const LineSampler& s, py::ssize_t n, std::uint64_t seed) {
                 Random rng(seed);
                 py::array_t<double> out(n);
                 auto r = out.mutable_unchecked<1>();
                 for (py::ssize_t i = 0; i < n; ++i) r(i) = s.sample(rng);
                 return out;
             },
             py::arg("n"), py::arg("seed"), "Draw n photon energies (keV).");

    py::class_<Transport>(m, "Transport")
        .def(py::init<Geometry, std::vector<AttenuationTable>, LineSampler, Vec3, int, double>(),
             py::arg("geometry"), py::arg("tables"), py::arg("source"),
             py::arg("source_position"), py::arg("detector_volume"), py::arg("energy_cut"))
        .def("run",
             [](const Transport& t, std::uint64_t n, std::uint64_t seed) {
                 std::vector<Event> events;
                 {
                     py::gil_scoped_release release;
                     events = t.run(n, seed);
                 }
                 const auto size = static_cast<py::ssize_t>(events.size());
                 py::array_t<double> deposited(size), emitted(size);
                 py::array_t<int> origin(size);
                 auto a = deposited.mutable_unchecked<1>();
                 auto b = emitted.mutable_unchecked<1>();
                 auto c = origin.mutable_unchecked<1>();
                 for (py::ssize_t i = 0; i < size; ++i) {
                     const auto& ev = events[static_cast<std::size_t>(i)];
                     a(i) = ev.deposited;
                     b(i) = ev.emitted;
                     c(i) = ev.origin;
                 }
                 py::dict out;
                 out["deposited"] = deposited;
                 out["emitted"] = emitted;
                 out["origin"] = origin;
                 return out;
             },
             py::arg("n"), py::arg("seed"),
             "Run n histories; returns arrays for the events that hit the detector.");
}
