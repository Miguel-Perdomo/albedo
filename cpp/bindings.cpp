#include <array>
#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "attenuation.hpp"
#include "compton.hpp"
#include "direction.hpp"
#include "kinematics.hpp"
#include "random.hpp"
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

std::array<double, 3> rotate_py(const std::array<double, 3>& d, double cos_theta, double phi) {
    const Vec3 v = rotate({d[0], d[1], d[2]}, cos_theta, phi);
    return {v.x, v.y, v.z};
}

}  // namespace

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
}
