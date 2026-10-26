#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
constexpr double kElectronMass = 510.99895;  // keV, CODATA 2018

double compton_energy(double energy, double cos_theta) {
    return energy / (1.0 + (energy / kElectronMass) * (1.0 - cos_theta));
}
}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "ALBEDO transport core";
    m.def("compton_energy", &compton_energy,
          py::arg("energy"), py::arg("cos_theta"),
          "Scattered photon energy (keV) for incident energy (keV) and cos(theta).");
}
