#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "attenuation.hpp"
#include "kinematics.hpp"

namespace py = pybind11;
using albedo::AttenuationTable;

PYBIND11_MODULE(_core, m) {
    m.doc() = "ALBEDO transport core";

    m.attr("ELECTRON_MASS") = albedo::kElectronMass;

    m.def("compton_energy", py::vectorize(&albedo::compton_energy),
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
}
