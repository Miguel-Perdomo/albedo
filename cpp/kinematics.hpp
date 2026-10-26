#pragma once

namespace albedo {

constexpr double kElectronMass = 510.99895;  // keV, CODATA 2018

// Energy (keV) of a photon of energy `energy` (keV) after Compton
// scattering through an angle theta, given cos(theta).
inline double compton_energy(double energy, double cos_theta) {
    return energy / (1.0 + (energy / kElectronMass) * (1.0 - cos_theta));
}

}  // namespace albedo
