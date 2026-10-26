#pragma once

#include <cmath>

#include "kinematics.hpp"
#include "random.hpp"

namespace albedo {

// Samples cos(theta) for Compton scattering on a free electron at rest,
// distributed according to the Klein-Nishina cross section.
//
// Method of Butcher and Messel, Nucl. Phys. 20 (1960) 15, as used in
// Geant4 (G4KleinNishinaCompton). The variable sampled is the energy
// ratio eps = E'/E in [eps0, 1], eps0 = 1/(1 + 2k), k = E/(m c^2). The
// cross section is written as f(eps) g(eps), with f a mixture of 1/eps and
// eps that is sampled exactly and g <= 1 a rejection function. The
// acceptance probability is above 0.65 between 30 keV and 3 MeV.
inline double sample_klein_nishina(double energy, Random& rng) {
    const double k = energy / kElectronMass;
    const double eps0 = 1.0 / (1.0 + 2.0 * k);
    const double eps0_sq = eps0 * eps0;
    const double alpha1 = -std::log(eps0);
    const double alpha2 = 0.5 * (1.0 - eps0_sq);

    while (true) {
        double eps, eps_sq;
        if (alpha1 > (alpha1 + alpha2) * rng.uniform()) {
            eps = std::exp(-alpha1 * rng.uniform());  // f ~ 1/eps
            eps_sq = eps * eps;
        } else {
            eps_sq = eps0_sq + (1.0 - eps0_sq) * rng.uniform();  // f ~ eps
            eps = std::sqrt(eps_sq);
        }
        const double one_minus_cos = (1.0 - eps) / (eps * k);
        const double sin_sq = one_minus_cos * (2.0 - one_minus_cos);
        const double g = 1.0 - eps * sin_sq / (1.0 + eps_sq);
        if (g >= rng.uniform()) {
            return 1.0 - one_minus_cos;
        }
    }
}

}  // namespace albedo
