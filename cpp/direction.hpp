#pragma once

#include <cmath>

#include "random.hpp"
#include "vec3.hpp"

namespace albedo {

constexpr double kPi = 3.14159265358979323846;

// Direction uniformly distributed over the unit sphere.
inline Vec3 isotropic_direction(Random& rng) {
    const double cos_t = 2.0 * rng.uniform() - 1.0;
    const double sin_t = std::sqrt(1.0 - cos_t * cos_t);
    const double phi = 2.0 * kPi * rng.uniform();
    return {sin_t * std::cos(phi), sin_t * std::sin(phi), cos_t};
}

// New direction after deflecting the unit vector `d` by a polar angle
// theta (given cos_theta) and an azimuthal angle phi measured around `d`.
//
// Standard closed form used in photon transport codes (e.g. PENELOPE,
// MCNP). When `d` is (anti)parallel to z the general expression is
// singular and the rotation is done directly about z.
inline Vec3 rotate(const Vec3& d, double cos_theta, double phi) {
    const double sin_t = std::sqrt(std::fmax(0.0, 1.0 - cos_theta * cos_theta));
    const double cos_p = std::cos(phi);
    const double sin_p = std::sin(phi);

    const double s2 = 1.0 - d.z * d.z;
    if (s2 < 1e-12) {
        const double sgn = d.z >= 0.0 ? 1.0 : -1.0;
        return {sin_t * cos_p, sin_t * sin_p, sgn * cos_theta};
    }
    const double s = std::sqrt(s2);
    Vec3 out{
        d.x * cos_theta + sin_t * (d.x * d.z * cos_p - d.y * sin_p) / s,
        d.y * cos_theta + sin_t * (d.y * d.z * cos_p + d.x * sin_p) / s,
        d.z * cos_theta - sin_t * s * cos_p,
    };
    return (1.0 / norm(out)) * out;  // remove accumulated round-off
}

}  // namespace albedo
