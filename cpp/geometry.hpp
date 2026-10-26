#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vec3.hpp"

namespace albedo {

constexpr int kVacuum = -1;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Distance (cm) a particle is moved past a boundary after reaching it, so
// that the next point lookup is unambiguous. Small compared with any
// physical length in the problem and large compared with round-off.
constexpr double kBoundaryPush = 1e-8;

// Solid cylinder with its axis along z.
struct Cylinder {
    double radius;
    double z_min;
    double z_max;
    int material;  // index into the material list, or kVacuum

    bool contains(const Vec3& p) const {
        return p.z >= z_min && p.z <= z_max && p.x * p.x + p.y * p.y <= radius * radius;
    }

    // Distance along the unit vector d from p to the first crossing of the
    // cylinder surface (side or caps) with t > 0; kInfinity if none.
    double distance_to_surface(const Vec3& p, const Vec3& d) const {
        double best = kInfinity;

        const double a = d.x * d.x + d.y * d.y;
        if (a > 0.0) {
            const double b = p.x * d.x + p.y * d.y;
            const double c = p.x * p.x + p.y * p.y - radius * radius;
            const double disc = b * b - a * c;
            if (disc >= 0.0) {
                const double sq = std::sqrt(disc);
                for (const double t : {(-b - sq) / a, (-b + sq) / a}) {
                    const double z = p.z + t * d.z;
                    if (t > 0.0 && t < best && z >= z_min && z <= z_max) best = t;
                }
            }
        }

        if (d.z != 0.0) {
            for (const double zc : {z_min, z_max}) {
                const double t = (zc - p.z) / d.z;
                if (t > 0.0 && t < best) {
                    const double x = p.x + t * d.x;
                    const double y = p.y + t * d.y;
                    if (x * x + y * y <= radius * radius) best = t;
                }
            }
        }
        return best;
    }
};

// Ordered list of coaxial cylinders. A point belongs to the first cylinder
// in the list that contains it; points outside every cylinder are outside
// the world. Nesting solid cylinders therefore produces shells.
class Geometry {
public:
    int add_cylinder(double radius, double z_min, double z_max, int material) {
        if (radius <= 0.0 || z_max <= z_min) {
            throw std::invalid_argument("Geometry: cylinder needs radius > 0 and z_max > z_min");
        }
        cylinders_.push_back({radius, z_min, z_max, material});
        return static_cast<int>(cylinders_.size()) - 1;
    }

    // Index of the volume containing p, or -1 if p is outside the world.
    int locate(const Vec3& p) const {
        for (std::size_t i = 0; i < cylinders_.size(); ++i) {
            if (cylinders_[i].contains(p)) return static_cast<int>(i);
        }
        return -1;
    }

    int material(int volume) const {
        return volume < 0 ? kVacuum : cylinders_[static_cast<std::size_t>(volume)].material;
    }

    // Distance to the nearest surface of any cylinder. The material can
    // only change on one of these surfaces.
    double distance_to_boundary(const Vec3& p, const Vec3& d) const {
        double best = kInfinity;
        for (const auto& c : cylinders_) best = std::fmin(best, c.distance_to_surface(p, d));
        return best;
    }

    std::size_t size() const { return cylinders_.size(); }

private:
    std::vector<Cylinder> cylinders_;
};

struct Segment {
    int volume;
    double length;
};

// Straight-line path from p along the unit vector d until it leaves the
// world, as consecutive (volume, length) segments.
inline std::vector<Segment> trace(const Geometry& geo, Vec3 p, const Vec3& d) {
    std::vector<Segment> segments;
    int volume = geo.locate(p);
    while (volume >= 0) {
        const double s = geo.distance_to_boundary(p, d);
        if (!std::isfinite(s)) break;
        p = p + (s + kBoundaryPush) * d;
        if (!segments.empty() && segments.back().volume == volume) {
            segments.back().length += s + kBoundaryPush;
        } else {
            segments.push_back({volume, s + kBoundaryPush});
        }
        volume = geo.locate(p);
    }
    return segments;
}

}  // namespace albedo
