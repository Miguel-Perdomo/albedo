#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "attenuation.hpp"
#include "compton.hpp"
#include "direction.hpp"
#include "geometry.hpp"
#include "kinematics.hpp"
#include "random.hpp"
#include "source.hpp"
#include "vec3.hpp"

namespace albedo {

// One photon history that deposited energy in the detector.
struct Event {
    double deposited;  // total energy deposited in the detector (keV)
    double emitted;    // energy of the emitted photon (keV)
    int origin;        // volume of the last interaction outside the detector
                       // before the first deposit; -1 if the photon arrived
                       // without interacting
};

// Analog photon transport: photoelectric absorption and Compton
// scattering on free electrons. Secondary electrons deposit their energy
// at the interaction point; fluorescence, Rayleigh scattering and pair
// production are not included.
class Transport {
public:
    Transport(Geometry geometry, std::vector<AttenuationTable> tables, LineSampler source,
              const Vec3& source_position, int detector_volume, double energy_cut)
        : geo_(std::move(geometry)),
          tables_(std::move(tables)),
          source_(std::move(source)),
          source_position_(source_position),
          detector_(detector_volume),
          energy_cut_(energy_cut) {
        if (detector_ < 0 || static_cast<std::size_t>(detector_) >= geo_.size()) {
            throw std::invalid_argument("Transport: detector volume out of range");
        }
        for (std::size_t v = 0; v < geo_.size(); ++v) {
            const int m = geo_.material(static_cast<int>(v));
            if (m != kVacuum && (m < 0 || static_cast<std::size_t>(m) >= tables_.size())) {
                throw std::invalid_argument("Transport: volume refers to a missing material");
            }
        }
        if (geo_.locate(source_position_) < 0) {
            throw std::invalid_argument("Transport: source is outside the world");
        }
    }

    // Simulate n histories; returns the ones that deposited energy.
    std::vector<Event> run(std::uint64_t n, std::uint64_t seed) const {
        Random rng(seed);
        std::vector<Event> events;
        for (std::uint64_t i = 0; i < n; ++i) {
            Event ev;
            if (history(rng, ev)) events.push_back(ev);
        }
        return events;
    }

private:
    Geometry geo_;
    std::vector<AttenuationTable> tables_;
    LineSampler source_;
    Vec3 source_position_;
    int detector_;
    double energy_cut_;

    bool history(Random& rng, Event& ev) const {
        double energy = source_.sample(rng);
        Vec3 p = source_position_;
        Vec3 d = isotropic_direction(rng);

        ev = {0.0, energy, -1};
        bool detected = false;
        int last_outside = -1;
        int volume = geo_.locate(p);

        while (volume >= 0) {
            const double to_boundary = geo_.distance_to_boundary(p, d);
            if (!std::isfinite(to_boundary)) break;

            const int material = geo_.material(volume);
            if (material == kVacuum) {
                p = p + (to_boundary + kBoundaryPush) * d;
                volume = geo_.locate(p);
                continue;
            }

            const AttenuationTable& table = tables_[static_cast<std::size_t>(material)];
            const double mu = table.mu_total(energy);
            const double to_interaction = -std::log(rng.uniform_pos()) / mu;
            if (to_interaction >= to_boundary) {
                p = p + (to_boundary + kBoundaryPush) * d;
                volume = geo_.locate(p);
                continue;
            }

            p = p + to_interaction * d;
            const bool in_detector = (volume == detector_);
            if (in_detector && !detected) {
                detected = true;
                ev.origin = last_outside;
            }

            if (rng.uniform() * mu < table.mu_photoelectric(energy)) {
                if (in_detector) ev.deposited += energy;
                break;
            }

            const double cos_theta = sample_klein_nishina(energy, rng);
            const double scattered = compton_energy(energy, cos_theta);
            if (in_detector) {
                ev.deposited += energy - scattered;
            } else {
                last_outside = volume;
            }
            d = rotate(d, cos_theta, 2.0 * kPi * rng.uniform());
            energy = scattered;

            if (energy < energy_cut_) {
                if (in_detector) ev.deposited += energy;
                break;
            }
        }
        return detected;
    }
};

}  // namespace albedo
