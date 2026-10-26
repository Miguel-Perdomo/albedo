#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "random.hpp"

namespace albedo {

// Samples the energy of an emitted photon from a discrete set of lines,
// each chosen with probability proportional to its intensity.
class LineSampler {
public:
    LineSampler(const std::vector<double>& energy, const std::vector<double>& intensity)
        : energy_(energy) {
        if (energy.empty() || energy.size() != intensity.size()) {
            throw std::invalid_argument("LineSampler: need equal, non-empty arrays");
        }
        double total = 0.0;
        for (const double w : intensity) {
            if (w < 0.0) throw std::invalid_argument("LineSampler: negative intensity");
            total += w;
        }
        if (total <= 0.0) throw std::invalid_argument("LineSampler: zero total intensity");

        cdf_.reserve(intensity.size());
        double running = 0.0;
        for (const double w : intensity) {
            running += w;
            cdf_.push_back(running / total);
        }
        cdf_.back() = 1.0;
    }

    double sample(Random& rng) const {
        const double u = rng.uniform();  // [0, 1)
        const auto it = std::upper_bound(cdf_.begin(), cdf_.end(), u);
        return energy_[static_cast<std::size_t>(it - cdf_.begin())];
    }

private:
    std::vector<double> energy_;
    std::vector<double> cdf_;
};

}  // namespace albedo
