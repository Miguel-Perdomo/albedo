#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace albedo {

// Tabulated linear attenuation coefficients of one material.
//
// Energies in keV, coefficients in cm^-1. The energy grid may contain
// repeated values at absorption edges (one row below and one above the
// edge). Interpolation is linear in (ln E, ln mu) and the interval is
// located with std::lower_bound, so an energy just below an edge uses
// the below-edge value and one just above uses the above-edge value.
// Outside the table the end values are returned.
class AttenuationTable {
public:
    AttenuationTable(const std::vector<double>& energy,
                     const std::vector<double>& mu_incoherent,
                     const std::vector<double>& mu_photoelectric) {
        const std::size_t n = energy.size();
        if (n < 2 || mu_incoherent.size() != n || mu_photoelectric.size() != n) {
            throw std::invalid_argument(
                "AttenuationTable: arrays must have equal length >= 2");
        }
        log_e_.reserve(n);
        log_inc_.reserve(n);
        log_ph_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (energy[i] <= 0.0 || mu_incoherent[i] <= 0.0 || mu_photoelectric[i] <= 0.0) {
                throw std::invalid_argument(
                    "AttenuationTable: energies and coefficients must be positive");
            }
            if (i > 0 && energy[i] < energy[i - 1]) {
                throw std::invalid_argument(
                    "AttenuationTable: energies must be non-decreasing");
            }
            log_e_.push_back(std::log(energy[i]));
            log_inc_.push_back(std::log(mu_incoherent[i]));
            log_ph_.push_back(std::log(mu_photoelectric[i]));
        }
    }

    double mu_incoherent(double energy) const { return interpolate(log_inc_, energy); }
    double mu_photoelectric(double energy) const { return interpolate(log_ph_, energy); }
    double mu_total(double energy) const {
        return mu_incoherent(energy) + mu_photoelectric(energy);
    }

    double energy_min() const { return std::exp(log_e_.front()); }
    double energy_max() const { return std::exp(log_e_.back()); }

private:
    std::vector<double> log_e_;
    std::vector<double> log_inc_;
    std::vector<double> log_ph_;

    double interpolate(const std::vector<double>& log_mu, double energy) const {
        const double x = std::log(energy);
        if (x <= log_e_.front()) return std::exp(log_mu.front());
        if (x >= log_e_.back()) return std::exp(log_mu.back());

        const auto it = std::lower_bound(log_e_.begin(), log_e_.end(), x);
        const std::size_t i = static_cast<std::size_t>(it - log_e_.begin());
        const double t = (x - log_e_[i - 1]) / (log_e_[i] - log_e_[i - 1]);
        return std::exp(log_mu[i - 1] + t * (log_mu[i] - log_mu[i - 1]));
    }
};

}  // namespace albedo
