#pragma once

#include <cstdint>
#include <random>

namespace albedo {

// Single source of random numbers for a simulation.
//
// The 64-bit Mersenne Twister is fully specified by the C++ standard, so a
// given seed produces the same sequence on every platform. The standard
// distributions (std::uniform_real_distribution, ...) are not, so uniform
// deviates are built here directly from the top 53 bits of each draw.
class Random {
public:
    explicit Random(std::uint64_t seed) : engine_(seed) {}

    // Uniform deviate in [0, 1).
    double uniform() {
        return static_cast<double>(engine_() >> 11) * 0x1.0p-53;
    }

    // Uniform deviate in (0, 1]; safe as the argument of std::log.
    double uniform_pos() { return 1.0 - uniform(); }

private:
    std::mt19937_64 engine_;
};

}  // namespace albedo
