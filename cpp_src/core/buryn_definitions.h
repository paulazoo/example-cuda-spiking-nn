#ifndef BURYN_DEFINITIONS_H
#define BURYN_DEFINITIONS_H

#include <cstddef>
#include <cstdint>

namespace buryn {

using Weight = float;

inline int rng_seed = 314; // default but changeable
inline float timestep = 1e-4;  // default 0.1 ms but changeable
inline uint32_t recording_interval_steps = 10000; // default 10000 steps (1s)

}  // namespace buryn

#endif  // BURYN_DEFINITIONS_H