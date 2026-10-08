#ifndef GENERATE_LOCATION_H
#define GENERATE_LOCATION_H

#include <cmath>
#include <cstdint>

#include "neuron_group.h"

namespace generate_location {

std::vector<std::vector<float>> generate_2d_grid_locations(const std::string& output_filepath,
                                    uint32_t num_neurons,
                                    float spacing,
                                    uint32_t rows,
                                    uint32_t cols);

}  // namespace generate_location

#endif  // GENERATE_LOCATION_H
