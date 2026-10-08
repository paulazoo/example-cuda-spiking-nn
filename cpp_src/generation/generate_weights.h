#ifndef GENERATE_WEIGHTS_H
#define GENERATE_WEIGHTS_H

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
#include <cstdint>

#include "buryn_definitions.h"
#include "math_functions.h"

namespace generate_weights {

struct WeightGenerationParams {
    float x_location_length;
    float y_location_length;
    float max_distance;
    float spread_scale;
    float max_weight_value;
    bool  weight_value_normal_dropoff;
    bool  probabilistic_normal_not_uniform;
    float probability_multiplier;
    bool  no_diagonal_connections;
};

void weights_file_header(std::ofstream& output);

std::vector<std::vector<float>> distance_2d_weights_file(const std::string& filename,
                                                        const std::vector<std::vector<float>>& pre_group_neuron_locations,
                                                        const std::vector<std::vector<float>>& post_group_neuron_locations,
                                                        const WeightGenerationParams& params);

float calculate_connection_probability(float distance, float spread_scale);

}  // namespace generate_weights

#endif  // GENERATE_WEIGHTS_H
