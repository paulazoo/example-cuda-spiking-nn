#ifndef GENERATE_STIMULUS_SCHEDULES_H
#define GENERATE_STIMULUS_SCHEDULES_H

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>
#include <cmath>

#include "buryn_definitions.h"
#include "math_functions.h"

namespace generate_stimulus_schedules {

struct StimulusGenerationParams {
    uint32_t max_timestep;
    std::vector<uint32_t> selected_input_ids;
    std::vector<uint32_t> selected_input_post_group_offsets;
    std::vector<uint32_t> start_clock_steps;
    std::vector<uint32_t> end_clock_steps;
    std::vector<float> stimulus_input_values;
};

struct StimulusGenerationResult {
    std::vector<uint32_t> timestep_event_counts;
    std::vector<uint32_t> neuron_ids;
    std::vector<uint32_t> post_group_offsets;
    std::vector<float> amplitude_changes;
};

// Helper: stable key sort/merge for (time, neuron)
struct Event {
    uint32_t t;
    uint32_t neuron;
    float delta;
};

StimulusGenerationResult partial_group_multiple_square_waves(
    const std::string& folder_name,
    StimulusGenerationParams& params);

}  // namespace generate_stimulus_schedules

#endif  // GENERATE_STIMULUS_SCHEDULES_H
