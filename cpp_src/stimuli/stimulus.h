#ifndef STIMULUS_H
#define STIMULUS_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <regex>
#include <unordered_set>

#include "buryn_definitions.h"
#include "neuron_group.h"

#include "stimulus_view.h"
#include "kernel_stimulus_apply.h"

class Stimulus {
public:
    Stimulus(std::vector<uint32_t> timestep_event_count,
                std::vector<uint32_t> stimulus_event_neuron_ids,
                std::vector<uint32_t> stimulus_event_post_group_offsets,
                std::vector<float> stimulus_event_amplitude_changes,
                GpuSimulationState& gpu_state);
    ~Stimulus() = default;

    StimulusInitHostData initialize_data(const std::vector<uint32_t>& timestep_event_count,
                                        const std::vector<uint32_t>& stimulus_event_neuron_ids,
                                        const std::vector<uint32_t>& stimulus_event_post_group_offsets,
                                        const std::vector<float>& stimulus_event_amplitude_changes);
    void make_gpu_view();
    void upload_data(const StimulusInitHostData& host_data);

    void apply();

private:
    // CPU only data
    std::vector<uint32_t> timestep_event_count_;
    uint32_t next_event_idx_ = 0;
    uint32_t current_timestep_ = 0;
    uint32_t total_stimulus_events_ = 0;

    // GPU data
    StimulusView gpu_view_;
    GpuSimulationState& gpu_state_;
};

#endif  // STIMULUS_H