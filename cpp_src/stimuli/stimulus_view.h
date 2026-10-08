#ifndef STIMULUS_VIEW_H
#define STIMULUS_VIEW_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct StimulusView {
    float* g_excitatory = nullptr;
    float* g_inhibitory = nullptr;
    float* i_stimulus = nullptr;
    uint32_t* timestep_event_count = nullptr;
    uint32_t* stimulus_event_neuron_ids = nullptr;
    uint32_t* stimulus_event_post_group_offsets = nullptr;
    float* stimulus_event_amplitude_changes = nullptr;
};


struct StimulusInitHostData {
    std::vector<uint32_t> timestep_event_count;
    std::vector<uint32_t> stimulus_event_neuron_ids;
    std::vector<uint32_t> stimulus_event_post_group_offsets;
    std::vector<float> stimulus_event_amplitude_changes;
};

#endif  // STIMULUS_VIEW_H
