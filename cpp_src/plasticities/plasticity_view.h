#ifndef PLASTICITY_VIEW_H
#define PLASTICITY_VIEW_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct PlasticityView {
    // Scalar params
    float ltp_A = 0.0f;
    float ltd_A = 0.0f;
    float ltp_B = 0.0f;
    float ltd_B = 0.0f;
    float pre_trace_decay = 0.0f;
    float post_trace_decay = 0.0f;
    float triplet_pre_trace_decay = 0.0f;
    float triplet_post_trace_decay = 0.0f;
    float weight_min = 0.0f;
    float weight_max = 1.0f;
    
    // Offsets and sizes
    uint32_t weight_matrix_idx = 0;
    uint32_t plasticity_offset = 0;
    uint32_t pre_group_offset = 0;
    uint32_t post_group_offset = 0;
    uint32_t pre_group_num_neurons = 0;
    uint32_t post_group_num_neurons = 0;

    // Pointers to GPU data
    float* pre_traces = nullptr;
    float* post_traces = nullptr;
    float* triplet_pre_traces = nullptr;
    float* triplet_post_traces = nullptr;
    uint32_t* pre_spikes = nullptr;
    uint32_t* post_spikes = nullptr;
    float* weight_matrix = nullptr;
    float* pre_weight_sums = nullptr;
};

struct PlasticityInitHostData {
    std::vector<float> pre_traces;
    std::vector<float> post_traces;
    std::vector<float> triplet_pre_traces;
    std::vector<float> triplet_post_traces;
    std::vector<float> pre_weight_sums;
};

#endif  // PLASTICITY_VIEW_H
