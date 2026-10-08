#ifndef PLASTICITY_PARAMS_H
#define PLASTICITY_PARAMS_H

#include <cstddef>
#include <cstdint>

struct PlasticityParams {
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

    uint32_t plasticity_offset = 0;
};


    // std::vector<float> pre_trace_;
    // std::vector<float> post_trace_;
    // uint32_t rows_;
    // uint32_t cols_;

#endif  // PLASTICITY_PARAMS_H
