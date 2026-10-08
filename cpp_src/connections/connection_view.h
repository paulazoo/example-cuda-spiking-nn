#ifndef CONNECTION_VIEW_H
#define CONNECTION_VIEW_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct ConnectionView {
    // Scalar params
    float postsynaptic_r_trace_decay = 0.0f;
    float postsynaptic_d_trace_decay = 0.0f;
    float postsynaptic_kernel_normalization_value = 1.0f;
    bool pre_group_is_inhibtory = false;
    
    // Offsets and sizes
    uint32_t weight_matrix_idx = 0;
    uint32_t pre_group_offset = 0;
    uint32_t post_group_offset = 0;
    uint32_t pre_group_num_neurons = 0;
    uint32_t post_group_num_neurons = 0;

    // Pointers to GPU data
    float* weight_matrix = nullptr;
    float* postsynaptic_r_traces = nullptr;
    float* postsynaptic_d_traces = nullptr;
    float* summed_postsynaptic_conductances = nullptr;
    float* post_group_g_excitatory = nullptr;
    float* post_group_g_inhibitory = nullptr;
    uint32_t* spikes = nullptr;
    void* cublas_handle = nullptr;
};


struct ConnectionInitHostData {
    std::vector<float> weight_matrix;
    std::vector<float> postsynaptic_r_traces;
    std::vector<float> postsynaptic_d_traces;
    std::vector<float> summed_postsynaptic_conductances;
};

#endif  // CONNECTION_VIEW_H
