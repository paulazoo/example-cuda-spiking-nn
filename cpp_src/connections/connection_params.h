#ifndef CONNECTION_PARAMS_H
#define CONNECTION_PARAMS_H

#include <cstddef>
#include <cstdint>

struct ConnectionParams {
    float timestep = 0.0f;
    float decay_time = 0.0f;
    float rise_time = 0.0f;
    float postsynaptic_kernel_normalization_value = 0.0f;
    
    uint32_t weight_matrix_idx = 0;
};

#endif  // CONNECTION_PARAMS_H
