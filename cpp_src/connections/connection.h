#ifndef CONNECTION_H
#define CONNECTION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "buryn_definitions.h"
#include "neuron_group.h"

#include "gpu_simulation_state.h"
#include "connection_view.h"
#include "connection_params.h"
#include "kernel_connection_propagate.h"

class Connection {
public:
    Connection(NeuronGroup& pre_group,
                NeuronGroup& post_group,
                std::vector<std::vector<float>>& weight_matrix,
                ConnectionParams params,
                GpuSimulationState& gpu_state);
    ~Connection() = default;

    ConnectionInitHostData initialize_data(const std::vector<std::vector<float>>& weight_matrix);
    void make_gpu_view();
    void upload_data(const ConnectionInitHostData& host_data);
    void propagate();
    
    // CPU only data getters
    uint32_t weight_matrix_idx() const { return params_.weight_matrix_idx; }
    
    // Offsets and sizes getters
    uint32_t pre_group_offset() const { return pre_group_offset_; }
    uint32_t post_group_offset() const { return post_group_offset_; }
    uint32_t pre_group_num_neurons() const { return pre_group_num_neurons_; }
    uint32_t post_group_num_neurons() const { return post_group_num_neurons_; }

private:
    NeuronGroup& pre_group_;
    NeuronGroup& post_group_;
    
    // CPU only data
    ConnectionParams params_;

    // Offsets and sizes
    uint32_t pre_group_offset_ = 0;
    uint32_t post_group_offset_ = 0;
    uint32_t pre_group_num_neurons_ = 0;
    uint32_t post_group_num_neurons_ = 0;

    // GPU data
    ConnectionView gpu_view_;
    GpuSimulationState& gpu_state_;
};

#endif  // CONNECTION_H