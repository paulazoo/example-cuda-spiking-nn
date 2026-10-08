#ifndef NEURON_GROUP_H
#define NEURON_GROUP_H

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "buryn_definitions.h"

#include "gpu_simulation_state.h"
#include "kernel_neuron_group_evolve.h"

#include "neuron_group_view.h"
#include "neuron_group_params.h"

class NeuronGroup {
public:
    explicit NeuronGroup(std::vector<std::vector<float>> neuron_locations,
                        NeuronGroupParams params,
                        GpuSimulationState& gpu_state);
    ~NeuronGroup() = default;
    NeuronGroupInitHostData initialize_data();
    void make_gpu_view();
    void upload_data(const NeuronGroupInitHostData& host_data);
    void evolve();
    
    // Getters for CPU data
    const std::vector<std::vector<float>>& neuron_locations() const { return neuron_locations_; }
    bool is_inhibitory() const { return params_.is_inhibitory; }
    uint32_t num_neurons() const { return params_.num_neurons; }
    uint32_t offset() const { return params_.offset; }

protected:
    // CPU only data
    std::vector<std::vector<float>> neuron_locations_;
    NeuronGroupParams params_;

    // GPU data
    NeuronGroupView gpu_view_;
    GpuSimulationState& gpu_state_;
};

#endif  // NEURON_GROUP_H
