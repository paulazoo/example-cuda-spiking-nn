#ifndef NEURON_GROUP_VIEW_H
#define NEURON_GROUP_VIEW_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct NeuronGroupView {
    // Scalar params
    float timestep = 1e-4f;
    float g_leak = 10.0f;
    float membrane_capacitance = 0.1f;
    float depolarization_slope_factor = 2.0f;
    float v_effective_spike_threshold = -50.0f;
    float v_excitatory_reversal = 0.0f;
    float v_inhibitory_reversal = -80.0f;
    float v_threshold = -45.0f;
    float v_reset = -55.0f;
    float v_rest = -65.0f;
    uint32_t refractory_steps = 0;
    bool is_inhibitory = false;
    float mean_background_excitatory_spikes = 0.0f;
    float mean_background_inhibitory_spikes = 0.0f;
    float background_excitatory_postsynaptic_r_trace_decay = 0.0f;
    float background_excitatory_postsynaptic_d_trace_decay = 0.0f;
    float background_inhibitory_postsynaptic_r_trace_decay = 0.0f;
    float background_inhibitory_postsynaptic_d_trace_decay = 0.0f;
    float background_excitatory_scaling_factor = 0.0f;
    float background_inhibitory_scaling_factor = 0.0f;

    // Offsets and sizes
    uint32_t num_neurons = 0;
    uint32_t offset = 0;

    // Pointers to GPU data
    float* membrane_potentials = nullptr;
    float* g_excitatory = nullptr;
    float* g_inhibitory = nullptr;
    float* i_stimulus = nullptr;
    uint32_t* refractory_counter = nullptr;
    uint32_t* spikes = nullptr;
    float* background_g_excitatory_synaptic_conductances = nullptr;
    float* background_g_inhibitory_synaptic_conductances = nullptr;
    float* background_excitatory_postsynaptic_r_traces = nullptr;
    float* background_excitatory_postsynaptic_d_traces = nullptr;
    float* background_inhibitory_postsynaptic_r_traces = nullptr;
    float* background_inhibitory_postsynaptic_d_traces = nullptr;
    void* neuron_rng_states = nullptr;
};

struct NeuronGroupInitHostData {
    std::vector<float> membrane_potentials;
    std::vector<float> g_excitatory;
    std::vector<float> g_inhibitory;
    std::vector<float> i_stimulus;
    std::vector<uint32_t> refractory_counter;
    std::vector<uint32_t> spikes;
    std::vector<float> background_g_excitatory_synaptic_conductances;
    std::vector<float> background_g_inhibitory_synaptic_conductances;
    std::vector<float> background_excitatory_postsynaptic_r_traces;
    std::vector<float> background_excitatory_postsynaptic_d_traces;
    std::vector<float> background_inhibitory_postsynaptic_r_traces;
    std::vector<float> background_inhibitory_postsynaptic_d_traces;
};

#endif  // NEURON_GROUP_VIEW_H
