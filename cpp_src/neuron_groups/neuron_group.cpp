// neuron_group.cu or neuron_group.cpp (if it only launches kernels, compiling with nvcc is simplest)
#include "neuron_group.h"

NeuronGroup::NeuronGroup(std::vector<std::vector<float>> neuron_locations,
                        NeuronGroupParams params,
                        GpuSimulationState& gpu_state)
: gpu_state_(gpu_state),
params_(params) {
    if (neuron_locations.size() != params_.num_neurons) {
        throw std::invalid_argument("Size of neuron_locations does not match num_neurons");
    }
    neuron_locations_ = std::move(neuron_locations);
    
    NeuronGroupInitHostData initial_host_data = initialize_data();
    make_gpu_view();
    upload_data(initial_host_data);
}

NeuronGroupInitHostData NeuronGroup::initialize_data() {
    NeuronGroupInitHostData host_data;
    host_data.membrane_potentials.resize(params_.num_neurons, params_.v_rest);
    host_data.g_excitatory.resize(params_.num_neurons, 0.0f);
    host_data.g_inhibitory.resize(params_.num_neurons, 0.0f);
    host_data.i_stimulus.resize(params_.num_neurons, 0.0f);
    host_data.refractory_counter.resize(params_.num_neurons, 0);
    host_data.spikes.resize(params_.num_neurons, 0);
    host_data.background_g_excitatory_synaptic_conductances.resize(params_.num_neurons, 0.0f);
    host_data.background_g_inhibitory_synaptic_conductances.resize(params_.num_neurons, 0.0f);
    host_data.background_excitatory_postsynaptic_r_traces.resize(params_.num_neurons, 0.0f);
    host_data.background_excitatory_postsynaptic_d_traces.resize(params_.num_neurons, 0.0f);
    host_data.background_inhibitory_postsynaptic_r_traces.resize(params_.num_neurons, 0.0f);
    host_data.background_inhibitory_postsynaptic_d_traces.resize(params_.num_neurons, 0.0f);
    return host_data;
}

void NeuronGroup::make_gpu_view() {
    // CPU only data
    gpu_view_.timestep = params_.timestep;
    gpu_view_.g_leak = params_.g_leak;
    gpu_view_.membrane_capacitance = params_.membrane_capacitance;
    gpu_view_.depolarization_slope_factor = params_.depolarization_slope_factor;
    gpu_view_.v_effective_spike_threshold = params_.v_effective_spike_threshold;
    gpu_view_.v_excitatory_reversal = params_.v_excitatory_reversal;
    gpu_view_.v_inhibitory_reversal = params_.v_inhibitory_reversal;
    gpu_view_.v_threshold = params_.v_threshold;
    gpu_view_.v_reset = params_.v_reset;
    gpu_view_.v_rest = params_.v_rest;
    gpu_view_.refractory_steps = params_.refractory_steps;
    gpu_view_.is_inhibitory = params_.is_inhibitory;
    gpu_view_.mean_background_excitatory_spikes = params_.mean_background_excitatory_spikes;
    gpu_view_.mean_background_inhibitory_spikes = params_.mean_background_inhibitory_spikes;
    gpu_view_.background_excitatory_postsynaptic_r_trace_decay = params_.background_excitatory_postsynaptic_r_trace_decay;
    gpu_view_.background_excitatory_postsynaptic_d_trace_decay = params_.background_excitatory_postsynaptic_d_trace_decay;
    gpu_view_.background_inhibitory_postsynaptic_r_trace_decay = params_.background_inhibitory_postsynaptic_r_trace_decay;
    gpu_view_.background_inhibitory_postsynaptic_d_trace_decay = params_.background_inhibitory_postsynaptic_d_trace_decay;
    gpu_view_.background_excitatory_scaling_factor = params_.background_excitatory_scaling_factor;
    gpu_view_.background_inhibitory_scaling_factor = params_.background_inhibitory_scaling_factor;

    // Offsets and sizes
    gpu_view_.num_neurons = params_.num_neurons;
    gpu_view_.offset = params_.offset;

    // Pointers to GPU data
    gpu_view_.membrane_potentials = gpu_state_.d_membrane_potentials() + params_.offset;
    gpu_view_.g_excitatory = gpu_state_.d_g_excitatory() + params_.offset;
    gpu_view_.g_inhibitory = gpu_state_.d_g_inhibitory() + params_.offset;
    gpu_view_.i_stimulus = gpu_state_.d_i_stimulus() + params_.offset;
    gpu_view_.refractory_counter = gpu_state_.d_refractory_counter() + params_.offset;
    gpu_view_.spikes = gpu_state_.d_spikes() + params_.offset;
    gpu_view_.background_g_excitatory_synaptic_conductances = gpu_state_.d_background_g_excitatory_synaptic_conductances() + params_.offset;
    gpu_view_.background_g_inhibitory_synaptic_conductances = gpu_state_.d_background_g_inhibitory_synaptic_conductances() + params_.offset;
    gpu_view_.background_excitatory_postsynaptic_r_traces = gpu_state_.d_background_excitatory_postsynaptic_r_traces() + params_.offset;
    gpu_view_.background_excitatory_postsynaptic_d_traces = gpu_state_.d_background_excitatory_postsynaptic_d_traces() + params_.offset;
    gpu_view_.background_inhibitory_postsynaptic_r_traces = gpu_state_.d_background_inhibitory_postsynaptic_r_traces() + params_.offset;
    gpu_view_.background_inhibitory_postsynaptic_d_traces = gpu_state_.d_background_inhibitory_postsynaptic_d_traces() + params_.offset;
    gpu_view_.neuron_rng_states = gpu_state_.d_neuron_rng_states();
}

void NeuronGroup::upload_data(const NeuronGroupInitHostData& host_data) {
    gpu_state_.upload_neuron_group_view(gpu_view_, host_data);
}

void NeuronGroup::evolve() {
  kernel_neuron_group_evolve_launch(gpu_view_);
}

