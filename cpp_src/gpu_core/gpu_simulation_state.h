#ifndef GPU_SIMULATION_STATE_H
#define GPU_SIMULATION_STATE_H

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "buryn_definitions.h"
#include "neuron_group_view.h"
#include "connection_view.h"
#include "stimulus_view.h"
#include "plasticity_view.h"
#include "kernel_init_neuron_rng_states.h"

class GpuSimulationState {
public:
    explicit GpuSimulationState(const uint32_t total_simulation_length, 
                                const uint32_t num_excitatory_neurons,
                                const uint32_t num_inhibitory_neurons,
                                const uint32_t num_stimulus_events,
                                const uint32_t num_plasticities);
    ~GpuSimulationState();

    // Delete copy constructor and copy assignment operator to prevent accidental copying of GPU resources
    GpuSimulationState(const GpuSimulationState&) = delete;
    GpuSimulationState& operator=(const GpuSimulationState&) = delete;

    uint32_t total_allocated_bytes() const;
    std::string memory_report() const;

    // Getters for device pointers
    // Kernel Neuron Group
    float* d_membrane_potentials() const { return d_membrane_potentials_; }
    float* d_g_excitatory() const { return d_g_excitatory_; }
    float* d_g_inhibitory() const { return d_g_inhibitory_; }
    float* d_i_stimulus() const { return d_i_stimulus_; }
    uint32_t* d_refractory_counter() const { return d_refractory_counter_; }
    uint32_t* d_spikes() const { return d_spikes_; }
    float* d_background_g_excitatory_synaptic_conductances() const { return d_background_g_excitatory_synaptic_conductances_; }
    float* d_background_g_inhibitory_synaptic_conductances() const { return d_background_g_inhibitory_synaptic_conductances_; }
    float* d_background_excitatory_postsynaptic_r_traces() const { return d_background_excitatory_postsynaptic_r_traces_; }
    float* d_background_excitatory_postsynaptic_d_traces() const { return d_background_excitatory_postsynaptic_d_traces_; }
    float* d_background_inhibitory_postsynaptic_r_traces() const { return d_background_inhibitory_postsynaptic_r_traces_; }
    float* d_background_inhibitory_postsynaptic_d_traces() const { return d_background_inhibitory_postsynaptic_d_traces_; }
    void* d_neuron_rng_states() const { return d_neuron_rng_states_; }
    // Kernel Connection
    float* d_w_ee() const { return d_w_ee_; }
    float* d_w_ei() const { return d_w_ei_; }
    float* d_w_ie() const { return d_w_ie_; }
    float* d_w_ii() const { return d_w_ii_; }
    float* d_ee_postsynaptic_r_traces() const { return d_ee_postsynaptic_r_traces_; }
    float* d_ee_postsynaptic_d_traces() const { return d_ee_postsynaptic_d_traces_; }
    float* d_ee_summed_postsynaptic_conductances() const { return d_ee_summed_postsynaptic_conductances_; }
    float* d_ei_postsynaptic_r_traces() const { return d_ei_postsynaptic_r_traces_; }
    float* d_ei_postsynaptic_d_traces() const { return d_ei_postsynaptic_d_traces_; }
    float* d_ei_summed_postsynaptic_conductances() const { return d_ei_summed_postsynaptic_conductances_; }
    float* d_ie_postsynaptic_r_traces() const { return d_ie_postsynaptic_r_traces_; }
    float* d_ie_postsynaptic_d_traces() const { return d_ie_postsynaptic_d_traces_; }
    float* d_ie_summed_postsynaptic_conductances() const { return d_ie_summed_postsynaptic_conductances_; }
    float* d_ii_postsynaptic_r_traces() const { return d_ii_postsynaptic_r_traces_; }
    float* d_ii_postsynaptic_d_traces() const { return d_ii_postsynaptic_d_traces_; }
    float* d_ii_summed_postsynaptic_conductances() const { return d_ii_summed_postsynaptic_conductances_; }
    void* cublas_handle() const { return cublas_handle_; }
    // Kernel Stimulus
    uint32_t* d_timestep_event_count() const { return d_timestep_event_count_; }
    uint32_t* d_stimulus_event_neuron_ids() const { return d_stimulus_event_neuron_ids_; }
    uint32_t* d_stimulus_event_post_group_offsets() const { return d_stimulus_event_post_group_offsets_; }
    float* d_stimulus_event_amplitude_changes() const { return d_stimulus_event_amplitude_changes_; }
    // Kernel Plasticity
    float* d_pre_traces() const { return d_pre_traces_; }
    float* d_post_traces() const { return d_post_traces_; }
    float* d_triplet_pre_traces() const { return d_triplet_pre_traces_; }
    float* d_triplet_post_traces() const { return d_triplet_post_traces_; }
    float* d_pre_weight_sums() const { return d_pre_weight_sums_; }
    // Kernel Monitors
    uint32_t* d_spike_history() const { return d_spike_history_; }
    float* d_membrane_potential_history() const { return d_membrane_potential_history_; }

    // Functions for uploading data to GPU gpu_views
    void upload_neuron_group_view(const NeuronGroupView& gpu_view, const NeuronGroupInitHostData& host_data);
    void upload_connection_view(const ConnectionView& gpu_view, const ConnectionInitHostData& host_data);
    void upload_stimulus_view(const StimulusView& gpu_view, const StimulusInitHostData& host_data);
    void upload_plasticity_view(const PlasticityView& gpu_view, const PlasticityInitHostData& host_data);

    // Functions for downloading data from GPU for monitors
    std::vector<uint32_t> download_and_flush_spike_history(const uint32_t offset, const uint32_t num_neurons, const uint32_t num_timesteps);
    std::vector<float> download_and_flush_membrane_potential_history(const uint32_t offset, const uint32_t num_neurons, const uint32_t num_timesteps);
    std::vector<float> download_weight_matrix(const uint32_t weight_matrix_idx) const;
    
    // Utility functions
    std::string lookup_weight_matrix_name(uint32_t weight_matrix_idx) const;

    // General getters
    uint32_t total_simulation_length() const { return total_simulation_length_; }
    uint32_t num_excitatory_neurons() const { return num_excitatory_neurons_; }
    uint32_t num_inhibitory_neurons() const { return num_inhibitory_neurons_; }
    uint32_t total_neurons() const { return total_neurons_; }
    uint32_t num_stimulus_events() const { return num_stimulus_events_; }
    uint32_t num_plasticities() const { return num_plasticities_; }
    
private:
    void require_upload_bounds(const std::string& field_name, const uint32_t offset, const uint32_t value_count);
    void require_nonnegative_weights(const std::string& field_name, const std::vector<float>& weights);

    void initialize_device_buffers();
    void initialize_cublas_handle();
    void initialize_rng_states();
    void initialize_monitor_histories();
    void free_device_buffers();


    uint32_t total_simulation_length_ = 0;
    uint32_t num_excitatory_neurons_ = 0;
    uint32_t num_inhibitory_neurons_ = 0;
    uint32_t total_neurons_ = 0;
    uint32_t num_stimulus_events_ = 0;
    uint32_t num_plasticities_ = 0;
    std::vector<std::string> weight_matrix_lookup_ ;

    // Kernel Neuron Group
    float* d_membrane_potentials_ = nullptr;
    float* d_g_excitatory_ = nullptr;
    float* d_g_inhibitory_ = nullptr;
    float* d_i_stimulus_ = nullptr;
    uint32_t* d_refractory_counter_ = nullptr;
    uint32_t* d_spikes_ = nullptr;
    float* d_background_g_excitatory_synaptic_conductances_ = nullptr;
    float* d_background_g_inhibitory_synaptic_conductances_ = nullptr;
    float* d_background_excitatory_postsynaptic_r_traces_ = nullptr;
    float* d_background_excitatory_postsynaptic_d_traces_ = nullptr;
    float* d_background_inhibitory_postsynaptic_r_traces_ = nullptr;
    float* d_background_inhibitory_postsynaptic_d_traces_ = nullptr;
    void* d_neuron_rng_states_ = nullptr;
    // Kernel Connection
    float* d_w_ee_ = nullptr;
    float* d_w_ei_ = nullptr;
    float* d_w_ie_ = nullptr;
    float* d_w_ii_ = nullptr;
    float* d_ee_postsynaptic_r_traces_ = nullptr;
    float* d_ee_postsynaptic_d_traces_ = nullptr;
    float* d_ee_summed_postsynaptic_conductances_ = nullptr;
    float* d_ei_postsynaptic_r_traces_ = nullptr;
    float* d_ei_postsynaptic_d_traces_ = nullptr;
    float* d_ei_summed_postsynaptic_conductances_ = nullptr;
    float* d_ie_postsynaptic_r_traces_ = nullptr;
    float* d_ie_postsynaptic_d_traces_ = nullptr;
    float* d_ie_summed_postsynaptic_conductances_ = nullptr;
    float* d_ii_postsynaptic_r_traces_ = nullptr;
    float* d_ii_postsynaptic_d_traces_ = nullptr;
    float* d_ii_summed_postsynaptic_conductances_ = nullptr;
    void* cublas_handle_ = nullptr;
    // Kernel Stimulus
    uint32_t* d_timestep_event_count_ = nullptr;
    uint32_t* d_stimulus_event_neuron_ids_ = nullptr;
    uint32_t* d_stimulus_event_post_group_offsets_ = nullptr;
    float* d_stimulus_event_amplitude_changes_ = nullptr;
    // Kernel Plasticity
    float* d_pre_traces_ = nullptr;
    float* d_post_traces_ = nullptr;
    float* d_triplet_pre_traces_ = nullptr;
    float* d_triplet_post_traces_ = nullptr;
    float* d_pre_weight_sums_ = nullptr;
    // Kernel Monitors
    uint32_t* d_spike_history_ = nullptr;
    float* d_membrane_potential_history_ = nullptr;

};

#endif  // GPU_SIMULATION_STATE_H