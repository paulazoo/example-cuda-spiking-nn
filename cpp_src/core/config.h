#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>
#include <string>
#include <vector>

#include "generate_weights.h"

struct Config {
    uint32_t simulation_steps = 10000;
    std::string output_directory = "../data/buryn_data/buryn_outputs_";
    std::string description;
    int rng_seed = 314;
    float timestep = 1e-4;
    
    // Neuron group settings
    uint32_t num_excitatory_neurons = 100;
    uint32_t num_inhibitory_neurons = 100;
    float g_leak = 0.05e-6;                      // S
    float membrane_capacitance = 1.0e-9;         // F
    float depolarization_slope_factor = 0.0065625; // V (ΔT)
    float v_effective_spike_threshold = -0.0606250; // V (VT)
    float v_excitatory_reversal = 0.0;           // V
    float v_inhibitory_reversal = -0.08;         // V
    float v_threshold = -0.04;                   // V
    float v_reset = -0.06;                       // V
    float v_rest = -0.07;                        // V
    uint32_t refractory_steps = 50;            // steps

    float group_e_mean_background_excitatory_spikes = 0.0; // Hz * timestep
    float group_e_mean_background_inhibitory_spikes = 0.0; // Hz * timestep
    float group_e_background_excitatory_postsynaptic_r_trace_decay = 0.0f; // per step
    float group_e_background_excitatory_postsynaptic_d_trace_decay = 0.0f; // per step
    float group_e_background_inhibitory_postsynaptic_r_trace_decay = 0.0f; // per step
    float group_e_background_inhibitory_postsynaptic_d_trace_decay = 0.0f; // per step
    float group_e_background_excitatory_scaling_factor = 0.0f;
    float group_e_background_inhibitory_scaling_factor = 0.0f;
    float group_i_mean_background_excitatory_spikes = 0.0; // Hz * timestep
    float group_i_mean_background_inhibitory_spikes = 0.0; // Hz * timestep
    float group_i_background_excitatory_postsynaptic_r_trace_decay = 0.0f; // per step
    float group_i_background_excitatory_postsynaptic_d_trace_decay = 0.0f; // per step
    float group_i_background_inhibitory_postsynaptic_r_trace_decay = 0.0f; // per step
    float group_i_background_inhibitory_postsynaptic_d_trace_decay = 0.0f; // per step
    float group_i_background_excitatory_scaling_factor = 0.0f;
    float group_i_background_inhibitory_scaling_factor = 0.0f;

    // Neuron location settings
    std::string excitatory_locations_file = "../data/buryn_data/buryn_outputs_/locations/group_e_locations.txt";
    std::string inhibitory_locations_file = "../data/buryn_data/buryn_outputs_/locations/group_i_locations.txt";
    float excitatory_location_spacing = 1.0;
    float inhibitory_location_spacing = 2.0;
    uint32_t excitatory_side_points = 10;
    uint32_t inhibitory_side_points = 10;
    float side_length = 10.0;

    // Weight generation settings
    float e_max_weight_value = 0.2235e-9;
    float i_max_weight_value = 0.0578e-9;
    float max_connection_distance = 3.0;
    float connection_spread_scale = 1.0;
    float connection_probability_multiplier = 1.0;
    generate_weights::WeightGenerationParams weight_generation_params_e_to_e = {
        10.0,
        10.0,
        3.0,
        1.0,
        0.2235e-9,
        false,
        true,
        1.0,
        true,
    };
    generate_weights::WeightGenerationParams weight_generation_params_e_to_i = {
        10.0,
        10.0,
        3.0,
        1.0,
        0.2235e-9,
        false,
        true,
        1.0,
        false,
    };
    generate_weights::WeightGenerationParams weight_generation_params_i_to_e = {
        10.0,
        10.0,
        3.0,
        1.0,
        0.0578e-9,
        false,
        false,
        1.0,
        false,
    };
    generate_weights::WeightGenerationParams weight_generation_params_i_to_i = {
        10.0,
        10.0,
        3.0,
        1.0,
        0.0578e-9,
        false,
        false,
        1.0,
        false,
    };

    // Connection settings
    std::string weights_e_to_i_filename = "../data/buryn_data/buryn_outputs_/weights/weights_e_to_i.txt";
    std::string weights_i_to_e_filename = "../data/buryn_data/buryn_outputs_/weights/weights_i_to_e.txt";
    std::string weights_e_to_e_filename = "../data/buryn_data/buryn_outputs_/weights/weights_e_to_e.txt";
    std::string weights_i_to_i_filename = "../data/buryn_data/buryn_outputs_/weights/weights_i_to_i.txt";
    float tau_decay_e = 2e-3;
    float tau_rise_e  = 0.3e-3;
    float tau_decay_i = 3e-3;
    float tau_rise_i  = 0.3e-3;
    float postsynaptic_kernel_normalization_value = 1.0;

    // Plasticity settings
    bool e_e_plasticity_enabled = true;
    float e_e_plasticity_ltp_A = 1.0e-12;
    float e_e_plasticity_ltd_A = 1.0e-12;
    float e_e_plasticity_ltp_B = 1.0e-12;
    float e_e_plasticity_ltd_B = 1.0e-12;
    float e_e_plasticity_tau_ltp = 0.02f;
    float e_e_plasticity_tau_ltd = 0.02f;
    float e_e_plasticity_triplet_tau_ltp = 0.02f;
    float e_e_plasticity_triplet_tau_ltd = 0.02f;
    float e_e_plasticity_weight_min = 0.0f;
    float e_e_plasticity_weight_max = 0.2235e-9;
    bool i_e_plasticity_enabled = true;
    float i_e_plasticity_ltp_A = 1.0e-12;
    float i_e_plasticity_ltd_A = 1.0e-12;
    float i_e_plasticity_ltd_B = 1.0e-12;
    float i_e_plasticity_tau_ltp = 0.02f;
    float i_e_plasticity_tau_ltd = 0.02f;
    float i_e_plasticity_weight_min = 0.0f;
    float i_e_plasticity_weight_max = 0.2235e-9;

    // Stimulus settings
    std::vector<float> stimulus_input_values = {0.25e-9}; // under 1 nA is good
    std::string stimulus_schedules_folder = "../data/buryn_data/buryn_outputs_/stimulus_schedules";
    std::vector<uint32_t> stimulus_selected_input_ids = {0,1,2,3,4,5,6,7,8,9};
    std::vector<uint32_t> stimulus_selected_input_post_group_offsets = std::vector<uint32_t>(10, 0);
    std::vector<uint32_t> stimulus_start_clock_steps = {5000};
    std::vector<uint32_t> stimulus_end_clock_steps = {6000};

    // Monitor settings
    std::string e_spike_recording_file = "../data/buryn_data/buryn_outputs_/recordings/spikes_e.txt";
    std::string i_spike_recording_file = "../data/buryn_data/buryn_outputs_/recordings/spikes_i.txt";
    std::string voltage_recording_file = "../data/buryn_data/buryn_outputs_/recordings/voltages_step0.txt";
    std::vector<uint32_t> voltage_recording_neuron_ids = {10, 426};
    std::string e_e_weight_recording_file = "../data/buryn_data/buryn_outputs_/weights/weights_e_to_e_recordings.txt";
};

Config load_config_json(const std::string& path);

#endif  // CONFIG_H
