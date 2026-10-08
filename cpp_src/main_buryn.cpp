#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

#include "buryn_definitions.h"
#include "config.h"
#include "system.h"

// Generation
#include "generate_location.h"
#include "generate_weights.h"
#include "generate_stimulus_schedules.h"

// Objects
#include "neuron_group.h"
#include "stimulus.h"
#include "stdp_plasticity.h"
#include "istdp_plasticity.h"
#include "triplet_plasticity.h"
#include "spike_recorder.h"
#include "voltage_recorder.h"
#include "weight_recorder.h"


void clean_up(std::string output_directory, const std::string& description, const std::string& config_path) {
    std::filesystem::path output_dir = output_directory;

    // Remove contents of output_directory (equivalent to: rm -rf output_directory/*)
    if (std::filesystem::exists(output_dir) && std::filesystem::is_directory(output_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
            std::filesystem::remove_all(entry.path());
        }
    } else {
        // Create top-level directory if it doesn't exist
        std::filesystem::create_directory(output_dir);
    }

    std::ofstream description_file(output_dir / "description.txt");
    if (!description_file) {
        throw std::runtime_error("Failed to write description file in output directory");
    }
    description_file << description;

    // Recreate expected subdirectories
    const std::vector<std::filesystem::path> subdirs = {
        "stimulus_schedules",
        "recordings",
        "locations",
        "videos",
        "weights",
        "plots"
    };

    for (const auto& subdir : subdirs) {
        std::filesystem::create_directory(output_dir / subdir);
    }

    // Copy config JSON into output directory
    std::filesystem::path config_file(config_path);
    if (!std::filesystem::exists(config_file)) {
        throw std::runtime_error("Config file does not exist: " + config_path);
    }
    std::filesystem::copy_file(
        config_file,
        output_dir / config_file.filename(),
        std::filesystem::copy_options::overwrite_existing
    );
}

void validate_plasticity_params(const PlasticityParams& params,
                                const bool requires_triplet,
                                const std::string& plasticity_name) {
    if (params.pre_trace_decay < 0.0f || params.pre_trace_decay > 1.0f) {
        throw std::runtime_error(plasticity_name + ": pre_trace_decay must be in [0, 1]");
    }
    if (params.post_trace_decay < 0.0f || params.post_trace_decay > 1.0f) {
        throw std::runtime_error(plasticity_name + ": post_trace_decay must be in [0, 1]");
    }
    if (requires_triplet) {
        if (params.triplet_pre_trace_decay < 0.0f || params.triplet_pre_trace_decay > 1.0f) {
            throw std::runtime_error(plasticity_name + ": triplet_pre_trace_decay must be in [0, 1]");
        }
        if (params.triplet_post_trace_decay < 0.0f || params.triplet_post_trace_decay > 1.0f) {
            throw std::runtime_error(plasticity_name + ": triplet_post_trace_decay must be in [0, 1]");
        }
    }
    if (params.weight_min > params.weight_max) {
        throw std::runtime_error(plasticity_name + ": weight_min must be <= weight_max");
    }
}

int main(int argc, char* argv[]) {
    Config buryn_config;
    const std::string config_path = (argc > 1) ? argv[1] : "config.json";
    buryn_config = load_config_json(config_path);
    clean_up(buryn_config.output_directory, buryn_config.description, config_path);
    buryn::rng_seed = buryn_config.rng_seed;
    buryn::timestep = buryn_config.timestep;

    System system;

    const uint32_t num_enabled_plasticities = static_cast<uint32_t>(buryn_config.e_e_plasticity_enabled) + static_cast<uint32_t>(buryn_config.i_e_plasticity_enabled);
    GpuSimulationState gpu_state(buryn_config.simulation_steps,
                                buryn_config.num_excitatory_neurons,
                                buryn_config.num_inhibitory_neurons,
                                buryn_config.stimulus_selected_input_ids.size() * buryn_config.stimulus_start_clock_steps.size() * 2, // *2 because each stimulus has a start and end step
                                num_enabled_plasticities);
    std::cout << ceil((5.0 * buryn_config.tau_decay_e) / buryn::timestep) << "\n";
    std::cout << gpu_state.memory_report() << "\n";

    // Generate locations for neurons
    std::cout << "Generating neuron locations...\n";
    std::vector<std::vector<float>> group_e_locations = generate_location::generate_2d_grid_locations(buryn_config.excitatory_locations_file, buryn_config.num_excitatory_neurons, buryn_config.excitatory_location_spacing, buryn_config.excitatory_side_points, buryn_config.excitatory_side_points);
    std::vector<std::vector<float>> group_i_locations = generate_location::generate_2d_grid_locations(buryn_config.inhibitory_locations_file, buryn_config.num_inhibitory_neurons, buryn_config.inhibitory_location_spacing, buryn_config.inhibitory_side_points, buryn_config.inhibitory_side_points);
    std::cout << "Neuron locations generated.\n";

    // Generate weight files
    std::cout << "Generating weight files...\n";
    const generate_weights::WeightGenerationParams& weight_gen_params_e_to_e = buryn_config.weight_generation_params_e_to_e;
    const generate_weights::WeightGenerationParams& weight_gen_params_e_to_i = buryn_config.weight_generation_params_e_to_i;
    const generate_weights::WeightGenerationParams& weight_gen_params_i_to_e = buryn_config.weight_generation_params_i_to_e;
    const generate_weights::WeightGenerationParams& weight_gen_params_i_to_i = buryn_config.weight_generation_params_i_to_i;
    std::vector<std::vector<float>> weight_matrix_e_to_e = generate_weights::distance_2d_weights_file(buryn_config.weights_e_to_e_filename, group_e_locations, group_e_locations, weight_gen_params_e_to_e);
    std::vector<std::vector<float>> weight_matrix_e_to_i = generate_weights::distance_2d_weights_file(buryn_config.weights_e_to_i_filename, group_e_locations, group_i_locations, weight_gen_params_e_to_i);
    std::vector<std::vector<float>> weight_matrix_i_to_e = generate_weights::distance_2d_weights_file(buryn_config.weights_i_to_e_filename, group_i_locations, group_e_locations, weight_gen_params_i_to_e);
    std::vector<std::vector<float>> weight_matrix_i_to_i = generate_weights::distance_2d_weights_file(buryn_config.weights_i_to_i_filename, group_i_locations, group_i_locations, weight_gen_params_i_to_i);
    std::cout << "Weight files generated.\n";

    // Generate stimulus schedules
    std::cout << "Generating stimulus schedules...\n";
    generate_stimulus_schedules::StimulusGenerationParams stimulus_gen_params{};
    stimulus_gen_params.max_timestep = buryn_config.simulation_steps;
    stimulus_gen_params.selected_input_ids = buryn_config.stimulus_selected_input_ids;
    stimulus_gen_params.selected_input_post_group_offsets = buryn_config.stimulus_selected_input_post_group_offsets;
    stimulus_gen_params.start_clock_steps = buryn_config.stimulus_start_clock_steps;
    stimulus_gen_params.end_clock_steps = buryn_config.stimulus_end_clock_steps;
    stimulus_gen_params.stimulus_input_values = buryn_config.stimulus_input_values;
    generate_stimulus_schedules::StimulusGenerationResult stimulus_schedules = generate_stimulus_schedules::partial_group_multiple_square_waves(
        buryn_config.stimulus_schedules_folder,
        stimulus_gen_params);
    std::cout << "Stimulus schedules generated.\n";

    // Make neuron groups
    std::cout << "Creating neuron groups...\n";
    NeuronGroupParams group_e_params{};
    group_e_params.timestep = buryn::timestep;
    group_e_params.g_leak = buryn_config.g_leak;
    group_e_params.membrane_capacitance = buryn_config.membrane_capacitance;
    group_e_params.depolarization_slope_factor = buryn_config.depolarization_slope_factor;
    group_e_params.v_effective_spike_threshold = buryn_config.v_effective_spike_threshold;
    group_e_params.v_excitatory_reversal = buryn_config.v_excitatory_reversal;
    group_e_params.v_inhibitory_reversal = buryn_config.v_inhibitory_reversal;
    group_e_params.v_threshold = buryn_config.v_threshold;
    group_e_params.v_reset = buryn_config.v_reset;
    group_e_params.v_rest = buryn_config.v_rest;
    group_e_params.refractory_steps = buryn_config.refractory_steps;
    group_e_params.offset = 0;
    group_e_params.num_neurons = buryn_config.num_excitatory_neurons;
    group_e_params.is_inhibitory = false;
    group_e_params.mean_background_excitatory_spikes = buryn_config.group_e_mean_background_excitatory_spikes;
    group_e_params.mean_background_inhibitory_spikes = buryn_config.group_e_mean_background_inhibitory_spikes;
    group_e_params.background_excitatory_postsynaptic_r_trace_decay = buryn_config.group_e_background_excitatory_postsynaptic_r_trace_decay;
    group_e_params.background_excitatory_postsynaptic_d_trace_decay = buryn_config.group_e_background_excitatory_postsynaptic_d_trace_decay;
    group_e_params.background_inhibitory_postsynaptic_r_trace_decay = buryn_config.group_e_background_inhibitory_postsynaptic_r_trace_decay;
    group_e_params.background_inhibitory_postsynaptic_d_trace_decay = buryn_config.group_e_background_inhibitory_postsynaptic_d_trace_decay;
    group_e_params.background_excitatory_scaling_factor = buryn_config.group_e_background_excitatory_scaling_factor;
    group_e_params.background_inhibitory_scaling_factor = buryn_config.group_e_background_inhibitory_scaling_factor;
    NeuronGroupParams group_i_params{};
    group_i_params.timestep = buryn::timestep;
    group_i_params.g_leak = buryn_config.g_leak;
    group_i_params.membrane_capacitance = buryn_config.membrane_capacitance;
    group_i_params.depolarization_slope_factor = buryn_config.depolarization_slope_factor;
    group_i_params.v_effective_spike_threshold = buryn_config.v_effective_spike_threshold;
    group_i_params.v_excitatory_reversal = buryn_config.v_excitatory_reversal;
    group_i_params.v_inhibitory_reversal = buryn_config.v_inhibitory_reversal;
    group_i_params.v_threshold = buryn_config.v_threshold;
    group_i_params.v_reset = buryn_config.v_reset;
    group_i_params.v_rest = buryn_config.v_rest;
    group_i_params.refractory_steps = buryn_config.refractory_steps;
    group_i_params.offset = buryn_config.num_excitatory_neurons;
    group_i_params.num_neurons = buryn_config.num_inhibitory_neurons;
    group_i_params.is_inhibitory = true;
    group_i_params.mean_background_excitatory_spikes = buryn_config.group_i_mean_background_excitatory_spikes;
    group_i_params.mean_background_inhibitory_spikes = buryn_config.group_i_mean_background_inhibitory_spikes;
    group_i_params.background_excitatory_postsynaptic_r_trace_decay = buryn_config.group_i_background_excitatory_postsynaptic_r_trace_decay;
    group_i_params.background_excitatory_postsynaptic_d_trace_decay = buryn_config.group_i_background_excitatory_postsynaptic_d_trace_decay;
    group_i_params.background_inhibitory_postsynaptic_r_trace_decay = buryn_config.group_i_background_inhibitory_postsynaptic_r_trace_decay;
    group_i_params.background_inhibitory_postsynaptic_d_trace_decay = buryn_config.group_i_background_inhibitory_postsynaptic_d_trace_decay;
    group_i_params.background_excitatory_scaling_factor = buryn_config.group_i_background_excitatory_scaling_factor;
    group_i_params.background_inhibitory_scaling_factor = buryn_config.group_i_background_inhibitory_scaling_factor;
    
    std::unique_ptr<NeuronGroup> group_e = std::make_unique<NeuronGroup>(
        group_e_locations,
        group_e_params,
        gpu_state
    );
    std::unique_ptr<NeuronGroup> group_i = std::make_unique<NeuronGroup>(
        group_i_locations,
        group_i_params,
        gpu_state
    );
    NeuronGroup& group_e_ref = *group_e;
    NeuronGroup& group_i_ref = *group_i;
    system.add_neuron_group(std::move(group_e));
    system.add_neuron_group(std::move(group_i));
    std::cout << "Neuron groups created.\n";


    // Make connections
    std::cout << "Creating connections...\n";
    ConnectionParams e_to_e_params{};
    e_to_e_params.timestep = buryn::timestep;
    e_to_e_params.decay_time = buryn_config.tau_decay_e;
    e_to_e_params.rise_time = buryn_config.tau_rise_e;
    e_to_e_params.postsynaptic_kernel_normalization_value = buryn_config.postsynaptic_kernel_normalization_value;
    e_to_e_params.weight_matrix_idx = 0;
    ConnectionParams e_to_i_params{};
    e_to_i_params.timestep = buryn::timestep;
    e_to_i_params.decay_time = buryn_config.tau_decay_e;
    e_to_i_params.rise_time = buryn_config.tau_rise_e;
    e_to_i_params.postsynaptic_kernel_normalization_value = buryn_config.postsynaptic_kernel_normalization_value;
    e_to_i_params.weight_matrix_idx = 1;
    ConnectionParams i_to_e_params{};
    i_to_e_params.timestep = buryn::timestep;
    i_to_e_params.decay_time = buryn_config.tau_decay_i;
    i_to_e_params.rise_time = buryn_config.tau_rise_i;
    i_to_e_params.postsynaptic_kernel_normalization_value = buryn_config.postsynaptic_kernel_normalization_value;
    i_to_e_params.weight_matrix_idx = 2;
    ConnectionParams i_to_i_params{};
    i_to_i_params.timestep = buryn::timestep;
    i_to_i_params.decay_time = buryn_config.tau_decay_i;
    i_to_i_params.rise_time = buryn_config.tau_rise_i;
    i_to_i_params.postsynaptic_kernel_normalization_value = buryn_config.postsynaptic_kernel_normalization_value;
    i_to_i_params.weight_matrix_idx = 3;
    // E -> E (excitatory pre-group)
    auto e_e_connection = std::make_unique<Connection>(group_e_ref,
                                                    group_e_ref,
                                                    weight_matrix_e_to_e,
                                                    e_to_e_params,
                                                    gpu_state);
    Connection& e_e_connection_ref = *e_e_connection;
    system.add_connection(std::move(e_e_connection));
    // E -> I (excitatory pre-group)
    auto e_i_connection = std::make_unique<Connection>(group_e_ref,
                                                    group_i_ref,
                                                    weight_matrix_e_to_i,
                                                    e_to_i_params,
                                                    gpu_state);
    Connection& e_i_connection_ref = *e_i_connection;
    system.add_connection(std::move(e_i_connection));
    // I -> E (inhibitory pre-group)
    auto i_e_connection = std::make_unique<Connection>(group_i_ref,
                                                    group_e_ref,
                                                    weight_matrix_i_to_e,
                                                    i_to_e_params,
                                                    gpu_state);
    Connection& i_e_connection_ref = *i_e_connection;
    system.add_connection(std::move(i_e_connection));
    // I -> I (inhibitory pre-group)
    auto i_i_connection = std::make_unique<Connection>(group_i_ref,
                                                    group_i_ref,
                                                    weight_matrix_i_to_i,
                                                    i_to_i_params,
                                                    gpu_state);
    Connection& i_i_connection_ref = *i_i_connection;
    system.add_connection(std::move(i_i_connection));
    std::cout << "Connections created.\n";

    // Make stimuli
    std::cout << "Creating stimuli...\n";
    system.add_stimulus(std::make_unique<Stimulus>(stimulus_schedules.timestep_event_counts,
                                                stimulus_schedules.neuron_ids,
                                                stimulus_schedules.post_group_offsets,
                                                stimulus_schedules.amplitude_changes,
                                                gpu_state));
    std::cout << "Stimuli generated.\n";

    // Add plasticities
    std::cout << "Adding plasticities...\n";
    uint32_t plasticity_offset = 0;
    if (buryn_config.e_e_plasticity_enabled) {
        PlasticityParams e_e_stdp_params{};
        e_e_stdp_params.ltp_A = buryn_config.e_e_plasticity_ltp_A;
        e_e_stdp_params.ltd_A = buryn_config.e_e_plasticity_ltd_A;
        e_e_stdp_params.ltp_B = buryn_config.e_e_plasticity_ltp_B;
        e_e_stdp_params.ltd_B = buryn_config.e_e_plasticity_ltd_B;
        e_e_stdp_params.weight_min = buryn_config.e_e_plasticity_weight_min;
        e_e_stdp_params.weight_max = buryn_config.e_e_plasticity_weight_max;
        e_e_stdp_params.pre_trace_decay = std::exp(-1.0f * buryn::timestep / buryn_config.e_e_plasticity_tau_ltp);
        e_e_stdp_params.post_trace_decay = std::exp(-1.0f * buryn::timestep / buryn_config.e_e_plasticity_tau_ltd);
        e_e_stdp_params.triplet_pre_trace_decay = std::exp(-1.0f * buryn::timestep / buryn_config.e_e_plasticity_triplet_tau_ltp);
        e_e_stdp_params.triplet_post_trace_decay = std::exp(-1.0f * buryn::timestep / buryn_config.e_e_plasticity_triplet_tau_ltd);
        e_e_stdp_params.plasticity_offset = plasticity_offset;
        system.add_plasticity(std::make_unique<TripletPlasticity>(e_e_connection_ref,
                                                                e_e_stdp_params,
                                                                gpu_state));
        validate_plasticity_params(e_e_stdp_params, true, "e_e excitatory Triplet plasticity");
        plasticity_offset += 1;
    }
    if (buryn_config.i_e_plasticity_enabled) {
        PlasticityParams i_e_stdp_params{};
        i_e_stdp_params.ltp_A = buryn_config.i_e_plasticity_ltp_A;
        i_e_stdp_params.ltd_A = buryn_config.i_e_plasticity_ltd_A;
        i_e_stdp_params.ltd_B = buryn_config.i_e_plasticity_ltd_B;
        i_e_stdp_params.weight_min = buryn_config.i_e_plasticity_weight_min;
        i_e_stdp_params.weight_max = buryn_config.i_e_plasticity_weight_max;
        i_e_stdp_params.pre_trace_decay = std::exp(-1.0f * buryn::timestep / buryn_config.i_e_plasticity_tau_ltp);
        i_e_stdp_params.post_trace_decay = std::exp(-1.0f * buryn::timestep / buryn_config.i_e_plasticity_tau_ltd);
        i_e_stdp_params.plasticity_offset = plasticity_offset;
        system.add_plasticity(std::make_unique<IstdpPlasticity>(i_e_connection_ref,
                                                                i_e_stdp_params,
                                                                gpu_state));
        validate_plasticity_params(i_e_stdp_params, false, "i_e inhibitory Istdp plasticity");
        plasticity_offset += 1;
    }
    std::cout << "Plasticities added.\n";


    // Set up spike recorders for both groups
    std::cout << "Setting up recorders...\n";
    auto group_e_recorder = std::make_unique<SpikeRecorder>(group_e_ref, buryn_config.e_spike_recording_file, buryn::recording_interval_steps, gpu_state);
    auto group_i_recorder = std::make_unique<SpikeRecorder>(group_i_ref, buryn_config.i_spike_recording_file, buryn::recording_interval_steps, gpu_state);
    auto group_e_voltage_recorder = std::make_unique<VoltageRecorder>(group_e_ref, buryn_config.voltage_recording_file, buryn_config.voltage_recording_neuron_ids, buryn::recording_interval_steps, gpu_state);
    auto e_e_weight_recorder = std::make_unique<WeightRecorder>(
        e_e_connection_ref,
        buryn_config.e_e_weight_recording_file,
        gpu_state
    );
    system.add_monitor(std::move(group_e_recorder));
    system.add_monitor(std::move(group_i_recorder));
    system.add_monitor(std::move(group_e_voltage_recorder));
    system.add_monitor(std::move(e_e_weight_recorder));
    std::cout << "Recorders set up.\n";

    // Run the simulation
    std::cout << "Running simulation...\n";
    system.run(buryn_config.simulation_steps);
    std::cout << "Simulation complete.\n";

    return 0;
}