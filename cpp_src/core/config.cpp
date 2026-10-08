#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "json.hpp"
#include "buryn_definitions.h"
#include "config.h"

// ChatGPT made most of this, so not thoroughly checked


static void apply_weight_generation_overrides(generate_weights::WeightGenerationParams& config,
                                              const nlohmann::json& obj,
                                              const std::string& prefix) {
    const std::string max_weight_value_key = prefix + "_max_weight_value";
    const std::string max_distance_key = prefix + "_max_distance";
    const std::string spread_scale_key = prefix + "_spread_scale";
    const std::string weight_value_normal_dropoff_key = prefix + "_weight_value_normal_dropoff";
    const std::string probabilistic_normal_not_uniform_key = prefix + "_probabilistic_normal_not_uniform";
    const std::string probability_multiplier_key = prefix + "_probability_multiplier";
    const std::string no_diagonal_connections_key = prefix + "_no_diagonal_connections";
    const std::string x_location_length_key = prefix + "_x_location_length";
    const std::string y_location_length_key = prefix + "_y_location_length";

    if (obj.contains(max_weight_value_key))
        config.max_weight_value = obj.value(max_weight_value_key, config.max_weight_value);
    if (obj.contains(max_distance_key))
        config.max_distance = obj.value(max_distance_key, config.max_distance);
    if (obj.contains(spread_scale_key))
        config.spread_scale = obj.value(spread_scale_key, config.spread_scale);
    if (obj.contains(weight_value_normal_dropoff_key))
        config.weight_value_normal_dropoff = obj.value(weight_value_normal_dropoff_key, config.weight_value_normal_dropoff);
    if (obj.contains(probabilistic_normal_not_uniform_key))
        config.probabilistic_normal_not_uniform = obj.value(probabilistic_normal_not_uniform_key, config.probabilistic_normal_not_uniform);
    if (obj.contains(probability_multiplier_key))
        config.probability_multiplier = obj.value(probability_multiplier_key, config.probability_multiplier);
    if (obj.contains(no_diagonal_connections_key))
        config.no_diagonal_connections = obj.value(no_diagonal_connections_key, config.no_diagonal_connections);
    if (obj.contains(x_location_length_key))
        config.x_location_length = obj.value(x_location_length_key, config.x_location_length);
    if (obj.contains(y_location_length_key))
        config.y_location_length = obj.value(y_location_length_key, config.y_location_length);
}

static std::string prefix_output_path(const std::string& output_directory, const std::string& path_value) {
    if (path_value.empty()) {
        return path_value;
    }

    std::filesystem::path path(path_value);
    if (path.is_absolute()) {
        return path_value;
    }

    std::filesystem::path output_dir(output_directory);
    if (!output_directory.empty()) {
        auto path_it = path.begin();
        auto output_it = output_dir.begin();
        if (path_it != path.end() && output_it != output_dir.end() && *path_it == *output_it) {
            return path_value;
        }
    }

    return (output_dir / path).lexically_normal().string();
}

Config load_config_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open config file: " + path);

    nlohmann::json j;
    try {
        in >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(std::string("JSON parse error: ") + e.what());
    }

    if (!j.is_object()) throw std::runtime_error("Config JSON must be an object at the top level");

    Config c;

    // Missing keys keep defaults
    c.simulation_steps = j.value("simulation_steps", c.simulation_steps);
    c.output_directory = j.value("output_directory", c.output_directory);
    c.description = j.value("description", c.description);
    c.rng_seed = j.value("rng_seed", c.rng_seed);
    c.timestep = j.value("timestep", c.timestep);

    // Neuron group settings
    c.num_excitatory_neurons = j.value("num_excitatory_neurons", c.num_excitatory_neurons);
    c.num_inhibitory_neurons = j.value("num_inhibitory_neurons", c.num_inhibitory_neurons);

    c.g_leak = j.value("g_leak", c.g_leak);
    c.membrane_capacitance = j.value("membrane_capacitance", c.membrane_capacitance);
    c.depolarization_slope_factor = j.value("depolarization_slope_factor", c.depolarization_slope_factor);

    c.v_effective_spike_threshold = j.value("v_effective_spike_threshold", c.v_effective_spike_threshold);
    c.v_excitatory_reversal = j.value("v_excitatory_reversal", c.v_excitatory_reversal);
    c.v_inhibitory_reversal = j.value("v_inhibitory_reversal", c.v_inhibitory_reversal);
    c.v_threshold = j.value("v_threshold", c.v_threshold);
    c.v_reset = j.value("v_reset", c.v_reset);
    c.v_rest = j.value("v_rest", c.v_rest);
    c.refractory_steps = j.value("refractory_steps", c.refractory_steps);
    
    c.group_e_mean_background_excitatory_spikes = j.value("group_e_mean_background_excitatory_spikes", c.group_e_mean_background_excitatory_spikes);
    c.group_e_mean_background_inhibitory_spikes = j.value("group_e_mean_background_inhibitory_spikes", c.group_e_mean_background_inhibitory_spikes);
    c.group_e_background_excitatory_postsynaptic_r_trace_decay = j.value("group_e_background_excitatory_postsynaptic_r_trace_decay", c.group_e_background_excitatory_postsynaptic_r_trace_decay);
    c.group_e_background_excitatory_postsynaptic_d_trace_decay = j.value("group_e_background_excitatory_postsynaptic_d_trace_decay", c.group_e_background_excitatory_postsynaptic_d_trace_decay);
    c.group_e_background_inhibitory_postsynaptic_r_trace_decay = j.value("group_e_background_inhibitory_postsynaptic_r_trace_decay", c.group_e_background_inhibitory_postsynaptic_r_trace_decay);
    c.group_e_background_inhibitory_postsynaptic_d_trace_decay = j.value("group_e_background_inhibitory_postsynaptic_d_trace_decay", c.group_e_background_inhibitory_postsynaptic_d_trace_decay);
    c.group_e_background_excitatory_scaling_factor = j.value("group_e_background_excitatory_scaling_factor", c.group_e_background_excitatory_scaling_factor);
    c.group_e_background_inhibitory_scaling_factor = j.value("group_e_background_inhibitory_scaling_factor", c.group_e_background_inhibitory_scaling_factor);
    c.group_i_mean_background_excitatory_spikes = j.value("group_i_mean_background_excitatory_spikes", c.group_i_mean_background_excitatory_spikes);
    c.group_i_mean_background_inhibitory_spikes = j.value("group_i_mean_background_inhibitory_spikes", c.group_i_mean_background_inhibitory_spikes);
    c.group_i_background_excitatory_postsynaptic_r_trace_decay = j.value("group_i_background_excitatory_postsynaptic_r_trace_decay", c.group_i_background_excitatory_postsynaptic_r_trace_decay);
    c.group_i_background_excitatory_postsynaptic_d_trace_decay = j.value("group_i_background_excitatory_postsynaptic_d_trace_decay", c.group_i_background_excitatory_postsynaptic_d_trace_decay);
    c.group_i_background_inhibitory_postsynaptic_r_trace_decay = j.value("group_i_background_inhibitory_postsynaptic_r_trace_decay", c.group_i_background_inhibitory_postsynaptic_r_trace_decay);
    c.group_i_background_inhibitory_postsynaptic_d_trace_decay = j.value("group_i_background_inhibitory_postsynaptic_d_trace_decay", c.group_i_background_inhibitory_postsynaptic_d_trace_decay);
    c.group_i_background_excitatory_scaling_factor = j.value("group_i_background_excitatory_scaling_factor", c.group_i_background_excitatory_scaling_factor);
    c.group_i_background_inhibitory_scaling_factor = j.value("group_i_background_inhibitory_scaling_factor", c.group_i_background_inhibitory_scaling_factor);

    // Neuron location settings
    c.excitatory_locations_file = j.value("excitatory_locations_file", c.excitatory_locations_file);
    c.inhibitory_locations_file = j.value("inhibitory_locations_file", c.inhibitory_locations_file);
    c.excitatory_location_spacing = j.value("excitatory_location_spacing", c.excitatory_location_spacing);
    c.inhibitory_location_spacing = j.value("inhibitory_location_spacing", c.inhibitory_location_spacing); 
    c.excitatory_side_points = j.value("excitatory_side_points", c.excitatory_side_points);
    c.inhibitory_side_points = j.value("inhibitory_side_points", c.inhibitory_side_points);
    c.side_length = j.value("side_length", c.side_length);

    // Weight generation settings
    c.e_max_weight_value = j.value("e_max_weight_value", c.e_max_weight_value);
    c.i_max_weight_value = j.value("i_max_weight_value", c.i_max_weight_value);
    c.max_connection_distance = j.value("max_connection_distance", c.max_connection_distance);
    c.connection_spread_scale = j.value("connection_spread_scale", c.connection_spread_scale);
    c.connection_probability_multiplier = j.value("connection_probability_multiplier", c.connection_probability_multiplier);
    c.weight_generation_params_e_to_e.max_weight_value = c.e_max_weight_value;
    c.weight_generation_params_e_to_i.max_weight_value = c.e_max_weight_value;
    c.weight_generation_params_i_to_e.max_weight_value = c.i_max_weight_value;
    c.weight_generation_params_i_to_i.max_weight_value = c.i_max_weight_value;
    c.weight_generation_params_e_to_e.max_distance = c.max_connection_distance;
    c.weight_generation_params_e_to_i.max_distance = c.max_connection_distance;
    c.weight_generation_params_i_to_e.max_distance = c.max_connection_distance;
    c.weight_generation_params_i_to_i.max_distance = c.max_connection_distance;
    c.weight_generation_params_e_to_e.spread_scale = c.connection_spread_scale;
    c.weight_generation_params_e_to_i.spread_scale = c.connection_spread_scale;
    c.weight_generation_params_i_to_e.spread_scale = c.connection_spread_scale;
    c.weight_generation_params_i_to_i.spread_scale = c.connection_spread_scale;
    c.weight_generation_params_e_to_e.probability_multiplier = c.connection_probability_multiplier;
    c.weight_generation_params_e_to_i.probability_multiplier = c.connection_probability_multiplier;
    c.weight_generation_params_i_to_e.probability_multiplier = c.connection_probability_multiplier;
    c.weight_generation_params_i_to_i.probability_multiplier = c.connection_probability_multiplier;
    c.weight_generation_params_e_to_e.x_location_length = c.side_length;
    c.weight_generation_params_e_to_i.x_location_length = c.side_length;
    c.weight_generation_params_i_to_e.x_location_length = c.side_length;
    c.weight_generation_params_i_to_i.x_location_length = c.side_length;
    c.weight_generation_params_e_to_e.y_location_length = c.side_length;
    c.weight_generation_params_e_to_i.y_location_length = c.side_length;
    c.weight_generation_params_i_to_e.y_location_length = c.side_length;
    c.weight_generation_params_i_to_i.y_location_length = c.side_length;

    apply_weight_generation_overrides(c.weight_generation_params_e_to_e, j, "e_to_e");
    apply_weight_generation_overrides(c.weight_generation_params_e_to_i, j, "e_to_i");
    apply_weight_generation_overrides(c.weight_generation_params_i_to_e, j, "i_to_e");
    apply_weight_generation_overrides(c.weight_generation_params_i_to_i, j, "i_to_i");

    // Connection settings
    c.weights_e_to_i_filename = j.value("weights_e_to_i_filename", j.value("weights_e_to_i", c.weights_e_to_i_filename));
    c.weights_i_to_e_filename = j.value("weights_i_to_e_filename", j.value("weights_i_to_e", c.weights_i_to_e_filename));
    c.weights_e_to_e_filename = j.value("weights_e_to_e_filename", j.value("weights_e_to_e", c.weights_e_to_e_filename));
    c.weights_i_to_i_filename = j.value("weights_i_to_i_filename", j.value("weights_i_to_i", c.weights_i_to_i_filename));
    c.tau_decay_e = j.value("tau_decay_e", c.tau_decay_e);
    c.tau_rise_e  = j.value("tau_rise_e", c.tau_rise_e);
    c.tau_decay_i = j.value("tau_decay_i", c.tau_decay_i);
    c.tau_rise_i  = j.value("tau_rise_i", c.tau_rise_i);
    c.postsynaptic_kernel_normalization_value =
        j.value("postsynaptic_kernel_normalization_value", c.postsynaptic_kernel_normalization_value);

    // Plasticity settings
    c.e_e_plasticity_enabled = j.value("e_e_plasticity_enabled", c.e_e_plasticity_enabled);
    c.e_e_plasticity_ltp_A = j.value("e_e_plasticity_ltp_A", c.e_e_plasticity_ltp_A);
    c.e_e_plasticity_ltd_A = j.value("e_e_plasticity_ltd_A", c.e_e_plasticity_ltd_A);
    c.e_e_plasticity_ltp_B = j.value("e_e_plasticity_ltp_B", c.e_e_plasticity_ltp_B);
    c.e_e_plasticity_ltd_B = j.value("e_e_plasticity_ltd_B", c.e_e_plasticity_ltd_B);
    c.e_e_plasticity_tau_ltp = j.value("e_e_plasticity_tau_ltp", c.e_e_plasticity_tau_ltp);
    c.e_e_plasticity_tau_ltd = j.value("e_e_plasticity_tau_ltd", c.e_e_plasticity_tau_ltd);
    c.e_e_plasticity_triplet_tau_ltp = j.value("e_e_plasticity_triplet_tau_ltp", c.e_e_plasticity_triplet_tau_ltp);
    c.e_e_plasticity_triplet_tau_ltd = j.value("e_e_plasticity_triplet_tau_ltd", c.e_e_plasticity_triplet_tau_ltd);
    c.e_e_plasticity_weight_min = j.value("e_e_plasticity_weight_min", c.e_e_plasticity_weight_min);
    c.e_e_plasticity_weight_max = j.value("e_e_plasticity_weight_max", c.e_e_plasticity_weight_max);
    c.i_e_plasticity_enabled = j.value("i_e_plasticity_enabled", c.i_e_plasticity_enabled);
    c.i_e_plasticity_ltp_A = j.value("i_e_plasticity_ltp_A", c.i_e_plasticity_ltp_A);
    c.i_e_plasticity_ltd_A = j.value("i_e_plasticity_ltd_A", c.i_e_plasticity_ltd_A);
    c.i_e_plasticity_ltd_B = j.value("i_e_plasticity_ltd_B", c.i_e_plasticity_ltd_B);
    c.i_e_plasticity_tau_ltp = j.value("i_e_plasticity_tau_ltp", c.i_e_plasticity_tau_ltp);
    c.i_e_plasticity_tau_ltd = j.value("i_e_plasticity_tau_ltd", c.i_e_plasticity_tau_ltd);
    c.i_e_plasticity_weight_min = j.value("i_e_plasticity_weight_min", c.i_e_plasticity_weight_min);
    c.i_e_plasticity_weight_max = j.value("i_e_plasticity_weight_max", c.i_e_plasticity_weight_max);

    // Stimulus settings
    if (j.contains("stimulus_input_values")) {
        const auto& arr = j.at("stimulus_input_values");
        if (!arr.is_array())
            throw std::runtime_error("stimulus_input_values must be an array");

        c.stimulus_input_values = arr.get<std::vector<float>>();
    }
    c.stimulus_schedules_folder = j.value("stimulus_schedules_folder", c.stimulus_schedules_folder);
    if (j.contains("stimulus_selected_input_ids")) {
        const auto& arr = j.at("stimulus_selected_input_ids");
        if (!arr.is_array())
            throw std::runtime_error("stimulus_selected_input_ids must be an array");

        // Enforce integer/unsigned elements for clearer errors
        for (uint32_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_number_unsigned() && !arr[i].is_number_integer())
                throw std::runtime_error("stimulus_selected_input_ids must contain only integers");
            if (arr[i].is_number_integer() && arr[i].get<long long>() < 0)
                throw std::runtime_error("stimulus_selected_input_ids must not contain negative values");
        }

        c.stimulus_selected_input_ids = arr.get<std::vector<uint32_t>>();
    }
    if (j.contains("stimulus_selected_input_post_group_offsets")) {
        const auto& arr = j.at("stimulus_selected_input_post_group_offsets");
        if (!arr.is_array())
            throw std::runtime_error("stimulus_selected_input_post_group_offsets must be an array");

        for (uint32_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_number_unsigned() && !arr[i].is_number_integer())
                throw std::runtime_error("stimulus_selected_input_post_group_offsets must contain only integers");
            if (arr[i].is_number_integer() && arr[i].get<long long>() < 0)
                throw std::runtime_error("stimulus_selected_input_post_group_offsets must not contain negative values");
        }

        c.stimulus_selected_input_post_group_offsets = arr.get<std::vector<uint32_t>>();
    }
    if (j.contains("stimulus_start_clock_steps")) {
        const auto& arr = j.at("stimulus_start_clock_steps");
        if (!arr.is_array())
            throw std::runtime_error("stimulus_start_clock_steps must be an array");

        // Enforce integer/unsigned elements for clearer errors
        for (uint32_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_number_unsigned() && !arr[i].is_number_integer())
                throw std::runtime_error("stimulus_start_clock_steps must contain only integers");
            if (arr[i].is_number_integer() && arr[i].get<long long>() < 0)
                throw std::runtime_error("stimulus_start_clock_steps must not contain negative values");
        }

        c.stimulus_start_clock_steps = arr.get<std::vector<uint32_t>>();
    }
    if (j.contains("stimulus_end_clock_steps")) {
        const auto& arr = j.at("stimulus_end_clock_steps");
        if (!arr.is_array())
            throw std::runtime_error("stimulus_end_clock_steps must be an array");

        // Enforce integer/unsigned elements for clearer errors
        for (uint32_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_number_unsigned() && !arr[i].is_number_integer())
                throw std::runtime_error("stimulus_end_clock_steps must contain only integers");
            if (arr[i].is_number_integer() && arr[i].get<long long>() < 0)
                throw std::runtime_error("stimulus_end_clock_steps must not contain negative values");
        }

        c.stimulus_end_clock_steps = arr.get<std::vector<uint32_t>>();
    }

    // Monitor settings
    c.e_spike_recording_file = j.value("e_spike_recording_file", c.e_spike_recording_file);
    c.i_spike_recording_file = j.value("i_spike_recording_file", c.i_spike_recording_file);
    c.voltage_recording_file = j.value("voltage_recording_file", c.voltage_recording_file);
    if (j.contains("voltage_recording_neuron_ids")) {
        const auto& arr = j.at("voltage_recording_neuron_ids");
        if (!arr.is_array())
            throw std::runtime_error("voltage_recording_neuron_ids must be an array");

        // Optional: enforce integer/unsigned elements for clearer errors
        for (uint32_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_number_unsigned() && !arr[i].is_number_integer())
                throw std::runtime_error("voltage_recording_neuron_ids must contain only integers");
            if (arr[i].is_number_integer() && arr[i].get<long long>() < 0)
                throw std::runtime_error("voltage_recording_neuron_ids must not contain negative values");
        }

        c.voltage_recording_neuron_ids = arr.get<std::vector<uint32_t>>();
    }
    c.e_e_weight_recording_file = j.value("e_e_weight_recording_file", c.e_e_weight_recording_file);

    c.excitatory_locations_file = prefix_output_path(c.output_directory, c.excitatory_locations_file);
    c.inhibitory_locations_file = prefix_output_path(c.output_directory, c.inhibitory_locations_file);
    c.weights_e_to_i_filename = prefix_output_path(c.output_directory, c.weights_e_to_i_filename);
    c.weights_i_to_e_filename = prefix_output_path(c.output_directory, c.weights_i_to_e_filename);
    c.weights_e_to_e_filename = prefix_output_path(c.output_directory, c.weights_e_to_e_filename);
    c.weights_i_to_i_filename = prefix_output_path(c.output_directory, c.weights_i_to_i_filename);
    c.stimulus_schedules_folder = prefix_output_path(c.output_directory, c.stimulus_schedules_folder);
    c.e_spike_recording_file = prefix_output_path(c.output_directory, c.e_spike_recording_file);
    c.i_spike_recording_file = prefix_output_path(c.output_directory, c.i_spike_recording_file);
    c.voltage_recording_file = prefix_output_path(c.output_directory, c.voltage_recording_file);
    c.e_e_weight_recording_file = prefix_output_path(c.output_directory, c.e_e_weight_recording_file);

    if (c.stimulus_selected_input_post_group_offsets.empty()) {
        c.stimulus_selected_input_post_group_offsets = std::vector<uint32_t>(c.stimulus_selected_input_ids.size(), 0);
    }

    if (c.stimulus_selected_input_post_group_offsets.size() != c.stimulus_selected_input_ids.size()) {
        throw std::runtime_error("stimulus_selected_input_post_group_offsets size must match stimulus_selected_input_ids size");
    }

    // Checks
    if (c.num_excitatory_neurons == 0 || c.num_inhibitory_neurons == 0)
        throw std::runtime_error("Neuron counts must be > 0");
    if (c.membrane_capacitance <= 0.0)
        throw std::runtime_error("membrane_capacitance must be > 0");

    return c;
}