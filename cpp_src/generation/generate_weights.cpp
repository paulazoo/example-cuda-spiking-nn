#include "generate_weights.h"

namespace generate_weights {

void weights_file_header(std::ofstream& output) {
    output << "pre_neuron_id, post_neuron_id, weight\n";
}

// Note that inhibitory weight values are still positive, but have inhibitory presynaptic neurons
std::vector<std::vector<float>> distance_2d_weights_file(
    const std::string& filename,
    const std::vector<std::vector<float>>& pre_group_neuron_locations,
    const std::vector<std::vector<float>>& post_group_neuron_locations,
    const WeightGenerationParams& params) {

    std::ofstream output(filename, std::ios::trunc);
    if (!output.is_open()) {
        std::cerr << "Failed to create weights file: " << filename << "\n";
        return {};  // empty matrix on file failure (keeps prior behavior of "return" on failure)
    }

    const uint32_t num_pre_group_neurons  = pre_group_neuron_locations.size();
    const uint32_t num_post_group_neurons = post_group_neuron_locations.size();
    if (num_pre_group_neurons == 0 || num_post_group_neurons == 0) {
        std::cerr << "Neuron groups must not be empty.\n";
        return {};
    }

    // Weight matrix: rows = pre, cols = post
    std::vector<std::vector<float>> weights(
        num_pre_group_neurons,
        std::vector<float>(num_post_group_neurons, 0.0f));

    weights_file_header(output);

    std::mt19937 rng(buryn::rng_seed);
    std::uniform_real_distribution<float> unif_distribution(0.0f, 1.0f);

    for (uint32_t pre_neuron_id = 0; pre_neuron_id < num_pre_group_neurons; ++pre_neuron_id) {
        for (uint32_t post_neuron_id = 0; post_neuron_id < num_post_group_neurons; ++post_neuron_id) {

            const std::vector<float>& pre_location  = pre_group_neuron_locations[pre_neuron_id];
            const std::vector<float>& post_location = post_group_neuron_locations[post_neuron_id];

            const float distance = math_functions::euclidean_distance_pbc(
                pre_location[0], pre_location[1],
                post_location[0], post_location[1],
                params.x_location_length, params.y_location_length);

            float weight = 0.0f;

            // No self-connections (only makes sense if IDs correspond; keeping original semantics)
            if (params.no_diagonal_connections && pre_neuron_id == post_neuron_id) {
                weights[pre_neuron_id][post_neuron_id] = weight;
                output << pre_neuron_id << ", " << post_neuron_id << ", " << weight << '\n';
                continue;
            }

            // Max distance cutoff
            if (std::abs(distance) > params.max_distance) {
                weights[pre_neuron_id][post_neuron_id] = weight;
                output << pre_neuron_id << ", " << post_neuron_id << ", " << weight << '\n';
                continue;
            }

            // Probabilistic connection
            const float random_value = unif_distribution(rng);
            float probability_of_connection = params.probability_multiplier;

            if (params.probabilistic_normal_not_uniform) {
                probability_of_connection *= calculate_connection_probability(distance, params.spread_scale);
            }

            if (random_value >= probability_of_connection) {
                weights[pre_neuron_id][post_neuron_id] = weight;
                output << pre_neuron_id << ", " << post_neuron_id << ", " << weight << '\n';
                continue;
            }

            // Max weight initially
            weight = params.max_weight_value;

            // Normal dropoff on weight magnitude
            if (params.weight_value_normal_dropoff) {
                weight *= std::exp(-1.0f * (std::pow(distance, 2.0f)) / (params.spread_scale));
            }

            // Check that weight is not negative before saving
            if (weight < 0) {
                throw std::runtime_error(
                    "generate_weights::distance_2d_weights_file: negative weight value calculated; see README notes for inhibitory weights"
                    "(pre=" + std::to_string(pre_neuron_id) +
                    ", post=" + std::to_string(post_neuron_id) +
                    ", weight=" + std::to_string(weight) + ")");
            }

            weights[pre_neuron_id][post_neuron_id] = weight;
            output << pre_neuron_id << ", " << post_neuron_id << ", " << weight << '\n';
        }
    }

    return weights;
}

float calculate_connection_probability(float distance, float spread_scale) {
    return std::exp(-1.0f * (std::pow(distance, 2.0f)) / (spread_scale));
}

}  // namespace generate_weights
