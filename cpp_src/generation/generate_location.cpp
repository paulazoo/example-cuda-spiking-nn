#include "generate_location.h"

namespace generate_location {

std::vector<std::vector<float>> generate_2d_grid_locations(const std::string& output_filepath,
                                                            uint32_t num_neurons,
                                                            float spacing,
                                                            uint32_t rows,
                                                            uint32_t cols) {
    std::vector<std::vector<float>> locations;
    if (num_neurons == 0) return locations;
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("rows and cols must be > 0");
    }
    locations.resize(num_neurons);

    // Overflow-safe: ensure rows*cols >= num_neurons without multiplying unless safe.
    if (rows < (num_neurons + cols - 1) / cols) {
        throw std::invalid_argument("grid too small: need at least num_neurons cells");
    }

    std::ofstream output(output_filepath, std::ios::trunc);
    if (!output.is_open() || !output.good()) {
        throw std::runtime_error("failed to open output file: " + output_filepath);
    }

    // Header: neuron_id + two dimensions for 2D grid
    output << "neuron_id, locationdim_0, locationdim_1\n";

    for (uint32_t id = 0; id < num_neurons; ++id) {
        const uint32_t r = id / cols;
        const uint32_t c = id % cols;

        const float x = static_cast<float>(c) * spacing;
        const float y = static_cast<float>(r) * spacing;

        locations[id] = {x, y};

        output << id << ", " << x << ", " << y << "\n";
    }

    if (!output.good()) {
        throw std::runtime_error("failed while writing output file: " + output_filepath);
    }

    return locations;
}

}  // namespace generate_location
