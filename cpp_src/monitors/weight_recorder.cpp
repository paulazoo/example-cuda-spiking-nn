#include "weight_recorder.h"


WeightRecorder::WeightRecorder(const Connection& connection,
                                const std::string& output_filepath,
                                GpuSimulationState& gpu_state)
    : connection_(connection),
    output_filepath_(output_filepath),
    Monitor(gpu_state) {
}


void WeightRecorder::write_header() {
    if (!output_.good()) {
        return;
    }

    output_ << "pre_neuron_id, post_neuron_id, weight\n";
}

void WeightRecorder::historicize(uint32_t clock_step) {
    // No historicization needed for weights for now
}

void WeightRecorder::record(uint32_t clock_step) {
    if (clock_step % 100000 != 0) {
        return; // Only record every so often (10s) to avoid excessive file I/O and storage usage
    }
    std::string filename = output_filepath_;
    uint32_t txt_position = filename.rfind(".txt");
    filename.insert(txt_position, "_step" + std::to_string(clock_step));
    if (output_.is_open()) output_.close(); // close old file
    output_.clear(); // clear any fail or eof flags
    output_.open(filename, std::ios::out | std::ios::trunc); // open new file
    std::cout << "clock step " << clock_step << " saved weights: " << filename << "\n";

    if (!output_.good()) {
        return;
    }

    write_header();

    std::vector<float> weight_matrix = gpu_state_.download_weight_matrix(connection_.weight_matrix_idx());
    for (uint32_t pre_id = 0; pre_id < connection_.pre_group_num_neurons(); ++pre_id) {
        for (uint32_t post_id = 0; post_id < connection_.post_group_num_neurons(); ++post_id) {
            float weight_matrix_value =
                weight_matrix[pre_id * connection_.post_group_num_neurons() + post_id];
            output_ << pre_id << ", " << post_id << ", " << weight_matrix_value << "\n";
        }
    }
}
