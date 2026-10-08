#include "voltage_recorder.h"

VoltageRecorder::VoltageRecorder(const NeuronGroup& group,
                                const std::string& output_filepath,
                                std::vector<uint32_t> selected_neurons,
                                uint32_t recording_interval_steps,
                                GpuSimulationState& gpu_state)
    : group_(group),
    selected_neurons_(std::move(selected_neurons)),
    output_filepath_(output_filepath),
    recording_interval_steps_(recording_interval_steps),
    Monitor(gpu_state) {

    if (selected_neurons_.empty()) {
        // If no neurons specified, record all neurons
        selected_neurons_.resize(group_.num_neurons());
        std::iota(selected_neurons_.begin(), selected_neurons_.end(), 0);
    } else {
        // Validate selected neuron IDs
        for (uint32_t neuron_id : selected_neurons_) {
            if (neuron_id >= group_.num_neurons()) {
                throw std::out_of_range("Selected neuron ID " + std::to_string(neuron_id) + " is out of range for the neuron group.");
            }
        }
    }

    make_view();
}

void VoltageRecorder::write_header() {
    if (!output_.good()) {
        return;
    }

    output_ << "neuron_id, voltage\n";
}

void VoltageRecorder::make_view() {
    gpu_view_.num_neurons = group_.num_neurons();
    gpu_view_.offset = group_.offset();
    gpu_view_.membrane_potentials = gpu_state_.d_membrane_potentials() + group_.offset();
    gpu_view_.membrane_potential_history = gpu_state_.d_membrane_potential_history() + (group_.offset()* buryn::recording_interval_steps);
}

void VoltageRecorder::historicize(uint32_t clock_step) {
    kernel_historicize_membrane_potentials_launch(
        gpu_view_.membrane_potentials, 
        gpu_view_.membrane_potential_history, 
        gpu_view_.num_neurons, 
        recording_interval_steps_,
        clock_step);
}

void VoltageRecorder::record(uint32_t clock_step) {
    std::cout << "Recording membrane potentials at clock step " << clock_step << "...\n";
    const std::vector<float>& potentials = gpu_state_.download_and_flush_membrane_potential_history(gpu_view_.offset,
        gpu_view_.num_neurons,
        buryn::recording_interval_steps);

    for (uint32_t history_step = 0; history_step < buryn::recording_interval_steps; history_step++) {
        std::string filename = output_filepath_;
        uint32_t txt_position = filename.rfind(".txt");
        filename.insert(txt_position, "_step" + std::to_string((clock_step - buryn::recording_interval_steps) + history_step + 1));
        if (output_.is_open()) output_.close(); // close old file
        output_.clear(); // clear any fail or eof flags
        output_.open(filename, std::ios::out | std::ios::trunc); // open new file
        
        if (!output_.good()) {
            return;
        }

        write_header();
        for (uint32_t neuron_id : selected_neurons_) {
            output_ << neuron_id << ", " << potentials[neuron_id * buryn::recording_interval_steps + history_step] << "\n";
        }
    }
    
}
