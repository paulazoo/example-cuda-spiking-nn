#include "spike_recorder.h"

SpikeRecorder::SpikeRecorder(const NeuronGroup& group,
                            const std::string& output_filepath,
                            uint32_t recording_interval_steps,
                            GpuSimulationState& gpu_state)
    : group_(group),
    output_(output_filepath, std::ios::trunc),
    recording_interval_steps_(recording_interval_steps),
    Monitor(gpu_state) {
        make_view();
        write_header();
}

void SpikeRecorder::make_view() {
    gpu_view_.num_neurons = group_.num_neurons();
    gpu_view_.offset = group_.offset();
    gpu_view_.spikes = gpu_state_.d_spikes() + group_.offset();
    gpu_view_.spike_history = gpu_state_.d_spike_history() + (group_.offset() * buryn::recording_interval_steps);
}

void SpikeRecorder::write_header() {
    if (!output_.good()) {
        return;
    }

    output_ << "clock_step, neuron_ids\n";
}

void SpikeRecorder::historicize(uint32_t clock_step) {
    kernel_historicize_spikes_launch(
        gpu_view_.spikes,
        gpu_view_.spike_history,
        gpu_view_.num_neurons,
        recording_interval_steps_,
        clock_step
    );
}

void SpikeRecorder::record(uint32_t clock_step) {
    if (clock_step < buryn::recording_interval_steps) {
        return;
    }
    std::vector<uint32_t> host_spike_history =
        gpu_state_.download_and_flush_spike_history(
            gpu_view_.offset,
            gpu_view_.num_neurons,
            buryn::recording_interval_steps);

    const uint32_t first_step = clock_step - buryn::recording_interval_steps;
    for (uint32_t inner_step = 0; inner_step < buryn::recording_interval_steps; inner_step++) {
        output_ << (first_step + inner_step) << ", ";
        for (uint32_t neuron_id = 0; neuron_id < gpu_view_.num_neurons; neuron_id++) {
            if (host_spike_history[neuron_id * buryn::recording_interval_steps + inner_step] == 1) {
                output_ << neuron_id << " ";
            }
        }
        output_ << "\n";
    }
}
