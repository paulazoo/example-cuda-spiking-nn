#include "stimulus.h"

Stimulus::Stimulus(std::vector<uint32_t> timestep_event_count,
                std::vector<uint32_t> stimulus_event_neuron_ids,
                std::vector<uint32_t> stimulus_event_post_group_offsets,
                std::vector<float> stimulus_event_amplitude_changes,
                GpuSimulationState& gpu_state)
    : gpu_state_(gpu_state) {
    StimulusInitHostData host_data = initialize_data(timestep_event_count,
                                                    stimulus_event_neuron_ids,
                                                    stimulus_event_post_group_offsets,
                                                    stimulus_event_amplitude_changes);
    timestep_event_count_ = host_data.timestep_event_count;
    total_stimulus_events_ = static_cast<uint32_t>(host_data.stimulus_event_neuron_ids.size());
    make_gpu_view();
    upload_data(host_data);
}

StimulusInitHostData Stimulus::initialize_data(const std::vector<uint32_t>& timestep_event_count,
                                         const std::vector<uint32_t>& stimulus_event_neuron_ids,
                                         const std::vector<uint32_t>& stimulus_event_post_group_offsets,
                                         const std::vector<float>& stimulus_event_amplitude_changes) {
    StimulusInitHostData host_data;
    host_data.timestep_event_count = timestep_event_count;
    host_data.stimulus_event_neuron_ids = stimulus_event_neuron_ids;
    host_data.stimulus_event_post_group_offsets = stimulus_event_post_group_offsets;
    host_data.stimulus_event_amplitude_changes = stimulus_event_amplitude_changes;
    return host_data;
}

void Stimulus::make_gpu_view() {
    // Pointers to GPU data
    gpu_view_.g_excitatory = gpu_state_.d_g_excitatory();
    gpu_view_.g_inhibitory = gpu_state_.d_g_inhibitory();
    gpu_view_.i_stimulus = gpu_state_.d_i_stimulus();
    gpu_view_.timestep_event_count = gpu_state_.d_timestep_event_count();
    gpu_view_.stimulus_event_neuron_ids = gpu_state_.d_stimulus_event_neuron_ids();
    gpu_view_.stimulus_event_post_group_offsets = gpu_state_.d_stimulus_event_post_group_offsets();
    gpu_view_.stimulus_event_amplitude_changes = gpu_state_.d_stimulus_event_amplitude_changes();
}

void Stimulus::upload_data(const StimulusInitHostData& host_data) {
    gpu_state_.upload_stimulus_view(gpu_view_, host_data);
}

void Stimulus::apply() {
    if (current_timestep_ >= timestep_event_count_.size()) {
        return;
    }

    const uint32_t num_events_to_apply = timestep_event_count_[current_timestep_];
    if (next_event_idx_ + num_events_to_apply > total_stimulus_events_) {
        throw std::runtime_error("Stimulus event indexing exceeded uploaded event buffers");
    }

    kernel_stimulus_apply_launch(gpu_view_, next_event_idx_, num_events_to_apply);

    next_event_idx_ += num_events_to_apply;
    ++current_timestep_;
}