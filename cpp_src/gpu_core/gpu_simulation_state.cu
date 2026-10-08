#include "gpu_simulation_state.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "cuda_utils.h"

GpuSimulationState::GpuSimulationState(const uint32_t total_simulation_length, 
                                const uint32_t num_excitatory_neurons,
                                const uint32_t num_inhibitory_neurons,
                                const uint32_t num_stimulus_events,
                                const uint32_t num_plasticities)
    : total_neurons_(num_excitatory_neurons + num_inhibitory_neurons),
    num_excitatory_neurons_(num_excitatory_neurons),
    num_inhibitory_neurons_(num_inhibitory_neurons),
    total_simulation_length_(total_simulation_length),
    num_stimulus_events_(num_stimulus_events),
    num_plasticities_(num_plasticities) {

    // NOTE: assuming two groups (excitatory and inhibitory) for this now
    weight_matrix_lookup_.push_back("w_ee");
    weight_matrix_lookup_.push_back("w_ei");
    weight_matrix_lookup_.push_back("w_ie");
    weight_matrix_lookup_.push_back("w_ii");
    
    free_device_buffers();
    initialize_device_buffers();
}


GpuSimulationState::~GpuSimulationState() {
    free_device_buffers();
}

void GpuSimulationState::initialize_device_buffers() {
    // Allocate device buffers and copy device pointer values to d_xxx_
    // Kernel Neuron Group
    cuda_check(cudaMalloc(&d_membrane_potentials_, total_neurons_ * sizeof(float)), "cudaMalloc d_membrane_potentials_");
    cuda_check(cudaMalloc(&d_g_excitatory_, total_neurons_ * sizeof(float)), "cudaMalloc d_g_excitatory_");
    cuda_check(cudaMalloc(&d_g_inhibitory_, total_neurons_ * sizeof(float)), "cudaMalloc d_g_inhibitory_");
    cuda_check(cudaMalloc(&d_i_stimulus_, total_neurons_ * sizeof(float)), "cudaMalloc d_i_stimulus_");
    cuda_check(cudaMalloc(&d_refractory_counter_, total_neurons_ * sizeof(uint32_t)), "cudaMalloc d_refractory_counter_");
    cuda_check(cudaMalloc(&d_spikes_, total_neurons_ * sizeof(uint32_t)), "cudaMalloc d_spikes_");
    cuda_check(cudaMalloc(&d_background_g_excitatory_synaptic_conductances_, total_neurons_ * sizeof(float)), "cudaMalloc d_background_g_excitatory_synaptic_conductances_");
    cuda_check(cudaMalloc(&d_background_g_inhibitory_synaptic_conductances_, total_neurons_ * sizeof(float)), "cudaMalloc d_background_g_inhibitory_synaptic_conductances_");
    cuda_check(cudaMalloc(&d_background_excitatory_postsynaptic_r_traces_, total_neurons_ * sizeof(float)), "cudaMalloc d_background_excitatory_postsynaptic_r_traces_");
    cuda_check(cudaMalloc(&d_background_excitatory_postsynaptic_d_traces_, total_neurons_ * sizeof(float)), "cudaMalloc d_background_excitatory_postsynaptic_d_traces_");
    cuda_check(cudaMalloc(&d_background_inhibitory_postsynaptic_r_traces_, total_neurons_ * sizeof(float)), "cudaMalloc d_background_inhibitory_postsynaptic_r_traces_");
    cuda_check(cudaMalloc(&d_background_inhibitory_postsynaptic_d_traces_, total_neurons_ * sizeof(float)), "cudaMalloc d_background_inhibitory_postsynaptic_d_traces_");
    cuda_check(cudaMalloc(&d_neuron_rng_states_, total_neurons_ * sizeof(curandStatePhilox4_32_10_t)), "cudaMalloc d_neuron_rng_states_");
    // Kernel Connection
    cuda_check(cudaMalloc(&d_w_ee_, num_excitatory_neurons_ * num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_w_ee_");
    cuda_check(cudaMalloc(&d_w_ei_, num_excitatory_neurons_ * num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_w_ei_");
    cuda_check(cudaMalloc(&d_w_ie_, num_inhibitory_neurons_ * num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_w_ie_");
    cuda_check(cudaMalloc(&d_w_ii_, num_inhibitory_neurons_ * num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_w_ii_");
    cuda_check(cudaMalloc(&d_ee_postsynaptic_r_traces_, num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_ee_postsynaptic_r_traces_");
    cuda_check(cudaMalloc(&d_ee_postsynaptic_d_traces_, num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_ee_postsynaptic_d_traces_");
    cuda_check(cudaMalloc(&d_ee_summed_postsynaptic_conductances_, num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_ee_summed_postsynaptic_conductances_");
    cuda_check(cudaMalloc(&d_ei_postsynaptic_r_traces_, num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_ei_postsynaptic_r_traces_");
    cuda_check(cudaMalloc(&d_ei_postsynaptic_d_traces_, num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_ei_postsynaptic_d_traces_");
    cuda_check(cudaMalloc(&d_ei_summed_postsynaptic_conductances_, num_excitatory_neurons_ * sizeof(float)), "cudaMalloc d_ei_summed_postsynaptic_conductances_");
    cuda_check(cudaMalloc(&d_ie_postsynaptic_r_traces_, num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_ie_postsynaptic_r_traces_");
    cuda_check(cudaMalloc(&d_ie_postsynaptic_d_traces_, num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_ie_postsynaptic_d_traces_");
    cuda_check(cudaMalloc(&d_ie_summed_postsynaptic_conductances_, num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_ie_summed_postsynaptic_conductances_");
    cuda_check(cudaMalloc(&d_ii_postsynaptic_r_traces_, num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_ii_postsynaptic_r_traces_");
    cuda_check(cudaMalloc(&d_ii_postsynaptic_d_traces_, num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_ii_postsynaptic_d_traces_");
    cuda_check(cudaMalloc(&d_ii_summed_postsynaptic_conductances_, num_inhibitory_neurons_ * sizeof(float)), "cudaMalloc d_ii_summed_postsynaptic_conductances_");
    // Kernel Stimulus
    cuda_check(cudaMalloc(&d_timestep_event_count_, total_simulation_length_ *sizeof(uint32_t)), "cudaMalloc d_timestep_event_count_");
    cuda_check(cudaMalloc(&d_stimulus_event_neuron_ids_, num_stimulus_events_ * sizeof(uint32_t)), "cudaMalloc d_stimulus_event_neuron_ids_");
    cuda_check(cudaMalloc(&d_stimulus_event_post_group_offsets_, num_stimulus_events_ * sizeof(uint32_t)), "cudaMalloc d_stimulus_event_post_group_offsets_");
    cuda_check(cudaMalloc(&d_stimulus_event_amplitude_changes_, num_stimulus_events_ * sizeof(float)), "cudaMalloc d_stimulus_event_amplitude_changes_");
    // Kernel Plasticity
    cuda_check(cudaMalloc(&d_pre_traces_, num_plasticities_ * total_neurons_ * sizeof(float)), "cudaMalloc d_pre_traces_");
    cuda_check(cudaMalloc(&d_post_traces_, num_plasticities_ * total_neurons_ * sizeof(float)), "cudaMalloc d_post_traces_");
    cuda_check(cudaMalloc(&d_triplet_pre_traces_, num_plasticities_ * total_neurons_ * sizeof(float)), "cudaMalloc d_triplet_pre_traces_");
    cuda_check(cudaMalloc(&d_triplet_post_traces_, num_plasticities_ * total_neurons_ * sizeof(float)), "cudaMalloc d_triplet_post_traces_");
    cuda_check(cudaMalloc(&d_pre_weight_sums_, num_plasticities_ * total_neurons_ * sizeof(float)), "cudaMalloc d_pre_weight_sums_");
    // Kernel Monitors
    cuda_check(cudaMalloc(&d_spike_history_, total_neurons_ * buryn::recording_interval_steps * sizeof(uint32_t)), "cudaMalloc d_spike_history_");
    cuda_check(cudaMalloc(&d_membrane_potential_history_, total_neurons_ * buryn::recording_interval_steps * sizeof(float)), "cudaMalloc d_membrane_potential_history_");

    initialize_rng_states();
    initialize_cublas_handle();
    initialize_monitor_histories();
}

void GpuSimulationState::initialize_rng_states() {
    if (d_neuron_rng_states_ == nullptr) {
        throw std::runtime_error("initialize_rng_states called before d_neuron_rng_states_ was allocated");
    }
    kernel_init_neuron_rng_states_launch(d_neuron_rng_states_, total_neurons_, static_cast<unsigned long long>(buryn::rng_seed));
}

void GpuSimulationState::initialize_cublas_handle() {
    cublasHandle_t handle = nullptr;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_handle_ = handle;
}

void GpuSimulationState::initialize_monitor_histories() {
    cuda_check(cudaMemset(d_membrane_potential_history_, 0.0f, total_neurons_ * buryn::recording_interval_steps * sizeof(float)), "cudaMemset d_membrane_potential_history_");
    cuda_check(cudaMemset(d_spike_history_, 0, total_neurons_ * buryn::recording_interval_steps * sizeof(uint32_t)), "cudaMemset d_spike_history_");
}

// YES a slice for all data
void GpuSimulationState::upload_neuron_group_view(const NeuronGroupView& gpu_view, const NeuronGroupInitHostData& host_data) {
    const uint32_t count = gpu_view.num_neurons;
    if (host_data.membrane_potentials.size() != count ||
        host_data.g_excitatory.size() != count ||
        host_data.g_inhibitory.size() != count ||
        host_data.i_stimulus.size() != count ||
        host_data.refractory_counter.size() != count ||
        host_data.spikes.size() != count ||
        host_data.background_g_excitatory_synaptic_conductances.size() != count ||
        host_data.background_g_inhibitory_synaptic_conductances.size() != count ||
        host_data.background_excitatory_postsynaptic_r_traces.size() != count ||
        host_data.background_excitatory_postsynaptic_d_traces.size() != count ||
        host_data.background_inhibitory_postsynaptic_r_traces.size() != count ||
        host_data.background_inhibitory_postsynaptic_d_traces.size() != count) {
        throw std::invalid_argument("upload_neuron_group_view host data vectors must all match gpu_view.num_neurons");
    }

    require_upload_bounds("neuron_group", gpu_view.offset, count);

    cuda_check(cudaMemcpy(gpu_view.membrane_potentials, host_data.membrane_potentials.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D membrane_potentials slice");
    cuda_check(cudaMemcpy(gpu_view.g_excitatory, host_data.g_excitatory.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D g_excitatory slice");
    cuda_check(cudaMemcpy(gpu_view.g_inhibitory, host_data.g_inhibitory.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D g_inhibitory slice");
    cuda_check(cudaMemcpy(gpu_view.i_stimulus, host_data.i_stimulus.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D i_stimulus slice");
    cuda_check(cudaMemcpy(gpu_view.refractory_counter, host_data.refractory_counter.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D refractory_counter slice");
    cuda_check(cudaMemcpy(gpu_view.spikes, host_data.spikes.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D spikes slice");
    cuda_check(cudaMemcpy(gpu_view.background_g_excitatory_synaptic_conductances, host_data.background_g_excitatory_synaptic_conductances.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D background_g_excitatory_synaptic_conductances slice");
    cuda_check(cudaMemcpy(gpu_view.background_g_inhibitory_synaptic_conductances, host_data.background_g_inhibitory_synaptic_conductances.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D background_g_inhibitory_synaptic_conductances slice");
    cuda_check(cudaMemcpy(gpu_view.background_excitatory_postsynaptic_r_traces, host_data.background_excitatory_postsynaptic_r_traces.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D background_excitatory_postsynaptic_r_traces slice");
    cuda_check(cudaMemcpy(gpu_view.background_excitatory_postsynaptic_d_traces, host_data.background_excitatory_postsynaptic_d_traces.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D background_excitatory_postsynaptic_d_traces slice");
    cuda_check(cudaMemcpy(gpu_view.background_inhibitory_postsynaptic_r_traces, host_data.background_inhibitory_postsynaptic_r_traces.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D background_inhibitory_postsynaptic_r_traces slice");
    cuda_check(cudaMemcpy(gpu_view.background_inhibitory_postsynaptic_d_traces, host_data.background_inhibitory_postsynaptic_d_traces.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D background_inhibitory_postsynaptic_d_traces slice");
}

// NOT a slice for all data
void GpuSimulationState::upload_connection_view(const ConnectionView& gpu_view, const ConnectionInitHostData& host_data) {
    require_nonnegative_weights("connection weight_matrix", host_data.weight_matrix);
    
    std::string weight_matrix_name = lookup_weight_matrix_name(gpu_view.weight_matrix_idx);
    if (weight_matrix_name == "w_ee") {
        cuda_check(cudaMemcpy(d_w_ee_, host_data.weight_matrix.data(), host_data.weight_matrix.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D w_ee slice");
        cuda_check(cudaMemcpy(d_ee_postsynaptic_r_traces_, host_data.postsynaptic_r_traces.data(), num_excitatory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ee_postsynaptic_r_traces slice");
        cuda_check(cudaMemcpy(d_ee_postsynaptic_d_traces_, host_data.postsynaptic_d_traces.data(), num_excitatory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ee_postsynaptic_d_traces slice");
        cuda_check(cudaMemcpy(d_ee_summed_postsynaptic_conductances_, host_data.summed_postsynaptic_conductances.data(), num_excitatory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ee_summed_postsynaptic_conductances slice");
    } else if (weight_matrix_name == "w_ei") {
        cuda_check(cudaMemcpy(d_w_ei_ , host_data.weight_matrix.data(), host_data.weight_matrix.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D w_ei slice");
        cuda_check(cudaMemcpy(d_ei_postsynaptic_r_traces_, host_data.postsynaptic_r_traces.data(), num_excitatory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ei_postsynaptic_r_traces slice");
        cuda_check(cudaMemcpy(d_ei_postsynaptic_d_traces_, host_data.postsynaptic_d_traces.data(), num_excitatory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ei_postsynaptic_d_traces slice");
        cuda_check(cudaMemcpy(d_ei_summed_postsynaptic_conductances_, host_data.summed_postsynaptic_conductances.data(), num_excitatory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ei_summed_postsynaptic_conductances slice");
    } else if (weight_matrix_name == "w_ie") {
        cuda_check(cudaMemcpy(d_w_ie_, host_data.weight_matrix.data(), host_data.weight_matrix.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D w_ie slice");
        cuda_check(cudaMemcpy(d_ie_postsynaptic_r_traces_, host_data.postsynaptic_r_traces.data(), num_inhibitory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ie_postsynaptic_r_traces slice");
        cuda_check(cudaMemcpy(d_ie_postsynaptic_d_traces_, host_data.postsynaptic_d_traces.data(), num_inhibitory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ie_postsynaptic_d_traces slice");
        cuda_check(cudaMemcpy(d_ie_summed_postsynaptic_conductances_, host_data.summed_postsynaptic_conductances.data(), num_inhibitory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ie_summed_postsynaptic_conductances slice");
    } else if (weight_matrix_name == "w_ii") {
        cuda_check(cudaMemcpy(d_w_ii_, host_data.weight_matrix.data(), host_data.weight_matrix.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D w_ii slice");
        cuda_check(cudaMemcpy(d_ii_postsynaptic_r_traces_, host_data.postsynaptic_r_traces.data(), num_inhibitory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ii_postsynaptic_r_traces slice");
        cuda_check(cudaMemcpy(d_ii_postsynaptic_d_traces_, host_data.postsynaptic_d_traces.data(), num_inhibitory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ii_postsynaptic_d_traces slice");
        cuda_check(cudaMemcpy(d_ii_summed_postsynaptic_conductances_, host_data.summed_postsynaptic_conductances.data(), num_inhibitory_neurons_ * sizeof(float), cudaMemcpyHostToDevice), "H2D ii_summed_postsynaptic_conductances slice");
    } else {
        throw std::invalid_argument("Unknown weight matrix name: " + weight_matrix_name);
    }

}

// NOT a slice for all data
void GpuSimulationState::upload_stimulus_view(const StimulusView& gpu_view, const StimulusInitHostData& host_data) {
    const uint32_t count = num_stimulus_events_;
    if (host_data.timestep_event_count.size() != total_simulation_length_ ||
        host_data.stimulus_event_neuron_ids.size() != count ||
        host_data.stimulus_event_post_group_offsets.size() != count ||
        host_data.stimulus_event_amplitude_changes.size() != count) {
        throw std::invalid_argument("upload_stimulus_view host data vectors must match gpu_view.num_stimulus_events and total_simulation_length");
    }

    cuda_check(cudaMemcpy(d_timestep_event_count_, host_data.timestep_event_count.data(), total_simulation_length_ * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D timestep_event_count slice");
    cuda_check(cudaMemcpy(d_stimulus_event_post_group_offsets_, host_data.stimulus_event_post_group_offsets.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D stimulus_event_post_group_offsets slice");
    cuda_check(cudaMemcpy(d_stimulus_event_neuron_ids_, host_data.stimulus_event_neuron_ids.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D stimulus_event_neuron_ids slice");
    cuda_check(cudaMemcpy(d_stimulus_event_amplitude_changes_, host_data.stimulus_event_amplitude_changes.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D stimulus_event_amplitude_changes slice");
}

// YES a slice for all data
void GpuSimulationState::upload_plasticity_view(const PlasticityView& gpu_view, const PlasticityInitHostData& host_data) {
    if (host_data.pre_traces.size() != gpu_view.pre_group_num_neurons ||
        host_data.post_traces.size() != gpu_view.post_group_num_neurons) {
        throw std::invalid_argument("upload_plasticity_view host data vectors must match gpu_view.pre_group_num_neurons and gpu_view.post_group_num_neurons");
    }

    cuda_check(cudaMemcpy(gpu_view.pre_traces, host_data.pre_traces.data(), gpu_view.pre_group_num_neurons * sizeof(float), cudaMemcpyHostToDevice), "H2D pre_traces slice");
    cuda_check(cudaMemcpy(gpu_view.post_traces, host_data.post_traces.data(), gpu_view.post_group_num_neurons * sizeof(float), cudaMemcpyHostToDevice), "H2D post_traces slice");
    
    // check if host_data also has triplet_pre_traces and triplet_post_traces
    if (host_data.triplet_pre_traces.empty() && host_data.triplet_post_traces.empty()) {
        return;
    }
    if (host_data.triplet_pre_traces.size() != gpu_view.pre_group_num_neurons ||
        host_data.triplet_post_traces.size() != gpu_view.post_group_num_neurons) {
        throw std::invalid_argument("upload_plasticity_view host data triplet trace vectors must match gpu_view.pre_group_num_neurons and gpu_view.post_group_num_neurons");
    }
    cuda_check(cudaMemcpy(gpu_view.triplet_pre_traces, host_data.triplet_pre_traces.data(), gpu_view.pre_group_num_neurons * sizeof(float), cudaMemcpyHostToDevice), "H2D triplet_pre_traces slice");
    cuda_check(cudaMemcpy(gpu_view.triplet_post_traces, host_data.triplet_post_traces.data(), gpu_view.post_group_num_neurons * sizeof(float), cudaMemcpyHostToDevice), "H2D triplet_post_traces slice");
    cuda_check(cudaMemcpy(gpu_view.pre_weight_sums, host_data.pre_weight_sums.data(), gpu_view.pre_group_num_neurons * sizeof(float), cudaMemcpyHostToDevice), "H2D pre_weight_sums slice");
}

// need to fix at some point to actually compare initalized buffer 
void GpuSimulationState::require_upload_bounds(const std::string& field_name, const uint32_t offset, const uint32_t value_count) {
    if (offset > total_neurons_ || value_count > (total_neurons_ - offset)) {
        throw std::invalid_argument(
            field_name + " slice is out of bounds. offset=" + std::to_string(offset) +
            ", count=" + std::to_string(value_count) +
            ", total_neurons=" + std::to_string(total_neurons_)
        );
    }
}

void GpuSimulationState::require_nonnegative_weights(const std::string& field_name, const std::vector<float>& weights) {
    const auto first_negative = std::find_if(weights.begin(), weights.end(), [](const float value) {
        return value < 0.0f;
    });

    if (first_negative != weights.end()) {
        throw std::invalid_argument(field_name + " must be non-negative to preserve inhibitory routing contract");
    }
}

std::string GpuSimulationState::lookup_weight_matrix_name(uint32_t weight_matrix_idx) const {
    if (weight_matrix_idx >= weight_matrix_lookup_.size()) {
        throw std::out_of_range("Invalid weight matrix index: " + std::to_string(weight_matrix_idx));
    }
    return weight_matrix_lookup_.at(weight_matrix_idx);
}


std::vector<uint32_t> GpuSimulationState::download_and_flush_spike_history(const uint32_t offset, const uint32_t num_neurons, const uint32_t num_timesteps) {
    if (num_neurons == 0) {
        return {};
    }

    if (offset + num_neurons > total_neurons_) {
        throw std::out_of_range("download_and_flush_spike_history: neuron range out of bounds");
    }

    const size_t elems = static_cast<size_t>(num_neurons) * num_timesteps;
    const size_t bytes = elems * sizeof(uint32_t);

    std::vector<uint32_t> host_spike_history(elems);

    const uint32_t* src = d_spike_history_ + static_cast<size_t>(offset) * num_timesteps;

    cuda_check(cudaMemcpy(host_spike_history.data(), src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy spike history device->host failed");

    // Flush the downloaded region on device
    cuda_check(cudaMemset(const_cast<uint32_t*>(src), 0, bytes), "cudaMemset spike history flush failed");

    return host_spike_history;
}

std::vector<float> GpuSimulationState::download_and_flush_membrane_potential_history(const uint32_t offset, const uint32_t num_neurons, const uint32_t num_timesteps) {
    if (num_neurons == 0) {
        return {};
    }

    if (offset + num_neurons > total_neurons_) {
        throw std::out_of_range("download_and_flush_membrane_potential_history: neuron range out of bounds");
    }

    const size_t elems = static_cast<size_t>(num_neurons) * num_timesteps;
    const size_t bytes = elems * sizeof(float);

    std::vector<float> host_membrane_potential_history(elems);

    const float* src = d_membrane_potential_history_ + static_cast<size_t>(offset) * num_timesteps;

    cuda_check(cudaMemcpy(host_membrane_potential_history.data(), src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy membrane potential history device->host failed");

    // Flush the downloaded region on device
    cuda_check(cudaMemset(const_cast<float*>(src), 0, bytes), "cudaMemset membrane potential history flush failed");

    return host_membrane_potential_history;
}

std::vector<float> GpuSimulationState::download_weight_matrix(const uint32_t weight_matrix_idx) const {
    std::string weight_matrix_name = lookup_weight_matrix_name(weight_matrix_idx);
    std::vector<float> host_weight_matrix;
    if (weight_matrix_name == "w_ee") {
        host_weight_matrix.resize(num_excitatory_neurons_ * num_excitatory_neurons_);
        cuda_check(cudaMemcpy(host_weight_matrix.data(), d_w_ee_, host_weight_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H w_ee");
    } else if (weight_matrix_name == "w_ei") {
        host_weight_matrix.resize(num_excitatory_neurons_ * num_inhibitory_neurons_);
        cuda_check(cudaMemcpy(host_weight_matrix.data(), d_w_ei_, host_weight_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H w_ei");
    } else if (weight_matrix_name == "w_ie") {
        host_weight_matrix.resize(num_inhibitory_neurons_ * num_excitatory_neurons_);
        cuda_check(cudaMemcpy(host_weight_matrix.data(), d_w_ie_, host_weight_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H w_ie");
    } else if (weight_matrix_name == "w_ii") {
        host_weight_matrix.resize(num_inhibitory_neurons_ * num_inhibitory_neurons_);
        cuda_check(cudaMemcpy(host_weight_matrix.data(), d_w_ii_, host_weight_matrix.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H w_ii");
    } else {
        throw std::invalid_argument("Unknown weight matrix name: " + weight_matrix_name);
    }
    return host_weight_matrix;
}

uint32_t GpuSimulationState::total_allocated_bytes() const {
    uint32_t total_bytes = 0;
    // Kernel Neuron Group
    total_bytes += total_neurons_ * sizeof(float);  // d_membrane_potentials_
    total_bytes += total_neurons_ * sizeof(float);  // d_g_excitatory_
    total_bytes += total_neurons_ * sizeof(float);  // d_g_inhibitory_
    total_bytes += total_neurons_ * sizeof(float);  // d_i_stimulus_
    total_bytes += total_neurons_ * sizeof(uint32_t);  // d_refractory_counter_
    total_bytes += total_neurons_ * sizeof(uint32_t);  // d_spikes_
    total_bytes += total_neurons_ * sizeof(float);  // d_background_g_excitatory_synaptic_conductances_
    total_bytes += total_neurons_ * sizeof(float);  // d_background_g_inhibitory_synaptic_conductances_
    total_bytes += total_neurons_ * sizeof(float);  // d_background_excitatory_postsynaptic_r_traces_
    total_bytes += total_neurons_ * sizeof(float);  // d_background_excitatory_postsynaptic_d_traces_
    total_bytes += total_neurons_ * sizeof(float);  // d_background_inhibitory_postsynaptic_r_traces_
    total_bytes += total_neurons_ * sizeof(float);  // d_background_inhibitory_postsynaptic_d_traces_
    total_bytes += total_neurons_ * sizeof(curandStatePhilox4_32_10_t);  // d_neuron_rng_states_
    // Kernel Connection
    total_bytes += num_excitatory_neurons_ * num_excitatory_neurons_ * sizeof(float);  // d_w_ee
    total_bytes += num_excitatory_neurons_ * num_inhibitory_neurons_ * sizeof(float);  // d_w_ei
    total_bytes += num_inhibitory_neurons_ * num_excitatory_neurons_ * sizeof(float);  // d_w_ie
    total_bytes += num_inhibitory_neurons_ * num_inhibitory_neurons_ * sizeof(float);  // d_w_ii
    total_bytes += num_excitatory_neurons_ * sizeof(float);  // d_ee_postsynaptic_r_traces_
    total_bytes += num_excitatory_neurons_ * sizeof(float);  // d_ee_postsynaptic_d_traces_
    total_bytes += num_excitatory_neurons_ * sizeof(float);  // d_ee_summed_postsynaptic_conductances_
    total_bytes += num_excitatory_neurons_ * sizeof(float);  // d_ei_postsynaptic_r_traces_
    total_bytes += num_excitatory_neurons_ * sizeof(float);  // d_ei_postsynaptic_d_traces_
    total_bytes += num_excitatory_neurons_ * sizeof(float);  // d_ei_summed_postsynaptic_conductances_
    total_bytes += num_inhibitory_neurons_ * sizeof(float);  // d_ie_postsynaptic_r_traces_
    total_bytes += num_inhibitory_neurons_ * sizeof(float);  // d_ie_postsynaptic_d_traces_
    total_bytes += num_inhibitory_neurons_ * sizeof(float);  // d_ie_summed_postsynaptic_conductances_
    total_bytes += num_inhibitory_neurons_ * sizeof(float);  // d_ii_postsynaptic_r_traces_
    total_bytes += num_inhibitory_neurons_ * sizeof(float);  // d_ii_postsynaptic_d_traces_
    total_bytes += num_inhibitory_neurons_ * sizeof(float);  // d_ii_summed_postsynaptic_conductances_
    // Kernel Stimulus
    total_bytes += total_simulation_length_ * sizeof(uint32_t);  // d_timestep_event_count_
    total_bytes += num_stimulus_events_ * sizeof(uint32_t);  // d_stimulus_event_neuron_ids_
    total_bytes += num_stimulus_events_ * sizeof(uint32_t);  // d_stimulus_event_post_group_offsets_
    total_bytes += num_stimulus_events_ * sizeof(float);  // d_stimulus_event_amplitude_changes_
    // Kernel Plasticity
    total_bytes += num_plasticities_ * total_neurons_ * sizeof(float);  // d_pre_traces_
    total_bytes += num_plasticities_ * total_neurons_ * sizeof(float);  // d_post_traces_
    total_bytes += num_plasticities_ * total_neurons_ * sizeof(float);  // d_triplet_pre_traces_
    total_bytes += num_plasticities_ * total_neurons_ * sizeof(float);  // d_triplet_post_traces_
    total_bytes += num_plasticities_ * total_neurons_ * sizeof(float);  // d_pre_weight_sums_
    // Kernel Monitors
    total_bytes += total_neurons_ * buryn::recording_interval_steps * sizeof(uint32_t);  // d_spike_history_
    total_bytes += total_neurons_ * buryn::recording_interval_steps * sizeof(float);  // d_membrane_potential_history_

    return total_bytes;
}

std::string GpuSimulationState::memory_report() const {
    const uint32_t total_bytes = total_allocated_bytes();
    const double total_megabytes = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    return "Total GPU Memory Allocated: " + std::to_string(total_bytes) +
           " bytes (" + std::to_string(total_megabytes) + " MB)";
}

void GpuSimulationState::free_device_buffers() {
    // Kernel Neuron Group
    if (d_membrane_potentials_ != nullptr) {
        cudaFree(d_membrane_potentials_);
        d_membrane_potentials_ = nullptr;
    }
    if (d_g_excitatory_ != nullptr) {
        cudaFree(d_g_excitatory_);
        d_g_excitatory_ = nullptr;
    }
    if (d_g_inhibitory_ != nullptr) {
        cudaFree(d_g_inhibitory_);
        d_g_inhibitory_ = nullptr;
    }
    if (d_i_stimulus_ != nullptr) {
        cudaFree(d_i_stimulus_);
        d_i_stimulus_ = nullptr;
    }
    if (d_refractory_counter_ != nullptr) {
        cudaFree(d_refractory_counter_);
        d_refractory_counter_ = nullptr;
    }
    if (d_spikes_ != nullptr) {
        cudaFree(d_spikes_);
        d_spikes_ = nullptr;
    }
    if (d_background_g_excitatory_synaptic_conductances_ != nullptr) {
        cudaFree(d_background_g_excitatory_synaptic_conductances_);
        d_background_g_excitatory_synaptic_conductances_ = nullptr;
    }
    if (d_background_g_inhibitory_synaptic_conductances_ != nullptr) {
        cudaFree(d_background_g_inhibitory_synaptic_conductances_);
        d_background_g_inhibitory_synaptic_conductances_ = nullptr;
    }
    if (d_background_excitatory_postsynaptic_r_traces_ != nullptr) {
        cudaFree(d_background_excitatory_postsynaptic_r_traces_);
        d_background_excitatory_postsynaptic_r_traces_ = nullptr;
    }
    if (d_background_excitatory_postsynaptic_d_traces_ != nullptr) {
        cudaFree(d_background_excitatory_postsynaptic_d_traces_);
        d_background_excitatory_postsynaptic_d_traces_ = nullptr;
    }
    if (d_background_inhibitory_postsynaptic_r_traces_ != nullptr) {
        cudaFree(d_background_inhibitory_postsynaptic_r_traces_);
        d_background_inhibitory_postsynaptic_r_traces_ = nullptr;
    }
    if (d_background_inhibitory_postsynaptic_d_traces_ != nullptr) {
        cudaFree(d_background_inhibitory_postsynaptic_d_traces_);
        d_background_inhibitory_postsynaptic_d_traces_ = nullptr;
    }
    if (d_neuron_rng_states_ != nullptr) {
        cudaFree(d_neuron_rng_states_);
        d_neuron_rng_states_ = nullptr;
    }
    // Kernel Connection
    if (d_w_ee_ != nullptr) {
        cudaFree(d_w_ee_);
        d_w_ee_ = nullptr;
    }
    if (d_w_ei_ != nullptr) {
        cudaFree(d_w_ei_);
        d_w_ei_ = nullptr;
    }
    if (d_w_ie_ != nullptr) {
        cudaFree(d_w_ie_);
        d_w_ie_ = nullptr;
    }
    if (d_w_ii_ != nullptr) {
        cudaFree(d_w_ii_);
        d_w_ii_ = nullptr;
    }
    if (d_ee_postsynaptic_r_traces_ != nullptr) {
        cudaFree(d_ee_postsynaptic_r_traces_);
        d_ee_postsynaptic_r_traces_ = nullptr;
    }
    if (d_ee_postsynaptic_d_traces_ != nullptr) {
        cudaFree(d_ee_postsynaptic_d_traces_);
        d_ee_postsynaptic_d_traces_ = nullptr;
    }
    if (d_ee_summed_postsynaptic_conductances_ != nullptr) {
        cudaFree(d_ee_summed_postsynaptic_conductances_);
        d_ee_summed_postsynaptic_conductances_ = nullptr;
    }
    if (d_ei_postsynaptic_r_traces_ != nullptr) {
        cudaFree(d_ei_postsynaptic_r_traces_);
        d_ei_postsynaptic_r_traces_ = nullptr;
    }
    if (d_ei_postsynaptic_d_traces_ != nullptr) {
        cudaFree(d_ei_postsynaptic_d_traces_);
        d_ei_postsynaptic_d_traces_ = nullptr;
    }
    if (d_ei_summed_postsynaptic_conductances_ != nullptr) {
        cudaFree(d_ei_summed_postsynaptic_conductances_);
        d_ei_summed_postsynaptic_conductances_ = nullptr;
    }
    if (d_ie_postsynaptic_r_traces_ != nullptr) {
        cudaFree(d_ie_postsynaptic_r_traces_);
        d_ie_postsynaptic_r_traces_ = nullptr;
    }
    if (d_ie_postsynaptic_d_traces_ != nullptr) {
        cudaFree(d_ie_postsynaptic_d_traces_);
        d_ie_postsynaptic_d_traces_ = nullptr;
    }
    if (d_ie_summed_postsynaptic_conductances_ != nullptr) {
        cudaFree(d_ie_summed_postsynaptic_conductances_);
        d_ie_summed_postsynaptic_conductances_ = nullptr;
    }
    if (d_ii_postsynaptic_r_traces_ != nullptr) {
        cudaFree(d_ii_postsynaptic_r_traces_);
        d_ii_postsynaptic_r_traces_ = nullptr;
    }
    if (d_ii_postsynaptic_d_traces_ != nullptr) {
        cudaFree(d_ii_postsynaptic_d_traces_);
        d_ii_postsynaptic_d_traces_ = nullptr;
    }
    if (d_ii_summed_postsynaptic_conductances_ != nullptr) {
        cudaFree(d_ii_summed_postsynaptic_conductances_);
        d_ii_summed_postsynaptic_conductances_ = nullptr;
    }
    if (cublas_handle_ != nullptr) {
        cublas_check(cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_)), "cublasDestroy");
        cublas_handle_ = nullptr;
    }
    // Kernel Stimulus
    if (d_timestep_event_count_ != nullptr) {
        cudaFree(d_timestep_event_count_);
        d_timestep_event_count_ = nullptr;
    }
    if (d_stimulus_event_neuron_ids_ != nullptr) {
        cudaFree(d_stimulus_event_neuron_ids_);
        d_stimulus_event_neuron_ids_ = nullptr;
    }
    if (d_stimulus_event_post_group_offsets_ != nullptr) {
        cudaFree(d_stimulus_event_post_group_offsets_);
        d_stimulus_event_post_group_offsets_ = nullptr;
    }
    if (d_stimulus_event_amplitude_changes_ != nullptr) {
        cudaFree(d_stimulus_event_amplitude_changes_);
        d_stimulus_event_amplitude_changes_ = nullptr;
    }
    // Kernel Plasticity
    if (d_pre_traces_ != nullptr) {
        cudaFree(d_pre_traces_);
        d_pre_traces_ = nullptr;
    }
    if (d_post_traces_ != nullptr) {
        cudaFree(d_post_traces_);
        d_post_traces_ = nullptr;
    }
    if (d_triplet_pre_traces_ != nullptr) {
        cudaFree(d_triplet_pre_traces_);
        d_triplet_pre_traces_ = nullptr;
    }
    if (d_triplet_post_traces_ != nullptr) {
        cudaFree(d_triplet_post_traces_);
        d_triplet_post_traces_ = nullptr;
    }
    if (d_pre_weight_sums_ != nullptr) {
        cudaFree(d_pre_weight_sums_);
        d_pre_weight_sums_ = nullptr;
    }
    // Kernel Monitors
    if (d_spike_history_ != nullptr) {
        cudaFree(d_spike_history_);
        d_spike_history_ = nullptr;
    }
    if (d_membrane_potential_history_ != nullptr) {
        cudaFree(d_membrane_potential_history_);
        d_membrane_potential_history_ = nullptr;
    }
}