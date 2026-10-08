#include "connection.h"

Connection::Connection(NeuronGroup& pre_group,
    NeuronGroup& post_group,
    std::vector<std::vector<float>>& weight_matrix,
    ConnectionParams params,
    GpuSimulationState& gpu_state)
    : pre_group_(pre_group),
    post_group_(post_group),
    params_(params),
    gpu_state_(gpu_state) {
        pre_group_offset_ = pre_group_.offset();
        post_group_offset_ = post_group_.offset();
        pre_group_num_neurons_ = pre_group_.num_neurons();
        post_group_num_neurons_ = post_group_.num_neurons();

        // Check weight_matrix dimensions
        if (weight_matrix.size() != pre_group_.num_neurons() || weight_matrix[0].size() != post_group_.num_neurons()) {
            throw std::invalid_argument("Weight matrix size does not match the number of connections between pre and post groups.");
        }
        
        ConnectionInitHostData host_data = initialize_data(weight_matrix);
        make_gpu_view();
        upload_data(host_data);
}

ConnectionInitHostData Connection::initialize_data(const std::vector<std::vector<float>>& weight_matrix) {
    ConnectionInitHostData host_data;
    host_data.weight_matrix.resize(pre_group_.num_neurons() * post_group_.num_neurons(), 0.0f);
    // Flatten the 2D weight matrix into a 1D vector in row-major order
    for (uint32_t i = 0; i < pre_group_.num_neurons(); ++i) {
        for (uint32_t j = 0; j < post_group_.num_neurons(); ++j) {
            host_data.weight_matrix[i * post_group_.num_neurons() + j] = weight_matrix[i][j];
            // For debugging:
            // std::cout << "Weight from pre neuron " << i << " to post neuron " << j << ": " << weight_matrix[i][j] << std::endl;
        }
    }
    host_data.postsynaptic_d_traces.resize(pre_group_.num_neurons(), 0.0f);
    host_data.postsynaptic_r_traces.resize(pre_group_.num_neurons(), 0.0f);
    host_data.summed_postsynaptic_conductances.resize(pre_group_.num_neurons(), 0.0f);
    return host_data;
}

void Connection::make_gpu_view() {
    // CPU only data
    gpu_view_.postsynaptic_r_trace_decay = std::exp(-1.0f * params_.timestep / params_.rise_time);
    gpu_view_.postsynaptic_d_trace_decay = std::exp(-1.0f * params_.timestep / params_.decay_time);
    gpu_view_.postsynaptic_kernel_normalization_value = params_.postsynaptic_kernel_normalization_value;
    gpu_view_.weight_matrix_idx = params_.weight_matrix_idx;
    gpu_view_.pre_group_is_inhibtory = pre_group_.is_inhibitory();

    // Offsets and sizes
    gpu_view_.pre_group_offset = pre_group_offset_;
    gpu_view_.post_group_offset = post_group_offset_;
    gpu_view_.pre_group_num_neurons = pre_group_.num_neurons();
    gpu_view_.post_group_num_neurons = post_group_.num_neurons();

    // Pointers to GPU data
    std::string weight_matrix_name = gpu_state_.lookup_weight_matrix_name(params_.weight_matrix_idx);
    if (weight_matrix_name == "w_ee") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ee();
        gpu_view_.postsynaptic_r_traces = gpu_state_.d_ee_postsynaptic_r_traces();
        gpu_view_.postsynaptic_d_traces = gpu_state_.d_ee_postsynaptic_d_traces();
        gpu_view_.summed_postsynaptic_conductances = gpu_state_.d_ee_summed_postsynaptic_conductances();
    } else if (weight_matrix_name == "w_ei") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ei();
        gpu_view_.postsynaptic_r_traces = gpu_state_.d_ei_postsynaptic_r_traces();
        gpu_view_.postsynaptic_d_traces = gpu_state_.d_ei_postsynaptic_d_traces();
        gpu_view_.summed_postsynaptic_conductances = gpu_state_.d_ei_summed_postsynaptic_conductances();
    } else if (weight_matrix_name == "w_ie") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ie();
        gpu_view_.postsynaptic_r_traces = gpu_state_.d_ie_postsynaptic_r_traces();
        gpu_view_.postsynaptic_d_traces = gpu_state_.d_ie_postsynaptic_d_traces();
        gpu_view_.summed_postsynaptic_conductances = gpu_state_.d_ie_summed_postsynaptic_conductances();
    } else if (weight_matrix_name == "w_ii") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ii();
        gpu_view_.postsynaptic_r_traces = gpu_state_.d_ii_postsynaptic_r_traces();
        gpu_view_.postsynaptic_d_traces = gpu_state_.d_ii_postsynaptic_d_traces();
        gpu_view_.summed_postsynaptic_conductances = gpu_state_.d_ii_summed_postsynaptic_conductances();
    } else {
        throw std::invalid_argument("Unknown weight matrix name: " + weight_matrix_name);
    }
    gpu_view_.post_group_g_excitatory = gpu_state_.d_g_excitatory() + post_group_offset_;
    gpu_view_.post_group_g_inhibitory = gpu_state_.d_g_inhibitory() + post_group_offset_;
    gpu_view_.spikes = gpu_state_.d_spikes() + pre_group_offset_;
    gpu_view_.cublas_handle = gpu_state_.cublas_handle();
}

void Connection::upload_data(const ConnectionInitHostData& host_data) {
    gpu_state_.upload_connection_view(gpu_view_, host_data);
}

void Connection::propagate() {
    kernel_connection_propagate_launch(gpu_view_);
}
