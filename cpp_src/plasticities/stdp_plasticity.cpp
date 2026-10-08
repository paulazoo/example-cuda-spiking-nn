#include "stdp_plasticity.h"

StdpPlasticity::StdpPlasticity(Connection& connection,
                                PlasticityParams params,
                                GpuSimulationState& gpu_state)
    : Plasticity(connection, params, gpu_state) {
    PlasticityInitHostData initial_host_data = initialize_data();
    make_gpu_view();
    upload_data(initial_host_data);
}


// NOTE: A little redundant because non-pre_group pre_traces and non-post_group post_traces values aren't touched, but simpler for now
PlasticityInitHostData StdpPlasticity::initialize_data() {
    PlasticityInitHostData host_data;
    host_data.pre_traces.resize(connection_.pre_group_num_neurons(), 0.0f);
    host_data.post_traces.resize(connection_.post_group_num_neurons(), 0.0f);
    return host_data;
}


void StdpPlasticity::make_gpu_view() {
    gpu_view_.ltp_A = params_.ltp_A;
    gpu_view_.ltd_A = params_.ltd_A;
    gpu_view_.weight_min = params_.weight_min;
    gpu_view_.weight_max = params_.weight_max;
    gpu_view_.pre_trace_decay = params_.pre_trace_decay;
    gpu_view_.post_trace_decay = params_.post_trace_decay;

    // Offsets and sizes
    gpu_view_.plasticity_offset = params_.plasticity_offset;
    gpu_view_.weight_matrix_idx = connection_.weight_matrix_idx();
    gpu_view_.pre_group_offset = connection_.pre_group_offset();
    gpu_view_.post_group_offset = connection_.post_group_offset();
    gpu_view_.pre_group_num_neurons = connection_.pre_group_num_neurons();
    gpu_view_.post_group_num_neurons = connection_.post_group_num_neurons();

    // Pointers to GPU data
    gpu_view_.pre_spikes = gpu_state_.d_spikes() + gpu_view_.pre_group_offset;
    gpu_view_.post_spikes = gpu_state_.d_spikes() + gpu_view_.post_group_offset;
    std::string weight_matrix_name = gpu_state_.lookup_weight_matrix_name(gpu_view_.weight_matrix_idx);
    if (weight_matrix_name == "w_ee") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ee();
    } else if (weight_matrix_name == "w_ei") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ei();
    } else if (weight_matrix_name == "w_ie") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ie();
    } else if (weight_matrix_name == "w_ii") {
        gpu_view_.weight_matrix = gpu_state_.d_w_ii();
    } else {
        throw std::runtime_error("Invalid weight matrix name in StdpPlasticity::make_gpu_view");
    }
    gpu_view_.pre_traces = gpu_state_.d_pre_traces() + (gpu_view_.plasticity_offset * gpu_state_.total_neurons() + gpu_view_.pre_group_offset);
    gpu_view_.post_traces = gpu_state_.d_post_traces() + (gpu_view_.plasticity_offset * gpu_state_.total_neurons() + gpu_view_.post_group_offset);
}


void StdpPlasticity::upload_data(const PlasticityInitHostData& host_data) {
    gpu_state_.upload_plasticity_view(gpu_view_, host_data);
}


void StdpPlasticity::update() {
    kernel_stdp_plasticity_update_launch(gpu_view_, params_);
}