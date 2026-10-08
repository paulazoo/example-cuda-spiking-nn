#include "triplet_plasticity.h"

TripletPlasticity::TripletPlasticity(Connection& connection,
                                PlasticityParams params,
                                GpuSimulationState& gpu_state)
    : Plasticity(connection, params, gpu_state) {
    PlasticityInitHostData initial_host_data = initialize_data();
    make_gpu_view();
    upload_data(initial_host_data);
}


// NOTE: A little redundant because non-pre_group pre_traces and non-post_group post_traces values aren't touched, but simpler for now
PlasticityInitHostData TripletPlasticity::initialize_data() {
    PlasticityInitHostData host_data;
    host_data.pre_traces.resize(connection_.pre_group_num_neurons(), 0.0f);
    host_data.post_traces.resize(connection_.post_group_num_neurons(), 0.0f);
    host_data.triplet_pre_traces.resize(connection_.pre_group_num_neurons(), 0.0f);
    host_data.triplet_post_traces.resize(connection_.post_group_num_neurons(), 0.0f);
    
    // fill pre_weight_sums based on initial weight matrix values
    host_data.pre_weight_sums.resize(connection_.post_group_num_neurons(), 0.0f);
    std::vector<float> initial_weights = gpu_state_.download_weight_matrix(connection_.weight_matrix_idx());
    for (uint32_t post_neuron_idx = 0; post_neuron_idx < connection_.post_group_num_neurons(); ++post_neuron_idx) {
        float weight_sum = 0.0f;
        for (uint32_t pre_neuron_idx = 0; pre_neuron_idx < connection_.pre_group_num_neurons(); ++pre_neuron_idx) {
            weight_sum += initial_weights[(pre_neuron_idx * connection_.post_group_num_neurons()) + post_neuron_idx];
        }
        host_data.pre_weight_sums[post_neuron_idx] = weight_sum;
    }
    return host_data;
}


void TripletPlasticity::make_gpu_view() {
    gpu_view_.ltp_A = params_.ltp_A;
    gpu_view_.ltd_A = params_.ltd_A;
    gpu_view_.ltp_B = params_.ltp_B;
    gpu_view_.ltd_B = params_.ltd_B;
    gpu_view_.weight_min = params_.weight_min;
    gpu_view_.weight_max = params_.weight_max;
    gpu_view_.pre_trace_decay = params_.pre_trace_decay;
    gpu_view_.post_trace_decay = params_.post_trace_decay;
    gpu_view_.triplet_pre_trace_decay = params_.triplet_pre_trace_decay;
    gpu_view_.triplet_post_trace_decay = params_.triplet_post_trace_decay;

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
        throw std::runtime_error("Invalid weight matrix name in TripletPlasticity::make_gpu_view");
    }
    gpu_view_.pre_traces = gpu_state_.d_pre_traces() + (gpu_view_.plasticity_offset * gpu_state_.total_neurons() + gpu_view_.pre_group_offset);
    gpu_view_.post_traces = gpu_state_.d_post_traces() + (gpu_view_.plasticity_offset * gpu_state_.total_neurons() + gpu_view_.post_group_offset);
    gpu_view_.triplet_pre_traces = gpu_state_.d_triplet_pre_traces() + (gpu_view_.plasticity_offset * gpu_state_.total_neurons() + gpu_view_.pre_group_offset);
    gpu_view_.triplet_post_traces = gpu_state_.d_triplet_post_traces() + (gpu_view_.plasticity_offset * gpu_state_.total_neurons() + gpu_view_.post_group_offset);
    gpu_view_.pre_weight_sums = gpu_state_.d_pre_weight_sums() + (gpu_view_.plasticity_offset * gpu_state_.total_neurons() + gpu_view_.post_group_offset);
}


void TripletPlasticity::upload_data(const PlasticityInitHostData& host_data) {
    gpu_state_.upload_plasticity_view(gpu_view_, host_data);
}


void TripletPlasticity::update() {
    kernel_triplet_plasticity_update_normalize_launch(gpu_view_, params_);
}