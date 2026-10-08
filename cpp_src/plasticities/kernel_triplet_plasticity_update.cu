#include "kernel_triplet_plasticity_update.h"

#include <cuda_runtime.h>
#include "cuda_utils.h"

__device__ inline float triplet_clamp_weight(const float weight, const PlasticityParams params) {
    if (weight < params.weight_min) {
        return params.weight_min;
    }
    if (weight > params.weight_max) {
        return params.weight_max;
    }
    return weight;
}

__global__ void kernel_triplet_plasticity_decay_pre_trace(PlasticityView gpu_view, const PlasticityParams params) {
    const uint32_t pre_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pre_neuron_id >= gpu_view.pre_group_num_neurons) {
        return;
    }

    float decayed_pre_trace = gpu_view.pre_traces[pre_neuron_id] * params.pre_trace_decay;
    if (decayed_pre_trace < 1e-30f) {
        decayed_pre_trace = 0.0f;
    }
    gpu_view.pre_traces[pre_neuron_id] = decayed_pre_trace;

    float decayed_triplet_pre_trace = gpu_view.triplet_pre_traces[pre_neuron_id] * params.triplet_pre_trace_decay;
    if (decayed_triplet_pre_trace < 1e-30f) {
        decayed_triplet_pre_trace = 0.0f;
    }
    gpu_view.triplet_pre_traces[pre_neuron_id] = decayed_triplet_pre_trace;
}

__global__ void kernel_triplet_plasticity_decay_post_trace(PlasticityView gpu_view, const PlasticityParams params) {
    const uint32_t post_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (post_neuron_id >= gpu_view.post_group_num_neurons) {
        return;
    }

    float decayed_post_trace = gpu_view.post_traces[post_neuron_id] * params.post_trace_decay;
    if (decayed_post_trace < 1e-30f) {
        decayed_post_trace = 0.0f;
    }
    gpu_view.post_traces[post_neuron_id] = decayed_post_trace;

    float decayed_triplet_post_trace = gpu_view.triplet_post_traces[post_neuron_id] * params.triplet_post_trace_decay;
    if (decayed_triplet_post_trace < 1e-30f) {
        decayed_triplet_post_trace = 0.0f;
    }
    gpu_view.triplet_post_traces[post_neuron_id] = decayed_triplet_post_trace;
}

__global__ void kernel_triplet_plasticity_ltp(PlasticityView gpu_view, const PlasticityParams params) {
    const uint32_t post_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (post_neuron_id >= gpu_view.post_group_num_neurons) {
        return;
    }
    if (gpu_view.post_spikes[post_neuron_id] != 1) {
        return;
    }

    for (uint32_t pre_neuron_id = 0; pre_neuron_id < gpu_view.pre_group_num_neurons; ++pre_neuron_id) {
        if (pre_neuron_id == post_neuron_id) {
            continue; // Skip self-connections
        }
        const uint32_t weight_idx = (pre_neuron_id * gpu_view.post_group_num_neurons) + post_neuron_id;
        const float delta = gpu_view.pre_traces[pre_neuron_id] * (params.ltp_A + (params.ltp_B * gpu_view.triplet_post_traces[post_neuron_id]));
        const float updated_weight = gpu_view.weight_matrix[weight_idx] + delta;
        gpu_view.weight_matrix[weight_idx] = triplet_clamp_weight(updated_weight, params);
    }
}

__global__ void kernel_triplet_plasticity_ltd(PlasticityView gpu_view, const PlasticityParams params) {
    const uint32_t pre_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pre_neuron_id >= gpu_view.pre_group_num_neurons) {
        return;
    }
    if (gpu_view.pre_spikes[pre_neuron_id] != 1) {
        return;
    }

    const uint32_t row_start = pre_neuron_id * gpu_view.post_group_num_neurons;
    for (uint32_t post_neuron_id = 0; post_neuron_id < gpu_view.post_group_num_neurons; ++post_neuron_id) {
        if (pre_neuron_id == post_neuron_id) {
            continue; // Skip self-connections
        }
        const uint32_t weight_idx = row_start + post_neuron_id;
        const float delta = -1.0f * gpu_view.post_traces[post_neuron_id] * (params.ltd_A + (params.ltd_B * gpu_view.triplet_pre_traces[pre_neuron_id]));
        const float updated_weight = gpu_view.weight_matrix[weight_idx] + delta;
        gpu_view.weight_matrix[weight_idx] = triplet_clamp_weight(updated_weight, params);
    }
}

__global__ void kernel_triplet_plasticity_increment_post_trace(PlasticityView gpu_view) {
    const uint32_t post_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (post_neuron_id >= gpu_view.post_group_num_neurons) {
        return;
    }
    if (gpu_view.post_spikes[post_neuron_id] == 1) {
        gpu_view.post_traces[post_neuron_id] += 1.0f;
        gpu_view.triplet_post_traces[post_neuron_id] += 1.0f;
    }
}

__global__ void kernel_triplet_plasticity_increment_pre_trace(PlasticityView gpu_view) {
    const uint32_t pre_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pre_neuron_id >= gpu_view.pre_group_num_neurons) {
        return;
    }
    if (gpu_view.pre_spikes[pre_neuron_id] == 1) {
        gpu_view.pre_traces[pre_neuron_id] += 1.0f;
        gpu_view.triplet_pre_traces[pre_neuron_id] += 1.0f;
    }
}

__global__ void kernel_triplet_plasticity_normalize_synapses(PlasticityView gpu_view) {
    const uint32_t pre_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pre_neuron_id >= gpu_view.pre_group_num_neurons) {
        return;
    }

    const uint32_t row_start = pre_neuron_id * gpu_view.post_group_num_neurons;

    // Compute current sum of outgoing weights
    float current_sum = 0.0f;
    for (uint32_t post_neuron_id = 0; post_neuron_id < gpu_view.post_group_num_neurons; ++post_neuron_id) {
        const uint32_t weight_idx = row_start + post_neuron_id;
        current_sum += gpu_view.weight_matrix[weight_idx];
    }

    // Avoid divide-by-zero
    if (current_sum <= 0.0f) {
        return;
    }

    const float target_sum = gpu_view.pre_weight_sums[pre_neuron_id];
    const float scale = target_sum / current_sum;

    // Rescale weights
    for (uint32_t post_neuron_id = 0; post_neuron_id < gpu_view.post_group_num_neurons; ++post_neuron_id) {
        const uint32_t weight_idx = row_start + post_neuron_id;
        gpu_view.weight_matrix[weight_idx] *= scale;
    }
}

void kernel_triplet_plasticity_update_launch(PlasticityView gpu_view, const PlasticityParams params) {
    constexpr int threads_per_block = 256;

    const int pre_blocks = static_cast<int>((gpu_view.pre_group_num_neurons + threads_per_block - 1) / threads_per_block);
    if (pre_blocks > 0) {
        kernel_triplet_plasticity_decay_pre_trace<<<pre_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_decay_pre_trace launch");
    }

    const int post_blocks = static_cast<int>((gpu_view.post_group_num_neurons + threads_per_block - 1) / threads_per_block);
    if (post_blocks > 0) {
        kernel_triplet_plasticity_decay_post_trace<<<post_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_decay_post_trace launch");

        kernel_triplet_plasticity_ltp<<<post_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_ltp launch");
    }

    if (pre_blocks > 0) {
        kernel_triplet_plasticity_ltd<<<pre_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_ltd launch");

        kernel_triplet_plasticity_increment_pre_trace<<<pre_blocks, threads_per_block>>>(gpu_view);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_increment_pre_trace launch");
    }

    if (post_blocks > 0) {
        kernel_triplet_plasticity_increment_post_trace<<<post_blocks, threads_per_block>>>(gpu_view);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_increment_post_trace launch");
    }

    cuda_check(cudaDeviceSynchronize(), "kernel_triplet_plasticity_update sync");
}

void kernel_triplet_plasticity_update_normalize_launch(PlasticityView gpu_view, const PlasticityParams params) {
    constexpr int threads_per_block = 256;

    const int pre_blocks = static_cast<int>((gpu_view.pre_group_num_neurons + threads_per_block - 1) / threads_per_block);
    if (pre_blocks > 0) {
        kernel_triplet_plasticity_decay_pre_trace<<<pre_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_decay_pre_trace launch");
    }

    const int post_blocks = static_cast<int>((gpu_view.post_group_num_neurons + threads_per_block - 1) / threads_per_block);
    if (post_blocks > 0) {
        kernel_triplet_plasticity_decay_post_trace<<<post_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_decay_post_trace launch");

        kernel_triplet_plasticity_ltp<<<post_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_ltp launch");
    }

    if (pre_blocks > 0) {
        kernel_triplet_plasticity_ltd<<<pre_blocks, threads_per_block>>>(gpu_view, params);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_ltd launch");

        kernel_triplet_plasticity_normalize_synapses<<<pre_blocks, threads_per_block>>>(gpu_view);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_normalize_synapses launch");

        kernel_triplet_plasticity_increment_pre_trace<<<pre_blocks, threads_per_block>>>(gpu_view);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_increment_pre_trace launch");
    }

    if (post_blocks > 0) {
        kernel_triplet_plasticity_increment_post_trace<<<post_blocks, threads_per_block>>>(gpu_view);
        cuda_check(cudaGetLastError(), "kernel_triplet_plasticity_increment_post_trace launch");
    }

    cuda_check(cudaDeviceSynchronize(), "kernel_triplet_plasticity_update sync");
}