#include "kernel_stimulus_apply.h"

#include <cuda_runtime.h>

#include "cuda_utils.h"

__global__ void kernel_stimulus_apply(StimulusView gpu_view,
                                      const uint32_t next_event_idx,
                                      const uint32_t num_events_to_apply) {
    const uint32_t thread_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= num_events_to_apply) {
        return;
    }

    const uint32_t assigned_event_idx = next_event_idx + thread_id;
    const uint32_t neuron_id = gpu_view.stimulus_event_neuron_ids[assigned_event_idx] +
        gpu_view.stimulus_event_post_group_offsets[assigned_event_idx];
    const float delta_amplitude = gpu_view.stimulus_event_amplitude_changes[assigned_event_idx];

    atomicAdd(&gpu_view.i_stimulus[neuron_id], delta_amplitude);
}

void kernel_stimulus_apply_launch(StimulusView gpu_view,
                                  const uint32_t next_event_idx,
                                  const uint32_t num_events_to_apply) {
    if (num_events_to_apply == 0) {
        return;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((num_events_to_apply + threads_per_block - 1) / threads_per_block);
    kernel_stimulus_apply<<<blocks, threads_per_block>>>(gpu_view, next_event_idx, num_events_to_apply);

    cuda_check(cudaGetLastError(), "kernel_stimulus_apply launch");
    cuda_check(cudaDeviceSynchronize(), "kernel_stimulus_apply sync");
}