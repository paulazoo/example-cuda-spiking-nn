#include "kernel_historicize_spikes.h"

#include <cuda_runtime.h>

#include "cuda_utils.h"

__global__ void kernel_historicize_spikes(
    const uint32_t* __restrict__ d_spikes,
    uint32_t* __restrict__ spike_history,
    uint32_t total_neurons,
    uint32_t recording_interval_steps,
    uint32_t clock_step) {
    uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= total_neurons) return;

    // Circular time index: 0..999
    uint32_t slot = clock_step % recording_interval_steps;

    // Layout: spike_history[neuron_id][time_slot]
    spike_history[neuron_id * recording_interval_steps + slot] = d_spikes[neuron_id];
}

void kernel_historicize_spikes_launch(
    const uint32_t* d_spikes,
    uint32_t* spike_history,
    uint32_t total_neurons,
    uint32_t recording_interval_steps,
    uint32_t clock_step) {

    uint32_t threads_per_block = 256;
    uint32_t blocks = (total_neurons + threads_per_block - 1) / threads_per_block;
    kernel_historicize_spikes<<<blocks, threads_per_block>>>(
        d_spikes, spike_history, total_neurons, recording_interval_steps, clock_step);
    
    cuda_check(cudaGetLastError(), "kernel_historicize_spikes launch");
    cuda_check(cudaDeviceSynchronize(), "kernel_historicize_spikes sync");
}