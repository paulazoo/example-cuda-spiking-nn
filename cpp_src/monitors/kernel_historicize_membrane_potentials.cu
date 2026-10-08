#include "kernel_historicize_membrane_potentials.h"

#include <cuda_runtime.h>

#include "cuda_utils.h"

__global__ void kernel_historicize_membrane_potentials(
    const float* __restrict__ d_membrane_potentials,
    float* __restrict__ membrane_potential_history,
    uint32_t total_neurons,
    uint32_t recording_interval_steps,
    uint32_t clock_step) {
    uint32_t neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= total_neurons) return;

    // Circular time index: 0..999
    uint32_t slot = clock_step % recording_interval_steps;

    // Layout: membrane_potential_history[neuron_id][time_slot]
    membrane_potential_history[neuron_id * recording_interval_steps + slot] = d_membrane_potentials[neuron_id];
}

void kernel_historicize_membrane_potentials_launch(
    const float* d_membrane_potentials,
    float* membrane_potential_history,
    uint32_t total_neurons,
    uint32_t recording_interval_steps,
    uint32_t clock_step) {

    uint32_t threads_per_block = 256;
    uint32_t blocks = (total_neurons + threads_per_block - 1) / threads_per_block;
    kernel_historicize_membrane_potentials<<<blocks, threads_per_block>>>(
        d_membrane_potentials, membrane_potential_history, total_neurons, recording_interval_steps, clock_step);
    
    cuda_check(cudaGetLastError(), "kernel_historicize_membrane_potentials launch");
    cuda_check(cudaDeviceSynchronize(), "kernel_historicize_membrane_potentials sync");
}