#include <curand_kernel.h>

#include "buryn_definitions.h"
#include "cuda_utils.h"

__global__ void kernel_init_neuron_rng_states(
    void* d_neuron_rng_states_void,
    const uint32_t total_neurons,
    const unsigned long long seed
) {
    const uint32_t neuron_id =
        static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (neuron_id >= total_neurons) {
        return;
    }

    auto* neuron_rng_states =
        static_cast<curandStatePhilox4_32_10_t*>(d_neuron_rng_states_void);

    curand_init(
        seed,       // global seed for reproducibility
        neuron_id,  // unique sequence per neuron
        0,          // offset within that sequence
        &neuron_rng_states[neuron_id]
    );
}

void kernel_init_neuron_rng_states_launch(
    void* d_neuron_rng_states_void,
    const uint32_t total_neurons,
    const unsigned long long seed
) {
    const uint32_t threads_per_block = 256;
    const uint32_t num_blocks =
        (total_neurons + threads_per_block - 1) / threads_per_block;

    kernel_init_neuron_rng_states<<<num_blocks, threads_per_block>>>(
        d_neuron_rng_states_void,
        total_neurons,
        seed
    );
    
    cuda_check(cudaGetLastError(), "kernel_init_neuron_rng_states launch");
    cuda_check(cudaDeviceSynchronize(), "kernel_init_neuron_rng_states sync");
}

