#ifndef KERNEL_NEURON_GROUP_INIT_RNG_STATES_H
#define KERNEL_NEURON_GROUP_INIT_RNG_STATES_H

#include <cstddef>
#include <cstdint>

void kernel_init_neuron_rng_states_launch(
    void* d_neuron_rng_states_void,
    const uint32_t total_neurons,
    const unsigned long long seed
);

#endif  // KERNEL_NEURON_GROUP_INIT_RNG_STATES_H