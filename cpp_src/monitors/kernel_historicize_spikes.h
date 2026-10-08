#ifndef KERNEL_HISTORICIZE_SPIKES_H
#define KERNEL_HISTORICIZE_SPIKES_H

#include <cstddef>
#include <cstdint>

void kernel_historicize_spikes_launch(
    const uint32_t* d_spikes, 
    uint32_t* spikes_history, 
    uint32_t total_neurons, 
    uint32_t recording_interval_steps,
    uint32_t clock_step);

#endif  // KERNEL_HISTORICIZE_SPIKES_H
