#ifndef KERNEL_HISTORICIZE_MEMBRANE_POTENTIALS_H
#define KERNEL_HISTORICIZE_MEMBRANE_POTENTIALS_H

#include <cstddef>
#include <cstdint>

void kernel_historicize_membrane_potentials_launch(
    const float* d_membrane_potentials, 
    float* membrane_potential_history, 
    uint32_t total_neurons, 
    uint32_t recording_interval_steps,
    uint32_t clock_step);

#endif  // KERNEL_HISTORICIZE_MEMBRANE_POTENTIALS_H
