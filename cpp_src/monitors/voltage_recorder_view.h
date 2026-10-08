#ifndef VOLTAGE_RECORDER_VIEW_H
#define VOLTAGE_RECORDER_VIEW_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct VoltageRecorderView {
    // Offsets and sizes
    uint32_t num_neurons = 0;
    uint32_t offset = 0;

    // Pointers to GPU data
    float* membrane_potentials = nullptr;
    float* membrane_potential_history = nullptr;
};

#endif  // VOLTAGE_RECORDER_VIEW_H
