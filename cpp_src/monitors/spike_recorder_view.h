#ifndef SPIKE_RECORDER_VIEW_H
#define SPIKE_RECORDER_VIEW_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct SpikeRecorderView {
    // Offsets and sizes
    uint32_t num_neurons = 0;
    uint32_t offset = 0;

    // Pointers to GPU data
    uint32_t* spikes = nullptr;
    uint32_t* spike_history = nullptr;
};

#endif  // SPIKE_RECORDER_VIEW_H
