#ifndef SPIKE_RECORDER_H
#define SPIKE_RECORDER_H

#include <fstream>
#include <cstddef>
#include <vector>
#include <numeric>

#include "buryn_definitions.h"
#include "monitor.h"
#include "neuron_group.h"

#include "spike_recorder_view.h"
#include "kernel_historicize_spikes.h"

class SpikeRecorder : public Monitor {
public:
    explicit SpikeRecorder(const NeuronGroup& group,
                            const std::string& output_filepath,
                            uint32_t recording_interval_steps,
                            GpuSimulationState& gpu_state);
    ~SpikeRecorder() = default;
    
    void record(uint32_t clock_step) override;
    void historicize(uint32_t clock_step) override;

    void make_view();

    const bool good() const { return output_.good(); }

private:
    const NeuronGroup& group_;

    void write_header();
    std::ofstream output_;
    uint32_t recording_interval_steps_;

    // GPU data
    SpikeRecorderView gpu_view_;
};

#endif  // SPIKE_RECORDER_H
