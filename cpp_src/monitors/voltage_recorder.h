#ifndef VOLTAGE_RECORDER_H
#define VOLTAGE_RECORDER_H

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "buryn_definitions.h"
#include "monitor.h"
#include "neuron_group.h"

#include "voltage_recorder_view.h"
#include "kernel_historicize_membrane_potentials.h"

class VoltageRecorder : public Monitor {
public:
    VoltageRecorder(const NeuronGroup& group, 
                    const std::string& output_filepath,
                    std::vector<uint32_t> selected_neurons,
                    uint32_t recording_interval_steps,
                    GpuSimulationState& gpu_state);
    ~VoltageRecorder() = default;
    
    void historicize(uint32_t clock_step) override;
    void record(uint32_t clock_step) override;
    
    void make_view();

    const bool good() const { return output_.good(); }

private:
    const NeuronGroup& group_;
    
    void write_header();

    std::ofstream output_;
    const std::string& output_filepath_;
    std::vector<uint32_t> selected_neurons_;
    uint32_t recording_interval_steps_;

    // GPU data
    VoltageRecorderView gpu_view_;
};

#endif  // VOLTAGE_RECORDER_H
