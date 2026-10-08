#ifndef WEIGHT_RECORDER_H
#define WEIGHT_RECORDER_H

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "buryn_definitions.h"
#include "monitor.h"
#include "connection.h"

class WeightRecorder : public Monitor {
public:
    WeightRecorder(const Connection& connection, 
                    const std::string& output_filepath,
                    GpuSimulationState& gpu_state);
    ~WeightRecorder() = default;

    void record(uint32_t clock_step) override;
    void historicize(uint32_t clock_step) override;

    const bool good() const { return output_.good(); }

private:
    void write_header();

    const Connection& connection_;
    std::ofstream output_;
    
    const std::string output_filepath_;
};

#endif  // WEIGHT_RECORDER_H
