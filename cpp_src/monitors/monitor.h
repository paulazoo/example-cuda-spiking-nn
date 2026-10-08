#ifndef MONITOR_H
#define MONITOR_H

#include "buryn_definitions.h"
#include "gpu_simulation_state.h"

class Monitor {
public:
    Monitor(GpuSimulationState& gpu_state);
    virtual ~Monitor() = default;

    virtual void historicize(uint32_t clock_step) = 0;
    virtual void record(uint32_t clock_step) = 0;
    
protected:
    GpuSimulationState& gpu_state_;
};

#endif  // MONITOR_H
