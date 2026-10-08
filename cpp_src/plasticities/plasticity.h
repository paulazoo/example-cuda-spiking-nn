#ifndef PLASTICITY_H
#define PLASTICITY_H

#include "buryn_definitions.h"
#include "connection.h"
#include "plasticity_params.h"
#include "plasticity_view.h"
#include "gpu_simulation_state.h"

class Plasticity {
public:
    Plasticity(Connection& connection, PlasticityParams params, GpuSimulationState& gpu_state);
    virtual ~Plasticity() = default;

    virtual void update() = 0;

protected:
    Connection& connection_;

    // CPU only data
    PlasticityParams params_;

    // GPU data
    PlasticityView gpu_view_;
    GpuSimulationState& gpu_state_;
};

#endif  // PLASTICITY_H
