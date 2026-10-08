#ifndef STDP_PLASTICITY_H
#define STDP_PLASTICITY_H

#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include "buryn_definitions.h"
#include "plasticity.h"
#include "kernel_stdp_plasticity_update.h"

class StdpPlasticity : public Plasticity {
public:
    StdpPlasticity(Connection& connection,
                PlasticityParams params,
                GpuSimulationState& gpu_state);
    ~StdpPlasticity() override = default;

    PlasticityInitHostData initialize_data();
    void make_gpu_view();
    void upload_data(const PlasticityInitHostData& host_data);
    void update() override;
    
    private:
};

#endif  // STDP_PLASTICITY_H
