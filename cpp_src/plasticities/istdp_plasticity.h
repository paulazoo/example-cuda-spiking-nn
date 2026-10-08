#ifndef ISTDP_PLASTICITY_H
#define ISTDP_PLASTICITY_H

#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include "buryn_definitions.h"
#include "plasticity.h"
#include "kernel_istdp_plasticity_update.h"

class IstdpPlasticity : public Plasticity {
public:
    IstdpPlasticity(Connection& connection,
                PlasticityParams params,
                GpuSimulationState& gpu_state);
    ~IstdpPlasticity() override = default;

    PlasticityInitHostData initialize_data();
    void make_gpu_view();
    void upload_data(const PlasticityInitHostData& host_data);
    void update() override;
    
    private:
};

#endif  // ISTDP_PLASTICITY_H
