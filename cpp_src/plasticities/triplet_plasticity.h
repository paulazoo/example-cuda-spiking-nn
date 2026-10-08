#ifndef TRIPLET_PLASTICITY_H
#define TRIPLET_PLASTICITY_H

#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include "buryn_definitions.h"
#include "plasticity.h"
#include "kernel_triplet_plasticity_update.h"

class TripletPlasticity : public Plasticity {
public:
    TripletPlasticity(Connection& connection,
                PlasticityParams params,
                GpuSimulationState& gpu_state);
    ~TripletPlasticity() override = default;

    PlasticityInitHostData initialize_data();
    void make_gpu_view();
    void upload_data(const PlasticityInitHostData& host_data);
    void update() override;
    
    private:
};

#endif  // TRIPLET_PLASTICITY_H
