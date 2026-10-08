#ifndef KERNEL_STDP_PLASTICITY_UPDATE_H
#define KERNEL_STDP_PLASTICITY_UPDATE_H

#include <cstdint>

#include "plasticity_view.h"
#include "plasticity_params.h"

void kernel_stdp_plasticity_update_launch(PlasticityView gpu_view, const PlasticityParams params);

#endif  // KERNEL_STDP_PLASTICITY_UPDATE_H