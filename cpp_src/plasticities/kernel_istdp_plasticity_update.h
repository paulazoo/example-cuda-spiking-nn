#ifndef KERNEL_ISTDP_PLASTICITY_UPDATE_H
#define KERNEL_ISTDP_PLASTICITY_UPDATE_H

#include <cstdint>

#include "plasticity_view.h"
#include "plasticity_params.h"

void kernel_istdp_plasticity_update_launch(PlasticityView gpu_view, const PlasticityParams params);

#endif  // KERNEL_ISTDP_PLASTICITY_UPDATE_H