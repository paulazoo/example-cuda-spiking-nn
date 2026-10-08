#ifndef KERNEL_TRIPLET_PLASTICITY_UPDATE_H
#define KERNEL_TRIPLET_PLASTICITY_UPDATE_H

#include <cstdint>

#include "plasticity_view.h"
#include "plasticity_params.h"

void kernel_triplet_plasticity_update_launch(PlasticityView gpu_view, const PlasticityParams params);
void kernel_triplet_plasticity_update_normalize_launch(PlasticityView gpu_view, const PlasticityParams params);

#endif  // KERNEL_TRIPLET_PLASTICITY_UPDATE_H