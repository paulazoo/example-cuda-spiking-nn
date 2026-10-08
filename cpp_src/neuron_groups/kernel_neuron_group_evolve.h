#ifndef KERNEL_NEURON_GROUP_EVOLVE_H
#define KERNEL_NEURON_GROUP_EVOLVE_H

#include <cstddef>
#include <cstdint>

#include "neuron_group_params.h"
#include "neuron_group_view.h"

void kernel_neuron_group_evolve_launch(NeuronGroupView gpu_view);

#endif  // KERNEL_NEURON_GROUP_EVOLVE_H
