#include "plasticity.h"

Plasticity::Plasticity(Connection& connection, PlasticityParams params, GpuSimulationState& gpu_state)
    : connection_(connection), params_(params), gpu_state_(gpu_state) {
}
