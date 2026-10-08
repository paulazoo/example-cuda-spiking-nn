#include "kernel_neuron_group_evolve.h"

#include <cstdint>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuda_utils.h"
#include <curand_kernel.h>

__global__ void kernel_update_background_conductances(NeuronGroupView gpu_view) {
    const uint32_t neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (neuron_id >= gpu_view.num_neurons) {
        return;
    }
    auto* neuron_rng_states = static_cast<curandStatePhilox4_32_10_t*>(gpu_view.neuron_rng_states);
    curandStatePhilox4_32_10_t local_rng_state = neuron_rng_states[gpu_view.offset + neuron_id];

    // Decay existing traces
    float decayed_trace =
        gpu_view.background_excitatory_postsynaptic_r_traces[neuron_id] *
        gpu_view.background_excitatory_postsynaptic_r_trace_decay;
    if (decayed_trace < 1e-30f) {
        decayed_trace = 0.0f;
    }
    gpu_view.background_excitatory_postsynaptic_r_traces[neuron_id] = decayed_trace;

    decayed_trace =
        gpu_view.background_inhibitory_postsynaptic_r_traces[neuron_id] *
        gpu_view.background_inhibitory_postsynaptic_r_trace_decay;
    if (decayed_trace < 1e-30f) {
        decayed_trace = 0.0f;
    }
    gpu_view.background_inhibitory_postsynaptic_r_traces[neuron_id] = decayed_trace;

    decayed_trace =
        gpu_view.background_excitatory_postsynaptic_d_traces[neuron_id] *
        gpu_view.background_excitatory_postsynaptic_d_trace_decay;
    if (decayed_trace < 1e-30f) {
        decayed_trace = 0.0f;
    }
    gpu_view.background_excitatory_postsynaptic_d_traces[neuron_id] = decayed_trace;

    decayed_trace =
        gpu_view.background_inhibitory_postsynaptic_d_traces[neuron_id] *
        gpu_view.background_inhibitory_postsynaptic_d_trace_decay;
    if (decayed_trace < 1e-30f) {
        decayed_trace = 0.0f;
    }
    gpu_view.background_inhibitory_postsynaptic_d_traces[neuron_id] = decayed_trace;

    // Generate Poisson spike counts for this timestep
    const unsigned int background_e_spikes = (gpu_view.mean_background_excitatory_spikes > 0.0f) ? curand_poisson(&local_rng_state, gpu_view.mean_background_excitatory_spikes) : 0u;
    const unsigned int background_i_spikes = (gpu_view.mean_background_inhibitory_spikes > 0.0f) ? curand_poisson(&local_rng_state, gpu_view.mean_background_inhibitory_spikes) : 0u;

    // Add excitatory shot noise
    if (background_e_spikes > 0u) {
        const float exc_increment = static_cast<float>(background_e_spikes);

        gpu_view.background_excitatory_postsynaptic_r_traces[neuron_id] += exc_increment;
        gpu_view.background_excitatory_postsynaptic_d_traces[neuron_id] += exc_increment;
    }

    gpu_view.background_g_excitatory_synaptic_conductances[neuron_id] =
        (gpu_view.background_excitatory_postsynaptic_d_traces[neuron_id] -
         gpu_view.background_excitatory_postsynaptic_r_traces[neuron_id]) *
        gpu_view.background_excitatory_scaling_factor;

    // Add inhibitory shot noise
    if (background_i_spikes > 0u) {
        const float inh_increment = static_cast<float>(background_i_spikes);

        gpu_view.background_inhibitory_postsynaptic_r_traces[neuron_id] += inh_increment;
        gpu_view.background_inhibitory_postsynaptic_d_traces[neuron_id] += inh_increment;
    }

    gpu_view.background_g_inhibitory_synaptic_conductances[neuron_id] =
        (gpu_view.background_inhibitory_postsynaptic_d_traces[neuron_id] -
         gpu_view.background_inhibitory_postsynaptic_r_traces[neuron_id]) *
        gpu_view.background_inhibitory_scaling_factor;

    neuron_rng_states[gpu_view.offset + neuron_id] = local_rng_state;
}

__global__ void kernel_neuron_group_evolve(NeuronGroupView gpu_view) {
    const uint32_t neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (neuron_id >= gpu_view.num_neurons) {
        return;
    }

    // Membrane pinned to reset potential during refractory period
    if (gpu_view.refractory_counter[neuron_id] > 0) {
        --gpu_view.refractory_counter[neuron_id];
        gpu_view.spikes[neuron_id] = 0;
        gpu_view.membrane_potentials[neuron_id] = gpu_view.v_reset;
    } else {
        // Calculate v_next
        const float v = gpu_view.membrane_potentials[neuron_id];
        const float linear_leak_term = -1.0f * gpu_view.g_leak * (v - gpu_view.v_rest);
        const float exponential_depolarization_term =
            gpu_view.g_leak * gpu_view.depolarization_slope_factor *
            expf((v - gpu_view.v_effective_spike_threshold) / gpu_view.depolarization_slope_factor);
        const float excitatory_synaptic_term =
            -1.0f * gpu_view.g_excitatory[neuron_id] * (v - gpu_view.v_excitatory_reversal);
        const float inhibitory_synaptic_term =
            -1.0f * gpu_view.g_inhibitory[neuron_id] * (v - gpu_view.v_inhibitory_reversal);
        const float external_stimulus_term = gpu_view.i_stimulus[neuron_id];
        const float dv =
            (gpu_view.timestep / gpu_view.membrane_capacitance) *
            (linear_leak_term + exponential_depolarization_term +
             excitatory_synaptic_term + inhibitory_synaptic_term + external_stimulus_term);
        const float v_next = v + dv;

        // Check for spikes
        if (v_next >= gpu_view.v_threshold) {
            gpu_view.refractory_counter[neuron_id] = gpu_view.refractory_steps;
            gpu_view.spikes[neuron_id] = 1;  // Mark spike
            gpu_view.membrane_potentials[neuron_id] = gpu_view.v_threshold;  // Cap at v_threshold for voltage recording
        } else {
            gpu_view.spikes[neuron_id] = 0;  // No spike
            gpu_view.membrane_potentials[neuron_id] = v_next;
        }
    }

    // Reset conductances for next connection propagation
    gpu_view.g_excitatory[neuron_id] =
        gpu_view.background_g_excitatory_synaptic_conductances[neuron_id];
    gpu_view.g_inhibitory[neuron_id] =
        gpu_view.background_g_inhibitory_synaptic_conductances[neuron_id];
}


void kernel_neuron_group_evolve_launch(NeuronGroupView gpu_view) {
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((gpu_view.num_neurons + threads_per_block - 1) / threads_per_block);

    kernel_update_background_conductances<<<blocks, threads_per_block>>>(gpu_view);
    cuda_check(cudaGetLastError(), "kernel_update_background_conductances launch");

    kernel_neuron_group_evolve<<<blocks, threads_per_block>>>(gpu_view);
    cuda_check(cudaGetLastError(), "kernel_neuron_group_evolve launch");

    cuda_check(cudaDeviceSynchronize(), "kernel_neuron_evolve_launch sync");
}