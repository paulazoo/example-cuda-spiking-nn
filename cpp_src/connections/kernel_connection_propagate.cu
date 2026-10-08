#include "kernel_connection_propagate.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_utils.h"


__global__ void kernel_connection_decay_r_traces(ConnectionView gpu_view) {
    const uint32_t pre_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pre_neuron_id >= gpu_view.pre_group_num_neurons) {
        return;
    }

    float decayed_r_trace = gpu_view.postsynaptic_r_traces[pre_neuron_id] * gpu_view.postsynaptic_r_trace_decay;
    if (decayed_r_trace < 1e-30f) {
        decayed_r_trace = 0.0f;
    }

    gpu_view.postsynaptic_r_traces[pre_neuron_id] = decayed_r_trace;
}

__global__ void kernel_connection_decay_d_traces(ConnectionView gpu_view) {
    const uint32_t pre_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pre_neuron_id >= gpu_view.pre_group_num_neurons) {
        return;
    }

    float decayed_d_trace = gpu_view.postsynaptic_d_traces[pre_neuron_id] * gpu_view.postsynaptic_d_trace_decay;
    if (decayed_d_trace < 1e-30f) {
        decayed_d_trace = 0.0f;
    }
    gpu_view.postsynaptic_d_traces[pre_neuron_id] = decayed_d_trace;
}

__global__ void kernel_connection_update_conductances_with_spikes(ConnectionView gpu_view) {
    const uint32_t pre_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pre_neuron_id >= gpu_view.pre_group_num_neurons) {
        return;
    }

    if (gpu_view.spikes[pre_neuron_id] == 1) {
        gpu_view.postsynaptic_r_traces[pre_neuron_id] += gpu_view.postsynaptic_kernel_normalization_value;
        gpu_view.postsynaptic_d_traces[pre_neuron_id] += gpu_view.postsynaptic_kernel_normalization_value;
    }
    gpu_view.summed_postsynaptic_conductances[pre_neuron_id] = gpu_view.postsynaptic_d_traces[pre_neuron_id] - gpu_view.postsynaptic_r_traces[pre_neuron_id];
}

// __global__ void kernel_connection_propagate_postsynaptic_conductances(ConnectionView gpu_view) {
//     const uint32_t post_neuron_id = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
//     if (post_neuron_id >= gpu_view.post_group_num_neurons) {
//         return;
//     }

//     for (uint32_t pre_neuron_id = 0; pre_neuron_id < gpu_view.pre_group_num_neurons; ++pre_neuron_id) {
//         const float weight = gpu_view.weight_matrix[pre_neuron_id * gpu_view.post_group_num_neurons + post_neuron_id];
//         if (gpu_view.pre_group_is_inhibtory) {
//             gpu_view.post_group_g_inhibitory[post_neuron_id] += weight * gpu_view.summed_postsynaptic_conductances[pre_neuron_id];
//         } else {
//             gpu_view.post_group_g_excitatory[post_neuron_id] += weight * gpu_view.summed_postsynaptic_conductances[pre_neuron_id];
//         }
//     }
// }

void kernel_connection_propagate_launch(ConnectionView gpu_view) {
    constexpr int threads_per_block = 256;

    const int pre_blocks = (gpu_view.pre_group_num_neurons + threads_per_block - 1) / threads_per_block;
    kernel_connection_decay_d_traces<<<pre_blocks, threads_per_block>>>(gpu_view);
    cuda_check(cudaGetLastError(), "kernel_connection_decay_d_traces launch");
    kernel_connection_decay_r_traces<<<pre_blocks, threads_per_block>>>(gpu_view);
    cuda_check(cudaGetLastError(), "kernel_connection_decay_r_traces launch");

    kernel_connection_update_conductances_with_spikes<<<pre_blocks, threads_per_block>>>(gpu_view);
    cuda_check(cudaGetLastError(), "kernel_connection_update_conductances_with_spikes launch");

    // const int post_blocks = (gpu_view.post_group_num_neurons + threads_per_block - 1) / threads_per_block;
    // kernel_connection_propagate_postsynaptic_conductances<<<post_blocks, threads_per_block>>>(gpu_view);
    // cuda_check(cudaGetLastError(), "kernel_connection_propagate_postsynaptic_conductances launch");
    const float alpha = 1.0f;
    const float beta = 1.0f;
    float* target_post_group_g = gpu_view.pre_group_is_inhibtory ? gpu_view.post_group_g_inhibitory : gpu_view.post_group_g_excitatory;
    cublas_check(cublasSgemv(static_cast<cublasHandle_t>(gpu_view.cublas_handle),
            CUBLAS_OP_N,
            static_cast<int>(gpu_view.post_group_num_neurons),
            static_cast<int>(gpu_view.pre_group_num_neurons),
            &alpha,
            gpu_view.weight_matrix,
            static_cast<int>(gpu_view.post_group_num_neurons),
            gpu_view.summed_postsynaptic_conductances,
            1,
            &beta,
            target_post_group_g,
            1),
        "cublasSgemv connection accumulate");
    // NOTE: GEMV calculates y = alpha * A * x + beta * y
    // cublasSgemv signature:
    // cuBLAS handle
    // CUBLAS_OP_N NOTE: cuBLAS expects column-major order so re-interpret weight_matrix as column-major matrix A with shape [post, pre]
    // m=num_post_neurons number of rows of A
    // n=num_pre_neurons number of columns of A
    // alpha=1
    // A=weight_matrix
    // lda=post leading dimension of A (number of rows in column-major order)
    // x=summed_postsynaptic_conductances of shape [pre, 1]
    // incx=1 increment for elements of x (not 1 means every incx-th element is used)
    // beta=1
    // y=target_post_group_g
    // incy=1 increment for elements of y (not 1 means every incy-th element is used)

    cuda_check(cudaDeviceSynchronize(), "kernel_connection_propagate sync");
}