#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

inline void cuda_check(cudaError_t code, const char* what) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(code));
    }
}

inline void cublas_check(cublasStatus_t code, const char* what) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(code)));
    }
}

#endif  // CUDA_UTILS_H
