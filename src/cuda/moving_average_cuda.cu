#include "moving_average_cuda.cuh"
#include "../sequential/moving_average_seq.hpp"
#include <cuda_runtime.h>
#include <sstream>

#define CUDA_CHECK(call, msg) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            if (errorMsg) { \
                std::ostringstream oss; \
                oss << "[CUDA Error] " << msg << ": " << cudaGetErrorString(err) \
                    << " (" << __FILE__ << ":" << __LINE__ << ")"; \
                *errorMsg = oss.str(); \
            } \
            return false; \
        } \
    } while (0)

#define CUDA_CHECK_CLEANUP(call, msg, cleanupAction) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            if (errorMsg) { \
                std::ostringstream oss; \
                oss << "[CUDA Error] " << msg << ": " << cudaGetErrorString(err) \
                    << " (" << __FILE__ << ":" << __LINE__ << ")"; \
                *errorMsg = oss.str(); \
            } \
            cleanupAction; \
            return false; \
        } \
    } while (0)

/**
 * Baseline CUDA Kernel for 1D Moving-Average Filter.
 * 
 * Each thread computes exactly one output element at index i.
 * Time complexity per thread: O(W).
 * Boundary condition: Clamped edge replication identical to sequential baseline.
 */
__global__ void movingAverageBaselineKernel(const SampleType* __restrict__ d_input,
                                           SampleType* __restrict__ d_output,
                                           size_t n,
                                           int windowSize,
                                           int k,
                                           double invW) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<int64_t>(n)) {
        return;
    }

    double sum = 0.0; // 64-bit accumulator prevents precision loss
    for (int j = -k; j <= k; ++j) {
        int64_t rawIndex = i + j;
        int64_t clampedIndex;
        if (rawIndex < 0) {
            clampedIndex = 0;
        } else if (rawIndex >= static_cast<int64_t>(n)) {
            clampedIndex = static_cast<int64_t>(n) - 1;
        } else {
            clampedIndex = rawIndex;
        }
        sum += static_cast<double>(d_input[clampedIndex]);
    }

    d_output[i] = static_cast<SampleType>(sum * invW);
}

bool movingAverageCUDA(const std::vector<SampleType>& input,
                       std::vector<SampleType>& output,
                       int windowSize,
                       int blockSize,
                       CUDATimings* timings,
                       std::string* errorMsg) {
    const size_t n = input.size();

    // 1. Parameter Validation (reusing verified rules from sequential baseline)
    if (!validateWindowParameters(n, windowSize, errorMsg)) {
        return false;
    }

    if (blockSize <= 0 || blockSize > 1024) {
        if (errorMsg) {
            *errorMsg = "CUDA block size must be between 1 and 1024 (received " + std::to_string(blockSize) + ").";
        }
        return false;
    }

    // 2. Prepare Host Output Buffer
    if (output.size() != n) {
        output.resize(n);
    }

    const int k = (windowSize - 1) / 2;
    const double invW = 1.0 / static_cast<double>(windowSize);
    const size_t bytes = n * sizeof(SampleType);

    // 3. Allocate Device Memory
    SampleType* d_input = nullptr;
    SampleType* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes), "Failed to allocate device input buffer");
    CUDA_CHECK_CLEANUP(cudaMalloc(&d_output, bytes), "Failed to allocate device output buffer", {
        cudaFree(d_input);
    });

    // 4. Create CUDA Events for Fine-Grained Profiling
    cudaEvent_t startH2D, stopH2D;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startD2H, stopD2H;

    CUDA_CHECK_CLEANUP(cudaEventCreate(&startH2D), "Failed to create startH2D event", {
        cudaFree(d_input); cudaFree(d_output);
    });
    CUDA_CHECK_CLEANUP(cudaEventCreate(&stopH2D), "Failed to create stopH2D event", {
        cudaEventDestroy(startH2D); cudaFree(d_input); cudaFree(d_output);
    });
    CUDA_CHECK_CLEANUP(cudaEventCreate(&startKernel), "Failed to create startKernel event", {
        cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
        cudaFree(d_input); cudaFree(d_output);
    });
    CUDA_CHECK_CLEANUP(cudaEventCreate(&stopKernel), "Failed to create stopKernel event", {
        cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
        cudaEventDestroy(startKernel);
        cudaFree(d_input); cudaFree(d_output);
    });
    CUDA_CHECK_CLEANUP(cudaEventCreate(&startD2H), "Failed to create startD2H event", {
        cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
        cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
        cudaFree(d_input); cudaFree(d_output);
    });
    CUDA_CHECK_CLEANUP(cudaEventCreate(&stopD2H), "Failed to create stopD2H event", {
        cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
        cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
        cudaEventDestroy(startD2H);
        cudaFree(d_input); cudaFree(d_output);
    });

    auto cleanupAll = [&]() {
        cudaEventDestroy(startH2D);
        cudaEventDestroy(stopH2D);
        cudaEventDestroy(startKernel);
        cudaEventDestroy(stopKernel);
        cudaEventDestroy(startD2H);
        cudaEventDestroy(stopD2H);
        cudaFree(d_input);
        cudaFree(d_output);
    };

    // 5. Host-to-Device Transfer (H2D)
    CUDA_CHECK_CLEANUP(cudaEventRecord(startH2D, 0), "Failed to record startH2D", cleanupAll());
    CUDA_CHECK_CLEANUP(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice),
                       "Failed to copy input signal from host to device", cleanupAll());
    CUDA_CHECK_CLEANUP(cudaEventRecord(stopH2D, 0), "Failed to record stopH2D", cleanupAll());

    // 6. Launch Baseline CUDA Kernel
    int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
    CUDA_CHECK_CLEANUP(cudaEventRecord(startKernel, 0), "Failed to record startKernel", cleanupAll());
    movingAverageBaselineKernel<<<gridSize, blockSize>>>(d_input, d_output, n, windowSize, k, invW);
    CUDA_CHECK_CLEANUP(cudaGetLastError(), "Baseline kernel launch failed", cleanupAll());
    CUDA_CHECK_CLEANUP(cudaEventRecord(stopKernel, 0), "Failed to record stopKernel", cleanupAll());

    // 7. Device-to-Host Transfer (D2H)
    CUDA_CHECK_CLEANUP(cudaEventRecord(startD2H, 0), "Failed to record startD2H", cleanupAll());
    CUDA_CHECK_CLEANUP(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost),
                       "Failed to copy output signal from device to host", cleanupAll());
    CUDA_CHECK_CLEANUP(cudaEventRecord(stopD2H, 0), "Failed to record stopD2H", cleanupAll());

    // 8. Synchronize and Compute Event Elapsed Times
    CUDA_CHECK_CLEANUP(cudaEventSynchronize(stopD2H), "Failed to synchronize on stopD2H event", cleanupAll());

    if (timings) {
        float h2d = 0.0f, kernel = 0.0f, d2h = 0.0f;
        cudaEventElapsedTime(&h2d, startH2D, stopH2D);
        cudaEventElapsedTime(&kernel, startKernel, stopKernel);
        cudaEventElapsedTime(&d2h, startD2H, stopD2H);

        timings->h2dMs = h2d;
        timings->kernelMs = kernel;
        timings->d2hMs = d2h;
        timings->totalMs = h2d + kernel + d2h;
    }

    // 9. Free Device Resources
    cleanupAll();

    return true;
}
