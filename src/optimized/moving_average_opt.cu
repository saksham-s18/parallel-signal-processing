#include "moving_average_opt.cuh"
#include "../sequential/moving_average_seq.hpp"
#include <cuda_runtime.h>
#include <sstream>

#define CUDA_CHECK(call, msg) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            if (errorMsg) { \
                std::ostringstream oss; \
                oss << "[CUDA Opt Error] " << msg << ": " << cudaGetErrorString(err) \
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
                oss << "[CUDA Opt Error] " << msg << ": " << cudaGetErrorString(err) \
                    << " (" << __FILE__ << ":" << __LINE__ << ")"; \
                *errorMsg = oss.str(); \
            } \
            cleanupAction; \
            return false; \
        } \
    } while (0)

/**
 * Shared-Memory Tiled CUDA Kernel for 1D Moving-Average Filter.
 * 
 * Each block processes a tile of size blockDim.x.
 * Required shared memory size per block: (blockDim.x + 2*k) * sizeof(SampleType).
 */
__global__ void movingAverageSharedMemKernel(const SampleType* __restrict__ d_input,
                                             SampleType* __restrict__ d_output,
                                             size_t n,
                                             int windowSize,
                                             int k,
                                             double invW) {
    extern __shared__ SampleType s_data[];

    const int tid = threadIdx.x;
    const int sharedSize = blockDim.x + 2 * k;
    const int64_t globalTileStart = static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t globalWindowStart = globalTileStart - k;

    // 1. Cooperative Shared-Memory Loading (with Halo Elements)
    // Threads cooperatively load (blockDim.x + 2*k) elements into on-chip shared memory.
    // Boundary handling: Clamped edge replication identical to sequential baseline.
    for (int sIdx = tid; sIdx < sharedSize; sIdx += blockDim.x) {
        int64_t gIdx = globalWindowStart + sIdx;
        int64_t clampedIdx;
        if (gIdx < 0) {
            clampedIdx = 0;
        } else if (gIdx >= static_cast<int64_t>(n)) {
            clampedIdx = static_cast<int64_t>(n) - 1;
        } else {
            clampedIdx = gIdx;
        }
        s_data[sIdx] = d_input[clampedIdx];
    }

    // 2. Barrier Synchronization
    // Ensure that all shared memory loads (internal tile + left/right halos) have completed
    // before any thread begins reading from s_data.
    __syncthreads();

    // 3. Stencil Computation from Shared Memory
    const int64_t i = globalTileStart + tid;
    if (i < static_cast<int64_t>(n)) {
        double sum = 0.0;
        // The window for output sample i is centered at s_data[tid + k].
        // The window covers [ (tid + k) - k, (tid + k) + k ] = [ tid, tid + 2*k ].
        // All accesses read strictly from fast on-chip shared memory.
        #pragma unroll 4
        for (int j = 0; j < windowSize; ++j) {
            sum += static_cast<double>(s_data[tid + j]);
        }
        d_output[i] = static_cast<SampleType>(sum * invW);
    }
}

bool movingAverageCUDAOptimized(const std::vector<SampleType>& input,
                                std::vector<SampleType>& output,
                                int windowSize,
                                int blockSize,
                                CUDATimings* timings,
                                std::string* errorMsg) {
    const size_t n = input.size();

    // 1. Parameter Validation
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
    const size_t sharedMemBytes = (blockSize + 2 * k) * sizeof(SampleType);

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

    // 6. Launch Shared-Memory Tiled Kernel
    int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
    CUDA_CHECK_CLEANUP(cudaEventRecord(startKernel, 0), "Failed to record startKernel", cleanupAll());
    movingAverageSharedMemKernel<<<gridSize, blockSize, sharedMemBytes>>>(
        d_input, d_output, n, windowSize, k, invW);
    CUDA_CHECK_CLEANUP(cudaGetLastError(), "Optimized shared memory kernel launch failed", cleanupAll());
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
