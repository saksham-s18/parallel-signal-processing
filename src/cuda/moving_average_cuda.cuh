#ifndef MOVING_AVERAGE_CUDA_CUH
#define MOVING_AVERAGE_CUDA_CUH

#include <vector>
#include <string>
#include "common.hpp"

// Detailed breakdown of CUDA execution times
struct CUDATimings {
    float h2dMs;       // Host-to-Device transfer time in milliseconds
    float kernelMs;    // Kernel execution time in milliseconds
    float d2hMs;       // Device-to-Host transfer time in milliseconds
    float totalMs;     // Total end-to-end time (h2d + kernel + d2h)
};

/**
 * Baseline CUDA 1D Moving-Average Filter.
 * 
 * Algorithm:
 *   - 1 CUDA thread computes exactly 1 output sample.
 *   - Uses naive O(N * W) global memory reads.
 *   - Applies edge-replication (clamping): min(max(p, 0), N - 1).
 *   - Accumulates using double precision matching the sequential baseline.
 * 
 * @param input Host input signal vector of length N
 * @param output Host output smoothed signal vector (resized to N)
 * @param windowSize Filter window size W (odd integer)
 * @param blockSize CUDA block size (threads per block, default 256)
 * @param timings Optional pointer to receive CUDA timing breakdown (via CUDA events)
 * @param errorMsg Optional pointer to string to receive error details on failure
 * @return true on success, false on error
 */
bool movingAverageCUDA(const std::vector<SampleType>& input,
                       std::vector<SampleType>& output,
                       int windowSize,
                       int blockSize = 256,
                       CUDATimings* timings = nullptr,
                       std::string* errorMsg = nullptr);

#endif // MOVING_AVERAGE_CUDA_CUH
