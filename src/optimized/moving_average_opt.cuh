#ifndef MOVING_AVERAGE_OPT_CUH
#define MOVING_AVERAGE_OPT_CUH

#include <vector>
#include <string>
#include "common.hpp"
#include "../cuda/moving_average_cuda.cuh"

/**
 * Optimized CUDA 1D Moving-Average Filter using Shared-Memory Tiling with Halo Cells.
 * 
 * Optimization Principles:
 *   1. Cooperative Global Memory Loading:
 *      Threads in a block cooperatively load a contiguous tile of the signal plus
 *      left and right halo elements (radius k = (W-1)/2) into on-chip shared memory.
 *   2. Massive Global Memory Bandwidth Reduction:
 *      Instead of each thread redundantly reading W samples from high-latency global memory
 *      (O(NW) global reads), all threads in the tile reuse shared memory data.
 *      Global memory reads drop from (BlockSize * W) to (BlockSize + 2*k) per block.
 *   3. Barrier Synchronization:
 *      __syncthreads() ensures all tile and halo elements are populated before computation.
 *   4. Zero-Conflict Shared Memory Reads:
 *      During stencil computation, consecutive threads read s_data[tid + j], which accesses
 *      consecutive 32-bit shared memory banks, avoiding bank conflicts.
 *   5. Accurate Divergence Characterization:
 *      Boundary clamping induces minor warp divergence during the cooperative load only
 *      for the first and last thread blocks of the grid. Interior blocks execute branch-free loads.
 * 
 * @param input Host input signal vector of length N
 * @param output Host output smoothed signal vector (resized to N)
 * @param windowSize Filter window size W (odd integer)
 * @param blockSize CUDA block size (threads per block, default 256)
 * @param timings Optional pointer to receive CUDA timing breakdown
 * @param errorMsg Optional pointer to string to receive error details on failure
 * @return true on success, false on error
 */
bool movingAverageCUDAOptimized(const std::vector<SampleType>& input,
                                std::vector<SampleType>& output,
                                int windowSize,
                                int blockSize = 256,
                                CUDATimings* timings = nullptr,
                                std::string* errorMsg = nullptr);

#endif // MOVING_AVERAGE_OPT_CUH
