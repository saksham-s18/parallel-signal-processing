#ifndef MOVING_AVERAGE_OMP_HPP
#define MOVING_AVERAGE_OMP_HPP

#include <vector>
#include <string>
#include <omp.h>
#include "common.hpp"

/**
 * OpenMP Multi-Threaded 1D Moving-Average Filter.
 * 
 * Mathematical Formulation:
 *   For an input signal x of length N and odd window size W = 2k + 1:
 *     y[i] = (1 / W) * sum_{j = -k}^{k} x[clamp(i + j, 0, N - 1)]
 * 
 * Boundary Strategy:
 *   Edge Replication (Clamping): min(max(p, 0), N - 1)
 * 
 * Parallelization Strategy:
 *   - Outer loop over sample indices i in [0, N-1] parallelized via #pragma omp parallel for
 *   - Static scheduling: schedule(static) partitions the N output elements evenly across threads
 *   - Each thread computes its assigned range of outputs completely independently
 *   - Shared input is read-only; output writes are mutually exclusive per index i (race-free)
 *   - Double precision accumulator matching the sequential reference baseline
 * 
 * @param input Input signal vector of length N
 * @param output Output smoothed signal vector (resized to N)
 * @param windowSize Odd window size W = 2k + 1
 * @param numThreads Desired OpenMP thread count (if <= 0, uses omp_get_max_threads())
 * @param errorMsg Optional pointer to string to receive error details on invalid input
 * @return true on success, false on invalid parameter
 */
bool movingAverageOpenMP(const std::vector<SampleType>& input,
                         std::vector<SampleType>& output,
                         int windowSize,
                         int numThreads = 0,
                         std::string* errorMsg = nullptr);

#endif // MOVING_AVERAGE_OMP_HPP
