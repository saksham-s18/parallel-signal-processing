#ifndef MOVING_AVERAGE_SEQ_HPP
#define MOVING_AVERAGE_SEQ_HPP

#include <vector>
#include <string>
#include "common.hpp"

/**
 * Validates the parameters for the moving-average filter.
 * Rules:
 * - input must not be empty (N > 0)
 * - windowSize W must be positive (W > 0)
 * - windowSize W must be an odd integer (W % 2 == 1)
 * - windowSize W must not exceed the signal size (W <= N)
 * 
 * @param n Signal length
 * @param windowSize Filter window size W
 * @param errorMsg Pointer to string to populate with error description on failure
 * @return true if valid, false otherwise
 */
bool validateWindowParameters(size_t n, int windowSize, std::string* errorMsg = nullptr);

/**
 * Sequential 1D Moving-Average Noise Reduction Filter (Reference Baseline).
 * 
 * Mathematical Formulation:
 *   For an input signal x of length N and odd window size W = 2k + 1:
 *     y[i] = (1 / W) * sum_{j = -k}^{k} x[clamp(i + j, 0, N - 1)]
 * 
 * Boundary Strategy:
 *   Edge Replication (Clamping):
 *     clamp(p, 0, N - 1) = min(max(p, 0), N - 1)
 * 
 * Complexity:
 *   - Time: O(N * W) arithmetic operations (intentionally naive baseline)
 *   - Space: O(N) output storage (no auxiliary memory during computation)
 * 
 * @param input Input signal vector of length N
 * @param output Output smoothed signal vector (resized to N)
 * @param windowSize Odd window size W = 2k + 1
 * @param errorMsg Optional pointer to string to receive error details on invalid input
 * @return true on success, false on invalid parameter
 */
bool movingAverageSequential(const std::vector<SampleType>& input,
                            std::vector<SampleType>& output,
                            int windowSize,
                            std::string* errorMsg = nullptr);

#endif // MOVING_AVERAGE_SEQ_HPP
