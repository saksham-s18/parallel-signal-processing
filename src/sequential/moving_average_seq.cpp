#include "moving_average_seq.hpp"
#include <sstream>

bool validateWindowParameters(size_t n, int windowSize, std::string* errorMsg) {
    if (n == 0) {
        if (errorMsg) *errorMsg = "Signal length N must be greater than 0.";
        return false;
    }
    if (windowSize <= 0) {
        if (errorMsg) *errorMsg = "Window size W must be a positive integer (received " + std::to_string(windowSize) + ").";
        return false;
    }
    if (windowSize % 2 == 0) {
        if (errorMsg) *errorMsg = "Window size W must be an odd integer (received even value " + std::to_string(windowSize) + ").";
        return false;
    }
    if (static_cast<size_t>(windowSize) > n) {
        if (errorMsg) *errorMsg = "Window size W (" + std::to_string(windowSize) + 
                                  ") cannot exceed signal length N (" + std::to_string(n) + ").";
        return false;
    }
    return true;
}

bool movingAverageSequential(const std::vector<SampleType>& input,
                            std::vector<SampleType>& output,
                            int windowSize,
                            std::string* errorMsg) {
    const size_t n = input.size();

    // 1. Parameter Validation
    if (!validateWindowParameters(n, windowSize, errorMsg)) {
        return false;
    }

    // 2. Prepare Output Buffer
    if (output.size() != n) {
        output.resize(n);
    }

    // Window radius: W = 2k + 1  ==>  k = (W - 1) / 2
    const int k = (windowSize - 1) / 2;
    const double invW = 1.0 / static_cast<double>(windowSize);

    // 3. Naive O(N * W) Sequential Moving-Average Computation
    // For each output sample i, sum the W samples within [i - k, i + k],
    // clamping any out-of-bounds indices to the nearest valid boundary (0 or N - 1).
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0; // 64-bit accumulator prevents precision loss during summation

        for (int j = -k; j <= k; ++j) {
            int64_t rawIndex = static_cast<int64_t>(i) + j;
            
            // Edge Replication (Clamping): min(max(rawIndex, 0), N - 1)
            size_t clampedIndex;
            if (rawIndex < 0) {
                clampedIndex = 0;
            } else if (rawIndex >= static_cast<int64_t>(n)) {
                clampedIndex = n - 1;
            } else {
                clampedIndex = static_cast<size_t>(rawIndex);
            }

            sum += static_cast<double>(input[clampedIndex]);
        }

        output[i] = static_cast<SampleType>(sum * invW);
    }

    return true;
}
