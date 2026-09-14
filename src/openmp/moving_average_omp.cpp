#include "moving_average_omp.hpp"
#include "../sequential/moving_average_seq.hpp"
#include <omp.h>

bool movingAverageOpenMP(const std::vector<SampleType>& input,
                         std::vector<SampleType>& output,
                         int windowSize,
                         int numThreads,
                         std::string* errorMsg) {
    const size_t n = input.size();

    // 1. Parameter Validation (reusing verified rules from sequential baseline)
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

    // Determine thread count: default to max available threads if unspecified (<= 0)
    int threadsToUse = (numThreads > 0) ? numThreads : omp_get_max_threads();

    // 3. OpenMP Parallel Moving-Average Computation
    //
    // Parallelization Design:
    // - Output domain decomposition: The iteration space [0, N - 1] is partitioned across threads.
    // - Schedule policy: 'schedule(static)' is used because the computational workload per output
    //   element is uniform (each sample requires exactly W reads, additions, and 1 scaling).
    //   Static chunking assigns N / threadsToUse contiguous iterations to each thread with zero runtime
    //   scheduling overhead and optimal spatial cache locality for consecutive reads.
    // - Thread safety:
    //   * 'input' is strictly read-only (shared among threads with no write conflicts).
    //   * 'output[i]' is written exclusively by the thread processing index i (no write-write hazard).
    //   * 'sum' is declared inside the iteration body, ensuring thread-private scope.
    // - Synchronization:
    //   * No mutexes, atomics, or explicit barriers are required within the loop.
    //   * An implicit barrier at the end of '#pragma omp parallel for' guarantees that all output
    //     elements are fully computed before this function returns.
    #pragma omp parallel for num_threads(threadsToUse) schedule(static) default(none) \
        shared(input, output, n, windowSize, k, invW)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        double sum = 0.0; // Thread-private accumulator matching sequential precision

        for (int j = -k; j <= k; ++j) {
            int64_t rawIndex = i + j;
            
            // Edge Replication (Clamping): min(max(rawIndex, 0), N - 1)
            int64_t clampedIndex;
            if (rawIndex < 0) {
                clampedIndex = 0;
            } else if (rawIndex >= static_cast<int64_t>(n)) {
                clampedIndex = static_cast<int64_t>(n) - 1;
            } else {
                clampedIndex = rawIndex;
            }

            sum += static_cast<double>(input[static_cast<size_t>(clampedIndex)]);
        }

        output[static_cast<size_t>(i)] = static_cast<SampleType>(sum * invW);
    }

    return true;
}
