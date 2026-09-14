#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

// Standard signal sample type (float is standard in GPU signal processing)
using SampleType = float;

// Benchmark result structure for consistent reporting across implementations
struct BenchmarkResult {
    std::string implementationName;
    size_t signalSize;
    int windowSize;
    int threadCount;          // For OpenMP (1 for sequential, 0 for CUDA if n/a)
    double executionTimeMs;   // Primary compute time (kernel time for CUDA)
    double totalTimeMs;       // End-to-end time (includes H2D + D2H for CUDA)
    double transferTimeMs;    // H2D + D2H for CUDA
    double speedup;           // Relative to sequential baseline
    double efficiency;        // Speedup / threadCount (for OpenMP)
};

// High-resolution CPU timer helper
class Timer {
public:
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = endTime - startTime;
        return duration.count(); // returns milliseconds
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

#endif // COMMON_HPP
