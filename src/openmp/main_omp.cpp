#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include "common.hpp"
#include "validator.hpp"
#include "signal_io.hpp"
#include "../sequential/moving_average_seq.hpp"
#include "moving_average_omp.hpp"

void printSignalSlice(const std::string& label, const std::vector<SampleType>& sig, size_t count = 5) {
    std::cout << label << " (total " << sig.size() << " samples): [";
    size_t previewCount = std::min(count, sig.size());
    for (size_t i = 0; i < previewCount; ++i) {
        std::cout << std::fixed << std::setprecision(4) << sig[i] << (i + 1 < previewCount ? ", " : "");
    }
    if (sig.size() > previewCount * 2) {
        std::cout << " ... ";
        for (size_t i = sig.size() - previewCount; i < sig.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << sig[i] << (i + 1 < sig.size() ? ", " : "");
        }
    }
    std::cout << "]\n";
}

// Runs mandatory correctness tests comparing OpenMP against Sequential reference
bool runCorrectnessTests() {
    std::cout << "\n=======================================================\n";
    std::cout << " RUNNING OPENMP CORRECTNESS VERIFICATION SUITE\n";
    std::cout << "=======================================================\n";
    bool allPassed = true;
    std::string err;

    // -------------------------------------------------------------
    // Test 1: Small hand-verifiable input (N = 5, W = 3)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 1] Hand-Verifiable Test (N = 5, W = 3):\n";
    std::vector<SampleType> testInput = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<SampleType> expectedOutput = {
        4.0f / 3.0f, 6.0f / 3.0f, 9.0f / 3.0f, 12.0f / 3.0f, 14.0f / 3.0f
    };
    std::vector<SampleType> seqOut, ompOut;
    movingAverageSequential(testInput, seqOut, 3, &err);
    movingAverageOpenMP(testInput, ompOut, 3, 2, &err);

    ValidationMetrics m1 = Validator::verify(expectedOutput, ompOut, 1e-5);
    ValidationMetrics m1_seq = Validator::verify(seqOut, ompOut, 1e-6);
    bool pass1 = m1.passed && m1_seq.passed;
    std::cout << "  - Comparison against hand calculation: " << (m1.passed ? "MATCH" : "FAIL") << "\n";
    std::cout << "  - Comparison against sequential reference: " << (m1_seq.passed ? "IDENTICAL" : "FAIL") << "\n";
    std::cout << "  - Max Absolute Error: " << m1_seq.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass1 ? "PASS" : "FAIL") << "\n";
    if (!pass1) allPassed = false;

    // -------------------------------------------------------------
    // Test 2: Identity Filter (W = 1)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 2] Identity Filter Test (W = 1):\n";
    std::vector<SampleType> sig10 = {10.5f, -2.3f, 4.0f, 9.1f, -1.0f, 0.0f, 7.7f, 3.2f, -8.4f, 5.0f};
    std::vector<SampleType> ompOutId;
    movingAverageOpenMP(sig10, ompOutId, 1, 4, &err);
    ValidationMetrics m2 = Validator::verify(sig10, ompOutId, 1e-6);
    bool pass2 = m2.passed && (m2.maxAbsoluteError == 0.0);
    std::cout << "  - Max Absolute Error: " << m2.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass2 ? "PASS" : "FAIL") << "\n";
    if (!pass2) allPassed = false;

    // -------------------------------------------------------------
    // Test 3: Boundary & Interior Check (N = 6, W = 5)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 3] Boundary & Interior Check (N = 6, W = 5):\n";
    std::vector<SampleType> sig6 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    std::vector<SampleType> expectedOut3 = {16.0f, 22.0f, 30.0f, 40.0f, 48.0f, 54.0f};
    std::vector<SampleType> ompOut3;
    movingAverageOpenMP(sig6, ompOut3, 5, 2, &err);
    ValidationMetrics m3 = Validator::verify(expectedOut3, ompOut3, 1e-5);
    bool pass3 = m3.passed;
    std::cout << "  - Max Absolute Error: " << m3.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass3 ? "PASS" : "FAIL") << "\n";
    if (!pass3) allPassed = false;

    // -------------------------------------------------------------
    // Test 4: Medium Scale Equivalence (N = 1000, W = 15) across thread counts
    // -------------------------------------------------------------
    std::cout << "\n[TEST 4] Medium Scale (N = 1000, W = 15) Cross-Thread Equivalence:\n";
    auto sig1000 = SignalGenerator::generateNoisySine(1000, 1000.0f, 5.0f, 0.5f, 42);
    std::vector<SampleType> seqOut1000;
    movingAverageSequential(sig1000, seqOut1000, 15, &err);

    std::vector<int> testThreads = {1, 2, 4, 8, 16, 24};
    bool pass4 = true;
    for (int t : testThreads) {
        std::vector<SampleType> candOut;
        movingAverageOpenMP(sig1000, candOut, 15, t, &err);
        ValidationMetrics m = Validator::verify(seqOut1000, candOut, 1e-5);
        if (!m.passed) {
            std::cout << "  - Thread " << t << ": FAILED (maxErr=" << m.maxAbsoluteError << ")\n";
            pass4 = false;
        } else {
            std::cout << "  - Threads " << std::setw(2) << t << ": MATCH (maxErr=" 
                      << std::scientific << std::setprecision(2) << m.maxAbsoluteError << ")\n";
        }
    }
    std::cout << "Result: " << (pass4 ? "PASS" : "FAIL") << "\n";
    if (!pass4) allPassed = false;

    // -------------------------------------------------------------
    // Test 5: Large Scale Equivalence (N = 100,000, W = 15)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 5] Large Scale (N = 100,000, W = 15) Validation:\n";
    auto sig100k = SignalGenerator::generateNoisySine(100000, 1000.0f, 5.0f, 0.5f, 999);
    std::vector<SampleType> seqOut100k;
    movingAverageSequential(sig100k, seqOut100k, 15, &err);

    bool pass5 = true;
    for (int t : {1, 4, 8, 24}) {
        std::vector<SampleType> candOut;
        movingAverageOpenMP(sig100k, candOut, 15, t, &err);
        ValidationMetrics m = Validator::verify(seqOut100k, candOut, 1e-5);
        if (!m.passed) {
            std::cout << "  - Threads " << std::setw(2) << t << ": FAILED (maxErr=" << m.maxAbsoluteError << ")\n";
            pass5 = false;
        } else {
            std::cout << "  - Threads " << std::setw(2) << t << ": MATCH (maxErr=" 
                      << std::scientific << std::setprecision(2) << m.maxAbsoluteError 
                      << ", RMSE=" << m.rootMeanSquareError << ")\n";
        }
    }
    std::cout << "Result: " << (pass5 ? "PASS" : "FAIL") << "\n";
    if (!pass5) allPassed = false;

    std::cout << "\n=======================================================\n";
    std::cout << " OVERALL TEST STATUS: " << (allPassed ? "ALL TESTS PASSED [SUCCESS]" : "SOME TESTS FAILED") << "\n";
    std::cout << "=======================================================\n\n";

    return allPassed;
}

// Executes thread-scaling experiment on N = 1,000,000, W = 31 across T in {1, 2, 4, 8, 16, 24}
void runThreadScalingBenchmark(size_t n = 1000000, int windowSize = 31, unsigned int seed = 42) {
    std::cout << "\n=========================================================================\n";
    std::cout << " OPENMP THREAD SCALING BENCHMARK EXPERIMENT\n";
    std::cout << " Workload: N = " << n << " samples, Window W = " << windowSize << " (radius k = " << (windowSize - 1) / 2 << ")\n";
    std::cout << " Logical Processors Available: " << omp_get_max_threads() << "\n";
    std::cout << " Runs per configuration: 5 (1 warm-up discarded + 4 averaged)\n";
    std::cout << "=========================================================================\n";

    std::cout << "Generating benchmark input signal..." << std::flush;
    std::vector<SampleType> inputSignal = SignalGenerator::generateNoisySine(n, 1000.0f, 5.0f, 0.5f, seed);
    std::cout << " Done.\n\n";

    std::string err;

    // 1. Measure Sequential Baseline on this EXACT SAME workload
    std::cout << "Measuring Sequential Baseline on identical workload (5 runs)...\n";
    std::vector<SampleType> seqReference;
    double seqTotalMs = 0.0;
    const int numRuns = 5;

    for (int run = 0; run < numRuns; ++run) {
        std::vector<SampleType> tmpSeq;
        Timer t;
        t.start();
        movingAverageSequential(inputSignal, tmpSeq, windowSize, &err);
        double ms = t.stop();

        if (run == 0) {
            seqReference = std::move(tmpSeq);
            std::cout << "  Run 0 (warm-up): " << std::fixed << std::setprecision(3) << ms << " ms (discarded)\n";
        } else {
            seqTotalMs += ms;
            std::cout << "  Run " << run << "          : " << std::fixed << std::setprecision(3) << ms << " ms\n";
        }
    }
    double seqAvgMs = seqTotalMs / (numRuns - 1);
    std::cout << ">>> Measured Sequential Baseline Average: " 
              << std::fixed << std::setprecision(3) << seqAvgMs << " ms <<<\n\n";

    // 2. Measure OpenMP Scaling across Thread Counts
    std::vector<int> threadCounts = {1, 2, 4, 8, 16, 24};
    struct ScalingRow {
        int threads;
        double avgTimeMs;
        double speedup;
        double efficiency;
        double maxError;
        bool valid;
    };
    std::vector<ScalingRow> results;

    std::cout << "Measuring OpenMP Scaling Across Thread Counts:\n";
    std::cout << "-------------------------------------------------------------------------\n";

    for (int t : threadCounts) {
        std::cout << "Evaluating Threads T = " << std::setw(2) << t << " ... " << std::flush;
        double totalOmpMs = 0.0;
        std::vector<SampleType> lastOmpOutput;

        for (int run = 0; run < numRuns; ++run) {
            std::vector<SampleType> tmpOmp;
            Timer timerOmp;
            timerOmp.start();
            movingAverageOpenMP(inputSignal, tmpOmp, windowSize, t, &err);
            double ms = timerOmp.stop();

            if (run == 0) {
                lastOmpOutput = std::move(tmpOmp); // warm-up
            } else {
                totalOmpMs += ms;
            }
        }

        double avgOmpMs = totalOmpMs / (numRuns - 1);
        double speedup = seqAvgMs / avgOmpMs;
        double efficiency = speedup / static_cast<double>(t);

        // Correctness verification against sequential reference
        ValidationMetrics vm = Validator::verify(seqReference, lastOmpOutput, 1e-4);

        ScalingRow row = {t, avgOmpMs, speedup, efficiency, vm.maxAbsoluteError, vm.passed};
        results.push_back(row);

        std::cout << "Avg: " << std::setw(8) << std::fixed << std::setprecision(3) << avgOmpMs << " ms | "
                  << "Speedup: " << std::setw(5) << std::fixed << std::setprecision(2) << speedup << "x | "
                  << "Eff: " << std::setw(5) << std::fixed << std::setprecision(1) << (efficiency * 100.0) << "% | "
                  << "Correctness: " << (vm.passed ? "PASS" : "FAIL") << "\n";
    }

    // 3. Print Markdown Result Table
    std::cout << "\n=========================================================================\n";
    std::cout << " SUMMARY TABLE: OPENMP THREAD SCALING (N = " << n << ", W = " << windowSize << ")\n";
    std::cout << "=========================================================================\n";
    std::cout << "| Threads | Avg Time (ms) | Speedup | Efficiency (%) | Correctness |\n";
    std::cout << "|:-------:|:-------------:|:-------:|:--------------:|:-----------:|\n";
    for (const auto& r : results) {
        std::cout << "| " << std::setw(7) << r.threads 
                  << " | " << std::setw(13) << std::fixed << std::setprecision(3) << r.avgTimeMs
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << " | " << std::setw(14) << std::fixed << std::setprecision(2) << (r.efficiency * 100.0)
                  << " | " << std::setw(11) << (r.valid ? "PASSED" : "FAILED") << " |\n";
    }
    std::cout << "=========================================================================\n\n";

    // 4. Save Table to CSV
    std::string csvPath = "results/tables/omp_thread_scaling.csv";
    std::ofstream csv(csvPath);
    if (csv) {
        csv << "Threads,AvgTimeMs,Speedup,EfficiencyPercent,MaxAbsoluteError,Status\n";
        for (const auto& r : results) {
            csv << r.threads << "," << r.avgTimeMs << "," << r.speedup << "," 
                << (r.efficiency * 100.0) << "," << r.maxError << "," << (r.valid ? "PASSED" : "FAILED") << "\n";
        }
        std::cout << "Raw scaling data saved to: " << csvPath << "\n\n";
    }
}

int main(int argc, char* argv[]) {
    size_t n = 100000;
    int windowSize = 15;
    unsigned int seed = 42;
    int numThreads = 0; // 0 means default to max threads
    std::string savePath = "";
    bool runTests = false;
    bool runBenchmark = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--size") && i + 1 < argc) {
            n = static_cast<size_t>(std::atoll(argv[++i]));
        } else if ((arg == "-w" || arg == "--window") && i + 1 < argc) {
            windowSize = std::atoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            seed = static_cast<unsigned int>(std::atoi(argv[++i]));
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            numThreads = std::atoi(argv[++i]);
        } else if ((arg == "--save") && i + 1 < argc) {
            savePath = argv[++i];
        } else if (arg == "--test") {
            runTests = true;
        } else if (arg == "--benchmark") {
            runBenchmark = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -n, --size <N>       Signal size (default: 100000)\n"
                      << "  -w, --window <W>     Moving-average window size (must be odd, default: 15)\n"
                      << "  -s, --seed <S>       Random seed for signal generator (default: 42)\n"
                      << "  -t, --threads <T>    Number of OpenMP threads (default: max available)\n"
                      << "  --test               Run built-in correctness test suite\n"
                      << "  --benchmark          Run full thread-scaling benchmark (N=1M, W=31, T=1..24)\n"
                      << "  --save <path>        Path to save output binary signal\n"
                      << "  -h, --help           Display this help message\n";
            return 0;
        }
    }

    if (runTests) {
        bool passed = runCorrectnessTests();
        return passed ? 0 : 1;
    }

    if (runBenchmark) {
        runThreadScalingBenchmark(n == 100000 ? 1000000 : n, 
                                  windowSize == 15 ? 31 : windowSize, 
                                  seed);
        return 0;
    }

    int actualThreads = (numThreads > 0) ? numThreads : omp_get_max_threads();

    std::cout << "=========================================================\n";
    std::cout << " CSS311 Assignment-1: OpenMP Moving-Average Parallel\n";
    std::cout << "=========================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Signal Size (N)  : " << n << " samples\n";
    std::cout << "  Window Size (W)  : " << windowSize << " (half-window radius k = " << (windowSize - 1) / 2 << ")\n";
    std::cout << "  Random Seed      : " << seed << "\n";
    std::cout << "  Threads Selected : " << actualThreads << " (of " << omp_get_max_threads() << " logical cores)\n";
    std::cout << "  Scheduling       : schedule(static)\n";
    std::cout << "---------------------------------------------------------\n";

    // 1. Generate Input Signal
    std::cout << "Generating input signal..." << std::flush;
    std::vector<SampleType> inputSignal = SignalGenerator::generateNoisySine(n, 1000.0f, 5.0f, 0.5f, seed);
    std::cout << " Done.\n";

    // 2. Sequential Reference Run for Validation
    std::cout << "Running sequential reference baseline for validation..." << std::flush;
    std::vector<SampleType> seqOutput;
    std::string err;
    Timer seqTimer;
    seqTimer.start();
    movingAverageSequential(inputSignal, seqOutput, windowSize, &err);
    double seqMs = seqTimer.stop();
    std::cout << " Done (" << std::fixed << std::setprecision(3) << seqMs << " ms).\n";

    // 3. OpenMP Parallel Run (TIMED ISOLATED COMPUTATION)
    std::vector<SampleType> ompOutput;
    Timer ompTimer;
    ompTimer.start();
    bool success = movingAverageOpenMP(inputSignal, ompOutput, windowSize, actualThreads, &err);
    double ompMs = ompTimer.stop();

    if (!success) {
        std::cerr << "\n[Error] OpenMP filtering failed: " << err << "\n";
        return 1;
    }

    // 4. Report Performance and Metrics
    std::cout << "\n>>> OpenMP filter time: " << std::fixed << std::setprecision(3) 
              << ompMs << " ms <<<\n";
    std::cout << "Number of OpenMP threads: " << actualThreads << "\n";
    double speedup = seqMs / ompMs;
    double efficiency = (speedup / actualThreads) * 100.0;
    std::cout << "Measured Speedup       : " << std::fixed << std::setprecision(2) << speedup << "x\n";
    std::cout << "Parallel Efficiency    : " << std::fixed << std::setprecision(1) << efficiency << "%\n";

    // 5. Correctness Verification against Sequential Reference
    ValidationMetrics vm = Validator::verify(seqOutput, ompOutput, 1e-4);
    Validator::printReport("OpenMP vs Sequential Reference", vm);

    // 6. Print Slices
    printSignalSlice("Input Signal ", inputSignal, 5);
    printSignalSlice("OpenMP Output", ompOutput, 5);

    // 7. Save Output if Requested
    if (!savePath.empty()) {
        if (SignalGenerator::saveBinary(savePath, ompOutput)) {
            std::cout << "\nOutput saved successfully to: " << savePath << "\n";
        }
    }

    std::cout << "=========================================================\n";
    return vm.passed ? 0 : 1;
}
