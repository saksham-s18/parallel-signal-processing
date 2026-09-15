#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include "common.hpp"
#include "validator.hpp"
#include "signal_io.hpp"
#include "../sequential/moving_average_seq.hpp"
#include "moving_average_opt.cuh"

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

bool runCorrectnessTests() {
    std::cout << "\n===========================================================\n";
    std::cout << " RUNNING OPTIMIZED CUDA (SHARED MEMORY) CORRECTNESS SUITE\n";
    std::cout << "===========================================================\n";
    bool allPassed = true;
    std::string err;

    // Test 1: Hand-verifiable test (N = 5, W = 3)
    std::cout << "\n[TEST 1] Hand-Verifiable Test (N = 5, W = 3):\n";
    std::vector<SampleType> testInput = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<SampleType> expectedOutput = {
        4.0f / 3.0f, 6.0f / 3.0f, 9.0f / 3.0f, 12.0f / 3.0f, 14.0f / 3.0f
    };
    std::vector<SampleType> seqOut, optOut;
    movingAverageSequential(testInput, seqOut, 3, &err);
    bool ok1 = movingAverageCUDAOptimized(testInput, optOut, 3, 256, nullptr, &err);

    ValidationMetrics m1 = Validator::verify(expectedOutput, optOut, 1e-5);
    ValidationMetrics m1_seq = Validator::verify(seqOut, optOut, 1e-6);
    bool pass1 = ok1 && m1.passed && m1_seq.passed;
    std::cout << "  - Comparison against hand calculation: " << (m1.passed ? "MATCH" : "FAIL") << "\n";
    std::cout << "  - Comparison against sequential reference: " << (m1_seq.passed ? "IDENTICAL" : "FAIL") << "\n";
    std::cout << "  - Max Absolute Error: " << m1_seq.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass1 ? "PASS" : "FAIL") << "\n";
    if (!pass1) allPassed = false;

    // Test 2: Identity Filter (W = 1)
    std::cout << "\n[TEST 2] Identity Filter Test (N = 10, W = 1):\n";
    std::vector<SampleType> sig10 = {10.5f, -2.3f, 4.0f, 9.1f, -1.0f, 0.0f, 7.7f, 3.2f, -8.4f, 5.0f};
    std::vector<SampleType> optOutId;
    bool ok2 = movingAverageCUDAOptimized(sig10, optOutId, 1, 256, nullptr, &err);
    ValidationMetrics m2 = Validator::verify(sig10, optOutId, 1e-6);
    bool pass2 = ok2 && m2.passed && (m2.maxAbsoluteError == 0.0);
    std::cout << "  - Max Absolute Error: " << m2.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass2 ? "PASS" : "FAIL") << "\n";
    if (!pass2) allPassed = false;

    // Test 3: Boundary & Interior Check (N = 6, W = 5)
    std::cout << "\n[TEST 3] Boundary & Interior Check (N = 6, W = 5):\n";
    std::vector<SampleType> sig6 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    std::vector<SampleType> expectedOut3 = {16.0f, 22.0f, 30.0f, 40.0f, 48.0f, 54.0f};
    std::vector<SampleType> optOut3;
    bool ok3 = movingAverageCUDAOptimized(sig6, optOut3, 5, 256, nullptr, &err);
    ValidationMetrics m3 = Validator::verify(expectedOut3, optOut3, 1e-5);
    bool pass3 = ok3 && m3.passed;
    std::cout << "  - Max Absolute Error: " << m3.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass3 ? "PASS" : "FAIL") << "\n";
    if (!pass3) allPassed = false;

    // Test 4: Medium Scale Equivalence (N = 1000, W = 15)
    std::cout << "\n[TEST 4] Medium Scale Equivalence (N = 1000, W = 15):\n";
    auto sig1000 = SignalGenerator::generateNoisySine(1000, 1000.0f, 5.0f, 0.5f, 42);
    std::vector<SampleType> seqOut1000, optOut1000;
    movingAverageSequential(sig1000, seqOut1000, 15, &err);
    bool ok4 = movingAverageCUDAOptimized(sig1000, optOut1000, 15, 256, nullptr, &err);
    ValidationMetrics m4 = Validator::verify(seqOut1000, optOut1000, 1e-5);
    bool pass4 = ok4 && m4.passed;
    std::cout << "  - Max Absolute Error: " << std::scientific << std::setprecision(2) << m4.maxAbsoluteError << "\n";
    std::cout << "  - Root Mean Sq Error: " << std::scientific << std::setprecision(2) << m4.rootMeanSquareError << "\n";
    std::cout << "Result: " << (pass4 ? "PASS" : "FAIL") << "\n";
    if (!pass4) allPassed = false;

    // Test 5: Invalid Parameter Handling
    std::cout << "\n[TEST 5] Invalid Parameter Handling:\n";
    std::vector<SampleType> dummy = {1.0f, 2.0f, 3.0f};
    std::vector<SampleType> dummyOut;
    std::string errMsg;

    bool rejEven = !movingAverageCUDAOptimized(dummy, dummyOut, 4, 256, nullptr, &errMsg);
    std::cout << "  - Reject even window (W=4):      " << (rejEven ? "PASS" : "FAIL") << "\n";

    bool rejZero = !movingAverageCUDAOptimized(dummy, dummyOut, 0, 256, nullptr, &errMsg);
    std::cout << "  - Reject zero window (W=0):      " << (rejZero ? "PASS" : "FAIL") << "\n";

    bool rejNeg = !movingAverageCUDAOptimized(dummy, dummyOut, -3, 256, nullptr, &errMsg);
    std::cout << "  - Reject negative window (W=-3): " << (rejNeg ? "PASS" : "FAIL") << "\n";

    bool rejOversize = !movingAverageCUDAOptimized(dummy, dummyOut, 7, 256, nullptr, &errMsg);
    std::cout << "  - Reject oversized window (W>N): " << (rejOversize ? "PASS" : "FAIL") << "\n";

    bool pass5 = rejEven && rejZero && rejNeg && rejOversize;
    std::cout << "Result: " << (pass5 ? "PASS" : "FAIL") << "\n";
    if (!pass5) allPassed = false;

    std::cout << "\n===========================================================\n";
    std::cout << " OVERALL OPTIMIZED CUDA STATUS: " << (allPassed ? "ALL TESTS PASSED [SUCCESS]" : "SOME TESTS FAILED") << "\n";
    std::cout << "===========================================================\n\n";

    return allPassed;
}

int main(int argc, char* argv[]) {
    size_t n = 100000;
    int windowSize = 15;
    int blockSize = 256;
    unsigned int seed = 42;
    int repetitions = 5;
    std::string savePath = "";
    std::string loadInputPath = "";
    bool runTests = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--size") && i + 1 < argc) {
            n = static_cast<size_t>(std::atoll(argv[++i]));
        } else if ((arg == "-w" || arg == "--window") && i + 1 < argc) {
            windowSize = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--block-size") && i + 1 < argc) {
            blockSize = std::atoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            seed = static_cast<unsigned int>(std::atoi(argv[++i]));
        } else if ((arg == "-r" || arg == "--runs") && i + 1 < argc) {
            repetitions = std::atoi(argv[++i]);
        } else if ((arg == "--load") && i + 1 < argc) {
            loadInputPath = argv[++i];
        } else if ((arg == "--save") && i + 1 < argc) {
            savePath = argv[++i];
        } else if (arg == "--test") {
            runTests = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -n, --size <N>          Signal size (default: 100000)\n"
                      << "  -w, --window <W>        Moving-average window size (must be odd, default: 15)\n"
                      << "  -b, --block-size <B>    CUDA block size (default: 256)\n"
                      << "  -s, --seed <S>          Random seed for signal generator (default: 42)\n"
                      << "  -r, --runs <R>          Number of benchmark repetitions (default: 5)\n"
                      << "  --load <path>           Load binary input signal\n"
                      << "  --save <path>           Path to save output binary signal\n"
                      << "  --test                  Run built-in correctness verification suite\n"
                      << "  -h, --help              Display this help message\n";
            return 0;
        }
    }

    if (runTests) {
        bool passed = runCorrectnessTests();
        return passed ? 0 : 1;
    }

    std::cout << "=========================================================\n";
    std::cout << " CSS311: CUDA Moving-Average (Shared-Memory Optimized)\n";
    std::cout << "=========================================================\n";

    // 1. Prepare Input Signal
    std::vector<SampleType> inputSignal;
    if (!loadInputPath.empty()) {
        std::cout << "Loading input signal from: " << loadInputPath << " ... " << std::flush;
        if (!SignalGenerator::loadBinary(loadInputPath, inputSignal)) {
            std::cerr << "Failed to load input.\n";
            return 1;
        }
        n = inputSignal.size();
        std::cout << "Done (" << n << " samples).\n";
    } else {
        std::cout << "Generating input signal (" << n << " samples, seed " << seed << ") ... " << std::flush;
        inputSignal = SignalGenerator::generateNoisySine(n, 1000.0f, 5.0f, 0.5f, seed);
        std::cout << "Done.\n";
    }

    std::cout << "Configuration:\n";
    std::cout << "  Signal Size (N)   : " << n << " samples\n";
    std::cout << "  Window Size (W)   : " << windowSize << " (radius k = " << (windowSize - 1) / 2 << ")\n";
    std::cout << "  CUDA Block Size   : " << blockSize << " threads/block\n";
    std::cout << "  Grid Size         : " << (n + blockSize - 1) / blockSize << " blocks\n";
    std::cout << "  Shared Mem/Block  : " << (blockSize + 2 * ((windowSize - 1) / 2)) * sizeof(SampleType) << " bytes\n";
    std::cout << "  Repetitions       : " << repetitions << " runs (1 warm-up discarded)\n";
    std::cout << "---------------------------------------------------------\n";

    // 2. Sequential Reference Run for Validation
    std::cout << "Computing sequential reference baseline for validation..." << std::flush;
    std::vector<SampleType> seqOutput;
    std::string err;
    Timer seqTimer;
    seqTimer.start();
    movingAverageSequential(inputSignal, seqOutput, windowSize, &err);
    double seqMs = seqTimer.stop();
    std::cout << " Done (" << std::fixed << std::setprecision(3) << seqMs << " ms).\n";

    // 3. Warm-up Optimized CUDA Run
    std::vector<SampleType> optOutput;
    CUDATimings warmupTimings;
    bool ok = movingAverageCUDAOptimized(inputSignal, optOutput, windowSize, blockSize, &warmupTimings, &err);
    if (!ok) {
        std::cerr << "[Error] Optimized CUDA execution failed: " << err << "\n";
        return 1;
    }

    // 4. Repeated Timed Benchmark Runs
    float totalH2d = 0.0f, totalKernel = 0.0f, totalD2h = 0.0f, totalE2e = 0.0f;
    for (int r = 0; r < repetitions; ++r) {
        CUDATimings t;
        movingAverageCUDAOptimized(inputSignal, optOutput, windowSize, blockSize, &t, &err);
        totalH2d += t.h2dMs;
        totalKernel += t.kernelMs;
        totalD2h += t.d2hMs;
        totalE2e += t.totalMs;
    }

    float avgH2d = totalH2d / repetitions;
    float avgKernel = totalKernel / repetitions;
    float avgD2h = totalD2h / repetitions;
    float avgE2e = totalE2e / repetitions;

    // 5. Verification against Sequential Reference
    ValidationMetrics vm = Validator::verify(seqOutput, optOutput, 1e-4);
    Validator::printReport("Optimized CUDA vs Sequential Reference", vm);

    // 6. Timing Report
    std::cout << "\n>>> OPTIMIZED CUDA TIMING BREAKDOWN (Averaged over " << repetitions << " runs) <<<\n";
    std::cout << "  Host-to-Device (H2D) : " << std::fixed << std::setprecision(4) << avgH2d << " ms\n";
    std::cout << "  Kernel Execution     : " << std::fixed << std::setprecision(4) << avgKernel << " ms\n";
    std::cout << "  Device-to-Host (D2H) : " << std::fixed << std::setprecision(4) << avgD2h << " ms\n";
    std::cout << "  Total End-to-End     : " << std::fixed << std::setprecision(4) << avgE2e << " ms\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "  Sequential Time      : " << std::fixed << std::setprecision(4) << seqMs << " ms\n";
    std::cout << "  Kernel-only Speedup  : " << std::fixed << std::setprecision(2) << (seqMs / avgKernel) << "x\n";
    std::cout << "  End-to-End Speedup   : " << std::fixed << std::setprecision(2) << (seqMs / avgE2e) << "x\n\n";

    // 7. Print Slices
    printSignalSlice("Input Signal ", inputSignal, 5);
    printSignalSlice("Opt Output   ", optOutput, 5);

    // 8. Save Output if Requested
    if (!savePath.empty()) {
        if (SignalGenerator::saveBinary(savePath, optOutput)) {
            std::cout << "\nOutput saved successfully to: " << savePath << "\n";
        }
    }

    std::cout << "=========================================================\n";
    return vm.passed ? 0 : 1;
}
