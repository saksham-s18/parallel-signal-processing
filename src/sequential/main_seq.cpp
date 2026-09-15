#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include "common.hpp"
#include "validator.hpp"
#include "signal_io.hpp"
#include "moving_average_seq.hpp"

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

// Executes comprehensive correctness verification test suite
bool runCorrectnessTests() {
    std::cout << "\n=======================================================\n";
    std::cout << " RUNNING SEQUENTIAL BASELINE CORRECTNESS TEST SUITE\n";
    std::cout << "=======================================================\n";
    bool allPassed = true;

    // -------------------------------------------------------------
    // Test 1: Small deterministic hand-verifiable test
    // Input: [1.0, 2.0, 3.0, 4.0, 5.0], Window W = 3 (k = 1)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 1] Hand-Verifiable Test (N = 5, W = 3):\n";
    std::vector<SampleType> testInput = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int w1 = 3;

    // Expected derivations using edge replication:
    // i=0: clamp([-1, 0, 1]) -> [0, 0, 1] -> (1 + 1 + 2) / 3 = 4/3  = 1.333333
    // i=1: clamp([ 0, 1, 2]) -> [0, 1, 2] -> (1 + 2 + 3) / 3 = 6/3  = 2.000000
    // i=2: clamp([ 1, 2, 3]) -> [1, 2, 3] -> (2 + 3 + 4) / 3 = 9/3  = 3.000000
    // i=3: clamp([ 2, 3, 4]) -> [2, 3, 4] -> (3 + 4 + 5) / 3 = 12/3 = 4.000000
    // i=4: clamp([ 3, 4, 5]) -> [3, 4, 4] -> (4 + 5 + 5) / 3 = 14/3 = 4.666667
    std::vector<SampleType> expectedOutput1 = {
        4.0f / 3.0f,
        6.0f / 3.0f,
        9.0f / 3.0f,
        12.0f / 3.0f,
        14.0f / 3.0f
    };

    std::vector<SampleType> actualOutput1;
    std::string err;
    bool ok1 = movingAverageSequential(testInput, actualOutput1, w1, &err);

    std::cout << "Input:    [";
    for (size_t i = 0; i < testInput.size(); ++i) {
        std::cout << testInput[i] << (i + 1 < testInput.size() ? ", " : "");
    }
    std::cout << "]\nWindow:   " << w1 << "\n";

    std::cout << "Expected: [";
    for (size_t i = 0; i < expectedOutput1.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << expectedOutput1[i] << (i + 1 < expectedOutput1.size() ? ", " : "");
    }
    std::cout << "]\nActual:   [";
    for (size_t i = 0; i < actualOutput1.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << actualOutput1[i] << (i + 1 < actualOutput1.size() ? ", " : "");
    }
    std::cout << "]\n";

    ValidationMetrics m1 = Validator::verify(expectedOutput1, actualOutput1, 1e-5);
    bool pass1 = ok1 && m1.passed;
    std::cout << "Result:   " << (pass1 ? "PASS" : "FAIL") << "\n";
    if (!pass1) allPassed = false;

    // -------------------------------------------------------------
    // Test 2: Identity Filter (W = 1)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 2] Identity Filter Test (W = 1):\n";
    std::vector<SampleType> sig10 = {10.5f, -2.3f, 4.0f, 9.1f, -1.0f, 0.0f, 7.7f, 3.2f, -8.4f, 5.0f};
    std::vector<SampleType> outIdentity;
    bool ok2 = movingAverageSequential(sig10, outIdentity, 1, &err);
    ValidationMetrics m2 = Validator::verify(sig10, outIdentity, 1e-6);
    bool pass2 = ok2 && m2.passed && (m2.maxAbsoluteError == 0.0);
    std::cout << "Window: 1 (Every output must strictly match input)\n";
    std::cout << "Max Absolute Error: " << m2.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass2 ? "PASS" : "FAIL") << "\n";
    if (!pass2) allPassed = false;

    // -------------------------------------------------------------
    // Test 3: Boundary Elements Check (N = 6, W = 5, k = 2)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 3] Boundary & Interior Check (N = 6, W = 5):\n";
    std::vector<SampleType> sig6 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    // Expected:
    // i=0: clamp([-2,-1,0,1,2]) -> (10+10+10+20+30)/5 = 80/5 = 16.0
    // i=1: clamp([-1,0,1,2,3])  -> (10+10+20+30+40)/5 = 110/5 = 22.0
    // i=2: clamp([0,1,2,3,4])   -> (10+20+30+40+50)/5 = 150/5 = 30.0 (interior)
    // i=3: clamp([1,2,3,4,5])   -> (20+30+40+50+60)/5 = 200/5 = 40.0 (interior)
    // i=4: clamp([2,3,4,5,6])   -> (30+40+50+60+60)/5 = 240/5 = 48.0
    // i=5: clamp([3,4,5,6,7])   -> (40+50+60+60+60)/5 = 270/5 = 54.0
    std::vector<SampleType> expectedOutput3 = {16.0f, 22.0f, 30.0f, 40.0f, 48.0f, 54.0f};
    std::vector<SampleType> out3;
    bool ok3 = movingAverageSequential(sig6, out3, 5, &err);
    ValidationMetrics m3 = Validator::verify(expectedOutput3, out3, 1e-5);
    bool pass3 = ok3 && m3.passed;
    std::cout << "Max Absolute Error: " << m3.maxAbsoluteError << "\n";
    std::cout << "Result: " << (pass3 ? "PASS" : "FAIL") << "\n";
    if (!pass3) allPassed = false;

    // -------------------------------------------------------------
    // Test 4: Parameter Validation & Invalid Configurations
    // -------------------------------------------------------------
    std::cout << "\n[TEST 4] Invalid Parameter Handling:\n";
    std::vector<SampleType> dummy = {1.0f, 2.0f, 3.0f};
    std::vector<SampleType> dummyOut;
    std::string errMsg;

    bool rejEven = !movingAverageSequential(dummy, dummyOut, 4, &errMsg);
    std::cout << "  - Reject even window (W=4):   " << (rejEven ? "PASS" : "FAIL") << " -> " << errMsg << "\n";

    bool rejZero = !movingAverageSequential(dummy, dummyOut, 0, &errMsg);
    std::cout << "  - Reject zero window (W=0):   " << (rejZero ? "PASS" : "FAIL") << " -> " << errMsg << "\n";

    bool rejNeg = !movingAverageSequential(dummy, dummyOut, -3, &errMsg);
    std::cout << "  - Reject negative window (W=-3): " << (rejNeg ? "PASS" : "FAIL") << " -> " << errMsg << "\n";

    bool rejOversize = !movingAverageSequential(dummy, dummyOut, 7, &errMsg);
    std::cout << "  - Reject oversized window (W > N): " << (rejOversize ? "PASS" : "FAIL") << " -> " << errMsg << "\n";

    bool pass4 = rejEven && rejZero && rejNeg && rejOversize;
    std::cout << "Result: " << (pass4 ? "PASS" : "FAIL") << "\n";
    if (!pass4) allPassed = false;

    // -------------------------------------------------------------
    // Test 5: Multi-Scale Execution Stability (N = 10, 1000, 100000)
    // -------------------------------------------------------------
    std::cout << "\n[TEST 5] Scale Stability Tests:\n";
    std::vector<size_t> testSizes = {10, 1000, 100000};
    std::vector<int> testWindows = {1, 3, 15};
    bool pass5 = true;

    for (size_t n : testSizes) {
        auto sig = SignalGenerator::generateNoisySine(n, 1000.0f, 5.0f, 0.5f, 12345);
        for (int w : testWindows) {
            if (static_cast<size_t>(w) > n) continue;
            std::vector<SampleType> out;
            Timer t;
            t.start();
            bool ok = movingAverageSequential(sig, out, w, &errMsg);
            double ms = t.stop();

            if (!ok || out.size() != n) {
                std::cout << "  - Failed on N=" << n << ", W=" << w << "\n";
                pass5 = false;
            } else {
                std::cout << "  - N=" << std::setw(6) << n 
                          << ", W=" << std::setw(2) << w 
                          << " -> Success (" << std::fixed << std::setprecision(3) << ms << " ms)\n";
            }
        }
    }
    std::cout << "Result: " << (pass5 ? "PASS" : "FAIL") << "\n";
    if (!pass5) allPassed = false;

    std::cout << "\n=======================================================\n";
    std::cout << " TEST SUITE SUMMARY: " << (allPassed ? "ALL TESTS PASSED [SUCCESS]" : "SOME TESTS FAILED") << "\n";
    std::cout << "=======================================================\n\n";

    return allPassed;
}

int main(int argc, char* argv[]) {
    // Default arguments
    size_t n = 100000;
    int windowSize = 15;
    unsigned int seed = 42;
    int repetitions = 5;
    std::string savePath = "";
    std::string loadInputPath = "";
    bool runTests = false;

    // Parse CLI options
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-n" || arg == "--size") && i + 1 < argc) {
            n = static_cast<size_t>(std::atoll(argv[++i]));
        } else if ((arg == "-w" || arg == "--window") && i + 1 < argc) {
            windowSize = std::atoi(argv[++i]);
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
                      << "  -n, --size <N>       Signal size (default: 100000)\n"
                      << "  -w, --window <W>     Moving-average window size (must be odd, default: 15)\n"
                      << "  -s, --seed <S>       Random seed for signal generator (default: 42)\n"
                      << "  -r, --runs <R>       Number of benchmark repetitions (default: 5)\n"
                      << "  --load <path>        Path to load binary input signal\n"
                      << "  --save <path>        Path to save output binary signal\n"
                      << "  --test               Run built-in verification test suite\n"
                      << "  -h, --help           Display this help message\n";
            return 0;
        }
    }

    if (runTests) {
        bool passed = runCorrectnessTests();
        return passed ? 0 : 1;
    }

    std::cout << "=========================================================\n";
    std::cout << " CSS311 Assignment-1: Sequential Moving-Average Baseline\n";
    std::cout << "=========================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Signal Size (N)  : " << n << " samples\n";
    std::cout << "  Window Size (W)  : " << windowSize << " (half-window radius k = " << (windowSize - 1) / 2 << ")\n";
    std::cout << "  Random Seed      : " << seed << "\n";
    std::cout << "  Boundary Mode    : Edge Replication (Clamping)\n";
    std::cout << "  Algorithm        : Naive O(N*W) Sequential Baseline\n";
    std::cout << "  Repetitions      : " << repetitions << " runs (1 warm-up discarded)\n";
    std::cout << "---------------------------------------------------------\n";

    // 1. Prepare Input Signal
    std::vector<SampleType> inputSignal;
    if (!loadInputPath.empty()) {
        std::cout << "Loading input signal from: " << loadInputPath << " ... " << std::flush;
        if (!SignalGenerator::loadBinary(loadInputPath, inputSignal)) {
            std::cerr << "Failed to load input.\n";
            return 1;
        }
        n = inputSignal.size();
        std::cout << " Done (" << n << " samples).\n";
    } else {
        std::cout << "Generating input signal..." << std::flush;
        inputSignal = SignalGenerator::generateNoisySine(n, 1000.0f, 5.0f, 0.5f, seed);
        std::cout << " Done.\n";
    }

    // 2. Perform Moving-Average Filter Computation (TIMED ISOLATED COMPUTATION)
    std::vector<SampleType> outputSignal;
    std::string errorMsg;
    double totalMs = 0.0;

    for (int run = 0; run < repetitions; ++run) {
        std::vector<SampleType> tmpOutput;
        Timer filterTimer;
        filterTimer.start();
        bool success = movingAverageSequential(inputSignal, tmpOutput, windowSize, &errorMsg);
        double ms = filterTimer.stop();

        if (!success) {
            std::cerr << "\n[Error] Moving-average filtering failed: " << errorMsg << "\n";
            return 1;
        }

        if (run == 0) {
            outputSignal = std::move(tmpOutput); // warm-up
        } else {
            totalMs += ms;
        }
    }

    double elapsedMs = (repetitions > 1) ? (totalMs / (repetitions - 1)) : totalMs;

    // 3. Print Performance and Results
    std::cout << "\n>>> Sequential filter time: " << std::fixed << std::setprecision(3) 
              << elapsedMs << " ms <<<\n";
    double throughputMsamples = (static_cast<double>(n) / (elapsedMs / 1000.0)) / 1e6;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << throughputMsamples << " MSamples/sec\n\n";

    // 4. Print Representative Slices
    printSignalSlice("Input Signal ", inputSignal, 5);
    printSignalSlice("Smoothed Output", outputSignal, 5);

    // 5. Save Output if Requested
    if (!savePath.empty()) {
        if (SignalGenerator::saveBinary(savePath, outputSignal)) {
            std::cout << "\nOutput saved successfully to: " << savePath << "\n";
        }
    }

    std::cout << "=========================================================\n";
    return 0;
}
