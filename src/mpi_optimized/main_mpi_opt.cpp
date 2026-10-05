#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <mpi.h>
#include "common.hpp"
#include "validator.hpp"
#include "signal_io.hpp"
#include "../sequential/moving_average_seq.hpp"
#include "moving_average_mpi_opt.hpp"

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

bool runCorrectnessTests(int rank, int worldSize) {
    if (rank == 0) {
        std::cout << "\n=======================================================\n";
        std::cout << " RUNNING OPTIMIZED MPI (NON-BLOCKING) TEST SUITE\n";
        std::cout << " MPI Communicator Size (P): " << worldSize << " processes\n";
        std::cout << " Overlap Strategy: Isend/Irecv with Interior Compute Overlap\n";
        std::cout << "=======================================================\n";
    }

    bool allPassed = true;
    std::string err;

    // -------------------------------------------------------------
    // Test 1: Deterministic Hand-Verifiable Test (N = 10, W = 3)
    // -------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n[TEST 1] Hand-Verifiable Test (N = 10, W = 3):\n";
    }
    std::vector<SampleType> testInput1;
    if (rank == 0) {
        testInput1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    }
    std::vector<SampleType> seqOut1, mpiOut1;
    if (rank == 0) {
        movingAverageSequential(testInput1, seqOut1, 3, &err);
    }
    movingAverageMPIOpt(testInput1, mpiOut1, 3, MPI_COMM_WORLD, nullptr, &err);

    if (rank == 0) {
        ValidationMetrics m1 = Validator::verify(seqOut1, mpiOut1, 1e-5);
        bool pass1 = m1.passed;
        std::cout << "  - Comparison against sequential reference: " << (pass1 ? "IDENTICAL" : "FAIL") << "\n";
        std::cout << "  - Max Absolute Error: " << m1.maxAbsoluteError << "\n";
        std::cout << "Result: " << (pass1 ? "PASS" : "FAIL") << "\n";
        if (!pass1) allPassed = false;
    }

    // -------------------------------------------------------------
    // Test 2: Identity Filter (W = 1, N = 24)
    // -------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n[TEST 2] Identity Filter Test (W = 1, N = 24):\n";
    }
    std::vector<SampleType> testInput2;
    if (rank == 0) {
        testInput2.resize(24);
        for (size_t i = 0; i < 24; ++i) {
            testInput2[i] = static_cast<SampleType>(i * 2.7f - 8.5f);
        }
    }
    std::vector<SampleType> mpiOut2;
    movingAverageMPIOpt(testInput2, mpiOut2, 1, MPI_COMM_WORLD, nullptr, &err);

    if (rank == 0) {
        ValidationMetrics m2 = Validator::verify(testInput2, mpiOut2, 1e-6);
        bool pass2 = m2.passed && (m2.maxAbsoluteError == 0.0);
        std::cout << "  - Max Absolute Error: " << m2.maxAbsoluteError << "\n";
        std::cout << "Result: " << (pass2 ? "PASS" : "FAIL") << "\n";
        if (!pass2) allPassed = false;
    }

    // -------------------------------------------------------------
    // Test 3: Boundary & Interior Check (N = 48, W = 5)
    // -------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n[TEST 3] Boundary & Interior Check (N = 48, W = 5):\n";
    }
    std::vector<SampleType> testInput3;
    if (rank == 0) {
        testInput3 = SignalGenerator::generateNoisySine(48, 1000.0f, 5.0f, 0.5f, 101);
    }
    std::vector<SampleType> seqOut3, mpiOut3;
    if (rank == 0) {
        movingAverageSequential(testInput3, seqOut3, 5, &err);
    }
    movingAverageMPIOpt(testInput3, mpiOut3, 5, MPI_COMM_WORLD, nullptr, &err);

    if (rank == 0) {
        ValidationMetrics m3 = Validator::verify(seqOut3, mpiOut3, 1e-5);
        bool pass3 = m3.passed;
        std::cout << "  - Max Absolute Error: " << m3.maxAbsoluteError << "\n";
        std::cout << "Result: " << (pass3 ? "PASS" : "FAIL") << "\n";
        if (!pass3) allPassed = false;
    }

    // -------------------------------------------------------------
    // Test 4: Medium Scale Equivalence (N = 10,000, W = 15)
    // -------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n[TEST 4] Medium Scale (N = 10,000, W = 15) Validation:\n";
    }
    std::vector<SampleType> testInput4;
    if (rank == 0) {
        testInput4 = SignalGenerator::generateNoisySine(10000, 1000.0f, 5.0f, 0.5f, 42);
    }
    std::vector<SampleType> seqOut4, mpiOut4;
    if (rank == 0) {
        movingAverageSequential(testInput4, seqOut4, 15, &err);
    }
    movingAverageMPIOpt(testInput4, mpiOut4, 15, MPI_COMM_WORLD, nullptr, &err);

    if (rank == 0) {
        ValidationMetrics m4 = Validator::verify(seqOut4, mpiOut4, 1e-5);
        bool pass4 = m4.passed;
        std::cout << "  - Max Absolute Error: " << std::scientific << std::setprecision(2) << m4.maxAbsoluteError << "\n";
        std::cout << "  - RMSE              : " << std::scientific << std::setprecision(2) << m4.rootMeanSquareError << "\n";
        std::cout << "Result: " << (pass4 ? "PASS" : "FAIL") << "\n";
        if (!pass4) allPassed = false;
    }

    // -------------------------------------------------------------
    // Test 5: Large Scale Equivalence (N = 100,000, W = 31)
    // -------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n[TEST 5] Large Scale (N = 100,000, W = 31) Validation:\n";
    }
    std::vector<SampleType> testInput5;
    if (rank == 0) {
        testInput5 = SignalGenerator::generateNoisySine(100000, 1000.0f, 5.0f, 0.5f, 999);
    }
    std::vector<SampleType> seqOut5, mpiOut5;
    if (rank == 0) {
        movingAverageSequential(testInput5, seqOut5, 31, &err);
    }
    movingAverageMPIOpt(testInput5, mpiOut5, 31, MPI_COMM_WORLD, nullptr, &err);

    if (rank == 0) {
        ValidationMetrics m5 = Validator::verify(seqOut5, mpiOut5, 1e-5);
        bool pass5 = m5.passed;
        std::cout << "  - Max Absolute Error: " << std::scientific << std::setprecision(2) << m5.maxAbsoluteError << "\n";
        std::cout << "  - RMSE              : " << std::scientific << std::setprecision(2) << m5.rootMeanSquareError << "\n";
        std::cout << "Result: " << (pass5 ? "PASS" : "FAIL") << "\n";
        if (!pass5) allPassed = false;

        std::cout << "\n=======================================================\n";
        std::cout << " OVERALL TEST STATUS: " << (allPassed ? "ALL TESTS PASSED [SUCCESS]" : "SOME TESTS FAILED") << "\n";
        std::cout << "=======================================================\n\n";
    }

    int passInt = allPassed ? 1 : 0;
    MPI_Bcast(&passInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return (passInt == 1);
}

void runBenchmarkSuite(size_t n, int windowSize, unsigned int seed, int rank, int worldSize) {
    if (rank == 0) {
        std::cout << "\n=========================================================================\n";
        std::cout << " OPTIMIZED MPI (NON-BLOCKING OVERLAP) BENCHMARK EXPERIMENT\n";
        std::cout << " Workload: N = " << n << " samples, Window W = " << windowSize << " (radius k = " << (windowSize - 1) / 2 << ")\n";
        std::cout << " MPI Processes (P): " << worldSize << "\n";
        std::cout << " Overlap: MPI_Isend / MPI_Irecv + Interior Stencil + MPI_Waitall\n";
        std::cout << " Runs: 5 (1 warm-up discarded + 4 averaged)\n";
        std::cout << "=========================================================================\n";
    }

    std::vector<SampleType> inputSignal;
    if (rank == 0) {
        std::cout << "Generating benchmark input signal..." << std::flush;
        inputSignal = SignalGenerator::generateNoisySine(n, 1000.0f, 5.0f, 0.5f, seed);
        std::cout << " Done.\n";
    }

    std::string err;
    std::vector<SampleType> seqReference;
    double seqAvgMs = 0.0;

    // 1. Measure Sequential Reference on Rank 0
    if (rank == 0) {
        std::cout << "Measuring Sequential Baseline (5 runs)...\n";
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
            } else {
                seqTotalMs += ms;
            }
        }
        seqAvgMs = seqTotalMs / (numRuns - 1);
        std::cout << ">>> Measured Sequential Baseline Average: " 
                  << std::fixed << std::setprecision(3) << seqAvgMs << " ms <<<\n\n";
    }

    // 2. Measure Optimized MPI Performance
    const int numRuns = 5;
    double totalInteriorMs = 0.0;
    double totalWaitMs = 0.0;
    double totalBoundaryMs = 0.0;
    double totalComputeMs = 0.0;
    double totalScatterMs = 0.0;
    double totalGatherMs = 0.0;
    double totalEndToEndMs = 0.0;
    std::vector<SampleType> lastOutput;

    for (int run = 0; run < numRuns; ++run) {
        std::vector<SampleType> tmpOutput;
        MPIOptTimings t = {};
        movingAverageMPIOpt(inputSignal, tmpOutput, windowSize, MPI_COMM_WORLD, &t, &err);

        if (run == 0) {
            if (rank == 0) lastOutput = std::move(tmpOutput);
        } else {
            if (rank == 0) {
                totalInteriorMs  += t.interiorComputeMs;
                totalWaitMs      += t.unhiddenWaitMs;
                totalBoundaryMs  += t.boundaryComputeMs;
                totalComputeMs   += t.totalComputeMs;
                totalScatterMs   += t.scatterMs;
                totalGatherMs    += t.gatherMs;
                totalEndToEndMs  += t.totalMs;
            }
        }
    }

    if (rank == 0) {
        int validRuns = numRuns - 1;
        double avgInterior  = totalInteriorMs / validRuns;
        double avgWait      = totalWaitMs / validRuns;
        double avgBoundary  = totalBoundaryMs / validRuns;
        double avgCompute   = totalComputeMs / validRuns;
        double avgScatter   = totalScatterMs / validRuns;
        double avgGather    = totalGatherMs / validRuns;
        double avgEndToEnd  = totalEndToEndMs / validRuns;

        double speedup = seqAvgMs / avgEndToEnd;
        double computeSpeedup = seqAvgMs / avgCompute;
        double efficiency = (speedup / worldSize) * 100.0;

        ValidationMetrics vm = Validator::verify(seqReference, lastOutput, 1e-4);

        std::cout << "\n=========================================================================\n";
        std::cout << " OPTIMIZED MPI TIMING BREAKDOWN (Averaged over " << validRuns << " runs)\n";
        std::cout << "=========================================================================\n";
        std::cout << "  Input Scatter Time        : " << std::setw(8) << std::fixed << std::setprecision(3) << avgScatter << " ms\n";
        std::cout << "  Overlapped Interior Compute: " << std::setw(8) << std::fixed << std::setprecision(3) << avgInterior << " ms (latency hidden)\n";
        std::cout << "  Unhidden Halo Wait Time   : " << std::setw(8) << std::fixed << std::setprecision(3) << avgWait << " ms (MPI_Waitall stall)\n";
        std::cout << "  Boundary Stencil Compute  : " << std::setw(8) << std::fixed << std::setprecision(3) << avgBoundary << " ms\n";
        std::cout << "  Total Pure Compute Time   : " << std::setw(8) << std::fixed << std::setprecision(3) << avgCompute << " ms\n";
        std::cout << "  Output Gather Time        : " << std::setw(8) << std::fixed << std::setprecision(3) << avgGather << " ms\n";
        std::cout << "  ---------------------------------------------------------\n";
        std::cout << "  Total End-to-End Time     : " << std::setw(8) << std::fixed << std::setprecision(3) << avgEndToEnd << " ms\n";
        std::cout << "  Measured Speedup          : " << std::setw(8) << std::fixed << std::setprecision(2) << speedup << "x (End-to-End)\n";
        std::cout << "  Compute Speedup           : " << std::setw(8) << std::fixed << std::setprecision(2) << computeSpeedup << "x (Computation Only)\n";
        std::cout << "  Parallel Efficiency       : " << std::setw(8) << std::fixed << std::setprecision(1) << efficiency << "%\n";
        std::cout << "  Correctness Status        : " << (vm.passed ? "PASSED [OK]" : "FAILED [MISMATCH]") << "\n";
        std::cout << "=========================================================================\n\n";

        // Append to results table if results directory exists
        std::string csvPath = "results/tables/mpi_opt_scaling.csv";
        bool fileExists = std::ifstream(csvPath).good();
        std::ofstream csv(csvPath, std::ios::app);
        if (csv) {
            if (!fileExists) {
                csv << "Processes,N,Window,SeqAvgMs,ComputeMs,InteriorMs,WaitMs,BoundaryMs,ScatterMs,GatherMs,EndToEndMs,Speedup,EfficiencyPercent,Status\n";
            }
            csv << worldSize << "," << n << "," << windowSize << "," << seqAvgMs << ","
                << avgCompute << "," << avgInterior << "," << avgWait << "," << avgBoundary << ","
                << avgScatter << "," << avgGather << "," << avgEndToEnd << ","
                << speedup << "," << efficiency << "," << (vm.passed ? "PASSED" : "FAILED") << "\n";
            std::cout << "Logged benchmark result to: " << csvPath << "\n\n";
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t n = 100000;
    int windowSize = 15;
    unsigned int seed = 42;
    std::string savePath = "";
    std::string loadInputPath = "";
    bool runTests = false;
    bool runBenchmark = false;
    int repetitions = 5;

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
        } else if (arg == "--benchmark") {
            runBenchmark = true;
        } else if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: mpiexec -n <P> " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  -n, --size <N>       Signal size (default: 100000)\n"
                          << "  -w, --window <W>     Moving-average window size (must be odd, default: 15)\n"
                          << "  -s, --seed <S>       Random seed for signal generator (default: 42)\n"
                          << "  -r, --runs <R>       Number of benchmark repetitions (default: 5)\n"
                          << "  --load <path>        Path to load binary input signal\n"
                          << "  --test               Run built-in correctness test suite\n"
                          << "  --benchmark          Run benchmark suite\n"
                          << "  --save <path>        Path to save output binary signal\n"
                          << "  -h, --help           Display this help message\n";
            }
            MPI_Finalize();
            return 0;
        }
    }

    if (runTests) {
        bool passed = runCorrectnessTests(rank, worldSize);
        MPI_Finalize();
        return passed ? 0 : 1;
    }

    if (runBenchmark) {
        runBenchmarkSuite(n == 100000 ? 1000000 : n,
                          windowSize == 15 ? 31 : windowSize,
                          seed, rank, worldSize);
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        std::cout << "=========================================================\n";
        std::cout << " CSS311 Assignment-2: Optimized MPI Moving-Average Parallel\n";
        std::cout << "=========================================================\n";
        std::cout << "Configuration:\n";
        std::cout << "  Signal Size (N)     : " << n << " samples\n";
        std::cout << "  Window Size (W)     : " << windowSize << " (radius k = " << (windowSize - 1) / 2 << ")\n";
        std::cout << "  Random Seed         : " << seed << "\n";
        std::cout << "  MPI Processes (P)   : " << worldSize << "\n";
        std::cout << "  Halo Exchange Policy: Non-Blocking (MPI_Isend / MPI_Irecv)\n";
        std::cout << "  Latency Hiding      : Overlapped Interior Stencil Computation\n";
        std::cout << "  Repetitions         : " << repetitions << " runs (1 warm-up discarded)\n";
        std::cout << "---------------------------------------------------------\n";
    }

    // 1. Prepare Input Signal on Rank 0
    std::vector<SampleType> inputSignal;
    if (rank == 0) {
        if (!loadInputPath.empty()) {
            std::cout << "Loading input signal from: " << loadInputPath << " ... " << std::flush;
            if (!SignalGenerator::loadBinary(loadInputPath, inputSignal)) {
                std::cerr << "Failed to load input file: " << loadInputPath << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            n = inputSignal.size();
            std::cout << " Done (" << n << " samples).\n";
        } else {
            std::cout << "Generating input signal..." << std::flush;
            inputSignal = SignalGenerator::generateNoisySine(n, 1000.0f, 5.0f, 0.5f, seed);
            std::cout << " Done.\n";
        }
    }

    // 2. Sequential Reference Run for Validation (Rank 0)
    std::vector<SampleType> seqOutput;
    double seqMs = 0.0;
    std::string err;
    if (rank == 0) {
        std::cout << "Running sequential reference baseline for validation..." << std::flush;
        Timer seqTimer;
        seqTimer.start();
        movingAverageSequential(inputSignal, seqOutput, windowSize, &err);
        seqMs = seqTimer.stop();
        std::cout << " Done (" << std::fixed << std::setprecision(3) << seqMs << " ms).\n";
    }

    // 3. Optimized MPI Parallel Filter Run
    std::vector<SampleType> mpiOutput;
    double totalInteriorMs = 0.0;
    double totalWaitMs = 0.0;
    double totalBoundaryMs = 0.0;
    double totalComputeMs = 0.0;
    double totalEndToEndMs = 0.0;

    for (int run = 0; run < repetitions; ++run) {
        std::vector<SampleType> tmpOutput;
        MPIOptTimings t = {};
        bool ok = movingAverageMPIOpt(inputSignal, tmpOutput, windowSize, MPI_COMM_WORLD, &t, &err);
        if (!ok) {
            if (rank == 0) {
                std::cerr << "\n[Error] Optimized MPI filtering failed: " << err << "\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (run == 0) {
            if (rank == 0) mpiOutput = std::move(tmpOutput); // warm-up
        } else {
            if (rank == 0) {
                totalInteriorMs += t.interiorComputeMs;
                totalWaitMs     += t.unhiddenWaitMs;
                totalBoundaryMs += t.boundaryComputeMs;
                totalComputeMs  += t.totalComputeMs;
                totalEndToEndMs += t.totalMs;
            }
        }
    }

    int exitCode = 0;
    if (rank == 0) {
        int validRuns = (repetitions > 1) ? (repetitions - 1) : 1;
        double avgInteriorMs = totalInteriorMs / validRuns;
        double avgWaitMs     = totalWaitMs / validRuns;
        double avgBoundaryMs = totalBoundaryMs / validRuns;
        double avgComputeMs  = totalComputeMs / validRuns;
        double avgEndToEndMs = totalEndToEndMs / validRuns;

        std::cout << "\n>>> Optimized MPI filter time (Compute): " << std::fixed << std::setprecision(3)
                  << avgComputeMs << " ms (Interior: " << avgInteriorMs << " ms, Boundary: " << avgBoundaryMs << " ms) <<<\n";
        std::cout << ">>> MPI_Waitall Stalling Latency: " << std::fixed << std::setprecision(3)
                  << avgWaitMs << " ms <<<\n";
        std::cout << ">>> Optimized MPI filter time (End-to-End): " << std::fixed << std::setprecision(3)
                  << avgEndToEndMs << " ms (includes scatter/gather/halo) <<<\n";
        std::cout << "MPI Processes: " << worldSize << "\n";

        double speedup = seqMs / avgEndToEndMs;
        double computeSpeedup = seqMs / avgComputeMs;
        double efficiency = (speedup / worldSize) * 100.0;

        std::cout << "Measured Speedup (End-to-End) : " << std::fixed << std::setprecision(2) << speedup << "x\n";
        std::cout << "Measured Speedup (Compute Only): " << std::fixed << std::setprecision(2) << computeSpeedup << "x\n";
        std::cout << "Parallel Efficiency           : " << std::fixed << std::setprecision(1) << efficiency << "%\n";

        // 4. Correctness Verification against Sequential Reference
        ValidationMetrics vm = Validator::verify(seqOutput, mpiOutput, 1e-4);
        Validator::printReport("Optimized MPI vs Sequential Reference", vm);

        // 5. Print Slices
        printSignalSlice("Input Signal         ", inputSignal, 5);
        printSignalSlice("Optimized MPI Output ", mpiOutput, 5);

        // 6. Save Output if Requested
        if (!savePath.empty()) {
            if (SignalGenerator::saveBinary(savePath, mpiOutput)) {
                std::cout << "\nOutput saved successfully to: " << savePath << "\n";
            }
        }

        std::cout << "=========================================================\n";
        exitCode = vm.passed ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
