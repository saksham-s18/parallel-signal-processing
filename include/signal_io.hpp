#ifndef SIGNAL_IO_HPP
#define SIGNAL_IO_HPP

#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include "common.hpp"

class SignalGenerator {
public:
    // Generate a deterministic synthetic noisy signal (sine wave + gaussian noise)
    // Uses a fixed seed to guarantee identical logical input across all implementations
    static std::vector<SampleType> generateNoisySine(size_t n, 
                                                    SampleType samplingFreq = 1000.0f,
                                                    SampleType signalFreq = 5.0f, 
                                                    SampleType noiseStdDev = 0.5f,
                                                    unsigned int seed = 42) {
        std::vector<SampleType> signal(n);
        std::mt19937 rng(seed);
        std::normal_distribution<SampleType> dist(0.0f, noiseStdDev);

        const SampleType twoPi = 2.0f * 3.14159265358979323846f;
        for (size_t i = 0; i < n; ++i) {
            SampleType t = static_cast<SampleType>(i) / samplingFreq;
            SampleType clean = std::sin(twoPi * signalFreq * t);
            SampleType noise = dist(rng);
            signal[i] = clean + noise;
        }
        return signal;
    }

    // Save signal to simple binary file for rapid I/O across large sizes
    static bool saveBinary(const std::string& filepath, const std::vector<SampleType>& signal) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            std::cerr << "[SignalIO Error] Failed to open file for writing: " << filepath << std::endl;
            return false;
        }
        uint64_t n = signal.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(signal.data()), n * sizeof(SampleType));
        return true;
    }

    // Load signal from binary file
    static bool loadBinary(const std::string& filepath, std::vector<SampleType>& signal) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) {
            std::cerr << "[SignalIO Error] Failed to open file for reading: " << filepath << std::endl;
            return false;
        }
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        signal.resize(n);
        in.read(reinterpret_cast<char*>(signal.data()), n * sizeof(SampleType));
        return true;
    }
};

#endif // SIGNAL_IO_HPP
