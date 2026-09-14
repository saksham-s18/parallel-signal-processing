#ifndef VALIDATOR_HPP
#define VALIDATOR_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include "common.hpp"

struct ValidationMetrics {
    bool passed;
    double maxAbsoluteError;
    double maxRelativeError;
    double rootMeanSquareError;
    size_t mismatchCount;
    size_t totalElements;
    double toleranceUsed;
};

class Validator {
public:
    // Compare a test output signal against the reference sequential output signal
    static ValidationMetrics verify(const std::vector<SampleType>& reference,
                                   const std::vector<SampleType>& candidate,
                                   double absoluteTolerance = 1e-4,
                                   double relativeTolerance = 1e-4) {
        ValidationMetrics metrics = {};
        metrics.totalElements = reference.size();
        metrics.toleranceUsed = absoluteTolerance;
        metrics.passed = true;

        if (reference.size() != candidate.size()) {
            std::cerr << "[Validator Error] Size mismatch: reference=" 
                      << reference.size() << ", candidate=" << candidate.size() << std::endl;
            metrics.passed = false;
            return metrics;
        }

        double sumSquaredError = 0.0;
        double maxAbsErr = 0.0;
        double maxRelErr = 0.0;
        size_t mismatches = 0;

        for (size_t i = 0; i < reference.size(); ++i) {
            double ref = static_cast<double>(reference[i]);
            double cand = static_cast<double>(candidate[i]);
            double absErr = std::abs(cand - ref);
            double relErr = (std::abs(ref) > 1e-8) ? (absErr / std::abs(ref)) : absErr;

            if (absErr > maxAbsErr) {
                maxAbsErr = absErr;
            }
            if (relErr > maxRelErr) {
                maxRelErr = relErr;
            }

            sumSquaredError += absErr * absErr;

            if (absErr > absoluteTolerance && relErr > relativeTolerance) {
                mismatches++;
            }
        }

        metrics.maxAbsoluteError = maxAbsErr;
        metrics.maxRelativeError = maxRelErr;
        metrics.rootMeanSquareError = std::sqrt(sumSquaredError / static_cast<double>(reference.size()));
        metrics.mismatchCount = mismatches;
        metrics.passed = (mismatches == 0);

        return metrics;
    }

    static void printReport(const std::string& candidateName, const ValidationMetrics& m) {
        std::cout << "\n========================================\n";
        std::cout << " Validation Report: " << candidateName << "\n";
        std::cout << "========================================\n";
        std::cout << " Status              : " << (m.passed ? "PASSED [OK]" : "FAILED [MISMATCH]") << "\n";
        std::cout << " Total Elements      : " << m.totalElements << "\n";
        std::cout << " Max Absolute Error  : " << std::scientific << std::setprecision(6) << m.maxAbsoluteError << "\n";
        std::cout << " Max Relative Error  : " << std::scientific << std::setprecision(6) << m.maxRelativeError << "\n";
        std::cout << " Root Mean Sq Error  : " << std::scientific << std::setprecision(6) << m.rootMeanSquareError << "\n";
        std::cout << " Tolerance Limit     : " << std::scientific << std::setprecision(6) << m.toleranceUsed << "\n";
        std::cout << " Mismatched Elements : " << m.mismatchCount << " (" 
                  << std::fixed << std::setprecision(2) 
                  << (static_cast<double>(m.mismatchCount) / m.totalElements * 100.0) << "%)\n";
        std::cout << "========================================\n\n";
    }
};

#endif // VALIDATOR_HPP
