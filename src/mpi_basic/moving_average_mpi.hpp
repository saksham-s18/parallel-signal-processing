#ifndef MOVING_AVERAGE_MPI_HPP
#define MOVING_AVERAGE_MPI_HPP

#include <vector>
#include <string>
#include <mpi.h>
#include "common.hpp"

// MPI Datatype matching SampleType (float)
#define MPI_SAMPLE_TYPE MPI_FLOAT

// Structure to hold detailed timing breakdown for MPI runs
struct MPITimings {
    double scatterMs;      // Time to distribute input signal to ranks
    double haloExchangeMs; // Time spent in blocking halo/ghost-cell exchange
    double computeMs;      // Time spent in pure local moving-average stencil computation
    double gatherMs;       // Time spent gathering local results back to root
    double totalMs;        // End-to-end elapsed time
};

/**
 * Validates the parameters for the MPI moving-average filter.
 * Rules:
 * - Signal length N must be greater than 0
 * - Window size W must be positive and odd (W % 2 == 1)
 * - Window size W must not exceed signal length N (W <= N)
 * - Each rank's chunk must be at least window radius k = (W - 1) / 2
 *
 * @param n Total signal length
 * @param windowSize Filter window size W
 * @param worldSize Total number of MPI processes
 * @param errorMsg Pointer to string to populate with error description on failure
 * @return true if valid, false otherwise
 */
bool validateMPIParameters(size_t n, int windowSize, int worldSize, std::string* errorMsg = nullptr);

/**
 * Computes contiguous block decomposition (counts and displacements)
 * for distributing N elements across P processes.
 *
 * @param n Total signal length
 * @param worldSize Number of MPI processes
 * @param counts Output vector of element counts per rank (size worldSize)
 * @param displs Output vector of displacements/offsets per rank (size worldSize)
 */
void computePartition(size_t n, int worldSize, std::vector<int>& counts, std::vector<int>& displs);

/**
 * Performs Basic 1D Moving-Average Filtering on a locally partitioned signal chunk
 * using blocking halo/ghost-cell exchange (MPI_Sendrecv).
 *
 * @param localInput Local owned input samples (size local_n)
 * @param localOutput Local computed output buffer (size local_n)
 * @param local_n Number of owned elements on this rank
 * @param global_n Total signal size N across all ranks
 * @param windowSize Filter window size W = 2k + 1
 * @param rank MPI rank of the current process
 * @param worldSize Total number of MPI processes in communicator
 * @param comm MPI Communicator (default MPI_COMM_WORLD)
 * @param timings Optional pointer to MPITimings structure to record fine-grained time
 * @param errorMsg Optional pointer to string to receive error details on failure
 * @return true on success, false on error
 */
bool movingAverageMPIChunk(const SampleType* localInput,
                          SampleType* localOutput,
                          size_t local_n,
                          size_t global_n,
                          int windowSize,
                          int rank,
                          int worldSize,
                          MPI_Comm comm = MPI_COMM_WORLD,
                          MPITimings* timings = nullptr,
                          std::string* errorMsg = nullptr);

/**
 * End-to-end Basic MPI Moving-Average Filter.
 * Rank 0 distributes the input signal via MPI_Scatterv, all ranks compute their
 * partition with halo exchange, and results are gathered to Rank 0 via MPI_Gatherv.
 *
 * @param input Full input signal vector (valid on Rank 0)
 * @param output Full output signal vector (assembled on Rank 0)
 * @param windowSize Filter window size W = 2k + 1
 * @param comm MPI Communicator (default MPI_COMM_WORLD)
 * @param timings Optional pointer to MPITimings structure
 * @param errorMsg Optional pointer to string to receive error details
 * @return true on success, false on error
 */
bool movingAverageMPI(const std::vector<SampleType>& input,
                     std::vector<SampleType>& output,
                     int windowSize,
                     MPI_Comm comm = MPI_COMM_WORLD,
                     MPITimings* timings = nullptr,
                     std::string* errorMsg = nullptr);

#endif // MOVING_AVERAGE_MPI_HPP
