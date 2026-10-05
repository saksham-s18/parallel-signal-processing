#ifndef MOVING_AVERAGE_MPI_OPT_HPP
#define MOVING_AVERAGE_MPI_OPT_HPP

#include <vector>
#include <string>
#include <mpi.h>
#include "common.hpp"

// MPI Datatype matching SampleType (float)
#define MPI_SAMPLE_TYPE MPI_FLOAT

// Fine-grained timing breakdown for Optimized Non-Blocking MPI
struct MPIOptTimings {
    double scatterMs;             // Signal distribution time from root
    double postCommMs;            // Time to post non-blocking Isend/Irecv
    double interiorComputeMs;     // Overlapped interior compute time (during transmission)
    double unhiddenWaitMs;        // Remaining wait time in MPI_Waitall
    double boundaryComputeMs;     // Boundary compute time after halos arrive
    double totalComputeMs;        // interiorComputeMs + boundaryComputeMs
    double gatherMs;              // Gathering local results back to root
    double totalMs;               // Total end-to-end elapsed time
};

/**
 * Validates the parameters for the optimized MPI moving-average filter.
 */
bool validateMPIOptParameters(size_t n, int windowSize, int worldSize, std::string* errorMsg = nullptr);

/**
 * Computes contiguous block decomposition (counts and displacements).
 */
void computePartitionOpt(size_t n, int worldSize, std::vector<int>& counts, std::vector<int>& displs);

/**
 * Performs Optimized 1D Moving-Average Filtering on a locally partitioned signal chunk
 * using non-blocking halo exchange (MPI_Isend / MPI_Irecv) with communication-computation overlap.
 *
 * Overlap Strategy:
 * 1. Post non-blocking Isend and Irecv for left and right halos.
 * 2. Immediately compute the interior elements [k, local_n - k - 1] which strictly depend
 *    on local owned samples, hiding communication latency behind CPU arithmetic.
 * 3. Invoke MPI_Waitall to synchronize the halo arrivals.
 * 4. Compute boundary elements [0, k - 1] and [local_n - k, local_n - 1] using halo data.
 *
 * @param localInput Local owned input samples (size local_n)
 * @param localOutput Local computed output buffer (size local_n)
 * @param local_n Number of owned elements on this rank
 * @param global_n Total signal size N across all ranks
 * @param windowSize Filter window size W = 2k + 1
 * @param rank MPI rank of the current process
 * @param worldSize Total number of MPI processes in communicator
 * @param comm MPI Communicator (default MPI_COMM_WORLD)
 * @param timings Optional pointer to MPIOptTimings structure to record fine-grained time
 * @param errorMsg Optional pointer to string to receive error details on failure
 * @return true on success, false on error
 */
bool movingAverageMPIOptChunk(const SampleType* localInput,
                             SampleType* localOutput,
                             size_t local_n,
                             size_t global_n,
                             int windowSize,
                             int rank,
                             int worldSize,
                             MPI_Comm comm = MPI_COMM_WORLD,
                             MPIOptTimings* timings = nullptr,
                             std::string* errorMsg = nullptr);

/**
 * End-to-end Optimized MPI Moving-Average Filter.
 * Root process (rank 0) provides the full input signal and receives the full output signal.
 *
 * @param input Full input signal vector (valid on Rank 0)
 * @param output Full output signal vector (assembled on Rank 0)
 * @param windowSize Filter window size W = 2k + 1
 * @param comm MPI Communicator (default MPI_COMM_WORLD)
 * @param timings Optional pointer to MPIOptTimings structure
 * @param errorMsg Optional pointer to string to receive error details
 * @return true on success, false on error
 */
bool movingAverageMPIOpt(const std::vector<SampleType>& input,
                        std::vector<SampleType>& output,
                        int windowSize,
                        MPI_Comm comm = MPI_COMM_WORLD,
                        MPIOptTimings* timings = nullptr,
                        std::string* errorMsg = nullptr);

#endif // MOVING_AVERAGE_MPI_OPT_HPP
