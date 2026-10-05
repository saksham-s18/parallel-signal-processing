#include "moving_average_mpi.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <sstream>

bool validateMPIParameters(size_t n, int windowSize, int worldSize, std::string* errorMsg) {
    if (n == 0) {
        if (errorMsg) *errorMsg = "Signal length N must be greater than 0.";
        return false;
    }
    if (windowSize <= 0) {
        if (errorMsg) *errorMsg = "Window size W must be a positive integer (received " + std::to_string(windowSize) + ").";
        return false;
    }
    if (windowSize % 2 == 0) {
        if (errorMsg) *errorMsg = "Window size W must be an odd integer (received even value " + std::to_string(windowSize) + ").";
        return false;
    }
    if (static_cast<size_t>(windowSize) > n) {
        if (errorMsg) *errorMsg = "Window size W (" + std::to_string(windowSize) + 
                                  ") cannot exceed signal length N (" + std::to_string(n) + ").";
        return false;
    }
    if (worldSize <= 0) {
        if (errorMsg) *errorMsg = "MPI world size must be at least 1 (received " + std::to_string(worldSize) + ").";
        return false;
    }

    int k = (windowSize - 1) / 2;
    size_t minChunk = n / static_cast<size_t>(worldSize);
    if (minChunk < static_cast<size_t>(k)) {
        if (errorMsg) {
            *errorMsg = "Domain decomposition constraint: Minimum partition size per rank (N/P = " +
                        std::to_string(minChunk) + ") must be >= window radius k = " + std::to_string(k) +
                        ". Increase signal size N or reduce process count P.";
        }
        return false;
    }

    return true;
}

void computePartition(size_t n, int worldSize, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(worldSize);
    displs.resize(worldSize);

    int base = static_cast<int>(n / static_cast<size_t>(worldSize));
    int rem  = static_cast<int>(n % static_cast<size_t>(worldSize));

    int offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

bool movingAverageMPIChunk(const SampleType* localInput,
                          SampleType* localOutput,
                          size_t local_n,
                          size_t global_n,
                          int windowSize,
                          int rank,
                          int worldSize,
                          MPI_Comm comm,
                          MPITimings* timings,
                          std::string* errorMsg) {
    if (local_n == 0) {
        if (errorMsg) *errorMsg = "Local chunk size cannot be zero on rank " + std::to_string(rank);
        return false;
    }

    const int k = (windowSize - 1) / 2;
    const double invW = 1.0 / static_cast<double>(windowSize);

    // 1. Allocate Extended Local Buffer with Halos
    // Layout:
    // [0 ... k-1]: Left halo (k elements)
    // [k ... k + local_n - 1]: Owned local elements (local_n elements)
    // [k + local_n ... k + local_n + k - 1]: Right halo (k elements)
    const size_t totalExtended = local_n + 2 * static_cast<size_t>(k);
    std::vector<SampleType> extendedBuf(totalExtended);

    // Copy local owned samples into the interior of the extended buffer
    std::memcpy(&extendedBuf[k], localInput, local_n * sizeof(SampleType));

    // 2. Blocking Halo Exchange
    double tHaloStart = MPI_Wtime();

    int leftNeighbor  = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    int rightNeighbor = (rank < worldSize - 1) ? (rank + 1) : MPI_PROC_NULL;

    // Shift 1 (Rightward transfer):
    // Send rightmost k owned samples to rightNeighbor; receive left halo from leftNeighbor
    MPI_Sendrecv(&extendedBuf[k + local_n - k], k, MPI_SAMPLE_TYPE, rightNeighbor, 101,
                 &extendedBuf[0],              k, MPI_SAMPLE_TYPE, leftNeighbor,  101,
                 comm, MPI_STATUS_IGNORE);

    // Shift 2 (Leftward transfer):
    // Send leftmost k owned samples to leftNeighbor; receive right halo from rightNeighbor
    MPI_Sendrecv(&extendedBuf[k],           k, MPI_SAMPLE_TYPE, leftNeighbor,  102,
                 &extendedBuf[k + local_n], k, MPI_SAMPLE_TYPE, rightNeighbor, 102,
                 comm, MPI_STATUS_IGNORE);

    // Global Boundary Clamping (Edge Replication):
    // Rank 0: Replicate first sample x[0] into left halo
    if (rank == 0) {
        SampleType edgeVal = extendedBuf[k];
        for (int j = 0; j < k; ++j) {
            extendedBuf[j] = edgeVal;
        }
    }

    // Rank P - 1: Replicate last sample x[N - 1] into right halo
    if (rank == worldSize - 1) {
        SampleType edgeVal = extendedBuf[k + local_n - 1];
        for (int j = 0; j < k; ++j) {
            extendedBuf[k + local_n + j] = edgeVal;
        }
    }

    double tHaloEnd = MPI_Wtime();
    double haloMs = (tHaloEnd - tHaloStart) * 1000.0;

    // 3. Local Moving-Average Stencil Computation
    double tCompStart = MPI_Wtime();

    for (size_t i = 0; i < local_n; ++i) {
        double sum = 0.0;
        // Window is centered at extendedBuf[k + i].
        // Span: [ (k + i) - k ... (k + i) + k ] = [ i ... i + windowSize - 1 ]
        const SampleType* windowStart = &extendedBuf[i];
        for (int j = 0; j < windowSize; ++j) {
            sum += static_cast<double>(windowStart[j]);
        }
        localOutput[i] = static_cast<SampleType>(sum * invW);
    }

    double tCompEnd = MPI_Wtime();
    double compMs = (tCompEnd - tCompStart) * 1000.0;

    if (timings) {
        timings->haloExchangeMs = haloMs;
        timings->computeMs = compMs;
    }

    return true;
}

bool movingAverageMPI(const std::vector<SampleType>& input,
                     std::vector<SampleType>& output,
                     int windowSize,
                     MPI_Comm comm,
                     MPITimings* timings,
                     std::string* errorMsg) {
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &worldSize);

    double tStartTotal = MPI_Wtime();

    // 1. Validation & Parameter Synchronization
    size_t n = 0;
    int validFlag = 1;
    std::string localErr;

    if (rank == 0) {
        n = input.size();
        if (!validateMPIParameters(n, windowSize, worldSize, &localErr)) {
            validFlag = 0;
        }
    }

    MPI_Bcast(&validFlag, 1, MPI_INT, 0, comm);
    if (!validFlag) {
        if (rank == 0 && errorMsg) {
            *errorMsg = localErr;
        }
        return false;
    }

    uint64_t n_u64 = static_cast<uint64_t>(n);
    MPI_Bcast(&n_u64, 1, MPI_UINT64_T, 0, comm);
    n = static_cast<size_t>(n_u64);

    // 2. Compute Partition on all ranks
    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    computePartition(n, worldSize, counts, displs);

    size_t local_n = static_cast<size_t>(counts[rank]);
    std::vector<SampleType> localInput(local_n);
    std::vector<SampleType> localOutput(local_n);

    if (rank == 0 && output.size() != n) {
        output.resize(n);
    }

    // 3. Scatterv: Distribute chunks from Rank 0
    double tScatterStart = MPI_Wtime();
    const SampleType* sendbuf = (rank == 0) ? input.data() : nullptr;

    MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_SAMPLE_TYPE,
                 localInput.data(), counts[rank], MPI_SAMPLE_TYPE,
                 0, comm);

    double tScatterEnd = MPI_Wtime();
    double scatterMs = (tScatterEnd - tScatterStart) * 1000.0;

    // 4. Compute Chunk with Blocking Halo Exchange
    MPITimings chunkTimings = {};
    bool ok = movingAverageMPIChunk(localInput.data(), localOutput.data(), local_n,
                                   n, windowSize, rank, worldSize,
                                   comm, &chunkTimings, errorMsg);
    if (!ok) {
        return false;
    }

    // 5. Gatherv: Assemble Output Chunks to Rank 0
    double tGatherStart = MPI_Wtime();
    SampleType* recvbuf = (rank == 0) ? output.data() : nullptr;

    MPI_Gatherv(localOutput.data(), counts[rank], MPI_SAMPLE_TYPE,
                recvbuf, counts.data(), displs.data(), MPI_SAMPLE_TYPE,
                0, comm);

    double tGatherEnd = MPI_Wtime();
    double gatherMs = (tGatherEnd - tGatherStart) * 1000.0;

    double tEndTotal = MPI_Wtime();
    double totalMs = (tEndTotal - tStartTotal) * 1000.0;

    // 6. Aggregate Timings across Ranks (find maximum time spent in each phase)
    if (timings) {
        double maxHalo = 0.0, maxCompute = 0.0, maxTotal = 0.0;
        MPI_Reduce(&chunkTimings.haloExchangeMs, &maxHalo,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&chunkTimings.computeMs,      &maxCompute, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&totalMs,                     &maxTotal,   1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (rank == 0) {
            timings->scatterMs      = scatterMs;
            timings->haloExchangeMs = maxHalo;
            timings->computeMs      = maxCompute;
            timings->gatherMs       = gatherMs;
            timings->totalMs        = maxTotal;
        }
    }

    return true;
}
