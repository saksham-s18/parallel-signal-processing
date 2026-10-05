#include "moving_average_mpi_opt.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <sstream>

bool validateMPIOptParameters(size_t n, int windowSize, int worldSize, std::string* errorMsg) {
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

void computePartitionOpt(size_t n, int worldSize, std::vector<int>& counts, std::vector<int>& displs) {
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

bool movingAverageMPIOptChunk(const SampleType* localInput,
                             SampleType* localOutput,
                             size_t local_n,
                             size_t global_n,
                             int windowSize,
                             int rank,
                             int worldSize,
                             MPI_Comm comm,
                             MPIOptTimings* timings,
                             std::string* errorMsg) {
    if (local_n == 0) {
        if (errorMsg) *errorMsg = "Local chunk size cannot be zero on rank " + std::to_string(rank);
        return false;
    }

    const int k = (windowSize - 1) / 2;
    const double invW = 1.0 / static_cast<double>(windowSize);

    // 1. Prepare Extended Buffer
    const size_t totalExtended = local_n + 2 * static_cast<size_t>(k);
    std::vector<SampleType> extendedBuf(totalExtended);
    std::memcpy(&extendedBuf[k], localInput, local_n * sizeof(SampleType));

    // Edge Replication on global boundary endpoints
    if (rank == 0) {
        SampleType edgeVal = extendedBuf[k];
        for (int j = 0; j < k; ++j) {
            extendedBuf[j] = edgeVal;
        }
    }
    if (rank == worldSize - 1) {
        SampleType edgeVal = extendedBuf[k + local_n - 1];
        for (int j = 0; j < k; ++j) {
            extendedBuf[k + local_n + j] = edgeVal;
        }
    }

    // 2. Post Non-Blocking Halo Exchanges (MPI_Irecv / MPI_Isend)
    double tPostStart = MPI_Wtime();

    int leftNeighbor  = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    int rightNeighbor = (rank < worldSize - 1) ? (rank + 1) : MPI_PROC_NULL;

    MPI_Request reqs[4];

    // Post asynchronous receives for incoming halo regions
    MPI_Irecv(&extendedBuf[0],              k, MPI_SAMPLE_TYPE, leftNeighbor,  201, comm, &reqs[0]);
    MPI_Irecv(&extendedBuf[k + local_n],    k, MPI_SAMPLE_TYPE, rightNeighbor, 202, comm, &reqs[1]);

    // Post asynchronous sends of outgoing local boundary data
    MPI_Isend(&extendedBuf[k + local_n - k], k, MPI_SAMPLE_TYPE, rightNeighbor, 201, comm, &reqs[2]);
    MPI_Isend(&extendedBuf[k],              k, MPI_SAMPLE_TYPE, leftNeighbor,  202, comm, &reqs[3]);

    double tPostEnd = MPI_Wtime();
    double postMs = (tPostEnd - tPostStart) * 1000.0;

    // 3. Overlapped Computation: Process Interior Elements
    // Interior elements [k, local_n - k - 1] strictly read local data [k ... local_n + k - 1].
    // They NEVER access halo regions [0 ... k - 1] or [k + local_n ... totalExtended - 1].
    // Thus, interior computation proceeds concurrently with background network / IPC transmission!
    double tInteriorStart = MPI_Wtime();

    int64_t interiorStart = static_cast<int64_t>(k);
    int64_t interiorEnd   = static_cast<int64_t>(local_n) - static_cast<int64_t>(k);

    if (interiorEnd > interiorStart) {
        for (int64_t i = interiorStart; i < interiorEnd; ++i) {
            double sum = 0.0;
            const SampleType* windowStart = &extendedBuf[i];
            for (int j = 0; j < windowSize; ++j) {
                sum += static_cast<double>(windowStart[j]);
            }
            localOutput[i] = static_cast<SampleType>(sum * invW);
        }
    }

    double tInteriorEnd = MPI_Wtime();
    double interiorMs = (tInteriorEnd - tInteriorStart) * 1000.0;

    // 4. Synchronize Halo Exchanges
    // Wait only for any remaining network latency not fully hidden by interior computation.
    double tWaitStart = MPI_Wtime();
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    double tWaitEnd = MPI_Wtime();
    double waitMs = (tWaitEnd - tWaitStart) * 1000.0;

    // 5. Boundary Computation: Compute Left and Right Edges Using Arrived Halos
    double tBoundaryStart = MPI_Wtime();

    // Left Boundary: i in [0, min(k, local_n) - 1]
    int64_t leftBoundEnd = std::min(static_cast<int64_t>(k), static_cast<int64_t>(local_n));
    for (int64_t i = 0; i < leftBoundEnd; ++i) {
        double sum = 0.0;
        const SampleType* windowStart = &extendedBuf[i];
        for (int j = 0; j < windowSize; ++j) {
            sum += static_cast<double>(windowStart[j]);
        }
        localOutput[i] = static_cast<SampleType>(sum * invW);
    }

    // Right Boundary: i in [max(k, local_n - k), local_n - 1]
    int64_t rightBoundStart = std::max(static_cast<int64_t>(k), static_cast<int64_t>(local_n) - static_cast<int64_t>(k));
    for (int64_t i = rightBoundStart; i < static_cast<int64_t>(local_n); ++i) {
        double sum = 0.0;
        const SampleType* windowStart = &extendedBuf[i];
        for (int j = 0; j < windowSize; ++j) {
            sum += static_cast<double>(windowStart[j]);
        }
        localOutput[i] = static_cast<SampleType>(sum * invW);
    }

    double tBoundaryEnd = MPI_Wtime();
    double boundaryMs = (tBoundaryEnd - tBoundaryStart) * 1000.0;

    if (timings) {
        timings->postCommMs        = postMs;
        timings->interiorComputeMs = interiorMs;
        timings->unhiddenWaitMs    = waitMs;
        timings->boundaryComputeMs = boundaryMs;
        timings->totalComputeMs    = interiorMs + boundaryMs;
    }

    return true;
}

bool movingAverageMPIOpt(const std::vector<SampleType>& input,
                        std::vector<SampleType>& output,
                        int windowSize,
                        MPI_Comm comm,
                        MPIOptTimings* timings,
                        std::string* errorMsg) {
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &worldSize);

    double tStartTotal = MPI_Wtime();

    // 1. Validation & Synchronization
    size_t n = 0;
    int validFlag = 1;
    std::string localErr;

    if (rank == 0) {
        n = input.size();
        if (!validateMPIOptParameters(n, windowSize, worldSize, &localErr)) {
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

    // 2. Compute Partition
    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    computePartitionOpt(n, worldSize, counts, displs);

    size_t local_n = static_cast<size_t>(counts[rank]);
    std::vector<SampleType> localInput(local_n);
    std::vector<SampleType> localOutput(local_n);

    if (rank == 0 && output.size() != n) {
        output.resize(n);
    }

    // 3. Scatterv: Distribute chunks
    double tScatterStart = MPI_Wtime();
    const SampleType* sendbuf = (rank == 0) ? input.data() : nullptr;

    MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_SAMPLE_TYPE,
                 localInput.data(), counts[rank], MPI_SAMPLE_TYPE,
                 0, comm);

    double tScatterEnd = MPI_Wtime();
    double scatterMs = (tScatterEnd - tScatterStart) * 1000.0;

    // 4. Overlapped Non-Blocking Halo Execution
    MPIOptTimings chunkTimings = {};
    bool ok = movingAverageMPIOptChunk(localInput.data(), localOutput.data(), local_n,
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

    // 6. Aggregate Timings across Ranks (find maximum in each phase)
    if (timings) {
        double maxPost = 0.0, maxInterior = 0.0, maxWait = 0.0, maxBoundary = 0.0, maxCompute = 0.0, maxTotal = 0.0;
        MPI_Reduce(&chunkTimings.postCommMs,        &maxPost,     1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&chunkTimings.interiorComputeMs, &maxInterior, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&chunkTimings.unhiddenWaitMs,    &maxWait,     1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&chunkTimings.boundaryComputeMs, &maxBoundary, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&chunkTimings.totalComputeMs,    &maxCompute,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&totalMs,                       &maxTotal,    1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (rank == 0) {
            timings->scatterMs         = scatterMs;
            timings->postCommMs        = maxPost;
            timings->interiorComputeMs = maxInterior;
            timings->unhiddenWaitMs    = maxWait;
            timings->boundaryComputeMs = maxBoundary;
            timings->totalComputeMs    = maxCompute;
            timings->gatherMs          = gatherMs;
            timings->totalMs           = maxTotal;
        }
    }

    return true;
}
