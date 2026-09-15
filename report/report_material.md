# CSS311: Parallel & Distributed Computing — Programming Assignment 1
## Final Technical Report: Parallel Signal Processing / Moving-Average Noise Reduction

---

### Front Page Metadata (Required Assignment Fields)

| Field | Details / Placeholders |
|:---|:---|
| **Group Number** | `[Insert Group Number, e.g., Group 12]` |
| **Group Leader** | `[Insert Group Leader Name & Student ID]` |
| **Group Member 2** | `[Insert Member 2 Name & Student ID]` |
| **Group Member 3** | `[Insert Member 3 Name & Student ID]` |
| **Semester & Batch Number** | `[Insert Semester & Batch, e.g., Fall 2026 / Batch 2024]` |
| **Application Theme** | Parallel Signal Processing / Noise Reduction |
| **Assigned Problem / Title** | 1D Moving-Average Noise Reduction Filter Across Sequential, OpenMP, and CUDA Paradigms |
| **Submission Date** | `[Insert Submission Date]` |
| **Google Drive Link** | `[Insert Link to Video Demo / Git Repository / Project Artifacts]` |

---

## 1. Problem Statement & Background

Digital signal processing systems frequently capture continuous physical quantities (audio streams, seismic sensor data, biomedical telemetry such as electrocardiograms, radar returns, and environmental measurements) that are inevitably corrupted by high-frequency thermal, electromagnetic, or quantization noise. A foundational technique for signal conditioning is the **Moving-Average (MA) Filter**.

The moving-average filter is a low-pass finite impulse response (FIR) filter that smoothes a sampled discrete-time 1D signal by replacing each sample with the unweighted arithmetic mean of its adjacent samples within a sliding window of size $W$. While conceptually simple, computing an unweighted moving-average filter over long time-series signals with wide filtering windows is computationally intensive, requiring $\mathcal{O}(N \times W)$ arithmetic operations.

In high-throughput real-time systems (e.g., multi-channel acoustic sensor arrays or continuous radar monitoring), single-threaded CPU processing fails to keep pace with incoming data acquisition rates. This project investigates the parallel decomposition, implementation, and scalability of the 1D moving-average noise reduction filter across three computing architectures:
1. **Sequential CPU Baseline (C++):** Serves as the unoptimized algorithmic ground truth.
2. **Multi-Core Shared-Memory Parallelism (OpenMP):** Exploits multi-threading across physical CPU cores and Hyper-Threads.
3. **Massively Parallel Many-Core GPU (CUDA Baseline):** Maps the workload onto thousands of concurrent GPU threads.
4. **Optimized GPU Architecture (CUDA Shared-Memory Tiling with Halo Cells):** Harnesses high-speed on-chip SRAM to eliminate redundant global memory bandwidth bottlenecks.

---

## 2. Mathematical Formulation & Processing Workflow

### 2.1 Filter Equation
Let $x = [x[0], x[1], \dots, x[N-1]]$ be a discrete input signal of length $N$. Let $W$ be an odd positive integer denoting the window width ($W = 2k + 1$, where $k = \frac{W - 1}{2}$ represents the half-window filter radius). The filtered output signal $y$ of length $N$ is defined as:

$$y[i] = \frac{1}{W} \sum_{j = -k}^{k} x[\text{boundary}(i + j)] \quad \text{for } i \in [0, N-1]$$

### 2.2 Boundary Handling: Edge Replication (Clamping)
At signal boundaries ($i < k$ or $i \ge N - k$), index offsets $i + j$ fall outside the valid signal index range $[0, N-1]$. In this project, **edge replication (clamping)** is enforced across all implementations:

$$\text{boundary}(p) = \min(\max(p, 0), N - 1)$$

- If $p < 0$, the sample is clamped to $x[0]$.
- If $p \ge N$, the sample is clamped to $x[N-1]$.

**Architectural Justification:**
Clamping ensures that every output element averages exactly $W$ samples and divides by the constant scaling factor $\frac{1}{W}$. This eliminates variable-divisor conditional branches across threads, avoiding thread divergence in GPU warps.

### 2.3 End-to-End Processing Workflow
The processing pipeline is structured into clear phases:
1. **Input Signal Acquisition / Generation:** Deterministic synthetic noisy sine wave generation ($x[t] = \sin(2\pi f t) + \mathcal{N}(0, \sigma^2)$) or loading binary signal files.
2. **Parameter Validation:** Verifying that $N > 0$, $W > 0$, $W \pmod 2 = 1$, and $W \le N$.
3. **Buffer Allocation:** Allocating input/output buffers on host and device.
4. **Data Transfer (GPU):** Copying input data across PCIe from Host to Device ($H2D$).
5. **Parallel Kernel Execution:** Performing $\mathcal{O}(N \times W)$ additions and multiplications in parallel.
6. **Data Transfer (GPU):** Copying results from Device to Host ($D2H$).
7. **Correctness Verification & Profiling:** Validating candidates against the sequential reference using root-mean-square error (RMSE) and maximum absolute error ($L_\infty$).

---

## 3. Parallel Decomposition, Dependencies, & Synchronization

### 3.1 Sequential vs. Parallel Portions (Amdahl's Law Analysis)
According to Amdahl's Law, the maximum theoretical speedup $S_{\text{max}}$ achievable by parallelizing a program is governed by the fraction of execution time that is strictly parallelizable ($P$) versus strictly sequential ($1 - P$):

$$S(T) = \frac{1}{(1 - P) + \frac{P}{T}}$$

In our signal filtering pipeline:
- **Strictly Sequential Portions ($1 - P$):**
  - Command-line argument parsing and parameter validation.
  - Signal allocation and disk file I/O.
  - GPU initialization, driver context creation, and PCIe host-device memory transfers ($H2D$ and $D2H$).
  - CPU-side synchronization and validation verification.
- **Parallel Portions ($P$):**
  - Moving-average stencil arithmetic: for each sample $i$, reading $W$ values, accumulating their sum, and multiplying by $\frac{1}{W}$.

Because PCIe bus transfers operate at approximately $16 \text{ GB/s}$ (PCIe Gen 4 x8) while GPU internal registers and shared memory operate at hundreds of gigabytes or terabytes per second, $1 - P$ is dominated by data marshaling for small signal sizes.

### 3.2 Data Decomposition & Race-Free Concurrency
- **Output Domain Decomposition:** The problem is decomposed by output samples. Because calculating $y[i]$ depends only on reading a window of $x$ and writing exclusively to $y[i]$, there are **zero loop-carried dependencies** between different output indices $i_1 \ne i_2$.
- **Concurrent Safe Reads:** The input signal $x$ is strictly immutable (read-only) throughout computation. All threads read overlapping regions of $x$ concurrently without read-write hazards.
- **Mutually Exclusive Writes:** Each output element $y[i]$ is written by exactly one thread or one OpenMP iteration. There are **zero write-write conflicts**.
- **Private Accumulators:** Each thread maintains a private 64-bit `double` accumulator `sum`, eliminating the need for atomics, locks, or reduction trees.

### 3.3 Synchronization Requirements
- **OpenMP:** No mutexes or locks are used inside the parallel loop. An implicit barrier at the conclusion of `#pragma omp parallel for` synchronizes all threads before returning to the caller.
- **CUDA Baseline:** Threads compute independently without inter-thread communication. Kernel completion is synchronized on the host via `cudaEventSynchronize()`.
- **CUDA Optimized (Shared Memory):** A block-wide barrier synchronization `__syncthreads()` is strictly required after cooperative loading into shared memory before any thread begins calculating moving averages.

---

## 4. Implementation Approaches

### 4.1 Sequential Implementation (C++ Reference)
The sequential baseline in [`src/sequential/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/sequential/) uses a nested loop structure:
- Outer loop traverses $i$ from $0$ to $N - 1$.
- Inner loop traverses offset $j$ from $-k$ to $+k$, clamping index $i + j$ to $[0, N-1]$.
- Accumulation is carried out in 64-bit `double` precision before casting to `float`, preventing accumulation drift over wide windows.
- Operates in $\mathcal{O}(N \times W)$ time and $\mathcal{O}(1)$ auxiliary space.

### 4.2 OpenMP Implementation
Located in [`src/openmp/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/openmp/):
```cpp
#pragma omp parallel for num_threads(threadsToUse) schedule(static) default(none) \
    shared(input, output, n, windowSize, k, invW)
for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    double sum = 0.0;
    for (int j = -k; j <= k; ++j) {
        int64_t rawIndex = i + j;
        int64_t clampedIndex = (rawIndex < 0) ? 0 : 
                               (rawIndex >= n ? n - 1 : rawIndex);
        sum += static_cast<double>(input[clampedIndex]);
    }
    output[i] = static_cast<SampleType>(sum * invW);
}
```
**Static Scheduling Rationale:**
Because edge-clamping guarantees that every iteration executes exactly $W$ reads and additions, computational work per sample is completely uniform. `schedule(static)` splits the range $[0, N-1]$ into $T$ contiguous partitions of size $\lceil N / T \rceil$, maximizing L1/L2 cache spatial locality with zero dynamic runtime scheduling overhead.

### 4.3 CUDA Baseline Implementation
Located in [`src/cuda/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/cuda/):
- **Thread Hierarchy:** 1D grid of 1D thread blocks. Block size is set to $B = 256$ threads. Grid size is $\lceil N / B \rceil$.
- **Mapping:** One thread computes one output sample: $i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$.
- **Global Memory Access:** Each thread independently loads $W$ elements from high-latency GPU DRAM (global memory).
- **Error Handling:** Every CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaEventRecord`, kernel launch `cudaGetLastError()`) is wrapped in a strict error-checking macro (`CUDA_CHECK`).
- **Timing:** Non-intrusive GPU timing using CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`) measures $H2D$, $Kernel$, and $D2H$ intervals with sub-microsecond precision.

### 4.4 CUDA Optimized Implementation (Shared-Memory Tiling with Halo Cells)
Located in [`src/optimized/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/optimized/):

```cpp
__global__ void movingAverageSharedMemKernel(const SampleType* __restrict__ d_input,
                                             SampleType* __restrict__ d_output,
                                             size_t n, int windowSize, int k, double invW) {
    extern __shared__ SampleType s_data[];

    const int tid = threadIdx.x;
    const int sharedSize = blockDim.x + 2 * k;
    const int64_t globalTileStart = static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t globalWindowStart = globalTileStart - k;

    // 1. Cooperative Shared-Memory Loading (Tile + Left & Right Halos)
    for (int sIdx = tid; sIdx < sharedSize; sIdx += blockDim.x) {
        int64_t gIdx = globalWindowStart + sIdx;
        int64_t clampedIdx = (gIdx < 0) ? 0 : (gIdx >= n ? n - 1 : gIdx);
        s_data[sIdx] = d_input[clampedIdx];
    }

    // 2. Block-wide Barrier Synchronization
    __syncthreads();

    // 3. Stencil Computation from On-Chip SRAM
    const int64_t i = globalTileStart + tid;
    if (i < n) {
        double sum = 0.0;
        #pragma unroll 4
        for (int j = 0; j < windowSize; ++j) {
            sum += static_cast<double>(s_data[tid + j]);
        }
        d_output[i] = static_cast<SampleType>(sum * invW);
    }
}
```

#### Detailed Optimization Principles:
1. **Theoretical Load-Count Reuse Model:** In the baseline kernel, each thread reads $W$ global memory words. For block size $B = 256$ and $W = 63$, a block issues $256 \times 63 = 16,128$ global memory read instructions. In the tiled kernel, cooperative loading requires only $B + 2k = 256 + 62 = 318$ global loads per block. This theoretical data-reuse model predicts a $\approx 50.7\times$ reduction in issued global memory load instructions. (Note: actual physical DRAM traffic reduction on hardware depends on L1/L2 hardware cache hit rates).
2. **Shared-Memory Bank Access Pattern:** Shared memory is organized into 32 independent banks (4-byte width per bank). During the inner stencil loop, thread $tid$ reads `s_data[tid + j]`. Across all active threads $0 \dots 31$ within a warp, memory addresses increment by 4 bytes per thread, mapping each access to an independent bank: $\text{bank} = (tid + j) \pmod{32}$. This linear access pattern satisfies conflict-free shared memory access across the warp during the inner compute loop.
3. **Warp Divergence Analysis:** Execution is not completely free of divergence. During the cooperative loading phase, boundary clamping causes branch divergence only in the first block (`blockIdx.x == 0`) and terminal block of the grid. For all interior blocks ($>99\%$ of the workload), threads follow uniform non-divergent paths. During the moving-average computation loop, all active threads in every block execute the identical loop trip count $W$, resulting in uniform, non-divergent instruction execution.

---

## 5. Experimental Environment

### Hardware Specifications
- **Processor (CPU):** 13th Gen Intel(R) Core(TM) i7-13700HX
  - Physical Cores: 16 (8 Performance-cores @ 5.0 GHz max turbo + 8 Efficient-cores @ 3.7 GHz)
  - Logical Threads: 24 (Hyper-Threading on P-cores)
  - L3 Cache: 30 MB Intel Smart Cache
- **Graphics Card (GPU):** NVIDIA GeForce RTX 4050 Laptop GPU
  - Architecture: Ada Lovelace (Compute Capability 8.9 / `sm_89`)
  - Streaming Multiprocessors (SMs): 20 SMs (2560 CUDA Cores)
  - GPU Memory: 6 GB GDDR6 (96-bit bus, ~192 GB/s theoretical bandwidth)
  - Power Limit: 50 W TGP

### Software & Toolchain Specifications
- **Operating System:** Microsoft Windows 11 Home (64-bit, Version 24H2)
- **NVIDIA GPU Driver:** 616.92
- **CUDA Toolkit:** CUDA 13.4.59 (`nvcc` compiler)
- **Host C++ Compilers:** 
  - MSVC x64 v19.51.36257 (Visual Studio 2026 Build Tools v18.10.0) for CUDA compilation
  - GCC / G++ 16.1.0 (MSYS2 UCRT64) with native `-fopenmp`
- **Python Environment:** Python 3.13.5 with NumPy 2.1.3, Pandas, and Matplotlib 3.10.0

---

## 6. Input Dataset & Correctness Verification

### 6.1 Benchmark Workload Sizes
To rigorously benchmark performance without risking laptop thermal throttling, three practical, deterministic workloads were chosen:

| Workload Tier | Signal Size ($N$) | Window Size ($W$) | Radius ($k$) | Total Arithmetic Ops ($N \times W$) | Input File |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Small** | $10,000$ | $15$ | $7$ | $150,000$ ops | `data/input/input_small_N10000.bin` |
| **Medium** | $100,000$ | $31$ | $15$ | $3,100,000$ ops | `data/input/input_medium_N100000.bin` |
| **Large** | $1,000,000$ | $63$ | $31$ | $63,000,000$ ops | `data/input/input_large_N1000000.bin` |

All signals were generated deterministically using a fixed random seed (`seed = 42`) and stored in binary format matching [`include/signal_io.hpp`](file:///c:/Users/Predator/Desktop/Signal%20Processing/include/signal_io.hpp).

### 6.2 Human-Readable Hand-Calculated Demonstration ($N = 5, W = 3$)
To verify edge-clamping arithmetic, a small deterministic test was calculated by hand and validated against all four implementations:
- Input: $x = [1.0, 2.0, 3.0, 4.0, 5.0]$, $W = 3$, $k = 1$:
  - $i=0$: $\text{clamp}([-1, 0, 1]) \rightarrow [0, 0, 1] \rightarrow \frac{1+1+2}{3} = \frac{4}{3} \approx 1.333333$
  - $i=1$: $\text{clamp}([0, 1, 2]) \rightarrow [0, 1, 2] \rightarrow \frac{1+2+3}{3} = \frac{6}{3} = 2.000000$
  - $i=2$: $\text{clamp}([1, 2, 3]) \rightarrow [1, 2, 3] \rightarrow \frac{2+3+4}{3} = \frac{9}{3} = 3.000000$
  - $i=3$: $\text{clamp}([2, 3, 4]) \rightarrow [2, 3, 4] \rightarrow \frac{3+4+5}{3} = \frac{12}{3} = 4.000000$
  - $i=4$: $\text{clamp}([3, 4, 5]) \rightarrow [3, 4, 4] \rightarrow \frac{4+5+5}{3} = \frac{14}{3} \approx 4.666667$

**Measured Equivalence Table (Logged from `results/tables/small_demo_table.csv`):**

| Index ($i$) | Input $x[i]$ | Hand Calculation | Sequential | OpenMP ($T=4$) | CUDA Baseline | CUDA Opt | Absolute Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | $1.0000$ | $1.333333$ | $1.333333$ | $1.333333$ | $1.333333$ | $1.333333$ | **0.000000** |
| **1** | $2.0000$ | $2.000000$ | $2.000000$ | $2.000000$ | $2.000000$ | $2.000000$ | **0.000000** |
| **2** | $3.0000$ | $3.000000$ | $3.000000$ | $3.000000$ | $3.000000$ | $3.000000$ | **0.000000** |
| **3** | $4.0000$ | $4.000000$ | $4.000000$ | $4.000000$ | $4.000000$ | $4.000000$ | **0.000000** |
| **4** | $5.0000$ | $4.666667$ | $4.666667$ | $4.666667$ | $4.666667$ | $4.666667$ | **0.000000** |

### 6.3 Comprehensive Correctness Across All Workload Sizes
For every benchmark size, the outputs produced by OpenMP, CUDA Baseline, and CUDA Optimized were checked against the Sequential reference output. **In all cases, Maximum Absolute Error was $0.00\times 10^0$ (maximum absolute error of 0 within measured single-precision floating-point precision).**

---

## 7. Performance Benchmarks & Results

*All measurements reflect real executions on the target machine. Each timing represents the arithmetic mean of 4 timed runs following 1 discarded warm-up run.*

### 7.1 Full Performance Comparison Table Across All Workloads

| Implementation | Input Size | $N$ | Window $W$ | Threads / Block | Kernel Time (ms) | H2D Time (ms) | D2H Time (ms) | End-to-End Time (ms) | Speedup vs Seq | Efficiency (%) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sequential** | Small | $10,000$ | $15$ | $1$ | N/A | N/A | N/A | **0.1290** | $1.00\times$ | $100.0\%$ | REFERENCE |
| **OpenMP ($T=1$)** | Small | $10,000$ | $15$ | $1$ | N/A | N/A | N/A | **0.0930** | $1.39\times$ | $138.7\%$ | PASSED |
| **OpenMP ($T=8$)** | Small | $10,000$ | $15$ | $8$ | N/A | N/A | N/A | **0.0870** | $1.48\times$ | $18.5\%$ | PASSED |
| **OpenMP ($T=24$)** | Small | $10,000$ | $15$ | $24$ | N/A | N/A | N/A | **0.3120** | **0.41x (SLOWER)** | $1.7\%$ | PASSED |
| **CUDA Baseline** | Small | $10,000$ | $15$ | $256$ | $0.0164$ | $0.0161$ | $0.0254$ | **0.0579** | $2.23\times$ (Kernel: $7.87\times$) | N/A | PASSED |
| **CUDA Optimized** | Small | $10,000$ | $15$ | $256$ | $0.0340$ | $0.0390$ | $0.0598$ | **0.1327** | **0.97x (SLOWER)** (Kernel: $3.79\times$) | N/A | PASSED |
| | | | | | | | | | | | |
| **Sequential** | Medium | $100,000$ | $31$ | $1$ | N/A | N/A | N/A | **1.6540** | $1.00\times$ | $100.0\%$ | REFERENCE |
| **OpenMP ($T=1$)** | Medium | $100,000$ | $31$ | $1$ | N/A | N/A | N/A | **2.0270** | $0.82\times$ | $81.6\%$ | PASSED |
| **OpenMP ($T=8$)** | Medium | $100,000$ | $31$ | $8$ | N/A | N/A | N/A | **0.5650** | $2.93\times$ | $36.6\%$ | PASSED |
| **OpenMP ($T=24$)** | Medium | $100,000$ | $31$ | $24$ | N/A | N/A | N/A | **0.4160** | $3.98\times$ | $16.6\%$ | PASSED |
| **CUDA Baseline** | Medium | $100,000$ | $31$ | $256$ | $0.1039$ | $0.0773$ | $0.1164$ | **0.2976** | $5.56\times$ (Kernel: $15.92\times$) | N/A | PASSED |
| **CUDA Optimized** | Medium | $100,000$ | $31$ | $256$ | $0.1059$ | $0.0915$ | $0.1284$ | **0.3258** | $5.08\times$ (Kernel: $15.62\times$) | N/A | PASSED |
| | | | | | | | | | | | |
| **Sequential** | Large | $1,000,000$ | $63$ | $1$ | N/A | N/A | N/A | **39.5710** | $1.00\times$ | $100.0\%$ | REFERENCE |
| **OpenMP ($T=1$)** | Large | $1,000,000$ | $63$ | $1$ | N/A | N/A | N/A | **37.3770** | $1.06\times$ | $105.9\%$ | PASSED |
| **OpenMP ($T=8$)** | Large | $1,000,000$ | $63$ | $8$ | N/A | N/A | N/A | **6.9710** | $5.68\times$ | $71.0\%$ | PASSED |
| **OpenMP ($T=24$)** | Large | $1,000,000$ | $63$ | $24$ | N/A | N/A | N/A | **5.7330** | $6.90\times$ | $28.8\%$ | PASSED |
| **CUDA Baseline** | Large | $1,000,000$ | $63$ | $256$ | $1.6661$ | $0.6565$ | $0.7314$ | **3.0540** | $12.96\times$ (Kernel: $23.75\times$) | N/A | PASSED |
| **CUDA Optimized** | Large | $1,000,000$ | $63$ | $256$ | **1.5797** | $0.5723$ | $0.6970$ | **2.8489** | **13.89x** (Kernel: **25.05x**) | N/A | PASSED |

*(Data source: `results/tables/full_benchmark_comparison.csv`)*

---

### 7.2 OpenMP Thread Scaling Analysis ($N = 1,000,000, W = 31$)

*Tested across thread counts $T \in \{1, 2, 4, 8, 16, 24\}$ on Intel Core i7-13700HX (Reference sequential baseline on identical workload: **15.816 ms**):*

| Threads ($T$) | Average Time (ms) | Speedup ($S = T_{\text{seq}} / T_{\text{omp}}$) | Parallel Efficiency ($E = S / T$) | Correctness |
|:---:|:---:|:---:|:---:|:---:|
| **1** | $15.373$ ms | $1.03\times$ | $102.9\%$ | PASSED |
| **2** | $9.316$ ms | $1.70\times$ | $84.9\%$ | PASSED |
| **4** | $5.206$ ms | $3.04\times$ | $75.9\%$ | PASSED |
| **8** | $3.982$ ms | $3.97\times$ | $49.6\%$ | PASSED |
| **16** | **3.282 ms** | **4.82x** | $30.1\%$ | PASSED |
| **24** | $3.427$ ms | $4.62\times$ | $19.2\%$ | PASSED |

*(Data source: `results/tables/omp_thread_scaling.csv`)*

#### OpenMP Scalability Observations:
1. **Single-Thread Efficiency Interpretation ($102.9\%$):** At $T=1$, the OpenMP run completed in $15.373$ ms compared to the sequential baseline's $15.816$ ms, producing a nominal efficiency of $102.9\%$. This should be understood strictly as normal run-to-run system measurement variance, slight CPU frequency turbo boost differences, and minor compiler loop scheduling variations rather than genuine super-linear parallel scaling.
2. **Linear Scaling up to 4 Threads:** From $T=1$ ($15.37$ ms) to $T=4$ ($5.21$ ms), speedup scales cleanly to $3.04\times$ ($76\%$ efficiency) as computation maps directly onto physical P-cores without thread contention.
3. **Diminishing Returns beyond 8 Cores:** The processor features 8 Performance cores and 8 Efficient cores. When utilizing 16 threads, speedup peaks at **$4.82\times$ ($3.282$ ms)**.
4. **Hyper-Threading Inversion at $T = 24$:** At 24 threads, execution time slightly degrades to $3.427$ ms ($4.62\times$ speedup, efficiency dropping to $19.2\%$). Moving-average is a memory-bound stencil with high memory bandwidth saturation. SMT (Hyper-Threading) threads share the same L1/L2 caches and memory channels, introducing cache eviction conflicts and thread scheduling overhead that exceed any computational gain.

---

### 7.3 CUDA Timing Breakdown & PCIe Transfer Overhead

| Implementation | Input Size | $N$ | Window $W$ | H2D Transfer | Kernel Time | D2H Transfer | End-to-End Time | Transfer Overhead Ratio (%) | Kernel Speedup | End-to-End Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CUDA Baseline** | Small | $10,000$ | $15$ | $0.0161$ ms | $0.0164$ ms | $0.0254$ ms | $0.0579$ ms | **71.7%** | $7.87\times$ | $2.23\times$ |
| **CUDA Optimized** | Small | $10,000$ | $15$ | $0.0390$ ms | $0.0340$ ms | $0.0598$ ms | $0.1327$ ms | **74.5%** | $3.79\times$ | **0.97x (Slower)** |
| **CUDA Baseline** | Medium | $100,000$ | $31$ | $0.0773$ ms | $0.1039$ ms | $0.1164$ ms | $0.2976$ ms | **65.1%** | $15.92\times$ | $5.56\times$ |
| **CUDA Optimized** | Medium | $100,000$ | $31$ | $0.0915$ ms | $0.1059$ ms | $0.1284$ ms | $0.3258$ ms | **67.5%** | $15.62\times$ | $5.08\times$ |
| **CUDA Baseline** | Large | $1,000,000$ | $63$ | $0.6565$ ms | $1.6661$ ms | $0.7314$ ms | $3.0540$ ms | **45.4%** | $23.75\times$ | $12.96\times$ |
| **CUDA Optimized** | Large | $1,000,000$ | $63$ | $0.5723$ ms | **1.5797 ms** | $0.6970$ ms | **2.8489 ms** | **44.6%** | **25.05x** | **13.89x** |

*(Data source: `results/tables/cuda_breakdown.csv`)*

---

## 8. Deep Technical Discussion

### 8.1 When Parallelization Does Not Provide Significant Benefit or Is Slower
A critical takeaway required by the assignment is identifying when parallelization is ineffective or detrimental:

1. **Small Workloads and OpenMP Over-Threading ($N=10,000, T=24$):**
   - Sequential time on Small input is only **$0.1290$ ms**.
   - Running OpenMP with 24 threads takes **$0.3120$ ms** ($0.41\times$ speedup — **2.4 times slower than sequential!**).
   - *Cause:* Assigning only $\approx 416$ samples per thread creates more thread dispatching, barrier synchronization, and cache coherency traffic than the computation itself takes.
2. **Small Workloads and GPU PCIe Transfer Dominance ($N=10,000$):**
   - For the Small input, CUDA Optimized End-to-End time is **$0.1327$ ms**, which is slower than Sequential ($0.1290$ ms, speedup $0.97\times$).
   - As shown in the breakdown table, **$74.5\%$ of the total GPU time is consumed by PCIe transfers** ($H2D$ + $D2H$), while the kernel executes in only $0.034$ ms.
   - For tiny datasets, copying data to the GPU and back over PCIe introduces an latency penalty that exceeds the CPU computation time.
3. **GPU Kernel vs. End-to-End Speedup Gap:**
   - On the Large dataset ($N=1,000,000, W=63$), the CUDA Optimized **kernel executes in $1.58$ ms, achieving a $25.05\times$ compute speedup**.
   - However, when accounting for memory transfers ($0.57$ ms $H2D$ + $0.70$ ms $D2H$), the **End-to-End execution time is $2.85$ ms, yielding a $13.89\times$ speedup**.
   - Hiding PCIe overhead is impossible in single-burst filter passes unless overlapping asynchronous streams (`cudaMemcpyAsync`) or pipelined double-buffering are employed.

### 8.2 Baseline CUDA vs. Optimized Shared-Memory CUDA
- On Small ($W=15$) and Medium ($W=31$) workloads, CUDA Baseline is slightly faster than CUDA Optimized because the window radius is small ($k=7$ and $k=15$) and GPU L1/L2 caches successfully cache contiguous global reads, whereas shared memory tiling incurs cooperative load loop overhead and a `__syncthreads()` barrier.
- On Large workloads with wide windows ($N=1,000,000, W=63$), the **shared-memory optimization outperforms baseline** ($1.58$ ms vs $1.67$ ms kernel time). At $W=63$, each thread must perform 63 memory operations; loading them once into shared memory reduces global memory bus traffic by over $50\times$, overcoming the barrier synchronization overhead.

---

## 9. Visualizations & Plot Summary

The following charts were generated from actual CSV benchmarks and are saved under [`results/plots/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/results/plots/):

1. **`execution_time_comparison.png`:** Compares execution times on a logarithmic scale across Sequential, OpenMP, CUDA Baseline, and CUDA Optimized for Small, Medium, and Large datasets.
2. **`speedup_comparison.png`:** Plots measured speedup across sizes, highlighting how GPU speedup scales dramatically with workload size while illustrating the gap between Kernel-only and End-to-End speedups.
3. **`omp_thread_scaling.png`:** Displays speedup and parallel efficiency curves for OpenMP from $T=1$ to $T=24$, showing peak performance at 16 threads and efficiency decline due to Hyper-Threading contention.
4. **`cuda_breakdown.png`:** Stacked bar chart showing absolute milliseconds and relative percentage of $H2D$, $Kernel$, and $D2H$ times, proving that transfer overhead dominates small workloads ($74\%$).
5. **`signal_denoising_demo.png`:** Plots the raw noisy sine signal alongside the moving-average filtered output, visually demonstrating effective high-frequency noise attenuation.

---

## 10. Conclusion

This project successfully implemented, verified, and benchmarked a 1D moving-average noise reduction filter across Sequential, OpenMP, CUDA Baseline, and CUDA Shared-Memory Optimized implementations:
1. **Correctness:** All parallel implementations achieved $100\%$ numerical equivalence with the sequential baseline ($0.00\times 10^0$ maximum absolute error) and matched hand-calculated boundary derivations.
2. **CPU Scalability:** OpenMP achieved a maximum speedup of **$4.82\times$** on 16 threads for $N = 1,000,000$. Over-threading beyond physical cores ($T=24$) caused memory bus saturation.
3. **GPU Acceleration:** CUDA achieved a massive **$25.05\times$ kernel speedup** and **$13.89\times$ end-to-end speedup** on the RTX 4050 Laptop GPU for large workloads ($N=1\text{M}, W=63$).
4. **Memory Bottlenecks:** PCIe data transfers represent up to $74\%$ of execution time on small workloads, proving that GPU acceleration is only advantageous when workload arithmetic intensity is high enough to amortize communication overhead.

---

## 11. Individual Contribution Placeholders

| Member Name | Student ID | Primary Responsibilities & Contributions |
|:---|:---:|:---|
| `[Group Leader Name]` | `[Student ID]` | `[Project coordination, CUDA shared-memory optimization implementation, profiling]` |
| `[Group Member 2]` | `[Student ID]` | `[Sequential baseline, OpenMP parallel implementation, thread scalability analysis]` |
| `[Group Member 3]` | `[Student ID]` | `[CUDA baseline implementation, dataset generator, plotting scripts, report compilation]` |
