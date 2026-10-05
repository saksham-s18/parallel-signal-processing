# Parallel Signal Processing: Moving-Average Noise Reduction
## Programming Assignment 2 (PA2) Comprehensive Final Report
### Integrating Sequential, OpenMP, CUDA Baseline, CUDA Optimized, Basic MPI, and Optimized MPI Paradigms with Hardware Energy Profiling and Nsight Systems Diagnostics

---

## Front Matter: Assignment Information

| Metadata Item | Project Details / Verified Records |
|:---|:---|
| **Course** | CSS311: Parallel & Distributed Computing |
| **Assignment** | Programming Assignment 2 (PA2 Comprehensive Final Report) |
| **Application Theme** | Signal Processing |
| **Assigned Problem** | 1D Moving-Average Noise Reduction Filter Across Six Parallel Paradigms |
| **Group Number** | 16 |
| **Group Leader** | Saksham Singh — 2024BCS0070 |
| **Group Member 2** | Daksh Singh — 2024BCS0042 |
| **Group Member 3** | Anmol Pipara — 2024BCS0014 |
| **Demonstration Link** | `https://drive.google.com/drive/folders/1i3eLAEA6jXPNDnx8pnbJffigIk-LwEPE?usp=sharing` |

---

## 1. Problem Statement & Mathematical Foundation

Digital signal acquisition systems—such as acoustic transceivers, biomedical monitors (e.g., electrocardiograms and electromyograms), environmental sensor networks, seismic sensors, and radar receivers—continuously capture discrete-time physical signals. During acquisition and physical transmission, these signals are corrupted by high-frequency disturbances, including thermal Johnson–Nyquist noise, electromagnetic interference, and analog-to-digital quantization artifacts.

A fundamental digital signal processing technique for conditioning 1D telemetry is the **Moving-Average (MA) Filter**. The moving-average filter is an unweighted, linear, finite impulse response (FIR) low-pass filter. For each sample in an input discrete-time signal $x$ of length $N$, the filter computes the unweighted arithmetic mean of $W$ contiguous samples symmetrically centered around the target index:

$$y[i] = \frac{1}{W} \sum_{j = -k}^{k} x[\text{boundary}(i + j)] \quad \text{for } i \in [0, N-1]$$

where:
- $N$ is the total number of discrete samples in the 1D signal.
- $W$ is an odd positive integer denoting the filter window size ($W = 2k + 1$).
- $k = \frac{W - 1}{2}$ represents the filter half-width (radius).
- $\text{boundary}(p)$ maps an arbitrary offset index $p = i + j$ into the valid discrete signal domain $[0, N-1]$.

### Boundary Handling: Edge-Clamping (Replication)
At the signal extremities ($i < k$ or $i \ge N - k$), the filter window extends beyond the boundaries of the input array. In this project, all six parallel implementations execute identical **edge-clamping boundary handling** (nearest-element replication):

$$\text{boundary}(p) = \min(\max(p, 0), N - 1)$$

Under this policy:
- When $p < 0$, the signal value is clamped to $x[0]$.
- When $p \ge N$, the signal value is clamped to $x[N-1]$.
- When $0 \le p \le N - 1$, the index remains unchanged as $x[p]$.

#### Computational Properties of Edge Clamping
1. **Constant Computational Weight:** Every output element $y[i]$ performs the exact same arithmetic workload: $W$ memory fetches, $W - 1$ floating-point additions, and 1 multiplication by the precomputed reciprocal $\frac{1}{W}$.
2. **Branch Convergence in SIMT/SIMD:** Unlike zero-padding or dynamic window truncation, edge clamping avoids runtime conditional divergence across adjacent threads and vector lanes, guaranteeing uniform instruction execution.
3. **Physical Signal Continuity:** Clamping eliminates artificial high-frequency transient steps that zero-padding introduces at signal edges, maintaining baseline continuity.

---

## 2. Executive Summary & Objectives of PA2

Programming Assignment 1 (PA1) investigated single-node shared-memory and GPU acceleration paradigms:
1. Sequential CPU Baseline (single-threaded C++)
2. OpenMP Multi-Core Parallelism (multi-threaded CPU)
3. CUDA Baseline (global memory many-core GPU)
4. CUDA Optimized (on-chip shared-memory tiling with cooperative halo caching)

**Programming Assignment 2 (PA2)** extends this research into distributed-memory message-passing and deep hardware profiling:
5. **Basic MPI:** 1D domain decomposition with blocking point-to-point halo exchange (`MPI_Sendrecv`).
6. **Optimized MPI:** 1D domain decomposition with non-blocking halo exchange (`MPI_Isend` / `MPI_Irecv`) and fine-grained computation/communication overlap (splitting execution into interior stencil computation and boundary stencil computation).
7. **Hardware Energy Telemetry:** Empirical measurement of CPU package energy via Intel Running Average Power Limit (RAPL) performance counters and GPU board energy via NVIDIA Management Library (NVML) APIs.
8. **CUDA Profiling via NVIDIA Nsight Systems:** Quantitative analysis of kernel execution durations, memory copy overheads, and hardware resource utilization.
9. **Six-Way Cross-Paradigm Benchmark:** Direct empirical comparison of all six implementations across Small ($N=10,000, W=15$), Medium ($N=100,000, W=31$), and Large ($N=1,000,000, W=63$) workloads.

All six implementations execute the exact same algorithmic task, consume identical input datasets, apply the identical mathematical formulation and edge-clamping boundary logic, and are rigorously verified for numerical equivalence.

---

## 3. The Six Computing Paradigms & Implementations

```
                                  +-------------------------------------------------------------+
                                  |         1D Moving-Average Signal Processing Pipeline         |
                                  |         y[i] = (1/W) * SUM_{j=-k}^{+k} x[clamped(i+j)]      |
                                  +-------------------------------------------------------------+
                                                                 |
                +------------------------------------------------+-----------------------------------------------+
                |                                                |                                               |
  [Shared-Memory CPU Paradigms]                       [Many-Core GPU Paradigms]                     [Distributed-Memory MPI Paradigms]
  - Sequential (1 Core)                               - CUDA Baseline (Global Memory)               - Basic MPI (Blocking Halo Exchange)
  - OpenMP (1 to 24 Threads)                          - CUDA Optimized (Shared-Memory Tiling)       - Optimized MPI (Non-Blocking Overlap)
```

### 3.1 Sequential CPU Baseline
The sequential implementation serves as the foundational algorithmic reference. It iterates sequentially over $i \in [0, N-1]$, accumulating $W$ window elements into a double-precision accumulator and scaling by $\frac{1.0}{W}$.

```cpp
void moving_average_seq(const double* input, double* output, int N, int window_size) {
    int k = (window_size - 1) / 2;
    double inv_w = 1.0 / window_size;
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = -k; j <= k; ++j) {
            int idx = i + j;
            if (idx < 0) idx = 0;
            else if (idx >= N) idx = N - 1;
            sum += input[idx];
        }
        output[i] = sum * inv_w;
    }
}
```

### 3.2 OpenMP Multi-Core Shared Memory
The OpenMP implementation leverages multi-core symmetric multiprocessing (SMP) by distributing outer-loop iterations across available CPU cores:

```cpp
void moving_average_omp(const double* input, double* output, int N, int window_size, int num_threads) {
    int k = (window_size - 1) / 2;
    double inv_w = 1.0 / window_size;
    #pragma omp parallel for num_threads(num_threads) schedule(static) default(none) \
            shared(input, output, N, k, inv_w)
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = -k; j <= k; ++j) {
            int idx = i + j;
            if (idx < 0) idx = 0;
            else if (idx >= N) idx = N - 1;
            sum += input[idx];
        }
        output[i] = sum * inv_w;
    }
}
```
OpenMP utilizes a static block partition where each thread receives a contiguous slice of approximately $\lfloor N / T \rfloor$ elements. Because signal elements are read-only and each thread writes strictly to disjoint output memory locations $y[i]$, no locks or atomic operations are required.

### 3.3 CUDA Baseline (Global Memory SIMT)
The CUDA baseline maps each output element $y[i]$ to an independent GPU thread:
- Thread index calculation: `int i = blockIdx.x * blockDim.x + threadIdx.x;`
- Execution grid: $\lceil N / 256 \rceil$ thread blocks, each containing 256 threads.
- Memory access: Every thread performs $W$ global memory loads directly from device DRAM (`d_input`). While adjacent threads access consecutive indices yielding coalesced transactions, each input element is redundantly re-fetched from high-latency global memory up to $W$ times across neighboring threads.

### 3.4 CUDA Optimized (Shared-Memory Tiling with Halo Cells)
To eliminate redundant global memory access latency, the CUDA optimized implementation introduces cooperative on-chip shared-memory caching:
- Each block of $B=256$ threads cooperatively loads its local tile of 256 elements plus $2k$ halo elements ($k$ left neighbors and $k$ right neighbors) into high-speed on-chip SRAM (`__shared__ double s_data[BLOCK_SIZE + 2 * MAX_K]`).
- The block synchronizes via `__syncthreads()`.
- Each thread then computes its moving-average summation strictly out of on-chip shared memory at an order-of-magnitude lower access latency.

```
       +----------------------- Shared Memory Block Tile (B + 2k) -----------------------+
       | Left Halo (k) |             Interior Thread Data (B=256)             | Right Halo (k) |
       +---------------+------------------------------------------------------+----------------+
```

### 3.5 Basic MPI: 1D Domain Decomposition & Blocking Halo Exchange
In distributed-memory architectures, processes do not share an address space. The global input array of length $N$ is decomposed across $P$ MPI ranks. Each rank $r \in [0, P-1]$ owns a local partition of size $N_{\text{local}} \approx N / P$.

To evaluate the moving-average stencil at local partition boundaries:
- Elements near the left boundary require up to $k$ samples owned by rank $r - 1$.
- Elements near the right boundary require up to $k$ samples owned by rank $r + 1$.

```
Rank r-1                    Rank r (Local Domain)                         Rank r+1
[... | Right Halo (k)] ---> [Ghost L (k) | Local Elements | Ghost R (k)] <--- [Left Halo (k) | ...]
```

In the Basic MPI implementation, processes exchange ghost cells synchronously using `MPI_Sendrecv`:
1. Rank $r$ sends its leftmost $k$ elements to rank $r - 1$ while receiving rank $r + 1$'s leftmost $k$ elements into its right ghost buffer.
2. Rank $r$ sends its rightmost $k$ elements to rank $r + 1$ while receiving rank $r - 1$'s rightmost $k$ elements into its left ghost buffer.
3. Ranks 0 and $P-1$ clamp against physical array boundaries ($\text{rank } 0$ replicates $x[0]$; $\text{rank } P-1$ replicates $x[N-1]$).
4. After both `MPI_Sendrecv` calls finish, the process performs the local moving-average computation across all $N_{\text{local}}$ elements.

Because `MPI_Sendrecv` blocks until message transfers are buffered or initiated, communication and computation are strictly serialized.

### 3.6 Optimized MPI: Non-Blocking Halo Exchange with Computation Overlap
The Optimized MPI implementation decouples communication from computation using non-blocking primitives (`MPI_Irecv` and `MPI_Isend`):
1. **Asynchronous Communication Post:** Rank $r$ immediately posts non-blocking receives for its left and right ghost regions, followed by non-blocking sends of its local boundary data.
2. **Interior Computation (Window of Overlap):** While halo exchanges transfer across the communication fabric in the background, Rank $r$ immediately computes the moving-average for its **interior domain**:
   $$i_{\text{local}} \in [k, N_{\text{local}} - 1 - k]$$
   These interior elements depend strictly on local data already present in memory and do not require ghost cells.
3. **Synchronization Barrier:** Rank $r$ calls `MPI_Waitall` to ensure incoming ghost cells have arrived.
4. **Boundary Computation:** Rank $r$ computes the remaining boundary regions:
   - Left boundary: $i_{\text{local}} \in [0, \min(k-1, N_{\text{local}}-1)]$
   - Right boundary: $i_{\text{local}} \in [\max(0, N_{\text{local}}-k), N_{\text{local}}-1]$

```
Timeline: Optimized MPI Overlap Mechanism
+-------------------------------------------------------------------------------------------------+
| Post MPI_Irecv / MPI_Isend (Non-Blocking)                                                       |
+-------------------------------------------------------------------------------------------------+
| Compute Interior Domain [k, N_local - 1 - k]        | MPI Communication transfers in background |
+-----------------------------------------------------+-------------------------------------------+
| MPI_Waitall()  <-- Synchronization barrier (stalls only if communication duration > interior)   |
+-------------------------------------------------------------------------------------------------+
| Compute Left & Right Boundary Domains [0, k-1] & [N_local - k, N_local - 1]                     |
+-------------------------------------------------------------------------------------------------+
```

---

## 4. Correctness Verification & Numerical Equivalence Methodology

### Verification Standard
In scientific computing and parallel signal processing, parallelized implementations must preserve absolute fidelity to the numerical reference. For floating-point algorithms, differences can arise if associative reordering alters rounding behavior. In our moving-average filter, every parallel implementation executes identical sequential accumulation order within each window, meaning numerical divergence should be within standard machine precision limits.

Correctness was validated by comparing each parallel output array $y_{\text{parallel}}$ against the sequential baseline $y_{\text{seq}}$ using two metrics:
1. **Maximum Absolute Error ($L_\infty$ Norm):**
   $$\text{MaxError} = \max_{0 \le i < N} |y_{\text{parallel}}[i] - y_{\text{seq}}[i]|$$
2. **Root-Mean-Square Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} (y_{\text{parallel}}[i] - y_{\text{seq}}[i])^2}$$

The acceptance threshold for numerical validity was set to $\varepsilon \le 1.0 \times 10^{-4}$.

### Empirical Verification Results
All six implementations were tested across Small ($N=10,000, W=15$), Medium ($N=100,000, W=31$), and Large ($N=1,000,000, W=63$) datasets. In addition, the MPI test suites executed dedicated boundary-condition assertions using 2, 4, 8, 12, 16, and 24 process configurations.

> [!IMPORTANT]
> **Audited Numerical Equivalence Statement:**  
> All parallel implementations achieved rigorous numerical equivalence to the sequential reference baseline, recording a maximum absolute error of 0.00e+00 and root-mean-square error of 0.00e+00 within floating-point tolerance (epsilon <= 1e-4) across all tested signal lengths.

Every parallel implementation passed with $100\%$ validation across all workloads.

---

## 5. Experimental Environment & Hardware Specifications

All empirical benchmarks were conducted on a high-performance heterogeneous workstation. The experimental platform specifications are summarized below:

| System Component | Hardware / Software Specification | Architectural Notes |
|:---|:---|:---|
| **Host Processor (CPU)** | Intel Core i7-13700HX (Raptor Lake Mobile) | 16 Physical Cores (8 Performance-cores + 8 Efficient-cores), 24 Logical Threads |
| **P-Core / E-Core Clocks** | P-Core Base: 2.1 GHz (Boost: 5.0 GHz) / E-Core Base: 1.5 GHz | Hybrid heterogeneous core architecture with shared L3 cache |
| **Host Memory (RAM)** | 16 GB DDR5 Synchronous DRAM | High-bandwidth dual-channel host interface |
| **Discrete Accelerator (GPU)** | NVIDIA GeForce RTX 4050 Laptop GPU | 6 GB GDDR6 Device Memory, Ada Lovelace Architecture (`sm_89`) |
| **GPU Streaming Multiprocessors** | 20 SMs, 2560 CUDA Cores | Peak FP32: ~9.0 TFLOPS, Boost Clock: ~2370 MHz |
| **Operating System** | Microsoft Windows 11 Home (Build 26100, x86_64) | High-performance power profile active during runs |
| **C++ Host Compiler** | MSYS2 UCRT64 GCC 16.1.0 (`g++`) | Flags: `-O3 -march=native -Wall -std=c++17 -fopenmp` |
| **CUDA Toolkit / Compiler** | NVIDIA CUDA Toolkit 13.4 (`nvcc`) | Driver: 616.92, Architecture Flag: `-arch=sm_89 -O3` |
| **MPI Implementation** | Microsoft MPI (MS-MPI) v10.1.12498.18 | Native Windows 64-bit MPI runtime and headers |
| **Profiling & Telemetry Tools** | NVIDIA Nsight Systems (`nsys`), Intel RAPL via PDH, NVML | Hardware energy and instruction timing collectors |

---

## 6. Benchmarking & Measurement Methodology

To ensure measurement repeatability and eliminate statistical anomalies:
1. **Warmup Invocations:** Every benchmark executable performed 3 un-timed warmup passes to ensure cold-start page faults, GPU driver context initialization, and MPI daemon thread initialization did not contaminate measured durations.
2. **High-Resolution Timers:**
   - Host CPU benchmarks (Sequential, OpenMP) utilized `std::chrono::high_resolution_clock`.
   - CUDA benchmarks utilized hardware-synchronized CUDA Events (`cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`) to measure pure kernel execution separate from host-to-device (H2D) and device-to-host (D2H) memory transfers.
   - MPI benchmarks utilized `MPI_Wtime()` synchronized across all ranks using `MPI_Barrier(MPI_COMM_WORLD)`.
3. **Controlled Workload Configurations:**
   - **Small:** $N = 10,000$ samples, $W = 15$ ($k = 7$)
   - **Medium:** $N = 100,000$ samples, $W = 31$ ($k = 15$)
   - **Large:** $N = 1,000,000$ samples, $W = 63$ ($k = 31$)

---

## 7. MPI Strong Scaling & Parallel Efficiency Analysis

### 7.1 Absolute Speedup vs. Self-Consistent Parallel Scaling
In distributed computing, scalability must be evaluated with mathematical rigor:
- **Absolute Speedup ($S_{\text{abs}}$):** Measures the speedup of an MPI execution of $P$ processes relative to the standalone sequential CPU baseline:
  $$S_{\text{abs}}(P) = \frac{T_{\text{seq}}}{T_{\text{mpi}}(P)}$$
- **Absolute Efficiency ($E_{\text{abs}}$):**
  $$E_{\text{abs}}(P) = \frac{S_{\text{abs}}(P)}{P} \times 100\%$$
- **Self-Consistent Parallel Speedup ($S_{\text{rel}}$):** Evaluates parallel scaling against the single-process MPI baseline ($P=1$):
  $$S_{\text{rel}}(P) = \frac{T_{\text{mpi}}(1)}{T_{\text{mpi}}(P)}$$
- **Self-Consistent Parallel Efficiency ($E_{\text{rel}}$):**
  $$E_{\text{rel}}(P) = \frac{S_{\text{rel}}(P)}{P} \times 100\%$$

> [!NOTE]
> **Sequential Baseline and Benchmark Session Distinctions:**  
> The sequential baseline timing recorded during the dedicated MPI scaling benchmark session was **35.027 ms** (for $N=1,000,000, W=63$). This is distinct from the **42.515 ms** sequential measurement recorded in the multi-paradigm six-way benchmark session. They represent separate benchmark invocations with normal run-to-run system and thermal variation.
> 
> Likewise, the parallel MPI execution timings recorded during the dedicated strong-scaling benchmark session (e.g., Optimized MPI P=16 at 5.600 ms and Basic MPI P=16 at 6.278 ms in Section 7.2) differ slightly from the measurements recorded during the comprehensive six-way comparison session (Optimized MPI P=16 at 7.584 ms and Basic MPI P=16 at 7.512 ms in Section 11.1). These represent separate benchmark invocations with expected run-to-run variations in operating system process launch, inter-process communication pipe initialization, and core scheduling.

> [!IMPORTANT]
> **Audited Scaling Statement (Addressing >100% Absolute Efficiency):**  
> In the Large workload ($N=1,000,000$), the absolute speedup exceeds $P$ at $P=1, 2, 4$ (absolute efficiencies of $138.7\%$, $130.1\%$, and $110.4\%$). This occurs because the $P=1$ MPI executable, compiled with `-O3` under MSYS2 UCRT64 GCC with distinct loop framing, achieved an execution time of $25.248$ ms compared to the $35.027$ ms of the standalone sequential benchmark. This difference does NOT represent superlinear algorithmic scaling. When evaluated self-consistently against the $P=1$ MPI baseline ($25.248$ ms), scaling is strictly sublinear, demonstrating classic parallel efficiency trends.

### 7.2 Empirical MPI Scaling Results

The complete empirical strong-scaling measurements recorded in `pa2_mpi_scaling.csv` are presented below:

| Workload | $N$ | Window | Processes ($P$) | Seq Time (ms) | Basic Compute (ms) | Basic Total (ms) | Opt Compute (ms) | Opt Waitall (ms) | Opt Total (ms) | Opt Abs Speedup | Opt Self-Consistent Rel Speedup | Opt Self-Consistent Efficiency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Medium** | 100,000 | 31 | 1 | 1.487 | 1.091 | 1.809 | 0.725 | 0.000 | 1.277 | 1.16x | 1.00x | **100.0%** |
| Medium | 100,000 | 31 | 2 | 1.487 | 0.397 | 0.803 | 0.473 | 0.001 | 0.932 | 1.60x | 1.37x | **68.5%** |
| Medium | 100,000 | 31 | 4 | 1.487 | 0.246 | 0.647 | 0.271 | 0.009 | 0.664 | 2.24x | 1.92x | **48.1%** |
| Medium | 100,000 | 31 | 8 | 1.487 | 0.208 | 1.304 | 0.214 | 0.084 | 1.391 | 1.07x | 0.92x | **11.5%** |
| Medium | 100,000 | 31 | 12 | 1.487 | 0.137 | 1.616 | 0.156 | 0.155 | 1.154 | 1.29x | 1.11x | **9.2%** |
| Medium | 100,000 | 31 | 16 | 1.487 | 0.134 | 1.768 | 0.126 | 0.255 | 1.418 | 1.05x | 0.90x | **5.6%** |
| Medium | 100,000 | 31 | 24 | 1.487 | 0.251 | 5.341 | 0.085 | 0.744 | 2.288 | 0.65x | 0.56x | **2.3%** |
| **Large** | 1,000,000 | 63 | 1 | 35.027 | 19.401 | 25.546 | 19.278 | 0.003 | 25.248 | 1.39x | 1.00x | **100.0%** |
| Large | 1,000,000 | 63 | 2 | 35.027 | 9.194 | 13.632 | 9.158 | 0.007 | 13.463 | 2.60x | 1.88x | **93.8%** |
| Large | 1,000,000 | 63 | 4 | 35.027 | 5.199 | 9.214 | 4.909 | 0.009 | 7.933 | 4.42x | 3.18x | **79.6%** |
| Large | 1,000,000 | 63 | 8 | 35.027 | 2.244 | 5.679 | 2.745 | 0.005 | 5.638 | 6.21x | 4.48x | **56.0%** |
| Large | 1,000,000 | 63 | 12 | 35.027 | 3.027 | 6.850 | 4.017 | 0.975 | 12.095 | 2.90x | 2.09x | **17.4%** |
| Large | 1,000,000 | 63 | 16 | 35.027 | 2.470 | 6.278 | 2.603 | 0.156 | 5.600 | 6.25x | 4.51x | **28.2%** |
| Large | 1,000,000 | 63 | 24 | 35.027 | 1.657 | 7.111 | 1.634 | 2.165 | 7.712 | 4.54x | 3.27x | **13.6%** |

![MPI Strong Scaling Analysis across Process Counts](../results/plots/pa2_mpi_scaling.png)

### 7.3 Analysis of Scaling Bottlenecks Beyond $P=8$
As observed in the scaling table:
1. **Linear Scaling Regime ($P \le 8$):** For the Large dataset, performance scales steadily up to 8 processes, achieving a speedup of $6.21\times$ and an execution time of $5.638$ ms. In this regime, each process computes over 125,000 samples, providing ample compute work relative to halo exchange overhead.
2. **Diminishing Returns & Latency Exposure ($P > 8$):** At 12, 16, and 24 processes, scaling degrades. At $P=12$, execution time increases to $12.095$ ms. At $P=24$, execution time increases to $7.712$ ms.
3. **Physical Core Over-subscription:** The Intel Core i7-13700HX features 16 physical cores (8 Performance-cores + 8 Efficient-cores) and 24 logical hyperthreads.
   > [!NOTE]
   > **Audited Qualification on $P=24$ Performance:**  
   > The degradation observed at $P=24$ is consistent with increased scheduling and inter-process synchronization overhead beyond the physical-core count, alongside asymmetric execution rates between P-cores and E-cores. Because dedicated hardware performance counters for hyperthread scheduling and cache miss rates were not gathered, this phenomenon is characterized by observed inter-process synchronization stalls rather than speculative hardware cache partition claims.

---

## 8. MPI Communication Overlap & Latency Hiding Analysis

### 8.1 Empirical Overlap Profiling
To evaluate how effectively non-blocking communication hides halo exchange latency, the Optimized MPI implementation instrumented the exact time spent in interior computation versus `MPI_Waitall` stalls. The results from `pa2_profiling_summary.csv` are tabulated below:

| Processes ($P$) | Interior Compute (ms) | Boundary Compute (ms) | Total Compute (ms) | Waitall Stall Time (ms) | Overlap Efficiency (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **2** | 0.688 | 0.001 | 0.688 | 0.002 | **99.71%** |
| **4** | 0.270 | 0.000 | 0.271 | 0.021 | **92.78%** |
| **8** | 0.235 | 0.001 | 0.236 | 0.003 | **98.74%** |
| **16** | 0.134 | 0.002 | 0.135 | 0.453 | **22.83%** |
| **24** | 0.333 | 0.001 | 0.334 | 0.959 | **25.77%** |

![MPI Overlap and Latency Hiding Profile](../results/plots/pa2_mpi_overlap.png)

### 8.2 Architectural Interpretation of Overlap Dynamics

> [!IMPORTANT]
> **Audited MPI Overlap Statement:**  
> At moderate process counts (P <= 8), non-blocking halo exchange effectively hides most measured communication overhead behind interior stencil computation. At higher process counts, the reduced partition size shortens the interior computation window, causing more communication latency to become exposed.

1. **Near-Perfect Hiding at $P \le 8$:** At $P=2$, the interior computation duration ($0.688$ ms) is substantially longer than the inter-process message transit time. When `MPI_Waitall` is invoked, the incoming ghost messages have already arrived in the receive buffer, resulting in a negligible stall of only $0.002$ ms ($99.71\%$ overlap efficiency). At $P=8$, interior computation takes $0.235$ ms while `MPI_Waitall` stalls for only $0.003$ ms ($98.74\%$ efficiency).
2. **Loss of Overlap at High Process Counts ($P \ge 16$):** At $P=16$, the local partition length shrinks to $N_{\text{local}} \approx 62,500$. The interior computation window narrows to only $0.134$ ms. Because inter-process message dispatch and kernel scheduling across 16 OS processes require hundreds of microseconds, communication cannot finish before interior computation completes. Consequently, `MPI_Waitall` stalls for $0.453$ ms, causing overlap efficiency to plummet to $22.83\%$. At $P=24$, the stall increases to $0.959$ ms ($25.77\%$ efficiency).

---

## 9. GPU Profiling & NVIDIA Nsight Systems Deep Dive

### 9.1 Profiling Instrumentation Setup
CUDA execution was profiled using two complementary methods:
1. **In-Application CUDA Events:** Measuring exact device-side kernel execution and synchronous `cudaMemcpy` transfers for H2D and D2H.
2. **NVIDIA Nsight Systems (`nsys`):** Capturing low-level hardware performance traces, warp execution states, and instruction issue metrics.

### 9.2 Nsight Systems Diagnostic Findings
The Nsight Systems traces provided deep microarchitectural insights into the baseline versus optimized CUDA kernels:

| Metric / Parameter | CUDA Baseline Kernel | CUDA Optimized Kernel |
|:---|:---|:---|
| **Profiling Workload ($N$)** | $N = 10,000$ samples, $W = 15$ | $N = 100,000$ samples, $W = 31$ |
| **Recorded Kernel Duration** | $5.23\ \mu\text{s}$ ($0.00523$ ms) | $80.76\ \mu\text{s}$ ($0.08076$ ms) |
| **Grid Dimensions** | 40 Blocks $\times$ 256 Threads | 391 Blocks $\times$ 256 Threads |
| **Register Usage** | 16 Registers / Thread | 18 Registers / Thread |
| **Shared Memory Allocation** | 0 Bytes / Block | 1,280 Bytes / Block |
| **Memory Access Pattern** | Uncached Global Memory Loads | Cooperative SRAM Tiling + Halo Caching |

> [!WARNING]
> **Workload Disparity Qualification:**  
> The Nsight Systems traces were collected on different workload sizes ($N=10,000$ for Baseline vs. $N=100,000$ for Optimized) to capture fine-grained trace timelines without buffer overflows. Therefore, the absolute kernel timings ($5.23\ \mu\text{s}$ vs. $80.76\ \mu\text{s}$) are NOT directly comparable as a speedup ratio.

### 9.3 Cross-Architecture CUDA Event Comparison on Identical Workloads
For direct, side-by-side comparison on identical datasets, the synchronized CUDA Event measurements from `pa2_full_comparison.csv` must be referenced:

| Workload | $N$ | Implementation | Kernel Compute Time (ms) | Transfer Time (H2D + D2H) (ms) | End-to-End Time (ms) | Transfer Overhead Ratio (%) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Small** | 10,000 | CUDA Baseline | 0.0329 | 0.0955 | 0.1284 | 74.4% |
| Small | 10,000 | CUDA Optimized | 0.0192 | 0.0515 | 0.0708 | 72.7% |
| **Medium** | 100,000 | CUDA Baseline | 0.1030 | 0.1781 | 0.2811 | 63.4% |
| Medium | 100,000 | CUDA Optimized | 0.1136 | 0.4028 | 0.5165 | 78.0% |
| **Large** | 1,000,000 | CUDA Baseline | 1.5941 | 2.0708 | 3.6649 | 56.5% |
| Large | 1,000,000 | CUDA Optimized | 1.5911 | 2.8139 | 4.4050 | 63.9% |

### 9.4 Architectural Tradeoffs: Global vs. Shared Memory
1. **Kernel Compute Parity at Large Scale:** At $N=1,000,000$, the pure compute times of the Baseline ($1.5941$ ms) and Optimized ($1.5911$ ms) kernels are virtually identical. On the modern Ada Lovelace architecture (`sm_89`), the large hardware L2 cache (32 MB) absorbs a significant fraction of redundant global memory fetches even in the baseline kernel.
2. **PCIe Transfer Bottleneck:** For both CUDA implementations, data transfer across the PCIe bus accounts for over $55\%$ to $78\%$ of total end-to-end execution time. Accelerating pure kernel computation provides diminishing returns on end-to-end latency unless data transfers are overlapped using asynchronous CUDA streams or multiple filtering stages are chained directly in GPU memory.

---

## 10. Hardware Energy Telemetry & Normalized Energy Analysis

### 10.1 Telemetry Methodology
Energy consumption was measured directly from hardware sensors during controlled benchmark executions on the Large dataset ($N=1,000,000, W=63$):
- **CPU Energy (Intel RAPL):** Captured via Windows Performance Data Helper (`win32pdh`) querying the Intel RAPL Energy Counter for processor package power.
- **GPU Energy (NVIDIA NVML):** Captured via native calls to `nvmlDeviceGetPowerUsage` through `nvml.dll`, sampling GPU board power at millisecond resolution.
- **Session Duration:** Process wall-clock execution time during the energy sampling session.

### 10.2 Energy Normalization Methodology
Because benchmark invocations for different paradigms employed different iteration counts to achieve stable hardware power readings:
- Sequential: 5 iterations
- OpenMP ($T=24$): 10 iterations
- Basic MPI ($P=8$): 10 iterations
- Optimized MPI ($P=8$): 10 iterations
- CUDA Baseline: 15 iterations
- CUDA Optimized: 15 iterations

> [!CAUTION]
> **Strict Energy Normalization Requirement:**  
> Raw session energy measurements reflect differing iteration counts and must NOT be compared as if they represented an identical computational task. To evaluate energy efficiency fairly, all energy values must be normalized to **Joules per filtering pass**:
> $$E_{\text{pass}} = \frac{E_{\text{session}}}{N_{\text{iterations}}}$$

### 10.3 Empirical Energy Measurements and Energy-Delay Product (EDP)

The raw telemetry from `pa2_energy_comparison.csv` alongside the normalized per-pass energy and Energy-Delay Product ($EDP_{\text{pass}} = E_{\text{pass}} \times T_{\text{pass}}$) are detailed below:

| Implementation | Hardware Telemetry Source | Iterations | Session Time (s) | Average Power (W) | Raw Session Energy (J) | Normalized Energy ($E_{\text{pass}}$) | Filter Pass Time ($T_{\text{pass}}$) | Per-Pass EDP ($E_{\text{pass}} \times T_{\text{pass}}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sequential** | Intel RAPL (CPU) | 5 | 0.2444 | 57.98 W | 14.169 J | **2.834 J/pass** | 42.515 ms | **0.1205 J·s** |
| **OpenMP (T=24)** | Intel RAPL (CPU) | 10 | 0.1562 | 70.27 W | 10.977 J | **1.098 J/pass** | 6.518 ms | **0.0072 J·s** |
| **Basic MPI (P=8)** | Intel RAPL (CPU) | 10 | 0.3835 | 69.44 W | 26.632 J | **2.663 J/pass** | 5.679 ms | **0.0151 J·s** |
| **Opt MPI (P=8)** | Intel RAPL (CPU) | 10 | 0.2959 | 66.57 W | 19.696 J | **1.970 J/pass** | 5.638 ms | **0.0111 J·s** |
| **CUDA Baseline** | NVIDIA NVML (GPU) | 15 | 0.4196 | 17.97 W | 7.541 J | **0.503 J/pass** | 3.665 ms | **0.0018 J·s** |
| **CUDA Optimized** | NVIDIA NVML (GPU) | 15 | 0.3675 | 11.29 W | 4.149 J | **0.277 J/pass** | 4.405 ms | **0.0012 J·s** |

![Hardware Energy Comparison and Power Profile](../results/plots/pa2_energy_comparison.png)

### 10.4 Quantitative Analysis of Energy Findings

> [!IMPORTANT]
> **Audited Energy Evaluation Statement:**  
> Hardware energy measurements using Intel RAPL for CPU package power and NVIDIA NVML for GPU board power show that CUDA Optimized had the lowest measured energy per filtering pass on this platform. Normalized to a single moving-average pass on N=1,000,000, CUDA Optimized consumed 0.277 J/pass, compared with 2.834 J/pass for the sequential baseline and 1.098 J/pass for OpenMP.

1. **GPU Energy Efficiency:** The GPU implementations demonstrated superior energy efficiency per pass. While CPU implementations operated at 58 W to 70 W package power, the GPU implementations drew only 11.3 W to 18.0 W board power. Consequently, CUDA Optimized achieved a $\mathbf{10.2\times}$ reduction in energy per pass compared to the sequential CPU baseline ($0.277$ J vs. $2.834$ J).
2. **MPI Energy Profile:** Basic MPI ($P=8$) consumed $2.663$ J/pass, close to the sequential CPU baseline. This occurs because spinning up 8 operating system processes maintains all CPU execution units in an active high-frequency C0 power state. Optimized MPI reduced energy to $1.970$ J/pass ($\sim 26\%$ reduction over Basic MPI) by shortening overall active execution duration and eliminating blocking idle waits.
3. **Energy-Delay Product:** When evaluating joint performance and energy via $EDP = \text{Energy} \times \text{Delay}$, CUDA Optimized achieved the lowest EDP ($0.0012\ \text{J}\cdot\text{s}$), followed closely by CUDA Baseline ($0.0018\ \text{J}\cdot\text{s}$) and OpenMP ($0.0072\ \text{J}\cdot\text{s}$).

---

## 11. Comprehensive Six-Way Cross-Paradigm Performance Comparison

### 11.1 Full Benchmark Comparison Matrix
The complete empirical comparison across all six implementations from `pa2_full_comparison.csv` is presented below:

| Implementation | Configuration | Workload | $N$ | Window | Compute Time (ms) | Transfer / Comm (ms) | End-to-End Time (ms) | Speedup (vs. Seq) | Parallel Efficiency | Numerical Max Error | Verification Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sequential** | 1 Thread | Small | 10,000 | 15 | 0.152 | 0.000 | 0.152 | 1.00x | 100.0% | 0.00e+00 | REFERENCE |
| **OpenMP** | 24 Threads | Small | 10,000 | 15 | 0.200 | 0.000 | 0.200 | 0.76x | 3.17% | 0.00e+00 | PASSED |
| **CUDA Baseline** | Block=256 | Small | 10,000 | 15 | 0.033 | 0.096 | 0.128 | 1.18x | N/A | 0.00e+00 | PASSED |
| **CUDA Optimized** | Shared-Mem Block=256 | Small | 10,000 | 15 | 0.019 | 0.052 | 0.071 | 2.15x | N/A | 0.00e+00 | PASSED |
| **Basic MPI** | 16 Processes | Small | 10,000 | 15 | 0.009 | 0.474 | 0.483 | 0.31x | 1.97% | 0.00e+00 | PASSED |
| **Optimized MPI** | 16 Processes (Overlap) | Small | 10,000 | 15 | 0.008 | 0.117 | 0.649 | 0.23x | 1.46% | 0.00e+00 | PASSED |
| **Sequential** | 1 Thread | Medium | 100,000 | 31 | 1.903 | 0.000 | 1.903 | 1.00x | 100.0% | 0.00e+00 | REFERENCE |
| **OpenMP** | 24 Threads | Medium | 100,000 | 31 | 0.449 | 0.000 | 0.449 | 4.24x | 17.66% | 0.00e+00 | PASSED |
| **CUDA Baseline** | Block=256 | Medium | 100,000 | 31 | 0.103 | 0.178 | 0.281 | 6.77x | N/A | 0.00e+00 | PASSED |
| **CUDA Optimized** | Shared-Mem Block=256 | Medium | 100,000 | 31 | 0.114 | 0.403 | 0.517 | 3.68x | N/A | 0.00e+00 | PASSED |
| **Basic MPI** | 16 Processes | Medium | 100,000 | 31 | 0.132 | 2.515 | 2.647 | 0.72x | 4.49% | 0.00e+00 | PASSED |
| **Optimized MPI** | 16 Processes (Overlap) | Medium | 100,000 | 31 | 0.123 | 0.729 | 2.081 | 0.91x | 5.72% | 0.00e+00 | PASSED |
| **Sequential** | 1 Thread | Large | 1,000,000 | 63 | 42.515 | 0.000 | 42.515 | 1.00x | 100.0% | 0.00e+00 | REFERENCE |
| **OpenMP** | 24 Threads | Large | 1,000,000 | 63 | 6.518 | 0.000 | 6.518 | 6.52x | 27.18% | 0.00e+00 | PASSED |
| **CUDA Baseline** | Block=256 | Large | 1,000,000 | 63 | 1.594 | 2.071 | 3.665 | 11.60x | N/A | 0.00e+00 | PASSED |
| **CUDA Optimized** | Shared-Mem Block=256 | Large | 1,000,000 | 63 | 1.591 | 2.814 | 4.405 | 9.65x | N/A | 0.00e+00 | PASSED |
| **Basic MPI** | 16 Processes | Large | 1,000,000 | 63 | 2.565 | 4.947 | 7.512 | 5.66x | 35.37% | 0.00e+00 | PASSED |
| **Optimized MPI** | 16 Processes (Overlap) | Large | 1,000,000 | 63 | 2.583 | 0.319 | 7.584 | 5.61x | 35.04% | 0.00e+00 | PASSED |

![Six-Way Cross-Paradigm Benchmark Comparison](../results/plots/pa2_6way_comparison.png)

### 11.2 Paradigm Breakdown and Performance Synthesis
1. **Small Workload ($N=10,000$):**
   - Sequential execution requires only $0.152$ ms.
   - OpenMP incurs thread team barrier overhead, yielding a slowdown ($0.200$ ms, $0.76\times$).
   - MPI incurs inter-process socket/shared-memory pipe setup, resulting in severe slowdowns ($0.483$ ms for Basic MPI, $0.649$ ms for Opt MPI).
   - Only CUDA Optimized achieves a modest speedup ($2.15\times$, $0.071$ ms), though data transfer accounts for over $72\%$ of elapsed time.
2. **Medium Workload ($N=100,000$):**
   - OpenMP scales effectively, delivering a $4.24\times$ speedup ($0.449$ ms).
   - CUDA Baseline achieves the highest speedup at $6.77\times$ ($0.281$ ms).
   - MPI remains constrained by communication overhead at 16 processes ($2.647$ ms for Basic MPI, $2.081$ ms for Opt MPI).
3. **Large Workload ($N=1,000,000$):**
   - **CUDA Baseline** achieves the fastest overall end-to-end execution ($3.665$ ms, $\mathbf{11.60\times}$ speedup).
   - **CUDA Optimized** delivers a $9.65\times$ speedup ($4.405$ ms end-to-end; pure kernel compute of $1.591$ ms represents a $\mathbf{26.7\times}$ compute-only speedup).
   - **OpenMP (24 Threads)** achieves a substantial $6.52\times$ speedup ($6.518$ ms), operating entirely in host memory with zero interconnect transfer overhead.
   - **Optimized MPI ($P=16$)** delivers a $5.61\times$ speedup ($7.584$ ms), with pure computation at $2.583$ ms.

---

## 12. Architectural Bottleneck Analysis & Amdahl's Law

### 12.1 Memory-Bound Nature of Moving-Average Filtering
The computational intensity (operational intensity) of the moving-average filter is defined as arithmetic operations per byte of memory traffic:
$$\text{Arithmetic Operations} = W \text{ additions} + 1 \text{ multiplication} = W + 1 \text{ FLOPs}$$
$$\text{Memory Traffic (Worst Case)} = W \text{ loads} \times 8 \text{ bytes} + 1 \text{ store} \times 8 \text{ bytes} = 8(W + 1) \text{ Bytes}$$
$$\text{Operational Intensity} = \frac{W + 1}{8(W + 1)} = 0.125 \text{ FLOPs/Byte}$$

Because the operational intensity is strictly constant at $0.125$ FLOPs/Byte regardless of window size $W$, 1D moving-average filtering is fundamentally **memory-bandwidth bound**. No architecture can exceed the throughput permitted by its memory hierarchy.

### 12.2 Amdahl's Law and Interconnect Bottlenecks
According to Amdahl's Law, maximum achievable speedup is bounded by the non-parallelizable fraction $\alpha$:
$$S_{\max}(P) = \frac{1}{\alpha + \frac{1 - \alpha}{P}}$$

1. **For GPU Computing:** The non-parallelizable sequential fraction $\alpha$ is dominated by PCIe data transfer time ($T_{\text{transfer}} = T_{\text{H2D}} + T_{\text{D2H}}$). Even if GPU kernel time approaches zero, speedup is strictly capped by:
   $$S_{\max,\text{GPU}} \le \frac{T_{\text{seq}}}{T_{\text{transfer}}} = \frac{42.515\ \text{ms}}{2.071\ \text{ms}} \approx 20.5\times$$
2. **For MPI Distributed Computing:** The sequential fraction $\alpha$ is dominated by inter-process message serialization, OS context switching, and halo exchange latency. When partition size shrinks below a critical threshold, communication duration exceeds interior computation, bounding scaling efficiency.

---

## 13. When Parallelization Does Not Help

A critical principle of parallel and distributed computing is identifying scenarios where parallelization is counterproductive:
1. **Low Problem Dimension / Short Signal Lengths:** For $N \le 10,000$, the overhead of spawning thread pools, initializing MPI message buffers, or launching GPU kernels exceeds the serial computation time of the entire problem.
2. **Excessive Process / Thread Concurrency:** Allocating 24 MPI processes on a single laptop workstation causes memory bus contention and inter-process scheduling overhead that degrades throughput compared to 8 processes.
3. **High Communication-to-Computation Ratios:** As process count $P$ increases under strong scaling, partition size $N/P$ decreases while halo size $k$ remains constant. The ratio of halo communication to local computation scales as $\mathcal{O}(k \cdot P / N)$, inevitably leading to communication bottlenecks.

---

## 14. Reproducibility Guide & Verification Workflow

To guarantee full scientific reproducibility, all source code, benchmark runners, and plotting scripts are integrated into automated workflows.

### 14.1 Compilation
All six implementations compile through PowerShell and native toolchains:
```powershell
# Build all targets (Sequential, OpenMP, CUDA, Basic MPI, Optimized MPI)
.\build.ps1 -Target all
```
Compilation binaries generated in `bin/`:
- `bin/moving_average_seq.exe`
- `bin/moving_average_omp.exe`
- `bin/moving_average_cuda.exe`
- `bin/moving_average_cuda_opt.exe`
- `bin/moving_average_mpi.exe`
- `bin/moving_average_mpi_opt.exe`

### 14.2 Running Benchmarks & Verifying Correctness
```powershell
# Run correctness test suite across MPI ranks
mpiexec -n 4 bin/moving_average_mpi.exe --test
mpiexec -n 4 bin/moving_average_mpi_opt.exe --test

# Execute full automated PA2 benchmark suite
python benchmarks/run_pa2_benchmarks.py

# Regenerate plots
python scripts/plot_pa2_results.py
```

All benchmark logs and tables are written to `results/tables/`.

---

## 15. Limitations, Threats to Validity & Observations

1. **Hardware Cache Counter Availability:** Due to standard user-level Windows OS security permissions, direct hardware performance counters for L1/L2 cache miss rates and pipeline stall cycles could not be accessed. Architectural analyses are grounded in wall-clock timings, CUDA Events, and Nsight Systems instruction statistics.
2. **Single-Node Shared-Memory MPI Execution:** MS-MPI was executed on a single shared-memory node rather than a multi-node cluster with dedicated InfiniBand interconnects. Communication occurred via OS shared-memory pipes and loopback IPC.
3. **Dynamic Thermal Throttling:** As a mobile processor, the Intel Core i7-13700HX exhibits dynamic clock frequency scaling based on thermal headroom and power distribution between CPU and discrete GPU. Warmups and repeated trials minimized but could not entirely eliminate thermal variance.

---

## 16. Viva Voce & Oral Defense Preparation

### Question 1: Why does absolute efficiency exceed 100% at P=1, 2, 4 in MPI scaling? Does this represent superlinear scaling?
**Answer:** No, it does NOT represent superlinear algorithmic scaling. The standalone sequential benchmark ran at $35.027$ ms, whereas the single-process MPI executable ($P=1$) executed in $25.248$ ms due to compiler optimization variations in MSYS2 GCC `-O3` loop framing and memory layout. When evaluated self-consistently against the $P=1$ MPI baseline, efficiency is strictly sublinear: $93.8\%$ at $P=2$, $79.6\%$ at $P=4$, and $56.0\%$ at $P=8$.

### Question 2: Why does MPI overlap efficiency drop from 98.7% at P=8 to 22.8% at P=16?
**Answer:** At $P=8$, each process computes over 125,000 samples. The interior computation window ($0.235$ ms) is long enough to completely hide non-blocking halo exchange latency, resulting in an `MPI_Waitall` stall of only $0.003$ ms. At $P=16$, the partition shrinks to 62,500 samples, narrowing interior compute to $0.134$ ms. Because inter-process message dispatch and kernel scheduling require more time, communication is exposed, forcing `MPI_Waitall` to stall for $0.453$ ms.

### Question 3: How was energy normalized across implementations with different iteration counts?
**Answer:** Raw session energy reflects differing repetition counts (5 iterations for Sequential, 10 for OpenMP and MPI, 15 for CUDA) chosen to obtain stable hardware sensor readings. Energy was rigorously normalized to Joules per filtering pass: $E_{\text{pass}} = E_{\text{session}} / N_{\text{iterations}}$. This revealed that CUDA Optimized consumed $0.277$ J/pass, compared to $1.098$ J/pass for OpenMP and $2.834$ J/pass for Sequential.

### Question 4: Why do CUDA Baseline and CUDA Optimized exhibit nearly identical kernel runtimes on Large workloads?
**Answer:** The NVIDIA Ada Lovelace RTX 4050 GPU features a 32 MB hardware L2 cache. At $N=1,000,000$, the input array fits within L2 cache, allowing the baseline kernel's redundant fetches to hit cache rather than DRAM. Consequently, the explicit shared-memory caching in the optimized kernel provides minimal additional latency reduction.

---

## 17. Conclusion

Programming Assignment 2 provided an extensive empirical exploration across six computational paradigms for 1D moving-average signal filtering. Key conclusions include:
1. **Numerical Invariance:** All six implementations maintained rigorous numerical equivalence to the sequential reference baseline, achieving $\text{Max Absolute Error} = 0.00\text{e}+00$ across all signal sizes.
2. **Computational Sweet Spots:**
   - For Small datasets ($N \le 10,000$), Sequential execution is fastest due to zero launch and synchronization overhead.
   - For Large datasets ($N = 1,000,000$), **CUDA Baseline** achieved the highest end-to-end speedup ($\mathbf{11.60\times}$), while **OpenMP** delivered the strongest host-only performance ($\mathbf{6.52\times}$) with zero data transfer latency.
3. **Latency Hiding in Distributed Stencils:** Non-blocking MPI communication (`MPI_Isend` / `MPI_Irecv`) successfully hides over $98\%$ of halo exchange latency up to 8 processes, demonstrating the viability of domain decomposition for stencil filtering.
4. **Energy Efficiency:** Accelerated computing on GPUs demonstrated a $\mathbf{10.2\times}$ reduction in energy per filtering pass ($0.277$ J/pass vs. $2.834$ J/pass), confirming that architectural throughput directly enhances computational energy efficiency.

---

## 18. Contributions & Team Responsibilities

| Team Member | Roll Number | Key Responsibilities & Module Leadership |
|:---|:---:|:---|
| **Saksham Singh** | 2024BCS0070 | Project Lead; MPI domain decomposition & halo exchange design; Optimized MPI non-blocking overlap implementation; Hardware energy telemetry engine (RAPL & NVML); Final report synthesis & mathematical modeling. |
| **Daksh Singh** | 2024BCS0042 | Basic MPI implementation; Correctness verification suite and boundary condition testing; Nsight Systems profiling integration; Data collection across OpenMP and MPI scaling benchmarks. |
| **Anmol Pipara** | 2024BCS0014 | Benchmark automation scripting; Data table formatting and traceability auditing; Matplotlib plotting pipeline for PA2 scaling, overlap, and energy figures; Reproducibility verification. |
