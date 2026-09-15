# CSS311: Parallel & Distributed Computing — Programming Assignment 1
## Moving-Average Noise Reduction (Signal Processing)

This project investigates and benchmarks the computational performance and scalability of a 1D Moving-Average Noise Reduction filter across sequential and parallel execution paradigms:
1. **Sequential Baseline (C++)**
2. **Multi-core Shared Memory Parallelism (OpenMP)**
3. **Massively Parallel Many-core GPU (CUDA)**
4. **Optimized Parallel Implementation (Shared Memory Tiling / Algorithmic Optimizations)**

---

## Project Directory Structure

```
Signal Processing/
├── README.md               # Overview, execution instructions, and architecture
├── build.ps1               # Build automation script (MSYS2 g++ with OpenMP)
├── include/                # Header files shared across all implementations
│   ├── common.hpp          # Data types, benchmark metrics structure, high-res timer
│   ├── validator.hpp       # Numerical correctness validator (max error, RMSE, tolerance)
│   └── signal_io.hpp       # Deterministic input signal generator & binary/CSV I/O
├── src/
│   ├── sequential/         # Baseline sequential C++ implementation
│   ├── openmp/             # OpenMP parallel implementation
│   ├── cuda/               # CUDA GPU implementation
│   └── optimized/          # Optimized CUDA/OpenMP implementation
├── data/
│   ├── input/              # Test signals across Small, Medium, Large sizes
│   └── output/             # Output filtered signals for correctness validation
├── benchmarks/             # Benchmark scripts, test parameters, and run configs
├── results/
│   ├── tables/             # Performance tables (execution time, speedup, efficiency)
│   └── plots/              # Visualizations (speedup curves, efficiency, signal plots)
├── scripts/                # Python scripts for synthetic signal generation and plotting
├── docs/                   # Algorithmic notes, boundary analysis, design docs
└── report/                 # Final assignment report and documentation
```

---

## Environment Profile Summary

- **Operating System:** Windows 11 Home Single Language (64-bit, Build 26200)
- **CPU:** 13th Gen Intel(R) Core(TM) i7-13700HX (16 Cores: 8 P-Cores + 8 E-Cores, 24 Threads)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM, Driver 551.76, Compute Capability 8.9)
- **Host Compiler:** GCC / G++ 16.1.0 (MSYS2 UCRT64) with native `-fopenmp` support
- **Python:** Python 3.13.5 (Anaconda environment with NumPy, SciPy, Matplotlib)
- **CUDA Toolkit (NVCC):** Requires installation before Step 3 (GPU driver with CUDA 12.4 support is active)

---

---

## Sequential Baseline (Step 2)

### 1. Purpose & Overview
The sequential C++ implementation ([`src/sequential/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/sequential)) serves as the **unoptimized computational reference baseline**. All subsequent parallel implementations (OpenMP, CUDA, Optimized) will be strictly validated against the output of this implementation.

### 2. Mathematical Definition
For an input 1D signal $x[0], x[1], \dots, x[N-1]$ and an odd window size $W = 2k + 1$ (where $k = \frac{W - 1}{2}$ is the filter radius):

$$y[i] = \frac{1}{W} \sum_{j = -k}^{k} x[\text{boundary}(i + j)]$$

### 3. Edge-Replication Boundary Handling
Boundary conditions are handled via **edge replication (clamping)**:

$$\text{boundary}(p) = \min(\max(p, 0), N - 1)$$

For any sample where index $i+j < 0$, it is clamped to $0$ ($x[0]$). For any sample where index $i+j \ge N$, it is clamped to $N - 1$ ($x[N-1]$). This guarantees that every output sample averages exactly $W$ elements with a constant divisor $W$, eliminating thread divergence in parallel implementations.

### 4. Algorithmic Complexity
- **Time Complexity:** $\mathcal{O}(N \times W)$ arithmetic operations. For each of the $N$ outputs, the inner loop iterates $W$ times, performing 1 addition and clamped read, followed by 1 division.
- **Space Complexity:** $\mathcal{O}(N)$ output storage. The filter operates in-place into the output buffer with $\mathcal{O}(1)$ auxiliary memory.
- **Precision:** Accumulation uses 64-bit `double` precision to eliminate summation drift across large window sizes before storing into single-precision 32-bit `SampleType` (`float`).

### 5. Why Deliberately Naive & Unoptimized
In parallel and distributed computing, an honest and academically rigorous study requires comparing parallel implementations against the **standard naive algorithm** rather than an algorithmically altered version (such as sliding-window or prefix-sum). Sliding-window introduces serial loop-carried dependencies ($y[i]$ depends on $y[i-1]$), whereas the naive $\mathcal{O}(NW)$ stencil reflects the exact independent computation structure mapped onto OpenMP threads and GPU warps.

### 6. How to Build
```powershell
# Build sequential baseline using the automated PowerShell build script
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target seq
```

### 7. How to Run
```powershell
# Run with default settings (N = 100,000, W = 15, Seed = 42)
.\bin\moving_average_seq.exe

# Run with custom parameters
.\bin\moving_average_seq.exe -n 1000000 -w 31 -s 42

# Run and save filtered output to binary file
.\bin\moving_average_seq.exe -n 100000 -w 15 --save data/output/reference_N100000_W15.bin

# Run built-in correctness verification test suite
.\bin\moving_average_seq.exe --test
```

---

## OpenMP Parallel Implementation (Step 3)

### 1. Parallelization Strategy
The OpenMP parallel filter ([`src/openmp/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/openmp)) parallelizes the outer loop over output indices $i \in [0, N-1]$ using:

```cpp
#pragma omp parallel for num_threads(threadsToUse) schedule(static) default(none) \
    shared(input, output, n, windowSize, k, invW)
```

### 2. Data Decomposition & Race-Free Properties
- **Output Domain Decomposition:** The signal output domain $[0, N-1]$ is partitioned into contiguous blocks assigned to each thread.
- **Concurrent Safe Reads:** The `input` array is strictly read-only during filtering. Multiple threads safely read overlapping window samples simultaneously without conflicts.
- **Mutually Exclusive Writes:** Iteration $i$ writes strictly to `output[i]`. Because each index $i$ is uniquely assigned to one thread, there are **zero write-write conflicts**.
- **Thread-Private Accumulation:** The accumulator `double sum = 0.0;` is declared inside the iteration scope, ensuring it is allocated on the private stack of each thread.
- **No Atomic/Mutex Overhead:** The algorithm requires no mutexes, locks, or critical sections.
- **Synchronization:** An implicit barrier at the end of `#pragma omp parallel for` guarantees all threads complete computation before returning.

### 3. Scheduling Strategy: `schedule(static)`
`schedule(static)` partitions the $N$ outputs into equal chunks of size $\lceil N / T \rceil$ at launch time. Because our edge-clamping boundary strategy ensures that **every single output sample computes the exact same arithmetic work ($W$ reads, $W-1$ additions, 1 multiplication)**, the workload is completely homogeneous. Static scheduling eliminates runtime scheduling overhead while maximizing CPU cache locality.

### 4. How to Build
```powershell
# Build OpenMP implementation (and sequential reference)
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target omp

# Or build all targets
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target all
```

### 5. How to Run
```powershell
# Run OpenMP with 8 threads on N = 100,000, W = 15
.\bin\moving_average_omp.exe -n 100000 -w 15 -t 8

# Run built-in correctness verification test suite
.\bin\moving_average_omp.exe --test

# Run full thread-scaling benchmark (N = 1,000,000, W = 31, T = 1..24)
.\bin\moving_average_omp.exe --benchmark
```

### 6. Measured Thread Scaling Results ($N = 1,000,000, W = 31$)

*Tested on 13th Gen Intel Core i7-13700HX (16 Cores: 8 P-cores + 8 E-cores, 24 Threads), 5 runs per configuration (1 warm-up discarded + 4 averaged). Reference sequential baseline measured on identical workload: **21.622 ms**.*

| Threads ($T$) | Average Time (ms) | Speedup ($S$) | Parallel Efficiency ($E$) | Validation Status |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 19.467 ms | 1.11x | 111.1% | PASSED |
| **2** | 14.496 ms | 1.49x | 74.6% | PASSED |
| **4** | 11.746 ms | 1.84x | 46.0% | PASSED |
| **8** | **11.262 ms** | **1.92x** | 24.0% | PASSED |
| **16** | 12.144 ms | 1.78x | 11.1% | PASSED |
| **24** | 14.283 ms | 1.51x | 6.3% | PASSED |

*Raw data automatically logged to [`results/tables/omp_thread_scaling.csv`](file:///c:/Users/Predator/Desktop/Signal%20Processing/results/tables/omp_thread_scaling.csv).*

---

## Baseline CUDA GPU Implementation (Step 4)

### 1. GPU Architecture & Mapping
The baseline CUDA implementation ([`src/cuda/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/cuda)) maps the moving-average filter onto thousands of concurrent GPU threads:
- **Thread Mapping:** Each thread computes exactly one output element:
  $$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
- **Block Size:** $B = 256$ threads per block.
- **Grid Size:** $\lceil N / 256 \rceil$ blocks.
- **Memory Access:** Each thread loads $W$ elements directly from high-latency GPU global memory (DRAM).
- **Error Checking:** Every CUDA API call (`cudaMalloc`, `cudaMemcpy`, kernel launch, event recording) is wrapped in strict error-handling macros (`CUDA_CHECK`).
- **Timing:** Dedicated CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`) measure Host-to-Device ($H2D$), Kernel Execution, and Device-to-Host ($D2H$) times separately with microsecond resolution.

### 2. How to Build & Run CUDA Baseline
```powershell
# Build CUDA baseline
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target cuda

# Run built-in correctness verification suite
.\bin\moving_average_cuda.exe --test

# Run safe initial performance test (N = 10,000, W = 15)
.\bin\moving_average_cuda.exe --initial-test

# Run on custom workload
.\bin\moving_average_cuda.exe -n 100000 -w 31 -b 256
```

---

## Shared-Memory Optimized CUDA Implementation (Step 5)

### 1. Shared-Memory Tiling with Halo Cells
The optimized CUDA implementation ([`src/optimized/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/src/optimized)) exploits on-chip SRAM (Shared Memory) to eliminate redundant global memory fetches:
- **Contiguous Tile Processing:** Each block processes an output tile of size $B = \text{blockDim.x} = 256$.
- **Halo Loading:** Each block cooperatively loads $B + 2k$ elements into `extern __shared__ SampleType s_data[]`, where $k = (W - 1) / 2$ represents the left and right halo cells.
- **Theoretical Load-Count Reduction:** Global memory read instructions decrease from $B \times W$ to $B + 2k$ per block (a theoretical $\approx 50.7\times$ instruction reduction for $W=63$, while actual hardware DRAM traffic depends on cache hit rates).
- **Barrier Synchronization:** `__syncthreads()` guarantees all halo and tile elements are resident before computation starts.
- **Shared-Memory Bank Access:** Stencil access `s_data[tid + j]` accesses consecutive 32-bit banks across warp threads, resulting in conflict-free access during the inner compute loop.
- **Accurate Divergence Characterization:** Boundary clamping branches occur only in the first and terminal blocks of the grid during the cooperative load phase; interior blocks and the stencil computation loop execute branch-free.

### 2. How to Build & Run Optimized CUDA
```powershell
# Build optimized CUDA implementation
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target opt

# Run built-in correctness verification suite
.\bin\moving_average_opt.exe --test

# Run on custom workload
.\bin\moving_average_opt.exe -n 1000000 -w 63 -b 256
```

---

## Performance Comparison & Benchmark Results (Step 6)

### 1. Full Benchmark Comparison Table (Measured on Target Hardware)

| Implementation | Workload Size | $N$ | Window $W$ | Threads / Block | Kernel Time (ms) | End-to-End Time (ms) | Speedup vs Seq | Correctness |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sequential** | Small | $10,000$ | $15$ | $1$ | N/A | **0.1290** ms | $1.00\times$ | REFERENCE |
| **OpenMP ($T=8$)** | Small | $10,000$ | $15$ | $8$ | N/A | **0.0870** ms | $1.48\times$ | PASSED |
| **OpenMP ($T=24$)** | Small | $10,000$ | $15$ | $24$ | N/A | **0.3120** ms | **0.41x (Slower)** | PASSED |
| **CUDA Baseline** | Small | $10,000$ | $15$ | $256$ | $0.0164$ ms | **0.0579** ms | $2.23\times$ (Kernel: $7.87\times$) | PASSED |
| **CUDA Optimized** | Small | $10,000$ | $15$ | $256$ | $0.0340$ ms | **0.1327** ms | **0.97x (Slower)** (Kernel: $3.79\times$) | PASSED |
| | | | | | | | | |
| **Sequential** | Medium | $100,000$ | $31$ | $1$ | N/A | **1.6540** ms | $1.00\times$ | REFERENCE |
| **OpenMP ($T=8$)** | Medium | $100,000$ | $31$ | $8$ | N/A | **0.5650** ms | $2.93\times$ | PASSED |
| **OpenMP ($T=24$)** | Medium | $100,000$ | $31$ | $24$ | N/A | **0.4160** ms | $3.98\times$ | PASSED |
| **CUDA Baseline** | Medium | $100,000$ | $31$ | $256$ | $0.1039$ ms | **0.2976** ms | $5.56\times$ (Kernel: $15.92\times$) | PASSED |
| **CUDA Optimized** | Medium | $100,000$ | $31$ | $256$ | $0.1059$ ms | **0.3258** ms | $5.08\times$ (Kernel: $15.62\times$) | PASSED |
| | | | | | | | | |
| **Sequential** | Large | $1,000,000$ | $63$ | $1$ | N/A | **39.5710** ms | $1.00\times$ | REFERENCE |
| **OpenMP ($T=8$)** | Large | $1,000,000$ | $63$ | $8$ | N/A | **6.9710** ms | $5.68\times$ | PASSED |
| **OpenMP ($T=24$)** | Large | $1,000,000$ | $63$ | $24$ | N/A | **5.7330** ms | $6.90\times$ | PASSED |
| **CUDA Baseline** | Large | $1,000,000$ | $63$ | $256$ | $1.6661$ ms | **3.0540** ms | $12.96\times$ (Kernel: $23.75\times$) | PASSED |
| **CUDA Optimized** | Large | $1,000,000$ | $63$ | $256$ | **1.5797 ms** | **2.8489** ms | **13.89x** (Kernel: **25.05x**) | PASSED |

*(All results logged to [`results/tables/full_benchmark_comparison.csv`](file:///c:/Users/Predator/Desktop/Signal%20Processing/results/tables/full_benchmark_comparison.csv))*

### 2. Key Insights: When Parallelization is Slower
1. **CPU Over-threading:** On small inputs ($N=10,000$), OpenMP with 24 threads ($0.312$ ms) is **$2.4\times$ slower than sequential** ($0.129$ ms) due to thread dispatch and synchronization overhead.
2. **GPU PCIe Overhead:** On small inputs, PCIe host-to-device and device-to-host transfers represent over **$74\%$** of total execution time, making end-to-end GPU time slower than sequential CPU time.
3. **GPU Kernel vs. End-to-End Speedup:** For the Large dataset ($N=1\text{M}, W=63$), the optimized kernel achieves a **$25.05\times$ speedup**, but PCIe transfers limit the end-to-end speedup to **$13.89\times$**, demonstrating Amdahl's Law in heterogeneous computing.

---

## How to Reproduce All Results

```powershell
# 1. Build all executables
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target all

# 2. Generate deterministic benchmark input datasets
python scripts/generate_inputs.py

# 3. Execute all benchmarks (OpenMP scaling + multi-implementation comparison)
python benchmarks/run_all_benchmarks.py

# 4. Generate all publication plots
python scripts/plot_results.py
```

Generated plots are located in [`results/plots/`](file:///c:/Users/Predator/Desktop/Signal%20Processing/results/plots/):
- `execution_time_comparison.png`
- `speedup_comparison.png`
- `omp_thread_scaling.png`
- `cuda_breakdown.png`
- `signal_denoising_demo.png`

Full report materials with placeholders for submission are located in [`report/report_material.md`](file:///c:/Users/Predator/Desktop/Signal%20Processing/report/report_material.md).

---

## Development Roadmap

- [x] **Step 1:** Environment inspection, mathematical formulation, boundary selection, architecture design, and minimal scaffolding.
- [x] **Step 2:** Sequential baseline C++ implementation, input signal generator, parameter validation, and correctness verification suite.
- [x] **Step 3:** OpenMP parallel implementation, thread scalability analysis, and parallel efficiency measurement.
- [x] **Step 4:** CUDA GPU implementation (Host-to-Device transfer, kernel execution, Device-to-Host transfer).
- [x] **Step 5:** Optimized parallel implementation (Shared memory tiling / halo exchange).
- [x] **Step 6:** Comprehensive benchmarking (Small, Medium, Large inputs), performance tables, and speedup plots.
- [x] **Step 7:** Final technical report preparation and viva readiness.



