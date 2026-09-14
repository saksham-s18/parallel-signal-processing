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

## Development Roadmap

- [x] **Step 1:** Environment inspection, mathematical formulation, boundary selection, architecture design, and minimal scaffolding.
- [x] **Step 2:** Sequential baseline C++ implementation, input signal generator, parameter validation, and correctness verification suite.
- [x] **Step 3:** OpenMP parallel implementation, thread scalability analysis, and parallel efficiency measurement.
- [ ] **Step 4:** CUDA GPU implementation (Host-to-Device transfer, kernel execution, Device-to-Host transfer).
- [ ] **Step 5:** Optimized parallel implementation (Shared memory tiling / halo exchange).
- [ ] **Step 6:** Comprehensive benchmarking (Small, Medium, Large inputs), performance tables, and speedup plots.
- [ ] **Step 7:** Final technical report preparation and viva readiness.


