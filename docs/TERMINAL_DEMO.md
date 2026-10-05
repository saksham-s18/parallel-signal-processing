# Live Terminal Demonstration Guide
## Parallel Signal Processing: 1D Moving-Average Noise Reduction
**Course:** CSS311 Parallel & Distributed Computing  
**Author:** Saksham Sharma  
**Target Platform:** Windows 11 / PowerShell 7+ / NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)

---

## Pre-Presentation Checklist (Setup Before Professor Arrives)

1. Open **6 PowerShell Terminal Windows** and arrange or tab them.
2. In all terminals, navigate to the project directory:
   ```powershell
   cd "C:\Users\Predator\Desktop\Signal Processing"
   ```
3. Ensure the project executables are compiled and up to date:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target all
   ```
4. Verify NVIDIA GPU detection:
   ```powershell
   nvidia-smi
   ```

---

## Terminal 6 — Hardware & CUDA Environment Inspection

Run this **first** to prove the runtime environment to the professor.

### Commands:
```powershell
cd "C:\Users\Predator\Desktop\Signal Processing"
nvidia-smi
nvcc --version
```

### Expected Output:
- `nvidia-smi` displays **NVIDIA GeForce RTX 4050 Laptop GPU** (6141 MiB VRAM), Driver 616.92, CUDA 13.4.
- `nvcc --version` outputs: `Cuda compilation tools, release 13.4, V13.4.59`.

### What to Explain:
- *"This system uses an NVIDIA Ada Lovelace architecture RTX 4050 mobile GPU with Compute Capability 8.9 (`sm_89`)."*
- *"The CPU is a modern multi-core processor providing 24 logical execution threads, which we will use for OpenMP scaling."*

---

## Terminal 1 — Sequential Baseline Implementation

### 1. Small Hand-Verifiable Proof ($N = 5, W = 3$)
Demonstrates algorithmic correctness against manual calculation.

```powershell
.\bin\moving_average_seq.exe --load data/input/input_demo_N5.bin -w 3 -r 1
```
*(Alternative built-in self-test: `.\bin\moving_average_seq.exe --test`)*

#### Expected Output:
```text
Loading input signal from: data/input/input_demo_N5.bin ... Done (5 samples).
Input Signal  (total 5 samples): [1.0000, 2.0000, 3.0000, 4.0000, 5.0000]
Smoothed Output (total 5 samples): [1.3333, 2.0000, 3.0000, 4.0000, 4.6667]
```

### 2. Small Workload Benchmark ($N = 10,000, W = 15$)
```powershell
.\bin\moving_average_seq.exe --load data/input/input_small_N10000.bin -w 15
```

#### Expected Output:
```text
Loading input signal from: data/input/input_small_N10000.bin ...  Done (10000 samples).
>>> Sequential filter time: ~0.10 - 0.13 ms <<<
Throughput: ~95 - 100 MSamples/sec
```

### What to Explain:
- **Algorithm:** $O(N \times W)$ direct window summation. For each sample $i$, it sums $W$ neighbors from $i - k$ to $i + k$ ($k = \frac{W-1}{2}$).
- **Boundary Condition:** Edge replication (clamping indices $< 0$ to $0$, and $> N-1$ to $N-1$).
- **Verification:** For $x = [1, 2, 3, 4, 5]$ with $W = 3$:
  - $i=0$: $\frac{1 + 1 + 2}{3} = 1.3333$
  - $i=1$: $\frac{1 + 2 + 3}{3} = 2.0000$
  - $i=2$: $\frac{2 + 3 + 4}{3} = 3.0000$
  - $i=3$: $\frac{3 + 4 + 5}{3} = 4.0000$
  - $i=4$: $\frac{4 + 5 + 5}{3} = 4.6667$
- This provides the exact baseline reference against which all parallel implementations are validated.

---

## Terminal 2 — OpenMP Multi-Core Implementation

### 1. Small Hand-Verifiable Proof ($N = 5, W = 3$)
```powershell
.\bin\moving_average_omp.exe --load data/input/input_demo_N5.bin -w 3 -t 2 -r 1
```

#### Expected Output:
```text
Validation Report: OpenMP vs Sequential Reference
 Status              : PASSED [OK]
 Max Absolute Error  : 0.000000e+00
Input Signal  (total 5 samples): [1.0000, 2.0000, 3.0000, 4.0000, 5.0000]
OpenMP Output (total 5 samples): [1.3333, 2.0000, 3.0000, 4.0000, 4.6667]
```

### 2. Small Workload with 8 Threads ($N = 10,000, W = 15$)
```powershell
.\bin\moving_average_omp.exe --load data/input/input_small_N10000.bin -w 15 -t 8
```
- **Filter time:** $\approx 0.08 - 0.19\text{ ms}$
- **Speedup:** $\approx 1.07\text{x} - 1.48\text{x}$
- **Validation:** `PASSED [OK]` (Max Absolute Error: $0.0$)

### 3. Small Workload with 24 Threads ($N = 10,000, W = 15$)
```powershell
.\bin\moving_average_omp.exe --load data/input/input_small_N10000.bin -w 15 -t 24
```
- **Filter time:** $\approx 0.16 - 0.31\text{ ms}$
- **Speedup:** $\approx 0.41\text{x} - 0.47\text{x}$ (Sub-linear / slowdown!)
- **Validation:** `PASSED [OK]` (Max Absolute Error: $0.0$)

### What to Explain:
- **Parallel Strategy:** Domain decomposition using `#pragma omp parallel for schedule(static)`. Each thread independently computes an equal chunk of output indices without loop dependencies or data races.
- **Overhead Lesson (Crucial for Defense):** Point out why 24 threads are **slower** than 8 threads at $N = 10,000$:
  - Work per thread is tiny ($10,000 / 24 \approx 416$ elements).
  - OpenMP thread team fork/join barriers and thread synchronization latency exceed the actual compute time.
  - This demonstrates Amdahl's Law and synchronization overhead on small workloads.

---

## Terminal 3 — CUDA Baseline Implementation

### 1. Small Hand-Verifiable Proof ($N = 5, W = 3$)
```powershell
.\bin\moving_average_cuda.exe --load data/input/input_demo_N5.bin -w 3 -r 1
```

#### Expected Output:
```text
Validation Report: CUDA Baseline vs Sequential Reference
 Status              : PASSED [OK]
 Max Absolute Error  : 0.000000e+00
CUDA Output   (total 5 samples): [1.3333, 2.0000, 3.0000, 4.0000, 4.6667]
```

### 2. Small Workload Benchmark ($N = 10,000, W = 15$)
```powershell
.\bin\moving_average_cuda.exe --load data/input/input_small_N10000.bin -w 15
```

#### Expected Output:
```text
Validation Report: CUDA Baseline vs Sequential Reference -> PASSED [OK]
>>> CUDA TIMING BREAKDOWN (Averaged over 5 runs) <<<
  Host-to-Device (H2D) : ~0.05 ms
  Kernel Execution     : ~0.02 - 0.03 ms
  Device-to-Host (D2H) : ~0.05 ms
  Total End-to-End     : ~0.13 ms
  Kernel-only Speedup  : ~4.7x - 7.8x
  End-to-End Speedup   : ~1.05x - 2.2x
```

### What to Explain:
- **GPU Mapping:** 1D grid of thread blocks (block size = 256 threads). Global thread ID: `int i = blockDim.x * blockIdx.x + threadIdx.x`.
- **Memory Access:** Every thread fetches $W$ samples directly from GPU global DRAM. Redundant memory transactions occur across neighboring threads.
- **Transfer Bottleneck:** Notice that PCIe transfer time ($\text{H2D} + \text{D2H} \approx 0.10\text{ ms}$) accounts for $> 70\%$ of total runtime at $N = 10,000$.

---

## Terminal 4 — Shared-Memory Optimized CUDA Implementation

### 1. Small Hand-Verifiable Proof ($N = 5, W = 3$)
```powershell
.\bin\moving_average_opt.exe --load data/input/input_demo_N5.bin -w 3 -r 1
```

#### Expected Output:
```text
Validation Report: Optimized CUDA vs Sequential Reference
 Status              : PASSED [OK]
 Max Absolute Error  : 0.000000e+00
Opt Output    (total 5 samples): [1.3333, 2.0000, 3.0000, 4.0000, 4.6667]
```

### 2. Small Workload Benchmark ($N = 10,000, W = 15$)
```powershell
.\bin\moving_average_opt.exe --load data/input/input_small_N10000.bin -w 15
```

#### Expected Output:
```text
Shared Mem/Block  : 1080 bytes
Validation Report : Optimized CUDA vs Sequential Reference -> PASSED [OK]
>>> OPTIMIZED CUDA TIMING BREAKDOWN <<<
  Host-to-Device (H2D) : ~0.03 ms
  Kernel Execution     : ~0.019 ms
  Device-to-Host (D2H) : ~0.03 ms
  Total End-to-End     : ~0.08 ms
  Kernel-only Speedup  : ~7.9x
```

### What to Explain:
- **Optimization Strategy:** Collaborative tile loading into on-chip `__shared__` memory.
- **Halo Cells:** To compute $B$ outputs ($B = 256$) with window radius $k = \frac{W-1}{2}$, threads collaboratively load a tile of size $B + 2k$ elements into shared memory, including $k$ left boundary and $k$ right boundary halo cells.
- **Barrier Synchronization:** `__syncthreads()` ensures all halo and interior elements are loaded before window summation begins.
- **Memory Bandwidth:** Shared memory operates at on-chip cache speeds ($\sim\text{terabytes/sec}$), eliminating redundant global DRAM requests.

---

## Terminal 5 — Benchmark Results & Scaling Analysis

Run the read-only PowerShell presentation script to present full validated benchmark tables directly in the terminal without rerunning heavy workloads:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\show_results.ps1
```

### Direct Native PowerShell Commands (if professor asks to inspect raw CSVs):
```powershell
# 1. Full Multi-workload comparison:
Import-Csv results\tables\full_benchmark_comparison.csv | Format-Table -AutoSize

# 2. CUDA Memory Breakdown:
Import-Csv results\tables\cuda_breakdown.csv | Format-Table -AutoSize

# 3. OpenMP Thread Scaling (1 to 24 threads):
Import-Csv results\tables\omp_thread_scaling.csv | Format-Table -AutoSize
```

### Summary of What the Display Shows:

| Implementation | Small ($N=10\text{k}, W=15$) | Medium ($N=100\text{k}, W=31$) | Large ($N=1\text{M}, W=63$) |
| :--- | :---: | :---: | :---: |
| **Sequential Baseline** | 0.129 ms (1.00x) | 1.654 ms (1.00x) | 39.571 ms (1.00x) |
| **OpenMP (8 threads)** | 0.087 ms (1.48x) | 0.565 ms (2.93x) | 6.971 ms (5.68x) |
| **OpenMP (24 threads)**| 0.312 ms (0.41x) | 0.416 ms (3.98x) | 5.733 ms (6.90x) |
| **CUDA Baseline** | 0.016 ms (2.23x E2E) | 0.104 ms (5.56x E2E) | 1.666 ms (12.96x E2E) |
| **CUDA Optimized** | 0.034 ms (0.97x E2E) | 0.106 ms (5.08x E2E) | 1.580 ms (13.89x E2E) |

---

## 12-Step Recommended Presentation Sequence

1. **Terminal 6:** Run `nvidia-smi` and `nvcc --version`.  
   *“This project targets our local NVIDIA RTX 4050 GPU (Ada Lovelace, 6 GB VRAM) alongside a 24-logical-core CPU.”*
2. **Build Terminal:** Run `powershell -ExecutionPolicy Bypass -File .\build.ps1 -Target all`.  
   *“All 4 targets—Sequential, OpenMP, CUDA Baseline, and CUDA Shared Memory—build from clean C++17/CUDA sources.”*
3. **Terminal 1:** Run Sequential with $N=5, W=3$:  
   `.\bin\moving_average_seq.exe --load data/input/input_demo_N5.bin -w 3 -r 1`  
   *“For input [1, 2, 3, 4, 5] and window 3 with clamped edges, the exact output is [1.3333, 2.0, 3.0, 4.0, 4.6667].”*
4. **Terminal 2:** Run OpenMP with $N=5, W=3$:  
   `.\bin\moving_average_omp.exe --load data/input/input_demo_N5.bin -w 3 -t 2 -r 1`
5. **Terminal 3:** Run CUDA Baseline with $N=5, W=3$:  
   `.\bin\moving_average_cuda.exe --load data/input/input_demo_N5.bin -w 3 -r 1`
6. **Terminal 4:** Run CUDA Optimized with $N=5, W=3$:  
   `.\bin\moving_average_opt.exe --load data/input/input_demo_N5.bin -w 3 -r 1`
7. **Point out Identical Results:**  
   *“All 4 implementations produce identical numerical outputs with $0.0$ absolute error, confirming numerical correctness.”*
8. **Terminal 1:** Run Small Workload ($N=10,000, W=15$):  
   `.\bin\moving_average_seq.exe --load data/input/input_small_N10000.bin -w 15` (Takes $\approx 0.10\text{ ms}$).
9. **Terminal 2:** Run OpenMP Small ($T=8$ vs $T=24$):  
   Show $T=8$ ($1.07\text{x}-1.48\text{x}$) vs $T=24$ ($0.41\text{x}-0.47\text{x}$).  
   *“This vividly demonstrates thread contention and barrier overhead when task granularity is too small.”*
10. **Terminals 3 & 4:** Run CUDA Small on $N=10,000$:  
    Show that kernel execution is fast ($0.02\text{ ms}$), but PCIe transfers dominate the end-to-end time.
11. **Terminal 5:** Run `powershell -ExecutionPolicy Bypass -File .\scripts\show_results.ps1`.  
    Show the Medium ($N=100\text{k}$) and Large ($N=1\text{M}$) benchmark tables.  
    *“At $N=1,000,000$, OpenMP scales to 6.90x, while CUDA achieves 13.89x end-to-end speedup and 25.05x kernel-only speedup.”*
12. **Closing Conclusion:**  
    *“As problem size increases, the computational intensity shifts from memory transfer-bound to compute-bound, fully unlocking the parallelism of the GPU.”*
