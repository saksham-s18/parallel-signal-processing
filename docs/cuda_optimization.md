# CUDA Shared-Memory Tiling Optimization with Halo Elements

## 1. Motivation
In the naive baseline CUDA kernel, each thread performs $W$ reads from device global memory. Because DRAM latency on GPUs is 200–400 clock cycles, reading overlapping elements repeatedly across threads in a warp saturates memory bus bandwidth.

For a block of $B = 256$ threads and window $W = 63$ ($k = 31$):
- **Baseline Global Memory Load Instructions:** $256 \times 63 = 16,128$ load instructions issued per block.
- **Tiled Global Memory Load Instructions:** $B + 2k = 256 + 62 = 318$ load instructions issued per block.
- **Theoretical Load-Count Reduction:** $\approx 50.7\times$ fewer global memory load instructions issued. (Note: actual physical DRAM traffic reduction on hardware depends on hardware L1/L2 cache hit rates).

## 2. Shared Memory Geometry
For each block:
- **Output tile size:** `blockDim.x` (256 threads).
- **Left halo size:** $k = (W - 1) / 2$.
- **Right halo size:** $k = (W - 1) / 2$.
- **Shared array size:** `sharedSize = blockDim.x + 2 * k`.

### Cooperative Loading
Threads stride cooperatively through the shared memory array:
```cpp
for (int sIdx = tid; sIdx < sharedSize; sIdx += blockDim.x) {
    int64_t gIdx = globalWindowStart + sIdx;
    int64_t clampedIdx = (gIdx < 0) ? 0 : (gIdx >= n ? n - 1 : gIdx);
    s_data[sIdx] = d_input[clampedIdx];
}
```

## 3. Synchronization
`__syncthreads()` guarantees that all $B + 2k$ elements in shared memory are valid before any thread starts computing its output.

## 4. Bank Conflict Analysis
NVIDIA GPUs have 32 shared memory banks, 4 bytes wide. In the stencil loop:
```cpp
for (int j = 0; j < windowSize; ++j) {
    sum += static_cast<double>(s_data[tid + j]);
}
```
For fixed $j$, thread $tid$ reads address `&s_data[tid + j]`.
Since address differences between consecutive threads in a warp are exactly 4 bytes (1 float), each thread in the warp accesses a different 32-bit bank:
$$\text{Bank} = (\text{tid} + j) \pmod{32}$$
Thus, under this linear 4-byte strided addressing, all 32 lanes access distinct banks simultaneously, resulting in conflict-free shared memory access during the inner compute loop.

## 5. Warp Divergence Characterization
- **During Cooperative Load:** Clamping branches (`gIdx < 0` or `gIdx >= n`) cause divergent execution only in the very first block of the grid (`blockIdx.x == 0`) and the terminal block. For all interior blocks, threads execute uniform branches.
- **During Computation:** Every active thread executes the identical inner loop of length $W$, yielding uniform, non-divergent instruction execution.
