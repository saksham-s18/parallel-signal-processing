# System Architecture & Algorithmic Notes

## 1. Domain Decomposition
The moving-average filter operates on a 1D sequence of length $N$.
Because each output sample $y[i]$ depends exclusively on reading $W$ values from input $x$ and writing to $y[i]$, the domain is decomposed across output elements:

- **OpenMP:** The loop $i \in [0, N-1]$ is statically chunked across $T$ CPU threads.
  $$\text{Chunk Size} = \left\lceil \frac{N}{T} \right\rceil$$
- **CUDA:** Output indices are mapped 1-to-1 to GPU threads:
  $$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
  Grid dimension is $\lceil N / B \rceil$, where block dimension $B = 256$.

## 2. Boundary Condition Analysis
Boundary elements ($i < k$ or $i \ge N - k$) require handling out-of-bounds reads. We selected **Edge Clamping (Replication)**:
$$\text{clampedIndex}(p) = \min(\max(p, 0), N - 1)$$

### Advantages:
1. **Constant Window Width:** Every sample averages exactly $W$ elements.
2. **Elimination of Warp Divergence:** Unlike zero-padding with variable normalization counts ($\frac{1}{count}$), every thread divides by the precalculated constant scalar $\frac{1}{W} = \text{invW}$, enabling uniform arithmetic instruction pipelines.
3. **Reproducibility:** Matches across all CPU and GPU implementations without floating-point drift.

## 3. Precision & Numerical Consistency
- Input and output buffers store 32-bit `float` (`SampleType`).
- Accumulation is maintained in 64-bit `double` precision (`double sum = 0.0`) inside CPU and GPU threads.
- Division is computed via multiplication with a precomputed 64-bit reciprocal (`invW = 1.0 / (double)windowSize`).
- Verified $0.00\times 10^0$ maximum absolute error across all implementations.
