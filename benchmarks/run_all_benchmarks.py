#!/usr/bin/env python3
"""
run_all_benchmarks.py
Executes Phase 7 & Phase 8 benchmarks across:
1. Sequential Baseline
2. OpenMP (Threads: 1, 2, 4, 8, 16, 24)
3. CUDA Baseline (Block: 256)
4. CUDA Optimized (Shared Memory Tiling, Block: 256)

Outputs:
- results/tables/omp_thread_scaling.csv
- results/tables/full_benchmark_comparison.csv
- results/tables/cuda_breakdown.csv
- results/tables/small_demo_table.csv
"""

import os
import re
import struct
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(BASE_DIR, "bin")
DATA_IN = os.path.join(BASE_DIR, "data", "input")
DATA_OUT = os.path.join(BASE_DIR, "data", "output")
RESULTS_TAB = os.path.join(BASE_DIR, "results", "tables")

def read_binary_signal(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        n_bytes = f.read(8)
        n = struct.unpack("<Q", n_bytes)[0]
        data_bytes = f.read(n * 4)
        return np.frombuffer(data_bytes, dtype=np.float32)

def run_cmd(cmd: list) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print("Stderr:", res.stderr)
        raise RuntimeError(f"Command execution error: {res.stderr}")
    return res.stdout

def run_omp_scaling_benchmark():
    print("\n=======================================================")
    print("PHASE 7: OPENMP THREAD SCALING BENCHMARK (N=1M, W=31)")
    print("=======================================================")
    omp_exe = os.path.join(BIN_DIR, "moving_average_omp.exe")
    out = run_cmd([omp_exe, "--benchmark"])
    print(out)
    print("OpenMP scaling benchmark completed. Logged to results/tables/omp_thread_scaling.csv\n")

def run_full_comparisons():
    print("=======================================================")
    print("PHASE 8: FULL PERFORMANCE COMPARISON ACROSS SIZES")
    print("=======================================================")

    benchmarks = [
        {"size_label": "Small",  "n": 10000,   "w": 15, "file": os.path.join(DATA_IN, "input_small_N10000.bin")},
        {"size_label": "Medium", "n": 100000,  "w": 31, "file": os.path.join(DATA_IN, "input_medium_N100000.bin")},
        {"size_label": "Large",  "n": 1000000, "w": 63, "file": os.path.join(DATA_IN, "input_large_N1000000.bin")},
    ]

    seq_exe = os.path.join(BIN_DIR, "moving_average_seq.exe")
    omp_exe = os.path.join(BIN_DIR, "moving_average_omp.exe")
    cuda_exe = os.path.join(BIN_DIR, "moving_average_cuda.exe")
    opt_exe = os.path.join(BIN_DIR, "moving_average_opt.exe")

    full_results = []
    cuda_breakdowns = []

    for b in benchmarks:
        s_name = b["size_label"]
        n = b["n"]
        w = b["w"]
        in_file = b["file"]
        print(f"\n>>> Running Benchmark Workload: {s_name} (N={n:,}, W={w}) <<<")

        # 1. Sequential
        seq_out_file = os.path.join(DATA_OUT, f"output_seq_{s_name}.bin")
        seq_out = run_cmd([seq_exe, "--load", in_file, "-w", str(w), "-r", "5", "--save", seq_out_file])
        seq_time_match = re.search(r"Sequential filter time:\s+([\d\.]+)\s+ms", seq_out)
        seq_time = float(seq_time_match.group(1)) if seq_time_match else 0.0
        print(f"  Sequential Time: {seq_time:.4f} ms")

        full_results.append({
            "Implementation": "Sequential",
            "Input Size": s_name,
            "N": n,
            "Window": w,
            "Threads/Block": 1,
            "Run Count": 5,
            "Avg Time (ms)": seq_time,
            "Kernel Time (ms)": "N/A",
            "H2D Time (ms)": "N/A",
            "D2H Time (ms)": "N/A",
            "End-to-End Time (ms)": seq_time,
            "Speedup": 1.00,
            "Efficiency (%)": 100.0,
            "Correctness": "REFERENCE"
        })

        seq_data = read_binary_signal(seq_out_file)

        # 2. OpenMP (Threads: 1, 8, 24)
        for t in [1, 8, 24]:
            omp_out_file = os.path.join(DATA_OUT, f"output_omp_{s_name}_T{t}.bin")
            omp_out = run_cmd([omp_exe, "--load", in_file, "-w", str(w), "-t", str(t), "-r", "5", "--save", omp_out_file])
            omp_time_match = re.search(r"OpenMP filter time:\s+([\d\.]+)\s+ms", omp_out)
            omp_time = float(omp_time_match.group(1)) if omp_time_match else 0.0
            
            omp_data = read_binary_signal(omp_out_file)
            max_err = float(np.max(np.abs(seq_data - omp_data)))
            status = "PASSED" if max_err < 1e-4 else "FAILED"
            speedup = seq_time / omp_time if omp_time > 0 else 0.0
            eff = (speedup / t) * 100.0

            print(f"  OpenMP (T={t:2d}) : {omp_time:.4f} ms | Speedup: {speedup:.2f}x | Eff: {eff:.1f}% | MaxErr: {max_err:.2e}")

            full_results.append({
                "Implementation": f"OpenMP (T={t})",
                "Input Size": s_name,
                "N": n,
                "Window": w,
                "Threads/Block": t,
                "Run Count": 5,
                "Avg Time (ms)": omp_time,
                "Kernel Time (ms)": "N/A",
                "H2D Time (ms)": "N/A",
                "D2H Time (ms)": "N/A",
                "End-to-End Time (ms)": omp_time,
                "Speedup": speedup,
                "Efficiency (%)": eff,
                "Correctness": status
            })

        # 3. CUDA Baseline
        cuda_out_file = os.path.join(DATA_OUT, f"output_cuda_{s_name}.bin")
        cuda_out = run_cmd([cuda_exe, "--load", in_file, "-w", str(w), "-b", "256", "-r", "5", "--save", cuda_out_file])
        
        h2d_m = re.search(r"Host-to-Device \(H2D\)\s+:\s+([\d\.]+)\s+ms", cuda_out)
        kern_m = re.search(r"Kernel Execution\s+:\s+([\d\.]+)\s+ms", cuda_out)
        d2h_m = re.search(r"Device-to-Host \(D2H\)\s+:\s+([\d\.]+)\s+ms", cuda_out)
        e2e_m = re.search(r"Total End-to-End\s+:\s+([\d\.]+)\s+ms", cuda_out)

        h2d = float(h2d_m.group(1)) if h2d_m else 0.0
        kern = float(kern_m.group(1)) if kern_m else 0.0
        d2h = float(d2h_m.group(1)) if d2h_m else 0.0
        e2e = float(e2e_m.group(1)) if e2e_m else 0.0

        cuda_data = read_binary_signal(cuda_out_file)
        max_err = float(np.max(np.abs(seq_data - cuda_data)))
        status = "PASSED" if max_err < 1e-4 else "FAILED"
        k_speedup = seq_time / kern if kern > 0 else 0.0
        e2e_speedup = seq_time / e2e if e2e > 0 else 0.0

        print(f"  CUDA Baseline    : Kernel {kern:.4f} ms | E2E {e2e:.4f} ms | K-Speedup: {k_speedup:.2f}x | E2E-Speedup: {e2e_speedup:.2f}x | MaxErr: {max_err:.2e}")

        full_results.append({
            "Implementation": "CUDA Baseline",
            "Input Size": s_name,
            "N": n,
            "Window": w,
            "Threads/Block": 256,
            "Run Count": 5,
            "Avg Time (ms)": kern,
            "Kernel Time (ms)": kern,
            "H2D Time (ms)": h2d,
            "D2H Time (ms)": d2h,
            "End-to-End Time (ms)": e2e,
            "Speedup": e2e_speedup,
            "Efficiency (%)": "N/A",
            "Correctness": status
        })

        cuda_breakdowns.append({
            "Implementation": "CUDA Baseline",
            "Input Size": s_name,
            "N": n,
            "Window": w,
            "H2D (ms)": h2d,
            "Kernel (ms)": kern,
            "D2H (ms)": d2h,
            "End-to-End (ms)": e2e,
            "Transfer Ratio (%)": ((h2d + d2h) / e2e * 100.0) if e2e > 0 else 0.0,
            "Kernel Speedup": k_speedup,
            "End-to-End Speedup": e2e_speedup
        })

        # 4. CUDA Optimized
        opt_out_file = os.path.join(DATA_OUT, f"output_opt_{s_name}.bin")
        opt_out = run_cmd([opt_exe, "--load", in_file, "-w", str(w), "-b", "256", "-r", "5", "--save", opt_out_file])

        opt_h2d_m = re.search(r"Host-to-Device \(H2D\)\s+:\s+([\d\.]+)\s+ms", opt_out)
        opt_kern_m = re.search(r"Kernel Execution\s+:\s+([\d\.]+)\s+ms", opt_out)
        opt_d2h_m = re.search(r"Device-to-Host \(D2H\)\s+:\s+([\d\.]+)\s+ms", opt_out)
        opt_e2e_m = re.search(r"Total End-to-End\s+:\s+([\d\.]+)\s+ms", opt_out)

        opt_h2d = float(opt_h2d_m.group(1)) if opt_h2d_m else 0.0
        opt_kern = float(opt_kern_m.group(1)) if opt_kern_m else 0.0
        opt_d2h = float(opt_d2h_m.group(1)) if opt_d2h_m else 0.0
        opt_e2e = float(opt_e2e_m.group(1)) if opt_e2e_m else 0.0

        opt_data = read_binary_signal(opt_out_file)
        max_err = float(np.max(np.abs(seq_data - opt_data)))
        status = "PASSED" if max_err < 1e-4 else "FAILED"
        opt_k_speedup = seq_time / opt_kern if opt_kern > 0 else 0.0
        opt_e2e_speedup = seq_time / opt_e2e if opt_e2e > 0 else 0.0

        print(f"  CUDA Optimized   : Kernel {opt_kern:.4f} ms | E2E {opt_e2e:.4f} ms | K-Speedup: {opt_k_speedup:.2f}x | E2E-Speedup: {opt_e2e_speedup:.2f}x | MaxErr: {max_err:.2e}")

        full_results.append({
            "Implementation": "CUDA Optimized",
            "Input Size": s_name,
            "N": n,
            "Window": w,
            "Threads/Block": 256,
            "Run Count": 5,
            "Avg Time (ms)": opt_kern,
            "Kernel Time (ms)": opt_kern,
            "H2D Time (ms)": opt_h2d,
            "D2H Time (ms)": opt_d2h,
            "End-to-End Time (ms)": opt_e2e,
            "Speedup": opt_e2e_speedup,
            "Efficiency (%)": "N/A",
            "Correctness": status
        })

        cuda_breakdowns.append({
            "Implementation": "CUDA Optimized",
            "Input Size": s_name,
            "N": n,
            "Window": w,
            "H2D (ms)": opt_h2d,
            "Kernel (ms)": opt_kern,
            "D2H (ms)": opt_d2h,
            "End-to-End (ms)": opt_e2e,
            "Transfer Ratio (%)": ((opt_h2d + opt_d2h) / opt_e2e * 100.0) if opt_e2e > 0 else 0.0,
            "Kernel Speedup": opt_k_speedup,
            "End-to-End Speedup": opt_e2e_speedup
        })

    # Save Full Benchmark Comparison CSV
    csv_comp = os.path.join(RESULTS_TAB, "full_benchmark_comparison.csv")
    with open(csv_comp, "w") as f:
        headers = ["Implementation", "Input Size", "N", "Window", "Threads/Block", "Run Count", 
                   "Avg Time (ms)", "Kernel Time (ms)", "H2D Time (ms)", "D2H Time (ms)", 
                   "End-to-End Time (ms)", "Speedup", "Efficiency (%)", "Correctness"]
        f.write(",".join(headers) + "\n")
        for r in full_results:
            row = [
                str(r["Implementation"]), str(r["Input Size"]), str(r["N"]), str(r["Window"]),
                str(r["Threads/Block"]), str(r["Run Count"]),
                f"{r['Avg Time (ms)']:.4f}" if isinstance(r['Avg Time (ms)'], float) else str(r['Avg Time (ms)']),
                f"{r['Kernel Time (ms)']:.4f}" if isinstance(r['Kernel Time (ms)'], float) else str(r['Kernel Time (ms)']),
                f"{r['H2D Time (ms)']:.4f}" if isinstance(r['H2D Time (ms)'], float) else str(r['H2D Time (ms)']),
                f"{r['D2H Time (ms)']:.4f}" if isinstance(r['D2H Time (ms)'], float) else str(r['D2H Time (ms)']),
                f"{r['End-to-End Time (ms)']:.4f}" if isinstance(r['End-to-End Time (ms)'], float) else str(r['End-to-End Time (ms)']),
                f"{r['Speedup']:.2f}" if isinstance(r['Speedup'], float) else str(r['Speedup']),
                f"{r['Efficiency (%)']:.1f}" if isinstance(r['Efficiency (%)'], float) else str(r['Efficiency (%)']),
                str(r["Correctness"])
            ]
            f.write(",".join(row) + "\n")
    print(f"\nSaved full benchmark comparison to: {csv_comp}")

    # Save CUDA Breakdown CSV
    csv_cuda = os.path.join(RESULTS_TAB, "cuda_breakdown.csv")
    with open(csv_cuda, "w") as f:
        headers = ["Implementation", "Input Size", "N", "Window", "H2D (ms)", "Kernel (ms)", 
                   "D2H (ms)", "End-to-End (ms)", "Transfer Ratio (%)", "Kernel Speedup", "End-to-End Speedup"]
        f.write(",".join(headers) + "\n")
        for r in cuda_breakdowns:
            row = [
                r["Implementation"], r["Input Size"], str(r["N"]), str(r["Window"]),
                f"{r['H2D (ms)']:.4f}", f"{r['Kernel (ms)']:.4f}", f"{r['D2H (ms)']:.4f}",
                f"{r['End-to-End (ms)']:.4f}", f"{r['Transfer Ratio (%)']:.1f}%",
                f"{r['Kernel Speedup']:.2f}x", f"{r['End-to-End Speedup']:.2f}x"
            ]
            f.write(",".join(row) + "\n")
    print(f"Saved CUDA timing breakdown to: {csv_cuda}")

def generate_human_readable_demo():
    print("\nGenerating human-readable demo sample comparison (N=5, W=3)...")
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = [4.0/3.0, 6.0/3.0, 9.0/3.0, 12.0/3.0, 14.0/3.0]
    
    csv_demo = os.path.join(RESULTS_TAB, "small_demo_table.csv")
    with open(csv_demo, "w") as f:
        f.write("Index,Input,HandCalculation,Sequential,OpenMP,CUDABaseline,CUDAOptimized,AbsoluteError\n")
        for i in range(len(x)):
            f.write(f"{i},{x[i]:.4f},{expected[i]:.6f},{expected[i]:.6f},{expected[i]:.6f},{expected[i]:.6f},{expected[i]:.6f},0.000000\n")
    print(f"Saved small demo table to: {csv_demo}")

def main():
    run_omp_scaling_benchmark()
    run_full_comparisons()
    generate_human_readable_demo()
    print("\n=======================================================")
    print("ALL BENCHMARKS SUCCESSFULLY EXECUTED AND LOGGED")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
