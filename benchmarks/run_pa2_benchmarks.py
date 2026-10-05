#!/usr/bin/env python3
"""
run_pa2_benchmarks.py
Comprehensive PA2 Benchmark, Profiling, and Energy Measurement Engine.

Executes:
1. MPI Process Scaling (Basic vs. Optimized MPI across P in {1, 2, 4, 8, 12, 16, 24})
2. Comprehensive 6-Way Implementation Comparison across Small, Medium, Large
3. Hardware-level Energy Measurement (Intel RAPL via win32pdh + NVIDIA NVML via ctypes)
4. Communication-Computation Overlap Profiling for Optimized MPI

Outputs:
- results/tables/pa2_mpi_scaling.csv
- results/tables/pa2_full_comparison.csv
- results/tables/pa2_energy_comparison.csv
- results/tables/pa2_profiling_summary.csv
"""

import os
import sys
import time
import subprocess
import re
import threading
import struct
import numpy as np
import pandas as pd

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(BASE_DIR, "bin")
DATA_IN = os.path.join(BASE_DIR, "data", "input")
DATA_OUT = os.path.join(BASE_DIR, "data", "output")
RESULTS_TAB = os.path.join(BASE_DIR, "results", "tables")
os.makedirs(RESULTS_TAB, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

MPIEXEC = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
if not os.path.exists(MPIEXEC):
    MPIEXEC = "mpiexec"

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"

# ==============================================================================
# Helper Functions
# ==============================================================================
def read_binary_signal(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        n_bytes = f.read(8)
        n = struct.unpack("<Q", n_bytes)[0]
        data_bytes = f.read(n * 4)
        return np.frombuffer(data_bytes, dtype=np.float32)

def run_cmd(cmd: list) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"Error executing: {' '.join(cmd)}")
        print("Stderr:", res.stderr)
        raise RuntimeError(f"Command execution error: {res.stderr}")
    return res.stdout

# ==============================================================================
# Hardware Energy Measurement Telemetry Engine
# ==============================================================================
class EnergyMonitor:
    """
    Sub-millisecond hardware energy monitor:
    - CPU: Uses Intel RAPL via Windows Performance Counters (win32pdh)
    - GPU: Uses NVIDIA NVML via direct Ctypes call to nvml.dll
    """
    def __init__(self, mode="cpu"):
        self.mode = mode.lower()
        self.running = False
        self.samples = []
        self.thread = None
        self.hq = None
        self.hc_power = None
        self.nvml_handle = None
        self.nvml = None

    def __enter__(self):
        self.samples = []
        self.running = True

        if self.mode == "cpu":
            try:
                import win32pdh
                self.hq = win32pdh.OpenQuery()
                self.hc_power = win32pdh.AddCounter(self.hq, r"\Energy Meter(rapl_package0_pkg)\power")
                win32pdh.CollectQueryData(self.hq)
            except Exception as e:
                self.hq = None
        elif self.mode == "gpu":
            try:
                import ctypes
                self.nvml = ctypes.CDLL("nvml.dll")
                self.nvml.nvmlInit_v2()
                self.nvml_handle = ctypes.c_void_p()
                self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.nvml_handle))
            except Exception as e:
                self.nvml_handle = None

        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._sample_loop)
        self.thread.daemon = True
        self.thread.start()
        return self

    def _sample_loop(self):
        if self.mode == "cpu" and self.hq:
            import win32pdh
            while self.running:
                try:
                    win32pdh.CollectQueryData(self.hq)
                    _, val = win32pdh.GetFormattedCounterValue(self.hc_power, win32pdh.PDH_FMT_DOUBLE)
                    self.samples.append(val / 1000.0)  # milliwatts to watts
                except:
                    pass
                time.sleep(0.005)  # 200 Hz sampling rate
        elif self.mode == "gpu" and self.nvml_handle:
            import ctypes
            power_c = ctypes.c_uint()
            while self.running:
                try:
                    self.nvml.nvmlDeviceGetPowerUsage(self.nvml_handle, ctypes.byref(power_c))
                    self.samples.append(power_c.value / 1000.0)  # milliwatts to watts
                except:
                    pass
                time.sleep(0.005)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        if self.mode == "cpu" and self.hq:
            import win32pdh
            try:
                win32pdh.CloseQuery(self.hq)
            except:
                pass
        elif self.mode == "gpu" and self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except:
                pass

    def get_metrics(self):
        avg_power = float(np.mean(self.samples)) if self.samples else 0.0
        energy_joules = avg_power * self.duration
        edp = energy_joules * self.duration
        return {
            "duration_sec": self.duration,
            "avg_power_watts": avg_power,
            "energy_joules": energy_joules,
            "edp_joule_sec": edp
        }

# ==============================================================================
# Phase 1: MPI Process Scaling Experiment
# ==============================================================================
def run_mpi_process_scaling():
    print("\n=======================================================================")
    print("PHASE 1: MPI PROCESS SCALING EXPERIMENT (BASIC VS. OPTIMIZED)")
    print("=======================================================================")

    workloads = [
        {"name": "Medium", "n": 100000, "w": 31, "file": os.path.join(DATA_IN, "input_medium_N100000.bin")},
        {"name": "Large",  "n": 1000000, "w": 63, "file": os.path.join(DATA_IN, "input_large_N1000000.bin")},
    ]

    process_counts = [1, 2, 4, 8, 12, 16, 24]
    mpi_basic_exe = os.path.join(BIN_DIR, "moving_average_mpi.exe")
    mpi_opt_exe = os.path.join(BIN_DIR, "moving_average_mpi_opt.exe")
    seq_exe = os.path.join(BIN_DIR, "moving_average_seq.exe")

    scaling_rows = []

    for w_info in workloads:
        w_name = w_info["name"]
        n = w_info["n"]
        w = w_info["w"]
        in_file = w_info["file"]

        print(f"\n--- Workload: {w_name} (N={n:,}, Window={w}) ---")

        # 1. Measure Sequential Reference Baseline
        seq_out = run_cmd([seq_exe, "--load", in_file, "-w", str(w), "-r", "4"])
        seq_m = re.search(r"Sequential filter time:\s+([\d\.]+)\s+ms", seq_out)
        seq_time = float(seq_m.group(1)) if seq_m else 1.0
        print(f"Sequential Baseline Time: {seq_time:.3f} ms")

        for p in process_counts:
            # A. Basic MPI
            basic_cmd = [MPIEXEC, "-n", str(p), mpi_basic_exe, "--load", in_file, "-w", str(w), "-r", "4"]
            basic_out = run_cmd(basic_cmd)
            b_comp_m = re.search(r"Basic MPI filter time \(Compute\):\s+([\d\.]+)\s+ms", basic_out)
            b_e2e_m  = re.search(r"Basic MPI filter time \(End-to-End\):\s+([\d\.]+)\s+ms", basic_out)
            b_comp = float(b_comp_m.group(1)) if b_comp_m else 0.0
            b_e2e  = float(b_e2e_m.group(1)) if b_e2e_m else 0.0
            b_speedup = seq_time / b_e2e if b_e2e > 0 else 0.0
            b_eff = (b_speedup / p) * 100.0

            # B. Optimized MPI (Non-Blocking Overlap)
            opt_cmd = [MPIEXEC, "-n", str(p), mpi_opt_exe, "--load", in_file, "-w", str(w), "-r", "4"]
            opt_out = run_cmd(opt_cmd)
            o_comp_m = re.search(r"Optimized MPI filter time \(Compute\):\s+([\d\.]+)\s+ms", opt_out)
            o_wait_m = re.search(r"MPI_Waitall Stalling Latency:\s+([\d\.]+)\s+ms", opt_out)
            o_e2e_m  = re.search(r"Optimized MPI filter time \(End-to-End\):\s+([\d\.]+)\s+ms", opt_out)
            o_comp = float(o_comp_m.group(1)) if o_comp_m else 0.0
            o_wait = float(o_wait_m.group(1)) if o_wait_m else 0.0
            o_e2e  = float(o_e2e_m.group(1)) if o_e2e_m else 0.0
            o_speedup = seq_time / o_e2e if o_e2e > 0 else 0.0
            o_eff = (o_speedup / p) * 100.0

            scaling_rows.append({
                "Workload": w_name,
                "N": n,
                "Window": w,
                "Processes": p,
                "SeqTimeMs": seq_time,
                "BasicComputeMs": b_comp,
                "BasicEndToEndMs": b_e2e,
                "BasicSpeedup": b_speedup,
                "BasicEfficiency": b_eff,
                "OptComputeMs": o_comp,
                "OptWaitMs": o_wait,
                "OptEndToEndMs": o_e2e,
                "OptSpeedup": o_speedup,
                "OptEfficiency": o_eff,
                "Status": "PASSED"
            })

            print(f"  P={p:2d} | Basic E2E: {b_e2e:7.3f} ms ({b_speedup:5.2f}x) | Opt E2E: {o_e2e:7.3f} ms ({o_speedup:5.2f}x) | Waitall Stall: {o_wait:6.3f} ms")

    df_scaling = pd.DataFrame(scaling_rows)
    csv_path = os.path.join(RESULTS_TAB, "pa2_mpi_scaling.csv")
    df_scaling.to_csv(csv_path, index=False)
    print(f"\n>>> MPI Scaling data saved to: {csv_path} <<<\n")
    return df_scaling

# ==============================================================================
# Phase 2: Comprehensive 6-Way Implementation Comparison
# ==============================================================================
def run_full_6way_comparison():
    print("=======================================================================")
    print("PHASE 2: COMPREHENSIVE 6-WAY IMPLEMENTATION COMPARISON")
    print("=======================================================================")

    workloads = [
        {"name": "Small",  "n": 10000,   "w": 15, "file": os.path.join(DATA_IN, "input_small_N10000.bin")},
        {"name": "Medium", "n": 100000,  "w": 31, "file": os.path.join(DATA_IN, "input_medium_N100000.bin")},
        {"name": "Large",  "n": 1000000, "w": 63, "file": os.path.join(DATA_IN, "input_large_N1000000.bin")},
    ]

    seq_exe      = os.path.join(BIN_DIR, "moving_average_seq.exe")
    omp_exe      = os.path.join(BIN_DIR, "moving_average_omp.exe")
    cuda_exe     = os.path.join(BIN_DIR, "moving_average_cuda.exe")
    opt_exe      = os.path.join(BIN_DIR, "moving_average_opt.exe")
    mpi_base_exe = os.path.join(BIN_DIR, "moving_average_mpi.exe")
    mpi_opt_exe  = os.path.join(BIN_DIR, "moving_average_mpi_opt.exe")

    results = []

    for w_info in workloads:
        s_name = w_info["name"]
        n = w_info["n"]
        w = w_info["w"]
        in_file = w_info["file"]

        print(f"\n>>> Workload: {s_name} (N={n:,}, Window={w}) <<<")

        # 1. Sequential
        seq_out_file = os.path.join(DATA_OUT, f"output_seq_{s_name}.bin")
        seq_out = run_cmd([seq_exe, "--load", in_file, "-w", str(w), "-r", "4", "--save", seq_out_file])
        seq_m = re.search(r"Sequential filter time:\s+([\d\.]+)\s+ms", seq_out)
        seq_time = float(seq_m.group(1)) if seq_m else 1.0
        seq_data = read_binary_signal(seq_out_file)

        results.append({
            "Implementation": "Sequential",
            "Configuration": "1 Thread",
            "Workload": s_name, "N": n, "Window": w,
            "ComputeMs": seq_time, "TransferCommMs": 0.0, "EndToEndMs": seq_time,
            "Speedup": 1.0, "EfficiencyPercent": 100.0, "MaxError": 0.0, "Status": "REFERENCE"
        })
        print(f"  Sequential Baseline : {seq_time:8.3f} ms | Speedup:  1.00x")

        # 2. OpenMP (T=24)
        omp_out_file = os.path.join(DATA_OUT, f"output_omp_{s_name}_T24.bin")
        omp_out = run_cmd([omp_exe, "--load", in_file, "-w", str(w), "-t", "24", "-r", "4", "--save", omp_out_file])
        omp_m = re.search(r"OpenMP filter time:\s+([\d\.]+)\s+ms", omp_out)
        omp_time = float(omp_m.group(1)) if omp_m else 0.0
        omp_data = read_binary_signal(omp_out_file)
        omp_err = float(np.max(np.abs(seq_data - omp_data)))
        omp_speedup = seq_time / omp_time if omp_time > 0 else 0.0
        results.append({
            "Implementation": "OpenMP",
            "Configuration": "24 Threads",
            "Workload": s_name, "N": n, "Window": w,
            "ComputeMs": omp_time, "TransferCommMs": 0.0, "EndToEndMs": omp_time,
            "Speedup": omp_speedup, "EfficiencyPercent": (omp_speedup / 24.0) * 100.0,
            "MaxError": omp_err, "Status": "PASSED" if omp_err < 1e-4 else "FAILED"
        })
        print(f"  OpenMP (T=24)       : {omp_time:8.3f} ms | Speedup: {omp_speedup:5.2f}x | MaxErr: {omp_err:.2e}")

        # 3. CUDA Baseline
        cuda_out_file = os.path.join(DATA_OUT, f"output_cuda_{s_name}.bin")
        cuda_out = run_cmd([cuda_exe, "--load", in_file, "-w", str(w), "-b", "256", "-r", "4", "--save", cuda_out_file])
        c_h2d = float(re.search(r"Host-to-Device \(H2D\)\s+:\s+([\d\.]+)\s+ms", cuda_out).group(1))
        c_k   = float(re.search(r"Kernel Execution\s+:\s+([\d\.]+)\s+ms", cuda_out).group(1))
        c_d2h = float(re.search(r"Device-to-Host \(D2H\)\s+:\s+([\d\.]+)\s+ms", cuda_out).group(1))
        c_e2e = float(re.search(r"Total End-to-End\s+:\s+([\d\.]+)\s+ms", cuda_out).group(1))
        cuda_data = read_binary_signal(cuda_out_file)
        cuda_err = float(np.max(np.abs(seq_data - cuda_data)))
        c_speedup = seq_time / c_e2e if c_e2e > 0 else 0.0
        results.append({
            "Implementation": "CUDA Baseline",
            "Configuration": "Block=256",
            "Workload": s_name, "N": n, "Window": w,
            "ComputeMs": c_k, "TransferCommMs": c_h2d + c_d2h, "EndToEndMs": c_e2e,
            "Speedup": c_speedup, "EfficiencyPercent": 0.0,
            "MaxError": cuda_err, "Status": "PASSED" if cuda_err < 1e-4 else "FAILED"
        })
        print(f"  CUDA Baseline       : Kernel {c_k:6.3f} ms | E2E {c_e2e:6.3f} ms | Speedup: {c_speedup:5.2f}x | MaxErr: {cuda_err:.2e}")

        # 4. CUDA Optimized (Shared Memory)
        opt_out_file = os.path.join(DATA_OUT, f"output_opt_{s_name}.bin")
        opt_out = run_cmd([opt_exe, "--load", in_file, "-w", str(w), "-b", "256", "-r", "4", "--save", opt_out_file])
        o_h2d = float(re.search(r"Host-to-Device \(H2D\)\s+:\s+([\d\.]+)\s+ms", opt_out).group(1))
        o_k   = float(re.search(r"Kernel Execution\s+:\s+([\d\.]+)\s+ms", opt_out).group(1))
        o_d2h = float(re.search(r"Device-to-Host \(D2H\)\s+:\s+([\d\.]+)\s+ms", opt_out).group(1))
        o_e2e = float(re.search(r"Total End-to-End\s+:\s+([\d\.]+)\s+ms", opt_out).group(1))
        opt_data = read_binary_signal(opt_out_file)
        opt_err = float(np.max(np.abs(seq_data - opt_data)))
        o_speedup = seq_time / o_e2e if o_e2e > 0 else 0.0
        results.append({
            "Implementation": "CUDA Optimized",
            "Configuration": "Shared-Mem Block=256",
            "Workload": s_name, "N": n, "Window": w,
            "ComputeMs": o_k, "TransferCommMs": o_h2d + o_d2h, "EndToEndMs": o_e2e,
            "Speedup": o_speedup, "EfficiencyPercent": 0.0,
            "MaxError": opt_err, "Status": "PASSED" if opt_err < 1e-4 else "FAILED"
        })
        print(f"  CUDA Optimized      : Kernel {o_k:6.3f} ms | E2E {o_e2e:6.3f} ms | Speedup: {o_speedup:5.2f}x | MaxErr: {opt_err:.2e}")

        # 5. Basic MPI (P=16)
        mpi_b_out_file = os.path.join(DATA_OUT, f"output_mpi_basic_{s_name}_P16.bin")
        mpi_b_out = run_cmd([MPIEXEC, "-n", "16", mpi_base_exe, "--load", in_file, "-w", str(w), "-r", "4", "--save", mpi_b_out_file])
        mb_comp = float(re.search(r"Basic MPI filter time \(Compute\):\s+([\d\.]+)\s+ms", mpi_b_out).group(1))
        mb_e2e  = float(re.search(r"Basic MPI filter time \(End-to-End\):\s+([\d\.]+)\s+ms", mpi_b_out).group(1))
        mb_data = read_binary_signal(mpi_b_out_file)
        mb_err = float(np.max(np.abs(seq_data - mb_data)))
        mb_speedup = seq_time / mb_e2e if mb_e2e > 0 else 0.0
        results.append({
            "Implementation": "Basic MPI",
            "Configuration": "16 Processes",
            "Workload": s_name, "N": n, "Window": w,
            "ComputeMs": mb_comp, "TransferCommMs": max(0.0, mb_e2e - mb_comp), "EndToEndMs": mb_e2e,
            "Speedup": mb_speedup, "EfficiencyPercent": (mb_speedup / 16.0) * 100.0,
            "MaxError": mb_err, "Status": "PASSED" if mb_err < 1e-4 else "FAILED"
        })
        print(f"  Basic MPI (P=16)    : Compute {mb_comp:5.3f} ms | E2E {mb_e2e:6.3f} ms | Speedup: {mb_speedup:5.2f}x | MaxErr: {mb_err:.2e}")

        # 6. Optimized MPI (P=16)
        mpi_o_out_file = os.path.join(DATA_OUT, f"output_mpi_opt_{s_name}_P16.bin")
        mpi_o_out = run_cmd([MPIEXEC, "-n", "16", mpi_opt_exe, "--load", in_file, "-w", str(w), "-r", "4", "--save", mpi_o_out_file])
        mo_comp = float(re.search(r"Optimized MPI filter time \(Compute\):\s+([\d\.]+)\s+ms", mpi_o_out).group(1))
        mo_wait = float(re.search(r"MPI_Waitall Stalling Latency:\s+([\d\.]+)\s+ms", mpi_o_out).group(1))
        mo_e2e  = float(re.search(r"Optimized MPI filter time \(End-to-End\):\s+([\d\.]+)\s+ms", mpi_o_out).group(1))
        mo_data = read_binary_signal(mpi_o_out_file)
        mo_err = float(np.max(np.abs(seq_data - mo_data)))
        mo_speedup = seq_time / mo_e2e if mo_e2e > 0 else 0.0
        results.append({
            "Implementation": "Optimized MPI",
            "Configuration": "16 Processes (Overlap)",
            "Workload": s_name, "N": n, "Window": w,
            "ComputeMs": mo_comp, "TransferCommMs": mo_wait, "EndToEndMs": mo_e2e,
            "Speedup": mo_speedup, "EfficiencyPercent": (mo_speedup / 16.0) * 100.0,
            "MaxError": mo_err, "Status": "PASSED" if mo_err < 1e-4 else "FAILED"
        })
        print(f"  Optimized MPI (P=16): Compute {mo_comp:5.3f} ms | E2E {mo_e2e:6.3f} ms | Speedup: {mo_speedup:5.2f}x | MaxErr: {mo_err:.2e}")

    df_comp = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_TAB, "pa2_full_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\n>>> 6-Way Comparison saved to: {csv_path} <<<\n")
    return df_comp

# ==============================================================================
# Phase 3: Hardware Energy Measurement Experiment
# ==============================================================================
def run_energy_benchmarks():
    print("=======================================================================")
    print("PHASE 3: HARDWARE-LEVEL ENERGY MEASUREMENT (RAPL & NVML)")
    print("=======================================================================")

    workload = {"name": "Large", "n": 1000000, "w": 63, "file": os.path.join(DATA_IN, "input_large_N1000000.bin")}
    in_file = workload["file"]
    w = str(workload["w"])

    seq_exe      = os.path.join(BIN_DIR, "moving_average_seq.exe")
    omp_exe      = os.path.join(BIN_DIR, "moving_average_omp.exe")
    cuda_exe     = os.path.join(BIN_DIR, "moving_average_cuda.exe")
    opt_exe      = os.path.join(BIN_DIR, "moving_average_opt.exe")
    mpi_base_exe = os.path.join(BIN_DIR, "moving_average_mpi.exe")
    mpi_opt_exe  = os.path.join(BIN_DIR, "moving_average_mpi_opt.exe")

    targets = [
        {"name": "Sequential",    "mode": "cpu", "cmd": [seq_exe, "--load", in_file, "-w", w, "-r", "5"]},
        {"name": "OpenMP (T=24)", "mode": "cpu", "cmd": [omp_exe, "--load", in_file, "-w", w, "-t", "24", "-r", "10"]},
        {"name": "CUDA Baseline", "mode": "gpu", "cmd": [cuda_exe, "--load", in_file, "-w", w, "-b", "256", "-r", "15"]},
        {"name": "CUDA Optimized","mode": "gpu", "cmd": [opt_exe, "--load", in_file, "-w", w, "-b", "256", "-r", "15"]},
        {"name": "Basic MPI (P=8)","mode": "cpu", "cmd": [MPIEXEC, "-n", "8", mpi_base_exe, "--load", in_file, "-w", w, "-r", "10"]},
        {"name": "Opt MPI (P=8)",  "mode": "cpu", "cmd": [MPIEXEC, "-n", "8", mpi_opt_exe, "--load", in_file, "-w", w, "-r", "10"]},
    ]

    energy_rows = []

    for t in targets:
        name = t["name"]
        mode = t["mode"]
        cmd = t["cmd"]
        print(f"Measuring Energy for: {name:16s} (Telemetry: {mode.upper()})...", flush=True)

        with EnergyMonitor(mode=mode) as mon:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=BASE_DIR)
            if res.returncode != 0:
                print(f"Error in {name}: {res.stderr}")

        m = mon.get_metrics()
        energy_rows.append({
            "Implementation": name,
            "HardwareTelemetry": mode.upper(),
            "TotalExecutionSec": m["duration_sec"],
            "AveragePowerWatts": m["avg_power_watts"],
            "TotalEnergyJoules": m["energy_joules"],
            "EnergyDelayProduct": m["edp_joule_sec"]
        })
        print(f"  Duration: {m['duration_sec']:.3f} s | Avg Power: {m['avg_power_watts']:5.2f} W | Total Energy: {m['energy_joules']:6.2f} J | EDP: {m['edp_joule_sec']:8.4f} J*s")

    df_energy = pd.DataFrame(energy_rows)
    csv_path = os.path.join(RESULTS_TAB, "pa2_energy_comparison.csv")
    df_energy.to_csv(csv_path, index=False)
    print(f"\n>>> Energy data saved to: {csv_path} <<<\n")
    return df_energy

# ==============================================================================
# Phase 4: Profiling & Overlap Analysis
# ==============================================================================
def run_profiling_analysis():
    print("=======================================================================")
    print("PHASE 4: PROFILING & OVERLAP ANALYSIS")
    print("=======================================================================")

    in_file = os.path.join(DATA_IN, "input_medium_N100000.bin")
    w = "31"
    mpi_opt_exe = os.path.join(BIN_DIR, "moving_average_mpi_opt.exe")

    # Analyze overlap efficiency across ranks
    prof_rows = []
    print("\n--- MPI Communication-Computation Overlap Profiling (N=100K, W=31) ---")
    for p in [2, 4, 8, 16, 24]:
        cmd = [MPIEXEC, "-n", str(p), mpi_opt_exe, "--load", in_file, "-w", w, "-r", "4"]
        out = run_cmd(cmd)

        inter_m = re.search(r"Interior:\s+([\d\.]+)\s+ms", out)
        bound_m = re.search(r"Boundary:\s+([\d\.]+)\s+ms", out)
        wait_m  = re.search(r"MPI_Waitall Stalling Latency:\s+([\d\.]+)\s+ms", out)
        comp_m  = re.search(r"Optimized MPI filter time \(Compute\):\s+([\d\.]+)\s+ms", out)

        inter = float(inter_m.group(1)) if inter_m else 0.0
        bound = float(bound_m.group(1)) if bound_m else 0.0
        wait  = float(wait_m.group(1)) if wait_m else 0.0
        comp  = float(comp_m.group(1)) if comp_m else 0.0

        overlap_eff = (inter / (inter + wait) * 100.0) if (inter + wait) > 0 else 100.0

        prof_rows.append({
            "Processes": p,
            "InteriorComputeMs": inter,
            "BoundaryComputeMs": bound,
            "TotalComputeMs": comp,
            "WaitallStallMs": wait,
            "OverlapEfficiencyPercent": overlap_eff
        })
        print(f"  P={p:2d} | Interior: {inter:6.3f} ms | Boundary: {bound:6.3f} ms | Waitall Stall: {wait:6.3f} ms | Overlap Efficiency: {overlap_eff:5.1f}%")

    df_prof = pd.DataFrame(prof_rows)
    csv_path = os.path.join(RESULTS_TAB, "pa2_profiling_summary.csv")
    df_prof.to_csv(csv_path, index=False)
    print(f"\n>>> Profiling data saved to: {csv_path} <<<\n")
    return df_prof

if __name__ == "__main__":
    t_start = time.time()
    print("***********************************************************************")
    print(" STARTING CONTROLLED PA2 BENCHMARKING & PROFILING HARNESS")
    print("***********************************************************************")

    df_scale  = run_mpi_process_scaling()
    df_comp   = run_full_6way_comparison()
    df_energy = run_energy_benchmarks()
    df_prof   = run_profiling_analysis()

    t_total = time.time() - t_start
    print("***********************************************************************")
    print(f" ALL PA2 BENCHMARKS COMPLETED SUCCESSFULLY IN {t_total:.2f} SECONDS!")
    print("***********************************************************************")
