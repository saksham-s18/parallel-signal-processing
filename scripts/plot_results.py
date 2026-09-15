#!/usr/bin/env python3
"""
plot_results.py
Generates publication-quality charts from actual measured benchmark CSV results.
Reads:
- results/tables/full_benchmark_comparison.csv
- results/tables/omp_thread_scaling.csv
- results/tables/cuda_breakdown.csv

Saves to: results/plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set high-quality visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14

def plot_execution_time_comparison():
    csv_path = os.path.join(TABLES_DIR, "full_benchmark_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Filter key implementations
    targets = ["Sequential", "OpenMP (T=24)", "CUDA Baseline", "CUDA Optimized"]
    df_filtered = df[df["Implementation"].isin(targets)].copy()

    sizes = ["Small", "Medium", "Large"]
    x = np.arange(len(sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    colors = {
        "Sequential": "#e74c3c",
        "OpenMP (T=24)": "#3498db",
        "CUDA Baseline": "#f39c12",
        "CUDA Optimized": "#2ecc71"
    }

    for i, impl in enumerate(targets):
        subset = df_filtered[df_filtered["Implementation"] == impl]
        # Order by sizes
        subset = subset.set_index("Input Size").reindex(sizes).reset_index()
        times = pd.to_numeric(subset["End-to-End Time (ms)"])
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, times, width, label=impl, color=colors[impl], alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Small\n(N=10k, W=15)", "Medium\n(N=100k, W=31)", "Large\n(N=1M, W=63)"])
    ax.set_ylabel("Execution Time in ms (Log Scale)")
    ax.set_title("Execution Time Comparison Across Implementations (Log Scale)", fontweight="bold", pad=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "execution_time_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_speedup_comparison():
    csv_path = os.path.join(TABLES_DIR, "full_benchmark_comparison.csv")
    df = pd.read_csv(csv_path)

    sizes = ["Small", "Medium", "Large"]
    x = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Plot OpenMP (T=24), CUDA Baseline E2E, CUDA Baseline Kernel, CUDA Opt E2E, CUDA Opt Kernel
    configs = [
        ("OpenMP (T=24)", "End-to-End Time (ms)", "#3498db", "o-", "OpenMP (T=24)"),
        ("CUDA Baseline", "End-to-End Time (ms)", "#e67e22", "s--", "CUDA Baseline (End-to-End)"),
        ("CUDA Baseline", "Kernel Time (ms)", "#d35400", "s-", "CUDA Baseline (Kernel-Only)"),
        ("CUDA Optimized", "End-to-End Time (ms)", "#27ae60", "^--", "CUDA Optimized (End-to-End)"),
        ("CUDA Optimized", "Kernel Time (ms)", "#2ecc71", "^-", "CUDA Optimized (Kernel-Only)")
    ]

    seq_subset = df[df["Implementation"] == "Sequential"].set_index("Input Size").reindex(sizes)
    seq_times = pd.to_numeric(seq_subset["Avg Time (ms)"]).values

    width = 0.15
    for idx, (impl, time_col, color, marker, label) in enumerate(configs):
        sub = df[df["Implementation"] == impl].set_index("Input Size").reindex(sizes)
        times = pd.to_numeric(sub[time_col]).values
        speedups = seq_times / times
        offset = (idx - 2) * width
        ax.bar(x + offset, speedups, width, label=label, color=color, alpha=0.88, edgecolor="black", linewidth=0.5)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Sequential Baseline (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Small\n(N=10k, W=15)", "Medium\n(N=100k, W=31)", "Large\n(N=1M, W=63)"])
    ax.set_ylabel("Measured Speedup vs Sequential Baseline (x)")
    ax.set_title("Parallel Speedup Across Workload Sizes: CPU vs GPU", fontweight="bold", pad=12)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "speedup_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_omp_thread_scaling():
    csv_path = os.path.join(TABLES_DIR, "omp_thread_scaling.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    threads = df["Threads"].values
    times = df["AvgTimeMs"].values
    speedup = df["Speedup"].values
    eff = df["EfficiencyPercent"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Plot Speedup & Ideal
    ax1.plot(threads, speedup, "o-", color="#2980b9", linewidth=2.2, markersize=7, label="Measured Speedup")
    ax1.plot(threads, threads, "--", color="#7f8c8d", linewidth=1.5, label="Ideal Linear Speedup")
    ax1.set_xlabel("Number of Threads (T)")
    ax1.set_ylabel("Speedup Factor (x)")
    ax1.set_title("OpenMP Speedup Scaling (N = 1,000,000, W = 31)", fontweight="bold")
    ax1.set_xticks(threads)
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white")

    # Plot Efficiency
    ax2.plot(threads, eff, "s-", color="#e74c3c", linewidth=2.2, markersize=7, label="Parallel Efficiency")
    ax2.axhline(100.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="100% Ideal Efficiency")
    ax2.set_xlabel("Number of Threads (T)")
    ax2.set_ylabel("Parallel Efficiency (%)")
    ax2.set_title("OpenMP Parallel Efficiency vs Thread Count", fontweight="bold")
    ax2.set_xticks(threads)
    ax2.set_ylim(0, 115)
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor="white")

    plt.suptitle("Intel Core i7-13700HX OpenMP Scaling Analysis", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "omp_thread_scaling.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_cuda_time_breakdown():
    csv_path = os.path.join(TABLES_DIR, "cuda_breakdown.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    sizes = ["Small", "Medium", "Large"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Subplot 1: Absolute Time Breakdown
    labels = []
    h2d_vals, kern_vals, d2h_vals = [], [], []

    for s in sizes:
        for impl_short, impl_name in [("Base", "CUDA Baseline"), ("Opt", "CUDA Optimized")]:
            row = df[(df["Input Size"] == s) & (df["Implementation"] == impl_name)].iloc[0]
            labels.append(f"{s}\n{impl_short}")
            h2d_vals.append(row["H2D (ms)"])
            kern_vals.append(row["Kernel (ms)"])
            d2h_vals.append(row["D2H (ms)"])

    x = np.arange(len(labels))
    p1 = ax1.bar(x, h2d_vals, label="Host-to-Device (H2D)", color="#3498db", alpha=0.9, edgecolor="black", linewidth=0.5)
    p2 = ax1.bar(x, kern_vals, bottom=h2d_vals, label="Kernel Execution", color="#2ecc71", alpha=0.9, edgecolor="black", linewidth=0.5)
    bottom_d2h = np.array(h2d_vals) + np.array(kern_vals)
    p3 = ax1.bar(x, d2h_vals, bottom=bottom_d2h, label="Device-to-Host (D2H)", color="#e67e22", alpha=0.9, edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("CUDA Time Breakdown by Phase (Absolute ms)", fontweight="bold")
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, ls="--", alpha=0.5)

    # Subplot 2: Relative Percentage Stacked (Highlighting Transfer Dominance on Small)
    total_times = np.array(h2d_vals) + np.array(kern_vals) + np.array(d2h_vals)
    pct_h2d = np.array(h2d_vals) / total_times * 100.0
    pct_kern = np.array(kern_vals) / total_times * 100.0
    pct_d2h = np.array(d2h_vals) / total_times * 100.0

    ax2.bar(x, pct_h2d, label="H2D Transfer %", color="#3498db", alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.bar(x, pct_kern, bottom=pct_h2d, label="Kernel Execution %", color="#2ecc71", alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.bar(x, pct_d2h, bottom=pct_h2d + pct_kern, label="D2H Transfer %", color="#e67e22", alpha=0.9, edgecolor="black", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Percentage of Total Time (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Percentage Breakdown: Transfer Overhead vs Compute", fontweight="bold")
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(True, ls="--", alpha=0.5)

    plt.suptitle("RTX 4050 GPU Execution Profile: Demonstrating PCIe Transfer Overhead", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "cuda_breakdown.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_signal_denoising_demo():
    # Load demo noisy signal and filtered outputs to plot actual waveform smoothing
    in_path = os.path.join(BASE_DIR, "data", "input", "input_small_N10000.bin")
    out_path = os.path.join(BASE_DIR, "data", "output", "output_opt_Small.bin")

    if not os.path.exists(in_path) or not os.path.exists(out_path):
        return

    import struct
    def read_bin(p):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return np.frombuffer(f.read(n * 4), dtype=np.float32)

    raw = read_bin(in_path)[:400]
    smoothed = read_bin(out_path)[:400]
    t = np.arange(len(raw)) / 1000.0

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(t, raw, color="#95a5a6", alpha=0.75, label="Raw Noisy Input Signal (Sine + Gaussian Noise)")
    ax.plot(t, smoothed, color="#e74c3c", linewidth=2.0, label="Filtered Output (Moving-Average W=15)")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Moving-Average Signal Denoising Demonstration (First 400 Samples)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "signal_denoising_demo.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def main():
    plot_execution_time_comparison()
    plot_speedup_comparison()
    plot_omp_thread_scaling()
    plot_cuda_time_breakdown()
    plot_signal_denoising_demo()
    print("All plots successfully generated.\n")

if __name__ == "__main__":
    main()
