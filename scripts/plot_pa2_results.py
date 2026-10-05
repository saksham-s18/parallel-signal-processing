#!/usr/bin/env python3
"""
plot_pa2_results.py
Generates publication-quality charts for PA2 empirical results:
1. MPI Process Scaling (Speedup & Time across P)
2. 6-Way Architecture Comparison across Workload Sizes
3. Hardware-Level Energy & EDP Comparison (Joules & Watt-hours)
4. MPI Overlap Profiling Breakdown

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

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

def plot_mpi_scaling():
    csv_path = os.path.join(TABLES_DIR, "pa2_mpi_scaling.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    df_large = df[df["Workload"] == "Large"].sort_values("Processes")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # 1. Execution Time Plot
    p = df_large["Processes"].values
    ax1.plot(p, df_large["BasicEndToEndMs"].values, marker="o", linewidth=2.2, color="#e67e22", label="Basic MPI (Blocking)")
    ax1.plot(p, df_large["OptEndToEndMs"].values, marker="s", linewidth=2.2, color="#2980b9", label="Optimized MPI (Non-Blocking Overlap)")
    ax1.axhline(df_large["SeqTimeMs"].iloc[0], color="#c0392b", linestyle="--", linewidth=1.8, label="Sequential Reference")

    ax1.set_xlabel("Number of MPI Processes (P)")
    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_title("MPI Process Scaling: Execution Time (N=1M, W=63)", fontweight="bold")
    ax1.set_xticks(p)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Speedup Plot
    ax2.plot(p, df_large["BasicSpeedup"].values, marker="o", linewidth=2.2, color="#e67e22", label="Basic MPI Speedup")
    ax2.plot(p, df_large["OptSpeedup"].values, marker="s", linewidth=2.2, color="#2980b9", label="Optimized MPI Speedup")
    ax2.plot(p, p, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Ideal Linear Speedup")

    ax2.set_xlabel("Number of MPI Processes (P)")
    ax2.set_ylabel("Speedup (vs. Sequential)")
    ax2.set_title("MPI Process Scaling: Measured Speedup (N=1M, W=63)", fontweight="bold")
    ax2.set_xticks(p)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "pa2_mpi_scaling.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def plot_6way_comparison():
    csv_path = os.path.join(TABLES_DIR, "pa2_full_comparison.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    impl_order = ["Sequential", "OpenMP", "Basic MPI", "Optimized MPI", "CUDA Baseline", "CUDA Optimized"]
    df_large = df[df["Workload"] == "Large"].copy()
    
    # Filter and sort
    df_large = df_large.set_index("Implementation").reindex(impl_order).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    colors = ["#7f8c8d", "#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#2ecc71"]

    bars = ax.bar(df_large["Implementation"], df_large["Speedup"], color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    ax.set_ylabel("Speedup vs. Sequential Baseline")
    ax.set_title("6-Way Architectural Speedup Comparison (Large Workload: N=1M, W=63)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "pa2_6way_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def plot_energy_comparison():
    csv_path = os.path.join(TABLES_DIR, "pa2_energy_comparison.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    colors = ["#7f8c8d", "#3498db", "#e74c3c", "#2ecc71", "#e67e22", "#9b59b6"]

    # 1. Total Energy Consumed (Joules)
    bars1 = ax1.bar(df["Implementation"], df["TotalEnergyJoules"], color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax1.set_ylabel("Total Energy Consumed (Joules)")
    ax1.set_title("Hardware Energy Consumption (RAPL CPU & NVML GPU)", fontweight="bold")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f} J",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    # 2. Energy-Delay Product (EDP in Joule-seconds)
    bars2 = ax2.bar(df["Implementation"], df["EnergyDelayProduct"], color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax2.set_ylabel("Energy-Delay Product (J * s)")
    ax2.set_title("Energy-Delay Product (EDP) Lower is Greener/Better", fontweight="bold")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "pa2_energy_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def plot_mpi_overlap_efficiency():
    csv_path = os.path.join(TABLES_DIR, "pa2_profiling_summary.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    p = df["Processes"].values
    eff = df["OverlapEfficiencyPercent"].values

    bars = ax.bar([str(x) for x in p], eff, color="#1abc9c", edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_xlabel("MPI Process Count (P)")
    ax.set_ylabel("Communication-Computation Overlap Efficiency (%)")
    ax.set_title("Optimized MPI: Communication Latency Hiding Efficiency (N=100K)", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "pa2_mpi_overlap.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    plot_mpi_scaling()
    plot_6way_comparison()
    plot_energy_comparison()
    plot_mpi_overlap_efficiency()
