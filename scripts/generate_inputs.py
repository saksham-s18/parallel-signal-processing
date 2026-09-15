#!/usr/bin/env python3
"""
generate_inputs.py
Deterministic synthetic 1D noisy signal generator for CSS311 Assignment 1.
Generates reproducible input signals using a fixed random seed.
Binary file format matches include/signal_io.hpp:
- uint64_t n (8 bytes, little-endian)
- n * float32 (single precision floating point samples)
"""

import os
import struct
import numpy as np

def generate_noisy_sine(n: int, sampling_freq: float = 1000.0, signal_freq: float = 5.0, noise_std: float = 0.5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / sampling_freq
    two_pi = np.float32(2.0 * np.pi)
    clean = np.sin(two_pi * signal_freq * t, dtype=np.float32)
    noise = rng.normal(0.0, noise_std, n).astype(np.float32)
    return clean + noise

def save_binary(filepath: str, signal: np.ndarray):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n = len(signal)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(signal.astype(np.float32).tobytes())
    print(f"Saved {n} samples ({os.path.getsize(filepath):,} bytes) to: {filepath}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "data", "input")

    datasets = [
        {"name": "Small",  "filename": "input_small_N10000.bin",    "n": 10000,   "seed": 42},
        {"name": "Medium", "filename": "input_medium_N100000.bin",  "n": 100000,  "seed": 42},
        {"name": "Large",  "filename": "input_large_N1000000.bin", "n": 1000000, "seed": 42},
    ]

    print("==================================================")
    print("Generating Deterministic Benchmark Input Datasets")
    print("==================================================")

    for ds in datasets:
        path = os.path.join(input_dir, ds["filename"])
        sig = generate_noisy_sine(ds["n"], seed=ds["seed"])
        save_binary(path, sig)

    # Also generate human-readable sample for report demonstration (N=5, W=3)
    demo_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    demo_path = os.path.join(input_dir, "sample_demo_N5.csv")
    with open(demo_path, "w") as f:
        f.write("Index,InputSample\n")
        for i, val in enumerate(demo_input):
            f.write(f"{i},{val:.4f}\n")
    print(f"Saved human-readable demo signal to: {demo_path}")
    print("All benchmark datasets successfully generated.\n")

if __name__ == "__main__":
    main()
