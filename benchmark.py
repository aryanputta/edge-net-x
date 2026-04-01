"""
Standalone benchmark: measures GPU vs CPU inference throughput.

Run:
    python benchmark.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
from collections import deque
from ml.model import train_model, generate_training_data, INPUT_FEATURES
from ml.inference import InferenceEngine

SEQ_LEN = 50
BATCH_SIZES = [1, 8, 32, 64, 128]
WARMUP_RUNS = 5
BENCH_RUNS = 50


def make_window(seq_len: int) -> deque:
    d = deque(maxlen=seq_len)
    for _ in range(seq_len):
        d.append([0.5] * INPUT_FEATURES)
    return d


def run_benchmark(device_str: str):
    print(f"\n{'─'*50}")
    print(f"  Device: {device_str.upper()}")
    print(f"{'─'*50}")

    model = train_model(seq_len=SEQ_LEN, epochs=5, device=device_str)
    engine = InferenceEngine(model, SEQ_LEN)
    window = make_window(SEQ_LEN)

    results = {}
    for bs in BATCH_SIZES:
        batch = [window] * bs

        # Warmup
        for _ in range(WARMUP_RUNS):
            engine._infer_sync([list(w) for w in batch])

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(BENCH_RUNS):
            engine._infer_sync([list(w) for w in batch])
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / BENCH_RUNS) * 1000.0
        throughput = (bs * BENCH_RUNS) / elapsed

        results[bs] = (avg_ms, throughput)
        print(f"  batch={bs:4d}  avg={avg_ms:7.3f}ms  throughput={throughput:8.1f} inf/s")

    engine.shutdown()
    return results


def main():
    print("EDGE-NET-X PRO — Inference Benchmark")
    print("=" * 50)

    cpu_results = run_benchmark("cpu")

    if torch.cuda.is_available():
        gpu_results = run_benchmark("cuda")
        print(f"\n{'─'*50}")
        print("  CPU vs GPU Speedup")
        print(f"{'─'*50}")
        for bs in BATCH_SIZES:
            cpu_ms, _ = cpu_results[bs]
            gpu_ms, _ = gpu_results[bs]
            speedup = cpu_ms / max(gpu_ms, 1e-6)
            print(f"  batch={bs:4d}  speedup={speedup:.2f}x")
    else:
        print("\n[No CUDA device found — skipping GPU benchmark]")


if __name__ == "__main__":
    main()
