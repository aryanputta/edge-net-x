"""
GPU-accelerated (or CPU-fallback) inference engine.

Runs in a dedicated background thread via loop.run_in_executor so the
asyncio event loop is never blocked. Exposes an async interface.

Batch mode: collects pending windows into a single GPU call to maximise
throughput; measures and records single vs batch latency for benchmarking.
"""
import asyncio
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch

from ml.model import CongestionLSTM, INPUT_FEATURES


class InferenceEngine:
    def __init__(self, model: CongestionLSTM, seq_len: int = 50):
        self._model = model
        self._seq_len = seq_len
        self._device = next(model.parameters()).device
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-infer")
        self._lock = threading.Lock()

        # Benchmarking
        self.last_single_latency_ms: float = 0.0
        self.last_batch_latency_ms: float = 0.0
        self.inference_count: int = 0
        self.device_name: str = str(self._device)

        # Whether GPU is actually in use
        self.using_gpu: bool = self._device.type == "cuda"

    def _infer_sync(self, windows: List[List[List[float]]]) -> List[float]:
        with self._lock:
            t0 = time.perf_counter()
            # Pad/truncate each window to seq_len
            padded = []
            for w in windows:
                if len(w) < self._seq_len:
                    pad = [[0.0] * INPUT_FEATURES] * (self._seq_len - len(w))
                    w = pad + list(w)
                else:
                    w = list(w)[-self._seq_len:]
                padded.append(w)

            x = torch.tensor(padded, dtype=torch.float32).to(self._device)
            with torch.no_grad():
                scores = self._model(x).squeeze(-1).cpu().tolist()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.inference_count += len(windows)
            if len(windows) == 1:
                self.last_single_latency_ms = elapsed_ms
            else:
                self.last_batch_latency_ms = elapsed_ms
            return scores if isinstance(scores, list) else [scores]

    async def infer_single(self, window: deque) -> float:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor, self._infer_sync, [list(window)]
        )
        return results[0]

    async def infer_batch(self, windows: List[deque]) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._infer_sync, [list(w) for w in windows]
        )

    def benchmark(self, window: deque, batch_size: int = 32) -> Tuple[float, float]:
        """Run single vs batch inference and return (single_ms, batch_ms)."""
        single_start = time.perf_counter()
        self._infer_sync([list(window)])
        single_ms = (time.perf_counter() - single_start) * 1000.0

        batch = [list(window)] * batch_size
        batch_start = time.perf_counter()
        self._infer_sync(batch)
        batch_ms = (time.perf_counter() - batch_start) * 1000.0

        return single_ms, batch_ms

    def shutdown(self):
        self._executor.shutdown(wait=False)


def build_inference_engine(seq_len: int = 50) -> Tuple[InferenceEngine, str]:
    """Train model, move to best available device, return engine + device description."""
    from ml.model import train_model

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"[ML] Training CongestionLSTM on {device_str.upper()}...")
    model = train_model(seq_len=seq_len, device=device_str)
    engine = InferenceEngine(model, seq_len)

    if device_str == "cuda":
        desc = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        desc = "CPU"

    print(f"[ML] Model ready on {desc}")
    return engine, desc
