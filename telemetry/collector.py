"""
Telemetry pipeline.

Consumes PacketResult events from the slicer and RTT measurements from
clients. Maintains per-slice rolling windows and computes stats at each
tick (default 100 ms).
"""
import asyncio
import time
import math
from collections import deque
from typing import Dict, List, Tuple, Optional

from models import PacketResult, SliceMetrics, NetworkState
from config import SystemConfig


_WINDOW = 200  # samples per rolling window


class RollingStats:
    def __init__(self, maxlen: int = _WINDOW):
        self._latencies: deque = deque(maxlen=maxlen)
        self._sizes: deque = deque(maxlen=maxlen)
        self._dropped: deque = deque(maxlen=maxlen)
        self._timestamps: deque = deque(maxlen=maxlen)

    def record(self, result: PacketResult):
        self._timestamps.append(result.timestamp)
        self._dropped.append(1 if result.dropped else 0)
        if not result.dropped:
            self._latencies.append(result.latency_ms)
            self._sizes.append(result.size_bytes)

    def latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def jitter_ms(self) -> float:
        if len(self._latencies) < 2:
            return 0.0
        diffs = [abs(self._latencies[i] - self._latencies[i - 1])
                 for i in range(1, len(self._latencies))]
        return sum(diffs) / len(diffs)

    def packet_loss_pct(self) -> float:
        if not self._dropped:
            return 0.0
        return 100.0 * sum(self._dropped) / len(self._dropped)

    def throughput_mbps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        total_bytes = sum(self._sizes)
        return (total_bytes * 8) / (elapsed * 1_000_000)


class TelemetryCollector:
    def __init__(self, config: SystemConfig, slicer_results: asyncio.Queue):
        self._config = config
        self._results_q = slicer_results
        self._stats: Dict[str, RollingStats] = {
            name: RollingStats() for name in config.slices
        }
        self._active_flows: Dict[str, set] = {name: set() for name in config.slices}
        self._running = False

        # Telemetry window for ML: list of (timestamp, feature_vector)
        self._ml_window: deque = deque(maxlen=config.ml_window_size)
        self._state: Optional[NetworkState] = None

    @property
    def ml_window(self) -> deque:
        return self._ml_window

    @property
    def current_state(self) -> Optional[NetworkState]:
        return self._state

    async def run(self):
        self._running = True
        interval = self._config.telemetry_interval_ms / 1000.0
        consumer = asyncio.create_task(self._consume())
        ticker = asyncio.create_task(self._tick(interval))
        await asyncio.gather(consumer, ticker)

    async def _consume(self):
        while self._running:
            try:
                result: PacketResult = await asyncio.wait_for(
                    self._results_q.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            s = result.slice_type
            if s in self._stats:
                self._stats[s].record(result)
                self._active_flows[s].add(result.flow_id)

    async def _tick(self, interval: float):
        while self._running:
            await asyncio.sleep(interval)
            self._update_state()

    def _update_state(self):
        from config import DEFAULT_CONFIG
        slices = {}
        for name, cfg in self._config.slices.items():
            st = self._stats[name]
            slices[name] = SliceMetrics(
                name=name,
                bandwidth_mbps=cfg.bandwidth_mbps,
                latency_ms=st.latency_ms(),
                throughput_mbps=st.throughput_mbps(),
                packet_loss_pct=st.packet_loss_pct(),
                jitter_ms=st.jitter_ms(),
                queue_depth=0,  # filled in by slicer integration
                active_flows=len(self._active_flows[name]),
            )
        total_tp = sum(s.throughput_mbps for s in slices.values())
        self._state = NetworkState(slices=slices, total_throughput_mbps=total_tp)

        # Build feature vector for ML window: aggregate across all slices
        features = self._aggregate_features(slices)
        self._ml_window.append(features)

    def _aggregate_features(self, slices: Dict[str, SliceMetrics]) -> List[float]:
        names = ["CRITICAL", "STANDARD", "BACKGROUND"]
        row = []
        total_bw = max(sum(s.bandwidth_mbps for s in slices.values()), 1.0)
        for n in names:
            s = slices.get(n)
            if s:
                row += [
                    min(s.latency_ms / 100.0, 1.0),
                    min(s.throughput_mbps / 100.0, 1.0),
                    s.packet_loss_pct / 100.0,
                    min(s.jitter_ms / 50.0, 1.0),
                    s.bandwidth_mbps / total_bw,
                ]
            else:
                row += [0.0] * 5
        return row  # 15 features (5 per slice × 3 slices)

    def inject_queue_depths(self, depths: Dict[str, int]):
        if self._state:
            for name, d in depths.items():
                if name in self._state.slices:
                    self._state.slices[name].queue_depth = d

    def stop(self):
        self._running = False
