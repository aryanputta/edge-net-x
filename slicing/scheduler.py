"""
Network slicing scheduler with token-bucket bandwidth enforcement.

Each slice gets an AsyncTokenBucket. Packets consume tokens proportional
to their size. When tokens are exhausted the packet is dropped. Queuing
delay is computed from the virtual queue depth so that simulated latency
responds realistically to slice bandwidth changes.
"""
import asyncio
import time
from collections import deque
from typing import Dict, Optional

from config import SystemConfig, SliceConfig
from models import NetworkPacket, PacketResult, SliceType


class AsyncTokenBucket:
    def __init__(self, rate_mbps: float, burst_factor: float = 2.0):
        self._set_rate(rate_mbps, burst_factor)
        self._tokens: float = self._capacity
        self._last: float = time.monotonic()
        self._lock = asyncio.Lock()

    def _set_rate(self, rate_mbps: float, burst_factor: float = 2.0):
        self._rate = rate_mbps * 125_000.0   # bytes/s
        self._capacity = self._rate * burst_factor

    async def update_rate(self, rate_mbps: float):
        async with self._lock:
            self._set_rate(rate_mbps)
            self._capacity = max(self._capacity, self._tokens)

    async def consume(self, size_bytes: int) -> bool:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens >= size_bytes:
                self._tokens -= size_bytes
                return True
            return False

    @property
    def fill_ratio(self) -> float:
        return self._tokens / self._capacity if self._capacity > 0 else 0.0


class SliceQueue:
    """Priority queue for one network slice."""

    def __init__(self, config: SliceConfig, fault_state: dict):
        self.config = config
        self._fault_state = fault_state
        self.bucket = AsyncTokenBucket(config.bandwidth_mbps)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._results_out: Optional[asyncio.Queue] = None
        self._running = False

        # stats
        self.delivered = 0
        self.dropped = 0
        self.latencies: deque = deque(maxlen=500)

    def attach_results_queue(self, q: asyncio.Queue):
        self._results_out = q

    async def enqueue(self, packet: NetworkPacket):
        try:
            self._queue.put_nowait(packet)
        except asyncio.QueueFull:
            self.dropped += 1
            if self._results_out:
                await self._results_out.put(PacketResult(
                    flow_id=packet.flow_id,
                    slice_type=packet.slice_type,
                    size_bytes=packet.size_bytes,
                    latency_ms=0.0,
                    dropped=True,
                ))

    async def run(self):
        self._running = True
        while self._running:
            try:
                packet: NetworkPacket = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            passed = await self.bucket.consume(packet.size_bytes)
            if not passed:
                self.dropped += 1
                if self._results_out:
                    await self._results_out.put(PacketResult(
                        flow_id=packet.flow_id,
                        slice_type=packet.slice_type,
                        size_bytes=packet.size_bytes,
                        latency_ms=0.0,
                        dropped=True,
                    ))
                continue

            # Simulate propagation + queuing delay
            base_ms = self.config.base_latency_ms
            queue_depth = self._queue.qsize()
            rate_bps = max(self.bucket._rate, 1.0)
            queuing_ms = (queue_depth * packet.size_bytes / rate_bps) * 1000.0

            # Fault injection overlay
            fault_ms = 0.0
            fault = self._fault_state
            if fault.get("active"):
                import random
                if fault.get("type") == "latency_spike":
                    fault_ms = fault.get("magnitude_ms", 0.0)
                elif fault.get("type") == "packet_loss":
                    if random.random() < fault.get("loss_rate", 0.0):
                        self.dropped += 1
                        if self._results_out:
                            await self._results_out.put(PacketResult(
                                flow_id=packet.flow_id,
                                slice_type=packet.slice_type,
                                size_bytes=packet.size_bytes,
                                latency_ms=0.0,
                                dropped=True,
                            ))
                        continue
                elif fault.get("type") == "slowdown":
                    fault_ms = fault.get("slowdown_ms", 0.0)

            total_delay_s = (base_ms + queuing_ms + fault_ms) / 1000.0
            await asyncio.sleep(total_delay_s)

            latency_ms = (time.monotonic() - packet.created_at) * 1000.0
            self.latencies.append(latency_ms)
            self.delivered += 1

            if self._results_out:
                await self._results_out.put(PacketResult(
                    flow_id=packet.flow_id,
                    slice_type=packet.slice_type,
                    size_bytes=packet.size_bytes,
                    latency_ms=latency_ms,
                    dropped=False,
                ))

    def stop(self):
        self._running = False

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()


class NetworkSlicer:
    """Routes packets to the correct slice and manages bandwidth allocation."""

    def __init__(self, config: SystemConfig, fault_state: dict):
        self._config = config
        self._fault_state = fault_state
        self.queues: Dict[str, SliceQueue] = {
            name: SliceQueue(sc, fault_state)
            for name, sc in config.slices.items()
        }
        self._results_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        for q in self.queues.values():
            q.attach_results_queue(self._results_queue)

    async def dispatch(self, packet: NetworkPacket):
        q = self.queues.get(packet.slice_type)
        if q:
            await q.enqueue(packet)

    @property
    def results_queue(self) -> asyncio.Queue:
        return self._results_queue

    async def adjust_bandwidth(self, adjustments: Dict[str, float]):
        """Apply new bandwidth allocations from the decision engine."""
        for name, bw in adjustments.items():
            if name in self.queues:
                sc = self._config.slices[name]
                clamped = max(sc.min_bandwidth_mbps, min(sc.max_bandwidth_mbps, bw))
                sc.bandwidth_mbps = clamped
                await self.queues[name].bucket.update_rate(clamped)

    def current_allocations(self) -> Dict[str, float]:
        return {name: self._config.slices[name].bandwidth_mbps for name in self.queues}

    async def run(self):
        await asyncio.gather(*[q.run() for q in self.queues.values()])
