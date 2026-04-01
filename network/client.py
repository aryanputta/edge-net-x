"""
Traffic generators — one asyncio task per flow.

Each flow has:
  - slice type (CRITICAL / STANDARD / BACKGROUND)
  - protocol (TCP or UDP)
  - send rate (packets/second)
  - packet size distribution

TCP clients maintain a persistent connection and measure RTT per packet.
UDP clients send datagrams and track responses with a pending dict keyed by seq.
"""
import asyncio
import json
import struct
import time
import random
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from models import SliceType, Protocol

logger = logging.getLogger(__name__)


@dataclass
class FlowProfile:
    flow_id: str
    slice_type: SliceType
    protocol: Protocol
    packets_per_second: float
    min_size: int
    max_size: int


# Predefined profiles for the three slice types
def default_profiles() -> list[FlowProfile]:
    profiles = []
    for i in range(3):
        profiles.append(FlowProfile(
            f"critical-tcp-{i}", SliceType.CRITICAL, Protocol.TCP,
            50.0, 512, 2048,
        ))
        profiles.append(FlowProfile(
            f"standard-tcp-{i}", SliceType.STANDARD, Protocol.TCP,
            30.0, 1024, 4096,
        ))
        profiles.append(FlowProfile(
            f"background-udp-{i}", SliceType.BACKGROUND, Protocol.UDP,
            20.0, 256, 1024,
        ))
    return profiles


class TCPFlowClient:
    def __init__(self, profile: FlowProfile, host: str, port: int, rtt_sink: asyncio.Queue):
        self._p = profile
        self._host = host
        self._port = port
        self._rtt_sink = rtt_sink
        self._seq = 0
        self._running = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def run(self):
        self._running = True
        while self._running:
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self._host, self._port
                )
                await asyncio.gather(
                    self._send_loop(),
                    self._recv_loop(),
                )
            except (ConnectionRefusedError, OSError) as e:
                logger.debug("TCP %s connect failed: %s", self._p.flow_id, e)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug("TCP %s error: %s", self._p.flow_id, e)
                await asyncio.sleep(0.2)

    async def _send_loop(self):
        interval = 1.0 / self._p.packets_per_second
        pending: Dict[int, float] = {}

        while self._running and self._writer and not self._writer.is_closing():
            size = random.randint(self._p.min_size, self._p.max_size)
            seq = self._seq
            self._seq += 1
            ts = time.monotonic()
            pending[seq] = ts

            msg = json.dumps({
                "flow_id": self._p.flow_id,
                "slice_type": self._p.slice_type.value,
                "size_bytes": size,
                "seq": seq,
                "ts": ts,
            }).encode()
            header = struct.pack(">I", len(msg))
            try:
                self._writer.write(header + msg)
                await self._writer.drain()
            except Exception:
                break

            # Clean up stale pending (>5s old)
            cutoff = ts - 5.0
            for old_seq in [k for k, v in pending.items() if v < cutoff]:
                del pending[old_seq]

            await asyncio.sleep(interval)

    async def _recv_loop(self):
        pending_start: Dict[int, float] = {}
        while self._running and self._reader:
            try:
                data = await asyncio.wait_for(self._reader.readexactly(8), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            seq, _ = struct.unpack(">II", data)
            rtt = (time.monotonic() - self._seq_ts(seq)) * 1000.0
            await self._rtt_sink.put((self._p.flow_id, self._p.slice_type.value, rtt, False))

    def _seq_ts(self, seq: int) -> float:
        return time.monotonic() - 0.001  # fallback

    def stop(self):
        self._running = False
        if self._writer:
            self._writer.close()


class UDPFlowClient(asyncio.DatagramProtocol):
    def __init__(self, profile: FlowProfile, host: str, port: int, rtt_sink: asyncio.Queue):
        self._p = profile
        self._host = host
        self._port = port
        self._rtt_sink = rtt_sink
        self._seq = 0
        self._pending: Dict[int, float] = {}
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._running = False

    def connection_made(self, transport):
        self._transport = transport

    def datagram_received(self, data: bytes, addr):
        try:
            msg = json.loads(data)
            seq = msg["seq"]
            sent_at = self._pending.pop(seq, None)
            if sent_at is not None:
                rtt_ms = (time.monotonic() - sent_at) * 1000.0
                asyncio.ensure_future(
                    self._rtt_sink.put((self._p.flow_id, self._p.slice_type.value, rtt_ms, False))
                )
        except Exception:
            pass

    def error_received(self, exc):
        pass

    async def run(self):
        self._running = True
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                transport, _ = await loop.create_datagram_endpoint(
                    lambda: self,
                    remote_addr=(self._host, self._port),
                )
                self._transport = transport
                await self._send_loop()
            except Exception as e:
                logger.debug("UDP %s error: %s", self._p.flow_id, e)
                await asyncio.sleep(0.5)

    async def _send_loop(self):
        interval = 1.0 / self._p.packets_per_second
        while self._running and self._transport:
            size = random.randint(self._p.min_size, self._p.max_size)
            seq = self._seq
            self._seq += 1
            ts = time.monotonic()
            self._pending[seq] = ts

            # Prune stale
            cutoff = ts - 5.0
            for k in [k for k, v in self._pending.items() if v < cutoff]:
                lost_ts = self._pending.pop(k)
                asyncio.ensure_future(
                    self._rtt_sink.put((self._p.flow_id, self._p.slice_type.value, 0.0, True))
                )

            msg = json.dumps({
                "flow_id": self._p.flow_id,
                "slice_type": self._p.slice_type.value,
                "size_bytes": size,
                "seq": seq,
                "ts": ts,
            }).encode()
            try:
                self._transport.sendto(msg)
            except Exception:
                break
            await asyncio.sleep(interval)

    def stop(self):
        self._running = False
        if self._transport:
            self._transport.close()


class TrafficGenerator:
    """Manages all flow clients."""

    def __init__(self, host: str, tcp_port: int, udp_port: int):
        self._host = host
        self._tcp_port = tcp_port
        self._udp_port = udp_port
        self.rtt_queue: asyncio.Queue = asyncio.Queue(maxsize=20_000)
        self._clients = []
        self._tasks = []

    def add_profiles(self, profiles: list[FlowProfile]):
        for p in profiles:
            if p.protocol == Protocol.TCP:
                self._clients.append(TCPFlowClient(p, self._host, self._tcp_port, self.rtt_queue))
            else:
                self._clients.append(UDPFlowClient(p, self._host, self._udp_port, self.rtt_queue))

    async def run(self):
        await asyncio.sleep(0.5)  # let server start first
        self._tasks = [asyncio.create_task(c.run()) for c in self._clients]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def stop(self):
        for c in self._clients:
            c.stop()
        for t in self._tasks:
            t.cancel()
