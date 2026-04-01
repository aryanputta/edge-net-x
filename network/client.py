"""
Real-socket traffic generators.

Each flow client sends binary frames with actual payload bytes — measured
throughput is real bytes/second on the socket, not a synthetic number.

TCP clients:
  - Persistent connection per flow
  - Sends 20-byte header + payload, waits for 8-byte ACK
  - RTT = time from header write to ACK receipt (real socket round-trip)
  - TCP_NODELAY enabled on CRITICAL flows to minimize Nagle buffering

UDP clients:
  - Stateless datagrams, max 1380 bytes payload per datagram
  - Seq-tracked with 2s expiry for loss detection
  - No connection teardown overhead

Traffic patterns:
  - CONSTANT: fixed inter-packet interval
  - POISSON: exponential inter-arrival (models real traffic burstiness)
  - BURST: large bursts every N seconds with silence between
"""
import asyncio
import os
import random
import socket
import struct
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from models import SliceType, Protocol
from network.protocol import (
    HEADER_SIZE, ACK_SIZE, MAX_UDP_PAYLOAD,
    encode_header, decode_ack, PROTO_TCP, PROTO_UDP,
)

logger = logging.getLogger(__name__)


class TrafficPattern(str, Enum):
    CONSTANT = "constant"
    POISSON  = "poisson"
    BURST    = "burst"


@dataclass
class FlowProfile:
    flow_id: str
    slice_type: SliceType
    protocol: Protocol
    packets_per_second: float
    min_size: int
    max_size: int
    pattern: TrafficPattern = TrafficPattern.CONSTANT
    # BURST mode: send `burst_count` packets, then sleep `burst_interval_s`
    burst_count: int = 10
    burst_interval_s: float = 1.0


def default_profiles() -> list[FlowProfile]:
    profiles = []
    for i in range(3):
        profiles.append(FlowProfile(
            f"critical-tcp-{i}", SliceType.CRITICAL, Protocol.TCP,
            50.0, 512, 2048, TrafficPattern.CONSTANT,
        ))
        profiles.append(FlowProfile(
            f"standard-tcp-{i}", SliceType.STANDARD, Protocol.TCP,
            30.0, 1024, 4096, TrafficPattern.POISSON,
        ))
        profiles.append(FlowProfile(
            f"background-udp-{i}", SliceType.BACKGROUND, Protocol.UDP,
            20.0, 256, min(1380, 1024), TrafficPattern.BURST,
            burst_count=15, burst_interval_s=1.5,
        ))
    return profiles


def _next_interval(profile: FlowProfile) -> float:
    base = 1.0 / profile.packets_per_second
    if profile.pattern == TrafficPattern.CONSTANT:
        return base
    elif profile.pattern == TrafficPattern.POISSON:
        return random.expovariate(profile.packets_per_second)
    return base  # BURST handles its own pacing


def _make_payload(size: int) -> bytes:
    # os.urandom is realistic (non-compressible) but slow at scale;
    # use a repeating pattern with random salt for speed
    salt = os.urandom(16)
    repeats = (size + 15) // 16
    return (salt * repeats)[:size]


class TCPFlowClient:
    """Persistent TCP connection for one flow."""

    def __init__(self, profile: FlowProfile, host: str, port: int, rtt_sink: asyncio.Queue):
        self._p = profile
        self._host = host
        self._port = port
        self._rtt_sink = rtt_sink
        self._seq = 0
        self._pending: Dict[int, float] = {}   # seq -> send_ts (shared between loops)
        self._running = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def run(self):
        self._running = True
        while self._running:
            self._pending.clear()
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self._host, self._port,
                )
                # TCP_NODELAY on CRITICAL: avoid Nagle buffering, minimize latency
                if self._p.slice_type == SliceType.CRITICAL:
                    sock = self._writer.get_extra_info("socket")
                    if sock:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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
        p = self._p
        if p.pattern == TrafficPattern.BURST:
            await self._burst_send_loop()
            return

        while self._running and self._writer and not self._writer.is_closing():
            size = random.randint(p.min_size, p.max_size)
            seq = self._seq
            self._seq += 1
            payload = _make_payload(size)
            header = encode_header(seq, p.slice_type.value, PROTO_TCP, size)

            ts = time.monotonic()
            self._pending[seq] = ts

            try:
                self._writer.write(header + payload)
                await self._writer.drain()
            except Exception:
                break

            # Prune stale (>5s)
            cutoff = ts - 5.0
            for k in [k for k, v in self._pending.items() if v < cutoff]:
                del self._pending[k]
                await self._rtt_sink.put((p.flow_id, p.slice_type.value, 0.0, True))

            await asyncio.sleep(_next_interval(p))

    async def _burst_send_loop(self):
        p = self._p
        while self._running and self._writer and not self._writer.is_closing():
            for _ in range(p.burst_count):
                if not self._running:
                    return
                size = random.randint(p.min_size, p.max_size)
                seq = self._seq
                self._seq += 1
                payload = _make_payload(size)
                header = encode_header(seq, p.slice_type.value, PROTO_TCP, size)
                ts = time.monotonic()
                self._pending[seq] = ts
                try:
                    self._writer.write(header + payload)
                    await self._writer.drain()
                except Exception:
                    return
                await asyncio.sleep(0.002)  # 2ms between burst packets
            await asyncio.sleep(p.burst_interval_s)

    async def _recv_loop(self):
        while self._running and self._reader:
            try:
                data = await asyncio.wait_for(self._reader.readexactly(ACK_SIZE), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            seq, status = decode_ack(data)
            sent_at = self._pending.pop(seq, None)
            if sent_at is not None:
                rtt_ms = (time.monotonic() - sent_at) * 1000.0
                dropped = status != 0
                await self._rtt_sink.put((self._p.flow_id, self._p.slice_type.value, rtt_ms, dropped))

    def stop(self):
        self._running = False
        if self._writer:
            self._writer.close()


class UDPFlowClient:
    """Stateless UDP flow — datagram per packet with loss detection."""

    def __init__(self, profile: FlowProfile, host: str, port: int, rtt_sink: asyncio.Queue):
        self._p = profile
        self._host = host
        self._port = port
        self._rtt_sink = rtt_sink
        self._seq = 0
        self._pending: Dict[int, float] = {}
        self._running = False
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_UDPClientProtocol] = None

    async def run(self):
        self._running = True
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                proto = _UDPClientProtocol(self._pending, self._rtt_sink, self._p)
                transport, _ = await loop.create_datagram_endpoint(
                    lambda: proto,
                    remote_addr=(self._host, self._port),
                )
                self._transport = transport
                self._protocol = proto
                await self._send_loop()
            except Exception as e:
                logger.debug("UDP %s error: %s", self._p.flow_id, e)
                await asyncio.sleep(0.5)

    async def _send_loop(self):
        p = self._p
        while self._running and self._transport:
            # Cap payload to max UDP size
            size = min(random.randint(p.min_size, p.max_size), MAX_UDP_PAYLOAD)
            seq = self._seq
            self._seq += 1
            payload = _make_payload(size)
            header = encode_header(seq, p.slice_type.value, PROTO_UDP, size)
            ts = time.monotonic()
            self._pending[seq] = ts

            # Loss detection: prune entries older than 2s
            cutoff = ts - 2.0
            for k in [k for k, v in self._pending.items() if v < cutoff]:
                del self._pending[k]
                asyncio.ensure_future(
                    self._rtt_sink.put((p.flow_id, p.slice_type.value, 0.0, True))
                )

            try:
                self._transport.sendto(header + payload)
            except Exception:
                break

            if p.pattern == TrafficPattern.BURST:
                for _ in range(p.burst_count - 1):
                    await asyncio.sleep(0.002)
                    size2 = min(random.randint(p.min_size, p.max_size), MAX_UDP_PAYLOAD)
                    seq2 = self._seq
                    self._seq += 1
                    payload2 = _make_payload(size2)
                    h2 = encode_header(seq2, p.slice_type.value, PROTO_UDP, size2)
                    self._pending[seq2] = time.monotonic()
                    try:
                        self._transport.sendto(h2 + payload2)
                    except Exception:
                        break
                await asyncio.sleep(p.burst_interval_s)
            else:
                await asyncio.sleep(_next_interval(p))

    def stop(self):
        self._running = False
        if self._transport:
            self._transport.close()


class _UDPClientProtocol(asyncio.DatagramProtocol):
    def __init__(self, pending: dict, rtt_sink: asyncio.Queue, profile: FlowProfile):
        self._pending = pending
        self._rtt_sink = rtt_sink
        self._p = profile

    def datagram_received(self, data: bytes, addr):
        if len(data) < ACK_SIZE:
            return
        seq, status = decode_ack(data)
        sent_at = self._pending.pop(seq, None)
        if sent_at is not None:
            rtt_ms = (time.monotonic() - sent_at) * 1000.0
            dropped = status != 0
            asyncio.ensure_future(
                self._rtt_sink.put((self._p.flow_id, self._p.slice_type.value, rtt_ms, dropped))
            )

    def error_received(self, exc):
        pass

    def connection_lost(self, exc):
        pass


class TrafficGenerator:
    def __init__(self, host: str, tcp_port: int, udp_port: int):
        self._host = host
        self._tcp_port = tcp_port
        self._udp_port = udp_port
        self.rtt_queue: asyncio.Queue = asyncio.Queue(maxsize=50_000)
        self._clients = []
        self._tasks = []

    def add_profiles(self, profiles: list[FlowProfile]):
        for p in profiles:
            if p.protocol == Protocol.TCP:
                self._clients.append(TCPFlowClient(p, self._host, self._tcp_port, self.rtt_queue))
            else:
                self._clients.append(UDPFlowClient(p, self._host, self._udp_port, self.rtt_queue))

    async def run(self):
        await asyncio.sleep(0.5)
        self._tasks = [asyncio.create_task(c.run()) for c in self._clients]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def stop(self):
        for c in self._clients:
            c.stop()
        for t in self._tasks:
            t.cancel()
