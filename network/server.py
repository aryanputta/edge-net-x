"""
Async TCP + UDP server — real binary payload transport.

Each TCP connection maps to one flow. The server reads the 20-byte binary
header, then reads exactly payload_len bytes of payload (real data moving
through the socket), dispatches a PacketResult to the slicer, and sends an
8-byte ACK with the RTT-relevant sequence number.

UDP datagrams are self-contained: header + payload in one datagram.

This means all throughput metrics are derived from bytes that actually
traverse the socket — no simulation of "hypothetical" packet sizes.
"""
import asyncio
import struct
import time
import logging

from network.protocol import (
    HEADER_SIZE, ACK_SIZE, MAX_UDP_PAYLOAD,
    decode_header, encode_ack,
    PROTO_TCP, PROTO_UDP,
)
from models import NetworkPacket, SliceType, Protocol

logger = logging.getLogger(__name__)

# Maximum payload we'll accept — guards against malformed frames
_MAX_PAYLOAD = 64 * 1024   # 64 KB


class TCPSessionProtocol(asyncio.Protocol):
    """One instance per accepted TCP connection (one flow)."""

    def __init__(self, slicer, stats: dict):
        self._slicer = slicer
        self._stats = stats
        self._buf = bytearray()
        self._transport = None
        self._tasks = []

    def connection_made(self, transport):
        self._transport = transport
        transport.set_write_buffer_limits(high=256 * 1024, low=64 * 1024)
        self._stats["tcp_connections"] = self._stats.get("tcp_connections", 0) + 1

    def data_received(self, data: bytes):
        self._buf.extend(data)
        self._drain_buffer()

    def _drain_buffer(self):
        while True:
            if len(self._buf) < HEADER_SIZE:
                break
            hdr = decode_header(bytes(self._buf[:HEADER_SIZE]))
            if hdr.payload_len > _MAX_PAYLOAD:
                logger.debug("Oversized payload %d — dropping connection", hdr.payload_len)
                if self._transport:
                    self._transport.close()
                return
            total = HEADER_SIZE + hdr.payload_len
            if len(self._buf) < total:
                break
            # consume payload (real bytes)
            payload = bytes(self._buf[HEADER_SIZE:total])
            del self._buf[:total]
            task = asyncio.ensure_future(self._handle(hdr, payload))
            self._tasks.append(task)
            task.add_done_callback(self._tasks.remove)

    async def _handle(self, hdr, payload: bytes):
        latency_ms = (time.monotonic() - hdr.timestamp) * 1000.0
        pkt = NetworkPacket(
            flow_id=f"tcp-{hdr.slice_type}-{id(self)}",
            slice_type=SliceType(hdr.slice_type),
            protocol=Protocol.TCP,
            size_bytes=len(payload),
            sequence_num=hdr.seq,
            created_at=hdr.timestamp,
        )
        await self._slicer.dispatch(pkt)
        ack = encode_ack(hdr.seq)
        if self._transport and not self._transport.is_closing():
            self._transport.write(ack)

    def connection_lost(self, exc):
        self._stats["tcp_connections"] = max(0, self._stats.get("tcp_connections", 1) - 1)


class UDPServerProtocol(asyncio.DatagramProtocol):
    def __init__(self, slicer, stats: dict):
        self._slicer = slicer
        self._stats = stats
        self._transport = None

    def connection_made(self, transport):
        self._transport = transport

    def datagram_received(self, data: bytes, addr):
        if len(data) < HEADER_SIZE:
            return
        hdr = decode_header(data[:HEADER_SIZE])
        payload = data[HEADER_SIZE:HEADER_SIZE + hdr.payload_len]
        asyncio.ensure_future(self._handle(hdr, payload, addr))

    async def _handle(self, hdr, payload: bytes, addr):
        pkt = NetworkPacket(
            flow_id=f"udp-{hdr.slice_type}-{addr[1]}",
            slice_type=SliceType(hdr.slice_type),
            protocol=Protocol.UDP,
            size_bytes=max(len(payload), 1),
            sequence_num=hdr.seq,
            created_at=hdr.timestamp,
        )
        await self._slicer.dispatch(pkt)
        ack = encode_ack(hdr.seq)
        if self._transport:
            self._transport.sendto(ack, addr)

    def error_received(self, exc):
        pass


class EdgeNetServer:
    def __init__(self, slicer, host: str, tcp_port: int, udp_port: int):
        self._slicer = slicer
        self._host = host
        self._tcp_port = tcp_port
        self._udp_port = udp_port
        self.stats: dict = {"tcp_connections": 0}
        self._tcp_server = None
        self._udp_transport = None

    async def start(self):
        loop = asyncio.get_event_loop()
        self._tcp_server = await loop.create_server(
            lambda: TCPSessionProtocol(self._slicer, self.stats),
            self._host, self._tcp_port,
            reuse_address=True,
        )
        self._udp_transport, _ = await loop.create_datagram_endpoint(
            lambda: UDPServerProtocol(self._slicer, self.stats),
            local_addr=(self._host, self._udp_port),
        )
        logger.info("EdgeNetServer TCP:%d UDP:%d", self._tcp_port, self._udp_port)

    async def stop(self):
        if self._tcp_server:
            self._tcp_server.close()
            await self._tcp_server.wait_closed()
        if self._udp_transport:
            self._udp_transport.close()
