"""
Async TCP + UDP server.

Clients send JSON-encoded packet descriptors. The server parses them,
constructs NetworkPacket objects, dispatches to the slicer, then sends
back an ACK with the assigned sequence number so clients can measure RTT.

TCP uses length-prefixed framing (4-byte big-endian length header).
UDP is stateless — each datagram is one packet descriptor.
"""
import asyncio
import json
import struct
import logging

from models import NetworkPacket, SliceType, Protocol

logger = logging.getLogger(__name__)


class TCPServerProtocol(asyncio.Protocol):
    def __init__(self, slicer, stats: dict):
        self._slicer = slicer
        self._stats = stats
        self._buf = b""
        self._transport = None

    def connection_made(self, transport):
        self._transport = transport
        self._stats["tcp_connections"] = self._stats.get("tcp_connections", 0) + 1

    def data_received(self, data: bytes):
        self._buf += data
        while len(self._buf) >= 4:
            length = struct.unpack(">I", self._buf[:4])[0]
            if len(self._buf) < 4 + length:
                break
            payload = self._buf[4:4 + length]
            self._buf = self._buf[4 + length:]
            asyncio.ensure_future(self._handle(payload))

    async def _handle(self, payload: bytes):
        try:
            msg = json.loads(payload)
            pkt = NetworkPacket(
                flow_id=msg["flow_id"],
                slice_type=SliceType(msg["slice_type"]),
                protocol=Protocol.TCP,
                size_bytes=msg["size_bytes"],
                sequence_num=msg["seq"],
                created_at=msg["ts"],
            )
            await self._slicer.dispatch(pkt)
            ack = struct.pack(">II", msg["seq"], 0)
            if self._transport and not self._transport.is_closing():
                self._transport.write(ack)
        except Exception as e:
            logger.debug("TCP handle error: %s", e)

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
        asyncio.ensure_future(self._handle(data, addr))

    async def _handle(self, data: bytes, addr):
        try:
            msg = json.loads(data)
            pkt = NetworkPacket(
                flow_id=msg["flow_id"],
                slice_type=SliceType(msg["slice_type"]),
                protocol=Protocol.UDP,
                size_bytes=msg["size_bytes"],
                sequence_num=msg["seq"],
                created_at=msg["ts"],
            )
            await self._slicer.dispatch(pkt)
            ack = json.dumps({"seq": msg["seq"]}).encode()
            if self._transport:
                self._transport.sendto(ack, addr)
        except Exception as e:
            logger.debug("UDP handle error: %s", e)


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
            lambda: TCPServerProtocol(self._slicer, self.stats),
            self._host,
            self._tcp_port,
        )

        self._udp_transport, _ = await loop.create_datagram_endpoint(
            lambda: UDPServerProtocol(self._slicer, self.stats),
            local_addr=(self._host, self._udp_port),
        )
        logger.info("EdgeNetServer listening TCP:%d UDP:%d", self._tcp_port, self._udp_port)

    async def stop(self):
        if self._tcp_server:
            self._tcp_server.close()
            await self._tcp_server.wait_closed()
        if self._udp_transport:
            self._udp_transport.close()
