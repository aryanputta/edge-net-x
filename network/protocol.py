"""
Wire protocol for EDGE-NET-X traffic.

TCP frame:
  [4B seq][1B slice_type][1B protocol][2B reserved][4B payload_len][8B timestamp]
  followed by payload_len bytes of payload data.

  Total header: 20 bytes.

ACK frame (server → client):
  [4B seq][4B status]   (status=0 OK, status=1 dropped)

UDP datagram:
  Same 20-byte header, payload inline in the same datagram.
  Max payload capped at 1380 bytes to stay under Ethernet MTU without fragmentation.

Slice encoding:
  0 = CRITICAL, 1 = STANDARD, 2 = BACKGROUND

All multi-byte integers are big-endian.
"""
import struct
import time
from typing import NamedTuple

HEADER_FMT = ">IBBHI d"   # seq(4) slice(1) proto(1) reserved(2) length(4) ts(8) = 20 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # should be 20
ACK_FMT = ">II"
ACK_SIZE = struct.calcsize(ACK_FMT)         # 8 bytes
MAX_UDP_PAYLOAD = 1380

SLICE_ENCODE = {"CRITICAL": 0, "STANDARD": 1, "BACKGROUND": 2}
SLICE_DECODE = {v: k for k, v in SLICE_ENCODE.items()}
PROTO_TCP = 0
PROTO_UDP = 1


class FrameHeader(NamedTuple):
    seq: int
    slice_type: str
    protocol: int
    payload_len: int
    timestamp: float


def encode_header(seq: int, slice_type: str, protocol: int, payload_len: int) -> bytes:
    return struct.pack(
        HEADER_FMT,
        seq,
        SLICE_ENCODE[slice_type],
        protocol,
        0,
        payload_len,
        time.monotonic(),
    )


def decode_header(data: bytes) -> FrameHeader:
    seq, slice_enc, proto, _, length, ts = struct.unpack(HEADER_FMT, data)
    return FrameHeader(
        seq=seq,
        slice_type=SLICE_DECODE.get(slice_enc, "STANDARD"),
        protocol=proto,
        payload_len=length,
        timestamp=ts,
    )


def encode_ack(seq: int, dropped: bool = False) -> bytes:
    return struct.pack(ACK_FMT, seq, 1 if dropped else 0)


def decode_ack(data: bytes):
    return struct.unpack(ACK_FMT, data)
