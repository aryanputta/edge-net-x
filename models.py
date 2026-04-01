from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time


class SliceType(str, Enum):
    CRITICAL = "CRITICAL"
    STANDARD = "STANDARD"
    BACKGROUND = "BACKGROUND"


class Protocol(str, Enum):
    TCP = "TCP"
    UDP = "UDP"


@dataclass
class NetworkPacket:
    flow_id: str
    slice_type: SliceType
    protocol: Protocol
    size_bytes: int
    sequence_num: int
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class PacketResult:
    flow_id: str
    slice_type: str
    size_bytes: int
    latency_ms: float
    dropped: bool
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class SliceMetrics:
    name: str
    bandwidth_mbps: float
    latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    packet_loss_pct: float = 0.0
    jitter_ms: float = 0.0
    queue_depth: int = 0
    active_flows: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0


@dataclass
class NetworkState:
    slices: Dict[str, SliceMetrics]
    congestion_score: float = 0.0
    total_throughput_mbps: float = 0.0
    fault_active: bool = False
    fault_description: str = ""
    timestamp: float = field(default_factory=time.monotonic)

    def feature_vector(self, slice_name: str) -> List[float]:
        s = self.slices.get(slice_name)
        if s is None:
            return [0.0] * 6
        total_bw = max(sum(x.bandwidth_mbps for x in self.slices.values()), 1.0)
        return [
            min(s.latency_ms / 100.0, 1.0),
            min(s.throughput_mbps / 100.0, 1.0),
            s.packet_loss_pct / 100.0,
            min(s.jitter_ms / 50.0, 1.0),
            min(s.queue_depth / 1000.0, 1.0),
            s.bandwidth_mbps / total_bw,
        ]


@dataclass
class DecisionAction:
    timestamp: float
    trigger: str
    congestion_score: float
    adjustments: Dict[str, float]
    description: str
    rolled_back: bool = False
    baseline_metrics: Optional[Dict[str, float]] = None
