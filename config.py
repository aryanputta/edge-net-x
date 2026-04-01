from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SliceConfig:
    name: str
    bandwidth_mbps: float
    latency_budget_ms: float
    priority: int
    min_bandwidth_mbps: float
    max_bandwidth_mbps: float
    base_latency_ms: float


@dataclass
class SystemConfig:
    total_bandwidth_mbps: float = 100.0
    telemetry_interval_ms: float = 100.0
    ml_window_size: int = 50
    decision_cooldown_s: float = 3.0
    congestion_threshold: float = 0.70
    hysteresis_threshold: float = 0.50
    hysteresis_count: int = 3
    rollback_window_s: float = 5.0
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 7000
    udp_port: int = 7001
    api_port: int = 8080
    dashboard_refresh_hz: float = 5.0

    slices: Dict[str, SliceConfig] = field(default_factory=lambda: {
        "CRITICAL": SliceConfig("CRITICAL", 60.0, 2.0, 3, 30.0, 80.0, 1.0),
        "STANDARD": SliceConfig("STANDARD", 30.0, 10.0, 2, 10.0, 50.0, 5.0),
        "BACKGROUND": SliceConfig("BACKGROUND", 10.0, 50.0, 1, 5.0, 20.0, 20.0),
    })


DEFAULT_CONFIG = SystemConfig()
