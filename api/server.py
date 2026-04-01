"""
Async REST control plane API.

Provides runtime access to system state and control actions without
touching the main asyncio event loop. Uses only stdlib — no extra deps.

Endpoints:
  GET  /api/status              full system snapshot (JSON)
  GET  /api/bandwidth           current slice allocations
  POST /api/bandwidth           override allocations {"CRITICAL": 70, ...}
  POST /api/fault/inject/<type> inject fault (?duration=30)
  POST /api/fault/clear         clear active fault
  GET  /api/decisions           last 10 decision actions
  GET  /metrics                 Prometheus text format
"""
import asyncio
import json
import time
import logging
from typing import Callable, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class ControlAPI:
    def __init__(
        self,
        port: int,
        get_state: Callable,
        slicer,
        fault_injector,
        decision_engine,
        prometheus_registry,
    ):
        self._port = port
        self._get_state = get_state
        self._slicer = slicer
        self._fault = fault_injector
        self._decisions = decision_engine
        self._prometheus = prometheus_registry
        self._server = None

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", self._port
        )
        logger.info("Control API listening on http://127.0.0.1:%d", self._port)

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            raw = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        except asyncio.TimeoutError:
            writer.close()
            return

        lines = raw.decode(errors="replace").split("\r\n")
        if not lines:
            writer.close()
            return

        try:
            method, path, _ = lines[0].split(" ", 2)
        except ValueError:
            writer.close()
            return

        parsed = urlparse(path)
        route = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        body_start = raw.find(b"\r\n\r\n")
        body = raw[body_start + 4:] if body_start != -1 else b""

        try:
            status, response = await self._route(method, route, params, body)
        except Exception as e:
            logger.debug("API error: %s", e)
            status, response = 500, {"error": str(e)}

        if isinstance(response, str):
            content_type = "text/plain; version=0.0.4"
            encoded = response.encode()
        else:
            content_type = "application/json"
            encoded = json.dumps(response, default=str).encode()

        header = (
            f"HTTP/1.1 {status} OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(encoded)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()

        try:
            writer.write(header + encoded)
            await writer.drain()
        finally:
            writer.close()

    async def _route(self, method: str, path: str, params: dict, body: bytes):
        state = self._get_state()

        # GET /api/status
        if method == "GET" and path == "/api/status":
            if state is None:
                return 503, {"error": "system not ready"}
            return 200, {
                "timestamp": time.time(),
                "congestion_score": state.congestion_score,
                "total_throughput_mbps": state.total_throughput_mbps,
                "fault": {"active": state.fault_active, "description": state.fault_description},
                "slices": {
                    name: {
                        "bandwidth_mbps": s.bandwidth_mbps,
                        "latency_ms": round(s.latency_ms, 3),
                        "packet_loss_pct": round(s.packet_loss_pct, 3),
                        "throughput_mbps": round(s.throughput_mbps, 3),
                        "jitter_ms": round(s.jitter_ms, 3),
                        "queue_depth": s.queue_depth,
                        "active_flows": s.active_flows,
                    }
                    for name, s in state.slices.items()
                },
            }

        # GET /api/bandwidth
        if method == "GET" and path == "/api/bandwidth":
            alloc = self._slicer.current_allocations()
            return 200, {"allocations_mbps": alloc}

        # POST /api/bandwidth
        if method == "POST" and path == "/api/bandwidth":
            try:
                adjustments = json.loads(body)
            except json.JSONDecodeError:
                return 400, {"error": "invalid JSON body"}
            valid_slices = {"CRITICAL", "STANDARD", "BACKGROUND"}
            bad = set(adjustments) - valid_slices
            if bad:
                return 400, {"error": f"unknown slices: {bad}"}
            await self._slicer.adjust_bandwidth(adjustments)
            return 200, {"ok": True, "applied": adjustments}

        # POST /api/fault/inject/<type>
        if method == "POST" and path.startswith("/api/fault/inject/"):
            fault_type = path.split("/")[-1]
            duration = float(params.get("duration", [0])[0]) or None
            self._fault.inject(fault_type, duration_s=duration)
            return 200, {"ok": True, "fault": fault_type, "duration_s": duration}

        # POST /api/fault/clear
        if method == "POST" and path == "/api/fault/clear":
            self._fault.clear()
            return 200, {"ok": True}

        # GET /api/decisions
        if method == "GET" and path == "/api/decisions":
            actions = []
            for a in list(self._decisions.actions)[:10]:
                actions.append({
                    "timestamp": a.timestamp,
                    "trigger": a.trigger,
                    "congestion_score": round(a.congestion_score, 4),
                    "adjustments": a.adjustments,
                    "description": a.description,
                    "rolled_back": a.rolled_back,
                })
            return 200, {"decisions": actions}

        # GET /metrics (Prometheus)
        if method == "GET" and path == "/metrics":
            return 200, self._prometheus.render(state)

        return 404, {"error": "not found"}
