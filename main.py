"""
EDGE-NET-X PRO — main entry point.

Wires all subsystems together and runs the asyncio event loop.

Usage:
    python main.py [--no-demo] [--inject latency_spike|packet_loss|slowdown]
                   [--api-port 8080]
"""
import asyncio
import argparse
import logging
import sys
import time
import os

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)

sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_CONFIG
from models import NetworkState
from network.server import EdgeNetServer
from network.client import TrafficGenerator, default_profiles
from telemetry.collector import TelemetryCollector
from ml.inference import build_inference_engine
from decision.engine import DecisionEngine
from slicing.scheduler import NetworkSlicer
from simulation.fault import FaultInjector
from simulation.traffic import TrafficScenario
from dashboard.display import Dashboard
from api.server import ControlAPI
from metrics.prometheus import PrometheusRegistry


async def ml_inference_loop(engine, telemetry, decision_engine):
    """Runs every 500 ms — infers congestion score and pushes to decision engine."""
    while True:
        await asyncio.sleep(0.5)
        window = telemetry.ml_window
        if len(window) < 5:
            continue
        score = await engine.infer_single(window)
        decision_engine.update_congestion_score(score)


async def main(run_demo: bool, manual_fault: str | None, api_port: int):
    cfg = DEFAULT_CONFIG
    events: list = []
    start_time = time.time()

    print("Initialising EDGE-NET-X PRO...")

    # ── Shared fault state ───────────────────────────────────────────────────
    fault_state: dict = {"active": False}
    fault_injector = FaultInjector(fault_state, events)

    # ── Network slicer ───────────────────────────────────────────────────────
    slicer = NetworkSlicer(cfg, fault_state)

    # ── Telemetry ────────────────────────────────────────────────────────────
    telemetry = TelemetryCollector(cfg, slicer.results_queue)

    # ── Decision engine ──────────────────────────────────────────────────────
    decision_engine = DecisionEngine(cfg, slicer, telemetry, events)

    # ── ML inference ─────────────────────────────────────────────────────────
    inference_engine, device_desc = build_inference_engine(seq_len=cfg.ml_window_size)

    # ── Prometheus metrics ───────────────────────────────────────────────────
    prometheus = PrometheusRegistry(
        sla_budgets={name: sc.latency_budget_ms for name, sc in cfg.slices.items()},
        inference_engine=inference_engine,
        decision_engine=decision_engine,
    )

    # ── Network server ───────────────────────────────────────────────────────
    server = EdgeNetServer(slicer, cfg.tcp_host, cfg.tcp_port, cfg.udp_port)
    await server.start()

    # ── REST control API ─────────────────────────────────────────────────────
    api = ControlAPI(
        port=api_port,
        get_state=lambda: telemetry.current_state,
        slicer=slicer,
        fault_injector=fault_injector,
        decision_engine=decision_engine,
        prometheus_registry=prometheus,
    )
    await api.start()

    # ── Traffic generator ────────────────────────────────────────────────────
    generator = TrafficGenerator(cfg.tcp_host, cfg.tcp_port, cfg.udp_port)
    generator.add_profiles(default_profiles())

    # ── Dashboard ────────────────────────────────────────────────────────────
    dashboard = Dashboard(refresh_hz=cfg.dashboard_refresh_hz)

    if manual_fault:
        async def inject_after_delay():
            await asyncio.sleep(10.0)
            fault_injector.inject(manual_fault, duration_s=30.0)
        asyncio.create_task(inject_after_delay())

    scenario = TrafficScenario(fault_injector, events)

    print(f"ML device:   {device_desc}")
    print(f"Control API: http://127.0.0.1:{api_port}/api/status")
    print(f"Metrics:     http://127.0.0.1:{api_port}/metrics")
    print("Starting all subsystems...")

    try:
        await asyncio.gather(
            slicer.run(),
            telemetry.run(),
            generator.run(),
            decision_engine.run(),
            ml_inference_loop(inference_engine, telemetry, decision_engine),
            dashboard.run(
                get_state=lambda: telemetry.current_state,
                decision_engine=decision_engine,
                events=events,
                fault_injector=fault_injector,
                inference_engine=inference_engine,
                start_time=start_time,
            ),
            *([scenario.run_demo()] if run_demo else []),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        generator.stop()
        telemetry.stop()
        decision_engine.stop()
        dashboard.stop()
        inference_engine.shutdown()
        await server.stop()
        await api.stop()
        print("\nEDGE-NET-X PRO stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDGE-NET-X PRO")
    parser.add_argument("--no-demo", action="store_true", help="Skip demo scenario")
    parser.add_argument(
        "--inject", metavar="FAULT",
        choices=["latency_spike", "packet_loss", "slowdown"],
        help="Inject a specific fault 10s after startup",
    )
    parser.add_argument("--api-port", type=int, default=8080, help="REST API port")
    args = parser.parse_args()

    try:
        asyncio.run(main(
            run_demo=not args.no_demo,
            manual_fault=args.inject,
            api_port=args.api_port,
        ))
    except KeyboardInterrupt:
        pass
