"""
EDGE-NET-X PRO — main entry point.

Wires all subsystems together and runs the asyncio event loop.

Usage:
    python main.py [--no-demo] [--inject latency_spike|packet_loss|slowdown]
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

# ── add repo root to path ────────────────────────────────────────────────────
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


async def ml_inference_loop(engine, telemetry, decision_engine):
    """Runs every 500 ms — infers congestion score and pushes to decision engine."""
    while True:
        await asyncio.sleep(0.5)
        window = telemetry.ml_window
        if len(window) < 5:
            continue
        score = await engine.infer_single(window)
        decision_engine.update_congestion_score(score)


async def main(run_demo: bool, manual_fault: str | None):
    cfg = DEFAULT_CONFIG
    events: list = []
    start_time = time.time()

    print("Initialising EDGE-NET-X PRO...")

    # ── Shared fault state (dict passed by reference to SliceQueues) ─────────
    fault_state: dict = {"active": False}
    fault_injector = FaultInjector(fault_state, events)

    # ── Network slicer (bandwidth enforcement) ───────────────────────────────
    slicer = NetworkSlicer(cfg, fault_state)

    # ── Telemetry collector ──────────────────────────────────────────────────
    telemetry = TelemetryCollector(cfg, slicer.results_queue)

    # ── Decision engine ──────────────────────────────────────────────────────
    decision_engine = DecisionEngine(cfg, slicer, telemetry)

    # ── ML inference engine (trains synchronously before loop starts) ────────
    inference_engine, device_desc = build_inference_engine(seq_len=cfg.ml_window_size)

    # ── Network server ───────────────────────────────────────────────────────
    server = EdgeNetServer(slicer, cfg.tcp_host, cfg.tcp_port, cfg.udp_port)
    await server.start()

    # ── Traffic generator ────────────────────────────────────────────────────
    generator = TrafficGenerator(cfg.tcp_host, cfg.tcp_port, cfg.udp_port)
    generator.add_profiles(default_profiles())

    # ── Dashboard ────────────────────────────────────────────────────────────
    dashboard = Dashboard(refresh_hz=cfg.dashboard_refresh_hz)

    # ── Manual fault injection ───────────────────────────────────────────────
    if manual_fault:
        async def inject_after_delay():
            await asyncio.sleep(10.0)
            fault_injector.inject(manual_fault, duration_s=30.0)
        asyncio.create_task(inject_after_delay())

    # ── Demo scenario ────────────────────────────────────────────────────────
    scenario = TrafficScenario(fault_injector, events)

    print(f"ML device: {device_desc}")
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
            *(  [scenario.run_demo()] if run_demo else [] ),
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
        print("\nEDGE-NET-X PRO stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDGE-NET-X PRO")
    parser.add_argument(
        "--no-demo", action="store_true",
        help="Skip the automated demo scenario",
    )
    parser.add_argument(
        "--inject", metavar="FAULT",
        choices=["latency_spike", "packet_loss", "slowdown"],
        help="Inject a specific fault 10s after startup",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(run_demo=not args.no_demo, manual_fault=args.inject))
    except KeyboardInterrupt:
        pass
