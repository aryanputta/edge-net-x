"""
Traffic scenario controller — drives demo sequences automatically.
"""
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class TrafficScenario:
    """
    Runs the five-phase demo automatically:
      1. Normal traffic     (30s)
      2. Inject congestion  (20s) — latency spike fault
      3. ML predicts spike  — no direct control, just observation
      4. System reacts      — decision engine fires automatically
      5. Metrics improve    — observe recovery
    """

    def __init__(self, fault_injector, events: list):
        self._fault = fault_injector
        self._events = events

    async def run_demo(self):
        await asyncio.sleep(15.0)
        self._log("Phase 1: Normal traffic — baseline established")

        await asyncio.sleep(15.0)
        self._log("Phase 2: Injecting congestion — latency spike + packet loss")
        self._fault.inject("latency_spike")
        await asyncio.sleep(5.0)
        self._fault.inject("packet_loss", duration_s=15.0)

        await asyncio.sleep(20.0)
        self._log("Phase 3: ML congestion prediction rising...")

        await asyncio.sleep(15.0)
        self._log("Phase 4: Decision engine reallocating bandwidth")

        await asyncio.sleep(15.0)
        self._fault.clear()
        self._log("Phase 5: Fault cleared — metrics recovering")

        await asyncio.sleep(20.0)
        self._log("Demo complete — system stable")

    def _log(self, msg: str):
        self._events.insert(0, (time.time(), "DEMO", msg))
        logger.info("[Demo] %s", msg)
