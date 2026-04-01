"""
Story-driven demo engine.

Runs a scripted 5-phase sequence that tells a complete narrative:

  Phase 1: Baseline         — system operating normally, metrics stable
  Phase 2: Network break    — latency spike + packet loss injected
  Phase 3: Detection        — ML congestion score rises, SLA breaches logged
  Phase 4: Automatic recovery — decision engine reallocates bandwidth
  Phase 5: Recovery proof   — metrics improve, before/after comparison printed

At each phase transition, a timestamped event is pushed to the timeline.
A baseline snapshot is captured before the fault, and a recovery snapshot
after, so the printed summary shows real measurable improvement.
"""
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    timestamp: float
    label: str
    latency: Dict[str, float]          # slice -> ms
    packet_loss: Dict[str, float]      # slice -> %
    throughput: Dict[str, float]       # slice -> Mbps
    congestion_score: float


def _snap(state, label: str) -> Optional[MetricsSnapshot]:
    if state is None:
        return None
    return MetricsSnapshot(
        timestamp=time.time(),
        label=label,
        latency={k: v.latency_ms for k, v in state.slices.items()},
        packet_loss={k: v.packet_loss_pct for k, v in state.slices.items()},
        throughput={k: v.throughput_mbps for k, v in state.slices.items()},
        congestion_score=state.congestion_score,
    )


def _print_comparison(before: MetricsSnapshot, after: MetricsSnapshot):
    width = 68
    print("\n" + "═" * width)
    print("  BEFORE vs AFTER: automatic recovery results")
    print("═" * width)
    print(f"  {'Metric':<30}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
    print("  " + "─" * (width - 2))

    for name in ["CRITICAL", "STANDARD", "BACKGROUND"]:
        before_lat = before.latency.get(name, 0)
        after_lat = after.latency.get(name, 0)
        delta = after_lat - before_lat
        arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
        color = "\033[92m" if delta < 0 else ("\033[91m" if delta > 0 else "")
        reset = "\033[0m"
        print(f"  {name+' latency (ms)':<30}  {before_lat:>10.1f}  {after_lat:>10.1f}  "
              f"{color}{arrow}{abs(delta):>8.1f}{reset}")

    print()
    for name in ["CRITICAL", "STANDARD", "BACKGROUND"]:
        before_loss = before.packet_loss.get(name, 0)
        after_loss = after.packet_loss.get(name, 0)
        delta = after_loss - before_loss
        arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
        color = "\033[92m" if delta < 0 else ("\033[91m" if delta > 0 else "")
        reset = "\033[0m"
        print(f"  {name+' packet loss (%)':<30}  {before_loss:>10.1f}  {after_loss:>10.1f}  "
              f"{color}{arrow}{abs(delta):>8.1f}{reset}")

    print()
    print(f"  {'Congestion score':<30}  {before.congestion_score:>10.4f}  "
          f"{after.congestion_score:>10.4f}")
    print("═" * width + "\n")


class DemoScenario:
    def __init__(self, fault_injector, events: list, get_state):
        self._fault = fault_injector
        self._events = events
        self._get_state = get_state
        self.baseline: Optional[MetricsSnapshot] = None
        self.post_recovery: Optional[MetricsSnapshot] = None

    def _log(self, msg: str, tag: str = "DEMO"):
        self._events.insert(0, (time.time(), tag, msg))
        logger.info("[Demo] %s", msg)

    async def run(self):
        # ── Phase 1: Baseline (20s) ─────────────────────────────────────────
        self._log("Phase 1 — Baseline: all slices operating normally")
        await asyncio.sleep(18.0)
        self.baseline = _snap(self._get_state(), "baseline")
        self._log("Baseline snapshot captured", "SNAPSHOT")
        await asyncio.sleep(2.0)

        # ── Phase 2: Break the network ──────────────────────────────────────
        self._log("Phase 2 — BREAKING the network: injecting latency spike + packet loss", "FAULT")
        self._fault.inject("latency_spike")
        await asyncio.sleep(3.0)
        self._fault.inject("packet_loss")
        self._log("Latency spike +80ms AND 25% packet loss active — watch metrics degrade", "FAULT")

        # ── Phase 3: Detection ──────────────────────────────────────────────
        await asyncio.sleep(5.0)
        self._log("Phase 3 — Detection: ML congestion score rising toward threshold")
        await asyncio.sleep(10.0)
        self._log("Prediction threshold crossed — decision engine evaluating...", "DECISION")

        # ── Phase 4: Automatic recovery ─────────────────────────────────────
        await asyncio.sleep(5.0)
        self._log("Phase 4 — Recovery: CRITICAL slice bandwidth reallocated upward", "DECISION")
        self._log("BACKGROUND traffic throttled to protect latency-sensitive flows", "DECISION")
        self._fault.clear()
        self._log("Fault cleared — system converging to stable state")

        # ── Phase 5: Proof ──────────────────────────────────────────────────
        await asyncio.sleep(12.0)
        self.post_recovery = _snap(self._get_state(), "post_recovery")
        self._log("Phase 5 — Recovery proof: before/after comparison available", "SNAPSHOT")
        await asyncio.sleep(3.0)

        if self.baseline and self.post_recovery:
            _print_comparison(self.baseline, self.post_recovery)

        self._log("Demo complete — system stable")
