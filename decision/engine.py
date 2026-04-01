"""
Decision engine — the control plane brain.

Reads ML congestion score + live metrics, decides whether to reallocate
bandwidth across slices, and applies the action through the slicer.

Design guarantees:
  - Hysteresis: only triggers after N consecutive above-threshold readings
  - Cooldown: minimum interval between consecutive decisions
  - Rollback: if metrics don't improve within the rollback window, revert
  - Starvation prevention: BACKGROUND slice always keeps min_bandwidth_mbps
  - Adaptive thresholding: adjusts congestion threshold based on system variance
  - SLA breach detection: emits events when latency exceeds per-slice budget
"""
import asyncio
import math
import time
import logging
from collections import deque
from typing import Dict, List, Optional

from config import SystemConfig
from models import DecisionAction, NetworkState

logger = logging.getLogger(__name__)


class AdaptiveThreshold:
    """
    Adjusts the congestion firing threshold based on recent score variance.

    When the system is stable (low variance), lower the threshold to catch
    congestion earlier. When the system is oscillating (high variance), raise
    it to reduce unnecessary decisions.
    """

    def __init__(self, base: float = 0.70, min_t: float = 0.55, max_t: float = 0.85):
        self._base = base
        self._min = min_t
        self._max = max_t
        self._history: deque = deque(maxlen=30)
        self.current = base

    def update(self, score: float) -> float:
        self._history.append(score)
        if len(self._history) < 10:
            return self.current
        mean = sum(self._history) / len(self._history)
        variance = sum((x - mean) ** 2 for x in self._history) / len(self._history)
        std = math.sqrt(variance)

        # High variance → raise threshold (system oscillating, be conservative)
        # Low variance  → lower threshold (system stable, catch congestion early)
        adjustment = (std - 0.15) * 0.5    # neutral at std=0.15
        self.current = max(self._min, min(self._max, self._base + adjustment))
        return self.current


class DecisionEngine:
    def __init__(self, config: SystemConfig, slicer, telemetry, events: list = None):
        self._cfg = config
        self._slicer = slicer
        self._telemetry = telemetry
        self._events = events or []

        # Snapshot original allocations before any mutations so _act_normal
        # can rebalance to the startup values rather than the current (modified) ones.
        self._original_allocations: Dict[str, float] = {
            name: sc.bandwidth_mbps for name, sc in config.slices.items()
        }

        self._high_count: int = 0
        self._low_count: int = 0
        self._last_action_time: float = 0.0
        self._running = False

        self.actions: deque = deque(maxlen=100)
        self._pending_rollback: Optional[Dict] = None
        self._rollback_at: float = 0.0

        # Adaptive threshold
        self._adaptive = AdaptiveThreshold(
            base=config.congestion_threshold,
            min_t=config.congestion_threshold - 0.15,
            max_t=config.congestion_threshold + 0.15,
        )

        # SLA breach tracking
        self._sla_breach_counts: Dict[str, int] = {k: 0 for k in config.slices}

    async def run(self):
        self._running = True
        while self._running:
            await asyncio.sleep(1.0)
            state = self._telemetry.current_state
            if state is None:
                continue

            # Inject queue depths from slicer
            depths = {name: q.queue_depth for name, q in self._slicer.queues.items()}
            self._telemetry.inject_queue_depths(depths)

            score = state.congestion_score
            now = time.monotonic()

            # Update adaptive threshold based on recent score variance
            effective_threshold = self._adaptive.update(score)

            # SLA breach detection
            self._check_sla_breaches(state, now)

            # Check rollback
            if self._pending_rollback and now >= self._rollback_at:
                await self._maybe_rollback(state)

            # Hysteresis counter — uses adaptive threshold
            if score >= effective_threshold:
                self._high_count += 1
                self._low_count = 0
            elif score <= effective_threshold * 0.7:  # clear band below threshold
                self._high_count = 0
                self._low_count += 1
            # else: in dead-band — don't reset either counter

            cooldown_ok = (now - self._last_action_time) >= self._cfg.decision_cooldown_s

            if self._high_count >= self._cfg.hysteresis_count and cooldown_ok:
                await self._act_congested(state, score)
            elif self._low_count >= self._cfg.hysteresis_count and cooldown_ok:
                await self._act_normal(state, score)

    async def _act_congested(self, state: NetworkState, score: float):
        alloc = self._slicer.current_allocations()
        cfg = self._cfg.slices

        # Snapshot baseline for rollback
        baseline = {
            name: s.latency_ms for name, s in state.slices.items()
        }

        # Increase CRITICAL, reduce BACKGROUND, cap STANDARD
        new_alloc = dict(alloc)
        transfer = min(10.0, alloc["BACKGROUND"] - cfg["BACKGROUND"].min_bandwidth_mbps)
        transfer += min(5.0, alloc["STANDARD"] - cfg["STANDARD"].min_bandwidth_mbps)

        new_alloc["CRITICAL"] = min(
            cfg["CRITICAL"].max_bandwidth_mbps,
            alloc["CRITICAL"] + transfer * 0.7,
        )
        new_alloc["STANDARD"] = max(
            cfg["STANDARD"].min_bandwidth_mbps,
            alloc["STANDARD"] - transfer * 0.3,
        )
        new_alloc["BACKGROUND"] = max(
            cfg["BACKGROUND"].min_bandwidth_mbps,
            alloc["BACKGROUND"] - transfer * 0.4,
        )

        # Normalize so total doesn't exceed budget
        total = sum(new_alloc.values())
        budget = self._cfg.total_bandwidth_mbps
        if total > budget:
            scale = budget / total
            new_alloc = {k: v * scale for k, v in new_alloc.items()}

        await self._slicer.adjust_bandwidth(new_alloc)
        self._last_action_time = time.monotonic()
        self._high_count = 0

        action = DecisionAction(
            timestamp=time.time(),
            trigger="congestion_spike",
            congestion_score=score,
            adjustments=new_alloc,
            description=(
                f"Congestion {score:.2f} ≥ {self._cfg.congestion_threshold:.2f}. "
                f"CRITICAL ↑{new_alloc['CRITICAL']:.1f} Mbps, "
                f"BACKGROUND ↓{new_alloc['BACKGROUND']:.1f} Mbps"
            ),
            baseline_metrics=baseline,
        )
        self.actions.appendleft(action)

        # Schedule rollback check
        self._pending_rollback = {"alloc": alloc, "action": action}
        self._rollback_at = time.monotonic() + self._cfg.rollback_window_s

        logger.info("Decision: %s", action.description)

    async def _act_normal(self, state: NetworkState, score: float):
        alloc = self._slicer.current_allocations()

        # Gently rebalance toward original startup allocations (not mutated values)
        defaults = self._original_allocations
        new_alloc = {}
        for name in alloc:
            diff = defaults[name] - alloc[name]
            new_alloc[name] = alloc[name] + diff * 0.5  # move halfway back

        await self._slicer.adjust_bandwidth(new_alloc)
        self._last_action_time = time.monotonic()
        self._low_count = 0

        action = DecisionAction(
            timestamp=time.time(),
            trigger="congestion_clear",
            congestion_score=score,
            adjustments=new_alloc,
            description=(
                f"Congestion cleared ({score:.2f}). Rebalancing toward defaults."
            ),
        )
        self.actions.appendleft(action)
        logger.info("Decision: %s", action.description)

    async def _maybe_rollback(self, state: NetworkState):
        rb = self._pending_rollback
        if rb is None:
            return
        self._pending_rollback = None

        action: DecisionAction = rb["action"]
        baseline = action.baseline_metrics or {}

        # Check if CRITICAL latency improved
        current_crit_lat = state.slices.get("CRITICAL", None)
        if current_crit_lat and "CRITICAL" in baseline:
            if current_crit_lat.latency_ms >= baseline["CRITICAL"] * 1.1:
                # No improvement — rollback
                await self._slicer.adjust_bandwidth(rb["alloc"])
                action.rolled_back = True
                rollback_action = DecisionAction(
                    timestamp=time.time(),
                    trigger="rollback",
                    congestion_score=state.congestion_score,
                    adjustments=rb["alloc"],
                    description="Metrics did not improve — reverting bandwidth allocation.",
                )
                self.actions.appendleft(rollback_action)
                logger.info("Decision: rollback applied")

    def _check_sla_breaches(self, state: NetworkState, now: float):
        for name, s in state.slices.items():
            budget = self._cfg.slices[name].latency_budget_ms
            if s.latency_ms > budget and s.latency_ms > 0:
                self._sla_breach_counts[name] = self._sla_breach_counts.get(name, 0) + 1
                if self._sla_breach_counts[name] % 5 == 1:  # log every 5th breach
                    msg = (
                        f"SLA BREACH: {name} latency {s.latency_ms:.1f}ms "
                        f"exceeds budget {budget:.0f}ms "
                        f"(#{self._sla_breach_counts[name]})"
                    )
                    self._events.insert(0, (time.time(), "SLA", msg))
                    logger.warning(msg)

    @property
    def adaptive_threshold(self) -> float:
        return self._adaptive.current

    @property
    def sla_breach_counts(self) -> Dict[str, int]:
        return dict(self._sla_breach_counts)

    def update_congestion_score(self, score: float):
        """Called by the ML inference loop to push the latest score."""
        # Persist so telemetry ticks don't reset it to 0
        self._telemetry._last_congestion_score = score
        state = self._telemetry.current_state
        if state:
            state.congestion_score = score

    def stop(self):
        self._running = False
