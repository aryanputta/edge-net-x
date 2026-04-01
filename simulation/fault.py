"""
Fault injection controller.

The fault_state dict is shared with every SliceQueue so faults take effect
immediately without any message-passing overhead.
"""
import asyncio
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


FAULT_TYPES = {
    "latency_spike": {"magnitude_ms": 80.0},
    "packet_loss":   {"loss_rate": 0.25},
    "slowdown":      {"slowdown_ms": 40.0},
}


class FaultInjector:
    def __init__(self, fault_state: dict, events: list):
        self._state = fault_state
        self._events = events
        self._active_fault: Optional[str] = None

    def inject(self, fault_type: str, duration_s: Optional[float] = None):
        if fault_type not in FAULT_TYPES:
            logger.warning("Unknown fault type: %s", fault_type)
            return
        self._active_fault = fault_type
        params = FAULT_TYPES[fault_type]
        self._state.clear()
        self._state["active"] = True
        self._state["type"] = fault_type
        self._state.update(params)

        msg = f"FAULT INJECTED: {fault_type} ({params})"
        self._events.insert(0, (time.time(), "FAULT", msg))
        logger.warning(msg)

        if duration_s:
            asyncio.ensure_future(self._auto_clear(duration_s))

    def clear(self):
        fault_type = self._active_fault or "unknown"
        self._state.clear()
        self._state["active"] = False
        self._active_fault = None
        msg = f"FAULT CLEARED: {fault_type}"
        self._events.insert(0, (time.time(), "CLEAR", msg))
        logger.info(msg)

    async def _auto_clear(self, duration_s: float):
        await asyncio.sleep(duration_s)
        if self._state.get("active"):
            self.clear()

    @property
    def active(self) -> bool:
        return bool(self._state.get("active"))

    @property
    def description(self) -> str:
        if not self.active:
            return "None"
        ft = self._state.get("type", "unknown")
        if ft == "latency_spike":
            return f"Latency +{self._state.get('magnitude_ms', 0):.0f}ms"
        elif ft == "packet_loss":
            return f"Loss {self._state.get('loss_rate', 0) * 100:.0f}%"
        elif ft == "slowdown":
            return f"Slowdown +{self._state.get('slowdown_ms', 0):.0f}ms"
        return ft
