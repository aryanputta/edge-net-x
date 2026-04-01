"""
Rich terminal dashboard — live system introspection.

Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Header: system info, uptime, ML device, fault status           │
  ├──────────────────────┬──────────────────┬───────────────────────┤
  │  Slice metrics table │  ML / Congestion │  Event timeline       │
  │  (latency, loss, tp) │  score + history │  (decisions + faults) │
  └──────────────────────┴──────────────────┴───────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │  Bandwidth allocation bar chart per slice                       │
  └─────────────────────────────────────────────────────────────────┘
"""
import time
import math
from collections import deque
from typing import List, Tuple, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.bar import Bar
from rich import box
from rich.columns import Columns
from rich.progress import BarColumn, Progress, TaskID


_SCORE_HISTORY: deque = deque(maxlen=40)
_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: deque, vmin: float = 0.0, vmax: float = 1.0) -> str:
    if not values:
        return "─" * 20
    span = max(vmax - vmin, 0.001)
    chars = []
    for v in list(values)[-20:]:
        idx = int((v - vmin) / span * (len(_SPARKLINE_CHARS) - 1))
        idx = max(0, min(len(_SPARKLINE_CHARS) - 1, idx))
        chars.append(_SPARKLINE_CHARS[idx])
    return "".join(chars)


def _score_color(score: float) -> str:
    if score < 0.4:
        return "green"
    if score < 0.7:
        return "yellow"
    return "red"


def _latency_color(ms: float, budget_ms: float) -> str:
    ratio = ms / max(budget_ms, 1.0)
    if ratio < 0.5:
        return "green"
    if ratio < 0.9:
        return "yellow"
    return "red"


def build_layout(
    state,
    decision_actions: deque,
    events: list,
    fault_injector,
    inference_engine,
    start_time: float,
    score_history: deque,
) -> Layout:
    from config import DEFAULT_CONFIG

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="alloc", size=5),
    )
    layout["main"].split_row(
        Layout(name="metrics", ratio=2),
        Layout(name="ml", ratio=1),
        Layout(name="events", ratio=2),
    )

    uptime = time.time() - start_time
    fault_desc = fault_injector.description if fault_injector else "None"
    fault_color = "red bold" if fault_injector and fault_injector.active else "green"
    device_str = inference_engine.device_name if inference_engine else "CPU"
    infer_count = inference_engine.inference_count if inference_engine else 0
    single_ms = inference_engine.last_single_latency_ms if inference_engine else 0.0
    batch_ms = inference_engine.last_batch_latency_ms if inference_engine else 0.0

    header_text = Text()
    header_text.append("EDGE-NET-X PRO  ", style="bold cyan")
    header_text.append(f"uptime {uptime:.0f}s  ")
    header_text.append(f"ML device: {device_str}  inferences: {infer_count}  ")
    header_text.append(f"single: {single_ms:.2f}ms  batch: {batch_ms:.2f}ms  ")
    header_text.append("fault: ", style="bold")
    header_text.append(fault_desc, style=fault_color)
    layout["header"].update(Panel(header_text, box=box.SIMPLE))

    # ── Slice metrics table ─────────────────────────────────────────
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Slice", width=10)
    table.add_column("BW (Mbps)", justify="right", width=10)
    table.add_column("Latency (ms)", justify="right", width=13)
    table.add_column("Loss %", justify="right", width=8)
    table.add_column("Throughput", justify="right", width=11)
    table.add_column("Jitter (ms)", justify="right", width=11)
    table.add_column("Queue", justify="right", width=7)
    table.add_column("Flows", justify="right", width=6)

    cfg = DEFAULT_CONFIG.slices
    if state:
        for name, s in state.slices.items():
            budget = cfg[name].latency_budget_ms
            lat_color = _latency_color(s.latency_ms, budget)
            loss_color = "red" if s.packet_loss_pct > 10 else ("yellow" if s.packet_loss_pct > 2 else "green")
            slice_color = {"CRITICAL": "red", "STANDARD": "yellow", "BACKGROUND": "blue"}[name]
            table.add_row(
                Text(name, style=f"bold {slice_color}"),
                f"{s.bandwidth_mbps:.1f}",
                Text(f"{s.latency_ms:.1f}", style=lat_color),
                Text(f"{s.packet_loss_pct:.1f}", style=loss_color),
                f"{s.throughput_mbps:.2f}",
                f"{s.jitter_ms:.2f}",
                str(s.queue_depth),
                str(s.active_flows),
            )
    layout["metrics"].update(Panel(table, title="[bold]Slice Metrics", border_style="blue"))

    # ── ML Panel ────────────────────────────────────────────────────
    score = state.congestion_score if state else 0.0
    _SCORE_HISTORY.append(score)
    spark = _sparkline(score_history if score_history else _SCORE_HISTORY)
    score_color = _score_color(score)
    ml_text = Text()
    ml_text.append(f"Congestion Score\n\n", style="bold")
    ml_text.append(f"  {score:.4f}\n\n", style=f"bold {score_color} on default")
    ml_text.append("History (40 ticks):\n", style="dim")
    ml_text.append(f"  {spark}\n\n", style=score_color)
    ml_text.append("Thresholds:\n", style="dim")
    ml_text.append("  High:  0.70 → reallocate\n")
    ml_text.append("  Reset: 0.50 → rebalance\n\n")
    if state:
        tp = state.total_throughput_mbps
        ml_text.append(f"Total throughput: {tp:.2f} Mbps\n")
    layout["ml"].update(Panel(ml_text, title="[bold]ML Prediction", border_style="magenta"))

    # ── Event timeline ───────────────────────────────────────────────
    ev_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    ev_table.add_column("Time", style="dim", width=8)
    ev_table.add_column("Type", width=8)
    ev_table.add_column("Description")

    combined = []
    for action in list(decision_actions)[:5]:
        tag = "ROLLBACK" if action.rolled_back else "DECISION"
        color = "red" if action.rolled_back else "cyan"
        combined.append((action.timestamp, tag, color, action.description))
    for ts, tag, msg in (events or [])[:5]:
        color = "red" if tag == "FAULT" else ("green" if tag == "CLEAR" else "yellow")
        combined.append((ts, tag, color, msg))
    combined.sort(key=lambda x: x[0], reverse=True)

    for ts, tag, color, desc in combined[:8]:
        age = time.time() - ts
        time_str = f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago"
        ev_table.add_row(
            time_str,
            Text(tag, style=f"bold {color}"),
            Text(desc[:60], overflow="ellipsis"),
        )
    layout["events"].update(Panel(ev_table, title="[bold]Event Timeline", border_style="green"))

    # ── Bandwidth allocation bars ─────────────────────────────────────
    alloc_text = Text()
    total_bw = DEFAULT_CONFIG.total_bandwidth_mbps
    colors = {"CRITICAL": "red", "STANDARD": "yellow", "BACKGROUND": "blue"}
    if state:
        for name, s in state.slices.items():
            pct = s.bandwidth_mbps / total_bw
            bar_len = int(pct * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            alloc_text.append(f"  {name:<12}", style=f"bold {colors[name]}")
            alloc_text.append(f" {bar} ", style=colors[name])
            alloc_text.append(f" {s.bandwidth_mbps:5.1f} Mbps ({pct*100:.0f}%)\n")
    layout["alloc"].update(Panel(alloc_text, title="[bold]Bandwidth Allocation", border_style="cyan"))

    return layout


class Dashboard:
    def __init__(self, refresh_hz: float = 5.0):
        self._refresh = refresh_hz
        self._console = Console()
        self._running = False
        self.score_history: deque = deque(maxlen=40)

    async def run(
        self,
        get_state,
        decision_engine,
        events: list,
        fault_injector,
        inference_engine,
        start_time: float,
    ):
        self._running = True
        interval = 1.0 / self._refresh

        with Live(console=self._console, refresh_per_second=self._refresh, screen=True) as live:
            while self._running:
                state = get_state()
                if state:
                    self.score_history.append(state.congestion_score)
                layout = build_layout(
                    state=state,
                    decision_actions=decision_engine.actions if decision_engine else deque(),
                    events=events,
                    fault_injector=fault_injector,
                    inference_engine=inference_engine,
                    start_time=start_time,
                    score_history=self.score_history,
                )
                live.update(layout)
                await asyncio.sleep(interval)

    def stop(self):
        self._running = False


import asyncio
