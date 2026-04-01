"""
Story-driven terminal dashboard.

The layout tells a narrative, not just dumps numbers:

  ┌────────────────────────────────────────────────────────────────────┐
  │  System header: uptime, ML device, inference stats, fault status   │
  ├──────────────────┬──────────────────────────┬──────────────────────┤
  │  Slice metrics   │  ML + adaptive threshold │  Story timeline      │
  │  with SLA budget │  score history sparkline │  cause → effect      │
  ├──────────────────┴──────────────────────────┴──────────────────────┤
  │  Bandwidth allocation bars                                         │
  └────────────────────────────────────────────────────────────────────┘
"""
import asyncio
import time
from collections import deque
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


_SPARKLINE = "▁▂▃▄▅▆▇█"


def _spark(values: deque, lo: float = 0.0, hi: float = 1.0) -> str:
    if not values:
        return "─" * 20
    span = max(hi - lo, 0.001)
    out = []
    for v in list(values)[-24:]:
        idx = int((v - lo) / span * (len(_SPARKLINE) - 1))
        out.append(_SPARKLINE[max(0, min(len(_SPARKLINE) - 1, idx))])
    return "".join(out)


def _score_style(s: float) -> str:
    if s < 0.4: return "bold green"
    if s < 0.7: return "bold yellow"
    return "bold red"


def _lat_style(ms: float, budget: float) -> str:
    r = ms / max(budget, 0.001)
    if r < 0.5: return "green"
    if r < 0.9: return "yellow"
    return "red"


def _loss_style(pct: float) -> str:
    if pct < 1: return "green"
    if pct < 5: return "yellow"
    return "red"


def _age(ts: float) -> str:
    s = time.time() - ts
    if s < 60: return f"{s:.0f}s"
    return f"{s/60:.0f}m"


def build_layout(
    state,
    decision_engine,
    events: list,
    fault_injector,
    inference_engine,
    start_time: float,
    score_history: deque,
) -> Layout:
    from config import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG.slices

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="alloc", size=5),
    )
    layout["main"].split_row(
        Layout(name="metrics", ratio=5),
        Layout(name="ml", ratio=3),
        Layout(name="timeline", ratio=4),
    )

    # ── Header ──────────────────────────────────────────────────────────────
    uptime = time.time() - start_time
    fault_desc = fault_injector.description if fault_injector else "None"
    fault_active = fault_injector and fault_injector.active
    fault_style = "bold red" if fault_active else "green"

    ie = inference_engine
    hdr = Text()
    hdr.append("EDGE-NET-X PRO  ", style="bold cyan")
    hdr.append(f"uptime {uptime:.0f}s  ")
    if ie:
        hdr.append(f"ML {ie.device_name}  ")
        hdr.append(f"infer: {ie.last_single_latency_ms:.2f}ms  ")
        hdr.append(f"total: {ie.inference_count}  ")
    hdr.append("fault: ", style="bold")
    hdr.append(fault_desc, style=fault_style)
    layout["header"].update(Panel(hdr, box=box.SIMPLE))

    # ── Slice metrics table ──────────────────────────────────────────────────
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", padding=(0, 1))
    t.add_column("Slice",    width=11)
    t.add_column("BW Mbps",  justify="right", width=8)
    t.add_column("Budget",   justify="right", width=8)
    t.add_column("Lat ms",   justify="right", width=8)
    t.add_column("Loss %",   justify="right", width=8)
    t.add_column("TP Mbps",  justify="right", width=9)
    t.add_column("Jitter",   justify="right", width=8)
    t.add_column("Queue",    justify="right", width=7)

    colors = {"CRITICAL": "red", "STANDARD": "yellow", "BACKGROUND": "blue"}
    sla = getattr(decision_engine, "sla_breach_counts", {}) if decision_engine else {}

    if state:
        for name, s in state.slices.items():
            budget = cfg[name].latency_budget_ms
            breaches = sla.get(name, 0)
            label = Text(name, style=f"bold {colors[name]}")
            if breaches > 0:
                label.append(f" ⚠{breaches}", style="red")
            t.add_row(
                label,
                f"{s.bandwidth_mbps:.1f}",
                f"{budget:.0f}ms",
                Text(f"{s.latency_ms:.1f}", style=_lat_style(s.latency_ms, budget)),
                Text(f"{s.packet_loss_pct:.1f}", style=_loss_style(s.packet_loss_pct)),
                f"{s.throughput_mbps:.2f}",
                f"{s.jitter_ms:.2f}",
                str(s.queue_depth),
            )
    layout["metrics"].update(Panel(t, title="[bold]Slice Metrics", border_style="blue"))

    # ── ML + adaptive threshold ──────────────────────────────────────────────
    score = state.congestion_score if state else 0.0
    adaptive_t = getattr(decision_engine, "adaptive_threshold", 0.70) if decision_engine else 0.70
    spark = _spark(score_history)

    ml = Text()
    ml.append("Congestion Score\n", style="bold")
    ml.append(f"  {score:.4f}\n", style=_score_style(score))
    ml.append(f"\n  {spark}\n\n", style=_score_style(score))
    ml.append("Adaptive threshold\n", style="dim")
    ml.append(f"  fire:  {adaptive_t:.3f}\n")
    ml.append(f"  clear: {adaptive_t*0.7:.3f}\n\n")

    if sla:
        ml.append("SLA breaches\n", style="dim")
        for sn, cnt in sla.items():
            style = "red" if cnt > 10 else ("yellow" if cnt > 0 else "green")
            ml.append(f"  {sn}: {cnt}\n", style=style)

    if state:
        ml.append(f"\nTotal TP: {state.total_throughput_mbps:.2f} Mbps\n")

    layout["ml"].update(Panel(ml, title="[bold]ML Prediction", border_style="magenta"))

    # ── Story timeline ───────────────────────────────────────────────────────
    tl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tl.add_column("Age",  style="dim", width=5)
    tl.add_column("Tag",  width=9)
    tl.add_column("What happened")

    tag_styles = {
        "FAULT":    "bold red",
        "CLEAR":    "bold green",
        "DECISION": "bold cyan",
        "ROLLBACK": "bold yellow",
        "SLA":      "red",
        "SNAPSHOT": "bold blue",
        "DEMO":     "dim white",
    }

    combined = []
    de = decision_engine
    if de:
        for a in list(de.actions)[:6]:
            tag = "ROLLBACK" if a.rolled_back else "DECISION"
            combined.append((a.timestamp, tag, a.description))
    for ts, tag, msg in (events or [])[:8]:
        combined.append((ts, tag, msg))
    combined.sort(key=lambda x: x[0], reverse=True)

    for ts, tag, desc in combined[:10]:
        style = tag_styles.get(tag, "white")
        tl.add_row(_age(ts), Text(tag, style=style), Text(desc[:52], overflow="ellipsis"))

    layout["timeline"].update(Panel(tl, title="[bold]Story Timeline", border_style="green"))

    # ── Bandwidth bars ───────────────────────────────────────────────────────
    total_bw = 100.0
    bars = Text()
    if state:
        for name, s in state.slices.items():
            pct = s.bandwidth_mbps / total_bw
            filled = int(pct * 52)
            bar = "█" * filled + "░" * (52 - filled)
            budget = cfg[name].latency_budget_ms
            lat_ok = s.latency_ms <= budget
            status = "✓" if lat_ok else "✗ SLA"
            c = colors[name]
            bars.append(f"  {name:<12}", style=f"bold {c}")
            bars.append(f" {bar} ", style=c)
            bars.append(f" {s.bandwidth_mbps:5.1f} Mbps ({pct*100:.0f}%)  ")
            bars.append(status + "\n", style="green" if lat_ok else "red")
    layout["alloc"].update(Panel(bars, title="[bold]Bandwidth Allocation", border_style="cyan"))

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
                    decision_engine=decision_engine,
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
