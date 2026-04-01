"""
Prometheus-compatible metrics registry.

Exports slice-level gauges + histograms in the Prometheus text exposition format.
No external library required — renders the protocol directly.

Metrics exposed:
  edge_net_x_slice_latency_ms{slice}         gauge
  edge_net_x_slice_throughput_mbps{slice}    gauge
  edge_net_x_slice_packet_loss_pct{slice}    gauge
  edge_net_x_slice_jitter_ms{slice}          gauge
  edge_net_x_slice_bandwidth_mbps{slice}     gauge
  edge_net_x_slice_queue_depth{slice}        gauge
  edge_net_x_congestion_score               gauge
  edge_net_x_total_throughput_mbps          gauge
  edge_net_x_sla_violations_total{slice}    counter
  edge_net_x_ml_inference_latency_ms        gauge
  edge_net_x_decisions_total               counter
  edge_net_x_rollbacks_total               counter
"""
import time
from typing import Optional


class PrometheusRegistry:
    def __init__(self, sla_budgets: dict, inference_engine=None, decision_engine=None):
        self._sla_budgets = sla_budgets  # slice_name -> latency_budget_ms
        self._inference_engine = inference_engine
        self._decision_engine = decision_engine
        self._sla_violations: dict = {k: 0 for k in sla_budgets}
        self._start_time = time.time()

    def update_sla(self, slice_name: str, latency_ms: float):
        budget = self._sla_budgets.get(slice_name, float("inf"))
        if latency_ms > budget:
            self._sla_violations[slice_name] = self._sla_violations.get(slice_name, 0) + 1

    def render(self, state) -> str:
        lines = []
        ts_ms = int(time.time() * 1000)

        def gauge(name: str, labels: dict, value: float, help_text: str = ""):
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            lines.append(f"{name}{{{label_str}}} {value:.6g} {ts_ms}")

        def counter(name: str, labels: dict, value: int, help_text: str = ""):
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            lines.append(f"{name}{{{label_str}}} {value} {ts_ms}")

        if state:
            lines.append("# HELP edge_net_x_slice_latency_ms Mean slice latency")
            lines.append("# TYPE edge_net_x_slice_latency_ms gauge")
            for name, s in state.slices.items():
                self.update_sla(name, s.latency_ms)
                gauge("edge_net_x_slice_latency_ms", {"slice": name}, s.latency_ms)

            lines.append("# HELP edge_net_x_slice_throughput_mbps Slice throughput")
            lines.append("# TYPE edge_net_x_slice_throughput_mbps gauge")
            for name, s in state.slices.items():
                gauge("edge_net_x_slice_throughput_mbps", {"slice": name}, s.throughput_mbps)

            lines.append("# HELP edge_net_x_slice_packet_loss_pct Packet loss percent")
            lines.append("# TYPE edge_net_x_slice_packet_loss_pct gauge")
            for name, s in state.slices.items():
                gauge("edge_net_x_slice_packet_loss_pct", {"slice": name}, s.packet_loss_pct)

            lines.append("# HELP edge_net_x_slice_jitter_ms Slice jitter")
            lines.append("# TYPE edge_net_x_slice_jitter_ms gauge")
            for name, s in state.slices.items():
                gauge("edge_net_x_slice_jitter_ms", {"slice": name}, s.jitter_ms)

            lines.append("# HELP edge_net_x_slice_bandwidth_mbps Allocated bandwidth")
            lines.append("# TYPE edge_net_x_slice_bandwidth_mbps gauge")
            for name, s in state.slices.items():
                gauge("edge_net_x_slice_bandwidth_mbps", {"slice": name}, s.bandwidth_mbps)

            lines.append("# HELP edge_net_x_slice_queue_depth Virtual queue depth")
            lines.append("# TYPE edge_net_x_slice_queue_depth gauge")
            for name, s in state.slices.items():
                gauge("edge_net_x_slice_queue_depth", {"slice": name}, s.queue_depth)

            lines.append("# HELP edge_net_x_congestion_score ML congestion probability")
            lines.append("# TYPE edge_net_x_congestion_score gauge")
            lines.append(f"edge_net_x_congestion_score {state.congestion_score:.6g} {ts_ms}")

            lines.append("# HELP edge_net_x_total_throughput_mbps System total throughput")
            lines.append("# TYPE edge_net_x_total_throughput_mbps gauge")
            lines.append(f"edge_net_x_total_throughput_mbps {state.total_throughput_mbps:.6g} {ts_ms}")

        lines.append("# HELP edge_net_x_sla_violations_total Cumulative SLA breaches")
        lines.append("# TYPE edge_net_x_sla_violations_total counter")
        for name, count in self._sla_violations.items():
            counter("edge_net_x_sla_violations_total", {"slice": name}, count)

        if self._inference_engine:
            lines.append("# HELP edge_net_x_ml_single_inference_ms Single inference latency")
            lines.append("# TYPE edge_net_x_ml_single_inference_ms gauge")
            lines.append(
                f"edge_net_x_ml_single_inference_ms "
                f"{self._inference_engine.last_single_latency_ms:.6g} {ts_ms}"
            )
            lines.append("# HELP edge_net_x_ml_inferences_total Total inference count")
            lines.append("# TYPE edge_net_x_ml_inferences_total counter")
            lines.append(
                f"edge_net_x_ml_inferences_total "
                f"{self._inference_engine.inference_count} {ts_ms}"
            )

        if self._decision_engine:
            total = len(self._decision_engine.actions)
            rollbacks = sum(1 for a in self._decision_engine.actions if a.rolled_back)
            lines.append("# HELP edge_net_x_decisions_total Control plane decisions")
            lines.append("# TYPE edge_net_x_decisions_total counter")
            lines.append(f"edge_net_x_decisions_total {total} {ts_ms}")
            lines.append("# HELP edge_net_x_rollbacks_total Rolled-back decisions")
            lines.append("# TYPE edge_net_x_rollbacks_total counter")
            lines.append(f"edge_net_x_rollbacks_total {rollbacks} {ts_ms}")

        lines.append("# HELP edge_net_x_uptime_seconds System uptime")
        lines.append("# TYPE edge_net_x_uptime_seconds counter")
        lines.append(f"edge_net_x_uptime_seconds {time.time() - self._start_time:.1f} {ts_ms}")

        return "\n".join(lines) + "\n"
