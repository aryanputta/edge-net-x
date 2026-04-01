# EDGE-NET-X PRO

A real-time distributed networking control plane that simulates 5G-style network slicing with GPU-accelerated ML-driven congestion prediction and adaptive traffic management.

```
┌────────────────────────────────────────────────────────────────────┐
│  EDGE-NET-X PRO   uptime 42s  ML device: CPU  inferences: 84      │
├──────────────────────┬──────────────────┬──────────────────────────┤
│  Slice Metrics       │  ML Prediction   │  Event Timeline          │
│  CRITICAL  1.2ms  0% │  Score: 0.8831   │  12s  DECISION: CRIT↑   │
│  STANDARD  6.4ms  1% │  ▁▁▂▃▅▇████████ │  18s  FAULT: latency+80 │
│  BACKGROUND 22ms  4% │  High:  0.70     │  30s  DEMO: Phase 2      │
├──────────────────────┴──────────────────┴──────────────────────────┤
│  CRITICAL   ████████████████████████████████████░░░░░░  72.0 Mbps │
│  STANDARD   ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20.0 Mbps │
│  BACKGROUND ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8.0 Mbps │
└────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
Traffic Generators (TCP + UDP)
        │
        ▼
 EdgeNetServer ──────────────────────────────────┐
        │                                        │
        ▼                                        │
 NetworkSlicer (3 slices)                        │
  ├─ CRITICAL  [AsyncTokenBucket 60 Mbps]        │
  ├─ STANDARD  [AsyncTokenBucket 30 Mbps]        │
  └─ BACKGROUND[AsyncTokenBucket 10 Mbps]        │
        │                                        │
        ▼                                        │
 TelemetryCollector                              │
  └─ rolling windows (200 samples)               │
  └─ ML feature vectors (15 features/tick)       │
        │                                        │
        ▼                                        │
 CongestionLSTM (GPU/CPU)                        │
  └─ (batch, seq=50, 15) → congestion prob       │
        │                                        │
        ▼                                        │
 DecisionEngine ◄────────────────────────────────┘
  ├─ hysteresis (3 consecutive readings)
  ├─ cooldown (3s between decisions)
  └─ rollback (reverts if no improvement in 5s)
        │
        ▼
 NetworkSlicer.adjust_bandwidth()
```

**All components run in a single asyncio event loop.** The ML inference runs in a ThreadPoolExecutor thread so the event loop is never blocked by PyTorch. Bandwidth enforcement uses per-slice token buckets with burst capacity.

## Key Design Decisions

### Adaptive thresholding
The decision engine tracks the variance of recent ML congestion scores. When variance is low (system stable), it lowers the firing threshold — catching congestion earlier. When variance is high (system oscillating from too-frequent decisions), it raises the threshold to reduce churn. This self-tuning loop prevents the classic oscillation problem in control systems without requiring manual tuning.

### SLA breach detection
Every decision engine tick checks each slice's measured latency against its configured latency budget (CRITICAL: 2ms, STANDARD: 10ms, BACKGROUND: 50ms). Breaches are counted, logged to the event timeline, and exported as `edge_net_x_sla_violations_total` Prometheus counters. This gives operators actionable signal beyond the congestion score.

### REST API design
The control API uses only Python stdlib (`asyncio.start_server`) — no aiohttp, FastAPI, or extra deps. This demonstrates that async HTTP handling from first principles is ~80 lines of code. The API enables Grafana integration, external orchestrators, and scripted fault injection for testing.

### Why asyncio for networking?
A single-threaded asyncio event loop handles thousands of concurrent connections via I/O multiplexing, matching how real 5G control-plane software is structured. Each flow client is an asyncio task — no OS thread overhead.

### Latency simulation without real network hardware
Packets carry a `created_at` monotonic timestamp. The token bucket and fault injector apply `asyncio.sleep()` delays before acknowledging each packet. Since the OS loopback is effectively zero-latency, measured RTT equals the simulated delay — fully controllable in software.

### Token bucket bandwidth enforcement
Each slice has an `AsyncTokenBucket` that refills at the slice's current bandwidth rate (bytes/second). Packets that can't consume tokens are dropped immediately (tail drop). Queue depth is tracked as a virtual counter for telemetry.

### LSTM congestion prediction
- **Input**: sliding window of 50 telemetry snapshots × 15 features (5 metrics × 3 slices)
- **Model**: 2-layer LSTM(64) → Linear(32) → Linear(1) → Sigmoid
- **Training**: synthetic data with three patterns (normal, congested, transition), trained in ~2s on CPU before the demo starts
- **Inference**: runs every 500ms in a ThreadPoolExecutor thread; GPU batch inference is supported

### Hysteresis + cooldown prevent oscillation
The decision engine won't act on a single congestion spike. It requires 3 consecutive above-threshold readings (hysteresis counter) and enforces a 3-second cooldown between decisions. If metrics don't improve within 5 seconds of a reallocation, the action is automatically rolled back.

### Fault injection
A shared `fault_state` dict is read by every `SliceQueue` on each packet. Injecting a fault is instant — no message passing. Supported faults: `latency_spike`, `packet_loss`, `slowdown`.

## Components

| Directory | Purpose |
|-----------|---------|
| `network/` | Async TCP + UDP server; multi-flow traffic generators |
| `telemetry/` | Async rolling-window metrics collector (200 samples, 100ms ticks) |
| `ml/` | CongestionLSTM, synthetic training with early stopping, GPU inference |
| `decision/` | Adaptive control engine: hysteresis, cooldown, rollback, SLA detection |
| `slicing/` | Token-bucket scheduler, per-slice bandwidth enforcement |
| `simulation/` | Fault injector, automated demo scenario |
| `dashboard/` | Rich terminal live dashboard |
| `api/` | REST control plane API (stdlib asyncio, zero extra deps) |
| `metrics/` | Prometheus text format exporter |

## Installation

```bash
pip install torch rich
```

CUDA is detected automatically. Falls back to CPU if no GPU is available.

## Running

```bash
# Full automated demo (5 phases, ~2 minutes)
python main.py

# No demo, just run the system
python main.py --no-demo

# Inject a specific fault 10s after startup
python main.py --inject latency_spike
python main.py --inject packet_loss
python main.py --inject slowdown

# Custom API port
python main.py --api-port 9090
```

## REST Control API

The system exposes a REST API on `http://127.0.0.1:8080` (default):

```bash
# Full system snapshot
curl http://127.0.0.1:8080/api/status

# Current bandwidth allocations
curl http://127.0.0.1:8080/api/bandwidth

# Override allocations at runtime
curl -X POST http://127.0.0.1:8080/api/bandwidth \
     -d '{"CRITICAL": 70, "STANDARD": 20, "BACKGROUND": 10}'

# Inject a fault (optional ?duration=30 seconds)
curl -X POST "http://127.0.0.1:8080/api/fault/inject/latency_spike?duration=30"
curl -X POST "http://127.0.0.1:8080/api/fault/inject/packet_loss?duration=20"

# Clear active fault
curl -X POST http://127.0.0.1:8080/api/fault/clear

# Last 10 control-plane decisions
curl http://127.0.0.1:8080/api/decisions

# Prometheus metrics
curl http://127.0.0.1:8080/metrics
```

## Prometheus Metrics

The `/metrics` endpoint exports Prometheus-compatible gauges and counters:

| Metric | Type | Description |
|--------|------|-------------|
| `edge_net_x_slice_latency_ms{slice}` | gauge | Mean latency per slice |
| `edge_net_x_slice_throughput_mbps{slice}` | gauge | Throughput per slice |
| `edge_net_x_slice_packet_loss_pct{slice}` | gauge | Packet loss per slice |
| `edge_net_x_slice_bandwidth_mbps{slice}` | gauge | Current allocation |
| `edge_net_x_congestion_score` | gauge | ML congestion probability |
| `edge_net_x_sla_violations_total{slice}` | counter | Cumulative SLA breaches |
| `edge_net_x_decisions_total` | counter | Control-plane decisions |
| `edge_net_x_rollbacks_total` | counter | Rolled-back decisions |
| `edge_net_x_ml_inferences_total` | counter | Total ML inferences |
| `edge_net_x_ml_single_inference_ms` | gauge | Inference latency |

## Benchmark

```bash
python benchmark.py
```

Sample output (CPU only):
```
Device: CPU
──────────────────────────────────────────────────
  batch=   1  avg=  1.843ms  throughput=   542.5 inf/s
  batch=   8  avg=  3.217ms  throughput=  2487.1 inf/s
  batch=  32  avg=  8.901ms  throughput=  3595.0 inf/s
  batch=  64  avg= 15.442ms  throughput=  4145.2 inf/s
  batch= 128  avg= 28.134ms  throughput=  4548.3 inf/s
```

Sample output (with GPU):
```
Device: GPU (NVIDIA RTX 4090)
──────────────────────────────────────────────────
  batch=   1  avg=  0.412ms  throughput=  2427.1 inf/s
  batch=  32  avg=  0.893ms  throughput= 35836.4 inf/s
  batch= 128  avg=  1.241ms  throughput=103143.0 inf/s

CPU vs GPU Speedup
  batch=   1  speedup= 4.47x
  batch=  32  speedup= 9.97x
  batch= 128  speedup=22.67x
```

## Demo Walkthrough

The automated demo runs five phases:

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Normal traffic | 15s | All slices operating at baseline allocations |
| 2. Fault injection | 20s | Latency spike + packet loss injected |
| 3. ML prediction | ~10s | Congestion score rises above 0.70 threshold |
| 4. System reacts | instant | Decision engine reallocates: CRITICAL ↑, BACKGROUND ↓ |
| 5. Recovery | 20s | Fault cleared, system rebalances toward defaults |

Watch the **ML Prediction** panel for the score climbing toward 1.0, then the **Event Timeline** for the DECISION trigger, and finally the **Bandwidth Allocation** bars shift to reflect the new slice allocations.

## Performance Targets

| Metric | Normal | Under Congestion | After Reallocation |
|--------|--------|------------------|--------------------|
| CRITICAL latency | ~1ms | ~80ms | ~2ms |
| CRITICAL packet loss | 0% | ~25% | <1% |
| ML inference (CPU) | ~2ms | ~2ms | ~2ms |
| Decision latency | — | <1s after 3 readings | — |

## Flow Profiles

The system runs 9 concurrent flows by default:

| Flow | Slice | Protocol | Rate | Size |
|------|-------|----------|------|------|
| critical-tcp-0/1/2 | CRITICAL | TCP | 50 pkt/s | 512–2048 B |
| standard-tcp-0/1/2 | STANDARD | TCP | 30 pkt/s | 1024–4096 B |
| background-udp-0/1/2 | BACKGROUND | UDP | 20 pkt/s | 256–1024 B |
