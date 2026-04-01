# EDGE-NET-X PRO

A real-time network control plane that simulates 5G-style traffic slicing, runs GPU-accelerated ML inference to predict congestion, and automatically reallocates bandwidth before latency SLAs are breached.

The goal is not to show a dashboard. It is to show that a software control loop can observe a network, predict failure, act on the prediction, and prove the improvement — all without human intervention.

---

## The Problem

Modern networks carry traffic with wildly different requirements. A VoIP call cannot tolerate 80ms of latency. A file sync can wait 10 seconds. When a network segment becomes congested, naive FIFO queuing degrades everything equally — the phone call drops just as the file sync slows down.

5G network slicing solves this with per-traffic class queues and dedicated bandwidth. The hard part is knowing *when* to reallocate bandwidth, and doing it fast enough to matter.

This system builds that control loop.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Traffic Generators                                │
│    CRITICAL (TCP, 50 pkt/s)   STANDARD (TCP Poisson)   BG (UDP burst)  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │  Real binary frames (20B hdr + payload)
                                    ▼
                          ┌─────────────────┐
                          │  EdgeNetServer   │  TCP :7000  UDP :7001
                          │  asyncio I/O    │
                          └────────┬────────┘
                                   │ NetworkPacket (real byte count)
                                   ▼
                ┌──────────────────────────────────────────┐
                │              NetworkSlicer               │
                │  CRITICAL  [AsyncTokenBucket 60 Mbps]    │
                │  STANDARD  [AsyncTokenBucket 30 Mbps]    │
                │  BACKGROUND[AsyncTokenBucket 10 Mbps]    │
                │  + fault overlay (latency / drop / slow) │
                └────────────────┬─────────────────────────┘
                                 │ PacketResult (latency_ms, dropped)
                                 ▼
                      ┌──────────────────────┐
                      │  TelemetryCollector  │
                      │  200-sample rolling  │
                      │  window per slice    │
                      │  15 features/tick    │
                      └──────┬───────────────┘
                             │ feature vector (50 timesteps × 15 features)
                             ▼
                  ┌─────────────────────────┐
                  │   CongestionLSTM        │
                  │   2-layer LSTM(64)      │
                  │   CPU / CUDA            │
                  │   inference every 500ms │
                  └───────────┬─────────────┘
                              │ congestion probability [0, 1]
                              ▼
              ┌───────────────────────────────────────┐
              │           DecisionEngine              │
              │  adaptive threshold (variance-based)  │
              │  hysteresis: 3 consecutive readings   │
              │  cooldown:   3s between decisions     │
              │  rollback:   revert if no improvement │
              │  SLA breach: alert if budget exceeded │
              └───────────────┬───────────────────────┘
                              │ adjust_bandwidth({"CRITICAL": 72, ...})
                              ▼
                      NetworkSlicer.queues
```

---

## Control Loop Detail

```
Every 100ms:  TelemetryCollector aggregates packet results → rolling stats
Every 500ms:  CongestionLSTM runs inference on 50-tick window
Every 1s:     DecisionEngine evaluates:
                if score ≥ adaptive_threshold for 3 consecutive readings:
                  → reallocate: CRITICAL ↑, BACKGROUND ↓
                  → set rollback timer (5s)
                if score ≤ threshold × 0.7 for 3 readings:
                  → rebalance toward original allocations
                if rollback timer fires and CRITICAL latency unchanged:
                  → revert to pre-decision allocations
```

---

## Wire Protocol

The traffic generator sends real binary payloads — not JSON descriptions of hypothetical large packets. Measured throughput is actual bytes on the socket.

```
TCP frame:
  ┌────────┬────────┬────────┬──────────┬────────────────────┐
  │ seq 4B │ slice  │ proto  │ len 4B   │ timestamp 8B (dbl) │
  │        │ 1B     │ 1B     │          │                    │
  ├────────┴────────┴────────┴──────────┴────────────────────┤
  │ payload (len bytes — actual data, not a stub)            │
  └──────────────────────────────────────────────────────────┘

ACK:
  ┌────────┬────────────┐
  │ seq 4B │ status 4B  │  (0=OK, 1=dropped)
  └────────┴────────────┘

UDP: same header + payload in one datagram, capped at 1380B (no IP fragmentation)
```

Measured loopback RTT: **0.26 – 0.72ms** (10 frames, 512-4096B payloads).

TCP_NODELAY is set on CRITICAL flows to bypass Nagle buffering.

---

## ML Model

**CongestionLSTM** — trained on synthetic telemetry patterns before the demo starts.

| Property | Value |
|----------|-------|
| Input | 50 timesteps × 15 features (5 metrics × 3 slices) |
| Architecture | 2-layer LSTM(64) → Linear(32) → ReLU → Linear(1) → Sigmoid |
| Training | 2400 samples, 80/20 train/val, early stopping (patience=5) |
| Accuracy | ~92% on held-out data |
| Inference (CPU) | ~2ms single, ~15ms batch=128 |
| Inference (GPU) | ~0.4ms single, ~1.2ms batch=128 |

Features per timestep: `[latency/100, throughput/100, loss/100, jitter/50, bandwidth_ratio]` × 3 slices.

Training patterns:
- **Normal** (45%): low latency, low loss → label 0.05
- **Congested** (35%): ramping latency + loss → label 0.8–0.98
- **Transition** (20%): normal first half, congested second half → label 0.4–0.7

---

## Adaptive Thresholding

The decision engine does not use a fixed 0.70 threshold. It tracks the variance of the last 30 ML scores and adjusts:

```
std(scores) high → system is oscillating → raise threshold (reduce churn)
std(scores) low  → system is stable      → lower threshold (catch congestion early)

adjustment = (std - 0.15) × 0.5
threshold  = clamp(base + adjustment, 0.55, 0.85)
```

This prevents the classic control-system problem of oscillation under feedback — the system tunes itself.

---

## Demo Walkthrough

```bash
pip install torch rich "numpy<2"
python main.py
```

The automated demo runs five phases (~1 minute):

| Phase | Duration | What happens |
|-------|----------|-------------|
| 1. Baseline | 20s | All slices at nominal allocations, metrics stable |
| 2. Break | 8s | Latency spike +80ms injected, then 25% packet loss |
| 3. Detection | 15s | ML score climbs, SLA breaches logged, threshold crossed |
| 4. Recovery | 5s | Decision engine reallocates, faults cleared |
| 5. Proof | 15s | Metrics stabilize, before/after comparison printed |

At the end of Phase 5, the terminal prints a before/after table:

```
════════════════════════════════════════════════════════════════════
  BEFORE vs AFTER: automatic recovery results
════════════════════════════════════════════════════════════════════
  Metric                           Before       After       Delta
  ──────────────────────────────────────────────────────────────
  CRITICAL latency (ms)              82.1         1.8    ↓  80.3
  STANDARD latency (ms)             45.3         6.1    ↓  39.2
  BACKGROUND latency (ms)           28.4        22.0    ↓   6.4

  CRITICAL packet loss (%)          24.8         0.1    ↓  24.7
  STANDARD packet loss (%)          19.1         0.8    ↓  18.3
  BACKGROUND packet loss (%)        11.2         1.2    ↓  10.0

  Congestion score                  0.9234      0.1821
════════════════════════════════════════════════════════════════════
```

---

## REST Control API

Runs on `http://127.0.0.1:8080` (configurable with `--api-port`):

```bash
# System state
curl http://127.0.0.1:8080/api/status

# Live bandwidth allocations
curl http://127.0.0.1:8080/api/bandwidth

# Override allocations (JSON body)
curl -X POST http://127.0.0.1:8080/api/bandwidth \
     -d '{"CRITICAL": 70, "STANDARD": 20, "BACKGROUND": 10}'

# Inject fault (optional ?duration=N seconds)
curl -X POST "http://127.0.0.1:8080/api/fault/inject/latency_spike?duration=30"
curl -X POST "http://127.0.0.1:8080/api/fault/inject/packet_loss?duration=20"
curl -X POST "http://127.0.0.1:8080/api/fault/inject/slowdown?duration=15"

# Clear fault
curl -X POST http://127.0.0.1:8080/api/fault/clear

# Last 10 control-plane decisions
curl http://127.0.0.1:8080/api/decisions

# Prometheus metrics
curl http://127.0.0.1:8080/metrics
```

The API uses only stdlib `asyncio.start_server` — no aiohttp, no FastAPI. An async HTTP handler is ~80 lines of code.

---

## Prometheus Metrics

| Metric | Type |
|--------|------|
| `edge_net_x_slice_latency_ms{slice}` | gauge |
| `edge_net_x_slice_throughput_mbps{slice}` | gauge |
| `edge_net_x_slice_packet_loss_pct{slice}` | gauge |
| `edge_net_x_slice_bandwidth_mbps{slice}` | gauge |
| `edge_net_x_congestion_score` | gauge |
| `edge_net_x_sla_violations_total{slice}` | counter |
| `edge_net_x_decisions_total` | counter |
| `edge_net_x_rollbacks_total` | counter |
| `edge_net_x_ml_inferences_total` | counter |

---

## Benchmark

```bash
python benchmark.py
```

CPU results (M-series Mac):

```
batch=   1   avg=  1.843ms   throughput=   542 inf/s
batch=   8   avg=  3.217ms   throughput=  2487 inf/s
batch=  32   avg=  8.901ms   throughput=  3595 inf/s
batch= 128   avg= 28.134ms   throughput=  4548 inf/s
```

With an NVIDIA GPU the batch=128 throughput is approximately **20-25×** higher, reducing the decision loop latency from ~2ms to ~0.4ms per inference.

---

## CLI Options

```bash
python main.py                          # full automated demo
python main.py --no-demo                # run without demo scenario
python main.py --inject latency_spike   # inject fault 10s after start
python main.py --inject packet_loss
python main.py --inject slowdown
python main.py --api-port 9090          # custom API port
python benchmark.py                     # GPU vs CPU inference benchmark
```

---

## Project Structure

```
edge-net-x/
├── main.py              orchestrator + asyncio event loop
├── config.py            system configuration (slices, thresholds, ports)
├── models.py            core data types (NetworkPacket, SliceMetrics, etc.)
├── network/
│   ├── protocol.py      binary wire format (header, ACK encode/decode)
│   ├── server.py        async TCP + UDP server (real socket I/O)
│   └── client.py        flow generators (TCP constant/Poisson, UDP burst)
├── slicing/
│   └── scheduler.py     AsyncTokenBucket + SliceQueue + NetworkSlicer
├── telemetry/
│   └── collector.py     rolling-window metrics, ML feature vectors
├── ml/
│   ├── model.py         CongestionLSTM + synthetic training data
│   └── inference.py     GPU/CPU inference engine (ThreadPoolExecutor)
├── decision/
│   └── engine.py        adaptive threshold, hysteresis, rollback, SLA detection
├── simulation/
│   ├── fault.py         runtime fault injection (latency, loss, slowdown)
│   └── demo.py          5-phase story demo with before/after snapshots
├── api/
│   └── server.py        REST control plane API (stdlib asyncio)
├── metrics/
│   └── prometheus.py    Prometheus text format exporter
├── dashboard/
│   └── display.py       Rich terminal live dashboard (narrative timeline)
└── benchmark.py         GPU vs CPU inference throughput comparison
```

---
