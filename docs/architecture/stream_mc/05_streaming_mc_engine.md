# 05. Streaming Monte Carlo Engine (`stream_mc`)

## Current scientific path — DLG-SelectiveStream R01 (2026-10-11)

The authoritative R01 implementation is `src/gog_fraud/streaming/selective.py`,
`selective_engine.py` and `selective_state.py`, with project-owned configurations,
experiments and evidence under `projects/stream_mc`. Relations use the shared
`data/level2/relation_builder.py` historical reference index; no hybrid graph
artifact is consumed. Primary inference is deterministic GIN → frozen margin
router → optional GATv2 independent similarity star. MC is an ablation.

This is retrospective contract-snapshot classification plus bounded-state
systems replay. Cached edge-time lineage and label availability are unknown;
training/calibration can postdate historical targets. Zero future-reference
retrieval violations do not establish full temporal validity. No live AML
safety, <50ms SLA, universal tail-latency improvement, global exactly-once,
dynamic online adaptation or whole-process constant-memory guarantee is made.

Declared engine boundaries:5000 resident contracts,128 nodes/edges per contract,
90event-day TTL,10000/64MiB immutable-feature cache,512 retry queue and1000 trace
buffer. Immutable train references/search index/model, input preload, population
accounting and checkpoint copies are separate structures. Logical byte budget,
serialized bytes, RSS and GPU allocated/reserved bytes are distinct measurements.
References can exceed cache capacity; misses reload immutable arrays. Snapshot
degree features and live replay updates use the same three degree observables,
not amount/velocity features described in the historical concept below.

R01 uses25 independent model identities, six fixed budgets/four routers,
separate contract/prefix/long workloads, actual cap crossing/churn/cache/TTL/queue
tests and abrupt local checkpoint recovery. Exact configs, raw provenance,
failure scope and reproduction commands are in `projects/stream_mc/protocols`
and `REPRODUCE.md`. Public scientific evidence excludes unsubmitted paper sources.

## Historical concept (not the validated R01 path)

The following sections preserve an earlier aspirational streaming-AML design.
Their performance targets, synthetic diagram and legacy command are not R01
experimental results or production guarantees, and must not be cited as such.

This historical document outlined **`stream_mc`** as a dynamic streaming graph
anomaly/AML engine intended for real-time financial streams.

---

## 1. Motivation & Operational Requirements

In production financial environments, transactions arrive as high-velocity event streams (thousands of transactions per second). Standard batch GNN models cannot be retrained continuously from scratch. `stream_mc` solves this with a **bounded-state streaming architecture** satisfying:
1. **Constant-Bound Memory:** Memory footprint must not grow unboundedly with stream length $T$.
2. **Sub-Second Latency:** Per-transaction feature extraction, localized ego-net retrieval, and score inference in $< 50\text{ ms}$.
3. **Temporal Dynamic Updating:** Node embeddings and relation states must update incrementally as new transactions occur.

---

## 2. Streaming Architecture Overview

```mermaid
flowchart LR
    StreamInput["Transaction Stream<br>(Source, Target, Amount, Timestamp)"] --> QueueMgr["Queue Manager<br>(Event Buffering & Order)"]
    QueueMgr --> Engine["Streaming Engine<br>(src/gog_fraud/streaming/engine.py)"]

    subgraph StateStore["Bounded-State In-Memory Storage"]
        SubStore["Subgraph Store<br>(Active k-hop Neighborhoods)"]
        RelState["Relation State<br>(Temporal Edge Decay & Freq)"]
        EmbCache["Embedding Cache<br>(LRU Cached Node Vectors)"]
    end

    Engine <--> StateStore

    Engine --> ModelInfer["Incremental DLG Inference<br>(Local Ego-Net Scoring)"]
    ModelInfer --> Router["AML Selection Router<br>(Priority Queue & Triage)"]
    Router --> Alerts["AML Case Alerts<br>(High-Risk Transactions)"]
```

---

## 3. Core Component Modules

### 3.1 Streaming Engine (`src/gog_fraud/streaming/engine.py`)
The orchestrator managing incoming transaction events:
- Receives transaction tuples: `(src_id, dst_id, amount, timestamp, features)`.
- Fetches active neighborhood subgraphs from `subgraph_store`.
- Computes dynamic node feature updates (e.g., sliding window transaction velocity, amount variance).
- Calls incremental inference and updates the `embedding_cache`.

### 3.2 Bounded-State Subgraph Store (`src/gog_fraud/streaming/subgraph_store.py`)
Maintains localized $k$-hop subgraphs within a bounded sliding window:
- Evicts edges older than $t - \Delta t_{\text{window}}$ to ensure memory bounds:
$$E_{\text{active}}(t) = \{ (u, v, \tau) \in E \mid t - \Delta t_{\text{window}} \le \tau \le t \}$$
- Employs indexed adjacency lists with temporal pointers for $O(1)$ edge insertion and efficient neighborhood extraction.

### 3.3 Dynamic Relation State (`src/gog_fraud/streaming/relation_state.py`)
Tracks behavioral relationship dynamics between accounts:
- Accumulates exponential moving averages (EMA) of transaction intensity:
$$R_{uv}(t) = R_{uv}(t_{\text{prev}}) \cdot e^{-\lambda (t - t_{\text{prev}})} + w_{\text{tx}}$$
where $\lambda$ is the temporal decay factor and $w_{\text{tx}}$ is the normalized transaction weight.
- Detects bursty structuring (smurfing patterns) and sudden fan-out/fan-in spikes.

### 3.4 In-Memory Embedding Cache (`src/gog_fraud/streaming/embedding_cache.py`)
- Least-Recently-Used (LRU) cache storing Level-1 and Level-2 representations.
- Invalidates and recomputes embeddings only for nodes within the 1-hop neighborhood of a new transaction, avoiding full-graph re-encoding.

### 3.5 Selection & Triage Router (`src/gog_fraud/selection/router.py`)
Prioritizes transactions for compliance review:
- Routes transactions into risk tiers based on composite scores:
$$\text{Priority} = w_1 \cdot s_{\text{DLG}} + w_2 \cdot R_{\text{burst}} + w_3 \cdot \text{Amount}$$
- Low-risk transactions pass through without blocking; suspicious transactions trigger immediate alert payloads.

---

## 4. Replay & Simulation Pipeline

To validate streaming performance on historical logs, `stream_mc` includes a deterministic replay runner:

```bash
python -m src.gog_fraud.pipelines.run_streaming_replay \
    --dataset bitcoin_otc \
    --window-size 3600 \
    --batch-size 64
```

This simulates live Kafka streams from static datasets and measures latency, memory consumption, and detection accuracy under realistic operational conditions.
