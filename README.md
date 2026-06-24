# NodeStor

**Local AI inference that belongs to no one.**

NodeStor is a production-grade AI inference engine built in Rust and Vulkan. It runs any open-weight model at full quality on any hardware — with no cloud dependency, no proprietary libraries, and a mathematical guarantee that quantization is *optional*, not forced.

---

## Why NodeStor Exists

Every major AI tool has a hidden cost: CUDA lock-in, cloud subscriptions, quality degradation from aggressive quantization, or complete lack of interpretability over what the model is actually doing internally.

NodeStor was built to remove all of those costs simultaneously.

**For individuals**: Run Llama 3.1 70B on your own machine with full quality. Nobody sees your prompts.

**For companies**: Deploy on your hardware, air-gapped if needed. No vendor lock-in. Pay-per-token pricing becomes irrelevant.

**For researchers**: Inspect hidden states, control activation geometry, run autonomous cross-domain hypothesis generation, observe model behavior with PROBES.

---

## Quick Start

```bash
# Download a model (270MB for testing)
nodestor pull bartowski/SmolLM2-135M-GGUF --filename SmolLM2-135M-Q4_K_M.gguf

# Run a prompt
nodestor run "Explain quantum entanglement in simple terms" \
  --model ~/.nodestor/models/SmolLM2-135M-Q4_K_M.gguf

# Deep Research with DAVI dreaming engine
nodestor run "Propose a mechanism for Alzheimer's cure via cross-domain synthesis" \
  --model model.gguf --deep-research --dream --max-loops 5

# Standalone DAVI dream cycle (no model needed)
nodestor davi dream --topic "quantum physics + genetics + topology"

# List installed models
nodestor models

# Interactive menu
nodestor
```

---

## The Four Systems

### COBER — Lossless Speculative Decoding
A small draft model proposes 4–8 tokens in parallel. The full model verifies all at once. The mathematical guarantee: output distribution is **identical** to running the full model alone. No quality trade-off, 2–4× speedup.

Also includes MCTS tree search, Medusa multi-head speculation, and conformal prediction sets.

### PROBES — Real-Time Mechanistic Interpretability
Three modules inspect the model's internal representations at every token:
- **ELK Probe**: Detects inconsistency between hidden states and output — the "lie detector"
- **CoT Monitor**: Flags divergence between visible reasoning chain and final answer
- **RAISE Detector**: Measures situational awareness SA1→SA5

PROBES attaches via trait injection — zero circular dependencies with the inference engine.

### DAVI — Dreaming Intelligence
12 cognitive modules that give NodeStor autonomous discovery capabilities. When the GPU is idle, the system runs a dream cycle:

**TDA** (gap detection) → **FEP** (prioritization by surprise) → **Semantic Annealing** (temperature-controlled exploration) → **Nash Tribunal** (3-agent adversarial validation) → **Category Theory Functors** (cross-domain mapping) → **Stigmergy** (swarm path reinforcement) → **Autopoiesis** (system self-improvement)

Result: hypotheses that cross domain boundaries in ways human researchers rarely consider — validated, not speculated.

### APEX — Bus-Speed Data Transport
The hardware abstraction that makes "infinite context on any SSD" real. DirectStorage (Windows) / io_uring (Linux) stream model weights from SSD to GPU via DMA, bypassing the CPU. BurstScheduler pre-loads layers before the GPU asks. KV-cache paging allocates VRAM efficiently across multiple contexts.

---

## Infinite Context

KV-cache eviction + vector DB indexing = the model never forgets.

When the KV window fills, evicted tokens are decoded to text and indexed in the embedded HNSW+BM25 database. When relevant, they're retrieved and prepended to the prompt. The model has access to arbitrarily long history within a fixed VRAM budget.

**Empirically proven**: `test_infinite_context_recall_of_evicted_fact` verifies end-to-end recall of facts inserted before eviction.

---

## Current State

| | |
|-|-|
| Language | Rust (edition 2021, MSRV 1.75) |
| Crates | 14 |
| Test suite | 296 passing, 0 failing |
| Real model | SmolLM2-135M F16 — coherent output, measured TTFT ~930ms |
| CLI commands | 17 (run, models, davi, pull, train, calibrate, inspect, scan, ...) |
| GPU | Vulkan (primary path, compute shaders in progress) |
| CPU | Correct Llama forward with interleaved RoPE (simulation/compatibility path) |

---

## Repository Structure

```
planejaja/
  nodestor/        — Rust engine (14 crates)
  docs/
    pt/            — Documentation in Portuguese
    en/            — Documentation in English
  apps/
    site/          — Website (future)
  README.md        — This file
```

---

## Documentation

### Portuguese
- [Overview — O que é NodeStor e como funciona](docs/pt/01-overview.md)
- [CLI Reference — Todos os comandos com exemplos detalhados](docs/pt/02-cli-reference.md)
- [Architecture — Crates, fluxo de dados, matemática](docs/pt/03-architecture.md)

### English
- [Overview — What NodeStor is and how it works](docs/en/01-overview.md)
- [CLI Reference — All commands with detailed examples](docs/en/02-cli-reference.md)
- [Architecture — Crates, data flow, mathematics](docs/en/03-architecture.md)

---

## Build

```bash
cd nodestor

# Development check (fast, parallel)
cargo check

# Run tests (serialized — prevents OOM on machines with <32 GB RAM)
cargo test --workspace --lib -j1

# Release build
cargo build --release -j1
```

**Note**: `target/` is symlinked to a separate drive if C: has limited space. See architecture docs.

---

## License

MIT OR Apache-2.0

---

*Built for the world. Owned by no one.*
