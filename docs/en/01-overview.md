# NodeStor — Overview

NodeStor is a local AI inference engine. Its core principle: **run any open-weight model at full quality, on any hardware, with no cloud and no proprietary dependencies.**

## Philosophy

Most AI tools force a choice: quality *or* accessibility. NodeStor refuses both extremes.

- **Lossless first**: The base path is full-precision (F16/F32/BF16) inference with speculative decoding (COBER). Quantization (Q4/Q5/Q8) is an opt-in convenience — never a requirement.
- **Hardware-agnostic**: Vulkan compute shaders run identically on NVIDIA, AMD, Intel, and mobile GPUs. DirectStorage (Windows) and io_uring (Linux) bypass the CPU for zero-copy tensor streaming.
- **Sovereign**: No telemetry, no accounts, no servers. The model runs on your machine, your data stays local.
- **Infinite context**: KV-cache eviction with vector-DB recall means the model never forgets — evicted tokens are indexed semantically and retrieved via HNSW+BM25.

## The Four Systems

### COBER — Lossless Speculation Engine
Token generation accelerated by speculative decoding. A small draft model proposes token sequences; the full model verifies them in parallel (EAGLE-2 style). Mathematically guaranteed: the output distribution is identical to greedy/sampling from the full model alone.

Also includes: MCTS tree search, Medusa multi-head speculation, conformal prediction sets.

### PROBES — Interpretability Layer
Inspects the model's internal representations in real time during generation:
- **ELK Probe**: Detects factual inconsistency between hidden states and output tokens (the "lie detector").
- **CoT Monitor**: Flags when the chain-of-thought diverges from the final answer.
- **RAISE Detector**: Measures situational awareness levels SA1→SA5 (does the model know it's being evaluated?).

PROBES runs as a trait-object injected into the pipeline — zero circular dependencies.

### DAVI — Dreaming Intelligence
12 cognitive modules that give NodeStor autonomous discovery capabilities:

| Module | Role |
|--------|------|
| `topology` | Finds gaps in the knowledge space via Persistent Homology (TDA) |
| `free_energy` | Prioritizes exploration using Friston's Free Energy Principle |
| `annealing` | Semantic Annealing — temperature controls creative exploration |
| `nash_tribunal` | 3 adversarial agents validate every hypothesis |
| `functors` | Category Theory maps discoveries across domains |
| `stigmergy` | Digital pheromones guide swarm-like exploration |
| `autopoiesis` | The system improves its own search strategy |
| `dreaming_engine` | Orchestrates all 7 subsystems into one dream cycle |
| `latent_jump` | Leaps between distant regions of latent space |
| `intent_compiler` | Compiles user intent into an execution plan |
| `elk_probe` | Lie detection at the activation level |
| `raise_detector` | Situational awareness monitoring |

The dream cycle runs when the GPU is idle. It detects topological gaps in the knowledge graph, generates cross-domain hypotheses, and validates them through Nash equilibrium between adversarial agents.

### APEX — Transport Layer
The hardware abstraction that makes "infinite context on any SSD" real:
- NTFS junction / io_uring for streaming model weights directly from disk
- Vulkan compute shaders for tensor operations
- KV-cache paging (PagedAttention-style) for multi-tenant scenarios
- Hardware-adaptive routing: detects SSD generation, GPU vendor, available VRAM

## Key Metrics (SmolLM2-135M on CPU)

| Metric | Value |
|--------|-------|
| TTFT | ~930ms |
| Throughput | ~5 tok/s (CPU) |
| Context | Infinite (eviction + vector recall) |
| Tests passing | 296 |

GPU path via Vulkan is the primary performance target — CPU is the simulation/compatibility path.
