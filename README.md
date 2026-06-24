# NodeStor

**The AI engine that runs anywhere, belongs to no one.**

NodeStor is a local AI inference engine written in Rust and Vulkan — sovereign, lossless, and built for the edge. It runs open-weight models at full quality on any hardware (NVIDIA, AMD, Intel, mobile) with no cloud dependency, no proprietary libraries, and no quality compromises.

---

## Repository Structure

```
planejaja/
  nodestor/        — Rust engine (14 crates, the core)
  docs/
    pt/            — Documentation in Portuguese
    en/            — Documentation in English
  apps/
    site/          — Website (future)
```

## Quick Start

```bash
# Download a model
nodestor pull bartowski/SmolLM2-135M-GGUF --filename SmolLM2-135M-Q4_K_M.gguf

# Run a prompt
nodestor run "What is quantum entanglement?" --model ~/.nodestor/models/SmolLM2-135M-Q4_K_M.gguf

# Run with Deep Research Engine (DAVI)
nodestor run "Propose a novel cancer treatment" --model model.gguf --deep-research --dream

# Run DAVI dream cycle standalone
nodestor davi dream --topic "quantum physics + genetics + economics"

# List installed models
nodestor models

# Interactive menu
nodestor
```

## The Four Systems

| System | What it does |
|--------|-------------|
| **COBER** | Lossless speculative decoding — EAGLE-2 + MCTS + rejection sampling |
| **PROBES** | ELK lie detector, CoT monitor, situational awareness (SA1→SA5) |
| **DAVI** | 12-module dreaming engine — discovers hypotheses humans haven't thought of |
| **APEX** | Transport layer — DirectStorage/io_uring, Vulkan compute, zero-copy DMA |

## Documentation

- [Overview (PT)](docs/pt/01-overview.md)
- [CLI Reference (PT)](docs/pt/02-cli-reference.md)
- [Architecture (PT)](docs/pt/03-architecture.md)
- [Overview (EN)](docs/en/01-overview.md)
- [CLI Reference (EN)](docs/en/02-cli-reference.md)
- [Architecture (EN)](docs/en/03-architecture.md)

## Build

```bash
cd nodestor
cargo build --release -j1   # -j1 prevents OOM on 16 GB machines
cargo test --workspace --lib -j1
```

## License

MIT OR Apache-2.0
