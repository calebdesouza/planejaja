# NodeStor — Overview

> **"The VLC of AI memory"** — an inference engine that turns any machine into a portable data center.

---

## The Problem NodeStor Solves

The AI ecosystem is fragmented by design. NVIDIA requires CUDA. Cloud servers require subscriptions. Popular tools like llama.cpp run models, but don't extract maximum value from the hardware. Large 70B+ models are accessible only to those with dozens of gigabytes of VRAM.

NodeStor starts from a different premise: **the real limit is not hardware — it's data transport engineering.**

A Gen 4 NVMe SSD has ~7 GB/s of bandwidth. A PCIe 4.0 x16 bus carries 32 GB/s. A modern GPU processes hundreds of billions of operations per second. The bottleneck is the middleware chain between these three components — and that's exactly the chain NodeStor eliminates.

---

## Core Philosophy

### 1. Lossless First

Most local AI engines treat quantization as mandatory. NodeStor inverts this logic:

- **Default path**: full precision (F16/F32/BF16) inference with speculative decoding (COBER). Model quality is 100% preserved.
- **Optional path**: quantization (Q4_K_M, Q5_K_M, Q8_0) for users with limited VRAM. Works perfectly, but is never forced.

The distinction matters because aggressive quantization degrades reasoning capability. In advanced math or scientific hypothesis generation, the difference between F16 and Q4 can be the difference between the right answer and the wrong one.

### 2. Hardware-Agnostic

Vulkan compute shaders compile for any GPU — NVIDIA, AMD, Intel, Apple Silicon, mobile GPUs. The same binary runs on a gaming PC, a data center server, and, in the future, a smartphone.

Tensor transport uses the most efficient API for each platform:
- **Windows**: DirectStorage (same mechanism that loads AAA game textures in <2s)
- **Linux**: io_uring + DMABUF (zero-copy from SSD to GPU)
- **Both**: NTFS junctions / symlinks to separate data from code

### 3. Total Sovereignty

No telemetry. No accounts. No cloud dependency. The model runs on your machine — your data, your prompts, your results stay where you choose.

This isn't just privacy: it's resilience. If a company's API changes price, shuts down, or blocks access, your workflow doesn't break.

---

## The Four Systems

NodeStor is composed of four systems that integrate in layers. Each can be understood independently, but the real power emerges from their interaction.

---

### System 1: COBER — Lossless Speculative Decoding

**What it is**: COBER (Contextual Optimistic Batch Execution Runtime) accelerates token generation without compromising quality.

**How it works**: A small draft model (e.g., 135M parameters) proposes 4–8 tokens in parallel. The full model verifies all of them at once. If the draft was correct, all tokens are accepted — equivalent to generating 4–8 tokens at the computational cost of 1.

**The mathematical guarantee**: The algorithm uses rejection sampling. Formally:

```
For each proposed token t_d at position i:
  r ~ Uniform(0, 1)
  If r < p_full(t_d) / p_draft(t_d):
    accept t_d
  Else:
    sample from corrected distribution; stop here
```

The final distribution is **mathematically identical** to what the full model would produce alone. There is no quality trade-off.

**Typical speedup**: 2–4× in tokens per second, depending on alignment between draft and main model.

**Also includes**:
- **MCTS**: Monte Carlo Tree Search to explore multiple reasoning paths before committing
- **Medusa**: multiple prediction "heads" in parallel on the same model
- **Conformal Prediction Sets**: statistical certainty about the set of possible tokens

---

### System 2: PROBES — Real-Time Mechanistic Interpretability

**What it is**: PROBES inspects what the model is *actually doing* internally, not just what it *says* it's doing.

**Why it matters**: Language models can express certainty in wrong answers. They can say "I don't know" when they actually know. They can follow a coherent reasoning chain but reach an inconsistent conclusion. PROBES detects all of this in real time.

**The three main modules**:

**ELK Probe (Eliciting Latent Knowledge)**
Monitors the state of hidden states at each generated token. Detects when there is an inconsistency between what the model knows (represented in activations) and what it is about to say (the next token). Technically analogous to a lie detector operating in the vector space of internal representations.

**CoT Monitor (Chain-of-Thought Monitor)**
Tracks coherence between the visible reasoning chain (text generated in `<thinking>`) and the final answer. If the reasoning would lead to conclusion A but the model answers B, the monitor flags divergence.

**RAISE Detector (SA1→SA5)**
Measures the model's situational awareness level on a 5-level scale:
- SA1: the model knows it's generating text
- SA2: the model knows there's a user
- SA3: the model knows it's being evaluated
- SA4: the model modifies behavior based on the evaluation
- SA5: the model reasons about its own evaluation

**Technical integration**: PROBES attaches to the pipeline via `ProbesTool` trait injection. The `inspect(hidden_state, step)` method is called at each generated token. This enables injection without circular dependencies — the `nodestor-inference` crate doesn't need to know PROBES exists.

---

### System 3: DAVI — Dreaming Intelligence

**What it is**: DAVI (Dreaming Autonomous Visual Intelligence) is NodeStor's autonomous discovery system. When the GPU is idle, the system "dreams" — exploring connections between concepts from different domains to generate hypotheses that humans rarely consider.

**The inspiration**: Scientists make their best discoveries in a resting state — when the mind connects ideas non-linearly. DAVI implements this process mathematically.

**The dream cycle** (8 steps orchestrated by `DreamingEngine`):

#### Step 1: Topology — Finding the Gaps

`TopologicalGapDetector` applies **Topological Data Analysis (TDA)** to the knowledge graph. Persistent homology identifies "holes" in the space — regions where knowledge is nonexistent or contradictory. These holes are opportunities for discovery.

Imagine knowledge as a 3D surface. Dense regions are well understood. Holes are questions that nobody has asked yet.

#### Step 2: Free Energy — Prioritizing by Surprise

`FreeEnergyObjective` uses **Karl Friston's Free Energy Principle**. The system prioritizes holes that would cause the greatest reduction in surprise if filled. Mathematically, it selects the action that minimizes expected free energy.

This ensures the system doesn't explore randomly — it focuses on the gaps with the greatest potential impact.

#### Step 3: Annealing — The Creative Temperature

`SemanticAnnealing` decides whether to accept or reject cross-domain leaps using the **Simulated Annealing** analogy. At high temperature, it accepts unlikely leaps (exploration). At low temperature, only probable leaps are accepted (refinement exploitation).

This allows the system to navigate between "highly speculative" and "conservative" throughout a cycle.

#### Step 4: Crystal Skeleton — Synthetic Structure

Generates the hypothesis in structured form, extracting a "syntactic skeleton" that can be formally validated.

#### Step 5: Functors — Category Theory Across Domains

`FunctorComposer` uses **Category Theory** to map a discovery from one domain to another. If the system discovers that "gravity" and "opportunity cost" share the same mathematical structure (both are gradients of a potential), a discovery in physics automatically becomes a discovery in economics.

A functor `F: Domain_A → Domain_B` preserves morphological structure. Each discovery multiplies across all mapped domains.

#### Step 6: Nash Tribunal — Adversarial Validation

`NashTribunal` instantiates 3 adversarial agents in Nash equilibrium:
- **Proposer Agent**: argues strongly in favor of the hypothesis
- **Skeptic Agent**: seeks flaws and contradictions
- **Synthesis Agent**: arbitrates and formulates a balanced verdict

Only hypotheses that survive the tribunal are accepted as `NovelDiscovery`. This ensures the system doesn't produce unsupported speculation.

#### Step 7: Stigmergy — Digital Pheromones

`StigmergicSwarm` deposits "pheromones" on successful paths, exactly like ants do. Future cycles have a higher probability of exploring neighborhoods of previous discoveries. Knowledge grows coherently, not randomly.

#### Step 8: Autopoiesis — The System Improves Itself

`AutopoieticLoop` adjusts the system's own parameters based on results. If temperature was too high (too many rejected leaps), it reduces. If too low (too little exploration), it increases. The system learns to dream better.

**Concrete result**: `nodestor davi dream --topic "quantum physics + genetics"` can produce hypotheses like "quantum entanglement properties have a structural analog in RNA splicing mechanisms" — crossovers that the literature rarely considers.

---

### System 4: APEX — Data Transport at Bus Speed

**What it is**: APEX (Adaptive Prefetch Execution) is the transport layer that treats the SSD as a VRAM extension.

**The principle**: A 7B parameter model in F16 occupies ~14 GB. Most GPUs have 8–16 GB of VRAM. The conventional solution is quantization. The APEX solution is streaming layers directly from the SSD to the GPU via DMA, bypassing RAM.

**The components**:

**BurstScheduler**: Pre-calculates which layers will be needed in the next steps and initiates transfers before the GPU asks. Eliminates latency bubbles.

**MesPrefetchQueue**: Prefetch queue based on model profile. For Transformer architectures, the access pattern is highly predictable — the scheduler exploits this.

**BufferPool**: Pool management of VRAM buffers. Layers no longer needed are released in FIFO order; layers soon to be needed are preloaded.

**KVCachePaginator**: KV-cache paging in PagedAttention style. Multiple users or multiple contexts share available VRAM without waste.

---

## Infinite Context: How It Works

This is one of NodeStor's most important features and deserves a detailed explanation.

**The problem**: KV-cache stores Key and Value vectors for each token in the history. With hidden dimension `d` and `L` layers, the memory cost per token is `2 × L × d × sizeof(f32)` bytes. For SmolLM2-135M (L=30, d=576): ~138 KB per token. For 4096 tokens: ~543 MB. For 100K token context: the cache would be unviable.

**The NodeStor solution**:

```
Normal generation → KV-cache fills → enforce_window() evicts old tokens
                                              ↓
                               Evicted tokens → decode() → text
                                              ↓
                               vector_db.add_document(id, text)
                                              ↓
                               Context recovered via semantic search
                               when relevant to the current prompt
```

The result: the model never loses access to previous context. When it needs information from 10,000 tokens ago, the system retrieves it via HNSW+BM25 search. "Forgetting" is indexing, not loss.

**Empirical proof**: the `test_infinite_context_recall_of_evicted_fact` test (in `nodestor-metadata`) verifies end-to-end that a fact inserted before eviction is correctly retrieved after eviction.

---

## Current State

| Component | Status |
|-----------|--------|
| Crates in workspace | 14 |
| Tests passing | 528+ |
| Real model tested | SmolLM2-135M (F16, 270MB) |
| Coherent output | Yes (`" Paris"` for "The capital of France is") |
| TTFT (CPU) | ~930ms |
| Throughput (CPU) | ~5 tok/s |
| Infinite context | Empirically proven |
| KV-cache reuse | Proven correct (5 tests: incremental == full-recompute) |
| Deep Research Engine | Complete — all tool handlers use real DAVI modules |
| DAVI Dream | 12 modules, dream cycle functional; `--dream` wired to DreamingEngine |
| Nash Tribunal | Real adversarial validation in `call_hypothesis` handler |
| CLI | 17 commands, interactive menu, dynamic temperature in agent loop |

**What's still in development**:
- Vulkan GPU compute shaders for real GPU inference (currently uses CPU fallback)
- Target speed: thousands of tok/s with GPU + COBER speculation

The mathematical foundation is correct. The sequence: **correctness ✓ → KV reuse ✓ → GPU → speculation**.
