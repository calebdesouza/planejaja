# NodeStor — Architecture

## Crate Dependency Graph

```
nodestor-cli  ──────────────────────────────────────────────┐
    │                                                         │
    ├── nodestor-inference  ←── ALL inference logic           │
    │       ├── pipeline.rs        (7-layer orchestrator)     │
    │       ├── cober.rs           (speculative decoding)     │
    │       ├── kv_cache.rs        (H2O + infinite context)   │
    │       ├── agent_loop.rs      (Deep Research Engine)     │
    │       ├── tool_registry.rs   (tool calling by tag)      │
    │       ├── cpu_reference.rs   (correct Llama forward)    │
    │       ├── sampler.rs         (top-k, top-p, temp, rep)  │
    │       ├── lora_core.rs       (LoRA delta injection)     │
    │       ├── steering_engine.rs (activation steering)      │
    │       ├── persistent_memory.rs (episodic memory)        │
    │       └── 50+ other modules                             │
    │                                                         │
    ├── nodestor-davi  ←── Dreaming Intelligence              │
    │       ├── dreaming_engine.rs (D9 orchestrator)          │
    │       ├── topology.rs        (TDA gap detection)        │
    │       ├── free_energy.rs     (Friston FEP)              │
    │       ├── annealing.rs       (semantic temperature)     │
    │       ├── nash_tribunal.rs   (3-agent validation)       │
    │       ├── functors.rs        (category theory)          │
    │       ├── stigmergy.rs       (swarm pheromones)         │
    │       ├── autopoiesis.rs     (self-improvement)         │
    │       ├── elk_probe.rs       (lie detector)             │
    │       ├── cot_monitor.rs     (CoT divergence)           │
    │       └── raise_detector.rs  (situational awareness)    │
    │                                                         │
    ├── nodestor-metadata  ←── Vector DB + RAG                │
    │       ├── vector_store.rs    (HNSW + BM25 + RRF)        │
    │       └── search.rs          (RAG interface)            │
    │                                                         │
    ├── nodestor-vulkan   ←── GPU compute                     │
    ├── nodestor-streaming ←── APEX transport + COBER         │
    ├── nodestor-formats  ←── GGUF + SafeTensors parsing      │
    ├── nodestor-scanner  ←── Hardware detection              │
    └── nodestor-core     ←── Shared types, errors, config    │
                                                              │
nodestor-server  ←── HTTP API (axum) ─────────────────────────┘
```

## Dependency Rules

- `nodestor-inference` is the hub. It does NOT import `nodestor-davi`.
- `nodestor-davi` imports `nodestor-inference` (uses `semantic_attention`, pipeline types).
- PROBES attach to the pipeline via `ProbesTool` trait injection (no circular dep).
- CLI imports everything; it's the integration point.

## Infinite Context

When the KV cache fills, `enforce_window()` evicts the oldest tokens. The evicted token IDs are decoded to text and indexed in the vector DB (`VectorSearch`). On retrieval, semantically similar past content is prepended to the prompt — giving the model access to arbitrarily long history within a fixed VRAM budget.

```
generate() loop:
  → forward step (Vulkan or CPU)
  → sample token
  → check KV window
    → if evicted: decode tokens → vector_db.add_document()
  → stream token to user
  → check EOS
```

## Agent Loop (Deep Research)

```
AgentExecutionLoop::run()
  loop (max_loops):
    → generate text (pipeline.generate_stream)
    → ToolRegistry::scan_for_call(text)
      → None:  terminate (DirectAnswer)
      → Some:  invoke(tag, query) → ToolResult
                → inject tool_response into context
                → adapt_temperature (stagnation check)
                → next loop
```

Tool tags emitted by the model:
- `<call_vector_db>query</call_vector_db>` → semantic search
- `<call_dream>domain1+domain2</call_dream>` → DAVI cross-domain synthesis
- `<call_think>reasoning</call_think>` → internal reflection
- `<call_hypothesis>claim</call_hypothesis>` → Nash Tribunal validation

## DAVI Dream Cycle

```
DreamingEngine::dream_cycle(embeddings, insights):
  1. TopologicalGapDetector  → find knowledge gaps (persistent homology)
  2. FreeEnergyObjective     → select highest-surprise gap
  3. SemanticAnnealing       → accept/reject cross-domain jump (temperature)
  4. NashTribunal            → 3 agents debate the hypothesis
  5. FunctorComposer         → map discovery to other domains
  6. StigmergicSwarm         → deposit pheromone on successful path
  7. AutopoieticLoop         → update system health / exploration bias
```

## Activation Steering

When `--steer-vector` is used, the pipeline applies Orthogonal Dynamic Projection at every residual stream:

```
h_new = h - (h · d̂) * d̂ * intensity
```

Where `d̂` is the normalized direction vector extracted by `calibrate`. This modifies the geometry of all hidden states without touching model weights.

DSCP (`--auto-steer`) generates the direction vector at runtime from contrastive activation templates — no external dataset needed.

## Token Speculation (COBER)

```
COBER::verify_and_accept_probabilistic(draft_tokens, logits):
  for each draft token t_d at position i:
    r ~ Uniform(0, 1)
    if r < p_model(t_d) / p_draft(t_d):
      accept t_d    ← lossless
    else:
      sample from corrected distribution
      stop
```

Speedup = mean accepted tokens per step (typically 2–4×).
