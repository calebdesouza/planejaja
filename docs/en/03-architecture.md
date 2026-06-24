# NodeStor — Technical Architecture

This document explains how NodeStor's 14 crates are organized, how data flows between them, and the architectural decisions that make the system possible.

---

## Crate Structure

```
nodestor/
  nodestor-cli/           ← Main binary (nodestor) + benchmarks
  nodestor-server/        ← HTTP server (axum) with REST API
  nodestor-integration-tests/  ← End-to-end tests
  crates/
    nodestor-core/        ← Shared types, errors, config
    nodestor-scanner/     ← Hardware detection (GPU, CPU, SSD, VRAM)
    nodestor-formats/     ← GGUF v1/v2/v3 + SafeTensors parser
    nodestor-transport/   ← High-performance async I/O
    nodestor-vulkan/      ← Vulkan engine (compute shaders + CPU pipeline)
    nodestor-streaming/   ← APEX: BurstScheduler, BufferPool, prefetch
    nodestor-metadata/    ← HNSW+BM25 vector database + indexing
    nodestor-inference/   ← Complete inference pipeline (60+ modules)
    nodestor-davi/        ← Dream engine (12 cognitive modules)
    nodestor-python/      ← Python bindings via maturin/PyO3
    nodestor-fabric/      ← Multi-node orchestration (future)
    nodestor-gdeflate/    ← GPU-accelerated GDeflate compression
```

---

## Dependency Graph

The fundamental rule: **no base crate may depend on an application crate.**

```
nodestor-core  (no internal deps)
      ↑
nodestor-scanner  nodestor-formats  nodestor-gdeflate
      ↑                  ↑
nodestor-transport
      ↑
nodestor-vulkan  nodestor-metadata
      ↑                  ↑
nodestor-streaming
      ↑
nodestor-inference  ←── central hub (60+ modules)
      ↑
nodestor-davi  ←── depends on inference (not the other way around)
      ↑
nodestor-cli / nodestor-server  ←── integrate everything
```

**Critical PROBES rule**: `nodestor-inference` does NOT import `nodestor-davi`. PROBES is coupled via trait injection:

```rust
// In nodestor-inference/src/pipeline.rs
pub trait ProbesTool: Send + Sync {
    fn inspect(&self, hidden: &[f32], step: usize);
}

pub struct InferencePipeline {
    probes: Option<Arc<dyn ProbesTool>>,
    ...
}
```

`DaviProbesTool` (in `nodestor-davi`) implements this trait and is injected by the CLI via `pipeline.with_probes_tool(davi_tool)`. Zero circular dependency.

---

## Inference Pipeline (7 Layers)

```
Prompt (text)
      │
      ▼ Layer 1: Tokenization
  TokenizerManager::encode()
  ┌─ GGUF tokenizer (byte-level BPE)
  └─ HuggingFace tokenizers (fallback)
      │
      ▼ Layer 2: Weight Loading
  WeightStore + GGUF/SafeTensors parser
  ┌─ F32/F16/BF16 → f32 directly (lossless)
  └─ Q8_0/Q4_K/Q5_K → DequantDispatcher
      │
      ▼ Layer 3: KV-Cache
  KVCacheWindow (H2O eviction)
  ┌─ Sliding window with fixed VRAM budget
  ├─ Eviction: heap(score) → FIFO → sink (last resort)
  └─ Evicted tokens → vector_db.add_document()
      │
      ▼ Layer 4: Forward Pass
  [GPU] VulkanEngine::dispatch_attention() / dispatch_turbo_quant_attention()
  [CPU] cpu_reference::forward_last_logits()
  ┌─ RMSNorm → SwiGLU → RoPE → Causal Attention → Output
  └─ GQA (Grouped Query Attention) when num_kv_heads < num_heads
      │
      ▼ Layer 5: Sampling
  Sampler { temperature, top_k, top_p, repetition_penalty }
  ┌─ Temperature 0 → greedy (argmax)
  └─ Top-K → Top-P (nucleus) → WeightedIndex sample
      │
      ▼ Layer 6: Streaming
  mpsc::Sender<Result<String>>
  ┌─ Tokens sent via channel to the client
  └─ Metrics: TTFT, tok/s, total
      │
      ▼ Layer 7: EOS Detection
  token_id == eos_token_id → terminate
```

---

## Infinite Context: Detailed Implementation

### The Problem

KV-cache stores Key and Value vectors for each token in history. With hidden dimension `d` and `L` layers, the memory cost per token is `2 × L × d × sizeof(f32)` bytes. For SmolLM2-135M (L=30, d=576): ~138 KB per token. For 4096 tokens: ~543 MB. For a 100K token context: the cache would be unviable.

### The Solution

```rust
// In kv_cache.rs — generation loop
let evicted = main_kv.enforce_window();
if evicted > 0 {
    for j in 0..evicted {
        if let Some(&t) = full_seq.get(window_start + j) {
            evict_buf.push(t);
        }
    }
    window_start += evicted;
    if evict_buf.len() >= 48 {
        let text = tokenizer.decode(&evict_buf, true).unwrap_or_default();
        if !text.trim().is_empty() {
            let id = format!("ctx_{}", window_start);
            let _ = self.vector_db.add_document(&id, &text).await;
        }
        evict_buf.clear();
    }
}
```

When the KV-cache fills, `enforce_window()` removes the lowest-scored tokens from the heap. Those token IDs are accumulated in `evict_buf`. When the buffer reaches 48 tokens (~32 words), they're decoded to text and indexed in the vector database.

### Retrieval

When the user asks a question related to evicted information, the HNSW+BM25 search retrieves the relevant text and it's prepended to the prompt:

```
[Retrieved context]: "The capital of France is Paris..."
<|im_start|>user
Which country has Paris as its capital?
```

The model responds as if it never "forgot".

**Proof**: `test_infinite_context_recall_of_evicted_fact` in `nodestor-metadata` verifies that a fact inserted before eviction is correctly retrieved after eviction with similarity > 0.9.

---

## Embedded Vector Database (HNSW + BM25 + RRF)

The `VectorStore` in `nodestor-metadata/src/vector_store.rs` is a pure-Rust implementation with no heavy dependencies.

### HNSW (Hierarchical Navigable Small World)

Multi-layer graph structure for approximate nearest neighbor search:
- Upper layers: few connections, long jumps (fast search)
- Layer 0: many connections, precise local search

Complexity: O(log N) for search, O(log N) for insertion.

### BM25 (Best Match 25)

Classic text retrieval algorithm by relevance. Unlike TF-IDF, BM25 has term frequency saturation (documents with 1000 occurrences aren't 1000× more relevant than documents with 1):

```
score(d, q) = Σ_t IDF(t) × (TF(t,d) × (k1+1)) / (TF(t,d) + k1×(1-b+b×|d|/avgdl))
```

Where `k1=1.2`, `b=0.75`.

### RRF (Reciprocal Rank Fusion)

Combines HNSW and BM25 rankings robustly:

```
RRF_score(d) = Σ_r 1/(k + rank_r(d))   where k=60
```

Documents well-ranked in both systems rise to the top. Documents at the top of only one system land in middle positions.

### Text Embedding

Without an external embedding model, uses FNV-1a hashing with feature hashing to generate 64-dimensional vectors per text. Enables approximate semantic similarity based on token n-grams.

---

## Agent Loop (Deep Research Engine)

### Configuration

```rust
pub struct AgentLoopConfig {
    pub max_loops: usize,           // max cycles (default: 10)
    pub stagnation_window: usize,   // stagnation detection window (default: 20)
    pub stagnation_threshold: f32,  // repetition threshold (default: 0.5)
    pub temperature_bump: f32,      // temperature increment (default: 0.15)
    pub temperature_max: f32,       // temperature ceiling (default: 1.8)
    pub temperature_base: f32,      // base temperature (default: 0.7)
    pub max_tokens_per_step: usize, // max tokens per loop (default: 512)
    pub verbose_steps: bool,
}
```

### Stagnation Detection

Two methods:

**By token overlap** — measures the fraction of tokens in the current window that already appeared in the previous window:
```rust
fn detect_stagnation(tokens: &[u32], window: usize, threshold: f32) -> bool {
    let half = window / 2;
    if tokens.len() < window { return false; }
    let old = &tokens[tokens.len()-window..tokens.len()-half];
    let new = &tokens[tokens.len()-half..];
    let overlap = new.iter().filter(|t| old.contains(t)).count();
    overlap as f32 / new.len() as f32 > threshold
}
```

**By n-grams** — detects repetition of larger patterns (complete phrases):
```rust
fn detect_stagnation_ngram(tokens: &[u32], n: usize, window: usize) -> bool {
    // extracts unique n-grams from 2 halves of the window
    // returns true if >50% of n-grams are duplicates
}
```

### Temperature Adaptation

```rust
fn adapt_temperature(&mut self, stagnated: bool) {
    if stagnated {
        self.stagnation_streak += 1;
        let bump = self.config.temperature_bump * self.stagnation_streak as f32;
        self.current_temperature = (self.current_temperature + bump)
            .min(self.config.temperature_max);
    } else if self.stagnation_streak > 0 {
        self.stagnation_streak -= 1;
        // Gradual cooling: each step without stagnation lowers by 60% of bump
        let cooling = self.config.temperature_bump * 0.6;
        self.current_temperature = (self.current_temperature - cooling)
            .max(self.config.temperature_base);
    }
}
```

### Execution Flow

```
loop i in 0..max_loops:
  1. Call pipeline.generate_stream(context, max_tokens, agent.current_temperature())
     └─ temperature is dynamically set — bumped on stagnation, cooled on progress
  2. Accumulate tokens → step_text
  3. detect_stagnation(all_tokens) || detect_stagnation_ngram(all_tokens)
  4. adapt_temperature(stagnated) → new temperature used on NEXT call
  5. ToolRegistry::try_invoke_from_text(step_text)
     ├── None: TerminationReason::DirectAnswer → return
     └── Some(ToolResult { tag, query, response }):
           call_vector_db  → real pipeline.vector_db.search_text()
           call_dream      → real DreamingEngine::dream_cycle()
           call_hypothesis → real NashTribunal::verify_hypothesis()
           context += step_text + tool_result.to_context_block()
           continue loop
6. If i == max_loops: TerminationReason::MaxLoopsReached
```

---

## Activation Steering: The Mathematics

### The Direction Vector

Calibration extracts hidden states `h+` (positive behavior) and `h-` (negative) and computes:

```
d = mean(h+) - mean(h-)     # mean difference
d̂ = d / ||d||              # normalized
```

### Orthogonal Dynamic Projection

At each layer, during generation:

```
h_new = h - (h · d̂) × d̂ × intensity
```

Geometrically: removes the component of the hidden state in direction `d̂`. If `d̂` represents "refusal," refusal is removed from the internal geometry without modifying weights.

**Why it works**: Linear representations of concepts in high-dimensional embedding spaces (Mikolov et al., 2013 linearity hypothesis) allow simple vector operations to modify model behavior.

**Intensity**: 0.0 = no effect, 1.0 = complete component removal, >1.0 = amplification of the opposite direction.

---

## Type-Faithful Weight Loading

GGUF can store tensors in multiple formats. NodeStor preserves maximum precision:

```rust
fn tensor_to_f32(data: &[u8], dtype: u32, count: usize) -> Vec<f32> {
    match dtype {
        0 => { // F32: direct copy
            bytemuck::cast_slice(data).to_vec()
        }
        1 => { // F16: half→f32 conversion
            data.chunks(2).map(|b| f16::from_le_bytes([b[0],b[1]]).to_f32()).collect()
        }
        2 => { // BF16: bfloat16→f32 conversion
            data.chunks(2).map(|b| {
                let bits = u16::from_le_bytes([b[0],b[1]]) as u32;
                f32::from_bits(bits << 16)
            }).collect()
        }
        8  => DequantDispatcher::dequant_q8_0(data, count),
        12 => DequantDispatcher::dequant_q4_k(data, count),
        13 => DequantDispatcher::dequant_q5_k(data, count),
        _  => vec![0.0; count], // unknown type
    }
}
```

**Tied Embeddings**: SmolLM2 and some other models don't have a separate `output.weight` — the language head uses the same weights as the input embedding. If `output.weight` is not found in the weight bank, the pipeline automatically borrows `token_embd.weight`.

---

## Correct Forward Architecture (Interleaved RoPE)

The implementation in `cpu_reference.rs` fixes a historical bug in many inference engines.

### The RoPE NeoX vs. Llama Bug

Llama GGUF weights use **Interleaved RoPE** (llama.cpp format), not NeoX RoPE. The difference:

**RoPE NeoX** (wrong for Llama):
```
pos_emb = [q[0]*cos - q[d/2]*sin, q[1]*cos - q[d/2+1]*sin, ...]
```

**Interleaved RoPE** (correct for Llama):
```
for i in 0..d/2:
  x = q[2*i],  y = q[2*i+1]
  q[2*i]   = x*cos(θ_i) - y*sin(θ_i)
  q[2*i+1] = x*sin(θ_i) + y*cos(θ_i)
```

The wrong permutation results in incorrectly distributed logits, causing the model to generate nonsense tokens even with correct weights. NodeStor uses the interleaved format, producing coherent output ("Paris" for "The capital of France is").

### Complete Transformer Forward

```
cpu_reference::forward_last_logits(weights, tokens, config):
  1. Embed all tokens: x[i] = token_embd[token_ids[i]]
  2. For each layer l:
     a. RMSNorm(x) → x_norm
     b. Q = x_norm × W_Q  (with RoPE interleaved on Q and K)
     c. K = x_norm × W_K
     d. V = x_norm × W_V
     e. Causal attention: A[i,j] = softmax(Q_i · K_j / √d) * (j ≤ i)
     f. GQA: if num_kv_heads < num_heads, K/V heads are shared
     g. out = A × V
     h. x = x + out × W_O  (residual)
     i. FFN: x = x + SwiGLU(RMSNorm(x))
  3. Final: logits = RMSNorm(x[-1]) × W_output
```

---

## Build Rules

```toml
# Optimized release profile
[profile.release]
lto = true          # Link-Time Optimization across crates
codegen-units = 1   # Single compilation unit (maximum optimization)
opt-level = 3       # Maximum optimization
strip = true        # Strip debug symbols
panic = "abort"     # No stack unwinding (smaller binary, faster)
```

**Development build constraints** (specific to this 16 GB RAM machine):
- `cargo test -j1`: serializes compilation to prevent OOM with parallel rustc
- `cargo check`: can use default parallelism (only generates `.rmeta`)
- `tokio`: pinned at 1.50.0 (1.52.3 requires `windows-0.57.0` which causes rustc stack overflow)
- `mio`: pinned at 1.1.1

**`target/` on D:**  
NTFS junction from `nodestor/target/` to `D:\nodestor-target` (Ventoy USB SSD). Transparent to cargo — builds write to D: automatically, keeping C: clean.
