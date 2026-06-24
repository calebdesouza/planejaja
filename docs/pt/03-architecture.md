# NodeStor — Arquitetura Técnica

Este documento explica como os 14 crates do NodeStor se organizam, como os dados fluem entre eles, e as decisões arquiteturais que tornam o sistema possível.

---

## Estrutura de Crates

```
nodestor/
  nodestor-cli/           ← Binário principal (nodestor) + benchmarks
  nodestor-server/        ← Servidor HTTP (axum) com API REST
  nodestor-integration-tests/  ← Testes end-to-end
  crates/
    nodestor-core/        ← Tipos compartilhados, erros, config
    nodestor-scanner/     ← Detecção de hardware (GPU, CPU, SSD, VRAM)
    nodestor-formats/     ← Parser GGUF v1/v2/v3 + SafeTensors
    nodestor-transport/   ← I/O assíncrono de alta performance
    nodestor-vulkan/      ← Engine Vulkan (compute shaders + pipeline CPU)
    nodestor-streaming/   ← APEX: BurstScheduler, BufferPool, prefetch
    nodestor-metadata/    ← Banco vetorial HNSW+BM25 + indexação
    nodestor-inference/   ← Pipeline completo de inferência (60+ módulos)
    nodestor-davi/        ← Motor de sonho (12 módulos cognitivos)
    nodestor-python/      ← Bindings Python via maturin/PyO3
    nodestor-fabric/      ← Orchestração multi-nó (futuro)
    nodestor-gdeflate/    ← Compressão GDeflate acelerada por GPU
```

---

## Grafo de Dependências

A regra fundamental: **nenhum crate base pode depender de um crate de aplicação.**

```
nodestor-core  (sem deps internas)
      ↑
nodestor-scanner  nodestor-formats  nodestor-gdeflate
      ↑                  ↑
nodestor-transport
      ↑
nodestor-vulkan  nodestor-metadata
      ↑                  ↑
nodestor-streaming
      ↑
nodestor-inference  ←── hub central (60+ módulos)
      ↑
nodestor-davi  ←── depende de inference (não o contrário)
      ↑
nodestor-cli / nodestor-server  ←── integram tudo
```

**Regra crítica PROBES**: `nodestor-inference` não importa `nodestor-davi`. PROBES é acoplado via trait injection:

```rust
// Em nodestor-inference/src/pipeline.rs
pub trait ProbesTool: Send + Sync {
    fn inspect(&self, hidden: &[f32], step: usize);
}

pub struct InferencePipeline {
    probes: Option<Arc<dyn ProbesTool>>,
    ...
}
```

`DaviProbesTool` (em `nodestor-davi`) implementa esse trait e é injetado pela CLI com `pipeline.with_probes_tool(davi_tool)`. Zero dependência circular.

---

## Pipeline de Inferência (7 Camadas)

```
Prompt (texto)
      │
      ▼ Camada 1: Tokenização
  TokenizerManager::encode()
  ┌─ GGUF tokenizer (BPE byte-level)
  └─ HuggingFace tokenizers (fallback)
      │
      ▼ Camada 2: Carregamento de Pesos
  WeightStore + GGUF/SafeTensors parser
  ┌─ F32/F16/BF16 → f32 direto (lossless)
  └─ Q8_0/Q4_K/Q5_K → DequantDispatcher
      │
      ▼ Camada 3: KV-Cache
  KVCacheWindow (H2O eviction)
  ┌─ Window deslizante com orçamento VRAM fixo
  ├─ Evicção: heap(score) → FIFO → sink (último recurso)
  └─ Tokens evictados → vector_db.add_document()
      │
      ▼ Camada 4: Forward Pass
  [GPU] VulkanEngine::dispatch_attention() / dispatch_turbo_quant_attention()
  [CPU] cpu_reference::forward_last_logits()
  ┌─ RMSNorm → SwiGLU → RoPE → Causal Attention → Output
  └─ GQA (Grouped Query Attention) quando num_kv_heads < num_heads
      │
      ▼ Camada 5: Amostragem
  Sampler { temperature, top_k, top_p, repetition_penalty }
  ┌─ Temperatura 0 → greedy (argmax)
  └─ Top-K → Top-P (nucleus) → WeightedIndex sample
      │
      ▼ Camada 6: Streaming
  mpsc::Sender<Result<String>>
  ┌─ Tokens enviados via channel para o cliente
  └─ Métricas: TTFT, tok/s, total
      │
      ▼ Camada 7: EOS Detection
  token_id == eos_token_id → encerra
```

---

## Contexto Infinito: Implementação Detalhada

### O Problema

KV-cache armazena vetores Key e Value para cada token do histórico. Com dimensão oculta `d` e `L` camadas, o custo de memória por token é `2 × L × d × sizeof(f32)` bytes. Para SmolLM2-135M (L=30, d=576): ~138 KB por token. Para 4096 tokens: ~543 MB. Para contexto de 100K tokens: o cache seria inviável.

### A Solução

```rust
// Em kv_cache.rs — loop de geração
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

Quando o KV-cache enche, `enforce_window()` remove os tokens mais antigos do score heap. Os IDs desses tokens são acumulados no `evict_buf`. Quando o buffer atinge 48 tokens (~32 palavras), são decodificados para texto e indexados no banco vetorial.

### Recuperação

Quando o usuário faz uma pergunta relacionada a informação evictada, a busca HNSW+BM25 recupera o texto relevante e ele é preposto ao prompt:

```
[Contexto recuperado]: "A capital da França é Paris..."
<|im_start|>user
Qual país tem Paris como capital?
```

O modelo responde como se nunca tivesse "esquecido".

**Prova**: `test_infinite_context_recall_of_evicted_fact` em `nodestor-metadata` verifica que um fato inserido antes da evição é recuperado após ela com similaridade > 0.9.

---

## Banco Vetorial Embutido (HNSW + BM25 + RRF)

O `VectorStore` em `nodestor-metadata/src/vector_store.rs` é uma implementação pure-Rust, sem dependências pesadas.

### HNSW (Hierarchical Navigable Small World)

Estrutura de grafo multi-camada para busca aproximada de vizinhos mais próximos:
- Camadas superiores: poucas conexões, saltos longos (busca rápida)
- Camada 0: muitas conexões, busca local precisa

Complexidade: O(log N) para busca, O(log N) para inserção.

### BM25 (Best Match 25)

Algoritmo clássico de recuperação de texto por relevância. Diferentemente do TF-IDF, BM25 tem saturação de frequência (documentos com 1000 ocorrências não são 1000× mais relevantes que documentos com 1):

```
score(d, q) = Σ_t IDF(t) × (TF(t,d) × (k1+1)) / (TF(t,d) + k1×(1-b+b×|d|/avgdl))
```

Onde `k1=1.2`, `b=0.75`.

### RRF (Reciprocal Rank Fusion)

Combina os rankings HNSW e BM25 de forma robusta:

```
RRF_score(d) = Σ_r 1/(k + rank_r(d))   onde k=60
```

Documentos bem rankeados em ambos os sistemas sobem ao topo. Documentos no topo de apenas um sistema ficam em posições médias.

### Embed de Texto

Sem modelo de embedding externo, usa FNV-1a hashing com feature hashing para gerar vetores de 64 dimensões por texto. Permite similaridade semântica aproximada baseada em n-gramas de tokens.

---

## Agent Loop (Deep Research Engine)

### Configuração

```rust
pub struct AgentLoopConfig {
    pub max_loops: usize,           // max ciclos (padrão: 10)
    pub stagnation_window: usize,   // janela para detecção de estagnação (padrão: 20)
    pub stagnation_threshold: f32,  // limiar de repetição (padrão: 0.5)
    pub temperature_bump: f32,      // incremento de temperatura (padrão: 0.15)
    pub temperature_max: f32,       // teto de temperatura (padrão: 1.8)
    pub temperature_base: f32,      // temperatura base (padrão: 0.7)
    pub max_tokens_per_step: usize, // tokens máximos por loop (padrão: 512)
    pub verbose_steps: bool,
}
```

### Detecção de Estagnação

Dois métodos:

**Por token overlap** — mede a fração de tokens da janela atual que já apareceram na janela anterior:
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

**Por n-gramas** — detecta repetição de padrões maiores (frases completas):
```rust
fn detect_stagnation_ngram(tokens: &[u32], n: usize, window: usize) -> bool {
    // extrai n-gramas únicos de 2 metades da janela
    // retorna true se >50% dos n-gramas são duplicados
}
```

### Adaptação de Temperatura

```rust
fn adapt_temperature(&mut self, stagnated: bool) {
    if stagnated {
        self.stagnation_streak += 1;
        let bump = self.config.temperature_bump * self.stagnation_streak as f32;
        self.current_temperature = (self.current_temperature + bump)
            .min(self.config.temperature_max);
    } else if self.stagnation_streak > 0 {
        self.stagnation_streak -= 1;
        // Resfriamento gradual: cada step sem estagnação baixa 60% do bump
        let cooling = self.config.temperature_bump * 0.6;
        self.current_temperature = (self.current_temperature - cooling)
            .max(self.config.temperature_base);
    }
}
```

### Fluxo de Execução

```
loop i em 0..max_loops:
  1. Chama pipeline.generate_stream(contexto, max_tokens)
  2. Acumula tokens → step_text
  3. Verifica stagnation em all_tokens
  4. Adapta temperatura
  5. ToolRegistry::scan_for_call(step_text)
     ├── None: TerminationReason::DirectAnswer → retorna
     └── Some(tag, query):
           invoke(tag, query) → ToolResult
           contexto += step_text + tool_result.to_context_block()
           continua loop
6. Se loop_i == max_loops: TerminationReason::MaxLoopsReached
```

---

## Steering de Ativação: A Matemática

### O Vetor de Direção

A calibração extrai hidden states `h+` (comportamento positivo) e `h-` (negativo) e computa:

```
d = mean(h+) - mean(h-)     # diferença de médias
d̂ = d / ||d||              # normalizado
```

### Projeção Ortogonal Dinâmica

A cada camada, durante a geração:

```
h_novo = h - (h · d̂) × d̂ × intensity
```

Geometricamente: remove a componente do hidden state na direção `d̂`. Se `d̂` representa "recusa", a recusa é retirada da geometria interna sem modificar pesos.

**Por que funciona**: representações lineares de conceitos em espaços de embedding de alta dimensão (hipótese de linearidade de Mikolov et al., 2013) permitem que operações vetoriais simples modifiquem o comportamento do modelo.

**Intensidade**: 0.0 = sem efeito, 1.0 = remoção completa da componente, >1.0 = amplificação da direção oposta.

---

## Carregamento de Pesos Fiel ao Tipo

O GGUF pode armazenar tensores em múltiplos formatos. NodeStor preserva a precisão máxima:

```rust
fn tensor_to_f32(data: &[u8], dtype: u32, count: usize) -> Vec<f32> {
    match dtype {
        0 => { // F32: cópia direta
            bytemuck::cast_slice(data).to_vec()
        }
        1 => { // F16: conversão half→f32
            data.chunks(2).map(|b| f16::from_le_bytes([b[0],b[1]]).to_f32()).collect()
        }
        2 => { // BF16: conversão bfloat16→f32
            data.chunks(2).map(|b| {
                let bits = u16::from_le_bytes([b[0],b[1]]) as u32;
                f32::from_bits(bits << 16)
            }).collect()
        }
        8 => DequantDispatcher::dequant_q8_0(data, count),
        12 => DequantDispatcher::dequant_q4_k(data, count),
        13 => DequantDispatcher::dequant_q5_k(data, count),
        _ => vec![0.0; count], // tipo desconhecido
    }
}
```

**Pesos Emprestados (Tied Embeddings)**: SmolLM2 e alguns outros modelos não têm `output.weight` separado — a cabeça de linguagem usa os mesmos pesos que o embedding de entrada. Se `output.weight` não é encontrado no banco de pesos, o pipeline automaticamente empresta `token_embd.weight`.

---

## Arquitetura de Forward Correto (RoPE Interleaved)

A implementação em `cpu_reference.rs` resolve um bug histórico em muitos motores de inferência.

### O Bug do RoPE NeoX vs. Llama

Os pesos GGUF do Llama usam **RoPE Interleaved** (formato Llama.cpp), não o RoPE NeoX. A diferença:

**RoPE NeoX** (errado para Llama):
```
pos_emb = [q[0]*cos - q[d/2]*sin, q[1]*cos - q[d/2+1]*sin, ...]
```

**RoPE Interleaved** (correto para Llama):
```
for i in 0..d/2:
  x = q[2*i],  y = q[2*i+1]
  q[2*i]   = x*cos(θ_i) - y*sin(θ_i)
  q[2*i+1] = x*sin(θ_i) + y*cos(θ_i)
```

A permutação incorreta resulta em logits distribuídos de forma errada, fazendo o modelo gerar tokens sem sentido mesmo com pesos corretos. NodeStor usa o formato interleaved, o que produz output coerente ("Paris" para "The capital of France is").

---

## Regras de Build

```toml
# Perfil de release otimizado
[profile.release]
lto = true          # Link-Time Optimization entre crates
codegen-units = 1   # Um único unidade de compilação (máxima otimização)
opt-level = 3       # Otimização máxima
strip = true        # Remove símbolos de debug
panic = "abort"     # Sem stack unwinding (menor binário, mais rápido)
```

**Restrições de build em desenvolvimento** (específicas desta máquina, 16 GB RAM):
- `cargo test -j1`: serializa compilação para evitar OOM com rustc paralelo
- `cargo check`: pode usar paralelismo padrão (apenas `.rmeta`)
- `tokio`: fixo em 1.50.0 (1.52.3 requer `windows-0.57.0` que causa stack overflow no rustc)
- `mio`: fixo em 1.1.1

**`target/` em D:**  
Junção NTFS de `nodestor/target/` para `D:\nodestor-target` (SSD USB Ventoy). Transparente para o cargo — builds escrevem em D: automaticamente, liberando C: para o sistema.
