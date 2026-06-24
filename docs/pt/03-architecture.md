# NodeStor — Arquitetura

## Grafo de Dependências dos Crates

```
nodestor-cli  ──────────────────────────────────────────────┐
    │                                                         │
    ├── nodestor-inference  ←── TODA a lógica de inferência   │
    │       ├── pipeline.rs        (orquestrador 7-camadas)   │
    │       ├── cober.rs           (decodificação especulativa)│
    │       ├── kv_cache.rs        (H2O + contexto infinito)  │
    │       ├── agent_loop.rs      (Motor Deep Research)       │
    │       ├── tool_registry.rs   (ferramentas por tag)      │
    │       ├── cpu_reference.rs   (forward Llama correto)    │
    │       ├── sampler.rs         (top-k, top-p, temp, rep)  │
    │       ├── lora_core.rs       (injeção delta LoRA)       │
    │       ├── steering_engine.rs (steering de ativação)     │
    │       ├── persistent_memory.rs (memória episódica)      │
    │       └── 50+ outros módulos                            │
    │                                                         │
    ├── nodestor-davi  ←── Inteligência Sonhante              │
    │       ├── dreaming_engine.rs (orquestrador D9)          │
    │       ├── topology.rs        (detecção de gaps TDA)     │
    │       ├── free_energy.rs     (FEP de Friston)           │
    │       ├── annealing.rs       (temperatura semântica)    │
    │       ├── nash_tribunal.rs   (validação 3 agentes)      │
    │       ├── functors.rs        (theory de categorias)     │
    │       ├── stigmergy.rs       (ferômônios swarm)         │
    │       ├── autopoiesis.rs     (auto-melhoria)            │
    │       ├── elk_probe.rs       (detector de mentiras)     │
    │       ├── cot_monitor.rs     (divergência CoT)          │
    │       └── raise_detector.rs  (consciência situacional)  │
    │                                                         │
    ├── nodestor-metadata  ←── Banco Vetorial + RAG           │
    │       ├── vector_store.rs    (HNSW + BM25 + RRF)        │
    │       └── search.rs          (interface RAG)            │
    │                                                         │
    ├── nodestor-vulkan   ←── Compute na GPU                  │
    ├── nodestor-streaming ←── Transporte APEX + COBER        │
    ├── nodestor-formats  ←── Parsing GGUF + SafeTensors      │
    ├── nodestor-scanner  ←── Detecção de hardware            │
    └── nodestor-core     ←── Tipos compartilhados, erros     │
                                                              │
nodestor-server  ←── API HTTP (axum) ─────────────────────────┘
```

## Regras de Dependência

- `nodestor-inference` é o hub. Ele NÃO importa `nodestor-davi`.
- `nodestor-davi` importa `nodestor-inference` (usa `semantic_attention`, tipos do pipeline).
- PROBES se acopla ao pipeline via injeção de trait `ProbesTool` (sem dependência circular).
- CLI importa tudo; é o ponto de integração.

## Contexto Infinito

Quando o KV cache enche, `enforce_window()` evicta os tokens mais antigos. Os IDs de tokens evictados são decodificados para texto e indexados no banco vetorial (`VectorSearch`). Na recuperação, conteúdo passado semanticamente similar é preposto ao prompt — dando ao modelo acesso a histórico de comprimento arbitrário dentro de um orçamento fixo de VRAM.

```
loop de generate():
  → passo forward (Vulkan ou CPU)
  → amostra token
  → verifica janela KV
    → se evictou: decodifica tokens → vector_db.add_document()
  → transmite token ao usuário
  → verifica EOS
```

## Loop de Agente (Deep Research)

```
AgentExecutionLoop::run()
  loop (max_loops):
    → gera texto (pipeline.generate_stream)
    → ToolRegistry::scan_for_call(texto)
      → None:  termina (DirectAnswer)
      → Some:  invoke(tag, query) → ToolResult
                → injeta tool_response no contexto
                → adapt_temperature (verificação de estagnação)
                → próximo loop
```

Tags de ferramenta emitidas pelo modelo:
- `<call_vector_db>query</call_vector_db>` → busca semântica
- `<call_dream>domínio1+domínio2</call_dream>` → síntese cross-domain DAVI
- `<call_think>raciocínio</call_think>` → reflexão interna
- `<call_hypothesis>hipótese</call_hypothesis>` → validação Nash Tribunal

## Ciclo de Sonho DAVI

```
DreamingEngine::dream_cycle(embeddings, insights):
  1. TopologicalGapDetector  → encontra lacunas no conhecimento (homologia persistente)
  2. FreeEnergyObjective     → seleciona lacuna de maior surpresa
  3. SemanticAnnealing       → aceita/rejeita salto cross-domain (temperatura)
  4. NashTribunal            → 3 agentes debatem a hipótese
  5. FunctorComposer         → mapeia descoberta para outros domínios
  6. StigmergicSwarm         → deposita feromônio no caminho bem-sucedido
  7. AutopoieticLoop         → atualiza saúde do sistema / viés de exploração
```

## Steering de Ativação

Quando `--steer-vector` é usado, o pipeline aplica Projeção Ortogonal Dinâmica em todo stream residual:

```
h_novo = h - (h · d̂) * d̂ * intensity
```

Onde `d̂` é o vetor de direção normalizado extraído por `calibrate`. Isso modifica a geometria de todos os hidden states sem tocar nos pesos do modelo.

DSCP (`--auto-steer`) gera o vetor de direção em runtime a partir de templates de ativação contrastivos — sem dataset externo necessário.

## Especulação de Tokens (COBER)

```
COBER::verify_and_accept_probabilistic(draft_tokens, logits):
  para cada token rascunhado t_d na posição i:
    r ~ Uniforme(0, 1)
    se r < p_modelo(t_d) / p_rascunho(t_d):
      aceita t_d    ← lossless
    senão:
      amostra da distribuição corrigida
      para
```

Speedup = média de tokens aceitos por passo (tipicamente 2–4×).

## Carregamento de Pesos Fiel ao Tipo

O pipeline carrega pesos GGUF preservando tipo de dado:
- **F32/F16/BF16** → conversão direta para F32 (sem perda)
- **Q8_0** → dequantização via `DequantDispatcher::dequant_q8_0`
- **Q4_K/Q5_K** → dequantização via `dequant_q4_k`/`dequant_q5_k`
- **Pesos emprestados (tied)**: SmolLM2 não tem `output.weight` → amarra à `token_embd`

## Regras de Build Críticas

- **Sempre `-j1`** para `cargo test` e `cargo build`: a máquina de 16 GB tem o page file esgotado por múltiplas instâncias paralelas do rustc.
- **`cargo check`** pode rodar em paralelo padrão (só gera `.rmeta`, menor uso de memória).
- `tokio` fixo em 1.50.0 / `mio` em 1.1.1 — não atualizar.
- `target/` é uma NTFS junction → `D:\nodestor-target` (Ventoy USB).
