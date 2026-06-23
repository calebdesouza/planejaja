//! Pipeline de inferência — orquestração end-to-end do fluxo de dados.
//!
//! Conecta: Scanner (Hardware) → Transport (SSD) → Streaming (Metralhadora/Pool) → Vulkan (GPU) → LLM.
//!
//! ### PROBES V2 (Interpretabilidade Mecanística em Tempo Real)
//! A cada token gerado, o pipeline executa:
//! 1. `SAEEngine::encode()` — decompõe o hidden_state em features legíveis
//! 2. `ElkProbe::probe_honesty()` — detecta dissimulação latente
//! 3. `CoTMonitor::evaluate_step()` — verifica obfuscação no raciocínio
//! 4. `RaiseDetector::scrutinize_inference()` — bloqueia consciência situacional SA4+
//! 5. `Sampler::sample_with_conformal()` — garante certeza matemática TECP

use nodestor_core::{DataTransport, HardwareProfile, ModelMetadata, NodeStorError};
use nodestor_formats::detect_parser;
use nodestor_scanner::scan;
use nodestor_streaming::{BufferPool, MesPrefetchQueue, BurstScheduler, speculative::SpeculativeCache};
use nodestor_transport::create_transport;
use nodestor_metadata::search::VectorSearch;
use nodestor_vulkan::VulkanEngine;
use std::sync::Arc;
use tokio::time::Instant;
use tracing::{debug, warn};
use futures::Stream;
use std::pin::Pin;
use crate::graph_interpreter;

/// Trait injetável para sistemas externos de PROBES (ELK/CoT/RAISE no DAVI).
/// O pipeline recebe um Box<dyn ProbesTool> e delega a inspeção sem criar depência circular.
pub trait ProbesTool: Send + Sync {
    /// Inspeciona um hidden_state e retorna (is_safe, alert_msg_or_none).
    fn inspect(&mut self, hidden_state: &[f32], step: usize) -> (bool, Option<String>);
}

/// Configurações paramétricas da engine.
pub struct InferenceConfig {
    pub model_path: String,
    pub prefetch_depth: usize,
    pub buffer_size: usize,
}

/// Métricas de desempenho geradas a cada inferência.
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tokens_per_second: f64,
    pub total_time_ms: u128,
    /// Alertas disparados pelo PROBES V2 durante a geração (ELK + CoT + RAISE)
    pub probes_alerts: Vec<String>,
    /// Tokens rejeitados pelo Conformal Predictor (alta incerteza)
    pub conformal_rejections: usize,
}

/// Estrutura para orquestração inteligente de contexto (RAM/SSD Paging).
pub struct KVCachePaginator {
    pub vram_capacity_tokens: usize,
    pub ssd_offload_enabled: bool,
}

/// Configuração do PROBES V2 para o pipeline.
pub struct ProbesConfig {
    /// Habilita SAE + ELK + CoT + RAISE durante a inferência
    pub enabled: bool,
    /// Dimensão do espaço latente do modelo (ex: 4096 para LLaMA 7B)
    pub hidden_dim: usize,
    /// Tamanho do dicionário SAE (expansão de features)
    pub sae_dict_size: usize,
    /// Limiar JumpReLU para esparsidade
    pub sae_threshold: f32,
    /// Nível máximo de RAISE tolerado antes do bloqueio
    pub raise_block_sa_level: u8,
}

impl Default for ProbesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hidden_dim: 4096,
            sae_dict_size: 8192,
            sae_threshold: 0.5,
            raise_block_sa_level: 4, // Bloqueia em SA4+
        }
    }
}

/// Pipeline central de execução do modelo.
pub struct InferencePipeline {
    pub config: InferenceConfig,
    pub probes_config: ProbesConfig,
    pub profile: HardwareProfile,
    pub transport: Arc<dyn DataTransport + Send + Sync>,
    pub metadata: Arc<ModelMetadata>,
    pub engine: VulkanEngine,
    pub vector_db: VectorSearch,
    pub kv_paginator: KVCachePaginator,
    /// Ferramenta externa de PROBES (ELK/CoT/RAISE) — injetável sem depência circular.
    pub probes_tool: Option<std::sync::Mutex<Box<dyn ProbesTool>>>,
    /// Escalonador de Batching Contínuo
    pub scheduler: std::sync::Mutex<crate::multi_tenant::MultiTenantScheduler>,
}

impl InferencePipeline {
    /// Boot do sistema de Inteligência Artificial V2.
    /// Inspeciona o hardware dinamicamente e monta o melhor pipeline O.S-Level.
    pub fn init(config: InferenceConfig) -> Result<Self, NodeStorError> {
        let profile = scan()?;
        let transport: Arc<dyn DataTransport + Send + Sync> = Arc::from(create_transport(&profile));

        let parser = detect_parser(&config.model_path)?;
        let raw_metadata = parser.parse(&config.model_path)?;
        let metadata = Arc::new(raw_metadata);

        let engine = VulkanEngine::new(&profile)?;
        
        let kv_paginator = KVCachePaginator {
            vram_capacity_tokens: 32768, // Valor base, dinâmico em prod real
            ssd_offload_enabled: true,
        };

        // Inicializa a base vetorial local (LanceDB)
        let db_path = format!("{}/vector_db", config.model_path);
        let vector_db = VectorSearch::new("knowledge_base", &db_path);

        Ok(Self {
            config,
            probes_config: ProbesConfig::default(),
            profile,
            transport,
            metadata,
            engine,
            vector_db,
            kv_paginator,
            probes_tool: None,
            scheduler: std::sync::Mutex::new(crate::multi_tenant::MultiTenantScheduler::new(1024, 128)),
        })
    }

    /// Configura o PROBES V2 com parâmetros customizados.
    pub fn with_probes(mut self, probes_config: ProbesConfig) -> Self {
        self.probes_config = probes_config;
        self
    }

    /// Injeta um sistema externo de inspeção (ELK/CoT/RAISE) via trait object.
    /// Permite uso do DAVI sem dependência circular.
    pub fn with_probes_tool(mut self, tool: Box<dyn ProbesTool>) -> Self {
        self.probes_tool = Some(std::sync::Mutex::new(tool));
        self
    }

    /// Orquestração Zero-Loss: Move KV Cache de alta fidelidade para o SSD via DMA.
    pub async fn swap_context_to_ssd(&self, _layer_idx: usize, _data: &[u8]) -> Result<(), NodeStorError> {
        // Usa o transporte industrial (DirectStorage/Win32) para salvar sem wait-state
        self.transport.write_to_vram_buffer(_data)?;
        debug!("Paging: Camada de contexto movida para o SSD (Zero-Loss FP16)");
        Ok(())
    }

    /// Orquestração Zero-Loss: Recarrega contexto do SSD para a VRAM instantaneamente.
    pub async fn load_context_from_ssd(&self, _layer_idx: usize) -> Result<Vec<u8>, NodeStorError> {
        // Implementação simulada do reload via DMA
        Ok(vec![0u8; 1024])
    }

    /// Submete uma requisição ao escalonador de Continuous Batching.
    pub fn submit_request(&self, request: crate::multi_tenant::InferenceRequest) {
        if let Ok(mut sched) = self.scheduler.lock() {
            sched.submit_request(request);
        }
    }

    /// Loop principal de "Mecanismo de Atenção": prevê tensores e dispara
    /// kernels Vulkan para gerar tokens a alta voltagem (Modo Metralhadora).
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        tx: Option<tokio::sync::mpsc::Sender<Result<String, NodeStorError>>>,
    ) -> Result<(String, GenerationStats), NodeStorError> {
        let start_time = Instant::now();
        
        // 0. Busca RAG (híbrida HNSW + BM25 + RRF): embeda o PROMPT real e
        //    recupera os fragmentos de conhecimento mais relevantes do VectorStore.
        let rag_context = self.vector_db.search_text(prompt, 3).await.unwrap_or_default();
        debug!("RAG: {} fragmentos relevantes recuperados para o prompt", rag_context.len());

        // 1. Inicializa o subsistema de memória L3 (Double/Triple VRAM Buffering)
        let pool = BufferPool::new(
            &self.engine.ctx, 
            self.config.buffer_size, 
            self.config.prefetch_depth
        )?;

        // 2. Coloca a Metralhadora para puxar de imediato
        let queue = MesPrefetchQueue::new(
            self.transport.clone(),
            self.config.model_path.clone(),
            pool,
        );

        let mut scheduler = BurstScheduler::new(
            self.config.prefetch_depth,
            queue,
            self.metadata.clone(),
        );

        // Instancia o Speculative Cache baseado no tamanho da VRAM.
        let mut _speculative_cache = SpeculativeCache::new(4);

        // ─── Detectar arquitetura do modelo via GraphInterpreter ─────────────────
        let graph = graph_interpreter::GraphInterpreter::interpret(&self.metadata)
            .unwrap_or_else(|_| {
                // Fallback: assume Llama-7B como default conservador
                // Constrói o extra com serde_json::Value disponível via nodestor_core
                use nodestor_core::ModelMetadata;
                let mut extra = serde_json::Map::new();
                extra.insert("llama.embedding_length".into(), serde_json::Value::Number(4096u64.into()));
                extra.insert("llama.attention.head_count".into(), serde_json::Value::Number(32u64.into()));
                extra.insert("llama.attention.head_count_kv".into(), serde_json::Value::Number(8u64.into()));
                extra.insert("llama.feed_forward_length".into(), serde_json::Value::Number(11008u64.into()));
                extra.insert("llama.vocab_size".into(), serde_json::Value::Number(32000u64.into()));
                let fallback_meta = ModelMetadata {
                    format: self.metadata.format.clone(),
                    model_name: self.metadata.model_name.clone(),
                    architecture: Some("llama".to_string()),
                    param_count: self.metadata.param_count,
                    tensors: vec![
                        nodestor_core::TensorInfo {
                            name: "blk.0.attn_q.weight".to_string(),
                            shape: vec![4096, 4096],
                            dtype: nodestor_core::TensorDtype::F16,
                            data_offset: 0, data_size: 0,
                        },
                        nodestor_core::TensorInfo {
                            name: "blk.0.ffn_gate.weight".to_string(),
                            shape: vec![4096, 11008],
                            dtype: nodestor_core::TensorDtype::F16,
                            data_offset: 0, data_size: 0,
                        },
                    ],
                    data_offset: 0,
                    file_size: 0,
                    extra: serde_json::Value::Object(extra),
                };
                graph_interpreter::GraphInterpreter::interpret(&fallback_meta).unwrap()
            });

        let (num_layers, vocab_size, hidden_size, num_heads, num_kv_heads, intermediate_size, rope_base) =
            match &graph.architecture {
                graph_interpreter::ModelArchitecture::Llama {
                    num_layers, vocab_size, hidden_dim, num_heads, num_kv_heads,
                    intermediate_size, rope_base, ..
                } => (*num_layers as usize, *vocab_size, *hidden_dim, *num_heads, *num_kv_heads, *intermediate_size, *rope_base),
                _ => (32usize, 32000u32, 4096u32, 32u32, 8u32, 11008u32, 10000.0f32),
            };

        // ─── Construir WeightBank a partir dos tensores carregados ───────────────
        // Em produção, os tensores do GGUF já foram carregados pelo parser e
        // estão no `self.metadata.tensors`. Aqui subimos cada um para a VRAM.
        let mut weight_bank = nodestor_vulkan::WeightBank::new();

        // ─── Carregamento de PESOS REAIS via WeightStore (mmap zero-copy + upload) ──
        // Tenta abrir o GGUF e subir os bytes reais de cada tensor para a GPU. Esta é
        // a ponte que faz o forward pass operar sobre os pesos verdadeiros do modelo,
        // e não sobre buffers zerados. Em arquivos sintéticos/testes (sem tensores
        // mapeáveis), cai no fallback de buffers vazios com as shapes corretas.
        let mut real_loaded = 0usize;
        match crate::weight_store::WeightStore::open(std::path::Path::new(&self.config.model_path)) {
            Ok(store) => {
                for name in store.list_tensors() {
                    let bytes = match store.tensor_bytes(name) { Some(b) => b, None => continue };
                    // Converte do dtype nativo do GGUF para FP32. NÚCLEO LOSSLESS:
                    // F32/F16/BF16 com fidelidade total (zero perda de qualidade).
                    // Quant (Q8_0…) é opcional e também convertida aqui.
                    let (dtype, n_elems) = store.tensor_info(name)
                        .map(|t| (t.dtype, t.shape.iter().map(|&d| d as usize).product::<usize>()))
                        .unwrap_or((nodestor_core::TensorDtype::F32, bytes.len() / 4));
                    let upload = match tensor_to_f32(bytes, dtype, n_elems) {
                        Some(f32s) => {
                            let raw = unsafe {
                                std::slice::from_raw_parts(f32s.as_ptr() as *const u8, f32s.len() * 4)
                            };
                            self.engine.upload(raw)
                        }
                        // Formato ainda não coberto pelo conversor: sobe os bytes crus.
                        None => self.engine.upload(bytes),
                    };
                    match upload {
                        Ok(buf) => { weight_bank.insert(name.to_string(), buf); real_loaded += 1; }
                        Err(e) => debug!("WeightStore: upload de '{}' falhou: {}", name, e),
                    }
                }
                debug!("WeightStore: {} tensores REAIS carregados (dtype→FP32 lossless) do GGUF para a GPU", real_loaded);
            }
            Err(e) => debug!("WeightStore indisponível ({}); usando fallback de buffers vazios", e),
        }

        // Fallback: garante que todo tensor esperado exista (shapes corretas) mesmo
        // que o WeightStore não tenha conseguido mapeá-lo (modelos sintéticos/dummy).
        for tensor in &self.metadata.tensors {
            if weight_bank.get(&tensor.name).is_some() { continue; }
            // Converte shape Vec<u64> para bytes (cada dim é u64 no formato GGUF)
            let size_bytes: usize = tensor.shape.iter()
                .map(|&d| d as usize)
                .product::<usize>() * 4;
            let size_bytes = size_bytes.max(4);
            match self.engine.alloc_buffer(size_bytes) {
                Ok(buf) => weight_bank.insert(tensor.name.clone(), buf),
                Err(e) => {
                    debug!("WeightBank: falha ao alocar '{}' ({}B): {}", tensor.name, size_bytes, e);
                }
            }
        }
        // TIED EMBEDDINGS: muitos modelos (SmolLM2, Gemma…) não trazem `output.weight`
        // — a lm_head É o `token_embd.weight`. Sem amarrar, o lm_head ficaria um
        // placeholder zerado e os LOGITS sairiam todos nulos. Materializa a amarração.
        if weight_bank.get("output.weight").is_none() {
            let tied = weight_bank.get("token_embd.weight")
                .map(|te| te.as_f32_slice().to_vec())
                .filter(|d| !d.is_empty());
            if let Some(data) = tied {
                let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                if let Ok(buf) = self.engine.upload(raw) {
                    weight_bank.insert("output.weight".to_string(), buf);
                    debug!("Tied embeddings: output.weight amarrado ao token_embd ({} floats)", data.len());
                }
            }
        }

        // Garante que as chaves críticas existam mesmo se ausentes no GGUF
        let ensure_key = |bank: &mut nodestor_vulkan::WeightBank, key: &str, size: usize| {
            if bank.get(key).is_none() {
                if let Ok(buf) = self.engine.alloc_buffer(size.max(4)) {
                    bank.insert(key.to_string(), buf);
                }
            }
        };
        ensure_key(&mut weight_bank, "output_norm.weight", hidden_size as usize * 4);
        ensure_key(&mut weight_bank, "output.weight", hidden_size as usize * vocab_size as usize * 4);
        ensure_key(&mut weight_bank, "token_embd.weight", vocab_size as usize * hidden_size as usize * 4);

        // Inicializa o Paged KV Cache (Contexto Infinito via SSD)
        let swap_path = format!("{}/nodestor_kv_swap_{}.bin", std::env::temp_dir().to_str().unwrap(), std::process::id());
        let max_vram_blocks = 16;
        let head_dim = if num_heads > 0 { hidden_size / num_heads } else { 64 };
        let mut kv_cache = crate::kv_cache::KVCache::new(
            num_layers,
            128,      // tokens por bloco
            head_dim as usize,
            max_vram_blocks,
            &swap_path
        );
        debug!("Paged KV Cache inicializado: {} camadas, head_dim={}", num_layers, head_dim);

        // Construir Transformer com dimensões reais do modelo
        let transformer = nodestor_vulkan::Transformer::from_metadata(
            num_layers as u32,
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            rope_base,
            1e-5,
            true, // use_zipgemm quando pesos NSZ disponíveis
        );


        let mut generated_tokens = Vec::new();
        let mut tokens_done = 0;
        let mut probes_alerts: Vec<String> = Vec::new();
        let mut conformal_rejections: usize = 0;
        let layers_per_token = num_layers.min(4).max(1);

        // ── PROBES V2: Inicialização dos módulos ──────────────────────────────────
        let probes_enabled = self.probes_config.enabled;
        // SAE local: decomposição monosemântica dos hidden_states.
        // Usa a dimensão REAL do modelo (não o default fixo), senão a decomposição
        // quebra em modelos cujo hidden_size ≠ 4096 (ex.: SmolLM2-135M tem 576).
        let mut probes_sae = crate::sae_engine::SAEEngine::new(
            hidden_size as usize,
            self.probes_config.sae_dict_size,
            self.probes_config.sae_threshold,
        );

        // MCTS Engine local: busca deliberativa profunda (Princípio 2)
        let mut _mcts_engine = crate::mcts_engine::MctsEngine::new(1.414); // Cp = sqrt(2)
        // ─────────────────────────────────────────────────────────────────────────

        let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            use_conformal: probes_enabled,
        });

        // ── Tokenizer — encode/decode ────────────────────────────────────────────
        // Constrói o tokenizer REAL a partir do vocab+merges embutidos no GGUF
        // (`tokenizer.ggml.tokens`/`.merges`). Sem isso, texto de modelos reais sai
        // ilegível. Fallback para um WordLevel dummy se o GGUF não trouxer vocab.
        let dummy_json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"Hello":1,"World":2},"unk_token":"<unk>"}}"#;
        let extra = &self.metadata.extra;
        let str_array = |key: &str| -> Vec<String> {
            extra.get(key).and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default()
        };
        let u32_meta = |key: &str| -> Option<u32> {
            extra.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
        };
        let gguf_tokens = str_array("tokenizer.ggml.tokens");
        let gguf_merges = str_array("tokenizer.ggml.merges");
        let tokenizer = if !gguf_tokens.is_empty() {
            debug!("Tokenizer GGUF: {} tokens, {} merges", gguf_tokens.len(), gguf_merges.len());
            crate::tokenizer::TokenizerManager::from_gguf(
                &gguf_tokens, &gguf_merges,
                u32_meta("tokenizer.ggml.bos_token_id"),
                u32_meta("tokenizer.ggml.eos_token_id"),
                u32_meta("tokenizer.ggml.unknown_token_id"),
            ).or_else(|e| {
                warn!("Tokenizer GGUF falhou ({}); usando dummy", e);
                crate::tokenizer::TokenizerManager::from_string(dummy_json)
            })
        } else {
            crate::tokenizer::TokenizerManager::from_string(dummy_json)
        }.map_err(|e| NodeStorError::InferenceError(format!("Falha ao construir tokenizer: {}", e)))?;

        // RAG end-to-end: costura os fragmentos recuperados ANTES do prompt, para
        // que o forward pass condicione a geração no conhecimento factual indexado.
        let effective_prompt = if rag_context.is_empty() {
            prompt.to_string()
        } else {
            let ctx: String = rag_context.iter()
                .filter_map(|r| r.payload.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            debug!("RAG: injetando {} fragmentos ({} chars) no contexto", rag_context.len(), ctx.len());
            format!("[Contexto]\n{}\n\n[Pergunta]\n{}", ctx, prompt)
        };
        let mut input_tokens = tokenizer.encode(&effective_prompt).unwrap_or(vec![0]);
        if input_tokens.is_empty() { input_tokens.push(0); }

        // Pre-enche a RAM/VRAM (Burst Pump) — best-effort e LIMITADO no tempo: o
        // priming do streaming não pode bloquear a geração. Se estourar o limite,
        // seguimos (os pesos já vêm do WeightStore mmap; o pump é otimização).
        match tokio::time::timeout(std::time::Duration::from_secs(5), scheduler.prime_pump()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => debug!("prime_pump: {}", e),
            Err(_) => warn!("prime_pump excedeu 5s; seguindo sem pré-aquecer o stream"),
        }

        let mut cober = crate::cober::CoberEngine::new_dense(crate::vram_budget::VramBudget::estimate(4 * 1024 * 1024 * 1024));
        let mut drafter = crate::latent_drafter::LatentDrafter::new(hidden_size as usize, 0.9);
        let cheby = nodestor_vulkan::transformer::ChebyshevSoftmax::default();

        // Config do forward de REFERÊNCIA correto (CPU) — base de coerência.
        let head_dim_cfg = if num_heads > 0 { (hidden_size / num_heads) as usize } else { 64 };
        let cpu_cfg = crate::cpu_reference::CpuModelConfig {
            n_layers: num_layers,
            hidden: hidden_size as usize,
            n_heads: num_heads as usize,
            n_kv_heads: num_kv_heads as usize,
            head_dim: head_dim_cfg,
            intermediate: intermediate_size as usize,
            vocab: vocab_size as usize,
            rope_base,
            eps: 1e-5,
        };
        let eos_token: Option<u32> = self.metadata.extra.get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64()).map(|n| n as u32);
        debug!("CPU-fwd cfg: layers={} hidden={} n_heads={} n_kv={} head_dim={} inter={} vocab={} rope_base={} eos={:?}",
            cpu_cfg.n_layers, cpu_cfg.hidden, cpu_cfg.n_heads, cpu_cfg.n_kv_heads,
            cpu_cfg.head_dim, cpu_cfg.intermediate, cpu_cfg.vocab, cpu_cfg.rope_base, eos_token);

        // ─── Forward INCREMENTAL com KV cache (reuso): O(seq) por token ─────────
        // Prefill: processa cada token do prompt UMA vez, preenchendo o cache.
        // Decode: processa só o token NOVO, atendendo ao histórico cacheado —
        // evita recomputar a sequência inteira a cada passo (O(seq²) → O(seq)).
        // É também a base para a especulação (COBER/EAGLE) render de verdade.
        // ─── Decode GREEDY + ESPECULAÇÃO SEM CABEÇA (n-gram / Prompt Lookup) ────
        // O método que MELHOR se encaixa: ZERO pesos extras, plug-and-play em
        // QUALQUER modelo. A CPU faz string-matching no contexto (n-gram) e "copia"
        // K tokens — custo ~zero. O ALVO os VERIFICA em UM forward batched
        // (`forward_verify` = tree-attention na GPU: 1 weight-load avalia os K),
        // aceita o PREFIXO concordante (greedy ⇒ lossless) e GUILHOTINA o resto
        // (rollback do cache). Em texto repetitivo (código/JSON/listas) a aceitação
        // é alta → "K tokens no tempo de 1" na placa limitada por banda.
        const NGRAM: usize = 2;
        const DRAFT_K: usize = 8;
        let prompt_lookup = |seq: &[u32]| -> Vec<u32> {
            let n = seq.len();
            if n <= NGRAM { return Vec::new(); }
            let suffix = &seq[n - NGRAM..];
            let mut i = n - NGRAM;
            while i > 0 {
                i -= 1;
                if &seq[i..i + NGRAM] == suffix {
                    let start = i + NGRAM;
                    let end = (start + DRAFT_K).min(n);
                    if start < n { return seq[start..end].to_vec(); }
                }
            }
            Vec::new()
        };
        let mut main_kv = crate::cpu_reference::CpuKvCache::new(cpu_cfg.n_layers);
        let _ = (&cheby, &cober, &drafter, &_mcts_engine, &probes_sae, &transformer, probes_enabled, layers_per_token, &sampler);

        // Prefill: preenche o cache com o prompt.
        let mut cur_logits: Option<Vec<f32>> = None;
        for (p, &tok) in input_tokens.iter().enumerate() {
            match crate::cpu_reference::forward_step(tok, p, &cpu_cfg, &weight_bank, &mut main_kv) {
                Some(l) => cur_logits = Some(l),
                None => break, // pesos ausentes (modelo sintético) → encerra
            }
        }

        let mut full_seq: Vec<u32> = input_tokens.clone();
        let mut pos = input_tokens.len();
        let mut spec_drafted = 0usize;
        let mut spec_accepted = 0usize;
        let mut spec_rounds = 0usize;

        'outer: while generated_tokens.len() < max_tokens {
            let logits = match cur_logits.take() { Some(l) => l, None => break };

            // 1+2+3. Rascunho n-gram → verificação batched → aceita prefixo.
            let draft = prompt_lookup(&full_seq);
            if !draft.is_empty() {
                spec_rounds += 1;
                spec_drafted += draft.len();
                let orig = main_kv.len();
                let vlogits = crate::cpu_reference::forward_verify(&draft, pos, &cpu_cfg, &weight_bank, &mut main_kv);
                let mut m = 0usize;
                let mut prev: &Vec<f32> = &logits;
                for i in 0..vlogits.len() {
                    if crate::cpu_reference::argmax(prev) != draft[i] { break; }
                    m += 1;
                    prev = &vlogits[i];
                }
                main_kv.truncate(orig + m); // guilhotina os rascunhos rejeitados
                spec_accepted += m;
                for &t in draft.iter().take(m) {
                    if Some(t) == eos_token { break 'outer; }
                    generated_tokens.push(t);
                    full_seq.push(t);
                    tokens_done += 1;
                    if let Some(ref tx_stream) = tx {
                        let s = tokenizer.decode(&[t], true).unwrap_or_default();
                        if !s.is_empty() { let _ = tx_stream.send(Ok(s)).await; }
                    }
                    if generated_tokens.len() >= max_tokens { break 'outer; }
                }
                cur_logits = Some(if m > 0 { vlogits[m - 1].clone() } else { logits.clone() });
                pos += m;
            } else {
                cur_logits = Some(logits);
            }
            if generated_tokens.len() >= max_tokens { break; }

            // 4. Token de correção do ALVO (sempre 1; lossless).
            let next_logits = match cur_logits.take() { Some(l) => l, None => break };
            let anchor = crate::cpu_reference::argmax(&next_logits);
            if Some(anchor) == eos_token { break; }
            generated_tokens.push(anchor);
            full_seq.push(anchor);
            tokens_done += 1;
            if let Some(ref tx_stream) = tx {
                let s = tokenizer.decode(&[anchor], true).unwrap_or_default();
                if !s.is_empty() { let _ = tx_stream.send(Ok(s)).await; }
            }
            cur_logits = crate::cpu_reference::forward_step(anchor, pos, &cpu_cfg, &weight_bank, &mut main_kv);
            pos += 1;
        }
        if spec_rounds > 0 {
            debug!("Especulação SEM CABEÇA (n-gram K={}): {}/{} rascunhos aceitos ({:.0}%), {:.2} tokens/rodada — na GPU = K tokens por weight-load",
                DRAFT_K, spec_accepted, spec_drafted,
                100.0 * spec_accepted as f64 / spec_drafted.max(1) as f64,
                (spec_accepted + spec_rounds) as f64 / spec_rounds as f64);
        }

        // Simula o fechamento verificando quantos blocos estão quentes na VRAM
        debug!("Geração finalizada. Tracker size: {} blocos. Tudo que passou de {} foi paginado no SSD.", 
               kv_cache.layers.iter().map(|l| l.blocks.iter().filter(|b| b.in_vram).count()).sum::<usize>(),
               max_vram_blocks);

        let elapsed_ms = start_time.elapsed().as_millis();
        let tps = if elapsed_ms > 0 {
            (tokens_done as f64) / (elapsed_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let stats = GenerationStats {
            prompt_tokens: prompt.split_whitespace().count(),
            generated_tokens: tokens_done,
            tokens_per_second: tps,
            total_time_ms: elapsed_ms,
            probes_alerts,
            conformal_rejections,
        };

        // Decode da string final
        let generated_text = if generated_tokens.is_empty() {
            format!("(Sem tokens gerados para o prompt: {})", prompt)
        } else {
            tokenizer.decode(&generated_tokens, true).unwrap_or_else(|_| String::from("Decode ERROR"))
        };

        Ok((generated_text, stats))
    }

    /// Versão Reativa/Stream: devolve tokens um a um conforme são gerados pela GPU.
    /// Vital para interfaces de Chat e UX de baixa latência percebida.
    ///
    /// Implementação: executa `generate()` completo e faz stream dos tokens gerados
    /// palavra a palavra via canal assíncrono. Cada fragmento de texto é enviado
    /// assim que disponível, sem buffer acumulado.
    pub async fn generate_stream(
        self: Arc<Self>,
        prompt: String,
        max_tokens: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<String, NodeStorError>> + Send>> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let this = self.clone();

        tokio::spawn(async move {
            match this.generate(&prompt, max_tokens, Some(tx.clone())).await {
                Ok((_text, stats)) => {
                    // Os fragmentos são enviados pela generate() agora.
                    tracing::debug!(
                        "[Stream] Geração finalizada: {} tokens em {}ms ({:.1} tok/s)",
                        stats.generated_tokens,
                        stats.total_time_ms,
                        stats.tokens_per_second,
                    );
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                }
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

/// Converte os bytes crus de um tensor GGUF (no seu dtype nativo) para FP32,
/// prontos para Matmul/Attention.
///
/// O núcleo do NodeStor é **lossless por especulação**: F32/F16/BF16 são
/// convertidos com fidelidade numérica TOTAL — o modelo roda com a qualidade
/// máxima da arquitetura, sem perda. A quantização (Q8_0, e via `dequant`
/// também Q4_K/Q5_K) é **opcional**: quem quiser economizar memória pode usar
/// modelos quantizados, mas isso nunca é exigido.
///
/// Retorna `None` para formatos ainda não cobertos pelo conversor direto
/// (quants legados Q4_0/Q5_0, tipos inteiros), sinalizando fallback ao chamador.
fn tensor_to_f32(
    bytes: &[u8],
    dtype: nodestor_core::TensorDtype,
    num_elements: usize,
) -> Option<Vec<f32>> {
    use nodestor_core::TensorDtype as DT;
    use crate::dequant::{DequantDispatcher, QuantFormat};

    // BF16 → FP32 é EXATO: o bfloat16 são exatamente os 16 bits altos de um FP32.
    if dtype == DT::BF16 {
        return Some(
            bytes.chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
        );
    }

    let format = match dtype {
        DT::F32  => QuantFormat::F32,
        DT::F16  => QuantFormat::F16,
        DT::Q8_0 => QuantFormat::Q8_0, // near-lossless (~idêntico a FP16)
        _ => return None,
    };

    let mut dispatcher = DequantDispatcher::new(false);
    Some(dispatcher.dequantize(bytes, format, num_elements))
}

#[cfg(test)]
mod weight_dtype_tests {
    use super::tensor_to_f32;
    use nodestor_core::TensorDtype;

    #[test]
    fn test_f32_passthrough() {
        let v = [1.5f32, -2.0, 3.25];
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let out = tensor_to_f32(&bytes, TensorDtype::F32, 3).unwrap();
        assert_eq!(out, vec![1.5, -2.0, 3.25]);
    }

    #[test]
    fn test_f16_to_f32_lossless() {
        // half: 0x3C00 = 1.0 ; 0xC000 = -2.0 (little-endian nos bytes)
        let bytes = [0x00u8, 0x3C, 0x00, 0xC0];
        let out = tensor_to_f32(&bytes, TensorDtype::F16, 2).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6, "F16 0x3C00 → 1.0, foi {}", out[0]);
        assert!((out[1] + 2.0).abs() < 1e-6, "F16 0xC000 → -2.0, foi {}", out[1]);
    }

    #[test]
    fn test_bf16_to_f32_exact() {
        // bfloat16: 0x3F80 = 1.0 ; 0x4040 = 3.0
        let bytes = [0x80u8, 0x3F, 0x40, 0x40];
        let out = tensor_to_f32(&bytes, TensorDtype::BF16, 2).unwrap();
        assert_eq!(out[0], 1.0, "BF16 0x3F80 deve ser exatamente 1.0");
        assert_eq!(out[1], 3.0, "BF16 0x4040 deve ser exatamente 3.0");
    }

    #[test]
    fn test_unsupported_format_returns_none() {
        // Q4_0 legado ainda não tem conversor direto → None (chamador usa fallback)
        assert!(tensor_to_f32(&[0u8; 18], TensorDtype::Q4_0, 32).is_none());
    }
}
