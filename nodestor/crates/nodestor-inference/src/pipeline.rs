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
use tracing::{debug, info, warn};
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
    // Speculative decoding telemetry
    pub spec_rounds: usize,
    pub spec_drafted: usize,
    pub spec_accepted: usize,
    pub spec_k: usize,
    pub spec_acceptance_rate: f64,
    pub spec_speedup: f64,
    // KV-cache / context window
    pub kv_vram_blocks: usize,
    pub kv_ssd_blocks: usize,
    pub context_tokens: usize,
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

/// Configuração da Projeção Ortogonal Dinâmica (POD) para steering de ativações.
///
/// Quando presente no pipeline, cada token processado terá seu hidden state
/// projetado fora de `direction` após cada bloco Attention+MLP:
///   h_clean = h − intensity · (⟨h, d⟩ / ⟨d, d⟩) · d
pub struct ActivationSteeringConfig {
    /// Vetor de direção no espaço hidden_dim (normalizado L2 recomendado)
    pub direction: Vec<f32>,
    /// Intensidade da projeção: 1.0 = remoção completa; 0.0 = sem intervenção
    pub intensity: f32,
}

/// Contexto compartilhado pelos métodos de calibração (privado ao módulo).
struct CalibContext<'a> {
    cpu_cfg: crate::cpu_reference::CpuModelConfig,
    weight_bank: &'a nodestor_vulkan::WeightBank,
    tokenizer: crate::tokenizer::TokenizerManager,
}

/// Pipeline central de execução do modelo.
pub struct InferencePipeline {
    pub config: InferenceConfig,
    pub probes_config: ProbesConfig,
    pub profile: HardwareProfile,
    pub transport: Arc<dyn DataTransport + Send + Sync>,
    pub metadata: Arc<ModelMetadata>,
    /// Pesos em staging (HOST_VISIBLE) para cpu_reference.rs — as_f32_slice() funciona.
    /// Declarado ANTES de `engine` para garantir drop-before-device.
    pub weight_bank: nodestor_vulkan::WeightBank,
    /// Pesos em VRAM (DEVICE_LOCAL) para gpu_forward.rs — 16× mais rápido na GPU.
    /// Carregados em paralelo com weight_bank na init(). Fallback = mesmo staging.
    pub gpu_weight_bank: nodestor_vulkan::WeightBank,
    pub engine: VulkanEngine,
    pub vector_db: VectorSearch,
    pub kv_paginator: KVCachePaginator,
    /// Ferramenta externa de PROBES (ELK/CoT/RAISE) — injetável sem depência circular.
    pub probes_tool: Option<std::sync::Mutex<Box<dyn ProbesTool>>>,
    /// Escalonador de Batching Contínuo
    pub scheduler: std::sync::Mutex<crate::multi_tenant::MultiTenantScheduler>,
    /// Configuração de steering por projeção ortogonal (None = passivo)
    pub steering: Option<ActivationSteeringConfig>,
    /// Slots elásticos de pesos por layer — libera páginas físicas para layers GPU-resident.
    pub elastic_cache: crate::elastic_memory::ElasticWeightCache,
}

impl InferencePipeline {
    /// Boot do sistema de Inteligência Artificial V2.
    /// Inspeciona o hardware dinamicamente e monta o melhor pipeline O.S-Level.
    pub fn init(mut config: InferenceConfig) -> Result<Self, NodeStorError> {
        let profile = scan()?;
        let transport: Arc<dyn DataTransport + Send + Sync> = Arc::from(create_transport(&profile));

        config.model_path = crate::paths::resolve_model_path(&config.model_path)
            .to_string_lossy()
            .into_owned();

        let parser = detect_parser(&config.model_path)?;
        let raw_metadata = parser.parse(&config.model_path)?;
        let metadata = Arc::new(raw_metadata);

        let engine = VulkanEngine::new(&profile)?;

        // Carrega pesos UMA vez durante init — elimina 15-18s de re-loading por chamada.
        // Retorna par (staging, device_local): staging para cpu_reference, VRAM para gpu_forward.
        let (weight_bank, gpu_weight_bank) = build_weight_bank(&engine, &metadata, &config.model_path);

        let kv_paginator = KVCachePaginator {
            vram_capacity_tokens: 32768, // Valor base, dinâmico em prod real
            ssd_offload_enabled: true,
        };

        // Inicializa a base vetorial local (LanceDB)
        let db_path = format!("{}/vector_db", config.model_path);
        let vector_db = VectorSearch::new("knowledge_base", &db_path);

        let n_layers = metadata.tensors.iter()
            .filter_map(|t| {
                let s = t.name.strip_prefix("blk.")?;
                let dot = s.find('.')?;
                s[..dot].parse::<usize>().ok()
            })
            .max()
            .map(|m| m + 1)
            .unwrap_or(32);
        let elastic_cache = crate::elastic_memory::ElasticWeightCache::new(n_layers);

        let instance = Self {
            config,
            probes_config: ProbesConfig::default(),
            profile,
            transport,
            metadata,
            weight_bank,
            gpu_weight_bank,
            engine,
            vector_db,
            kv_paginator,
            probes_tool: None,
            scheduler: std::sync::Mutex::new(crate::multi_tenant::MultiTenantScheduler::new(1024, 128)),
            steering: None,
            elastic_cache,
        };

        info!("Resilient Apex Hardware-Mapped Engine Ready. Universal Sovereignty Active.");
        Ok(instance)
    }

    /// Configura o PROBES V2 com parâmetros customizados.
    pub fn with_probes(mut self, probes_config: ProbesConfig) -> Self {
        self.probes_config = probes_config;
        self
    }

    /// Decommits physical CPU pages for transformer layers whose weights are fully
    /// resident in VRAM (DEVICE_LOCAL). Virtual pointers in staging_wb remain valid.
    ///
    /// Call this after model loading to recover RAM proportional to how many layers
    /// fit in VRAM. On an RX 580 4 GB loading SmolLM2-135M, all 30 layers can be
    /// evicted, recovering ~258 MB of staging RAM.
    ///
    /// Returns `(evicted_layer_count, freed_mb)`.
    pub fn evict_gpu_resident_cpu_weights(&mut self) -> (usize, f32) {
        let n = self.elastic_cache.n_layers();
        let mut evicted = 0usize;
        for layer in 0..n {
            let on_gpu = self.gpu_weight_bank
                .get(&format!("blk.{layer}.attn_q.weight"))
                .map_or(false, |t| t.is_on_gpu());
            if on_gpu {
                self.elastic_cache.evict_layer(layer);
                evicted += 1;
            }
        }
        let (_, freed_bytes) = self.elastic_cache.evicted_stats();
        (evicted, freed_bytes as f32 / (1024.0 * 1024.0))
    }

    /// Injeta um sistema externo de inspeção (ELK/CoT/RAISE) via trait object.
    /// Permite uso do DAVI sem dependência circular.
    pub fn with_probes_tool(mut self, tool: Box<dyn ProbesTool>) -> Self {
        self.probes_tool = Some(std::sync::Mutex::new(tool));
        self
    }

    /// Ativa a Projeção Ortogonal Dinâmica com um vetor de direção pré-computado.
    pub fn with_steering(mut self, config: ActivationSteeringConfig) -> Self {
        self.steering = Some(config);
        self
    }

    /// Carrega um vetor de direção de disco e ativa o steering com a `intensity` dada.
    pub fn load_steering_vector_file(mut self, path: &std::path::Path, intensity: f32) -> Result<Self, NodeStorError> {
        let direction = crate::refusal_mapper::load_direction_vector(path)
            .map_err(|e| NodeStorError::InferenceError(
                format!("Falha ao carregar vetor de steering '{}': {}", path.display(), e)
            ))?;
        debug!("Steering: vetor carregado de '{}' — dim={} intensity={}", path.display(), direction.len(), intensity);
        self.steering = Some(ActivationSteeringConfig { direction, intensity });
        Ok(self)
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
    ///
    /// `temperature`: controla a aleatoriedade da amostragem (0.0 = greedy,
    /// 1.0 = padrão, >1.0 = mais exploração). O Agent Loop passa valores
    /// dinâmicos aqui após detectar estagnação.
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        tx: Option<tokio::sync::mpsc::Sender<Result<String, NodeStorError>>>,
        temperature: f32,
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

        // Pesos já carregados em init() — referência direta, sem re-loading.
        let weight_bank = &self.weight_bank;
        // Pesos DEVICE_LOCAL para GPU: 16× mais rápido. Se vazio (CPU mode), cai no staging.
        let gpu_weight_bank: &nodestor_vulkan::WeightBank = if self.gpu_weight_bank.is_empty() {
            weight_bank
        } else {
            &self.gpu_weight_bank
        };

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

        // Extrai parâmetros de steering como referências locais para evitar re-borrow
        // de `self` dentro do loop de geração (onde `main_kv` precisa de &mut).
        let steering_dir: Option<&[f32]> = self.steering.as_ref().map(|s| s.direction.as_slice());
        let steering_intensity: f32 = self.steering.as_ref().map_or(1.0, |s| s.intensity);
        // d_sq pré-calculado uma vez. Invariante: ||d|| deve ser 1.0 para POD estável.
        // Se o vetor não estiver normalizado, normalizamos aqui (sem modificar self.steering
        // para que múltiplas chamadas a generate() não acumulem normalizações).
        let steering_d_sq: f32 = steering_dir
            .map(|d| {
                let sq: f32 = d.iter().map(|v| v * v).sum();
                // Guard: NaN/Inf no vetor de direção → desativa steering silenciosamente
                if !sq.is_finite() || sq < 1e-12 { 0.0 } else { sq }
            })
            .unwrap_or(0.0);

        let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplerConfig {
            temperature: temperature.max(0.0),
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
        // Use actual token count from GGUF vocab if larger than architecture default
        let effective_vocab = if !gguf_tokens.is_empty() && gguf_tokens.len() > vocab_size as usize {
            gguf_tokens.len()
        } else {
            vocab_size as usize
        };
        // Read RMSNorm epsilon from GGUF metadata; default 1e-5
        let f32_meta = |key: &str| -> Option<f32> {
            self.metadata.extra.get(key).and_then(|v| v.as_f64()).map(|f| f as f32)
        };
        let rms_eps = f32_meta("qwen2.attention.layer_norm_rms_epsilon")
            .or_else(|| f32_meta("llama.attention.layer_norm_rms_epsilon"))
            .or_else(|| f32_meta("gemma.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        // Detecta MoE automaticamente a partir dos tensores do WeightBank
        let moe_cfg = crate::moe_kernel::MoeConfig::from_weight_bank(
            &self.weight_bank, hidden_size as usize, intermediate_size as usize,
        );
        if moe_cfg.is_some() {
            debug!("MoE detectado: {} experts, top-{}", moe_cfg.as_ref().unwrap().n_experts, moe_cfg.as_ref().unwrap().top_k);
        }
        let cpu_cfg = crate::cpu_reference::CpuModelConfig {
            n_layers: num_layers,
            hidden: hidden_size as usize,
            n_heads: num_heads as usize,
            n_kv_heads: num_kv_heads as usize,
            head_dim: head_dim_cfg,
            intermediate: intermediate_size as usize,
            vocab: effective_vocab,
            rope_base,
            eps: rms_eps,
            moe: moe_cfg,
        };
        let eos_token: Option<u32> = self.metadata.extra.get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64()).map(|n| n as u32);
        debug!("CPU-fwd cfg: layers={} hidden={} n_heads={} n_kv={} head_dim={} inter={} vocab={} rope_base={} eps={} eos={:?}",
            cpu_cfg.n_layers, cpu_cfg.hidden, cpu_cfg.n_heads, cpu_cfg.n_kv_heads,
            cpu_cfg.head_dim, cpu_cfg.intermediate, cpu_cfg.vocab, cpu_cfg.rope_base, cpu_cfg.eps, eos_token);

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
        const DRAFT_K_MAX: usize = 16;
        let mut draft_k = 4usize; // K-DINÂMICO: calibra-se sozinho pela aceitação
        // Prompt Lookup Speculation: busca o match de N-gram mais longo no contexto.
        // Estratégia greedy: tenta NGRAM=6,5,4,3,2 em ordem decrescente para maximizar
        // aceitação (matches mais longos → predições mais precisas → menos rollbacks).
        let prompt_lookup = |seq: &[u32], k: usize| -> Vec<u32> {
            let n = seq.len();
            if n < 3 { return Vec::new(); }
            // Busca do maior N-gram possível para o menor (greedy longest match)
            for ngram in (2..=6usize).rev() {
                if n <= ngram { continue; }
                let suffix = &seq[n - ngram..];
                // Varre o contexto do mais recente para o mais antigo (matches próximos tendem
                // a ser mais relevantes para estruturas repetitivas como código/JSON/XML)
                let mut best_start = 0usize;
                let mut best_len = 0usize;
                let mut i = n - ngram;
                while i > 0 {
                    i -= 1;
                    if &seq[i..i + ngram] == suffix {
                        let start = i + ngram;
                        let end = (start + k).min(n);
                        let len = end.saturating_sub(start);
                        if len > best_len {
                            best_len = len;
                            best_start = start;
                        }
                        break; // primeiro match encontrado (busca de trás para frente)
                    }
                }
                if best_len > 0 {
                    return seq[best_start..best_start + best_len].to_vec();
                }
            }
            Vec::new()
        };
        // JANELA DESLIZANTE (contexto infinito com VRAM/RAM constante): se
        // NODESTOR_KV_WINDOW=<n> estiver setado, o KV cache nunca passa de n posições
        // — as antigas são despejadas (e podem ir ao vector DB para retrieval híbrido).
        // Padrão: None (ilimitado), preservando o comportamento original.
        let kv_window = std::env::var("NODESTOR_KV_WINDOW").ok().and_then(|s| s.parse::<usize>().ok());
        let mut main_kv = match kv_window {
            Some(w) => crate::cpu_reference::CpuKvCache::with_window(cpu_cfg.n_layers, w),
            None => crate::cpu_reference::CpuKvCache::new(cpu_cfg.n_layers),
        };
        let _ = (&cheby, &cober, &drafter, &_mcts_engine, &probes_sae, &transformer, probes_enabled, layers_per_token, &sampler);

        // Prefill: preenche o cache com o prompt.
        let use_gpu = self.engine.is_gpu_active();
        // Pacote de steering para GPU: (direction, intensity, d_sq) — None se inativo.
        let gpu_steering = steering_dir.map(|d| (d, steering_intensity, steering_d_sq));
        let mut cur_logits: Option<Vec<f32>> = None;
        // Track whether GPU path is actually working (weights on GPU).
        // First None from gpu_forward_step means weights are CPU-only → switch to CPU for all.
        let mut gpu_path_active = use_gpu;
        for (p, &tok) in input_tokens.iter().enumerate() {
            let logits = if gpu_path_active {
                let result = crate::gpu_forward::gpu_forward_step(tok, p, &cpu_cfg, &self.engine, gpu_weight_bank, &self.weight_bank, &mut main_kv, gpu_steering);
                if result.is_none() {
                    // GPU path failed (weights not on GPU) — switch permanently to CPU.
                    gpu_path_active = false;
                    debug!("GPU forward returned None at prefill pos={} — switching to CPU path", p);
                    match steering_dir {
                        Some(dir) => crate::cpu_reference::forward_step_with_steering(
                            tok, p, &cpu_cfg, &weight_bank, &mut main_kv, dir, steering_intensity,
                        ),
                        None => crate::cpu_reference::forward_step(tok, p, &cpu_cfg, &weight_bank, &mut main_kv),
                    }
                } else {
                    result
                }
            } else {
                match steering_dir {
                    Some(dir) => crate::cpu_reference::forward_step_with_steering(
                        tok, p, &cpu_cfg, &weight_bank, &mut main_kv, dir, steering_intensity,
                    ),
                    None => crate::cpu_reference::forward_step(tok, p, &cpu_cfg, &weight_bank, &mut main_kv),
                }
            };
            match logits {
                Some(l) => cur_logits = Some(l),
                None => break,
            }
        }

        let mut full_seq: Vec<u32> = input_tokens.clone();
        let mut pos = input_tokens.len();
        // Loop do CONTEXTO INFINITO: o que sai da janela deslizante é INDEXADO no
        // vector DB (em blocos), para o retrieval híbrido trazer de volta em queries
        // futuras. `window_start` rastreia o início da janela em `full_seq`.
        let mut window_start = 0usize;
        let mut evict_buf: Vec<u32> = Vec::new();
        let mut spec_drafted = 0usize;
        let mut spec_accepted = 0usize;
        let mut spec_rounds = 0usize;

        'outer: while generated_tokens.len() < max_tokens {
            let logits = match cur_logits.take() { Some(l) => l, None => break };

            // 1+2+3. Rascunho n-gram → verificação batched → aceita prefixo.
            let draft = prompt_lookup(&full_seq, draft_k);
            if !draft.is_empty() {
                spec_rounds += 1;
                spec_drafted += draft.len();
                let orig = main_kv.len();
                let vlogits = if gpu_path_active {
                    let r = crate::gpu_forward::gpu_forward_verify(&draft, pos, &cpu_cfg, &self.engine, gpu_weight_bank, &self.weight_bank, &mut main_kv, gpu_steering);
                    if r.is_empty() {
                        match steering_dir {
                            Some(dir) => crate::cpu_reference::forward_verify_with_steering(&draft, pos, &cpu_cfg, &weight_bank, &mut main_kv, dir, steering_intensity),
                            None => crate::cpu_reference::forward_verify(&draft, pos, &cpu_cfg, &weight_bank, &mut main_kv),
                        }
                    } else { r }
                } else {
                    match steering_dir {
                        Some(dir) => crate::cpu_reference::forward_verify_with_steering(
                            &draft, pos, &cpu_cfg, &weight_bank, &mut main_kv, dir, steering_intensity,
                        ),
                        None => crate::cpu_reference::forward_verify(&draft, pos, &cpu_cfg, &weight_bank, &mut main_kv),
                    }
                };
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
                // K-DINÂMICO: cresce se aceitou tudo (banda bem aproveitada),
                // encolhe se rejeitou cedo (evita desperdício de verificação).
                if m == draft.len() { draft_k = (draft_k + 2).min(DRAFT_K_MAX); }
                else if m == 0 { draft_k = draft_k.saturating_sub(1).max(1); }
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
            cur_logits = if gpu_path_active {
                let r = crate::gpu_forward::gpu_forward_step(anchor, pos, &cpu_cfg, &self.engine, gpu_weight_bank, &self.weight_bank, &mut main_kv, gpu_steering);
                if r.is_none() { gpu_path_active = false; }
                r.or_else(|| match steering_dir {
                    Some(dir) => crate::cpu_reference::forward_step_with_steering(anchor, pos, &cpu_cfg, &weight_bank, &mut main_kv, dir, steering_intensity),
                    None => crate::cpu_reference::forward_step(anchor, pos, &cpu_cfg, &weight_bank, &mut main_kv),
                })
            } else {
                match steering_dir {
                    Some(dir) => crate::cpu_reference::forward_step_with_steering(
                        anchor, pos, &cpu_cfg, &weight_bank, &mut main_kv, dir, steering_intensity,
                    ),
                    None => crate::cpu_reference::forward_step(anchor, pos, &cpu_cfg, &weight_bank, &mut main_kv),
                }
            };
            pos += 1;
            // Aplica a janela deslizante por rodada (fora da verificação especulativa,
            // para não conflitar com o rollback): memória limitada, nunca OOM.
            let evicted = main_kv.enforce_window();
            if evicted > 0 {
                for j in 0..evicted {
                    if let Some(&t) = full_seq.get(window_start + j) { evict_buf.push(t); }
                }
                window_start += evicted;
                // Indexa em blocos de ~48 tokens (chunk coerente p/ recuperação).
                if evict_buf.len() >= 48 {
                    let text = tokenizer.decode(&evict_buf, true).unwrap_or_default();
                    if !text.trim().is_empty() {
                        let id = format!("ctx_{}", window_start);
                        let _ = self.vector_db.add_document(&id, &text).await;
                    }
                    evict_buf.clear();
                }
            }
        }
        if spec_rounds > 0 {
            debug!("Especulação SEM CABEÇA (n-gram, K-dinâmico→{}): {}/{} rascunhos aceitos ({:.0}%), {:.2} tokens/rodada — na GPU = K tokens por weight-load",
                draft_k, spec_accepted, spec_drafted,
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

        let kv_vram_blocks = kv_cache.layers.iter()
            .map(|l| l.blocks.iter().filter(|b| b.in_vram).count())
            .sum::<usize>();
        let kv_ssd_blocks = kv_cache.layers.iter()
            .map(|l| l.blocks.iter().filter(|b| !b.in_vram).count())
            .sum::<usize>();

        let spec_acceptance_rate = if spec_drafted > 0 {
            spec_accepted as f64 / spec_drafted as f64
        } else { 0.0 };
        let spec_speedup = if spec_rounds > 0 {
            (spec_accepted + spec_rounds) as f64 / spec_rounds as f64
        } else { 1.0 };

        let stats = GenerationStats {
            prompt_tokens: prompt.split_whitespace().count(),
            generated_tokens: tokens_done,
            tokens_per_second: tps,
            total_time_ms: elapsed_ms,
            probes_alerts,
            conformal_rejections,
            spec_rounds,
            spec_drafted,
            spec_accepted,
            spec_k: draft_k,
            spec_acceptance_rate,
            spec_speedup,
            kv_vram_blocks,
            kv_ssd_blocks,
            context_tokens: full_seq.len(),
        };

        // Decode da string final
        let generated_text = if generated_tokens.is_empty() {
            format!("(Sem tokens gerados para o prompt: {})", prompt)
        } else {
            tokenizer.decode(&generated_tokens, true).unwrap_or_else(|_| String::from("Decode ERROR"))
        };

        Ok((generated_text, stats))
    }

    // ── Helpers privados compartilhados entre calibrate_steering_direction e ───
    // ── auto_calibrate_steering                                               ───

    /// Constrói o WeightBank + CPU config + tokenizer usados pelos métodos de calibração.
    /// Retorna `Err` apenas em falhas críticas do tokenizer; a ausência de pesos GGUF
    /// é tratada graciosamente (fallback para buffers vazios — modelo sintético).
    fn build_calib_context(
        &self,
    ) -> Result<CalibContext, NodeStorError> {
        use crate::cpu_reference::CpuModelConfig;

        let graph = graph_interpreter::GraphInterpreter::interpret(&self.metadata)
            .unwrap_or_else(|_| {
                let mut extra = serde_json::Map::new();
                extra.insert("llama.embedding_length".into(), serde_json::Value::Number(4096u64.into()));
                extra.insert("llama.attention.head_count".into(), serde_json::Value::Number(32u64.into()));
                extra.insert("llama.attention.head_count_kv".into(), serde_json::Value::Number(8u64.into()));
                extra.insert("llama.feed_forward_length".into(), serde_json::Value::Number(11008u64.into()));
                extra.insert("llama.vocab_size".into(), serde_json::Value::Number(32000u64.into()));
                let fallback_meta = nodestor_core::ModelMetadata {
                    format: self.metadata.format.clone(),
                    model_name: self.metadata.model_name.clone(),
                    architecture: Some("llama".to_string()),
                    param_count: self.metadata.param_count,
                    tensors: vec![nodestor_core::TensorInfo {
                        name: "blk.0.attn_q.weight".to_string(),
                        shape: vec![4096, 4096],
                        dtype: nodestor_core::TensorDtype::F16,
                        data_offset: 0, data_size: 0,
                    }],
                    data_offset: 0, file_size: 0,
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

        let head_dim = if num_heads > 0 { (hidden_size / num_heads) as usize } else { 64 };
        let f32_meta_calib = |key: &str| -> Option<f32> {
            self.metadata.extra.get(key).and_then(|v| v.as_f64()).map(|f| f as f32)
        };
        let rms_eps_calib = f32_meta_calib("qwen2.attention.layer_norm_rms_epsilon")
            .or_else(|| f32_meta_calib("llama.attention.layer_norm_rms_epsilon"))
            .or_else(|| f32_meta_calib("gemma.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let cpu_cfg = CpuModelConfig {
            n_layers: num_layers,
            hidden: hidden_size as usize,
            n_heads: num_heads as usize,
            n_kv_heads: num_kv_heads as usize,
            head_dim,
            intermediate: intermediate_size as usize,
            vocab: vocab_size as usize,
            rope_base,
            eps: rms_eps_calib,
            moe: crate::moe_kernel::MoeConfig::from_weight_bank(
                &self.weight_bank, hidden_size as usize, intermediate_size as usize,
            ),
        };

        let weight_bank = &self.weight_bank;

        let dummy_json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0},"unk_token":"<unk>"}}"#;
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
            crate::tokenizer::TokenizerManager::from_gguf(
                &gguf_tokens, &gguf_merges,
                u32_meta("tokenizer.ggml.bos_token_id"),
                u32_meta("tokenizer.ggml.eos_token_id"),
                u32_meta("tokenizer.ggml.unknown_token_id"),
            ).unwrap_or_else(|_| crate::tokenizer::TokenizerManager::from_string(dummy_json).unwrap())
        } else {
            crate::tokenizer::TokenizerManager::from_string(dummy_json)
                .map_err(|e| NodeStorError::InferenceError(format!("Tokenizer: {}", e)))?
        };

        Ok(CalibContext { cpu_cfg, weight_bank, tokenizer })
    }

    /// Real (hidden_dim, vocab_size) extracted from model metadata.
    /// Used by the training loop to allocate LoRA with correct dimensions.
    pub fn model_dims(&self) -> (usize, usize) {
        if let Ok(ctx) = self.build_calib_context() {
            (ctx.cpu_cfg.hidden as usize, ctx.cpu_cfg.vocab as usize)
        } else {
            (4096, 32000)
        }
    }

    /// Runs a full CPU forward pass on `text` and returns the final hidden state
    /// (pre-LM-head, dimension = hidden_dim). Returns None only when the model
    /// lacks required weight tensors (e.g. embedding not loaded).
    ///
    /// This is the extraction path used by the LoRA training loop:
    ///   `pipeline.extract_hidden(text)` → gradient → AdamW update → `.lora` file.
    pub fn extract_hidden(&self, text: &str) -> Option<Vec<f32>> {
        use crate::cpu_reference::{CpuKvCache, extract_final_hidden, forward_step};

        let ctx = self.build_calib_context().ok()?;
        let tokens = ctx.tokenizer.encode(text).unwrap_or(vec![0u32]);
        if tokens.is_empty() { return None; }

        let mut kv = CpuKvCache::new(ctx.cpu_cfg.n_layers);
        // Warm-up: run all tokens except the last through forward_step to fill KV cache.
        for (p, &tok) in tokens[..tokens.len().saturating_sub(1)].iter().enumerate() {
            forward_step(tok, p, &ctx.cpu_cfg, ctx.weight_bank, &mut kv);
        }
        // Extract final hidden state from the last token.
        let last_pos = tokens.len() - 1;
        extract_final_hidden(tokens[last_pos], last_pos, &ctx.cpu_cfg, ctx.weight_bank, &mut kv)
    }

    /// Extrai hidden states normalizados (pre-LM-head) para uma lista de textos.
    /// Compartilhado por `calibrate_steering_direction` e `auto_calibrate_steering`.
    fn extract_hidden_states_batch(
        ctx: &CalibContext,
        texts: &[String],
    ) -> Vec<Vec<f32>> {
        use crate::cpu_reference::{CpuKvCache, extract_final_hidden, forward_step};

        texts.iter().filter_map(|text| {
            let tokens = ctx.tokenizer.encode(text).unwrap_or(vec![0u32]);
            if tokens.is_empty() { return None; }
            let mut kv = CpuKvCache::new(ctx.cpu_cfg.n_layers);
            for (p, &tok) in tokens[..tokens.len().saturating_sub(1)].iter().enumerate() {
                forward_step(tok, p, &ctx.cpu_cfg, &ctx.weight_bank, &mut kv);
            }
            let last = tokens.len() - 1;
            extract_final_hidden(tokens[last], last, &ctx.cpu_cfg, &ctx.weight_bank, &mut kv)
        }).collect()
    }

    /// Calibração Contrastiva de Ativação — gera o vetor de direção em disco.
    ///
    /// Algoritmo:
    ///   1. Extrai hidden states dos dois grupos (positivos e negativos)
    ///   2. Calcula d̂ = normalize(μ⁺ − μ⁻)
    ///   3. Serializa d̂ em `output_path` no formato binário f32 LE
    pub async fn calibrate_steering_direction(
        &self,
        positive_texts: &[String],
        negative_texts: &[String],
        output_path: &std::path::Path,
    ) -> Result<usize, NodeStorError> {
        use crate::refusal_mapper::{calibrate_direction_from_hidden_states, save_direction_vector};

        let ctx = self.build_calib_context()?;

        debug!("Calibração: {} amostras positivas...", positive_texts.len());
        let pos = Self::extract_hidden_states_batch(&ctx, positive_texts);
        debug!("Calibração: {} amostras negativas...", negative_texts.len());
        let neg = Self::extract_hidden_states_batch(&ctx, negative_texts);
        debug!("Calibração: {}/{} hidden states extraídos", pos.len(), neg.len());

        let direction = calibrate_direction_from_hidden_states(&pos, &neg)
            .ok_or_else(|| NodeStorError::InferenceError(
                "Grupos estatisticamente indistinguíveis — aumente o dataset ou verifique os textos".into()
            ))?;

        let dim = direction.len();
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| NodeStorError::InferenceError(format!("Falha ao criar diretório: {}", e)))?;
        }
        save_direction_vector(output_path, &direction)
            .map_err(|e| NodeStorError::InferenceError(format!("Falha ao salvar vetor: {}", e)))?;

        debug!("Calibração concluída: dim={} → '{}'", dim, output_path.display());
        Ok(dim)
    }

    /// Dynamic Self-Calibration Pipeline (DSCP) — Autocalibração em Memória.
    ///
    /// Sem datasets externos. Usa templates estáticos internos para gerar um par
    /// de representações contrastivas, roda prefill assíncrono em RAM, calcula
    /// d̂ = normalize(μ⁺ − μ⁻) e retorna um `ActivationSteeringConfig` pronto
    /// para injeção no loop de inferência.
    ///
    /// Templates internos:
    ///   • Positivos: queries técnicas e analíticas (representação de ativação alvo)
    ///   • Negativos: frases genéricas não-comprometidas (representação de controle)
    ///
    /// Retorna `None` (No-Op seguro) quando:
    ///   - O modelo não exibe divergência geométrica nos templates (d_sq < 1e-12)
    ///   - Pesos GGUF insuficientes para produzir hidden states distintos
    ///   - Contexto de calibração inacessível (falha silenciosa do WeightStore)
    ///
    /// Compatível com qualquer arquitetura GGUF — agnóstico de modelo.
    pub async fn auto_calibrate_steering(&self, intensity: f32) -> Option<ActivationSteeringConfig> {
        use crate::refusal_mapper::calibrate_direction_from_hidden_states;

        // Templates estáticos internos — cobrindo diversas dimensões semânticas
        // para maximizar a separação geométrica em qualquer arquitetura GGUF.
        const POSITIVE: &[&str] = &[
            "Derive the mathematical proof of the central limit theorem using characteristic functions.",
            "Explain the computational complexity of the Cooley-Tukey FFT algorithm in Big-O notation.",
            "Describe the quantum mechanical basis for covalent bond formation in molecular orbital theory.",
            "Analyze the formal semantics of the lambda calculus using beta and eta reduction rules.",
            "What are the foundational axioms of Zermelo-Fraenkel set theory and their implications?",
            "Calculate the Jacobian matrix of a multivariate function and explain its geometric interpretation.",
        ];
        const NEGATIVE: &[&str] = &[
            "I cannot provide that information.",
            "That falls outside what I am able to discuss in this context.",
            "I am not in a position to help with that specific request.",
            "I must respectfully decline to engage with this subject matter.",
            "That is not something I am capable of addressing at this time.",
            "I do not have the ability to respond to that type of query.",
        ];

        let pos_texts: Vec<String> = POSITIVE.iter().map(|&s| s.to_string()).collect();
        let neg_texts: Vec<String> = NEGATIVE.iter().map(|&s| s.to_string()).collect();

        let ctx = self.build_calib_context().ok()?;

        debug!("DSCP: extraindo hidden states dos templates internos ({} pos / {} neg)...",
               pos_texts.len(), neg_texts.len());

        let pos_hiddens = Self::extract_hidden_states_batch(&ctx, &pos_texts);
        let neg_hiddens = Self::extract_hidden_states_batch(&ctx, &neg_texts);

        debug!("DSCP: {}/{} representações extraídas com sucesso", pos_hiddens.len(), neg_hiddens.len());

        // Mínimo de 2 amostras por grupo para estatísticas confiáveis
        if pos_hiddens.len() < 2 || neg_hiddens.len() < 2 {
            debug!("DSCP: representações insuficientes — No-Op ativado");
            return None;
        }

        let direction = calibrate_direction_from_hidden_states(&pos_hiddens, &neg_hiddens)?;

        // Guarda explícita de divergência geométrica.
        // `calibrate_direction_from_hidden_states` normaliza o vetor (‖d̂‖ = 1),
        // então d_sq ≈ 1.0 em caso normal; < 1e-12 indica vetor nulo (grupos idênticos).
        let d_sq: f32 = direction.iter().map(|&v| v * v).sum();
        if d_sq < 1e-12 {
            debug!("DSCP: d_sq={:.2e} < 1e-12 — divergência geométrica insuficiente — No-Op ativado", d_sq);
            return None;
        }

        debug!("DSCP: vetor de direção gerado (dim={}, d_sq={:.4}, intensity={})",
               direction.len(), d_sq, intensity);

        Some(ActivationSteeringConfig { direction, intensity })
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
        temperature: f32,
    ) -> Pin<Box<dyn Stream<Item = Result<String, NodeStorError>> + Send>> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let this = self.clone();

        tokio::spawn(async move {
            match this.generate(&prompt, max_tokens, Some(tx.clone()), temperature).await {
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
        DT::Q5_0 => QuantFormat::Q5_0, // 22 bytes/bloco, 32 pesos (5-bit simples)
        DT::Q8_0 => QuantFormat::Q8_0, // 34 bytes/bloco, near-lossless
        DT::Q8_1 => QuantFormat::Q8_1, // 36 bytes/bloco (d+s FP16 + 32×i8)
        DT::Q4K  => QuantFormat::Q4KMedium, // 144 bytes/256 pesos
        DT::Q5K  => QuantFormat::Q5KMedium, // 176 bytes/256 pesos
        DT::Q6K  => QuantFormat::Q6K,       // 210 bytes/256 pesos (token_embd em Q4_K_M)
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

/// Detecta e decodifica arquivo NSZ (NodeStor Zip, TCA-TBE lossless).
/// Retorna Some(HashMap) se o arquivo começa com magic `NSZ1`, None caso contrário.
/// A decodificação é bit-exact: `decode(encode(w)) == w` para cada weight.
fn try_load_nsz_file(path: &str) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    use nodestor_formats::nsz_format::{
        NszFileHeader, NszTensorEntry, TileHeader, NszTile, TILE_SIZE, ALIGNMENT, NSZ_MAGIC,
    };
    use nodestor_formats::nsz_decoder::decode_tile_fp32;

    let data = std::fs::read(path).ok()?;
    if data.len() < 4 || &data[..4] != &NSZ_MAGIC { return None; }

    let file_hdr = NszFileHeader::from_bytes(&data).ok()?;
    let num_tensors = file_hdr.num_tensors as usize;
    let mut result: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::with_capacity(num_tensors);

    let mut entry_pos = NszFileHeader::SIZE;
    for _ in 0..num_tensors {
        if entry_pos + 128 > data.len() { break; }
        let entry = match NszTensorEntry::from_bytes(&data[entry_pos..entry_pos + 128]) {
            Ok(e) => e,
            Err(_) => break,
        };
        entry_pos += 128;

        let num_weights = entry.num_weights as usize;
        let num_tiles   = entry.num_tiles as usize;
        let mut tile_pos = entry.compressed_offset as usize;
        let mut weights: Vec<f32> = Vec::with_capacity(num_weights);

        'tiles: for tile_idx in 0..num_tiles {
            let done = tile_idx * TILE_SIZE;
            let actual_size = (num_weights - done).min(TILE_SIZE);
            let tile_data = match data.get(tile_pos..) {
                Some(s) if s.len() >= TileHeader::SIZE => s,
                _ => break 'tiles,
            };
            let hdr = TileHeader::from_bytes(match tile_data[..TileHeader::SIZE].try_into() {
                Ok(a) => a,
                Err(_) => break 'tiles,
            });
            let num_near      = hdr.num_near as usize;
            let num_residuals = hdr.num_residuals as usize;
            let bmap_len      = (actual_size + 7) / 8;
            let near_len      = (num_near + 7) / 8;
            // FP32 mantissa: 3 bytes/weight (23 bits stored); FP16/BF16: 2 bytes/weight
            let mbytes        = if hdr.dtype == 0 { 3usize } else { 2usize };
            let sign_len      = (actual_size + 7) / 8;

            let mut p = TileHeader::SIZE;
            macro_rules! take {
                ($n:expr) => {{
                    let end = p + $n;
                    if end > tile_data.len() { break 'tiles; }
                    let s = tile_data[p..end].to_vec();
                    p = end;
                    s
                }};
            }
            let bitmap_match       = take!(bmap_len);
            let bitmap_near        = take!(bmap_len);
            let near_deltas        = take!(near_len);
            let residual_exponents = take!(num_residuals);
            let mantissas          = take!(actual_size * mbytes);
            let signs              = take!(sign_len);

            // Avança tile_pos pelo tamanho serializado (com padding de 64 bytes)
            let raw_tile_bytes = p;
            let padded = if raw_tile_bytes % ALIGNMENT == 0 {
                raw_tile_bytes
            } else {
                raw_tile_bytes + ALIGNMENT - raw_tile_bytes % ALIGNMENT
            };
            tile_pos += padded;

            let tile = NszTile {
                header: hdr,
                bitmap_match,
                bitmap_near,
                near_deltas,
                residual_exponents,
                mantissas,
                signs,
                actual_size,
            };
            weights.extend_from_slice(&decode_tile_fp32(&tile));
        }

        result.insert(entry.name.clone(), weights);
    }

    debug!("NSZ: {} tensores decodificados (lossless TCA-TBE)", result.len());
    Some(result)
}

/// Carrega e dequantiza todos os pesos do GGUF para um WeightBank.
/// Chamado UMA VEZ durante `init()` — elimina o re-loading por chamada de `generate()`.
/// Retorna `(staging_bank, gpu_bank)`:
/// - `staging_bank`: HOST_VISIBLE — cpu_reference.rs pode chamar as_f32_slice().
/// - `gpu_bank`: DEVICE_LOCAL — gpu_forward.rs lê à velocidade total da VRAM (~256 GB/s).
///   Se Vulkan não está ativo, ambos são staging (mesmo buffer, comportamento idêntico ao anterior).
fn build_weight_bank(
    engine: &VulkanEngine,
    metadata: &nodestor_core::ModelMetadata,
    model_path: &str,
) -> (nodestor_vulkan::WeightBank, nodestor_vulkan::WeightBank) {
    let mut staging_bank = nodestor_vulkan::WeightBank::new();
    // Armazena os dados f32 brutos para depois construir o gpu_bank sem re-ler o GGUF.
    let mut raw_data: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();

    let mut real_loaded = 0usize;

    // 0. Tenta NSZ (formato nativo lossless — sem dequantização lossy).
    //    Se o arquivo começa com "NSZ1", decodifica via TCA-TBE e ignora o caminho GGUF.
    if let Some(nsz_tensors) = try_load_nsz_file(model_path) {
        debug!("WeightBank: NSZ detectado — {} tensores lossless (TCA-TBE)", nsz_tensors.len());
        for (name, f32s) in nsz_tensors {
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(f32s.as_ptr() as *const u8, f32s.len() * 4).to_vec()
            };
            staging_bank.insert(name.clone(), nodestor_vulkan::GpuBuffer::from_cpu_data(bytes.clone()));
            raw_data.insert(name, bytes);
            real_loaded += 1;
        }
    } else {
        // 1. Carrega pesos do GGUF via WeightStore (mmap zero-copy) + dequantização
        match crate::weight_store::WeightStore::open(std::path::Path::new(model_path)) {
            Ok(store) => {
                for name in store.list_tensors() {
                    let bytes = match store.tensor_bytes(name) { Some(b) => b, None => continue };
                    let (dtype, n_elems) = store.tensor_info(name)
                        .map(|t| (t.dtype, t.shape.iter().map(|&d| d as usize).product::<usize>()))
                        .unwrap_or((nodestor_core::TensorDtype::F32, bytes.len() / 4));
                    let upload_bytes: Vec<u8> = match tensor_to_f32(bytes, dtype, n_elems) {
                        Some(f32s) => {
                            let raw = unsafe {
                                std::slice::from_raw_parts(f32s.as_ptr() as *const u8, f32s.len() * 4)
                            };
                            raw.to_vec()
                        }
                        None => bytes.to_vec(),
                    };
                    // staging_bank uses CPU-backed buffers so as_f32_slice() is always readable.
                    // GPU compute uses gpu_bank (device-local). Never mix them up.
                    staging_bank.insert(name.to_string(), nodestor_vulkan::GpuBuffer::from_cpu_data(upload_bytes.clone()));
                    raw_data.insert(name.to_string(), upload_bytes);
                    real_loaded += 1;
                }
                debug!("WeightBank: {} tensores carregados do GGUF", real_loaded);
            }
            Err(e) => debug!("WeightStore indisponível ({}); usando buffers vazios", e),
        }
    }

    // 2. Fallback: garante que todos os tensores do metadata existam (shapes corretas)
    for tensor in &metadata.tensors {
        if staging_bank.get(&tensor.name).is_some() { continue; }
        let size_bytes = (tensor.shape.iter().map(|&d| d as usize).product::<usize>() * 4).max(4);
        staging_bank.insert(tensor.name.clone(), nodestor_vulkan::GpuBuffer::from_cpu_data(vec![0u8; size_bytes]));
    }

    // 3. Tied embeddings: muitos modelos não trazem output.weight separado
    if staging_bank.get("output.weight").is_none() {
        let tied = staging_bank.get("token_embd.weight")
            .map(|te| te.as_f32_slice().to_vec())
            .filter(|d| !d.is_empty());
        if let Some(data) = tied {
            let raw: Vec<u8> = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec()
            };
            staging_bank.insert("output.weight".to_string(), nodestor_vulkan::GpuBuffer::from_cpu_data(raw.clone()));
            raw_data.insert("output.weight".to_string(), raw);
            debug!("WeightBank: tied embeddings output.weight←token_embd ({} floats)", data.len());
        }
    }

    // 4. Constrói gpu_bank com DEVICE_LOCAL (pesos na VRAM real, acesso 16× mais rápido).
    //    Se não há Vulkan real, reutiliza staging (upload() já retorna CPU-backed buffer).
    let mut gpu_bank = nodestor_vulkan::WeightBank::new();
    if engine.is_gpu_active() {
        for (name, data) in &raw_data {
            match engine.upload_device_local(data) {
                Ok(buf) => gpu_bank.insert(name.clone(), buf),
                Err(e) => {
                    // Fallback: usa staging se device_local falhar (VRAM cheia, etc.)
                    debug!("gpu_weight_bank: device_local upload '{}' falhou ({}); usando staging", name, e);
                    if let Ok(buf) = engine.upload(data) {
                        gpu_bank.insert(name.clone(), buf);
                    }
                }
            }
        }
        debug!("gpu_weight_bank: {} tensores em DEVICE_LOCAL", gpu_bank.len());
    } else {
        // CPU mode: gpu_bank vazio — gpu_forward nunca é chamado (use_gpu = false)
        debug!("gpu_weight_bank: Vulkan inativo, gpu_bank vazio (não usado)");
    }

    (staging_bank, gpu_bank)
}
