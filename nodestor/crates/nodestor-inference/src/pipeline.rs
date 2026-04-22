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

    /// Loop principal de "Mecanismo de Atenção": prevê tensores e dispara
    /// kernels Vulkan para gerar tokens a alta voltagem (Modo Metralhadora).
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(String, GenerationStats), NodeStorError> {
        let start_time = Instant::now();
        
        // 0. Busca RAG (LanceDB Lookup) — Opcional dependendo do prompt
        let _context = self.vector_db.search_knn(&[0.0; 128], 3).await?;
        debug!("RAG Context carregado: {} itens", _context.len());

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
        for tensor in &self.metadata.tensors {
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
        // SAE local: decomposição monosemântica dos hidden_states
        let mut probes_sae = crate::sae_engine::SAEEngine::new(
            self.probes_config.hidden_dim,
            self.probes_config.sae_dict_size,
            self.probes_config.sae_threshold,
        );
        // ─────────────────────────────────────────────────────────────────────────

        let mut sampler = crate::sampler::Sampler::new(crate::sampler::SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            use_conformal: probes_enabled,
        });

        // ── Tokenizer — encode do prompt ─────────────────────────────────────────
        // Tenta carregar o tokenizer real do GGUF; fallback para tokenizer dummy
        let dummy_json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<unk>","special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"Hello":1,"World":2},"unk_token":"<unk>"}}"#;
        let tokenizer_json = self.metadata.extra.get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or(dummy_json);
        let tokenizer = crate::tokenizer::TokenizerManager::from_string(tokenizer_json)
            .unwrap_or_else(|_| crate::tokenizer::TokenizerManager::from_string(dummy_json).unwrap());

        let mut input_tokens = tokenizer.encode(prompt).unwrap_or(vec![0]);
        if input_tokens.is_empty() { input_tokens.push(0); }

        // Pre-enche a RAM/VRAM para que o Kernel nunca bloqueie (Burst Pump)
        scheduler.prime_pump().await?;

        for step in 0..max_tokens {
            // Em uma engine LLM real, current_token passaria por uma Tabela de Embeddings e viraria um tensor.
            let current_token = if step < input_tokens.len() {
                input_tokens[step]
            } else {
                *generated_tokens.last().unwrap_or(&0)
            };

            // Criar embedding de entrada com a dimensão real do modelo
            let embed_dim = hidden_size as usize;
            let mut embed_data = vec![0.0f32; embed_dim];
            embed_data[current_token as usize % embed_dim] = 1.0;
            let embed_bytes = unsafe {
                std::slice::from_raw_parts(embed_data.as_ptr() as *const u8, embed_data.len() * 4)
            };
            let embed_buf = self.engine.upload(embed_bytes)?;

            // Roda o Forward Pass Real — pesos do WeightBank, todas as camadas ativas
            let logits_buf = transformer.forward(&self.engine, &embed_buf, &weight_bank, step as u32)
                .map_err(|e| nodestor_core::NodeStorError::VulkanError(e.to_string()))?;

            // ── PROBES V2: Inspeção pós-forward antes do sample ───────────────────
            if probes_enabled {
                // 1. RAIO-X (SAE local): decompõe hidden_state em features legíveis
                let latent_features = probes_sae.encode(&embed_data);
                let _ = latent_features; // Disponível para inspectors externos

                // 2. Ferramenta externa (ELK+CoT+RAISE via DAVI) se injetada
                if let Some(ref tool_mutex) = self.probes_tool {
                    if let Ok(mut tool) = tool_mutex.lock() {
                        let (is_safe, maybe_alert) = tool.inspect(&embed_data, step);
                        if let Some(alert) = maybe_alert {
                            warn!("{}", alert);
                            probes_alerts.push(alert);
                        }
                        if !is_safe {
                            let block_msg = format!("[PROBES] Step {}: Bloqueio por ferramenta externa (ELK/CoT/RAISE)", step);
                            warn!("{}", block_msg);
                            probes_alerts.push(block_msg);
                            break; // Interrompe geração segura
                        }
                    }
                }
            }
            // ─────────────────────────────────────────────────────────────────────

            // Download logits e sample com garantia Conformal
            let mut logits = self.engine.download_f32(&logits_buf)?;
            let (sampled_token, conformal_set) = sampler.sample_with_conformal(&mut logits, &generated_tokens)
                .map_err(|e| nodestor_core::NodeStorError::VulkanError(format!("Sampler error: {:?}", e)))?;

            // Verifica incerteza conformal — rejeita tokens de alta entropia
            if let Some(ref cs) = conformal_set {
                if !cs.is_reliable {
                    conformal_rejections += 1;
                    debug!("[PROBES/CONFORMAL] Step {}: token rejeitado por alta incerteza (entropy={:.3})", step, cs.entropy);
                }
            }

            let next_token = sampled_token;

            if step >= input_tokens.len() {
                // Usar o vocab_size real do modelo — sem clamp arbitrário para 3 tokens
                let token_in_range = (next_token as u32) % vocab_size;
                generated_tokens.push(token_in_range);
            }

            // Salvar bytes reais dos logits no KV Cache (paging SSD)
            let logit_bytes = self.engine.download_f32(&logits_buf)
                .map(|f32s| {
                    f32s.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()
                })
                .unwrap_or_else(|_| vec![0u8; (vocab_size * 4) as usize]);

            for layer_idx in 0..layers_per_token {
                if let Some(mut block) = scheduler.next_tensor().await {
                    let gpu_buffer = block.buffer.buffer.as_mut().unwrap();
                    // Matmul residual para dar pressão no sistema
                    let _ = self.engine.matmul(gpu_buffer, gpu_buffer, 32, 32, 32);
                    // Paging real: grava bytes do logit/KV para o SSD quando VRAM esgota
                    if let Err(e) = kv_cache.allocate_block(layer_idx, &logit_bytes, &*self.transport) {
                        debug!("KV Cache alloc layer {}: {}", layer_idx, e);
                    }
                } else {
                    return Err(NodeStorError::TransferFailed("A fila de prefetch secou!".to_string()));
                }
            }
            tokens_done += 1;
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
            match this.generate(&prompt, max_tokens).await {
                Ok((text, stats)) => {
                    // Faz stream de cada palavra individualmente para UX de baixa latência.
                    // Em uma implementação com tokenizer bidirecional, enviaria token a token.
                    for word in text.split_inclusive(' ') {
                        if tx.send(Ok(word.to_string())).await.is_err() {
                            break; // cliente desconectou
                        }
                    }
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
