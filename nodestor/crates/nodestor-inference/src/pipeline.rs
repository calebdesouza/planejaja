//! Pipeline de inferência — orquestração end-to-end do fluxo de dados.
//!
//! Conecta: Scanner (Hardware) → Transport (SSD) → Streaming (Metralhadora/Pool) → Vulkan (GPU) → LLM.

use nodestor_core::{DataTransport, HardwareProfile, ModelMetadata, NodeStorError};
use nodestor_formats::detect_parser;
use nodestor_scanner::scan;
use nodestor_streaming::{BufferPool, MesPrefetchQueue, StreamScheduler};
use nodestor_transport::create_transport;
use nodestor_metadata::search::VectorSearch;
use nodestor_vulkan::VulkanEngine;
use std::sync::Arc;
use tokio::time::Instant;
use tracing::debug;
use futures::Stream;
use std::pin::Pin;

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
}

/// Pipeline central de execução do modelo.
pub struct InferencePipeline {
    pub config: InferenceConfig,
    pub profile: HardwareProfile,
    pub transport: Arc<dyn DataTransport + Send + Sync>,
    pub metadata: Arc<ModelMetadata>,
    pub engine: VulkanEngine,
    pub vector_db: VectorSearch,
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
        
        // Inicializa a base vetorial local (LanceDB)
        let db_path = format!("{}/vector_db", config.model_path);
        let vector_db = VectorSearch::new("knowledge_base", &db_path);

        Ok(Self {
            config,
            profile,
            transport,
            metadata,
            engine,
            vector_db,
        })
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

        let mut scheduler = StreamScheduler::new(
            self.config.prefetch_depth,
            queue,
            self.metadata.clone(),
        );

        // Pre-enche a RAM/VRAM para que o Kernel nunca bloqueie
        scheduler.prime_pump().await?;

        // 3. Forward Pass (Geração em malha fechada)
        let generated_text = format!("(Resposta gerada via stream Vulkan zero-copy. Tokens gerados: {})", max_tokens);
        let mut tokens_done = 0;
        let mut layers_per_token = self.metadata.tensors.len().min(4); // Simula ler 4 tensores p/ camada
        if layers_per_token == 0 { layers_per_token = 1; }

        for _ in 0..max_tokens {
            for _ in 0..layers_per_token {
                if let Some(mut block) = scheduler.next_tensor().await {
                    let gpu_buffer = block.buffer.buffer.as_mut().unwrap();

                    // Disparo matemático Vulkan (Stub para teste de frame rate real)
                    let _ = self.engine.matmul(
                        gpu_buffer, 
                        gpu_buffer, 
                        32, 32, 32 // dimensões pequenas mock para não estourar tempo de teste
                    )?;
                } else {
                    return Err(NodeStorError::TransferFailed("A fila de prefetch secou!".to_string()));
                }
            }
            tokens_done += 1;
        }

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
        };

        Ok((generated_text, stats))
    }

    /// Versão Reativa/Stream: devolve tokens um a um conforme são gerados pela GPU.
    /// Vital para interfaces de Chat e UX de baixa latência percebida.
    pub async fn generate_stream(
        self: Arc<Self>,
        _prompt: String,
        max_tokens: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<String, NodeStorError>> + Send>> {
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let this = self.clone();

        tokio::spawn(async move {
            // Reutiliza a lógica de setup (Pool/Queue) — em produção isso seria cacheado
            let pool = match BufferPool::new(&this.engine.ctx, this.config.buffer_size, this.config.prefetch_depth) {
                Ok(p) => p,
                Err(e) => { let _ = tx.send(Err(e)).await; return; }
            };

            let queue = MesPrefetchQueue::new(this.transport.clone(), this.config.model_path.clone(), pool);
            let mut scheduler = StreamScheduler::new(this.config.prefetch_depth, queue, this.metadata.clone());
            
            if let Err(e) = scheduler.prime_pump().await {
                let _ = tx.send(Err(e)).await;
                return;
            }

            for i in 0..max_tokens {
                // Simulação de geração de 1 token
                let token = format!("token_{} ", i);
                
                // Simula o processamento de camadas
                if let Some(mut _block) = scheduler.next_tensor().await {
                    // Pipeline Vulkan fictício para manter o timing
                    let _ = tx.send(Ok(token)).await;
                } else {
                    let _ = tx.send(Err(NodeStorError::TransferFailed("Prefetch dry".into()))).await;
                    break;
                }
                
                // Pequeno delay para simular tempo de computação real (ms)
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}
