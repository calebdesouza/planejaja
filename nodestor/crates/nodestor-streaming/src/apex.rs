//! ApexOrchestrator — O Cérebro Inteligente do NodeStor APEX v2.
//!
//! Unifica TODOS os subsistemas de transporte e streaming num único orquestrador
//! que toma decisões em tempo real:
//!
//! - **Direct I/O**: Lê do SSD sem passar pelo Page Cache do OS
//! - **Triple-Path Allocator**: ReBAR → Pinned DMA → Staging, automático
//! - **Triple Buffer Pipeline**: 3 baldes rotativos garantem que GPU nunca param
//! - **STS (Speculative Tensor Streaming)**: Pré-carrega tensores futuros
//! - **KV Cache Paging**: Contexto infinito via SSD como extensão da VRAM
//!
//! ## A Inteligência Central: Decision Engine
//!
//! Para cada tensor requisitado:
//! 1. Decide a rota: DirectToVram | PinnedPipeline | StagingFallback | UnifiedMemory
//! 2. Direct I/O: SSD → Pinned Memory (bypass do Page Cache do OS)
//! 3. Triple Buffer: rotaciona os 3 baldes sem bloquear a GPU
//! 4. Dispara prefetch especulativo para as próximas N camadas (STS)

use nodestor_core::{ModelMetadata, NodeStorError};
use nodestor_transport::{DirectIOReader, PlatformIOCapabilities};
use nodestor_vulkan::{TripleBufferPipeline, MemoryPath};
use crate::speculative::SpeculativeCache;
use crate::burst_reader::BurstReader;
use crate::layer_graph::{ExecutionPlan, GroupType, LayerGraph, TensorGroup};
use tracing::{info, debug};
use std::collections::VecDeque;

/// Tarefa de background para Ambient AI.
pub trait AmbientTask: Send + Sync {
    fn execute(&mut self);
}

/// Detector de Ociosidade para despachar AmbientTasks.
pub struct IdleDetector {
    pub last_inference_fence: Option<ash::vk::Fence>,
    pub ambient_tasks: VecDeque<Box<dyn AmbientTask>>,
    pub idle_threshold_ms: u64,
    pub last_activity: std::time::Instant,
}

impl IdleDetector {
    pub fn new() -> Self {
        Self {
            last_inference_fence: None,
            ambient_tasks: VecDeque::new(),
            idle_threshold_ms: 50, // 50ms de inatividade = GPU ociosa
            last_activity: std::time::Instant::now(),
        }
    }

    pub fn add_task(&mut self, task: Box<dyn AmbientTask>) {
        self.ambient_tasks.push_back(task);
    }

    pub fn tick(&mut self, apex: &ApexOrchestrator) {
        if self.ambient_tasks.is_empty() { return; }

        if let Some(_fence) = self.last_inference_fence {
            // Emulação de status do fence atrelada aos pipelines do APEX
            // Em prod real: device.get_fence_status(_fence)
            let is_idle = !apex.pipeline.is_active() || self.last_activity.elapsed().as_millis() as u64 >= self.idle_threshold_ms;
            
            if is_idle {
                if let Some(mut task) = self.ambient_tasks.pop_front() {
                    task.execute();
                }
            }
        } else {
            // Se não há fence (simulação), verifica tempo decorrido
            if self.last_activity.elapsed().as_millis() as u64 >= self.idle_threshold_ms {
                if let Some(mut task) = self.ambient_tasks.pop_front() {
                    task.execute();
                }
            }
        }
    }
    
    pub fn mark_activity(&mut self) {
        self.last_activity = std::time::Instant::now();
    }
}


/// Rota de transporte selecionada pelo Decision Engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportRoute {
    /// ReBAR ativo: SSD escreve diretamente na VRAM via PCIe. (~14 GB/s)
    DirectToVram,
    /// Pinned DMA: SSD → RAM travada → GPU puxa via PCIe DMA. (~6-8 GB/s)
    PinnedPipeline,
    /// Staging clássico. Fallback para GPUs integradas. (~3-5 GB/s)
    StagingFallback,
    /// Unified Memory (Apple Silicon): SSD → RAM = SSD → GPU. (sem PCIe)
    UnifiedMemory,
}

/// Estatísticas de telemetria adaptativas do APEX.
#[derive(Debug, Default)]
pub struct ApexStats {
    pub tensors_loaded: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub bytes_transferred: u64,
    pub rebar_transfers: u64,
    pub pinned_transfers: u64,
    pub staging_transfers: u64,
}

impl ApexStats {
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 * 100.0 }
    }

    pub fn report(&self) -> String {
        format!(
            "Tensores: {} | Cache Hit: {:.1}% | Bytes: {:.2} GB | ReBAR={} Pinned={} Staging={}",
            self.tensors_loaded,
            self.cache_hit_rate(),
            self.bytes_transferred as f64 / 1e9,
            self.rebar_transfers,
            self.pinned_transfers,
            self.staging_transfers,
        )
    }
}

/// O orquestrador central do NodeStor APEX v2.
pub struct ApexOrchestrator {
    pub pipeline: TripleBufferPipeline,
    reader: DirectIOReader,
    pub platform: PlatformIOCapabilities,
    pub speculative_cache: SpeculativeCache,
    pub burst_reader: Option<BurstReader>,
    pub plan: Option<ExecutionPlan>,
    pub stats: ApexStats,
    pub idle_detector: IdleDetector,
}

impl ApexOrchestrator {
    /// Cria o ApexOrchestrator base (compatibilidade reversa). Zero-Config.
    pub fn new(model_path: &str, bucket_size: usize) -> Result<Self, NodeStorError> {
        let platform = PlatformIOCapabilities::detect();
        let reader = DirectIOReader::open(model_path)?;
        let pipeline = TripleBufferPipeline::simulation(bucket_size);
        let speculative_cache = SpeculativeCache::new(8);
        let burst_reader = Some(BurstReader::new(model_path));

        info!(
            "ApexOrchestrator: Inicializado | Direct I/O: {} ({}) | Plataforma: {}",
            reader.is_direct(),
            platform.bypass_strategy,
            platform.platform_name,
        );

        Ok(Self { 
            pipeline, 
            reader, 
            platform, 
            speculative_cache, 
            burst_reader,
            plan: None,
            stats: ApexStats::default(),
            idle_detector: IdleDetector::new(),
        })
    }

    /// Cria ApexOrchestrator com inteligência topológica (LayerGraph + STS Burst) e Decision Engine
    pub fn from_model(model_path: &str, metadata: &ModelMetadata) -> Result<Self, NodeStorError> {
        let plan = LayerGraph::build(metadata);
        let platform = PlatformIOCapabilities::detect();
        let reader = DirectIOReader::open(model_path)?;
        let burst_reader = Some(BurstReader::new(model_path));
        
        let max_group_bytes = plan.groups.iter()
            .map(|g| g.total_bytes)
            .max()
            .unwrap_or(64 * 1024 * 1024) as usize;
            
        // Ajustamos pipeline pra aguentar até mesmo o grupo com a maior exigência de VRAM de uma vez 
        let pipeline = TripleBufferPipeline::simulation(max_group_bytes);
        
        // Cache especulativo limitará RAM ou memória virtual a um orçamento dinâmico tolerável
        // 512MB fallback em fallback para sistemas minúsculos
        let cache_budget = (max_group_bytes * plan.num_layers.max(1)) / 4; 
        let cache_budget = cache_budget.max(512 * 1024 * 1024);
        
        let max_cached_groups = (cache_budget / max_group_bytes.max(1)).max(2).min(8);
        
        let speculative_cache = SpeculativeCache::new(max_cached_groups)
                                .with_bytes_limit(cache_budget);

        Ok(Self {
            pipeline,
            reader,
            platform,
            speculative_cache,
            burst_reader,
            plan: Some(plan),
            stats: ApexStats::default(),
            idle_detector: IdleDetector::new(),
        })
    }

    // ─── Decision Engine ────────────────────────────────────────────────────

    fn decide_route(&self) -> TransportRoute {
        if self.platform.unified_memory { return TransportRoute::UnifiedMemory; }
        match self.pipeline.memory_path() {
            MemoryPath::RebarDirect  => TransportRoute::DirectToVram,
            MemoryPath::PinnedHostDma | MemoryPath::Simulation => TransportRoute::PinnedPipeline,
            _ => TransportRoute::StagingFallback,
        }
    }

    // ─── API Pública ─────────────────────────────────────────────────────────

    /// Carrega um tensor do SSD com throughput máximo.
    ///
    /// Usa o Cache Especulativo (STS) como prioridade 1 (Zero-I/O Hit).
    /// Se não houver, fallback pro `TripleBufferPipeline::read_via_sim` sem borrow conflict.
    pub fn load_tensor(
        &mut self,
        name: &str,
        offset: u64,
        size: usize,
    ) -> Result<Vec<u8>, NodeStorError> {
        debug!("APEX: load_tensor '{}' offset={} size={}", name, offset, size);

        // 1. SpeculativeCache Hit? = I/O Instantâneo da VRAM/RAM
        if let Some(cached) = self.speculative_cache.try_get_by_name(name) {
            self.stats.tensors_loaded += 1;
            self.stats.cache_hits += 1;
            return Ok(cached);
        }

        // 2. Fallback: Cold Load por I/O 
        let route = self.decide_route();
        let reader = &self.reader;

        let data = self.pipeline.read_via_sim(|buf| {
            reader.read_at(offset, size, buf)
        })?;

        let read_bytes = data.len();

        // Telemetria
        self.stats.tensors_loaded += 1;
        self.stats.cache_misses += 1;
        self.stats.bytes_transferred += read_bytes as u64;
        match route {
            TransportRoute::DirectToVram | TransportRoute::UnifiedMemory => self.stats.rebar_transfers += 1,
            TransportRoute::PinnedPipeline => self.stats.pinned_transfers += 1,
            TransportRoute::StagingFallback => self.stats.staging_transfers += 1,
        }

        debug!("APEX: '{}' carregado via {:?} ({} bytes)", name, route, read_bytes);
        Ok(data)
    }
    
    /// Novo Método STS Burst: Carrega Grupo Inteiro usando paralelismo multi-thread do SSD NVMe
    pub fn load_group_burst(&mut self, group: &TensorGroup) -> Result<Vec<Vec<u8>>, NodeStorError> {
        // Se há apenas 1 tensor, o burst é um overhead inútil para o SO. Usamos carga single-thread.
        if group.tensors.len() <= 1 {
            let mut results = Vec::new();
            for slice in &group.tensors {
                results.push(self.load_tensor(&slice.name, slice.offset, slice.size as usize)?);
            }
            return Ok(results);
        }

        let route = self.decide_route();
        let slices: Vec<(u64, usize)> = group.tensors.iter().map(|s| (s.offset, s.size as usize)).collect();
        let mut vecs: Vec<Vec<u8>> = slices.iter().map(|s| vec![0u8; s.1]).collect();
        
        let mut bufs: Vec<&mut [u8]> = vecs.iter_mut().map(|v| v.as_mut_slice()).collect();
        
        if let Some(burst) = &self.burst_reader {
            let read_sizes = burst.read_burst(&slices, &mut bufs)?;
            
            // Recorta e guarda metadados
            let mut total_bytes = 0;
            for (idx, size) in read_sizes.iter().enumerate() {
                vecs[idx].truncate(*size);
                total_bytes += size;
                // Popula o spec cache para a geração assíncrona poder usar
                let name = &group.tensors[idx].name;
                self.speculative_cache.insert_by_name(name, vecs[idx].clone());
            }

            self.stats.tensors_loaded += group.tensors.len() as u64;
            self.stats.cache_misses += group.tensors.len() as u64;
            self.stats.bytes_transferred += total_bytes as u64;
            
            match route {
                TransportRoute::DirectToVram | TransportRoute::UnifiedMemory => self.stats.rebar_transfers += 1,
                TransportRoute::PinnedPipeline => self.stats.pinned_transfers += 1,
                TransportRoute::StagingFallback => self.stats.staging_transfers += 1,
            }
            
            return Ok(vecs);
        }
        
        // Timeout de burst configurado inexistente, cai para linear
        let mut results = Vec::new();
        for slice in &group.tensors {
             results.push(self.load_tensor(&slice.name, slice.offset, slice.size as usize)?);
        }
        Ok(results)
    }

    /// Streaming de modelo completo: SSD → memória com throughput máximo.
    pub fn stream_model(
        &mut self,
        total_size: u64,
        chunk_size: usize,
        mut on_chunk: impl FnMut(usize, &[u8]) -> Result<(), NodeStorError>,
    ) -> Result<ApexStreamStats, NodeStorError> {
        let start = std::time::Instant::now();
        let num_chunks = ((total_size + chunk_size as u64 - 1) / chunk_size as u64) as usize;
        let mut total_bytes = 0usize;

        info!(
            "APEX: stream_model {} chunks | Direct I/O: {} | {}",
            num_chunks, self.reader.is_direct(), self.platform.platform_name,
        );

        let reader = &self.reader;

        for i in 0..num_chunks {
            let offset = (i * chunk_size) as u64;
            let remaining = (total_size - offset).min(chunk_size as u64) as usize;

            // read_via_sim: zero conflitos de borrow — closure pega apenas &reader
            let chunk_data = self.pipeline.read_via_sim(|buf| {
                reader.read_at(offset, remaining, buf)
            })?;

            on_chunk(i, &chunk_data)?;
            total_bytes += chunk_data.len();
        }

        self.pipeline.drain_sim();

        let elapsed = start.elapsed().as_secs_f64();
        let throughput_gbps = (total_bytes as f64 / 1e9) / elapsed;
        let route = self.decide_route();

        self.stats.bytes_transferred += total_bytes as u64;

        info!(
            "APEX: stream_model concluído | {:.2} GB/s | {:?} | Direct I/O: {}",
            throughput_gbps, route, self.reader.is_direct()
        );

        Ok(ApexStreamStats {
            bytes_transferred: total_bytes,
            duration_secs: elapsed,
            throughput_gbps,
            route,
            direct_io_active: self.reader.is_direct(),
            platform_name: self.platform.platform_name,
            bypass_strategy: self.platform.bypass_strategy,
        })
    }

    /// Relatório de performance para diagnóstico.
    pub fn report(&self) -> String {
        format!(
            "NodeStor APEX v2 | {} | Direct I/O: {} | {}",
            self.platform.platform_name,
            if self.reader.is_direct() {
                format!("✅ {}", self.platform.bypass_strategy)
            } else {
                "⚠️ Fallback".to_string()
            },
            self.stats.report(),
        )
    }

    pub fn is_direct_io_active(&self) -> bool { self.reader.is_direct() }
    pub fn platform(&self) -> &PlatformIOCapabilities { &self.platform }

    // ─── One-Token-Lag Prefetch (Pesquisa: Roteamento PCIe Sublinear) ────────
    //
    // Conceito: O resultado do roteamento MoE no token `t` nos diz qual expert
    // será ativado no token `t+1`. Enquanto a GPU processa o expert atual,
    // este método dispara a leitura do expert seguinte em background via
    // DirectIOReader. Quando o token `t+1` chegar, o expert já estará no
    // SpeculativeCache → cache hit → ZERO I/O.
    //
    // Isso elimina o "Paradoxo da Densidade Temporal": em vez de page faults
    // cascateados, o SSD faz prefetch com 1 token de antecedência.

    /// Dispara prefetch assíncrono do bloco FFN de um expert MoE.
    ///
    /// `layer_idx`: índice da camada MoE (0-based)
    /// `expert_idx`: índice do expert vencedor (resultado do routing do token atual)
    /// `expert_block_size`: tamanho em bytes do bloco FFN de cada expert
    /// `experts_base_offset`: offset base no arquivo onde os experts começam
    ///
    /// O dado é inserido no `SpeculativeCache` com chave `"moe.{layer}.expert.{idx}"`.
    /// No próximo token, `load_tensor()` dará cache hit instantâneo.
    pub fn prefetch_expert_async(
        &mut self,
        layer_idx: usize,
        expert_idx: u32,
        expert_block_size: usize,
        experts_base_offset: u64,
    ) {
        let cache_key = format!("moe.{}.expert.{}", layer_idx, expert_idx);

        // Se já está no cache, não faz nada (evita leitura duplicada)
        if self.speculative_cache.try_get_by_name(&cache_key).is_some() {
            tracing::trace!(
                "APEX Prefetch: '{}' já no cache — skip",
                cache_key
            );
            return;
        }

        // Calcula o offset absoluto do expert no arquivo
        let expert_offset = experts_base_offset
            + (expert_idx as u64) * (expert_block_size as u64);

        // Lê via DirectIOReader (bypass do page cache do OS)
        // Em produção com io_uring, isso seria uma submissão non-blocking.
        // No modo atual, fazemos a leitura síncrona mas com Direct I/O,
        // que é suficiente para não bloquear graças ao pipeline triple-buffer.
        let data = self.pipeline.read_via_sim(|buf| {
            self.reader.read_at(expert_offset, expert_block_size, buf)
        });

        match data {
            Ok(bytes) => {
                tracing::debug!(
                    "APEX Prefetch: '{}' carregado ({} bytes, offset={})",
                    cache_key, bytes.len(), expert_offset
                );
                self.speculative_cache.insert_by_name(&cache_key, bytes);
                self.stats.tensors_loaded += 1;
                // Conta como cache miss agora, mas será cache hit no próximo token
            }
            Err(e) => {
                tracing::warn!(
                    "APEX Prefetch: falha ao carregar '{}': {}",
                    cache_key, e
                );
                // Falha silenciosa — o sistema continua sem prefetch,
                // caindo no load_tensor normal (mais lento, mas correto).
            }
        }
    }

    /// Versão inferida: calcula expert_block_size e base_offset a partir do ExecutionPlan.
    /// Usa quando o plano topológico está disponível.
    pub fn prefetch_expert_from_plan(
        &mut self,
        layer_idx: usize,
        expert_idx: u32,
    ) {
        if let Some(ref plan) = self.plan {
            // Estima o tamanho do bloco FFN de um expert a partir dos grupos MLP do plano.
            // Cada grupo MLP contém gate+up+down; dividido por num_experts dá o bloco.
            let ffn_group_size: u64 = plan.groups.iter()
                .filter(|g| g.group_type == GroupType::Mlp)
                .map(|g| g.total_bytes)
                .next()
                .unwrap_or(0);

            if ffn_group_size == 0 {
                return; // Sem info de MLP no plano — skip
            }

            // Base offset: usamos o data_offset do primeiro tensor MLP
            let base_offset: u64 = plan.groups.iter()
                .filter(|g| g.group_type == GroupType::Mlp)
                .flat_map(|g| g.tensors.iter())
                .map(|t| t.offset)
                .next()
                .unwrap_or(0);

            // Assume divisão uniforme entre experts (Mixtral = 8, DeepSeek = 256)
            // Para detecção real, o num_experts viria dos metadados GGUF.
            let num_experts = 8u64; // Fallback conservador: Mixtral

            let expert_size = (ffn_group_size / num_experts) as usize;

            self.prefetch_expert_async(layer_idx, expert_idx, expert_size, base_offset);
        }
    }
}

/// Resultado de um streaming completo via APEX.
#[derive(Debug, Clone)]
pub struct ApexStreamStats {
    pub bytes_transferred: usize,
    pub duration_secs: f64,
    pub throughput_gbps: f64,
    pub route: TransportRoute,
    pub direct_io_active: bool,
    pub platform_name: &'static str,
    pub bypass_strategy: &'static str,
}

impl ApexStreamStats {
    pub fn multiplier_vs(&self, baseline_gbps: f64) -> f64 {
        if baseline_gbps <= 0.0 { 1.0 } else { self.throughput_gbps / baseline_gbps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use nodestor_core::{ModelFormat, TensorInfo, TensorDtype};

    fn create_test_file(size: usize) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("apex_test.bin");
        std::fs::File::create(&path).unwrap().write_all(&vec![0xABu8; size]).unwrap();
        (dir, path.to_str().unwrap().to_string())
    }

    fn tensor_info(name: &str, offset: u64, size: u64) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            shape: vec![],
            dtype: TensorDtype::F32,
            data_offset: offset,
            data_size: size,
        }
    }

    fn make_metadata(tensors: Vec<TensorInfo>) -> ModelMetadata {
        ModelMetadata {
            format: ModelFormat::Gguf,
            model_name: None,
            architecture: None,
            param_count: None,
            tensors,
            data_offset: 0,
            file_size: 0,
            extra: serde_json::Value::Null,
        }
    }

    #[test]
    fn test_apex_decide_route_returns_valid() {
        let (_dir, path) = create_test_file(8192);
        let orch = ApexOrchestrator::new(&path, 4096).unwrap();
        assert!(matches!(orch.decide_route(),
            TransportRoute::DirectToVram | TransportRoute::PinnedPipeline |
            TransportRoute::StagingFallback | TransportRoute::UnifiedMemory
        ));
    }

    #[test]
    fn test_apex_load_tensor_reads_correct_bytes() {
        let (_dir, path) = create_test_file(8192);
        let mut orch = ApexOrchestrator::new(&path, 8192).unwrap();
        let data = orch.load_tensor("test.weight", 0, 1024).unwrap();
        assert_eq!(data.len(), 1024);
        assert!(data.iter().all(|&b| b == 0xAB));
        
        // Agora load cached - load_tensor doesnt cache automatically. Check stats.
        let _data2 = orch.load_tensor("test.weight", 0, 1024).unwrap();
        assert_eq!(orch.stats.cache_misses, 2);
    }

    #[test]
    fn test_apex_from_model_builds_plan() {
        let (_dir, path) = create_test_file(8192);
        let metadata = make_metadata(vec![
            tensor_info("blk.0.attn_k.weight", 0, 4096),
            tensor_info("blk.0.attn_v.weight", 4096, 4096),
        ]);
        let orch = ApexOrchestrator::from_model(&path, &metadata).unwrap();
        assert!(orch.plan.is_some());
        assert_eq!(orch.plan.as_ref().unwrap().groups.len(), 1); // Group Attention apenas
    }

    #[test]
    fn test_apex_cache_hit_eliminates_io() {
        let (_dir, path) = create_test_file(8192);
        let mut orch = ApexOrchestrator::new(&path, 4096).unwrap();
        
        // Injeta manualmente
        orch.speculative_cache.insert_by_name("in_memory.weight", vec![0xCC; 128]);
        
        let data = orch.load_tensor("in_memory.weight", 0, 128).unwrap();
        assert_eq!(data[0], 0xCC);
        assert_eq!(orch.stats.cache_hits, 1);
        assert_eq!(orch.stats.cache_misses, 0); // Zero I/O
    }
}
