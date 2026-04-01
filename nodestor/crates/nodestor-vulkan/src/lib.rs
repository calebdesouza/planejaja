mod buffer;
mod error;
mod instance;
mod pipeline;
mod shader_loader;
pub mod transformer;
pub mod command_recycler;
pub mod triple_buffer;
pub mod external_memory;

pub use buffer::{GpuBuffer, GpuBufferUsage};
pub use error::VulkanError;
pub use instance::VulkanContext;
pub use pipeline::{ComputePipeline, PipelineKind};
pub use transformer::*;
pub use buffer::MemoryPath;
pub use command_recycler::CommandRecycler;
pub use triple_buffer::TripleBufferPipeline;
pub use external_memory::{try_import_host_memory, is_supported as ext_memory_supported};

use nodestor_core::{GpuCapabilities, HardwareProfile, NodeStorError};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub compute_throughput_tflops: f32,
    pub vram_usage_bytes: u64,
    pub vram_total_bytes: u64,
    pub active_kernels: u32,
}

pub struct VulkanEngine {
    pub ctx: VulkanContext,
    pipelines: std::collections::HashMap<PipelineKind, ComputePipeline>,
}

impl VulkanEngine {
    pub fn new(profile: &HardwareProfile) -> Result<Self, NodeStorError> {
        let gpu = profile.primary_gpu();
        let ctx = VulkanContext::new(gpu).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        let pipelines = pipeline::create_all_pipelines(&ctx).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        Ok(Self { ctx, pipelines })
    }

    /// Descompressão Líquida (Streaming de Micro-Fatias)
    pub fn decompress_liquid(
        &self,
        chunk: &[u8],
        output_elements: usize,
    ) -> Result<(GpuBuffer, ash::vk::Fence), NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::Lossless)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline Lossless not available".into()))?;

        let input_buf = self.ctx.upload_to_gpu(chunk)?;
        let mut output_buf = self.ctx.alloc_gpu_buffer(output_elements * 4)?;

        let fence = pipeline.dispatch_liquid(&self.ctx, &input_buf, &mut output_buf, output_elements as u32)?;
        Ok((output_buf, fence))
    }

    /// Descompressão Massiva GPU (GDeflate Universal)
    /// Recebe um stream GDeflate (com header NodeStor ou puro, ajustado pela engine)
    pub fn decompress_gdeflate(
        &self,
        chunk: &[u8],
        expected_size: usize,
    ) -> Result<(GpuBuffer, ash::vk::Fence), NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::GDeflate)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline GDeflate not available".into()))?;

        // 1. Upload do payload comprimido para a VRAM (DMA Zero-Copy)
        let input_buf = self.ctx.upload_to_gpu(chunk)?;
        // 2. Alocação do buffer de saída completo
        let mut output_buf = self.ctx.alloc_gpu_buffer(expected_size)?;

        // Em uma implementação real do pipeline Nodestor-G com headers de tile:
        // let (tiles, _tile_size, header_len) = nodestor_gdeflate::tile::SerializedGDeflateStream::parse_header(chunk)?;
        // Aqui assumiremos um mock simplificado de 1 bloco unificado para validação da fundação Vulkan
        let tile_offset = 0; // offset após o header
        let compressed_size = chunk.len() as u32;
        let uncompressed_size = expected_size as u32;
        let output_offset = 0;

        // 3. Dispatch do Compute Shader
        let fence = pipeline.dispatch_gdeflate(
            &self.ctx,
            &input_buf,
            &mut output_buf,
            tile_offset,
            compressed_size,
            uncompressed_size,
            output_offset,
        )?;

        Ok((output_buf, fence))
    }


    pub fn matmul(&self, a: &GpuBuffer, b: &GpuBuffer, m: u32, k: u32, n: u32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::Matmul).ok_or_else(|| NodeStorError::VulkanError("Pipeline Matmul not available".into()))?;
        let mut output = self.ctx.alloc_gpu_buffer((m * n * 4) as usize)?;
        pipeline.dispatch_matmul(&self.ctx, a, b, &mut output, m, k, n)?;
        Ok(output)
    }

    pub fn rmsnorm(&self, input: &GpuBuffer, weight: &GpuBuffer, seq_len: u32, hidden_size: u32, eps: f32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::RmsNorm).ok_or_else(|| NodeStorError::VulkanError("RmsNorm missing".into()))?;
        let mut output = self.ctx.alloc_gpu_buffer((seq_len * hidden_size * 4) as usize)?;
        pipeline.dispatch_rmsnorm(&self.ctx, input, weight, &mut output, seq_len, hidden_size, eps)?;
        Ok(output)
    }

    pub fn rope(&self, q: &mut GpuBuffer, k: &mut GpuBuffer, seq_len: u32, num_heads_q: u32, num_heads_k: u32, head_dim: u32, freq_base: f32, start_pos: u32) -> Result<(), NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::RoPe).ok_or_else(|| NodeStorError::VulkanError("RoPE missing".into()))?;
        pipeline.dispatch_rope(&self.ctx, q, k, seq_len, num_heads_q, num_heads_k, head_dim, freq_base, start_pos)
    }

    pub fn silu(&self, input: &GpuBuffer, total_elements: u32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::SiLu).ok_or_else(|| NodeStorError::VulkanError("SiLU missing".into()))?;
        let mut output = self.ctx.alloc_gpu_buffer((total_elements * 4) as usize)?;
        pipeline.dispatch_silu(&self.ctx, input, &mut output, total_elements)?;
        Ok(output)
    }

    pub fn softmax_in_place(&self, buffer: &mut GpuBuffer, seq_len: u32, vocab_size: u32) -> Result<(), NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::Softmax).ok_or_else(|| NodeStorError::VulkanError("Softmax missing".into()))?;
        pipeline.dispatch_softmax(&self.ctx, buffer, seq_len, vocab_size)
    }

    pub fn attention(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer, seq_len: u32, head_dim: u32, scale: f32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::Attention).ok_or_else(|| NodeStorError::VulkanError("Attention missing".into()))?;
        let mut out_attn = self.ctx.alloc_gpu_buffer((seq_len * head_dim * 4) as usize)?;
        pipeline.dispatch_attention(&self.ctx, q, k, v, &mut out_attn, seq_len, head_dim, scale)?;
        Ok(out_attn)
    }

    pub fn alloc_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, NodeStorError> {
        self.ctx.alloc_gpu_buffer(size_bytes)
    }

    /// Aloca buffer Via Expressa (Triple-Path: ReBAR → Pinned DMA → Staging Fallback).
    /// Zero configuração. Detecta o melhor caminho automaticamente.
    pub fn alloc_pinned_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, NodeStorError> {
        GpuBuffer::allocate_pinned(&self.ctx, size_bytes)
    }

    pub fn upload(&self, data: &[u8]) -> Result<GpuBuffer, NodeStorError> {
        self.ctx.upload_to_gpu(data)
    }

    /// Upload via Via Expressa com Write-Combining.
    pub fn upload_pinned(&self, data: &[u8]) -> Result<GpuBuffer, NodeStorError> {
        self.ctx.upload_pinned(data)
    }

    /// `true` se a GPU tem ReBAR ativo (SSD pode escrever direto na VRAM).
    pub fn has_rebar(&self) -> bool { self.ctx.has_rebar }

    /// `true` se há uma fila de transferência dedicada (DMA async sem bloquear compute).
    pub fn has_dedicated_transfer_queue(&self) -> bool { self.ctx.has_dedicated_transfer_queue }

    pub fn download_f32(&self, buffer: &GpuBuffer) -> Result<Vec<f32>, NodeStorError> {
        self.ctx.download_from_gpu(buffer)
    }

    pub fn cosine_similarity_batch(&self, query: &GpuBuffer, candidates: &GpuBuffer, num_candidates: u32, dim: u32) -> Result<Vec<f32>, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::CosineSim).ok_or_else(|| NodeStorError::VulkanError("Pipeline CosineSim not available".into()))?;
        let mut scores_buf = self.ctx.alloc_gpu_buffer((num_candidates * 4) as usize)?;
        pipeline.dispatch_cosine(&self.ctx, query, candidates, &mut scores_buf, num_candidates, dim)?;
        self.ctx.download_from_gpu(&scores_buf)
    }

    pub fn capabilities(&self) -> &GpuCapabilities { self.ctx.capabilities() }
    pub fn device_name(&self) -> &str { self.ctx.device_name() }

    pub fn probe_gpus() -> Vec<GpuCapabilities> {
        match instance::probe_physical_devices() {
            Ok(caps) => {
                info!("Vulkan probe: {} GPU(s) detectada(s)", caps.len());
                caps
            }
            Err(e) => {
                warn!("Vulkan probe falhou ({}), usando heurísticas de SO", e);
                vec![]
            }
        }
    }

    pub fn consume_stats(&self) -> PerformanceStats {
        PerformanceStats {
            compute_throughput_tflops: 0.0,
            vram_usage_bytes: 0,
            vram_total_bytes: self.ctx.capabilities().vram_bytes,
            active_kernels: 0,
        }
    }
}
