mod buffer;
mod error;
mod instance;
mod pipeline;
mod shader_loader;

pub use buffer::{GpuBuffer, GpuBufferUsage};
pub use error::VulkanError;
pub use instance::VulkanContext;
pub use pipeline::{ComputePipeline, PipelineKind};

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
