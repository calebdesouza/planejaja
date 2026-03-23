//! nodestor-vulkan — Motor de Compute Vulkan para NodeStor.
//!
//! Esta crate implementa:
//! - Inicialização Vulkan headless (sem janela) em qualquer plataforma
//! - Seleção do melhor dispositivo físico disponível
//! - Alocação e gerenciamento de GPU buffers via `gpu-allocator`
//! - Compute pipelines para: dequantização de tensores, matmul, similaridade cosseno
//! - Shaders SPIR-V embutidos no binário (zero dependências externas em runtime)
//!
//! ## Compatibilidade
//! - **NVIDIA** (Windows/Linux): driver vulkan próprio
//! - **AMD** (Windows/Linux): AMDVLK / Mesa RADV
//! - **Intel** (Windows/Linux): driver Intel
//! - **Apple Silicon / Intel Mac**: MoltenVK (Vulkan → Metal)
//!
//! ## Inicialização (2 passes — resolve o chicken-and-egg com o Scanner)
//! ```rust,ignore
//! // Passe 1: Scanner usa probe mínima para detectar capabilities
//! let caps = VulkanProbe::enumerate_devices()?;
//!
//! // Passe 2: Engine cria logical device + pipelines com as features corretas
//! let engine = VulkanEngine::new_from_profile(&hardware_profile)?;
//! ```

mod buffer;
mod error;
mod instance;
mod pipeline;
mod shader_loader;

pub use buffer::{GpuBuffer, GpuBufferUsage};
pub use error::VulkanError;
pub use instance::VulkanContext;
pub use pipeline::{ComputePipeline, PipelineKind};

use nodestor_core::{GpuCapabilities, GpuVendor, HardwareProfile, NodeStorError, TensorDtype};
use tracing::{debug, info, warn};

/// Motor Vulkan completo: gerencia context, pipelines e operações de compute.
pub struct VulkanEngine {
    pub ctx: VulkanContext,
    pipelines: std::collections::HashMap<PipelineKind, ComputePipeline>,
}

impl VulkanEngine {
    /// Cria o motor Vulkan para o perfil de hardware fornecido.
    ///
    /// Seleciona automaticamente o melhor dispositivo físico disponível e
    /// ativa as extensões corretas baseado nas capabilities detectadas.
    pub fn new(profile: &HardwareProfile) -> Result<Self, NodeStorError> {
        let gpu = profile.primary_gpu();

        let device_name = gpu
            .map(|g| g.device_name.as_str())
            .unwrap_or("desconhecido");

        info!("Inicializando Vulkan Engine para: {}", device_name);

        let ctx = VulkanContext::new(gpu).map_err(|e| {
            NodeStorError::VulkanError(format!("Falha ao criar contexto Vulkan: {e}"))
        })?;

        let supported = ctx.capabilities();
        debug!(
            "Vulkan: cooperative_matrix2={} bfloat16={}",
            supported.supports_cooperative_matrix2, supported.supports_bfloat16
        );

        let pipelines = pipeline::create_all_pipelines(&ctx).map_err(|e| {
            NodeStorError::VulkanError(format!("Falha ao criar compute pipelines: {e}"))
        })?;

        info!(
            "Vulkan Engine pronto — {} pipelines carregados",
            pipelines.len()
        );

        Ok(Self { ctx, pipelines })
    }

    /// Dequantiza tensor comprimido diretamente na GPU.
    ///
    /// Os dados entram comprimidos (Q4_0, Q8_0, etc.) e saem como F16
    /// prontos para cálculos de inferência. A CPU nunca processa os pesos.
    pub fn dequantize(
        &self,
        compressed: &[u8],
        dtype: TensorDtype,
        output_elements: usize,
    ) -> Result<GpuBuffer, NodeStorError> {
        let kind = match dtype {
            TensorDtype::Q4_0 | TensorDtype::Q4_1 => PipelineKind::DequantQ4,
            TensorDtype::Q8_0 => PipelineKind::DequantQ8,
            // F16/BF16/F32 não precisam de dequantização — copia direta
            _ => return self.upload_raw(compressed),
        };

        let pipeline = self.pipelines.get(&kind).ok_or_else(|| {
            NodeStorError::VulkanError(format!("Pipeline {:?} não disponível", kind))
        })?;

        let input_buf = self.ctx.upload_to_gpu(compressed)?;
        let mut output_buf = self.ctx.alloc_gpu_buffer(output_elements * 2)?; // F16 = 2 bytes

        pipeline.dispatch(&self.ctx, &input_buf, &mut output_buf, output_elements as u32)?;

        Ok(output_buf)
    }

    /// Realiza multiplicação de matrizes (A × B) direto na GPU.
    ///
    /// Usado para forward pass de camadas lineares durante a inferência.
    pub fn matmul(
        &self,
        a: &GpuBuffer,
        b: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self
            .pipelines
            .get(&PipelineKind::Matmul)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline Matmul não disponível".into()))?;

        let mut output = self.ctx.alloc_gpu_buffer((m * n * 2) as usize)?; // F16
        pipeline.dispatch_matmul(&self.ctx, a, b, &mut output, m, k, n)?;
        Ok(output)
    }

    /// Calcula similaridade cosseno em batch entre query e candidatos.
    ///
    /// Usado pelo DiskANN/HNSW para busca vetorial acelerada na GPU.
    pub fn cosine_similarity_batch(
        &self,
        query: &GpuBuffer,
        candidates: &GpuBuffer,
        num_candidates: u32,
        dim: u32,
    ) -> Result<Vec<f32>, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::CosineSim).ok_or_else(|| {
            NodeStorError::VulkanError("Pipeline CosineSim não disponível".into())
        })?;

        let mut scores_buf = self.ctx.alloc_gpu_buffer((num_candidates * 4) as usize)?; // f32
        pipeline.dispatch_cosine(&self.ctx, query, candidates, &mut scores_buf, num_candidates, dim)?;

        self.ctx.download_from_gpu(&scores_buf)
    }

    /// Faz upload de dados brutos para a GPU sem transformação.
    fn upload_raw(&self, data: &[u8]) -> Result<GpuBuffer, NodeStorError> {
        self.ctx.upload_to_gpu(data)
    }

    /// Retorna as capabilities da GPU em uso.
    pub fn capabilities(&self) -> &GpuCapabilities {
        self.ctx.capabilities()
    }

    /// Nome do dispositivo em uso.
    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    /// Enumera GPUs disponíveis sem criar um engine completo (Passe 1 do scanner).
    ///
    /// Retorna capabilities reais detectadas pelo Vulkan.
    /// Chamado pelo `nodestor-scanner` antes de qualquer pipeline ser criado.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodestor_core::{GpuVendor, HardwareProfile, OsType, TransportBackend};

    fn mock_profile() -> HardwareProfile {
        HardwareProfile {
            gpus: vec![],
            storage: vec![],
            os: OsType::Windows,
            os_version: "test".to_string(),
            recommended_transport: TransportBackend::PreadFallback,
            cpu_cores: 4,
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn test_probe_gpus_does_not_panic() {
        // Deve funcionar mesmo sem GPU Vulkan disponível
        let caps = VulkanEngine::probe_gpus();
        // Em CI sem GPU, retorna vec vazia — isso é correto
        println!("GPUs detectadas pelo Vulkan probe: {}", caps.len());
    }
}
