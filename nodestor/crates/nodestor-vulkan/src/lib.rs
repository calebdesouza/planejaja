mod buffer;
mod error;
mod instance;
mod pipeline;
mod shader_loader;
pub mod transformer;
pub mod command_recycler;
pub mod triple_buffer;
pub mod external_memory;
pub mod operator_registry;
pub mod unified_pool;

pub use buffer::{GpuBuffer, GpuBufferUsage, QuantKind};
pub use error::VulkanError;
pub use instance::VulkanContext;
pub use pipeline::{ComputePipeline, PipelineKind};
pub use transformer::*;
pub use buffer::MemoryPath;
pub use command_recycler::CommandRecycler;
pub use triple_buffer::TripleBufferPipeline;
pub use external_memory::{try_import_host_memory, is_supported as ext_memory_supported};
pub use operator_registry::{cpu_gelu_fallback, cpu_layer_norm_fallback, cpu_mean_pool, l2_normalize, TensorOp, ActivationKind};
pub use unified_pool::{UnifiedMemoryPool, PoolSlotKind};

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
        // "Rodar em qualquer máquina": sem GPU física no perfil, não tocamos a FFI
        // real do Vulkan (que faria access-violation num ambiente headless/CPU-only).
        // Caímos graciosamente no caminho de simulação RAM-backed.
        let gpu = match profile.primary_gpu() {
            Some(g) => g,
            None => return Ok(Self::new_simulation()),
        };
        let ctx = VulkanContext::new(Some(gpu)).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        // O device Vulkan não inicializou de fato (driver headless) → CPU completa.
        if !ctx.vulkan_available {
            return Ok(Self::new_simulation());
        }
        let pipelines = match pipeline::create_all_pipelines(&ctx) {
            Ok(p) => p,
            Err(e) => {
                warn!("Pipelines Vulkan falharam ({}): usando simulação CPU completa", e);
                return Ok(Self::new_simulation());
            }
        };
        // Se o conjunto de pipelines GPU estiver INCOMPLETO (ex.: o SPIR-V do
        // TurboQuantAttention não compila neste driver), caímos para o motor de
        // simulação COMPLETO (CPU) em vez de um estado GPU meio-carregado. Um
        // forward consistente vale mais — e roda em qualquer máquina.
        let required = [
            PipelineKind::Matmul, PipelineKind::RmsNorm, PipelineKind::RoPe,
            PipelineKind::SiLu, PipelineKind::Add, PipelineKind::Mul,
        ];
        if required.iter().any(|k| !pipelines.contains_key(k)) {
            tracing::warn!("Pipelines GPU incompletos neste driver — usando simulação CPU completa.");
            return Ok(Self::new_simulation());
        }
        Ok(Self { ctx, pipelines })
    }

    /// Cria um VulkanEngine em modo de simulação (sem GPU real).
    /// Usado em benchmarks, testes e ambientes headless (CI/CD, Docker).
    pub fn new_simulation() -> Self {
        // Simulação FORÇADA: vai direto ao contexto CPU/RAM, sem TENTAR Vulkan real
        // (o que poderia access-violation em máquinas headless/loader quebrado).
        let ctx = VulkanContext::simulation();
        let pipelines = pipeline::create_simulation_pipelines();
        Self { ctx, pipelines }
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


    pub fn fused_layernorm_gelu(
        &self,
        input: &GpuBuffer,
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        hidden_dim: u32,
        eps: f32,
    ) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::FusedLayerNormGelu)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline FusedLayerNormGelu not available".into()))?;
        let mut output = self.ctx.alloc_gpu_buffer(input.size)?;
        pipeline.dispatch_fused_layernorm_gelu(&self.ctx, input, gamma, beta, &mut output, hidden_dim, eps)?;
        Ok(output)
    }

    pub fn matmul(&self, a: &GpuBuffer, b: &GpuBuffer, m: u32, k: u32, n: u32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::Matmul).ok_or_else(|| NodeStorError::VulkanError("Pipeline Matmul not available".into()))?;
        let mut output = self.ctx.alloc_gpu_buffer((m * n * 4) as usize)?;
        pipeline.dispatch_matmul(&self.ctx, a, b, &mut output, m, k, n)?;
        Ok(output)
    }

    /// Matmul cujo output é HOST_VISIBLE: GPU escreve direto, CPU lê via as_f32_slice()
    /// sem staging copy. Ideal para Q/K/V (precisam ir para CPU para atenção).
    /// Despacha automaticamente para Q4K shader quando `a.quant_kind == Q4K`.
    pub fn matmul_to_host(&self, a: &GpuBuffer, b: &GpuBuffer, m: u32, k: u32, n: u32) -> Result<GpuBuffer, NodeStorError> {
        if a.quant_kind == crate::buffer::QuantKind::Q4K {
            return self.matmul_q4k_to_host(a, b, m, k, n);
        }
        let pipeline = self.pipelines.get(&PipelineKind::Matmul)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline Matmul not available".into()))?;
        let mut output = if self.ctx.vulkan_available {
            // Staging = HOST_VISIBLE + STORAGE_BUFFER: shader escreve, CPU lê sem cópia
            GpuBuffer::allocate(&self.ctx, (m * n * 4) as usize, GpuBufferUsage::Staging)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?
        } else {
            GpuBuffer::new_storage((m * n * 4) as usize)
        };
        pipeline.dispatch_matmul(&self.ctx, a, b, &mut output, m, k, n)?;
        Ok(output)
    }

    /// Fused Q4_K dequant + matrix-vector mul com output HOST_VISIBLE.
    ///
    /// `weight` deve conter bytes Q4_K (144 bytes / 256 elementos por bloco).
    /// `m` = linhas de saída (N de peso), `k` = dimensão interna, `n` deve ser 1.
    pub fn matmul_q4k_to_host(&self, weight: &GpuBuffer, input: &GpuBuffer, m: u32, k: u32, _n: u32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::MatmulQ4K)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline MatmulQ4K não disponível — usando FP32 fallback".into()))?;
        let mut output = if self.ctx.vulkan_available {
            GpuBuffer::allocate(&self.ctx, (m * 4) as usize, GpuBufferUsage::Staging)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?
        } else {
            GpuBuffer::new_storage((m * 4) as usize)
        };
        pipeline.dispatch_matmul_q4k(&self.ctx, weight, input, &mut output, m, k)?;
        Ok(output)
    }

    /// Executa múltiplos matmuls em um ÚNICO command buffer — 1 submit + 1 queue_wait_idle.
    ///
    /// Cada op é `(weight, input, m, k, n)`. Retorna um `GpuBuffer` HOST_VISIBLE por op,
    /// na mesma ordem. Reduz N syncs GPU→CPU para 1 por chamada — crítico para QKV e gate+up.
    ///
    /// Fallback: se Vulkan não está ativo, executa cada op via `matmul_to_host` individualmente.
    pub fn batch_matmul_to_host(
        &self,
        ops: &[(&GpuBuffer, &GpuBuffer, u32, u32, u32)],
    ) -> Result<Vec<GpuBuffer>, NodeStorError> {
        if ops.is_empty() { return Ok(vec![]); }

        let pipeline = match self.pipelines.get(&PipelineKind::Matmul) {
            Some(p) if p.is_gpu_active() => p,
            _ => {
                // Fallback: individual calls (simulation mode)
                let mut out = Vec::with_capacity(ops.len());
                for &(a, b, m, k, n) in ops {
                    out.push(self.matmul_to_host(a, b, m, k, n)?);
                }
                return Ok(out);
            }
        };

        if !self.ctx.vulkan_available {
            let mut out = Vec::with_capacity(ops.len());
            for &(a, b, m, k, n) in ops {
                out.push(self.matmul_to_host(a, b, m, k, n)?);
            }
            return Ok(out);
        }

        unsafe {
            let device = self.ctx.device.as_ref().unwrap();
            let descriptor_pool = self.ctx.descriptor_pool.unwrap();
            let command_pool = self.ctx.command_pool.unwrap();

            // Allocate one command buffer for all ops.
            let alloc_cmds = ash::vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(ash::vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd_bufs = device.allocate_command_buffers(&alloc_cmds)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            let cmd_buf = cmd_bufs[0];

            device.begin_command_buffer(cmd_buf, &ash::vk::CommandBufferBeginInfo::default())
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let mut outputs = Vec::with_capacity(ops.len());
            let mut ds_collector: Vec<ash::vk::DescriptorSet> = Vec::with_capacity(ops.len());

            for &(a, b, m, k, n) in ops {
                let output = if a.quant_kind == crate::buffer::QuantKind::Q4K {
                    GpuBuffer::allocate(&self.ctx, (m * 4) as usize, GpuBufferUsage::Staging)
                } else {
                    GpuBuffer::allocate(&self.ctx, (m * n * 4) as usize, GpuBufferUsage::Staging)
                }.map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

                pipeline.record_matmul_into(device, descriptor_pool, cmd_buf, a, b, &output, m, k, n, &mut ds_collector)?;
                outputs.push(output);
            }

            device.end_command_buffer(cmd_buf)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            device.queue_submit(
                self.ctx.queue.unwrap(),
                &[ash::vk::SubmitInfo::default().command_buffers(&[cmd_buf])],
                ash::vk::Fence::null(),
            ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_wait_idle(self.ctx.queue.unwrap())
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            device.free_command_buffers(command_pool, &[cmd_buf]);
            let _ = device.free_descriptor_sets(descriptor_pool, &ds_collector);

            Ok(outputs)
        }
    }

    /// Upload para VRAM device-local (256 GB/s de largura de banda pela GPU).
    ///
    /// Três caminhos em ordem de prioridade:
    /// - **UMA / ReBAR**: DEVICE_LOCAL + HOST_VISIBLE → escrita direta, sem staging copy.
    ///   Cobre Apple Silicon via MoltenVK (todo heap é UMA) + AMD/NVIDIA com ReBAR ativo.
    /// - **Transfer Queue**: staging → vkCmdCopyBuffer via fila de transferência dedicada.
    ///   Permite overlap com compute em GPUs com fila separada (RX 580, RTX série).
    /// - **Compute Queue**: staging → vkCmdCopyBuffer via fila de compute (fallback universal).
    pub fn upload_device_local(&self, data: &[u8]) -> Result<GpuBuffer, NodeStorError> {
        if !self.ctx.vulkan_available {
            return Ok(GpuBuffer::from_cpu_data(data.to_vec()));
        }

        // Caminho A — UMA / ReBAR: DEVICE_LOCAL é também HOST_VISIBLE.
        // allocate_pinned() tenta ReBAR primeiro; se o heap tiver >= 80% da VRAM como UMA,
        // retorna buffer mapeado diretamente na VRAM. Zero staging copy.
        if self.ctx.has_rebar {
            if let Ok(mut buf) = GpuBuffer::allocate_pinned(&self.ctx, data.len()) {
                if buf.copy_from_slice(data).is_ok() {
                    return Ok(buf);
                }
            }
        }

        // Caminho B/C — staging → vkCmdCopyBuffer → DEVICE_LOCAL.
        let mut staging = GpuBuffer::allocate(&self.ctx, data.len(), GpuBufferUsage::Staging)
            .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        staging.copy_from_slice(data)
            .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        let src = staging.handle.ok_or_else(|| NodeStorError::VulkanError("staging handle missing".into()))?;

        let device_local = GpuBuffer::allocate(&self.ctx, data.len(), GpuBufferUsage::Storage)
            .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
        let dst = device_local.handle.ok_or_else(|| NodeStorError::VulkanError("storage handle missing".into()))?;

        unsafe {
            let device = self.ctx.device.as_ref().unwrap();

            // Seleciona fila: transfer dedicada (DMA paralelo ao compute) ou compute (fallback).
            let (xfer_queue, pool_family) =
                if self.ctx.has_dedicated_transfer_queue {
                    if let Some(tq) = self.ctx.transfer_queue {
                        (tq, self.ctx.transfer_queue_family)
                    } else {
                        (self.ctx.queue.unwrap(), self.ctx.queue_family_index)
                    }
                } else {
                    (self.ctx.queue.unwrap(), self.ctx.queue_family_index)
                };

            // Pool TRANSIENT no family correto — criado e destruído por upload (carregamento único).
            let xfer_pool = device.create_command_pool(
                &ash::vk::CommandPoolCreateInfo::default()
                    .queue_family_index(pool_family)
                    .flags(ash::vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            ).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let cmd = device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(xfer_pool).level(ash::vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            ).map_err(|e| {
                let _ = device.destroy_command_pool(xfer_pool, None);
                NodeStorError::VulkanError(e.to_string())
            })?[0];

            device.begin_command_buffer(cmd, &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.cmd_copy_buffer(cmd, src, dst, &[ash::vk::BufferCopy::default().size(data.len() as u64)]);
            device.end_command_buffer(cmd).map_err(|e| NodeStorError::VulkanError(e.to_string()))?;

            let fence = device.create_fence(&ash::vk::FenceCreateInfo::default(), None)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.queue_submit(xfer_queue, &[ash::vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| NodeStorError::VulkanError(e.to_string()))?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(xfer_pool, &[cmd]);
            device.destroy_command_pool(xfer_pool, None);
        }
        Ok(device_local)
    }

    /// Executa Fused Decompress (TCA-TBE) + Matmul (ZipGEMM) na VRAM.
    pub fn zipgemm(
        &self,
        compressed_weights: &GpuBuffer,
        activations: &GpuBuffer,
        tile_meta: &GpuBuffer,
        m: u32, k: u32, n: u32,
        tile_stride: u32,
        num_k_tiles: u32,
        weights_per_tile: u32,
    ) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::ZipGEMM)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline ZipGEMM not available".into()))?;
        let mut output = self.ctx.alloc_gpu_buffer((m * n * 4) as usize)?;
        
        // Passa None para o recycler no stub atual
        pipeline.dispatch_zipgemm(
            &self.ctx,
            &None,
            compressed_weights,
            activations,
            &mut output,
            tile_meta,
            m, k, n,
            tile_stride, num_k_tiles, weights_per_tile
        )?;
        Ok(output)
    }

    pub fn matmul_ternary(&self, a: &GpuBuffer, b_packed: &GpuBuffer, m: u32, k: u32, n: u32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::MatmulTernary)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline MatmulTernary not available".into()))?;
        let mut output = self.ctx.alloc_gpu_buffer((m * n * 4) as usize)?;
        pipeline.dispatch_matmul_ternary(&self.ctx, a, b_packed, &mut output, m, k, n)?;
        Ok(output)
    }

    pub fn mamba_selective_scan(
        &self, u: &GpuBuffer, delta: &GpuBuffer, a: &GpuBuffer, b: &GpuBuffer, c: &GpuBuffer,
        state: &GpuBuffer, seq_len: u32, d_inner: u32, d_state: u32
    ) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::MambaSelectiveScan)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline MambaSelectiveScan not available".into()))?;
        let mut y = self.ctx.alloc_gpu_buffer((seq_len * d_inner * 4) as usize)?;
        pipeline.dispatch_mamba_selective_scan(&self.ctx, u, delta, a, b, c, state, &mut y, seq_len, d_inner, d_state)?;
        Ok(y)
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
        // A saída de atenção multi-head tem o MESMO formato do Q projetado
        // ([seq_len, num_heads, head_dim] = n_q elementos), não apenas `head_dim`.
        // Dimensionar por `q.size` evita truncar para 1 head (OOB no matmul de
        // output projection no caminho simulação, e leitura de lixo no GPU).
        let mut out_attn = self.ctx.alloc_gpu_buffer(q.size)?;
        pipeline.dispatch_attention(&self.ctx, q, k, v, &mut out_attn, seq_len, head_dim, scale)?;
        Ok(out_attn)
    }

    pub fn turbo_quant_attention(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer, seq_len: u32, head_dim: u32, scale: f32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::TurboQuantAttention).ok_or_else(|| NodeStorError::VulkanError("TurboQuantAttention missing".into()))?;
        // A saída de atenção multi-head tem o MESMO formato do Q projetado
        // ([seq_len, num_heads, head_dim] = n_q elementos), não apenas `head_dim`.
        // Dimensionar por `q.size` evita truncar para 1 head (OOB no matmul de
        // output projection no caminho simulação, e leitura de lixo no GPU).
        let mut out_attn = self.ctx.alloc_gpu_buffer(q.size)?;
        pipeline.dispatch_turbo_quant_attention(&self.ctx, q, k, v, &mut out_attn, seq_len, head_dim, scale)?;
        Ok(out_attn)
    }

    /// Soma elementwise: `out[i] = a[i] + b[i]`.
    /// Usado para residual connections entre camadas Transformer.
    pub fn add(&self, a: &GpuBuffer, b: &GpuBuffer, elements: u32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::Add)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline Add not available".into()))?;
        let mut out = self.ctx.alloc_gpu_buffer((elements * 4) as usize)?;
        pipeline.dispatch_add(&self.ctx, &None, a, b, &mut out, elements)?;
        Ok(out)
    }

    /// Multiplicação elementwise: `out[i] = a[i] * b[i]`.
    /// Usado para SwiGLU: `SiLU(gate) * up_proj`.
    pub fn mul(&self, a: &GpuBuffer, b: &GpuBuffer, elements: u32) -> Result<GpuBuffer, NodeStorError> {
        let pipeline = self.pipelines.get(&PipelineKind::Mul)
            .ok_or_else(|| NodeStorError::VulkanError("Pipeline Mul not available".into()))?;
        let mut out = self.ctx.alloc_gpu_buffer((elements * 4) as usize)?;
        pipeline.dispatch_mul(&self.ctx, &None, a, b, &mut out, elements)?;
        Ok(out)
    }

    pub fn alloc_buffer(&self, size_bytes: usize) -> Result<GpuBuffer, NodeStorError> {
        self.ctx.alloc_gpu_buffer(size_bytes)
    }

    /// `true` se o motor está rodando em GPU real (não simulação CPU).
    pub fn is_gpu_active(&self) -> bool {
        self.ctx.vulkan_available
    }

    /// Upload de slice f32 direto para GPU.
    pub fn upload_f32(&self, data: &[f32]) -> Result<GpuBuffer, NodeStorError> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.ctx.upload_to_gpu(bytes)
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

    /// Dequantiza blocos Q5_0 → f32.
    ///
    /// Caminho GPU: shader `dequant_q5_0.comp` com workgroup correto (local_size_x=256,
    /// 16 invocações/bloco). Se Vulkan não disponível: CPU fallback inline bit-idêntico.
    ///
    /// `raw_blocks`: N * 22 bytes (formato Q5_0: [d:FP16 2B][qh:4B][qs:16B]).
    /// Retorna N * 32 f32.
    pub fn dequant_q5_0(&self, raw_blocks: &[u8]) -> Result<Vec<f32>, NodeStorError> {
        const BLOCK_BYTES: usize = 22;
        const WEIGHTS_PER_BLOCK: usize = 32;

        if raw_blocks.len() % BLOCK_BYTES != 0 {
            return Err(NodeStorError::InferenceError(
                format!("dequant_q5_0: raw_blocks.len()={} não é múltiplo de {}", raw_blocks.len(), BLOCK_BYTES)
            ));
        }
        let num_blocks = (raw_blocks.len() / BLOCK_BYTES) as u32;
        let out_bytes = (num_blocks as usize) * WEIGHTS_PER_BLOCK * 4;

        // Tenta caminho GPU se o pipeline DequantQ5_0 estiver ativo
        if let Some(pipeline) = self.pipelines.get(&PipelineKind::DequantQ5_0) {
            if pipeline.is_gpu_active() {
                let input_buf = self.ctx.upload_to_gpu(raw_blocks)?;
                let mut output_buf = self.ctx.alloc_gpu_buffer(out_bytes)?;
                pipeline.dispatch_q5_0(&self.ctx, &input_buf, &mut output_buf, num_blocks)?;
                return self.ctx.download_from_gpu(&output_buf);
            }
        }

        // CPU fallback: mesma fórmula do shader
        let mut out = vec![0.0f32; (num_blocks as usize) * WEIGHTS_PER_BLOCK];
        for b in 0..(num_blocks as usize) {
            let off = b * BLOCK_BYTES;
            let d_bits = (raw_blocks[off] as u16) | ((raw_blocks[off + 1] as u16) << 8);
            let d = Self::fp16_to_f32(d_bits);
            let qh = (raw_blocks[off + 2] as u32)
                | ((raw_blocks[off + 3] as u32) << 8)
                | ((raw_blocks[off + 4] as u32) << 16)
                | ((raw_blocks[off + 5] as u32) << 24);
            for j in 0..16usize {
                let qs = raw_blocks[off + 6 + j];
                let xh_lo = (qh >> j) & 1;
                let xh_hi = (qh >> (j + 16)) & 1;
                let x0 = ((qs & 0x0F) as i32 | ((xh_lo as i32) << 4)) - 16;
                let x1 = ((qs >> 4) as i32 | ((xh_hi as i32) << 4)) - 16;
                out[b * 32 + j]      = x0 as f32 * d;
                out[b * 32 + j + 16] = x1 as f32 * d;
            }
        }
        Ok(out)
    }

    fn fp16_to_f32(h: u16) -> f32 {
        let s = (h >> 15) & 1;
        let e = (h >> 10) & 0x1F;
        let m = h & 0x3FF;
        if e == 0 {
            if m == 0 { return 0.0; }
            return (if s != 0 { -1.0f32 } else { 1.0 }) * 5.960_464_5e-8 * m as f32;
        } else if e == 31 {
            return if m == 0 { f32::INFINITY * (if s != 0 { -1.0 } else { 1.0 }) } else { f32::NAN };
        }
        let v = (if s != 0 { -1.0f32 } else { 1.0 })
            * 2f32.powi(e as i32 - 15)
            * (1.0 + m as f32 / 1024.0);
        v
    }
}
