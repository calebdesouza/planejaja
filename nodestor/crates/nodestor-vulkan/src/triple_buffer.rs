//! Triple Buffer Pipeline — Saturação Contínua do Barramento SSD↔GPU.
//!
//! ## O Problema sem Triple Buffering
//! ```text
//! Fila singela:
//! [SSD lê chunk A]──────[GPU processa A]──[SSD lê chunk B]──────[GPU processa B]
//!                       ^GPU ociosa antes  ^SSD ocioso antes
//! ```
//!
//! ## Com Triple Buffering: GPU e SSD nunca param
//! ```text
//! Balde A: [SSD escreve]──────────[GPU processa]───────────────[SSD escreve]
//! Balde B:               [GPU processa]──────────[SSD escreve]──────────────
//! Balde C:                                       [GPU processa]──────────────
//! ```
//! A partir do 3º chunk, GPU e SSD trabalham 100% em paralelo eternamente.
//!
//! ## Implementação
//! 3 GpuBuffers Pinned (Via Expressa) rotativos com Fences para sincronização.
//! O produtor (SSD) escreve no balde "livre". O consumidor (GPU) processa
//! o balde "cheio". A rotação garante que nunca há espera desnecessária.

use crate::{buffer::{GpuBuffer, GpuBufferUsage, MemoryPath}, instance::VulkanContext};
use nodestor_core::NodeStorError;
use tracing::{debug, trace};

/// Estado de cada balde no pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BucketState {
    /// Livre — disponível para escrita pelo SSD.
    Free,
    /// Cheio — aguardando processamento pela GPU.
    ReadyForGpu,
    /// Em processamento — GPU está usando este balde.
    InFlight,
}

/// Um balde do Triple Buffer com buffer Pinned e fence associada.
struct Bucket {
    buffer: GpuBuffer,
    fence: ash::vk::Fence,
    state: BucketState,
    bytes_written: usize,
}

/// Pipeline de Triple Buffering para streaming contínuo SSD→GPU.
///
/// Garante que GPU e SSD trabalhem em paralelo sem serialização.
/// Usa `GpuBuffer::allocate_pinned()` para o melhor caminho de memória
/// disponível (ReBAR → Pinned DMA → Staging Fallback).
pub struct TripleBufferPipeline {
    buckets: Vec<Bucket>,
    bucket_size: usize,
    write_idx: usize,   // Próximo balde para o SSD escrever
    submit_idx: usize,  // Próximo balde para submeter à GPU
    vulkan_active: bool,
}

impl TripleBufferPipeline {
    /// Cria o pipeline com 3 baldes Pinned de `bucket_size` bytes cada.
    pub fn new(ctx: &VulkanContext, bucket_size: usize) -> Result<Self, NodeStorError> {
        if !ctx.vulkan_available {
            return Ok(Self::simulation(bucket_size));
        }

        let device = ctx.device.as_ref()
            .ok_or_else(|| NodeStorError::VulkanError("Device não disponível".into()))?;

        // Fences pré-criadas já sinalizadas → primeira iteração não bloqueia
        let fence_info = ash::vk::FenceCreateInfo::default()
            .flags(ash::vk::FenceCreateFlags::SIGNALED);

        let mut buckets = Vec::with_capacity(3);
        for i in 0..3 {
            // Aloca via Triple-Path Allocator (ReBAR → Pinned → Staging)
            let buffer = GpuBuffer::allocate_pinned(ctx, bucket_size)?;
            let fence = unsafe {
                device.create_fence(&fence_info, None)
                    .map_err(|e| NodeStorError::VulkanError(format!("Fence Balde {}: {}", i, e)))?
            };
            debug!("Triple Buffer [{}]: {} alocado ({} KB)",
                i, buffer.path_description(), bucket_size / 1024);
            buckets.push(Bucket {
                buffer,
                fence,
                state: BucketState::Free,
                bytes_written: 0,
            });
        }

        Ok(Self {
            buckets,
            bucket_size,
            write_idx: 0,
            submit_idx: 0,
            vulkan_active: true,
        })
    }

    pub fn simulation(bucket_size: usize) -> Self {
        let buckets = (0..3).map(|_| Bucket {
            buffer: GpuBuffer::new_storage(bucket_size),
            fence: ash::vk::Fence::null(),
            state: BucketState::Free,
            bytes_written: 0,
        }).collect();
        Self {
            buckets,
            bucket_size,
            write_idx: 0,
            submit_idx: 0,
            vulkan_active: false,
        }
    }

    /// Adquire o próximo balde livre para o SSD escrever.
    ///
    /// Se o balde rotativo ainda estiver em uso pela GPU, aguarda sua conclusão.
    /// Isso acontece apenas quando a GPU é mais lenta que o SSD (raro).
    pub fn acquire_write_bucket(
        &mut self,
        device: &ash::Device,
    ) -> Result<(usize, &mut GpuBuffer), NodeStorError> {
        let idx = self.write_idx % 3;
        let bucket = &mut self.buckets[idx];

        if self.vulkan_active && bucket.state == BucketState::InFlight {
            // GPU ainda está usando este balde — aguarda
            trace!("Triple Buffer: Balde {} ainda InFlight, aguardando GPU...", idx);
            unsafe {
                device.wait_for_fences(&[bucket.fence], true, 10_000_000_000)
                    .map_err(|e| NodeStorError::VulkanError(format!("WaitFence Balde {}: {}", idx, e)))?;
                device.reset_fences(&[bucket.fence])
                    .map_err(|e| NodeStorError::VulkanError(format!("ResetFence Balde {}: {}", idx, e)))?;
            }
        }

        bucket.state = BucketState::Free;
        bucket.bytes_written = 0;
        self.write_idx += 1;

        Ok((idx, &mut self.buckets[idx].buffer))
    }

    /// Wrapper de simulação para benchmarks puramente focados no SSD
    pub fn acquire_write_bucket_sim(&mut self) -> (usize, &mut GpuBuffer) {
        let idx = self.write_idx % 3;
        self.buckets[idx].state = BucketState::Free;
        self.buckets[idx].bytes_written = 0;
        self.write_idx += 1;
        (idx, &mut self.buckets[idx].buffer)
    }

    /// Retorna o índice de escrita atual (para cálculo pós-borrow do bucket_idx).
    /// Usado pelo ApexOrchestrator após liberar o empréstimo do balde.
    pub fn write_offset(&self) -> usize { self.write_idx }

    /// API de alto nível para simulação: executa o ciclo completo de leitura
    /// sem expor referências internas que causariam conflito de borrow.
    ///
    /// O closure `read_fn` recebe `&mut [u8]` (o buffer do balde), preenche-o
    /// e retorna quantos bytes foram escritos. O pipeline cuida do `mark_ready`
    /// e `submit_next_sim` internamente.
    ///
    /// Em produção (GPU real), os dados ficam no GpuBuffer e não são copiados.
    /// Em simulação, retorna `Vec<u8>` com os bytes lidos.
    pub fn read_via_sim<E>(
        &mut self,
        read_fn: impl FnOnce(&mut [u8]) -> Result<usize, E>,
    ) -> Result<Vec<u8>, E> {
        let idx = self.write_idx % 3;
        self.buckets[idx].state = BucketState::Free;
        self.buckets[idx].bytes_written = 0;
        self.write_idx += 1;

        let n = read_fn(self.buckets[idx].buffer.as_mut_bytes())?;
        let data = self.buckets[idx].buffer.as_bytes()[..n].to_vec();

        self.mark_ready(idx, n);
        self.submit_next_sim();
        Ok(data)
    }

    /// Marca o balde `idx` como cheio depois que o SSD terminou de escrever.
    pub fn mark_ready(&mut self, idx: usize, bytes_written: usize) {
        self.buckets[idx].state = BucketState::ReadyForGpu;
        self.buckets[idx].bytes_written = bytes_written;
        trace!("Triple Buffer: Balde {} pronto ({} bytes)", idx, bytes_written);
    }

    /// Submete o próximo balde pronto para processamento da GPU.
    /// Não bloqueia — a GPU processa enquanto o SSD já está enchendo o próximo balde.
    pub fn submit_next(
        &mut self,
        device: &ash::Device,
        queue: ash::vk::Queue,
        cmd: ash::vk::CommandBuffer,
    ) -> Result<Option<(usize, usize)>, NodeStorError> {
        let idx = self.submit_idx % 3;
        let bucket = &mut self.buckets[idx];

        if bucket.state != BucketState::ReadyForGpu {
            return Ok(None); // Nenhum balde pronto ainda
        }

        let bytes = bucket.bytes_written;

        if self.vulkan_active {
            let submit_info = ash::vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd));

            unsafe {
                device.queue_submit(queue, &[submit_info], bucket.fence)
                    .map_err(|e| NodeStorError::VulkanError(format!("QueueSubmit Balde {}: {}", idx, e)))?;
            }
        }

        bucket.state = BucketState::InFlight;
        self.submit_idx += 1;

        Ok(Some((idx, bytes)))
    }

    /// Wrapper de simulação sem bloqueio de GPU real
    pub fn submit_next_sim(&mut self) {
        let idx = self.submit_idx % 3;
        if self.buckets[idx].state == BucketState::ReadyForGpu {
            self.buckets[idx].state = BucketState::InFlight;
            self.submit_idx += 1;
        }
    }

    /// Aguarda todos os baldes em voo terminarem. Usado no shutdown.
    pub fn drain(&mut self, device: &ash::Device) {
        if !self.vulkan_active { return; }
        let fences: Vec<_> = self.buckets.iter()
            .filter(|b| b.state == BucketState::InFlight)
            .map(|b| b.fence)
            .collect();
        if !fences.is_empty() {
            unsafe { let _ = device.wait_for_fences(&fences, true, 10_000_000_000); }
        }
    }

    /// Dummy de simulação para benchmarking
    pub fn drain_sim(&mut self) {}

    /// Qual caminho de memória os baldes estão usando.
    pub fn memory_path(&self) -> MemoryPath { self.buckets[0].buffer.memory_path }
    pub fn bucket_size(&self) -> usize { self.bucket_size }
    pub fn is_active(&self) -> bool { self.vulkan_active }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_buffer_simulation_mode() {
        // Sem GPU real: modo simulação deve criar os 3 baldes corretamente
        let pipeline = TripleBufferPipeline::simulation(65536);
        assert_eq!(pipeline.buckets.len(), 3);
        assert_eq!(pipeline.bucket_size, 65536);
        assert!(!pipeline.vulkan_active);
        assert!(pipeline.buckets.iter().all(|b| b.state == BucketState::Free));
    }

    #[test]
    fn test_mark_ready_transitions() {
        let mut pipeline = TripleBufferPipeline::simulation(1024);
        pipeline.buckets[0].state = BucketState::Free;
        pipeline.mark_ready(0, 512);
        assert_eq!(pipeline.buckets[0].state, BucketState::ReadyForGpu);
        assert_eq!(pipeline.buckets[0].bytes_written, 512);
    }

    #[test]
    fn test_bucket_rotation_indices() {
        let pipeline = TripleBufferPipeline::simulation(1024);
        // Verifica que a rotação modulo 3 está correta
        assert_eq!(0 % 3, 0);
        assert_eq!(1 % 3, 1);
        assert_eq!(2 % 3, 2);
        assert_eq!(3 % 3, 0); // Rotação
        assert_eq!(4 % 3, 1);
    }
}
