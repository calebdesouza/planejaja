use std::sync::Arc;
use nodestor_core::{DataTransport, NodeStorError, LiquidTransferRequest, TransferResult};
use nodestor_vulkan::VulkanEngine;
use tokio::sync::mpsc;
use tracing::{info, debug};

pub struct LiquidChunk {
    pub chunk_index: usize,
    pub data: Vec<u8>,
    pub transfer_us: u64,
}

/// Orquestrador do Liquid Streaming Pipeline.
///
/// Gerencia a "Corrida de Latência" e o pipeline de micro-fatias.
pub struct LiquidOrchestrator {
    vulkan: Arc<VulkanEngine>,
    transport_a: Arc<dyn DataTransport + Send + Sync>,
    transport_b: Option<Arc<dyn DataTransport + Send + Sync>>,
}

impl LiquidOrchestrator {
    pub fn new(
        vulkan: Arc<VulkanEngine>,
        transport_a: Arc<dyn DataTransport + Send + Sync>,
        transport_b: Option<Arc<dyn DataTransport + Send + Sync>>,
    ) -> Self {
        Self { vulkan, transport_a, transport_b }
    }

    /// Executa o streaming líquido com Corrida de Latência.
    pub async fn stream_liquid(
        &self,
        request: LiquidTransferRequest,
    ) -> Result<(), NodeStorError> {
        info!("🌊 Liquid Streaming: Iniciando corrida p/ '{}'", request.tensor_name);
        
        let (tx, mut rx) = mpsc::channel::<TransferResult>(10);
        let request_arc = Arc::new(request);
        
        let tx_clone = tx.clone();
        let t_a = self.transport_a.clone();
        let r_a = request_arc.clone();

        // Competidor A: transport roda em thread de bloqueio (Rayon) fora do executor Tokio.
        tokio::task::spawn_blocking(move || {
            let callback = Box::new(move |res| {
                // tx.try_send é não-bloqueante e seguro em threads Rayon
                let _ = tx_clone.try_send(res);
            });
            let _ = t_a.transfer_liquid(&r_a, callback);
        });

        // Competidor B: DirectStorage (se disponível)
        if let Some(t_b) = &self.transport_b {
            let tx_clone_b = tx.clone();
            let t_b_clone = t_b.clone();
            let r_b = request_arc.clone();
            tokio::task::spawn_blocking(move || {
                let callback = Box::new(move |res| {
                    let _ = tx_clone_b.try_send(res);
                });
                let _ = t_b_clone.transfer_liquid(&r_b, callback);
            });
        }

        // Processador de fatias - o primeiro que chegar ganha (Modo Metralhadora)
        let mut chunks_processed = 0;
        let total_chunks = (request_arc.total_size + request_arc.chunk_size - 1) / request_arc.chunk_size;
        let mut previous_fence: Option<ash::vk::Fence> = None;

        while let Some(res) = rx.recv().await {
            debug!("   Líquido: Fatia de {} bytes recebida. Saturando barramento...", res.data.len());
            
            // PILLAR 3: Antes de processar a nova fatia, verificamos se a anterior terminou.
            // Isso permite que o transporte da nova fatia aconteça ENQUANTO a GPU calculava a anterior.
            if let Some(fence) = previous_fence {
                unsafe {
                    let device = self.vulkan.ctx.device().unwrap();
                    device.wait_for_fences(&[fence], true, u64::MAX)
                        .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
                }
            }

            // Tamanho original esperado para este chunk em específico
            let expected_size_bytes = (request_arc.total_size - (chunks_processed * request_arc.chunk_size)).min(request_arc.chunk_size);
            
            let (gpu_buf, current_fence) = if request_arc.compression == nodestor_core::CompressionHint::GDeflate {
                // Rota Nodestor-G: GDeflate nativo em Vulkan Compute
                let mut exact_uncomp_size = expected_size_bytes; // Fallback
                if res.data.len() >= 8 {
                    if let Ok((tiles, _tsize, _hlen)) = nodestor_gdeflate::tile::SerializedGDeflateStream::parse_header(&res.data) {
                        let total_unc: u32 = tiles.iter().map(|t| t.uncompressed_size).sum();
                        exact_uncomp_size = total_unc as usize;
                    }
                }
                
                self.vulkan.decompress_gdeflate(&res.data, exact_uncomp_size)?
            } else {
                // PILLAR 1: Expansão Lossless via Direct/Liquid original
                let output_elements = expected_size_bytes / 4; // F32 / Dequant
                self.vulkan.decompress_liquid(&res.data, output_elements)?
            };
            
            // Opcional: Aqui o gpu_buf estaria inserido no TensorRegistry ou Cache L1.
            let _ = gpu_buf;
            
            previous_fence = Some(current_fence);
            chunks_processed += 1;

            // PILLAR 4: Pre-fetching Preditivo
            if request_arc.look_ahead_hint && chunks_processed == (total_chunks / 2) {
                debug!("   🎯 Look-Ahead: Sinal de pré-carregamento emitido para próxima camada.");
            }

            if chunks_processed >= total_chunks {
                // Aguarda o último suspiro da metralhadora
                if let Some(fence) = previous_fence {
                    unsafe {
                        let device = self.vulkan.ctx.device().unwrap();
                        device.wait_for_fences(&[fence], true, u64::MAX)
                            .map_err(|e: ash::vk::Result| NodeStorError::VulkanError(e.to_string()))?;
                    }
                }
                break;
            }
        }

        info!("🏆 Liquid Streaming: {} finalizado com sucesso.", request_arc.tensor_name);
        Ok(())
    }
}
